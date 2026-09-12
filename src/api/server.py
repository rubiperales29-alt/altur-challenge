"""
Servidor de evaluación del reto Altur.

POST /detect
    body: {"audio_base64": "<wav base64>"}   (stereo, 8kHz, 16-bit PCM,
                                               canal 0 = caller, canal 1 = agente)
    resp: {"is_synthetic": true, "confidence": 0.87}

Arranca con:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
(ejecutar desde la carpeta src/, o ajustar PYTHONPATH)
"""
from __future__ import annotations
import base64
import io
import os
import sys

import joblib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.turn_detection import detect_turns
from features.timing_features import compute_timing_features
from features.acoustic_features import compute_acoustic_features
from features.linguistic_features import compute_linguistic_features
from features.feature_pipeline import features_to_vector

MODEL_PATH = os.environ.get("ALTUR_MODEL_PATH", "models/model.joblib")

app = FastAPI(title="Altur Challenge - Human vs Synthetic Caller Detector")

_bundle = None


def _load_model():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"No se encontró el modelo en {MODEL_PATH}. Corre primero: python src/train.py"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class DetectRequest(BaseModel):
    audio_base64: str
    use_asr: bool = False  # dejar en False para latencia baja en evaluación


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@app.on_event("startup")
def startup():
    try:
        _load_model()
        print(f"Modelo cargado desde {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Modelo no cargado todavía: {e}")


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    bundle = _load_model()
    clf = bundle["model"]
    feature_names = bundle["feature_names"]

    try:
        raw = base64.b64decode(req.audio_base64)
        audio, sr = sf.read(io.BytesIO(raw), always_2d=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"WAV inválido: {e}")

    if audio.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Se esperaba WAV estéreo (2 canales)")

    ch0 = audio[:, 0]
    ch1 = audio[:, 1]
    duration_s = len(ch0) / sr

    caller_turns = detect_turns(ch0, sr)
    agent_turns = detect_turns(ch1, sr)

    feats = {}
    feats.update(compute_timing_features(caller_turns, agent_turns, duration_s))
    feats.update(compute_acoustic_features(ch0, sr, caller_turns))
    feats.update(compute_linguistic_features(""))  # placeholder si use_asr=False

    if req.use_asr:
        from features.linguistic_features import transcribe_caller
        transcript = transcribe_caller(ch0, sr, caller_turns)
        feats.update(compute_linguistic_features(transcript))

    vec, _ = features_to_vector(feats, feature_names=feature_names)
    proba_synth = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])

    return DetectResponse(
        is_synthetic=proba_synth >= 0.5,
        confidence=proba_synth if proba_synth >= 0.5 else 1.0 - proba_synth,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
