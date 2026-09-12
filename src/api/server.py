"""
Servidor de evaluación del reto Altur — versión fusionada (features propias +
features de timing de Gabi), con manejo flexible del nombre del campo de audio.

POST /detect
    body: JSON con el WAV en base64 bajo uno de varios nombres de campo
          posibles (el reto no especifica el nombre exacto, así que aceptamos
          los más comunes: audio_base64, audio, wav_base64, wav,
          audio_wav_base64).
    resp: {"is_synthetic": true, "confidence": 0.87}

Arranca con:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import base64
import io
import os
import sys

import joblib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.turn_detection import detect_turns
from features.timing_features import compute_timing_features
from features.acoustic_features import compute_acoustic_features
from features.linguistic_features import compute_linguistic_features
from features.gabi_timing_features import turn_features_gabi, add_duration_features_gabi
from features.feature_pipeline import features_to_vector, _turns_to_dicts

MODEL_PATH = os.environ.get("ALTUR_MODEL_PATH", "models/model.joblib")

# El reto nunca especifica el nombre exacto del campo JSON del audio en base64
# -- aceptamos varios nombres comunes para no fallar por un detalle de formato.
AUDIO_FIELD_NAMES = ("audio_base64", "audio", "wav_base64", "wav", "audio_wav_base64")

app = FastAPI(title="Altur Challenge - Human vs Synthetic Caller Detector (fusion)")

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
async def detect(request: Request):
    bundle = _load_model()
    clf = bundle["model"]
    feature_names = bundle["feature_names"]

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="el body debe ser JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="el body debe ser un objeto JSON")

    b64 = next((body[k] for k in AUDIO_FIELD_NAMES if body.get(k)), None)
    if b64 is None:
        raise HTTPException(
            status_code=400,
            detail=f"falta el audio en base64; se espera uno de estos campos: {AUDIO_FIELD_NAMES}",
        )

    use_asr = bool(body.get("use_asr", False))

    try:
        raw = base64.b64decode(b64, validate=True)
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

    gabi_turns = _turns_to_dicts(caller_turns, agent_turns)
    gabi_feats = turn_features_gabi(gabi_turns)
    gabi_feats = add_duration_features_gabi(gabi_feats, duration_s)
    feats.update(gabi_feats)

    feats.update(compute_linguistic_features(""))

    if use_asr:
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
