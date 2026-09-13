"""
Servidor de evaluación del reto Altur — versión fusionada (features propias +
features de timing de Gabi).

POST /detect
    body (contrato oficial confirmado por los organizadores):
        {"call_id": "...", "audio_base64": "<WAV completo en base64>",
         "sample_rate": 8000, "channels": 2}
    resp: {"is_synthetic": true, "confidence": 0.87}

Regla del juez: un timeout, un status != 200, o una respuesta sin
`is_synthetic` booleano cuenta como respuesta incorrecta. Por eso este
servidor NUNCA deja escapar un error como status 4xx/5xx: cualquier
excepción durante el procesamiento cae a una respuesta de fallback
válida con status 200.

Arranca con:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import base64
import io
import os
import sys
import traceback

import joblib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.turn_detection import detect_turns
from features.timing_features import compute_timing_features
from features.acoustic_features import compute_acoustic_features
from features.linguistic_features import compute_linguistic_features
from features.gabi_timing_features import turn_features_gabi, add_duration_features_gabi
from features.feature_pipeline import features_to_vector, _turns_to_dicts

MODEL_PATH = os.environ.get("ALTUR_MODEL_PATH", "models/model.joblib")

# Campo oficial: audio_base64. Se mantienen alias como respaldo por si algún
# cliente de prueba usa un nombre distinto (no debería costar nada y evita
# fallar por un detalle de formato).
AUDIO_FIELD_NAMES = ("audio_base64", "audio", "wav_base64", "wav", "audio_wav_base64")

# Respuesta de emergencia: siempre HTTP 200 y con is_synthetic booleano, tal
# como exige el juez, incluso si no pudimos ni parsear el request.
FALLBACK_RESPONSE = {"is_synthetic": False, "confidence": 0.5}

app = FastAPI(title="Altur Challenge - Human vs Synthetic Caller Detector (fusion)")

# Permite que una página web (la herramienta de demo, o el harness de los
# jueces) llame al endpoint directo desde el navegador sin ser bloqueada.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/detect", response_model=None)
async def detect(request: Request):
    """Wrapper a prueba de fallos: cualquier excepción de `_detect_impl` cae
    a FALLBACK_RESPONSE con status 200, nunca a un 4xx/5xx ni a una excepción
    sin manejar (eso el juez lo cuenta como respuesta incorrecta igual, pero
    al menos no como timeout ni como error de transporte)."""
    try:
        return await _detect_impl(request)
    except Exception as e:
        print(f"[ERROR] /detect: {e}", file=sys.stderr)
        traceback.print_exc()
        return JSONResponse(content=FALLBACK_RESPONSE, status_code=200)


def _decode_audio(raw: bytes, declared_sample_rate: int | None, declared_channels: int | None):
    """Decodifica el WAV. Si el header viene corrupto pero el juez nos dio
    sample_rate/channels en el JSON, reintenta como PCM16 crudo con esos
    parámetros en vez de fallar."""
    try:
        audio, sr = sf.read(io.BytesIO(raw), always_2d=True)
        return audio, sr
    except Exception:
        if not declared_sample_rate or not declared_channels:
            raise
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        n_full_frames = len(pcm) - (len(pcm) % declared_channels)
        audio = pcm[:n_full_frames].reshape(-1, declared_channels)
        return audio, declared_sample_rate


async def _detect_impl(request: Request) -> DetectResponse:
    bundle = _load_model()
    clf = bundle["model"]
    feature_names = bundle["feature_names"]

    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("el body debe ser un objeto JSON")

    # Contrato oficial: call_id, audio_base64, sample_rate, channels.
    call_id = body.get("call_id")
    declared_sample_rate = body.get("sample_rate")
    declared_channels = body.get("channels")

    b64 = next((body[k] for k in AUDIO_FIELD_NAMES if body.get(k)), None)
    if b64 is None:
        raise ValueError(f"falta el audio en base64; se espera el campo 'audio_base64'")

    use_asr = bool(body.get("use_asr", False))

    raw = base64.b64decode(b64, validate=True)
    audio, sr = _decode_audio(raw, declared_sample_rate, declared_channels)

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    if audio.shape[1] < 2:
        # Llegó mono (no debería pasar según el contrato, pero por si acaso
        # no lo tratamos como error): usamos el único canal como caller y
        # dejamos el canal del agente en silencio.
        ch0 = audio[:, 0]
        ch1 = np.zeros_like(ch0)
    else:
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

    if call_id:
        print(f"[detect] call_id={call_id} is_synthetic={proba_synth >= 0.5} confidence={max(proba_synth, 1 - proba_synth):.3f}")

    return DetectResponse(
        is_synthetic=proba_synth >= 0.5,
        confidence=proba_synth if proba_synth >= 0.5 else 1.0 - proba_synth,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
