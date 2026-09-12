"""
Features lingüísticas (OPCIONALES, requieren transcribir con un ASR).

Idea diferenciadora del reto: cuando el agente pregunta por algo que NO EXISTE,
  - un humano muestra confusión genuina: pausas largas, "¿cómo?", "no le entiendo",
    repeticiones, autocorrección.
  - un caller-LLM tiende a alucinar una respuesta FLUIDA y coherente, porque está
    instruido a "responder algo".

Esta etapa es opcional (cuesta más tiempo/CPU) — actívala con use_asr=True en
feature_pipeline.py. Usa faster-whisper (CPU-friendly, modelo "small"/"base").
Si no está instalado o falla, se degrada a features en cero (no rompe el pipeline).
"""
from __future__ import annotations
import re

FILLERS_ES_MX = [
    "este", "o sea", "pos", "pus", "digo", "mmm", "eh", "ajá", "este pues",
    "no sé", "cómo le digo", "a ver", "fíjese que",
]
CONFUSION_MARKERS = [
    "cómo", "qué", "no entendí", "no le entiendo", "perdón", "disculpe",
    "puede repetir", "mande", "cómo dice", "no comprendo",
]

_ASR_MODEL = None


def _get_model():
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    try:
        from faster_whisper import WhisperModel
        _ASR_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
    except Exception:
        _ASR_MODEL = False  # marca de "no disponible"
    return _ASR_MODEL


def transcribe_caller(x, sr, caller_turns) -> str:
    model = _get_model()
    if not model:
        return ""
    import numpy as np
    segs = []
    for t in caller_turns:
        i0, i1 = int(t.start * sr), int(t.end * sr)
        if i1 > i0:
            segs.append(x[i0:i1])
    if not segs:
        return ""
    audio = np.concatenate(segs).astype("float32")
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    try:
        segments, _ = model.transcribe(audio, language="es", vad_filter=True)
        return " ".join(s.text for s in segments).strip().lower()
    except Exception:
        return ""


def compute_linguistic_features(transcript: str, n_words_agent_nonsense: int = 0) -> dict:
    """n_words_agent_nonsense: opcional, si quieres marcar manualmente turnos donde
    el agente preguntó algo sin sentido (no implementado por default, placeholder)."""
    feats = {}
    text = transcript or ""
    words = re.findall(r"\w+", text)
    n_words = len(words)

    feats["n_words"] = float(n_words)
    feats["filler_count"] = float(sum(text.count(f) for f in FILLERS_ES_MX))
    feats["filler_rate"] = float(feats["filler_count"] / max(1, n_words))
    feats["confusion_marker_count"] = float(sum(text.count(c) for c in CONFUSION_MARKERS))
    feats["confusion_marker_rate"] = float(feats["confusion_marker_count"] / max(1, n_words))

    # repeticiones / autocorrección: palabra inmediatamente repetida ("no no", "yo yo creo")
    repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
    feats["immediate_repeat_count"] = float(repeats)

    # longitud media de "oración" (proxy de fluidez) usando puntos/comas si el ASR los da
    chunks = re.split(r"[.,;]", text)
    chunk_lens = [len(re.findall(r"\w+", c)) for c in chunks if c.strip()]
    feats["mean_chunk_len"] = float(sum(chunk_lens) / len(chunk_lens)) if chunk_lens else 0.0

    return feats
