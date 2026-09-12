"""
Pipeline único: WAV (+ turns.json opcional) -> dict de features -> vector.

Uso:
    from features.feature_pipeline import extract_features_from_file
    feats = extract_features_from_file("audio/call_XXXX.wav", turns_json_path=None, use_asr=False)
"""
from __future__ import annotations
import json
import numpy as np
import soundfile as sf

from .turn_detection import detect_turns, Turn
from .timing_features import compute_timing_features
from .acoustic_features import compute_acoustic_features
from .linguistic_features import transcribe_caller, compute_linguistic_features


def _load_stereo_wav(path: str):
    audio, sr = sf.read(path, always_2d=True)  # (n_samples, n_channels)
    channel0 = audio[:, 0]
    channel1 = audio[:, 1] if audio.shape[1] > 1 else np.zeros_like(channel0)
    return channel0, channel1, sr


def _turns_from_json(path: str) -> tuple[list[Turn], list[Turn]]:
    with open(path) as f:
        data = json.load(f)
    caller, agent = [], []
    for t in data.get("turns", []):
        turn = Turn(float(t["start"]), float(t["end"]))
        if t["channel"] == 0:
            caller.append(turn)
        else:
            agent.append(turn)
    caller.sort(key=lambda t: t.start)
    agent.sort(key=lambda t: t.start)
    return caller, agent


def extract_features_from_file(
    wav_path: str,
    turns_json_path: str | None = None,
    use_asr: bool = False,
    use_own_vad: bool = True,
) -> dict:
    """
    use_own_vad=True (default y recomendado): usa el VAD propio (turn_detection.py)
      para AMBOS canales, igual que se hará en /detect. Así entrenas con la misma
      distribución de errores de segmentación que verás en producción.
    Si turns_json_path se provee Y use_own_vad=False, usa los turnos "gold" del
      dataset (útil solo para auditar qué tanto degrada tu VAD, ver evaluate_vad.py).
    """
    ch0, ch1, sr = _load_stereo_wav(wav_path)
    duration_s = len(ch0) / sr

    if use_own_vad or turns_json_path is None:
        caller_turns = detect_turns(ch0, sr)
        agent_turns = detect_turns(ch1, sr)
    else:
        caller_turns, agent_turns = _turns_from_json(turns_json_path)

    feats = {}
    feats.update(compute_timing_features(caller_turns, agent_turns, duration_s))
    feats.update(compute_acoustic_features(ch0, sr, caller_turns))

    if use_asr:
        transcript = transcribe_caller(ch0, sr, caller_turns)
        feats.update(compute_linguistic_features(transcript))
    else:
        feats.update(compute_linguistic_features(""))  # ceros, mantiene el esquema de columnas

    return feats


FEATURE_NAMES_CACHE: list[str] | None = None


def features_to_vector(feats: dict, feature_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Convierte el dict a un vector ordenado. Si no se pasa feature_names, usa
    el orden alfabético de llaves (consistente siempre y cuando el dict tenga
    siempre las mismas llaves, que es el caso aquí)."""
    names = feature_names or sorted(feats.keys())
    vec = np.array([feats.get(n, 0.0) for n in names], dtype=np.float64)
    return vec, names
