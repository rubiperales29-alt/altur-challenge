"""
Detección de turnos de habla por canal (VAD basado en energía).

En entrenamiento tenemos turns/<id>.json, pero en /detect (evaluación) SOLO
recibimos el WAV. Por eso el pipeline completo (train + inferencia) usa este
detector propio, para que no haya un mismatch entre lo que el modelo vio en
entrenamiento y lo que ve en producción.

Si existe un turns.json para una llamada, se puede usar sample_turns_from_json
para comparar / auditar qué tan bien generaliza este VAD (ver evaluate_vad.py).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Turn:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _frame_energy(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = 1 + max(0, (len(x) - frame_len) // hop_len)
    energies = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop_len: i * hop_len + frame_len]
        energies[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    times = (np.arange(n_frames) * hop_len) / sr
    return times, energies


def detect_turns(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    min_turn_s: float = 0.15,
    min_gap_s: float = 0.20,
    energy_percentile: float = 55.0,
) -> list[Turn]:
    """VAD simple por umbral adaptativo de energía RMS por frame.

    - Umbral = percentil (energy_percentile) de la energía de TODO el canal,
      para adaptarse al piso de ruido de cada llamada (telefonía 8kHz varía mucho).
    - Rellena micro-huecos menores a min_gap_s (para no partir una palabra en 2 turnos).
    - Descarta turnos menores a min_turn_s (ruido/click).
    """
    times, energies = _frame_energy(x, sr, frame_ms, hop_ms)
    if len(energies) == 0:
        return []

    # umbral robusto: mezcla de percentil global y piso de silencio (percentil bajo)
    noise_floor = np.percentile(energies, 15)
    speech_ref = np.percentile(energies, energy_percentile)
    threshold = noise_floor + 0.35 * (speech_ref - noise_floor) + 1e-6

    is_speech = energies > threshold

    # cerrar micro-huecos
    hop_s = hop_ms / 1000.0
    min_gap_frames = max(1, int(min_gap_s / hop_s))
    gap_start = None
    for i, v in enumerate(is_speech):
        if not v:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                if (i - gap_start) <= min_gap_frames:
                    is_speech[gap_start:i] = True
                gap_start = None

    # extraer segmentos contiguos
    turns: list[Turn] = []
    in_turn = False
    t0 = 0.0
    for i, v in enumerate(is_speech):
        if v and not in_turn:
            in_turn = True
            t0 = times[i]
        elif not v and in_turn:
            in_turn = False
            t1 = times[i]
            if (t1 - t0) >= min_turn_s:
                turns.append(Turn(t0, t1))
    if in_turn:
        t1 = times[-1] + hop_ms / 1000.0
        if (t1 - t0) >= min_turn_s:
            turns.append(Turn(t0, t1))

    return turns
