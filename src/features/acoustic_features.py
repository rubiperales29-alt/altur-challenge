"""
Features acústicas anti-spoofing sobre el canal del caller.

No intentamos reconocer AL hablante (eso rompería con speaker-disjoint splits).
Buscamos artefactos de síntesis que generalizan a voces nunca vistas:
  - Micro-inestabilidad de pitch (jitter) y de amplitud (shimmer): el TTS suele
    ser o demasiado estable o tener patrones periódicos distintos a la voz humana.
  - Ausencia/presencia de respiraciones cortas entre frases.
  - Piso de ruido de fondo: un humano real casi siempre tiene ruido ambiental
    no estacionario; un TTS limpio tiene silencio "perfecto" o ruido sintético plano.
  - Planitud espectral (spectral flatness) y MFCC: capturan textura de voz sintética.

Requiere: numpy, librosa
"""
from __future__ import annotations
import numpy as np
import librosa

from .turn_detection import Turn


def _extract_segment(x: np.ndarray, sr: int, turn: Turn) -> np.ndarray:
    i0 = max(0, int(turn.start * sr))
    i1 = min(len(x), int(turn.end * sr))
    return x[i0:i1]


def _pitch_track(seg: np.ndarray, sr: int) -> np.ndarray:
    if len(seg) < sr * 0.05:
        return np.array([])
    try:
        f0, voiced_flag, _ = librosa.pyin(
            seg.astype(np.float32), fmin=70, fmax=400, sr=sr,
            frame_length=1024, hop_length=160,
        )
        f0 = f0[voiced_flag] if f0 is not None else np.array([])
        f0 = f0[~np.isnan(f0)] if len(f0) else f0
        return f0
    except Exception:
        return np.array([])


def compute_acoustic_features(x: np.ndarray, sr: int, caller_turns: list[Turn]) -> dict:
    feats: dict = {}

    # Concatenar segmentos de habla del caller (máx ~30s para no gastar tiempo)
    segs = [_extract_segment(x, sr, t) for t in caller_turns]
    segs = [s for s in segs if len(s) > 0]
    if not segs:
        return _empty_acoustic_features()

    concat = np.concatenate(segs)
    max_samples = sr * 30
    if len(concat) > max_samples:
        concat = concat[:max_samples]
    concat_f = concat.astype(np.float32) / (np.max(np.abs(concat)) + 1e-6)

    # --- Pitch: jitter (variación ciclo a ciclo) ---
    f0 = _pitch_track(concat_f, sr)
    if len(f0) > 3:
        feats["f0_mean"] = float(np.mean(f0))
        feats["f0_std"] = float(np.std(f0))
        diffs = np.abs(np.diff(f0))
        feats["jitter_mean"] = float(np.mean(diffs) / (np.mean(f0) + 1e-6))
    else:
        feats["f0_mean"] = 0.0
        feats["f0_std"] = 0.0
        feats["jitter_mean"] = 0.0

    # --- Shimmer: variación de amplitud ciclo a ciclo (aprox por RMS de ventanas cortas) ---
    frame_len = int(sr * 0.02)
    hop_len = int(sr * 0.01)
    if len(concat_f) > frame_len * 2:
        rms = librosa.feature.rms(y=concat_f, frame_length=frame_len, hop_length=hop_len)[0]
        rms = rms[rms > 1e-4]
        if len(rms) > 3:
            shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-6)
            feats["shimmer_mean"] = float(shimmer)
        else:
            feats["shimmer_mean"] = 0.0
    else:
        feats["shimmer_mean"] = 0.0

    # --- Spectral flatness (textura sintética vs natural) ---
    flatness = librosa.feature.spectral_flatness(y=concat_f)[0]
    feats["spectral_flatness_mean"] = float(np.mean(flatness))
    feats["spectral_flatness_std"] = float(np.std(flatness))

    # --- MFCCs (resumen estadístico, no identidad) ---
    mfcc = librosa.feature.mfcc(y=concat_f, sr=sr, n_mfcc=13)
    feats["mfcc_mean_std"] = float(np.mean(np.std(mfcc, axis=1)))  # variabilidad promedio entre coefs

    # --- Micro-pausas de respiración dentro de los turnos (no entre turnos) ---
    breath_gaps = []
    for seg in segs:
        if len(seg) < sr * 0.3:
            continue
        seg_f = seg.astype(np.float32)
        r = librosa.feature.rms(y=seg_f, frame_length=frame_len, hop_length=hop_len)[0]
        if len(r) < 5:
            continue
        thresh = np.percentile(r, 20) * 1.5
        low = r < thresh
        # contar micro-huecos cortos (5-25 frames ~ 50-250ms) dentro del turno = candidatos a respiración
        run = 0
        for v in low:
            if v:
                run += 1
            else:
                if 5 <= run <= 25:
                    breath_gaps.append(run)
                run = 0
    feats["breath_gap_count"] = float(len(breath_gaps))
    feats["breath_gap_rate"] = float(len(breath_gaps) / max(1, len(segs)))

    # --- Piso de ruido de fondo entre turnos (silencio "de verdad" vs plano de TTS) ---
    if len(caller_turns) > 1:
        gaps = []
        for a, b in zip(caller_turns[:-1], caller_turns[1:]):
            i0, i1 = int(a.end * sr), int(b.start * sr)
            if i1 > i0:
                gaps.append(x[i0:i1])
        if gaps:
            noise = np.concatenate(gaps)
            feats["noise_floor_rms"] = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2) + 1e-12))
            feats["noise_floor_std"] = float(np.std(noise.astype(np.float64)))
        else:
            feats["noise_floor_rms"] = 0.0
            feats["noise_floor_std"] = 0.0
    else:
        feats["noise_floor_rms"] = 0.0
        feats["noise_floor_std"] = 0.0

    return feats


def _empty_acoustic_features() -> dict:
    keys = [
        "f0_mean", "f0_std", "jitter_mean", "shimmer_mean",
        "spectral_flatness_mean", "spectral_flatness_std", "mfcc_mean_std",
        "breath_gap_count", "breath_gap_rate", "noise_floor_rms", "noise_floor_std",
    ]
    return {k: 0.0 for k in keys}
