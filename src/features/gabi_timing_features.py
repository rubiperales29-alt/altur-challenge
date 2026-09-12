"""
Features de timing/turn-taking de Gabi, portadas a este pipeline.

Aporta señales que nuestras timing_features.py no tenía:
  - resp_lat_cv: coeficiente de variación de la latencia de respuesta
    (la VARIANZA discrimina más que la media sola).
  - stop_delay: cuánto sigue hablando el caller tras ser interrumpido
    (similar a nuestro cutoff_react, calculado distinto -> complementario).
  - silence_broken_by_caller: en silencios largos (>1s), cuántas veces
    es el CALLER quien los rompe (vs el agente).
  - Normalización anti-fuga por duración (_per_min) en vez de exponer
    la duración cruda como feature (evita que el modelo aprenda a base
    de "cuánto dura la llamada" en vez de cómo se comporta el caller).

Recibe turns en el mismo formato dict que usa el dataset de Altur:
{"channel": 0|1, "start": float, "end": float}
"""
from __future__ import annotations
import numpy as np


def _stats(xs, prefix):
    if len(xs) == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "std", "min", "max", "med", "n")}
    a = np.array(xs, dtype=float)
    return {
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_std":  float(a.std()),
        f"{prefix}_min":  float(a.min()),
        f"{prefix}_max":  float(a.max()),
        f"{prefix}_med":  float(np.median(a)),
        f"{prefix}_n":    float(len(a)),
    }


PER_MIN_KEYS = ("caller_turns", "agent_turns", "interrupted_n",
                "silence_broken_by_caller", "gap_n", "resp_lat_n")


def turn_features_gabi(turns: list[dict]) -> dict:
    """turns: lista de {"channel": 0|1, "start": float, "end": float}."""
    caller = sorted([t for t in turns if t["channel"] == 0], key=lambda t: t["start"])
    agent = sorted([t for t in turns if t["channel"] == 1], key=lambda t: t["start"])

    f = {}

    # --- 1. Latencia de respuesta ---
    lat = []
    for a in agent:
        nxt = [c for c in caller if c["start"] >= a["end"]]
        if nxt:
            d = nxt[0]["start"] - a["end"]
            if d < 10:
                lat.append(d)
    f.update(_stats(lat, "gabi_resp_lat"))
    f["gabi_resp_lat_cv"] = f["gabi_resp_lat_std"] / (f["gabi_resp_lat_mean"] + 1e-6)

    # --- 2. Solapamiento / barge-in ---
    overlap_total, barge_in = 0.0, 0
    for c in caller:
        for a in agent:
            ov = min(c["end"], a["end"]) - max(c["start"], a["start"])
            if ov > 0:
                overlap_total += ov
                if a["start"] > c["start"]:
                    barge_in += 1
    caller_speech = sum(c["end"] - c["start"] for c in caller) or 1e-6
    f["gabi_overlap_ratio"] = overlap_total / caller_speech
    f["gabi_interrupted_n"] = float(barge_in)

    # --- 3. Recuperación tras interrupción ---
    stop_delay = []
    for a in agent:
        for c in caller:
            if c["start"] < a["start"] < c["end"]:
                stop_delay.append(c["end"] - a["start"])
    f.update(_stats(stop_delay, "gabi_stop_delay"))

    # --- 4. Respuesta al silencio ---
    gaps = []
    events = sorted(turns, key=lambda t: t["start"])
    for i in range(len(events) - 1):
        g = events[i + 1]["start"] - events[i]["end"]
        if g > 1.0:
            gaps.append((g, events[i + 1]["channel"]))
    f["gabi_silence_broken_by_caller"] = float(sum(1 for g, ch in gaps if ch == 0))
    f.update(_stats([g for g, _ in gaps], "gabi_gap"))

    # --- 5. Forma de los turnos ---
    f.update(_stats([c["end"] - c["start"] for c in caller], "gabi_caller_turn"))
    f["gabi_caller_turns"] = float(len(caller))
    f["gabi_agent_turns"] = float(len(agent))
    f["gabi_turn_ratio"] = len(caller) / (len(agent) + 1e-6)

    return f


def add_duration_features_gabi(feats: dict, duration_s: float) -> dict:
    """Normaliza por duración SIN exponer la duración cruda (evita fuga:
    ver el análisis de Gabi en inspect_data.py -- 'LA TRAMPA')."""
    dur = max(float(duration_s), 1e-6)
    out = dict(feats)
    keys = tuple(f"gabi_{k}" for k in PER_MIN_KEYS)
    for k in keys:
        if k in out:
            out[f"{k}_per_min"] = out[k] / (dur / 60.0)
    return out
