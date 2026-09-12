"""
Features de turn-taking entre caller (canal 0) y agente (canal 1).

Hipótesis del reto: un caller sintético es un pipeline ASR -> LLM -> TTS.
Eso deja huellas en el TIMING que no dependen de la voz en sí:
  - Latencia de respuesta tras hablar el agente (más angosta/uniforme en pipelines)
  - Habla superpuesta genuina (backchannels "ajá", "sí") -> rara en sistemas por turnos
  - Reacción a interrupciones del agente (corte "demasiado limpio" = sospechoso)
  - Regularidad de la duración de los turnos del caller

Todo esto se calcula solo con listas de Turn (start, end) por canal.
"""
from __future__ import annotations
import numpy as np
from .turn_detection import Turn


def _stats(values: list[float], prefix: str) -> dict:
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0, f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0, f"{prefix}_n": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_n": float(len(arr)),
    }


def compute_timing_features(caller_turns: list[Turn], agent_turns: list[Turn], duration_s: float) -> dict:
    feats: dict = {}

    # --- 1. Latencia de respuesta: caller.start - agent.end previo ---
    latencies = []
    interruptions = []  # latencia negativa = el caller entró ANTES de que el agente terminara
    for ct in caller_turns:
        prev_agent_ends = [at.end for at in agent_turns if at.end <= ct.start + 0.05]
        if prev_agent_ends:
            last_agent_end = max(prev_agent_ends)
            lat = ct.start - last_agent_end
            latencies.append(lat)
            if lat < 0:
                interruptions.append(-lat)  # cuánto se metió encima

    feats.update(_stats(latencies, "resp_latency"))
    feats["frac_negative_latency"] = float(np.mean([l < 0 for l in latencies])) if latencies else 0.0

    # --- 2. Overlap: turnos del caller que se solapan con turnos del agente ---
    overlap_durs = []
    for ct in caller_turns:
        for at in agent_turns:
            ov = min(ct.end, at.end) - max(ct.start, at.start)
            if ov > 0:
                overlap_durs.append(ov)
    feats.update(_stats(overlap_durs, "overlap_dur"))
    feats["overlap_count"] = float(len(overlap_durs))
    feats["overlap_rate"] = float(len(overlap_durs)) / max(1, len(caller_turns))

    # --- 3. Reacción del caller cuando el AGENTE lo interrumpe a ÉL ---
    # (agente empieza a hablar antes de que el caller termine)
    cutoff_reactions = []  # cuánto sigue hablando el caller tras ser interrumpido
    for at in agent_turns:
        for ct in caller_turns:
            if ct.start < at.start < ct.end:
                cutoff_reactions.append(ct.end - at.start)
    feats.update(_stats(cutoff_reactions, "cutoff_react"))

    # --- 4. Duración y regularidad de los turnos del caller ---
    caller_durs = [t.duration for t in caller_turns]
    feats.update(_stats(caller_durs, "caller_turn_dur"))
    # coeficiente de variación: pipelines TTS tienden a ser más regulares
    if caller_durs and np.mean(caller_durs) > 1e-6:
        feats["caller_turn_dur_cv"] = float(np.std(caller_durs) / np.mean(caller_durs))
    else:
        feats["caller_turn_dur_cv"] = 0.0

    # --- 5. Silencios internos / ritmo global ---
    total_caller_speech = sum(caller_durs)
    feats["caller_speech_ratio"] = float(total_caller_speech / duration_s) if duration_s > 0 else 0.0
    feats["n_caller_turns"] = float(len(caller_turns))
    feats["n_agent_turns"] = float(len(agent_turns))
    feats["turns_per_min"] = float(len(caller_turns) / (duration_s / 60.0)) if duration_s > 0 else 0.0

    return feats
