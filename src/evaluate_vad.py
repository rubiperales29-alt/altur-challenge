"""
Audita qué tan bien el VAD propio (usado en /detect) se parece a los turns.json
"gold" que sí tienes en train/val. Esto te dice si vale la pena afinar los
parámetros de turn_detection.detect_turns antes de confiar en las features de timing.

Uso:
    python src/evaluate_vad.py --data-dir data
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from features.turn_detection import detect_turns
from features.feature_pipeline import _turns_from_json


def frame_level_agreement(gold_turns, pred_turns, duration_s, hop=0.01):
    n = int(duration_s / hop)
    gold_mask = np.zeros(n, dtype=bool)
    pred_mask = np.zeros(n, dtype=bool)
    for t in gold_turns:
        gold_mask[int(t.start / hop):int(t.end / hop)] = True
    for t in pred_turns:
        pred_mask[int(t.start / hop):int(t.end / hop)] = True
    agree = np.mean(gold_mask == pred_mask)
    return float(agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "manifest.csv")) as f:
        rows = list(csv.DictReader(f))

    scores = []
    for row in rows:
        anon_id = row["anon_id"]
        wav_path = os.path.join(args.data_dir, "audio", f"{anon_id}.wav")
        turns_path = os.path.join(args.data_dir, "turns", f"{anon_id}.json")
        if not (os.path.exists(wav_path) and os.path.exists(turns_path)):
            continue
        audio, sr = sf.read(wav_path, always_2d=True)
        ch0, ch1 = audio[:, 0], audio[:, 1]
        duration_s = len(ch0) / sr

        gold_caller, gold_agent = _turns_from_json(turns_path)
        pred_caller = detect_turns(ch0, sr)
        pred_agent = detect_turns(ch1, sr)

        agree0 = frame_level_agreement(gold_caller, pred_caller, duration_s)
        agree1 = frame_level_agreement(gold_agent, pred_agent, duration_s)
        scores.append((agree0 + agree1) / 2)
        print(f"{anon_id}: acuerdo caller={agree0:.3f} agente={agree1:.3f}")

    if scores:
        print(f"\nAcuerdo promedio VAD propio vs turns.json: {np.mean(scores):.3f}")
    else:
        print("No se encontraron pares wav+turns.json para comparar.")


if __name__ == "__main__":
    main()
