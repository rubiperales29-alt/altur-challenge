"""
Entrena el clasificador humano vs sintético.

Uso:
    python src/train.py --data-dir data --out models/model.joblib

Estructura esperada en data-dir:
    data/manifest.csv
    data/audio/<anon_id>.wav
    data/turns/<anon_id>.json   (opcional, no se usa por defecto en el entrenamiento
                                  real -- ver feature_pipeline.use_own_vad)

Modelo: GradientBoostingClassifier (robusto con pocos datos y features heterogéneas)
envuelto en CalibratedClassifierCV para que `confidence` sea honesto (Platt scaling),
tal como pide la evaluación del reto.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from features.feature_pipeline import extract_features_from_file, features_to_vector


def load_manifest(data_dir: str):
    rows = []
    with open(os.path.join(data_dir, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_dataset(rows, data_dir: str, split: str, use_asr: bool = False, feature_names=None):
    X, y, ids = [], [], []
    names = feature_names
    for row in rows:
        if row["split"] != split:
            continue
        anon_id = row["anon_id"]
        wav_path = os.path.join(data_dir, "audio", f"{anon_id}.wav")
        turns_path = os.path.join(data_dir, "turns", f"{anon_id}.json")
        if not os.path.exists(wav_path):
            print(f"  [skip] no existe {wav_path}")
            continue
        turns_path = turns_path if os.path.exists(turns_path) else None
        feats = extract_features_from_file(wav_path, turns_json_path=turns_path, use_asr=use_asr)
        vec, names = features_to_vector(feats, feature_names=names)
        X.append(vec)
        y.append(1 if row["label"] == "synthetic" else 0)
        ids.append(anon_id)
        print(f"  [ok] {anon_id} label={row['label']}")
    return np.array(X), np.array(y), ids, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="models/model.joblib")
    ap.add_argument("--use-asr", action="store_true", help="Activa features lingüísticas (más lento)")
    args = ap.parse_args()

    rows = load_manifest(args.data_dir)

    print("Extrayendo features de TRAIN...")
    X_train, y_train, ids_train, names = build_dataset(rows, args.data_dir, "train", use_asr=args.use_asr)
    print(f"\nTrain: {X_train.shape[0]} llamadas, {X_train.shape[1]} features")
    print(f"Distribución: {np.bincount(y_train)}  (0=human, 1=synthetic)")

    if len(np.unique(y_train)) < 2:
        print("ERROR: se necesitan ambas clases en train para entrenar.")
        return

    base_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )

    n_pos = int(np.sum(y_train))
    n_folds = min(5, n_pos, len(y_train) - n_pos) if n_pos > 1 else 2
    n_folds = max(2, n_folds)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv)
    clf.fit(X_train, y_train)

    # feature importance aproximada (del último base estimator fiteado internamente)
    try:
        importances = np.mean(
            [c.estimator.feature_importances_ for c in clf.calibrated_classifiers_], axis=0
        )
        top = sorted(zip(names, importances), key=lambda t: -t[1])[:15]
        print("\nTop features (importancia promedio):")
        for n, imp in top:
            print(f"  {n:30s} {imp:.4f}")
    except Exception as e:
        print(f"(no se pudo calcular importancia: {e})")

    print("\nEvaluando en VAL...")
    X_val, y_val, ids_val, _ = build_dataset(rows, args.data_dir, "val", use_asr=args.use_asr, feature_names=names)
    if len(X_val) > 0:
        proba = clf.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_val, pred)
        print(f"Accuracy val: {acc:.3f}")
        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, proba)
            print(f"AUC val: {auc:.3f}")
        print(classification_report(y_val, pred, target_names=["human", "synthetic"]))
    else:
        print("No hay datos de val disponibles todavía.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": clf, "feature_names": names}, args.out)
    print(f"\nModelo guardado en {args.out}")


if __name__ == "__main__":
    main()
