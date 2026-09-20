"""Repeated group CV with fold-internal MCC threshold selection."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def best_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, np.quantile(probability, np.linspace(0, 1, 201)), 1.0])
    score = [matthews_corrcoef(y, probability >= value) for value in candidates]
    return float(candidates[int(np.argmax(score))])


def model(seed: int, y: np.ndarray) -> CatBoostClassifier:
    weight = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    return CatBoostClassifier(iterations=350, depth=5, learning_rate=.03,
                              l2_leaf_reg=5, class_weights=[1.0, weight],
                              loss_function="Logloss", random_seed=seed,
                              verbose=False, allow_writing_files=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark/manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("outputs/frtpf_features.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nested_cv_predictions.csv"))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    features = pd.read_parquet(args.features)
    frame = manifest[["id_measurement", "target_any"]].merge(
        features, on="id_measurement", validate="one_to_one")
    x = frame.drop(columns=["id_measurement", "target_any"]).to_numpy(np.float32)
    y = frame.target_any.to_numpy(int)
    groups = frame.id_measurement.to_numpy()
    rows = []
    for repeat in range(args.repeats):
        outer = StratifiedGroupKFold(args.splits, shuffle=True,
                                     random_state=args.seed + repeat)
        for fold, (train, test) in enumerate(outer.split(x, y, groups), 1):
            inner = StratifiedGroupKFold(3, shuffle=True,
                                         random_state=args.seed + 100 + repeat * 10 + fold)
            inner_probability = np.full(len(train), np.nan)
            for inner_fold, (fit, validation) in enumerate(
                    inner.split(x[train], y[train], groups[train])):
                fitted = model(args.seed + inner_fold, y[train][fit])
                fitted.fit(x[train][fit], y[train][fit])
                inner_probability[validation] = fitted.predict_proba(x[train][validation])[:, 1]
            threshold = best_threshold(y[train], inner_probability)
            fitted = model(args.seed + 1000 + repeat * 10 + fold, y[train])
            fitted.fit(x[train], y[train])
            probability = fitted.predict_proba(x[test])[:, 1]
            for local, index in enumerate(test):
                rows.append({"id_measurement": int(groups[index]), "target": int(y[index]),
                             "probability": float(probability[local]),
                             "prediction": int(probability[local] >= threshold),
                             "threshold": threshold, "repeat": repeat + 1, "fold": fold})
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"PR-AUC={average_precision_score(output.target, output.probability):.4f}")
    print(f"ROC-AUC={roc_auc_score(output.target, output.probability):.4f}")
    print(f"MCC={matthews_corrcoef(output.target, output.prediction):.4f}")


if __name__ == "__main__":
    main()

