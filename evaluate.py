"""Evaluate saved group-level predictions and write a compact metrics table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=Path("outputs/nested_cv_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.json"))
    args = parser.parse_args()
    data = pd.read_csv(args.predictions)
    required = {"target", "probability", "prediction", "id_measurement"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    tn, fp, fn, tp = confusion_matrix(data.target, data.prediction).ravel()
    metrics = {"groups_scored": int(len(data)),
               "unique_groups": int(data.id_measurement.nunique()),
               "accuracy": float(accuracy_score(data.target, data.prediction)),
               "precision": float(precision_score(data.target, data.prediction, zero_division=0)),
               "recall": float(recall_score(data.target, data.prediction, zero_division=0)),
               "mcc": float(matthews_corrcoef(data.target, data.prediction)),
               "roc_auc": float(roc_auc_score(data.target, data.probability)),
               "pr_auc": float(average_precision_score(data.target, data.probability)),
               "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                                    "fn": int(fn), "tp": int(tp)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

