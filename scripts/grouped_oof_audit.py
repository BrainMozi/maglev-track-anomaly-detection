"""Leave-one-segment-out audit for the lightweight region classifier.

This script reads frame_index.csv produced by scripts/run_full.py. It does not
re-run the visual backbone or Graph Cut. Ambiguous candidates (y=-1) are
excluded. The report is an out-of-fold audit of the region classifier only;
it is not an independent end-to-end test because front-end parameters may have
been selected on the same development frames.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

FEATURE_NAMES = [
    "heat_peak", "heat_mean", "pix_peak", "pix_mean",
    "area", "compactness", "boundary_grad",
]


def load_candidates(path: Path):
    frames = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    candidates = []
    for frame in frames:
        for region in json.loads(frame["region_records"]):
            if int(region["y"]) < 0:
                continue
            candidates.append({
                "filename": frame["filename"],
                "group": frame["group"],
                "segment": frame["segment"],
                "y": int(region["y"]),
                "features": [float(region["features"][name]) for name in FEATURE_NAMES],
            })
    return frames, candidates


def fit_model(rows):
    x = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.asarray([r["y"] for r in rows], dtype=int)
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    weights = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float32)
    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=2, learning_rate=0.1, random_state=0
    )
    model.fit(x, y, sample_weight=weights)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_index", type=Path)
    parser.add_argument("--out", type=Path, default=Path("grouped_oof_report.json"))
    args = parser.parse_args()

    frames, rows = load_candidates(args.frame_index)
    segments = sorted({r["segment"] for r in rows})
    predictions = []
    folds = []

    for segment in segments:
        train = [r for r in rows if r["segment"] != segment]
        test = [r for r in rows if r["segment"] == segment]
        train_labels = {r["y"] for r in train}
        if train_labels != {0, 1}:
            raise RuntimeError(f"fold {segment} lacks one class in training: {train_labels}")
        model = fit_model(train)
        x_test = np.asarray([r["features"] for r in test], dtype=np.float32)
        y_true = np.asarray([r["y"] for r in test], dtype=int)
        probability = model.predict_proba(x_test)[:, 1]
        y_pred = (probability >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        folds.append({
            "held_out_segment": segment,
            "train_candidates": len(train),
            "test_candidates": len(test),
            "train_pos": sum(r["y"] == 1 for r in train),
            "train_neg": sum(r["y"] == 0 for r in train),
            "test_pos": int((y_true == 1).sum()),
            "test_neg": int((y_true == 0).sum()),
            "confusion_matrix_labels_0_1": cm.tolist(),
        })
        for row, pred, prob in zip(test, y_pred, probability):
            predictions.append({**row, "prediction": int(pred), "probability": float(prob)})

    y_true = np.asarray([r["y"] for r in predictions], dtype=int)
    y_pred = np.asarray([r["prediction"] for r in predictions], dtype=int)
    probability = np.asarray([r["probability"] for r in predictions], dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    by_frame = defaultdict(list)
    for row in predictions:
        by_frame[row["filename"]].append(row)
    abnormal = [f for f in frames if f["group"] == "excluded_mixed"]
    normal = [f for f in frames if f["group"] == "val_normal"]
    abnormal_hits = sum(
        any(r["y"] == 1 and r["prediction"] == 1 for r in by_frame[f["filename"]])
        for f in abnormal
    )
    normal_false_frames = sum(
        any(r["prediction"] == 1 for r in by_frame[f["filename"]])
        for f in normal
    )

    report = {
        "scope": "region classifier leave-one-segment-out audit; front-end remains development-selected",
        "feature_names": FEATURE_NAMES,
        "segments": segments,
        "folds": folds,
        "overall": {
            "candidates": len(predictions),
            "positives": int((y_true == 1).sum()),
            "negatives": int((y_true == 0).sum()),
            "confusion_matrix_labels_0_1": cm.tolist(),
            "positive_precision": float(cm[1, 1] / max(1, cm[0, 1] + cm[1, 1])),
            "positive_recall": float(cm[1, 1] / max(1, cm[1, 0] + cm[1, 1])),
            "roc_auc": float(roc_auc_score(y_true, probability)),
            "average_precision": float(average_precision_score(y_true, probability)),
            "development_abnormal_frame_hits_after_oof_classifier": abnormal_hits,
            "development_abnormal_frames": len(abnormal),
            "normal_frames_with_oof_false_region": normal_false_frames,
            "normal_frames": len(normal),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
