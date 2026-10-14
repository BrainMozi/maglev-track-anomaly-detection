#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train controlled YOLO ablations on A800 and produce an auditable report.

Selection uses the held-out validation video only.  The second 2025-11-14
video is touched once after model/config/threshold selection.  The final model
is then retrained for a fixed number of epochs on train+val and evaluated on
that untouched test video.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BUSINESS_NAMES = {0: "person", 1: "vehicle"}
COCO_VEHICLE_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
THRESHOLDS = np.round(np.arange(0.01, 0.81, 0.01), 2).tolist()


@dataclass
class Candidate:
    name: str
    yaml_name: str
    imgsz: int
    freeze: int | None = None


def read_yolo_gt(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = map(float, line.split()[:5])
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append((int(c), np.array([x1, y1, x2, y2], dtype=np.float32)))
    return boxes


def iou_one_to_many(box: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    if not others:
        return np.empty(0, dtype=np.float32)
    arr = np.stack(others)
    xx1 = np.maximum(box[0], arr[:, 0])
    yy1 = np.maximum(box[1], arr[:, 1])
    xx2 = np.minimum(box[2], arr[:, 2])
    yy2 = np.minimum(box[3], arr[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    b = np.maximum(0, arr[:, 2] - arr[:, 0]) * np.maximum(0, arr[:, 3] - arr[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def map_prediction_class(class_id: int, mode: str) -> int | None:
    if mode == "custom":
        return class_id if class_id in (0, 1) else None
    if class_id == 0:
        return 0
    if class_id in COCO_VEHICLE_IDS:
        return 1
    return None


def collect_predictions(model_path: Path, image_dir: Path, label_dir: Path, imgsz: int, mode: str) -> list[dict[str, Any]]:
    images = sorted(image_dir.glob("*.jpg"))
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(p) for p in images], imgsz=imgsz, conf=0.001, iou=0.70,
        max_det=300, device=0, batch=min(16, len(images)), verbose=False,
    )
    records = []
    for image_path, result in zip(images, results):
        h, w = result.orig_shape
        gt = read_yolo_gt(label_dir / f"{image_path.stem}.txt", w, h)
        pred = []
        if result.boxes is not None:
            for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                mapped = map_prediction_class(int(cls), mode)
                if mapped is not None:
                    pred.append((mapped, float(conf), xyxy.astype(np.float32)))
        records.append({"image": image_path, "shape": (h, w), "gt": gt, "pred": pred})
    del model, results
    gc.collect()
    torch.cuda.empty_cache()
    return records


def evaluate(records: list[dict[str, Any]], thresholds: dict[int, float], iou_threshold: float = 0.5) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    totals = [0, 0, 0]
    for class_id in (0, 1):
        tp = fp = fn = 0
        threshold = thresholds[class_id]
        for rec in records:
            gt = [b for c, b in rec["gt"] if c == class_id]
            pred = sorted([(s, b) for c, s, b in rec["pred"] if c == class_id and s >= threshold], reverse=True, key=lambda x: x[0])
            unmatched = list(range(len(gt)))
            for score, box in pred:
                if not unmatched:
                    fp += 1
                    continue
                overlaps = iou_one_to_many(box, [gt[i] for i in unmatched])
                best_local = int(np.argmax(overlaps))
                if float(overlaps[best_local]) >= iou_threshold:
                    tp += 1
                    unmatched.pop(best_local)
                else:
                    fp += 1
            fn += len(unmatched)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
        per_class[BUSINESS_NAMES[class_id]] = {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "f2": f2}
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
    tp, fp, fn = totals
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
    macro_f2 = float(np.mean([v["f2"] for v in per_class.values()]))
    return {"iou": iou_threshold, "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "f2": f2, "macro_f2": macro_f2}, "per_class": per_class}


def tune_thresholds(records: list[dict[str, Any]]) -> tuple[dict[int, float], dict[str, Any], list[dict[str, float]]]:
    chosen: dict[int, float] = {}
    curves: list[dict[str, float]] = []
    for class_id in (0, 1):
        best = None
        for threshold in THRESHOLDS:
            metrics = evaluate(records, {class_id: threshold, 1 - class_id: 1.0})["per_class"][BUSINESS_NAMES[class_id]]
            row = {"class_id": class_id, "threshold": threshold, "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "f2": metrics["f2"]}
            curves.append(row)
            key = (row["f2"], row["f1"], row["precision"], threshold)
            if best is None or key > best[0]:
                best = (key, threshold)
        assert best is not None
        chosen[class_id] = best[1]
    return chosen, evaluate(records, chosen), curves


def best_epoch(run_dir: Path) -> tuple[int, dict[str, float]]:
    with (run_dir / "results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    metrics = {k: float(v) for k, v in row.items() if k.startswith("metrics/")}
    return int(float(row["epoch"])), metrics


def train_one(base_weights: Path, data_yaml: Path, project: Path, c: Candidate, epochs: int = 140, val: bool = True, name: str | None = None) -> tuple[Path, Path, int, dict[str, float]]:
    model = YOLO(str(base_weights))
    run_name = name or c.name
    model.train(
        data=str(data_yaml), epochs=epochs, patience=30, batch=12 if c.imgsz >= 1536 else 20,
        imgsz=c.imgsz, device=0, workers=6, project=str(project), name=run_name,
        exist_ok=False, pretrained=True, cls_remap=True, optimizer="AdamW",
        lr0=3e-4, lrf=0.05, cos_lr=True, warmup_epochs=3.0,
        weight_decay=5e-4, freeze=c.freeze, deterministic=True, seed=42,
        cache="ram", amp=True, plots=True, val=val,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        translate=0.05, scale=0.15, hsv_h=0.01, hsv_s=0.40, hsv_v=0.30,
        erasing=0.0, close_mosaic=0, save_period=-1, verbose=True,
    )
    run_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)
    if val:
        epoch, metrics = best_epoch(run_dir)
    else:
        epoch, metrics = epochs, {}
        best = run_dir / "weights" / "last.pt"
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best, run_dir, epoch, metrics


def draw_boxes(image: Image.Image, boxes: Iterable[tuple[int, float, np.ndarray]], title: str) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    colors = {0: (35, 220, 100), 1: (255, 170, 30)}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw.rectangle((0, 0, out.width, 38), fill=(15, 18, 24))
    draw.text((12, 6), title, fill="white", font=title_font)
    for class_id, score, box in boxes:
        x1, y1, x2, y2 = map(float, box)
        color = colors[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = BUSINESS_NAMES[class_id] if score < 0 else f"{BUSINESS_NAMES[class_id]} {score:.2f}"
        draw.text((x1 + 2, max(40, y1 - 20)), label, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def prediction_boxes(record: dict[str, Any], thresholds: dict[int, float]) -> list[tuple[int, float, np.ndarray]]:
    return [(c, s, b) for c, s, b in record["pred"] if s >= thresholds[c]]


def make_montage(out_path: Path, baseline_records: list[dict[str, Any]], final_records: list[dict[str, Any]], baseline_thr: dict[int, float], final_thr: dict[int, float], limit: int = 6) -> None:
    chosen = np.linspace(0, len(final_records) - 1, min(limit, len(final_records)), dtype=int).tolist()
    rows = []
    panel_w = 640
    for idx in chosen:
        rec_b, rec_f = baseline_records[idx], final_records[idx]
        with Image.open(rec_f["image"]) as im:
            gt_boxes = [(c, -1.0, b) for c, b in rec_f["gt"]]
            panels = [
                draw_boxes(im, gt_boxes, f"Ground truth | {rec_f['image'].stem}"),
                draw_boxes(im, prediction_boxes(rec_b, baseline_thr), "COCO pretrained baseline"),
                draw_boxes(im, prediction_boxes(rec_f, final_thr), "Fine-tuned final model"),
            ]
        scaled = []
        for panel in panels:
            h = round(panel.height * panel_w / panel.width)
            scaled.append(panel.resize((panel_w, h), Image.Resampling.LANCZOS))
        row = Image.new("RGB", (panel_w * 3, scaled[0].height), "white")
        for j, panel in enumerate(scaled):
            row.paste(panel, (j * panel_w, 0))
        rows.append(row)
    canvas = Image.new("RGB", (panel_w * 3, sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(out_path, quality=94)


def make_metric_plot(out_path: Path, baseline: dict[str, Any], final: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    labels = ["Precision", "Recall", "F1", "F2"]
    keys = ["precision", "recall", "f1", "f2"]
    base = [baseline["overall"][k] for k in keys]
    tuned = [final["overall"][k] for k in keys]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    ax.bar(x - 0.18, base, 0.36, label="COCO pretrained", color="#718096")
    ax.bar(x + 0.18, tuned, 0.36, label="Fine-tuned final", color="#2b6cb0")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score on held-out test video")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, vals in enumerate((base, tuned)):
        for j, v in enumerate(vals):
            ax.text(j + (-0.18 if i == 0 else 0.18), v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    roi = dataset_root / "dataset_roi"
    work = args.work.resolve()
    if work.exists():
        raise SystemExit(f"refusing to overwrite existing work directory: {work}")
    (work / "report").mkdir(parents=True)
    project = work / "runs"

    candidates = [
        Candidate("vehicle_random_1280", "data_vehicle.yaml", 1280, None),
        Candidate("car_init_1280", "data_car.yaml", 1280, None),
        Candidate("car_init_freeze10_1280", "data_car.yaml", 1280, 10),
        Candidate("car_init_1536", "data_car.yaml", 1536, None),
    ]

    summary: dict[str, Any] = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "candidates": [],
        "selection_rule": "highest macro-F2 on held-out validation video; per-class thresholds tuned on validation only",
    }
    candidate_runtime: list[dict[str, Any]] = []
    val_image_dir, val_label_dir = roi / "images" / "val", roi / "labels" / "val"

    for candidate in candidates:
        best, run_dir, epoch, train_metrics = train_one(args.base_weights, roi / candidate.yaml_name, project, candidate)
        val_records = collect_predictions(best, val_image_dir, val_label_dir, candidate.imgsz, "custom")
        thresholds, val_metrics, curves = tune_thresholds(val_records)
        item = {
            "candidate": asdict(candidate), "best_weights": str(best), "run_dir": str(run_dir),
            "best_epoch": epoch, "ultralytics_best_row": train_metrics,
            "thresholds": {BUSINESS_NAMES[k]: v for k, v in thresholds.items()},
            "validation": val_metrics,
        }
        summary["candidates"].append(item)
        candidate_runtime.append({"candidate": candidate, "best": best, "epoch": epoch, "thresholds": thresholds, "val": val_metrics})
        with (work / "report" / f"threshold_curve_{candidate.name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("class_id", "threshold", "precision", "recall", "f1", "f2"))
            writer.writeheader(); writer.writerows(curves)

    selected = max(candidate_runtime, key=lambda x: (x["val"]["overall"]["macro_f2"], x["val"]["overall"]["f2"], x["val"]["overall"]["precision"]))
    summary["selected_candidate"] = selected["candidate"].name

    # Resolution sweep is validation-only; threshold is retuned at each resolution.
    resolution_sweep = []
    for imgsz in (960, 1280, 1536):
        rec = collect_predictions(selected["best"], val_image_dir, val_label_dir, imgsz, "custom")
        thresholds, metrics, _ = tune_thresholds(rec)
        resolution_sweep.append({"imgsz": imgsz, "thresholds": {BUSINESS_NAMES[k]: v for k, v in thresholds.items()}, "validation": metrics})
    chosen_resolution = max(resolution_sweep, key=lambda x: (x["validation"]["overall"]["macro_f2"], x["validation"]["overall"]["f2"]))
    final_imgsz = int(chosen_resolution["imgsz"])
    final_thresholds = {0: float(chosen_resolution["thresholds"]["person"]), 1: float(chosen_resolution["thresholds"]["vehicle"])}
    summary["resolution_sweep"] = resolution_sweep
    summary["selected_inference_resolution"] = final_imgsz
    summary["selected_thresholds"] = chosen_resolution["thresholds"]

    # Tune a COCO-pretrained baseline on the same validation view for a fair comparison.
    baseline_val = collect_predictions(args.base_weights, val_image_dir, val_label_dir, final_imgsz, "coco")
    baseline_thresholds, baseline_val_metrics, _ = tune_thresholds(baseline_val)
    summary["baseline_validation"] = {"thresholds": {BUSINESS_NAMES[k]: v for k, v in baseline_thresholds.items()}, "metrics": baseline_val_metrics}

    # Final deployment training: fixed epoch count, train+val, no test-driven checkpointing.
    final_epochs = max(12, int(selected["epoch"]))
    selected_c: Candidate = selected["candidate"]
    final_yaml_name = "data_final_car.yaml" if "car" in selected_c.yaml_name else "data_final_vehicle.yaml"
    final_candidate = Candidate("final_deployment", final_yaml_name, selected_c.imgsz, selected_c.freeze)
    final_best, final_run_dir, _, _ = train_one(
        args.base_weights, roi / final_yaml_name, project, final_candidate,
        epochs=final_epochs, val=False, name="final_deployment",
    )
    summary["final_training"] = {"weights": str(final_best), "run_dir": str(final_run_dir), "epochs": final_epochs, "train_images": "train+val", "test_used_for_training_or_checkpointing": False}

    test_image_dir, test_label_dir = roi / "images" / "test", roi / "labels" / "test"
    final_test_records = collect_predictions(final_best, test_image_dir, test_label_dir, final_imgsz, "custom")
    baseline_test_records = collect_predictions(args.base_weights, test_image_dir, test_label_dir, final_imgsz, "coco")
    final_test_metrics = evaluate(final_test_records, final_thresholds)
    baseline_test_metrics = evaluate(baseline_test_records, baseline_thresholds)
    summary["heldout_test"] = {
        "fine_tuned_final": final_test_metrics,
        "coco_pretrained_baseline": baseline_test_metrics,
        "threshold_policy": "thresholds fixed from validation before test evaluation",
    }

    # Test PR curves are diagnostic only and do not alter the selected thresholds.
    _, _, final_test_curves = tune_thresholds(final_test_records)
    with (work / "report" / "test_threshold_curve_diagnostic_only.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("class_id", "threshold", "precision", "recall", "f1", "f2"))
        writer.writeheader(); writer.writerows(final_test_curves)

    make_montage(work / "report" / "heldout_test_comparison.jpg", baseline_test_records, final_test_records, baseline_thresholds, final_thresholds)
    make_metric_plot(work / "report" / "heldout_test_metrics.png", baseline_test_metrics, final_test_metrics)
    (work / "report" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # A concise machine-readable row for shell monitoring.
    concise = {
        "selected_candidate": summary["selected_candidate"],
        "imgsz": final_imgsz,
        "thresholds": summary["selected_thresholds"],
        "baseline_test": baseline_test_metrics["overall"],
        "final_test": final_test_metrics["overall"],
        "final_weights": str(final_best),
    }
    print("FINAL_SUMMARY=" + json.dumps(concise, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
