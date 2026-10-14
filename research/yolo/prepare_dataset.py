#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit X-AnyLabeling JSON and build leakage-resistant YOLO datasets.

The split is by source video, never by random adjacent frames:
  train: 2025-11-04 videos (+ one early labeled frame)
  val:   first 2025-11-14 video
  test:  second 2025-11-14 video

Two image views are produced:
  full: original 1920x1080 frame
  roi:  crop (640, 270, 1920, 1080), the actual monitored safety area

Class id 1 remains the business class "vehicle" in label files.  Separate YAML
files expose it as either "vehicle" (random custom head row) or "car" (reuse
COCO car head initialization); inference maps it back to "vehicle".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

CLASSES = {"person": 0, "vehicle": 1}
ROI = (640, 270, 1920, 1080)
SEGMENT_SPLIT = {
    "ch07_20251104161952_20251112171033": "train",
    "ch07_20251104180001_20251112174239": "train",
    "ch07_20251104184021_20251112174322": "train",
    "ch07_20251114141501_20251118171616": "val",
    "ch07_20251114144033_20251118171804": "test",
}


def segment_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def yolo_line(class_id: int, box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    vals = (cx, cy, bw, bh)
    if not all(0.0 <= v <= 1.0 for v in vals) or bw <= 0 or bh <= 0:
        raise ValueError(f"invalid normalized box: {vals}")
    return f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def clip_box(box: tuple[float, float, float, float], crop: tuple[int, int, int, int]) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    l, t, r, b = crop
    ox1, oy1, ox2, oy2 = max(x1, l), max(y1, t), min(x2, r), min(y2, b)
    if ox2 <= ox1 or oy2 <= oy1:
        return None
    original_area = max(1e-9, (x2 - x1) * (y2 - y1))
    retained = (ox2 - ox1) * (oy2 - oy1) / original_area
    if retained < 0.50:
        return None
    return ox1 - l, oy1 - t, ox2 - l, oy2 - t


def write_yaml(path: Path, root: Path, class1_name: str, final_train: bool = False, yaml_root: str | None = None) -> None:
    root_posix = yaml_root or root.resolve().as_posix()
    if final_train:
        train_value = "\n  - images/train\n  - images/val"
    else:
        train_value = "images/train"
    path.write_text(
        f"path: {root_posix}\n"
        f"train: {train_value}\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: person\n"
        f"  1: {class1_name}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-root", type=str, default=None, help="remote absolute directory corresponding to --out")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    if not src.is_dir():
        raise SystemExit(f"source does not exist: {src}")
    if out.exists():
        raise SystemExit(f"output already exists: {out}")

    json_paths = sorted(src.glob("*.json"))
    if not json_paths:
        raise SystemExit("no JSON labels found")

    records: list[dict] = []
    class_counts = Counter()
    split_counts = Counter()
    segment_counts = Counter()
    size_counts = Counter()
    issues: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    background_images = 0

    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))
        stem = jp.stem
        seg = segment_of(stem)
        split = SEGMENT_SPLIT.get(seg)
        if split is None:
            issues.append(f"unknown segment: {stem}")
            continue
        image_name = Path(data.get("imagePath") or f"{stem}.jpg").name
        image_path = src / image_name
        if not image_path.exists():
            issues.append(f"missing image: {image_name}")
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        if (width, height) != (1920, 1080):
            issues.append(f"unexpected size {width}x{height}: {image_name}")

        boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        for idx, shape in enumerate(data.get("shapes", [])):
            label = shape.get("label")
            if label not in CLASSES:
                issues.append(f"unknown label {label!r}: {stem} shape {idx}")
                continue
            if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) < 2:
                issues.append(f"non-rectangle/invalid points: {stem} shape {idx}")
                continue
            (xa, ya), (xb, yb) = shape["points"][:2]
            x1, x2 = sorted((float(xa), float(xb)))
            y1, y2 = sorted((float(ya), float(yb)))
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                issues.append(f"empty box: {stem} shape {idx}")
                continue
            box = (x1, y1, x2, y2)
            boxes.append((CLASSES[label], box))
            class_counts[label] += 1
            area = (x2 - x1) * (y2 - y1)
            size_counts["tiny<32^2" if area < 32**2 else "small<64^2" if area < 64**2 else "medium<96^2" if area < 96**2 else "large"] += 1

        if not boxes:
            # A JSON file with an empty shapes list is an explicitly reviewed
            # negative/background image, unlike an unlabeled JPG.
            background_images += 1
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[digest].append(stem)
        records.append({"stem": stem, "segment": seg, "split": split, "image": image_path, "width": width, "height": height, "boxes": boxes})
        split_counts[split] += 1
        segment_counts[seg] += 1

    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    if duplicate_groups:
        issues.extend(f"exact duplicate images: {group}" for group in duplicate_groups)
    if issues:
        fatal = [x for x in issues if not x.startswith("unexpected size")]
        (out.parent / "prepare_issues.txt").write_text("\n".join(issues) + "\n", encoding="utf-8")
        if fatal:
            raise SystemExit("fatal annotation issues found; see prepare_issues.txt")

    for view in ("full", "roi"):
        root = out / f"dataset_{view}"
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        for rec in records:
            stem, split = rec["stem"], rec["split"]
            dst_img = root / "images" / split / f"{stem}.jpg"
            dst_lab = root / "labels" / split / f"{stem}.txt"
            lines: list[str] = []
            if view == "full":
                with Image.open(rec["image"]) as im:
                    im.save(dst_img, quality=95, subsampling=0)
                for class_id, box in rec["boxes"]:
                    lines.append(yolo_line(class_id, box, rec["width"], rec["height"]))
                out_w, out_h = rec["width"], rec["height"]
            else:
                with Image.open(rec["image"]) as im:
                    im.crop(ROI).save(dst_img, quality=95, subsampling=0)
                out_w, out_h = ROI[2] - ROI[0], ROI[3] - ROI[1]
                for class_id, box in rec["boxes"]:
                    clipped = clip_box(box, ROI)
                    if clipped is not None:
                        lines.append(yolo_line(class_id, clipped, out_w, out_h))
            if rec["boxes"] and not lines:
                raise SystemExit(f"all positive boxes lost in {view} view: {stem}")
            dst_lab.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
            manifest_rows.append((stem, rec["segment"], split, view, out_w, out_h, len(lines)))

        yaml_root = f"{args.remote_root.rstrip('/')}/dataset_{view}" if args.remote_root else None
        write_yaml(root / "data_car.yaml", root, "car", final_train=False, yaml_root=yaml_root)
        write_yaml(root / "data_vehicle.yaml", root, "vehicle", final_train=False, yaml_root=yaml_root)
        write_yaml(root / "data_final_car.yaml", root, "car", final_train=True, yaml_root=yaml_root)
        write_yaml(root / "data_final_vehicle.yaml", root, "vehicle", final_train=True, yaml_root=yaml_root)
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(("stem", "segment", "split", "view", "width", "height", "instances"))
            writer.writerows(manifest_rows)

    audit = {
        "source": str(src),
        "output": str(out),
        "roi": ROI,
        "labeled_images": len(records),
        "instances": sum(class_counts.values()),
        "background_images_with_empty_json": background_images,
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "segment_counts": dict(segment_counts),
        "size_counts_original_pixels": dict(size_counts),
        "exact_duplicate_groups": duplicate_groups,
        "unlabeled_images_excluded": len(list(src.glob("*.jpg"))) - len(records),
        "notes": [
            "Unlabeled JPG files are excluded, not treated as negatives, because they were not verified empty.",
            "Split is by source video to avoid adjacent-frame leakage.",
            "The business class vehicle can be exposed as car during training to reuse the COCO car head row.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
