"""Build frozen artifacts required by open-set single-frame inference.

Run once on a GPU node. The script does not search thresholds. It builds:
- pixel_normal_bank.npz from train-normal canonical images;
- wrn_normal_memory.npz from frozen WRN Layer2+Layer3 features;
- region_classifier.joblib from an audited development frame_index.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import dump

from maglev_anomaly.data import canonical_image, load_groups, load_map
from maglev_anomaly.pixel_anomaly import build_bank
from maglev_anomaly.region import FEATURE_NAMES, fit_predict
from maglev_anomaly.wrn_memory import WideResNetFeatureExtractor, build_wrn_memory_artifact


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_region_samples(frame_index: Path):
    features, labels = [], []
    with frame_index.open(newline="", encoding="utf-8-sig") as f:
        for frame in csv.DictReader(f):
            for region in json.loads(frame["region_records"]):
                y = int(region["y"])
                if y < 0:
                    continue
                features.append([float(region["features"][name]) for name in FEATURE_NAMES])
                labels.append(y)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    groups, _rejected = load_groups(config)
    map_x, map_y = load_map(config)
    canonical = []
    valid_reference = None
    for row in groups["train_normal"]:
        image, valid, _shape = canonical_image(row, map_x, map_y, True)
        canonical.append(image)
        if valid_reference is None:
            valid_reference = valid

    pixel_bank = build_bank(np.stack(canonical).astype(np.float32), valid_reference, config)
    pixel_path = args.out / "pixel_normal_bank.npz"
    np.savez_compressed(pixel_path, **{key: value.astype(np.float32) for key, value in pixel_bank.items()})

    extractor = WideResNetFeatureExtractor(device=args.device)
    wrn_path = args.out / "wrn_normal_memory.npz"
    try:
        build_wrn_memory_artifact(canonical, extractor, wrn_path, device=args.device)
    finally:
        extractor.close()

    x, y = load_region_samples(args.frame_index)
    classifier, _probability = fit_predict(x, y, config["region_classifier"]["model"])
    classifier_path = args.out / "region_classifier.joblib"
    dump(classifier, classifier_path)

    meta = {
        "train_normal_frames": len(canonical),
        "canonical_shape": list(map_x.shape),
        "wrn_model": "wide_resnet50_2",
        "wrn_layers": ["layer2", "layer3"],
        "region_feature_names": FEATURE_NAMES,
        "region_samples": len(y),
        "region_positive": int((y == 1).sum()),
        "region_negative": int((y == 0).sum()),
        "frozen_params": config["frozen_params"],
        "files": {},
    }
    for path in (pixel_path, wrn_path, classifier_path):
        meta["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (args.out / "artifacts_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
