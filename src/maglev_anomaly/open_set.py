"""Frozen single-frame open-set track anomaly inference.

This module performs no parameter search and no training. It consumes frozen
artifacts: canonical map, pixel normal bank, WRN normal memory, region
classifier, and fixed thresholds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from . import graphcut as gc
from . import region as rg
from .data import (
    backproject_fillpoly,
    canonical_from_array,
    gray_from_rgb,
    rgb_to_lab,
)
from .pixel_anomaly import score_frame


class OpenSetSingleFrameProcessor:
    def __init__(
        self,
        config: dict,
        *,
        heatmap_scorer: Any,
        pixel_bank: str | Path | dict[str, np.ndarray],
        region_classifier: str | Path | Any,
    ):
        self.config = config
        canonical_path = config["assets"]["canonical_map"]
        z = np.load(canonical_path)
        self.map_x = (z["map_x"] if "map_x" in z else z["canonical_map_x"]).astype(np.float32)
        self.map_y = (z["map_y"] if "map_y" in z else z["canonical_map_y"]).astype(np.float32)
        if tuple(self.map_x.shape) != tuple(config["canonical_shape"]):
            raise RuntimeError("canonical map shape does not match config")
        if isinstance(pixel_bank, (str, Path)):
            bank_file = np.load(pixel_bank)
            self.pixel_bank = {key: bank_file[key].astype(np.float32) for key in bank_file.files}
        else:
            self.pixel_bank = {key: np.asarray(value, dtype=np.float32) for key, value in pixel_bank.items()}
        if isinstance(region_classifier, (str, Path)):
            from joblib import load
            region_classifier = load(region_classifier)
        self.classifier = region_classifier
        self.classifier_threshold = float(config.get("region_classifier", {}).get("threshold", 0.5))
        self.heatmap_scorer = heatmap_scorer
        self.params = dict(config["frozen_params"])

    def _run_cut(self, frame_cache):
        downsample = int(self.config["graphcut"]["downsample"])
        anomaly = frame_cache["anomaly"]
        heat = frame_cache["heat_full"]
        lab = frame_cache["lab"]
        valid = frame_cache["valid"]
        if downsample > 1:
            shape = (max(1, anomaly.shape[0] // downsample), max(1, anomaly.shape[1] // downsample))
            anomaly_work = gc.resize_map(anomaly, shape)
            heat_work = gc.resize_map(heat, shape)
            lab_work = np.stack([gc.resize_map(lab[..., c], shape) for c in range(3)], axis=-1)
            valid_work = gc.resize_map(valid.astype(np.float32), shape, order=0) > 0.5
        else:
            anomaly_work, heat_work, lab_work, valid_work = anomaly, heat, lab, valid
        seed_work = heat_work >= self.params["seed_threshold"]
        support = anomaly_work >= self.params["support_threshold"]
        mask = gc.graph_cut(
            seed_work,
            anomaly_work,
            lab_work,
            valid_work,
            support,
            self.params["unary_weight"],
            self.params["smooth"],
            self.params["bias"],
            self.config["graphcut"]["edge_sigma"],
            self.config["graphcut"]["hard_capacity"],
        )
        mask = gc.keep_seed_components(mask, seed_work)
        full = gc.resize_map(mask.astype(np.float32), anomaly.shape, order=0) > 0.5
        seed_full = heat >= self.params["seed_threshold"]
        return full, seed_full

    def process(
        self,
        image_rgb: np.ndarray,
        *,
        dx_sub: float,
        dy_sub: float,
        calibrated_heatmap: np.ndarray | None = None,
    ) -> dict[str, Any]:
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        canonical, valid, original_shape = canonical_from_array(
            image_rgb, self.map_x, self.map_y, dx_sub, dy_sub, return_valid=True
        )
        if calibrated_heatmap is None:
            _raw, calibrated_heatmap = self.heatmap_scorer.score(canonical)
        anomaly = score_frame(canonical.astype(np.float32), self.pixel_bank, valid, self.config)
        heat_full = gc.resize_map(np.asarray(calibrated_heatmap, dtype=np.float32), anomaly.shape)
        gray = gray_from_rgb(canonical)
        gradient = np.hypot(np.gradient(gray, axis=1), np.gradient(gray, axis=0)).astype(np.float32)
        cache = {
            "anomaly": anomaly,
            "heat_full": heat_full,
            "lab": rgb_to_lab(canonical),
            "valid": valid,
        }
        candidate_mask, seed = self._run_cut(cache)
        labels, count = ndi.label(candidate_mask)
        features = rg.extract_features(labels, count, heat_full, anomaly, gradient)
        final_mask = np.zeros_like(candidate_mask)
        regions = []
        for index in range(count):
            label = index + 1
            region_mask = labels == label
            probability = float(self.classifier.predict_proba(features[index:index+1])[0, 1])
            verdict = "keep" if probability >= self.classifier_threshold else "drop"
            if verdict == "keep":
                final_mask |= region_mask
            ys, xs = np.where(region_mask)
            regions.append({
                "label": label,
                "verdict": verdict,
                "probability": probability,
                "area_canonical": int(region_mask.sum()),
                "bbox_canonical_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                "features": {name: float(features[index, i]) for i, name in enumerate(rg.FEATURE_NAMES)},
            })
        row = {"dx_sub": float(dx_sub), "dy_sub": float(dy_sub)}
        original_mask = backproject_fillpoly(final_mask, row, self.map_x, self.map_y, original_shape)
        return {
            "canonical_image": canonical,
            "canonical_mask": final_mask,
            "original_mask": original_mask,
            "seed": seed,
            "pixel_anomaly": anomaly,
            "calibrated_heatmap": np.asarray(calibrated_heatmap, dtype=np.float32),
            "regions": regions,
            "summary": {
                "candidate_regions": count,
                "kept_regions": sum(r["verdict"] == "keep" for r in regions),
                "canonical_mask_pixels": int(final_mask.sum()),
                "original_mask_pixels": int(original_mask.sum()),
            },
        }
