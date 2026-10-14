"""Unified per-frame output for known YOLO objects and open-set anomalies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UnifiedSingleFrameProcessor:
    def __init__(self, known_detector, open_set_processor, registrar):
        self.known_detector = known_detector
        self.open_set_processor = open_set_processor
        self.registrar = registrar

    @staticmethod
    def _box_mask_overlap(box, mask):
        h, w = mask.shape
        x1, y1, x2, y2 = box
        x1 = max(0, min(w, int(np.floor(x1))))
        y1 = max(0, min(h, int(np.floor(y1))))
        x2 = max(0, min(w, int(np.ceil(x2))))
        y2 = max(0, min(h, int(np.ceil(y2))))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        intersection = int(mask[y1:y2, x1:x2].sum()) if area else 0
        return intersection, intersection / max(1, area)

    @staticmethod
    def render(image_rgb, detections, anomaly_mask):
        output = np.asarray(image_rgb, dtype=np.uint8).copy()
        anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
        output[anomaly_mask] = (
            0.55 * output[anomaly_mask] + 0.45 * np.array([255, 30, 20])
        ).astype(np.uint8)
        image = Image.fromarray(output)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        colors = {"person": (30, 220, 110), "vehicle": (255, 165, 30)}
        for detection in detections:
            x1, y1, x2, y2 = detection["xyxy"]
            color = colors.get(detection["class_name"], (50, 180, 255))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            if detection.get("overlaps_open_set"):
                label += " + anomaly"
            text_box = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_box, fill=(0, 0, 0))
            draw.text((x1, y1), label, fill=color, font=font)
        return np.asarray(image)

    def process(self, image_rgb: np.ndarray, *, filename: str | None = None) -> dict[str, Any]:
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        detections = self.known_detector.predict(image_rgb)
        registration = self.registrar.estimate(image_rgb)
        if registration["valid"]:
            open_result = self.open_set_processor.process(
                image_rgb,
                dx_sub=registration["dx_sub"],
                dy_sub=registration["dy_sub"],
            )
            anomaly_mask = open_result["original_mask"]
            open_summary = open_result["summary"]
            regions = open_result["regions"]
        else:
            anomaly_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
            open_result = None
            open_summary = {"status": "skipped_registration_rejected", "kept_regions": 0}
            regions = []
        for detection in detections:
            intersection, ratio = self._box_mask_overlap(detection["xyxy"], anomaly_mask)
            detection["open_set_intersection_pixels"] = intersection
            detection["open_set_box_overlap_ratio"] = ratio
            detection["overlaps_open_set"] = intersection > 0
        overlay = self.render(image_rgb, detections, anomaly_mask)
        known_status = "ok" if detections else "ok_empty"
        open_status = ("ok" if open_summary.get("kept_regions", 0) else "ok_empty") if registration["valid"] else "registration_rejected"
        record = {
            "schema_version": "ufp/0.1",
            "filename": filename,
            "image_shape": list(image_rgb.shape),
            "coordinates": {"known_objects": "original", "open_set_comparison": "original"},
            "registration": registration,
            "known_objects": detections,
            "open_set": {"summary": open_summary, "regions": regions},
            "producer_status": {"known_object_yolo": known_status, "open_set_anomaly": open_status},
            "combined": {
                "known_object_count": len(detections),
                "open_set_region_count": int(open_summary.get("kept_regions", 0)),
                "known_objects_overlapping_open_set": sum(d["overlaps_open_set"] for d in detections),
                "has_any_detection": bool(detections or open_summary.get("kept_regions", 0)),
            },
        }
        record["alarm"] = {
            "level": "candidate" if record["combined"]["has_any_detection"] else "none",
            "confirmed": False,
            "note": "Single-frame output only; temporal confirmation is intentionally not applied.",
        }
        record["status"] = {
            "frame": "ok" if registration["valid"] else "partial",
            "failures": [] if registration["valid"] else ["open_set_registration_rejected"],
        }
        return {"record": record, "overlay": overlay, "open_set_result": open_result}

    def process_path(self, image_path: str | Path, output_dir: str | Path):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        result = self.process(image, filename=image_path.name)
        stem = image_path.stem
        Image.fromarray(result["overlay"]).save(output_dir / f"{stem}__combined.jpg", quality=92)
        (output_dir / f"{stem}__combined.json").write_text(
            json.dumps(result["record"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result["record"]
