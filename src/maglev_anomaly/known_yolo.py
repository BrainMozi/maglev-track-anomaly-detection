"""Frozen known-object detector for the fixed-view ch07 safety area.

YOLO is deliberately independent from the open-set track anomaly model. This
module maps the training class named ``car`` back to the business label
``vehicle`` and returns boxes in full-frame coordinates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROI = (640, 270, 1920, 1080)
DEFAULT_THRESHOLDS = {"person": 0.02, "vehicle": 0.15}


class KnownObjectDetector:
    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        model: Any | None = None,
        roi=DEFAULT_ROI,
        imgsz: int = 960,
        thresholds: dict[str, float] | None = None,
        nms_iou: float = 0.70,
        device: str | int = 0,
    ):
        self.roi = tuple(int(x) for x in roi)
        self.imgsz = int(imgsz)
        self.thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        self.nms_iou = float(nms_iou)
        self.device = device
        if model is None:
            if weights is None:
                raise ValueError("weights or a model instance is required")
            from ultralytics import YOLO
            model = YOLO(str(weights))
        self.model = model

    @staticmethod
    def _business_name(class_id: int) -> str | None:
        if class_id == 0:
            return "person"
        if class_id == 1:
            return "vehicle"
        return None

    def predict(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape HxWx3")
        left, top, right, bottom = self.roi
        h, w = image_rgb.shape[:2]
        if not (0 <= left < right <= w and 0 <= top < bottom <= h):
            raise ValueError(f"ROI {self.roi} is outside image shape {(h, w)}")
        crop_rgb = image_rgb[top:bottom, left:right]
        # Ultralytics interprets numpy arrays as OpenCV-style BGR.
        crop_bgr = crop_rgb[..., ::-1].copy()
        min_conf = min(self.thresholds.values())
        result = self.model.predict(
            source=crop_bgr,
            imgsz=self.imgsz,
            conf=min_conf,
            iou=self.nms_iou,
            max_det=300,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, confidence, class_id in zip(xyxy, confidences, classes):
            name = self._business_name(int(class_id))
            if name is None or float(confidence) < self.thresholds[name]:
                continue
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "class_id": int(class_id),
                "class_name": name,
                "confidence": float(confidence),
                "xyxy": [x1 + left, y1 + top, x2 + left, y2 + top],
                "roi_xyxy": [x1, y1, x2, y2],
            })
        return detections
