import tempfile
from pathlib import Path

import numpy as np

from maglev_anomaly.known_yolo import KnownObjectDetector
import maglev_anomaly.open_set as open_set_module
from maglev_anomaly.open_set import OpenSetSingleFrameProcessor
from maglev_anomaly.pixel_anomaly import build_bank
from maglev_anomaly.registration import TranslationRegistrar
from maglev_anomaly.unified import UnifiedSingleFrameProcessor


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _Boxes:
    def __init__(self):
        self.xyxy = _Tensor([[10, 20, 60, 80], [80, 20, 120, 70]])
        self.conf = _Tensor([0.50, 0.10])
        self.cls = _Tensor([0, 1])


class _Result:
    boxes = _Boxes()


class _YoloModel:
    def predict(self, **kwargs):
        assert kwargs["imgsz"] == 960
        return [_Result()]


class _Known:
    def predict(self, image):
        return [{"class_id": 0, "class_name": "person", "confidence": 0.8, "xyxy": [1, 1, 5, 5]}]


class _Registrar:
    def estimate(self, image):
        return {"valid": True, "status": "valid", "dx_sub": 0.0, "dy_sub": 0.0}


class _Scorer:
    def score(self, canonical):
        heat = np.zeros((4, 2), dtype=np.float32)
        heat[1:3, :] = 5.0
        return heat.copy(), heat


class _Classifier:
    def predict_proba(self, features):
        return np.tile(np.array([[0.1, 0.9]], dtype=np.float32), (len(features), 1))


class _OpenSet:
    def process(self, image, **kwargs):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[2:6, 2:6] = True
        return {
            "original_mask": mask,
            "summary": {"kept_regions": 1, "original_mask_pixels": int(mask.sum())},
            "regions": [{"label": 1, "verdict": "keep", "probability": 0.9}],
        }


def test_yolo_roi_mapping_and_class_thresholds():
    detector = KnownObjectDetector(
        model=_YoloModel(), roi=(100, 200, 300, 400), imgsz=960,
        thresholds={"person": 0.02, "vehicle": 0.15}, device="cpu",
    )
    detections = detector.predict(np.zeros((500, 500, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0]["class_name"] == "person"
    assert detections[0]["xyxy"] == [110.0, 220.0, 160.0, 280.0]


def test_translation_registration_identity():
    rng = np.random.default_rng(4)
    image = (rng.random((30, 30, 3)) * 255).astype(np.uint8)
    yy, xx = np.mgrid[5:25, 5:25]
    canonical = {
        "map_x": xx.astype(np.float32),
        "map_y": yy.astype(np.float32),
        "u_grid": np.linspace(0, 1, 20, dtype=np.float32),
        "n_s_reliable_v3": np.array(20),
    }
    registrar = TranslationRegistrar(
        canonical, image,
        {"search_radius_px": 2, "second_peak_radius": 0, "min_peak_margin": 0.0},
    )
    result = registrar.estimate(image)
    assert result["valid"] is True
    assert abs(result["dx_sub"]) < 0.25
    assert abs(result["dy_sub"]) < 0.25


def test_open_set_single_frame_contract():
    yy, xx = np.mgrid[0:20, 0:10]
    normal = np.full((6, 20, 10, 3), 100, dtype=np.float32)
    valid = np.ones((20, 10), dtype=bool)
    pixel_cfg = {
        "color_mad_eps": 6.0, "grad_mad_eps": 10.0, "excess_eps": 0.5,
        "color_weight": 0.68, "grad_weight": 0.32,
        "gaussian_color_sigma": 1.2, "gaussian_grad_sigma": 1.0,
        "gaussian_fused_sigma": 1.0,
    }
    bank = build_bank(normal, valid, {"pixel": pixel_cfg})
    with tempfile.TemporaryDirectory() as tmp:
        canonical_path = Path(tmp) / "map.npz"
        np.savez(canonical_path, map_x=xx.astype(np.float32), map_y=yy.astype(np.float32))
        config = {
            "assets": {"canonical_map": str(canonical_path)},
            "canonical_shape": [20, 10],
            "frozen_params": {"seed_threshold": 3.0, "support_threshold": 0.01,
                              "unary_weight": 1.0, "smooth": 1.0, "bias": 0.0},
            "pixel": pixel_cfg,
            "graphcut": {"downsample": 2, "edge_sigma": 0.18, "hard_capacity": 1e6},
        }
        processor = OpenSetSingleFrameProcessor(
            config, heatmap_scorer=_Scorer(), pixel_bank=bank, region_classifier=_Classifier()
        )
        image = np.full((20, 10, 3), 100, dtype=np.uint8)
        image[5:15] = 220
        original_graph_cut = open_set_module.gc.graph_cut
        open_set_module.gc.graph_cut = lambda seed, anomaly, lab, valid, support, *args: (seed | support) & valid
        try:
            result = processor.process(image, dx_sub=0, dy_sub=0)
        finally:
            open_set_module.gc.graph_cut = original_graph_cut
        assert result["canonical_mask"].shape == (20, 10)
        assert result["original_mask"].shape == (20, 10)
        assert result["summary"]["candidate_regions"] >= 1


def test_unified_output_keeps_both_model_results():
    processor = UnifiedSingleFrameProcessor(_Known(), _OpenSet(), _Registrar())
    result = processor.process(np.zeros((10, 10, 3), dtype=np.uint8), filename="frame.jpg")
    record = result["record"]
    assert record["combined"]["known_object_count"] == 1
    assert record["combined"]["open_set_region_count"] == 1
    assert record["known_objects"][0]["overlaps_open_set"] is True
    assert result["overlay"].shape == (10, 10, 3)
