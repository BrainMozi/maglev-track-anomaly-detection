import numpy as np

from maglev_anomaly.data import segment_of
from maglev_anomaly.graphcut import _unary_costs, keep_seed_components
from maglev_anomaly.pixel_anomaly import build_bank, score_frame
from maglev_anomaly.region import label_candidates


def test_segment_of_removes_frame_suffix():
    assert segment_of("ch07_abc_000123.jpg") == "ch07_abc"


def test_keep_seed_components_drops_unseeded_component():
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[3:5, 5:7] = True
    seed = np.zeros_like(mask)
    seed[1, 1] = True
    kept = keep_seed_components(mask, seed)
    assert kept[1:3, 1:3].all()
    assert not kept[3:5, 5:7].any()


def test_unary_hard_constraints():
    anomaly = np.full((3, 3), 0.8, dtype=np.float32)
    seed = np.zeros((3, 3), dtype=bool)
    seed[1, 1] = True
    valid = np.ones((3, 3), dtype=bool)
    support = np.zeros((3, 3), dtype=bool)
    fg, bg = _unary_costs(anomaly, seed, valid, support, 1.0, 0.0, 1e6)
    assert fg[1, 1] == 0
    assert bg[1, 1] == 1e6
    assert fg[0, 0] == 1e6
    assert bg[0, 0] == 0


def test_region_labels_positive_negative_ambiguous():
    coarse = np.zeros((4, 6), dtype=bool)
    coarse[:, :3] = True
    positive = np.zeros_like(coarse); positive[:, :2] = True
    negative = np.zeros_like(coarse); negative[:, 4:] = True
    ambiguous = np.zeros_like(coarse); ambiguous[:, 2:5] = True
    y = label_candidates([(1, positive), (2, negative), (3, ambiguous)], coarse, 0.5)
    assert y.tolist() == [1, 0, -1]


def test_pixel_anomaly_output_contract():
    rng = np.random.default_rng(0)
    stack = np.clip(100 + rng.normal(0, 2, size=(6, 24, 12, 3)), 0, 255).astype(np.float32)
    valid = np.ones((24, 12), dtype=bool)
    cfg = {"pixel": {
        "color_mad_eps": 6.0,
        "grad_mad_eps": 10.0,
        "excess_eps": 0.5,
        "color_weight": 0.68,
        "grad_weight": 0.32,
        "gaussian_color_sigma": 1.2,
        "gaussian_grad_sigma": 1.0,
        "gaussian_fused_sigma": 1.0,
    }}
    bank = build_bank(stack, valid, cfg)
    score = score_frame(stack[0], bank, valid, cfg)
    assert score.shape == valid.shape
    assert score.dtype == np.float32
    assert np.isfinite(score).all()
    assert score.min() >= 0
    assert score.max() <= 1
