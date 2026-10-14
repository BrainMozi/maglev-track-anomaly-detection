"""Candidate-region feature extraction and a lightweight tree classifier that only
learns anomaly-evidence combinations, never object appearance (person/car)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FEATURE_NAMES = ["heat_peak", "heat_mean", "pix_peak", "pix_mean",
                 "area", "compactness", "boundary_grad"]


def extract_features(labels, n_labels, heatmap, anomaly, gray_grad):
    """Per-region feature matrix (n_labels x 7)."""
    feats = np.zeros((n_labels, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(1, n_labels + 1):
        m = labels == i
        area = float(m.sum())
        heat = heatmap[m]
        pix = anomaly[m]
        # perimeter via morphological boundary (count of edge pixels)
        edge = m & ~ndi.binary_erosion(m)
        perimeter = float(edge.sum())
        compactness = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-6)
        boundary_grad = float(gray_grad[edge].mean()) if edge.sum() else 0.0
        feats[i - 1] = [
            float(heat.max()), float(heat.mean()),
            float(pix.max()), float(pix.mean()),
            area, compactness, boundary_grad,
        ]
    return feats


def label_candidates(candidates, coarse, positive_cover=0.5):
    """Return per-candidate label: 1 positive, 0 negative, -1 ambiguous(ignore).

    candidates: list of (label_id, bool mask at canonical resolution).
    coarse: bool coarse golden mask (canonical), or None (normal frame).
    """
    y = []
    for _, mask in candidates:
        if coarse is None:
            y.append(0)  # normal-frame candidate is always negative
            continue
        area = float(mask.sum())
        inter = float((mask & coarse).sum())
        cover = inter / (area + 1e-6)
        if cover >= positive_cover:
            y.append(1)
        elif cover <= 1e-6:
            y.append(0)
        else:
            y.append(-1)  # ambiguous inside-coarse region: ignore
    return np.array(y, dtype=int)


def fit_predict(feats, y, kind="gradient_boosting", seed=0):
    """Fit a lightweight tree classifier; returns (model, proba)."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    m = feats.shape[0]
    if m < 4:
        return None, None
    y = np.asarray(y, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    sw = np.ones(m, dtype=np.float32)
    if n_pos > 0 and n_neg > 0:
        sw[y == 1] = 1.0 / n_pos
        sw[y == 0] = 1.0 / n_neg
    if kind == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=2,
                                         learning_rate=0.1, random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                                     random_state=seed, class_weight="balanced")
        sw = None
    clf.fit(feats, y, sample_weight=sw)
    proba = clf.predict_proba(feats)[:, 1]
    return clf, proba
