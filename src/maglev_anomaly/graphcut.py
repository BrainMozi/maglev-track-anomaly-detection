"""Graph Cut with WideResNet seed as hard foreground constraint, pixel anomaly as
unary, and true image colour as pairwise. Only components connected to a seed
survive. This deliberately does NOT feed the patch probability into the unary,
which was the failure mode of the earlier pixel-fusion route (person-region
over-expansion).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def resize_map(a, shape, order=1):
    """Resize a 2D float map to `shape` (H,W)."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape == shape:
        return a
    try:
        import cv2
        return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR if order else cv2.INTER_NEAREST)
    except ImportError:
        return ndi.zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=order)


def _unary_costs(anomaly, seed, valid, support, w_unary, bias, hard):
    eps = 1e-6
    s = np.clip(anomaly, eps, 1 - eps)
    logit = np.log(s / (1 - s))
    data_fg = np.clip(w_unary * (-logit + bias), 0.0, hard)
    data_bg = np.clip(w_unary * (logit - bias), 0.0, hard)

    # hard constraints
    forced_bg = (~valid) | (~support)
    forced_bg = forced_bg & (~seed)
    data_fg[seed] = 0.0
    data_bg[seed] = hard
    data_fg[forced_bg] = hard
    data_bg[forced_bg] = 0.0
    return data_fg, data_bg


def _pairwise(lab, valid, smooth, sigma):
    labf = np.asarray(lab, dtype=np.float32)
    h, w = labf.shape[:2]
    dx = np.linalg.norm(labf[:, :-1] - labf[:, 1:], axis=2)
    dy = np.linalg.norm(labf[:-1] - labf[1:], axis=2)
    wx = np.zeros((h, w), np.float32)
    wy = np.zeros((h, w), np.float32)
    wx[:, :-1] = smooth * np.exp(-(dx * dx) / (2 * sigma * sigma))
    wy[:-1, :] = smooth * np.exp(-(dy * dy) / (2 * sigma * sigma))
    # zero edges crossing invalid pixels
    wx[:, :-1] *= (valid[:, :-1] & valid[:, 1:])
    wy[:-1, :] *= (valid[:-1, :] & valid[1:, :])
    return wx, wy


def graph_cut(seed, anomaly, lab, valid, support, w_unary, smooth, bias, sigma, hard):
    """Return bool foreground mask (same shape as inputs).

    seed: bool hard-foreground from WideResNet high threshold.
    anomaly: continuous pixel anomaly in [0,1] (unary source).
    lab: HxWx3 true image colour (pairwise).
    valid: bool valid raster pixels.
    support: bool pixels above the pixel low threshold (else forced background).
    """
    import maxflow
    h, w = anomaly.shape
    data_fg, data_bg = _unary_costs(anomaly, seed, valid, support=support,
                                    w_unary=w_unary, bias=bias, hard=hard)
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((h, w))
    g.add_grid_tedges(nodes, data_bg, data_fg)  # source=foreground
    wx, wy = _pairwise(lab, valid, smooth, sigma)
    g.add_grid_edges(nodes, weights=wx, structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8), symmetric=True)
    g.add_grid_edges(nodes, weights=wy, structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.int8), symmetric=True)
    g.maxflow()
    seg = g.get_grid_segments(nodes)  # True=sink(background), False=source(foreground)
    return (~seg) & valid


def keep_seed_components(mask, seed):
    """Keep only connected foreground components that contain at least one seed pixel."""
    lab, n = ndi.label(mask)
    keep = np.zeros_like(mask)
    if n == 0:
        return keep
    has_seed = np.zeros(n + 1, dtype=bool)
    for lbl in np.unique(lab[seed]):
        if lbl != 0:
            has_seed[lbl] = True
    return has_seed[lab]
