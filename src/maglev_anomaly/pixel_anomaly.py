"""Pixel-level normal evidence: colour / gradient residuals with Median/MAD and
position-wise q90/q99 calibration, reusing the effective parts of the early
`evaluate_background_anomaly.py` route but on the frozen canonical raster with a
global (cross-segment) normal bank built from the 116 confirmed normal frames.

Only provides boundary / local anomaly support; it never triggers a final alarm
independently of the WideResNet seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from .data import gray_from_rgb


def _stack(rows, mx, my, canonical_image):
    imgs = []
    for r in rows:
        im = canonical_image(r, mx, my)
        imgs.append(im.astype("float32"))
    return np.stack(imgs, axis=0)


def _channel_median_match(stack, valid, target=None):
    """Align per-frame channel medians to a target (defaults to global median).
    Returns (matched_stack, target)."""
    medians = np.array([[np.median(f[..., c][valid]) for c in range(3)] for f in stack],
                       dtype=np.float32)
    if target is None:
        target = np.median(medians, axis=0)
    return np.clip(stack + (target[None, None, None, :] - medians[:, None, None, :]), 0, 255), target


def match_single(frame, valid, target):
    """Match one frame's channel medians to a frozen target."""
    med = np.array([np.median(frame[..., c][valid]) for c in range(3)], dtype=np.float32)
    return np.clip(frame + (target[None, None, :] - med[None, None, :]), 0, 255)


def build_bank(stack, valid, cfg):
    """Build the frozen normal bank from the normal canonical stack (N,H,W,3)."""
    px = cfg["pixel"]
    stack, target = _channel_median_match(stack, valid)
    bg = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - bg[None]), axis=0)

    gray = gray_from_rgb(stack)
    gx = sobel(gray, axis=2)  # horizontal
    gy = sobel(gray, axis=1)  # vertical
    grad = np.hypot(gx, gy)
    grad_bg = np.median(grad, axis=0)
    grad_mad = np.median(np.abs(grad - grad_bg[None]), axis=0)

    color_maps = np.stack([
        gaussian_filter(np.mean(np.abs(f - bg) / (mad + px["color_mad_eps"]), axis=2),
                        px["gaussian_color_sigma"])
        for f in stack
    ])
    grad_maps = np.stack([
        gaussian_filter(np.abs(g - grad_bg) / (grad_mad + px["grad_mad_eps"]),
                        px["gaussian_grad_sigma"])
        for g in grad
    ])
    color_q90 = np.quantile(color_maps, 0.90, axis=0)
    color_q99 = np.quantile(color_maps, 0.99, axis=0)
    grad_q90 = np.quantile(grad_maps, 0.90, axis=0)
    grad_q99 = np.quantile(grad_maps, 0.99, axis=0)

    return {
        "bg": bg.astype("float32"), "mad": mad.astype("float32"),
        "grad_bg": grad_bg.astype("float32"), "grad_mad": grad_mad.astype("float32"),
        "color_q90": color_q90.astype("float32"), "color_q99": color_q99.astype("float32"),
        "grad_q90": grad_q90.astype("float32"), "grad_q99": grad_q99.astype("float32"),
        "target": target.astype("float32"),
    }


def score_frame(frame, bank, valid, cfg):
    """Score one canonical frame -> fused continuous pixel anomaly map in [0,1]."""
    px = cfg["pixel"]
    bg, mad = bank["bg"], bank["mad"]
    grad_bg, grad_mad = bank["grad_bg"], bank["grad_mad"]

    frame = match_single(frame, valid, bank["target"])

    color_z = np.mean(np.abs(frame - bg) / (mad + px["color_mad_eps"]), axis=2)
    gray = gray_from_rgb(frame)
    grad = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    grad_z = np.abs(grad - grad_bg) / (grad_mad + px["grad_mad_eps"])

    color_raw = gaussian_filter(color_z, px["gaussian_color_sigma"])
    grad_raw = gaussian_filter(grad_z, px["gaussian_grad_sigma"])

    color_excess = np.maximum(0.0, (color_raw - bank["color_q90"]) /
                              (bank["color_q99"] - bank["color_q90"] + px["excess_eps"]))
    grad_excess = np.maximum(0.0, (grad_raw - bank["grad_q90"]) /
                             (bank["grad_q99"] - bank["grad_q90"] + px["excess_eps"]))
    calibrated_color = (1.0 - np.exp(-color_excess)) * valid
    calibrated_grad = (1.0 - np.exp(-grad_excess)) * valid
    fused = gaussian_filter(px["color_weight"] * calibrated_color + px["grad_weight"] * calibrated_grad,
                            px["gaussian_fused_sigma"]) * valid
    return fused.astype("float32")
