"""Per-frame translation registration on the two stable stator bands."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

DEFAULT_REGISTRATION = {
    "left_band_u": [0.08, 0.23],
    "right_band_u": [0.77, 0.92],
    "row_limit": "reliable",
    "search_radius_px": 12,
    "subpixel": True,
    "min_corr": 0.50,
    "min_side_corr": 0.40,
    "min_peak_margin": 0.01,
    "second_peak_radius": 3,
}


def _gray(image_rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(image_rgb, dtype=np.float64)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _sample(gray: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coords = np.vstack([y.ravel(), x.ravel()])
    return ndi.map_coordinates(gray, coords, order=1, mode="nearest").reshape(x.shape)


def _znorm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    std = a.std()
    return a - a.mean() if std < 1e-9 else (a - a.mean()) / std


def _subpixel_1d(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    a, b, c = values[index - 1], values[index], values[index + 1]
    denominator = a - 2 * b + c
    return 0.0 if abs(denominator) < 1e-12 else float(0.5 * (a - c) / denominator)


class TranslationRegistrar:
    def __init__(
        self,
        canonical_map: str | Path | dict[str, np.ndarray],
        reference_image: str | Path | np.ndarray,
        config: dict | None = None,
    ):
        if isinstance(canonical_map, (str, Path)):
            z = np.load(canonical_map)
            canonical = {k: z[k] for k in z.files}
        else:
            canonical = canonical_map
        self.canonical = canonical
        self.config = {**DEFAULT_REGISTRATION, **(config or {})}
        if isinstance(reference_image, (str, Path)):
            reference_image = np.asarray(Image.open(reference_image).convert("RGB"))
        self.reference_gray = _gray(reference_image)
        self._bands = [
            self._band_coords(*self.config["left_band_u"]),
            self._band_coords(*self.config["right_band_u"]),
        ]
        self._reference = [_znorm(_sample(self.reference_gray, x, y)) for x, y in self._bands]

    def _band_coords(self, u_lo: float, u_hi: float):
        u = self.canonical["u_grid"].astype(float)
        cols = np.where((u >= u_lo) & (u <= u_hi))[0]
        if not len(cols):
            raise ValueError(f"canonical map has no columns for band {u_lo}-{u_hi}")
        x = self.canonical["map_x"].astype(float)[:, cols]
        y = self.canonical["map_y"].astype(float)[:, cols]
        limit = self.config["row_limit"]
        if limit == "reliable":
            key = "n_s_reliable_v3" if "n_s_reliable_v3" in self.canonical else "n_s_reliable"
            rows = int(self.canonical[key])
        elif limit == "full":
            rows = x.shape[0]
        else:
            rows = min(int(limit), x.shape[0])
        return x[:rows], y[:rows]

    def _corr_map(self, target_gray, x, y, reference, radius):
        scores = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        for iy, dy in enumerate(range(-radius, radius + 1)):
            for ix, dx in enumerate(range(-radius, radius + 1)):
                patch = _sample(target_gray, x + dx, y + dy)
                scores[iy, ix] = float(np.mean(reference * _znorm(patch)))
        return scores

    def estimate(self, image_rgb: np.ndarray) -> dict:
        target = _gray(image_rgb)
        radius = int(self.config["search_radius_px"])
        maps = [
            self._corr_map(target, x, y, reference, radius)
            for (x, y), reference in zip(self._bands, self._reference)
        ]
        fused = 0.5 * (maps[0] + maps[1])
        iy, ix = np.unravel_index(int(np.argmax(fused)), fused.shape)
        best = float(fused[iy, ix])
        suppressed = fused.copy()
        suppress = int(self.config["second_peak_radius"])
        suppressed[max(0, iy-suppress):iy+suppress+1, max(0, ix-suppress):ix+suppress+1] = -np.inf
        second = float(np.max(suppressed))
        dx = float(ix - radius)
        dy = float(iy - radius)
        if self.config["subpixel"]:
            dx += _subpixel_1d(fused[iy], ix)
            dy += _subpixel_1d(fused[:, ix], iy)
        left_corr = float(maps[0][iy, ix])
        right_corr = float(maps[1][iy, ix])
        margin = best - second
        reasons = []
        if best < self.config["min_corr"]:
            reasons.append("low_fused_correlation")
        if margin < self.config["min_peak_margin"]:
            reasons.append("ambiguous_peak")
        if min(left_corr, right_corr) < self.config["min_side_corr"]:
            reasons.append("weak_single_side")
        if not reasons:
            status = "valid"
        elif best >= self.config["min_corr"] and min(left_corr, right_corr) >= self.config["min_side_corr"] * 0.7:
            status = "degraded"
        else:
            status = "rejected"
        return {
            "dx_sub": dx,
            "dy_sub": dy,
            "best_corr": best,
            "second_corr": second,
            "margin": margin,
            "corr_left": left_corr,
            "corr_right": right_corr,
            "status": status,
            "valid": status in {"valid", "degraded"},
            "reason": reasons,
        }
