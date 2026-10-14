"""Project asset and geometry helpers for the joint visual detection pipeline.

Geometry follows the canonical map convention: sample via
`map_x + dx_sub`, `map_y + dy_sub`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _key(row, names):
    for n in names:
        for k, v in row.items():
            if n in str(k).lower() and str(v).strip():
                return str(v).strip()
    return None


def load_groups(cfg):
    """Return dict train_normal/val_normal/excluded_mixed, plus rejected list."""
    A = cfg["assets"]
    with Path(A["split_csv"]).open(newline="", encoding="utf-8-sig") as f:
        selected = list(csv.DictReader(f))
    split = {}
    for r in selected:
        fn = Path(_key(r, ["filename", "image", "name", "file", "path", "sha256"]) or "").name
        s = (_key(r, ["split", "set", "partition"]) or "").lower()
        split[fn] = "train_normal" if "train" in s else ("val_normal" if ("val" in s or "valid" in s) else "")

    with Path(A["registration_csv"]).open(newline="", encoding="utf-8-sig") as f:
        regs = list(csv.DictReader(f))
    src = Path(A["image_dir"])
    valid, rejected = [], []
    for r in regs:
        fn = Path(_key(r, ["filename", "image", "frame", "name", "file", "path"]) or "").name
        p = src / fn
        if not p.exists():
            hits = list(src.rglob(fn))
            p = hits[0] if hits else p
        dx, dy = _key(r, ["dx_sub"]), _key(r, ["dy_sub"])
        st = (_key(r, ["status", "valid", "accepted"]) or "").lower()
        try:
            dx, dy = float(dx), float(dy)
            ok = bool(fn and p.exists() and "reject" not in st and st not in ("0", "false", "invalid"))
        except Exception:
            ok = False
            dx = dy = 0.0
        item = {"filename": fn, "image": str(p), "dx_sub": dx, "dy_sub": dy,
                "registration_status": "valid" if ok else "rejected"}
        (valid if ok else rejected).append(item)

    by = {r["filename"]: r for r in valid}
    train, val = [], []
    for fn, s in split.items():
        if fn not in by:
            raise RuntimeError("selected image missing valid registration: " + fn)
        x = dict(by[fn])
        x["group"] = s
        (train if s == "train_normal" else val).append(x)
    mixed = [dict(r, group="excluded_mixed") for r in valid if r["filename"] not in split]

    exp = cfg["expected"]
    got = (len(train), len(val), len(mixed), len(rejected))
    want = (exp["train_normal"], exp["val_normal"], exp["excluded_mixed"], exp["registration_rejected"])
    if got != want:
        raise RuntimeError("asset counts %s != %s" % (got, want))
    return {"train_normal": train, "val_normal": val, "excluded_mixed": mixed}, rejected


def load_map(cfg):
    z = np.load(cfg["assets"]["canonical_map"])
    def get(ns):
        for n in ns:
            if n in z:
                return z[n]
        raise RuntimeError("canonical map lacks %s; keys=%s" % (ns, z.files))
    return get(["map_x", "canonical_map_x"]).astype("float32"), get(["map_y", "canonical_map_y"]).astype("float32")


def segment_of(filename: str) -> str:
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _load_rgb(path):
    try:
        import cv2
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError("cannot read " + str(path))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))


def canonical_from_array(im, map_x, map_y, dx_sub=0.0, dy_sub=0.0, return_valid=False):
    """Sample a canonical raster from an in-memory RGB image."""
    im = np.asarray(im, dtype=np.uint8)
    mx = map_x + float(dx_sub)
    my = map_y + float(dy_sub)
    try:
        import cv2
        out = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    except ImportError:
        import scipy.ndimage as ndi
        coords = np.vstack([my.ravel(), mx.ravel()])
        out = np.stack([ndi.map_coordinates(im[..., c].astype("float32"), coords,
                                            order=1, mode="constant", cval=0).reshape(mx.shape)
                        for c in range(3)], axis=-1)
        out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    valid = (mx >= 0) & (my >= 0) & (mx < im.shape[1] - 1) & (my < im.shape[0] - 1)
    if return_valid:
        return out, valid, im.shape
    return out


def canonical_image(row, map_x, map_y, return_valid=False):
    """Sample the canonical raster of a frame via map + (dx_sub, dy_sub)."""
    im = _load_rgb(row["image"])
    return canonical_from_array(im, map_x, map_y, row["dx_sub"], row["dy_sub"], return_valid)


def read_heat(heat_root, group, filename, calibrated=True):
    """Read WRN concat heatmap (296,40) float32 from the frozen result tree."""
    stem = Path(filename).stem
    p = Path(heat_root) / "wide_resnet50_2_l2_l3" / group / (stem + "__concat__normal_calibrated.npz")
    if not calibrated:
        p = Path(heat_root) / "wide_resnet50_2_l2_l3" / group / (stem + "__concat__raw.npz")
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p)
    a = z["map"] if "map" in z else z[list(z.files)[0]]
    if tuple(a.shape) != (296, 40):
        raise RuntimeError("expected WRN concat heatmap (296,40), got " + str(a.shape))
    return a.astype("float32")


def load_coarse_mask(cfg, filename):
    from PIL import Image
    p = Path(cfg["assets"]["coarse_masks"]) / filename
    if not p.exists():
        stem = Path(filename).stem
        hits = list(Path(cfg["assets"]["coarse_masks"]).glob(stem + "*"))
        if hits:
            p = hits[0]
    if not p.exists():
        return None
    return np.asarray(Image.open(p).convert("L")) > 0


def rgb_to_lab(rgb):
    """sRGB uint8 -> Lab (L in [0,100], a/b roughly [-128,127]) via numpy."""
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    mask = rgb > 0.04045
    lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = lin @ M.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def gray_from_rgb(rgb):
    r = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * r[..., 0] + 0.7152 * r[..., 1] + 0.0722 * r[..., 2]


def backproject_fillpoly(canonical_mask, row, map_x, map_y, original_shape):
    """Backproject a canonical bool mask to the original frame via mapped contours."""
    m = canonical_mask.astype(np.uint8)
    try:
        import cv2
        out = np.zeros(original_shape[:2], dtype=np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            pts = []
            for x, y in c[:, 0, :]:
                y = int(np.clip(y, 0, map_x.shape[0] - 1))
                x = int(np.clip(x, 0, map_x.shape[1] - 1))
                pts.append([map_x[y, x] + row["dx_sub"], map_y[y, x] + row["dy_sub"]])
            if len(pts) >= 3:
                polys.append(np.rint(pts).astype("int32"))
            else:
                for x, y in c[:, 0, :]:
                    quad = []
                    for ddy, ddx in ((0, 0), (0, 1), (1, 1), (1, 0)):
                        cy = min(int(y) + ddy, map_x.shape[0] - 1)
                        cx = min(int(x) + ddx, map_x.shape[1] - 1)
                        quad.append([map_x[cy, cx] + row["dx_sub"], map_y[cy, cx] + row["dy_sub"]])
                    polys.append(np.rint(quad).astype("int32"))
        if polys:
            cv2.fillPoly(out, polys, 255)
        return out > 0
    except ImportError:
        from PIL import Image, ImageDraw
        canvas = Image.new("L", (original_shape[1], original_shape[0]), 0)
        draw = ImageDraw.Draw(canvas)
        yy, xx = np.where(canonical_mask)
        for y, x in zip(yy.tolist(), xx.tolist()):
            pts = []
            for dy, dx in ((0, 0), (0, 1), (1, 1), (1, 0)):
                cy = min(y + dy, map_x.shape[0] - 1)
                cx = min(x + dx, map_x.shape[1] - 1)
                pts.append((int(round(map_x[cy, cx] + row["dx_sub"])),
                            int(round(map_y[cy, cx] + row["dy_sub"]))))
            draw.polygon(pts, fill=255)
        return np.asarray(canvas) > 0
