"""Feature seed + pixel refinement: full pipeline (local smoke and HPC full run).

Frozen chain:
  WRN heatmap -> high-threshold seeds -> pixel anomaly (colour/gradient Median/MAD
  + q90/q99) -> Graph Cut (seed hard constraint, pixel unary, colour pairwise)
  -> seed-connected components -> region classifier removes false responses.

The patch probability never enters the Graph Cut unary; it only produces seeds.
Coarse golden masks only judge regions; we never optimise coarse-mask IoU/coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from maglev_anomaly.data import (load_groups, load_map, canonical_image, read_heat,
                                 load_coarse_mask, backproject_fillpoly, rgb_to_lab,
                                 gray_from_rgb, segment_of)
from maglev_anomaly.pixel_anomaly import build_bank, score_frame
from maglev_anomaly import graphcut as gc
from maglev_anomaly import region as rg


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_search_frames(groups):
    return groups["val_normal"] + groups["excluded_mixed"]


def group_of(groups, filename):
    for g, rows in groups.items():
        for r in rows:
            if r["filename"] == filename:
                return g
    return None


def build_cache(cfg, groups, mx, my, out, smoke=False):
    """Compute and cache: normal bank + per-frame pixel anomaly + heatmap + rgb.

    Returns dict with cached per-frame info kept on disk; heavy arrays returned
    for the search/final stages via loaders.
    """
    cache_dir = Path(out) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bank_path = cache_dir / "normal_bank.npz"

    train = groups["train_normal"][:20] if smoke else groups["train_normal"]
    if bank_path.exists() and not smoke:
        bank = dict(np.load(bank_path))
        for k in bank:
            bank[k] = bank[k].astype("float32")
        print("bank loaded", bank_path, flush=True)
    else:
        stack = []
        valid_ref = None
        for r in train:
            im, valid, _ = canonical_image(r, mx, my, True)
            stack.append(im.astype("float32"))
            if valid_ref is None:
                valid_ref = valid
        stack = np.stack(stack, axis=0)
        valid_ref = valid_ref if valid_ref is not None else np.ones(mx.shape, bool)
        bank = build_bank(stack, valid_ref, cfg)
        np.savez_compressed(bank_path, **{k: v.astype("float32") for k, v in bank.items()})
        print("bank built ->", bank_path, flush=True)
        del stack

    # per-frame pixel anomaly
    frames = iter_search_frames(groups)
    if smoke:
        frames = [frames[0]]
    frame_cache = {}
    for r in frames:
        g = r["group"]
        stem = Path(r["filename"]).stem
        ap = cache_dir / g / (stem + "__frame.npz")
        ap.parent.mkdir(parents=True, exist_ok=True)
        if ap.exists():
            continue
        im, valid, original_shape = canonical_image(r, mx, my, True)
        anomaly = score_frame(im.astype("float32"), bank, valid, cfg)
        heat = read_heat(cfg["heatmap_root"], g, r["filename"], calibrated=True)
        heat_full = gc.resize_map(heat, anomaly.shape)
        lab = rgb_to_lab(im)
        grad = np.hypot(np.gradient(gray_from_rgb(im), axis=1), np.gradient(gray_from_rgb(im), axis=0))
        np.savez_compressed(ap,
                            anomaly=anomaly.astype(np.float16),
                            heat_full=heat_full.astype(np.float16),
                            lab=lab.astype(np.float16),
                            valid=valid,
                            grad=grad.astype(np.float16),
                            original_shape=np.array(original_shape, dtype=np.int32))
        frame_cache[(g, r["filename"])] = str(ap)
    return bank, cache_dir


def load_frame_cache(cache_dir, group, filename):
    stem = Path(filename).stem
    ap = cache_dir / group / (stem + "__frame.npz")
    z = np.load(ap)
    return {
        "anomaly": z["anomaly"].astype("float32"),
        "heat_full": z["heat_full"].astype("float32"),
        "lab": z["lab"].astype("float32"),
        "valid": z["valid"],
        "grad": z["grad"].astype("float32"),
        "original_shape": tuple(int(x) for x in z["original_shape"]),
    }


def downsample_frame(fc, ds):
    if ds <= 1:
        return (fc["anomaly"], fc["heat_full"], fc["lab"], fc["valid"], fc["valid"])
    h, w = fc["anomaly"].shape
    shape = (max(1, h // ds), max(1, w // ds))
    anomaly = gc.resize_map(fc["anomaly"], shape)
    heat = gc.resize_map(fc["heat_full"], shape)
    lab = np.stack([gc.resize_map(fc["lab"][..., c], shape) for c in range(3)], axis=-1)
    valid = gc.resize_map(fc["valid"].astype("float32"), shape, order=0) > 0.5
    return anomaly, heat, lab, valid, valid


def run_cut(fc, params, cfg):
    ds = cfg["graphcut"]["downsample"]
    anomaly, heat, lab, valid, _ = downsample_frame(fc, ds)
    seed_work = heat >= params["seed_threshold"]
    support = anomaly >= params["support_threshold"]
    mask = gc.graph_cut(seed_work, anomaly, lab, valid, support,
                        params["unary_weight"], params["smooth"], params["bias"],
                        cfg["graphcut"]["edge_sigma"], cfg["graphcut"]["hard_capacity"])
    mask = gc.keep_seed_components(mask, seed_work)
    # upscale to canonical full resolution
    mask_full = gc.resize_map(mask.astype("float32"), fc["anomaly"].shape, order=0) > 0.5
    seed_full = fc["heat_full"] >= params["seed_threshold"]
    return mask_full, seed_full


def candidate_regions(mask_full):
    labels, n = ndi_label(mask_full)
    cands = []
    for i in range(1, n + 1):
        m = labels == i
        if m.sum() >= 1:
            cands.append((i, m))
    return cands, labels


def ndi_label(mask):
    from scipy import ndimage as ndi
    return ndi.label(mask)


def evaluate_candidates(cands, coarse, positive_cover=0.5):
    """Region-level labels and stats for a frame's candidate list."""
    return rg.label_candidates(cands, coarse, positive_cover=positive_cover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--params", default=None, help="json file with chosen params")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    out = Path(a.out or cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        groups, rejected = load_groups(cfg)
        mx, my = load_map(cfg)
    except (FileNotFoundError, RuntimeError) as e:
        if a.smoke:
            print("SMOKE_FAILED_ASSET:", e)
            return 2
        raise
    if tuple(mx.shape) != tuple(cfg["canonical_shape"]):
        raise RuntimeError("canonical shape must be %s, got %s" % (cfg["canonical_shape"], mx.shape))

    bank, cache_dir = build_cache(cfg, groups, mx, my, out, a.smoke)
    frames = iter_search_frames(groups)
    if a.limit:
        frames = frames[:a.limit]
    if a.smoke:
        frames = [frames[0]]

    # ---- parameter search (reads cache only) ----
    search = cfg["search"]
    if not a.no_search and not a.params:
        ds = cfg["graphcut"]["downsample"]
        candidates = []
        for st in search["seed_threshold"]:
            for sup in search["support_threshold"]:
                for uw in search["unary_weight"]:
                    for sm in search["smooth"]:
                        for bi in search["bias"]:
                            candidates.append({"seed_threshold": st, "support_threshold": sup,
                                               "unary_weight": uw, "smooth": sm, "bias": bi})
        scored = []
        for p in candidates:
            hit, nfp, ufp = 0, 0, 0
            total_abn, total_nrm = 0, 0
            for r in frames:
                fc = load_frame_cache(cache_dir, r["group"], r["filename"])
                mask_full, seed_full = run_cut(fc, p, cfg)
                coarse = load_coarse_mask(cfg, r["filename"])
                cands, _labels = candidate_regions(mask_full)
                y = evaluate_candidates(cands, coarse, cfg["region_classifier"]["positive_cover"])
                pos = int((y == 1).sum())
                neg_unlabeled = int((y == 0).sum()) if coarse is not None else 0
                if r["group"] == "val_normal":
                    nfp += len(cands)
                    total_nrm += 1
                else:
                    total_abn += 1
                    if coarse is not None and pos >= 1:
                        hit += 1
                    ufp += neg_unlabeled
            hit_rate = hit / max(1, total_abn)
            scored.append((hit_rate, -nfp, -ufp, p))
        # lexicographic: hit rate first, then fewer normal fp, then fewer unlabeled fp
        chosen = max(scored, key=lambda x: x[:3])[3]
        json.dump({"chosen": chosen, "candidate_count": len(candidates),
                   "search_frames": len(frames),
                   "selection": "coarse-region hit rate -> normal false candidates -> unlabeled false candidates; no coverage/IoU"},
                  open(out / "parameters.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        with (out / "parameter_search.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["hit_rate", "normal_false_candidates", "unlabeled_false_candidates",
                      "seed_threshold", "support_threshold", "unary_weight", "smooth", "bias"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for hit_rate, neg_nfp, neg_ufp, p in sorted(scored, key=lambda x: x[:3], reverse=True):
                w.writerow({"hit_rate": hit_rate,
                            "normal_false_candidates": -neg_nfp,
                            "unlabeled_false_candidates": -neg_ufp,
                            **p})
    elif a.params:
        chosen = json.load(open(a.params, encoding="utf-8"))
    else:
        chosen = None

    params = chosen or {"seed_threshold": 4.0, "support_threshold": 0.4,
                        "unary_weight": 1.0, "smooth": 1.0, "bias": 0.0}

    # ---- region classifier ----
    all_feats, all_y, all_meta = [], [], []
    for r in frames:
        fc = load_frame_cache(cache_dir, r["group"], r["filename"])
        mask_full, seed_full = run_cut(fc, params, cfg)
        coarse = load_coarse_mask(cfg, r["filename"])
        cands, labels = candidate_regions(mask_full)
        if not cands:
            continue
        feats = rg.extract_features(labels, max(i for i, _ in cands), fc["heat_full"],
                                    fc["anomaly"], fc["grad"])
        y = evaluate_candidates(cands, coarse, cfg["region_classifier"]["positive_cover"])
        for k, (lbl, m) in enumerate(cands):
            if y[k] == -1:
                continue
            all_feats.append(feats[k])
            all_y.append(y[k])
            all_meta.append({"filename": r["filename"], "group": r["group"], "label": int(lbl)})

    clf = None
    if all_feats and len(set(all_y)) >= 2:
        X = np.stack(all_feats).astype("float32")
        Y = np.array(all_y, dtype=int)
        clf, proba = rg.fit_predict(X, Y, cfg["region_classifier"]["model"])
        from joblib import dump
        dump(clf, out / "region_classifier.joblib")
        json.dump({"model": cfg["region_classifier"]["model"],
                   "feature_names": rg.FEATURE_NAMES,
                   "positive_cover": cfg["region_classifier"]["positive_cover"],
                   "n_samples": len(Y), "n_pos": int((Y == 1).sum()), "n_neg": int((Y == 0).sum()),
                   "note": "Training-set probabilities are not an independent validation metric. Use scripts/grouped_oof_audit.py."},
                  open(out / "classifier_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- final output ----
    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for r in frames:
        fc = load_frame_cache(cache_dir, r["group"], r["filename"])
        mask_full, seed_full = run_cut(fc, params, cfg)
        coarse = load_coarse_mask(cfg, r["filename"])
        cands, labels = candidate_regions(mask_full)
        feats = rg.extract_features(labels, max(i for i, _ in cands) if cands else 0,
                                    fc["heat_full"], fc["anomaly"], fc["grad"])
        y = evaluate_candidates(cands, coarse, cfg["region_classifier"]["positive_cover"])

        # classifier decision per candidate
        keep = np.zeros_like(mask_full)
        rec_rows = []
        kept_regions = 0
        for k, (lbl, m) in enumerate(cands):
            verdict = "keep"
            probability = None
            if clf is not None:
                probability = float(clf.predict_proba(feats[k:k + 1])[0, 1])
                verdict = "keep" if probability >= 0.5 else "drop"
            if verdict == "keep":
                keep |= m
                kept_regions += 1
            rec_rows.append({"label": int(lbl), "y": int(y[k]), "verdict": verdict,
                             "probability": probability, "area": int(m.sum()),
                             "features": {nm: float(feats[k][i]) for i, nm in enumerate(rg.FEATURE_NAMES)}})

        g = r["group"]
        stem = Path(r["filename"]).stem
        d = final_dir / g
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / (stem + "__canonical_mask.npz"), map=keep.astype("uint8"))
        np.savez_compressed(d / (stem + "__seed.npz"), map=seed_full.astype("uint8"))
        np.savez_compressed(d / (stem + "__pixel_anomaly.npz"), map=fc["anomaly"].astype("float16"))

        # overlay on canonical + original
        from PIL import Image
        im = canonical_image(r, mx, my)
        orig = np.asarray(Image.open(r["image"]).convert("RGB"))
        orig_mask = backproject_fillpoly(keep, r, mx, my, fc["original_shape"])
        _save_overlay(im, keep, d / (stem + "__canonical_overlay.png"))
        _save_overlay(orig, orig_mask, d / (stem + "__original_overlay.png"))
        Image.fromarray((orig_mask * 255).astype("uint8")).save(d / (stem + "__original_mask.png"))
        Image.fromarray((keep * 255).astype("uint8")).save(d / (stem + "__canonical_mask.png"))

        index_rows.append({"filename": r["filename"], "group": g, "segment": segment_of(r["filename"]),
                           "n_candidates": len(cands),
                           "n_kept": kept_regions,
                           "has_kept_region": int(kept_regions > 0),
                           "region_records": json.dumps(rec_rows)})
        with (d / (stem + "__regions.json")).open("w", encoding="utf-8") as f:
            json.dump(rec_rows, f, ensure_ascii=False, indent=2)

    with (out / "frame_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "group", "segment", "n_candidates", "n_kept", "has_kept_region", "region_records"])
        w.writeheader()
        w.writerows(index_rows)

    # SHA-256 manifest of final outputs
    manifest = []
    for p in sorted(final_dir.rglob("*")):
        if p.is_file():
            manifest.append({"relative_path": str(p.relative_to(out)), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    with (out / "results_sha256.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "sha256", "bytes"])
        w.writeheader()
        w.writerows(manifest)

    meta = {"canonical_shape": list(mx.shape), "groups": {k: len(v) for k, v in groups.items()},
            "frames_processed": len(frames), "params": params,
            "classifier": bool(clf is not None),
            "runtime_seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(out / "run_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def _save_overlay(img, mask, path):
    from PIL import Image
    out = np.asarray(img).copy()
    m = mask > 0
    out[m] = (0.55 * out[m] + 0.45 * np.array([255, 30, 20])).astype(np.uint8)
    Image.fromarray(out.astype("uint8")).save(path)


if __name__ == "__main__":
    main()
