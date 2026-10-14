"""Unified frozen inference for one image or a directory of frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from maglev_anomaly.known_yolo import KnownObjectDetector
from maglev_anomaly.open_set import OpenSetSingleFrameProcessor
from maglev_anomaly.registration import TranslationRegistrar
from maglev_anomaly.unified import UnifiedSingleFrameProcessor
from maglev_anomaly.wrn_memory import WideResNetFeatureExtractor, WRNNormalMemoryScorer


def collect_images(path: Path, pattern: str):
    if path.is_file():
        return [path]
    return sorted(path.glob(pattern))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--save-overlays", choices=["all", "alerts", "none"], default="alerts")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.input, args.glob)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("require 0 <= shard-index < shard-count")
    images = [path for index, path in enumerate(images) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise SystemExit("no input images found for this shard")

    yolo = KnownObjectDetector(
        config["yolo"]["weights"],
        roi=config["yolo"]["roi"],
        imgsz=config["yolo"]["imgsz"],
        thresholds=config["yolo"]["thresholds"],
        nms_iou=config["yolo"]["nms_iou"],
        device=config["device"],
    )
    registrar = TranslationRegistrar(
        config["open_set"]["assets"]["canonical_map"],
        config["registration"]["reference_image"],
        config=config["registration"].get("parameters"),
    )
    extractor = WideResNetFeatureExtractor(device=config["device"])
    scorer = WRNNormalMemoryScorer(
        extractor, config["open_set"]["artifacts"]["wrn_normal_memory"], device=config["device"]
    )
    open_set = OpenSetSingleFrameProcessor(
        config["open_set"],
        heatmap_scorer=scorer,
        pixel_bank=config["open_set"]["artifacts"]["pixel_normal_bank"],
        region_classifier=config["open_set"]["artifacts"]["region_classifier"],
    )
    processor = UnifiedSingleFrameProcessor(yolo, open_set, registrar)

    manifest = args.out / f"frame_results.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.jsonl"
    failures_path = args.out / f"failures.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.jsonl"
    try:
        with manifest.open("w", encoding="utf-8") as output, failures_path.open("w", encoding="utf-8") as failures:
            for index, image_path in enumerate(images, start=1):
                try:
                    image = np.asarray(Image.open(image_path).convert("RGB"))
                    result = processor.process(image, filename=image_path.name)
                    record = result["record"]
                    save = args.save_overlays == "all" or (
                        args.save_overlays == "alerts" and record["combined"]["has_any_detection"]
                    )
                    if save:
                        Image.fromarray(result["overlay"]).save(
                            args.out / f"{image_path.stem}__combined.jpg", quality=92
                        )
                except Exception as error:
                    failure = {
                        "filename": image_path.name,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                    failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    record = {
                        "schema_version": "ufp/0.1",
                        "filename": image_path.name,
                        "status": {"frame": "failed", "failures": [failure]},
                        "known_objects": [],
                        "open_set": {"summary": {"kept_regions": 0}, "regions": []},
                        "combined": {"known_object_count": 0, "open_set_region_count": 0,
                                     "known_objects_overlapping_open_set": 0, "has_any_detection": False},
                        "registration": {"status": "not_run", "valid": False},
                    }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                (args.out / f"{image_path.stem}__combined.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[{index}/{len(images)}] {image_path.name} status={record['status']['frame']} alert={record['combined']['has_any_detection']}", flush=True)
    finally:
        extractor.close()

    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
