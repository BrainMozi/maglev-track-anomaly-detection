"""Merge sharded frame JSONL outputs into a manual-review index and summary."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    manifests = sorted(args.result_dir.glob("frame_results.shard_*.jsonl"))
    if not manifests:
        raise SystemExit("no sharded manifests found")
    rows = []
    class_counts = Counter()
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            for detection in record.get("known_objects", []):
                class_counts[detection["class_name"]] += 1
            rows.append({
                "filename": record["filename"],
                "frame_status": record.get("status", {}).get("frame", "unknown"),
                "registration_status": record.get("registration", {}).get("status", "unknown"),
                "person_count": sum(d["class_name"] == "person" for d in record.get("known_objects", [])),
                "vehicle_count": sum(d["class_name"] == "vehicle" for d in record.get("known_objects", [])),
                "open_set_regions": record["combined"]["open_set_region_count"],
                "overlap_count": record["combined"]["known_objects_overlapping_open_set"],
                "has_any_detection": record["combined"]["has_any_detection"],
                "manual_truth": "",
                "manual_notes": "",
            })
    rows.sort(key=lambda row: row["filename"] or "")
    with (args.result_dir / "manual_review_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "frames": len(rows),
        "failed_frames": sum(r["frame_status"] == "failed" for r in rows),
        "partial_frames": sum(r["frame_status"] == "partial" for r in rows),
        "registration_rejected": sum(r["registration_status"] == "rejected" for r in rows),
        "frames_with_any_detection": sum(bool(r["has_any_detection"]) for r in rows),
        "frames_with_open_set_region": sum(int(r["open_set_regions"]) > 0 for r in rows),
        "known_object_boxes": dict(class_counts),
        "note": "Precision/recall require manual_truth or independent labels; this file alone is an operational summary.",
    }
    (args.result_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
