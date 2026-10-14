"""Extract video frames without deduplication using OpenCV.

The default stride=1 keeps every decoded frame. Filenames preserve video stem
and zero-based frame index so results can be sorted back into temporal order.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    import cv2
    if args.stride < 1:
        raise SystemExit("stride must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for video in args.videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        frame_index = 0
        written = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.stride == 0:
                path = args.out / f"{video.stem}__frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                    raise RuntimeError(f"failed to write {path}")
                written += 1
            frame_index += 1
        capture.release()
        total_written += written
        print(f"{video.name}: decoded={frame_index} written={written}", flush=True)
    print(f"total_written={total_written} out={args.out}")


if __name__ == "__main__":
    main()
