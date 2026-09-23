#!/usr/bin/env python
"""
GECV occlusion statistics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


CLASSES = ("positive", "negative", "neutral")


def sample_frames_uniform(video_path: Path, num_frames: int) -> List[np.ndarray]:
    """Sample num_frames frames uniformly across the full clip duration."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames in {video_path}")

    indices = np.linspace(0, total - 1, num=num_frames).round().astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def count_faces_per_frame(detector: MTCNN, frames: List[np.ndarray]) -> List[int]:
    """Return number of faces detected in each frame."""
    counts = []
    for frame in frames:
        boxes, _ = detector.detect(frame)
        counts.append(0 if boxes is None else len(boxes))
    return counts


def variation_above_threshold(counts: List[int], threshold: float = 0.5) -> bool:
    """(max - min) / max > threshold. True means the clip shows substantial
    face-count variation across sampled frames (occlusion / camera change / etc.)."""
    if not counts:
        return False
    cmax = max(counts)
    cmin = min(counts)
    if cmax == 0:
        return False
    return (cmax - cmin) / cmax > threshold


def aggregate(per_video: List[dict]) -> dict:
    """Overall summary statistics across a list of per-video records."""
    all_counts = [c for v in per_video for c in v["counts"]]
    if not all_counts:
        return {
            "num_videos": 0,
            "mean_faces_per_frame": 0.0,
            "median_faces_per_frame": 0.0,
            "fraction_frames_with_zero_faces": 0.0,
            "fraction_videos_with_50pct_count_variation": 0.0,
        }
    return {
        "num_videos": len(per_video),
        "mean_faces_per_frame": float(np.mean(all_counts)),
        "median_faces_per_frame": float(np.median(all_counts)),
        "fraction_frames_with_zero_faces": float(np.mean([c == 0 for c in all_counts])),
        "fraction_videos_with_50pct_count_variation": float(
            np.mean([variation_above_threshold(v["counts"]) for v in per_video])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=Path, required=True,
                        help="Root containing positive/, negative/, neutral/ subdirs.")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames sampled per video. Default 16 to match VGAF analysis.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video_extension", type=str, default=".mp4")
    parser.add_argument("--limit_per_class", type=int, default=None,
                        help="Optional cap on videos per class (for a smoke test).")
    parser.add_argument("--save_per_video", action="store_true",
                        help="Also save per-video face counts (larger JSON).")
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    detector = MTCNN(
        keep_all=True,
        device=args.device,
        thresholds=[0.6, 0.7, 0.7],  # Loose, matches VGAF script.
        min_face_size=20,
    )

    per_class_records: Dict[str, List[dict]] = {c: [] for c in CLASSES}
    num_failed = 0

    for cls in CLASSES:
        cls_dir = args.video_dir / cls
        if not cls_dir.is_dir():
            raise SystemExit(f"Missing class directory: {cls_dir}")
        videos = sorted(cls_dir.glob(f"*{args.video_extension}"))
        if args.limit_per_class:
            videos = videos[: args.limit_per_class]
        logger.info("Class %s: %d videos", cls, len(videos))

        for vp in tqdm(videos, desc=f"{cls:>9}"):
            try:
                frames = sample_frames_uniform(vp, args.num_frames)
                counts = count_faces_per_frame(detector, frames)
                per_class_records[cls].append({
                    "video": vp.name,
                    "counts": counts,
                })
            except Exception as e:
                num_failed += 1
                logger.warning("Failed on %s: %s", vp, e)

    all_records = [r for cls in CLASSES for r in per_class_records[cls]]
    result = {
        "config": {
            "video_dir": str(args.video_dir),
            "num_frames_per_video": args.num_frames,
            "classes": list(CLASSES),
            "label_source": "parent_directory",
        },
        "processing": {
            "num_videos_succeeded": len(all_records),
            "num_videos_failed": num_failed,
            "per_class_counts": {c: len(per_class_records[c]) for c in CLASSES},
        },
        "overall": aggregate(all_records),
        "per_class": {c: aggregate(per_class_records[c]) for c in CLASSES},
    }

    if args.save_per_video:
        result["per_video"] = {
            c: per_class_records[c] for c in CLASSES
        }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Wrote %s", args.output_json)
    print("\n=== OVERALL ===")
    print(json.dumps(result["overall"], indent=2))
    print("\n=== PER CLASS ===")
    print(json.dumps(result["per_class"], indent=2))


if __name__ == "__main__":
    main()