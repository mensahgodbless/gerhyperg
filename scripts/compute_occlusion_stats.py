"""
Compute occlusion / face-availability statistics on VGAF.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Label file parsing 

def parse_label_mapping(spec: Optional[str]) -> dict[str, str]:
    
    if not spec:
        return {}
    out: dict[str, str] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(
                f"Bad --label_mapping entry {piece!r}; expected 'src:dst'."
            )
        src, dst = piece.split(":", 1)
        out[src.strip()] = dst.strip()
    return out


def load_labels(
    label_file: Path,
    skip_header_rows: int,
    label_mapping: dict[str, str],
    video_extension: str,
) -> dict[str, str]:
    
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    labels: dict[str, str] = {}
    n_lines = 0
    n_skipped_blank = 0
    n_skipped_header = 0
    n_skipped_malformed = 0
    n_unknown_label = 0

    with open(label_file, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            n_lines += 1
            line = raw.strip()
            if not line or line.startswith("#"):
                n_skipped_blank += 1
                continue
            if n_skipped_header < skip_header_rows:
                n_skipped_header += 1
                logger.info("Skipping header row: %r", line)
                continue
            parts = line.split()
            if len(parts) < 2:
                logger.warning(
                    "Skipping malformed line %d in %s: %r",
                    line_no, label_file, raw.rstrip(),
                )
                n_skipped_malformed += 1
                continue

            raw_name = parts[0]
            raw_label = parts[1]

            # Append extension if the name doesn't already have one
            if video_extension and not raw_name.endswith(video_extension):
                video_name = raw_name + video_extension
            else:
                video_name = raw_name

            # Apply optional label mapping; if mapping is provided and
            # the raw label isn't in it, warn loudly.
            if label_mapping:
                if raw_label in label_mapping:
                    label = label_mapping[raw_label]
                else:
                    n_unknown_label += 1
                    label = raw_label
            else:
                label = raw_label

            if video_name in labels:
                logger.warning(
                    "Duplicate entry for %s at line %d (was %r, now %r)",
                    video_name, line_no, labels[video_name], label,
                )
            labels[video_name] = label

    logger.info(
        "Label file stats: %d total lines, %d header, %d blank/comment, "
        "%d malformed, %d unknown-label, %d valid entries",
        n_lines, n_skipped_header, n_skipped_blank,
        n_skipped_malformed, n_unknown_label, len(labels),
    )
    return labels


# Frame sampling and face counting

def sample_frame_indices(num_total_frames: int, num_samples: int) -> list[int]:
    """Uniformly sample frame indices across a video.

    Returns ``num_samples`` indices in ``[0, num_total_frames)``.
    If the video is shorter than ``num_samples``, repeats the last frame.
    """
    if num_total_frames <= 0:
        return []
    if num_total_frames <= num_samples:
        idx = list(range(num_total_frames))
        idx += [num_total_frames - 1] * (num_samples - num_total_frames)
        return idx
    return list(np.linspace(0, num_total_frames - 1, num_samples).astype(int))


def count_faces_in_video(
    video_path: Path,
    detector: MTCNN,
    num_frames: int,
) -> Optional[list[int]]:
    """Return list of face counts per sampled frame, or None if unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s", video_path)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = sample_frame_indices(total, num_frames)
    if not sample_idx:
        cap.release()
        return None

    counts: list[int] = []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            counts.append(0)
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(frame_rgb)
        counts.append(0 if boxes is None else int(boxes.shape[0]))

    cap.release()
    return counts


# ─── Statistics computation ───────────────────────────────────────────

def varies_by_more_than_50pct(counts: list[int]) -> bool:
    """Whether per-frame counts vary by more than 50% within a video."""
    arr = np.array(counts, dtype=np.float32)
    if arr.max() == 0:
        return False
    return float((arr.max() - arr.min()) / max(arr.max(), 1.0)) > 0.5


def summarise(per_video_counts: dict[str, list[int]]) -> dict[str, float]:
    """Compute the four headline statistics on a group of videos."""
    if not per_video_counts:
        return {
            "num_videos": 0,
            "mean_faces_per_frame": 0.0,
            "median_faces_per_frame": 0.0,
            "fraction_frames_with_zero_faces": 0.0,
            "fraction_videos_with_50pct_count_variation": 0.0,
        }

    all_counts = np.array(
        [c for counts in per_video_counts.values() for c in counts],
        dtype=np.int32,
    )
    n_varying = sum(
        1 for counts in per_video_counts.values()
        if varies_by_more_than_50pct(counts)
    )

    return {
        "num_videos": len(per_video_counts),
        "mean_faces_per_frame": float(all_counts.mean()),
        "median_faces_per_frame": float(np.median(all_counts)),
        "fraction_frames_with_zero_faces": float((all_counts == 0).mean()),
        "fraction_videos_with_50pct_count_variation":
            n_varying / len(per_video_counts),
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=Path, required=True,
                        help="Directory containing VGAF .mp4 videos.")
    parser.add_argument("--label_file", type=Path, required=True,
                        help="Text file: 'video_name label' per line.")
    parser.add_argument("--skip_header_rows", type=int, default=1,
                        help="Number of header rows to skip (VGAF has 1).")
    parser.add_argument("--video_extension", type=str, default=".mp4",
                        help="Extension appended to label-file names if missing.")
    parser.add_argument("--label_mapping", type=str, default="",
                        help="Mapping 'raw1:pretty1,raw2:pretty2'. "
                             "For VGAF, use '1:negative,2:neutral,3:positive'.")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames sampled per 5-second clip (default: 16).")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on number of videos processed (for testing).")
    parser.add_argument("--allow_missing_label", action="store_true",
                        help="If set, process videos even if not in label file.")
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    label_mapping = parse_label_mapping(args.label_mapping)
    if label_mapping:
        logger.info("Label mapping: %s", label_mapping)

    labels = load_labels(
        label_file=args.label_file,
        skip_header_rows=args.skip_header_rows,
        label_mapping=label_mapping,
        video_extension=args.video_extension,
    )

    # Find videos on disk
    videos_on_disk = sorted(args.video_dir.glob(f"*{args.video_extension}"))
    logger.info(
        "Found %d %s files under %s",
        len(videos_on_disk), args.video_extension, args.video_dir,
    )
    if not videos_on_disk:
        raise SystemExit("No video files found — check --video_dir / --video_extension.")

    # Match disk videos against label file
    matched: list[tuple[Path, str]] = []
    missing_label: list[str] = []
    for vp in videos_on_disk:
        if vp.name in labels:
            matched.append((vp, labels[vp.name]))
        else:
            missing_label.append(vp.name)

    labels_missing_video = sorted(
        set(labels.keys()) - {vp.name for vp, _ in matched}
    )

    logger.info("Videos with labels: %d", len(matched))
    logger.info("Videos on disk but NOT in label file: %d", len(missing_label))
    logger.info("Labels with no video on disk: %d", len(labels_missing_video))

 
    if len(matched) == 0 and not args.allow_missing_label:
        # Print examples to help debug
        ex_label_keys = list(labels.keys())[:3]
        ex_disk_names = [vp.name for vp in videos_on_disk[:3]]
        raise SystemExit(
            f"Zero matches between disk and label file.\n"
            f"  Example label-file entries: {ex_label_keys}\n"
            f"  Example disk filenames:     {ex_disk_names}\n"
            f"Check --video_extension and --skip_header_rows."
        )

    if missing_label and not args.allow_missing_label:
        logger.info(
            "Skipping %d videos with no label (use --allow_missing_label to include them).",
            len(missing_label),
        )

    to_process: list[tuple[Path, Optional[str]]] = list(matched)
    if args.allow_missing_label:
        to_process += [(args.video_dir / name, None) for name in missing_label]

    if args.limit:
        to_process = to_process[: args.limit]
    logger.info("Will process %d videos", len(to_process))

    #Initialise MTCNN
    detector = MTCNN(
        keep_all=True,
        device=args.device,
        thresholds=[0.6, 0.7, 0.7],
        min_face_size=20,
    )

    #Process videos 
    per_video_counts: dict[str, list[int]] = {}
    per_video_label: dict[str, str] = {}
    failed: list[str] = []
    for vp, label in tqdm(to_process, desc="Counting faces"):
        counts = count_faces_in_video(vp, detector, args.num_frames)
        if counts is None:
            failed.append(str(vp))
            continue
        per_video_counts[vp.name] = counts
        per_video_label[vp.name] = label if label is not None else "_unlabeled"

    # ── Stats ────────────────────────────────────────────────────
    overall = summarise(per_video_counts)

    by_class: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for vid_name, counts in per_video_counts.items():
        cls = per_video_label[vid_name]
        by_class[cls][vid_name] = counts
    per_class_stats = {cls: summarise(d) for cls, d in by_class.items()}

    all_counts = np.array(
        [c for counts in per_video_counts.values() for c in counts],
        dtype=np.int32,
    )
    if all_counts.size:
        keys, vals = np.unique(all_counts, return_counts=True)
        hist = {str(int(k)): int(v) for k, v in zip(keys, vals)}
    else:
        hist = {}

    summary = {
        "config": {
            "video_dir": str(args.video_dir),
            "label_file": str(args.label_file),
            "skip_header_rows": args.skip_header_rows,
            "video_extension": args.video_extension,
            "label_mapping": label_mapping,
            "num_frames_per_video": args.num_frames,
        },
        "matching": {
            "num_videos_on_disk": len(videos_on_disk),
            "num_labels_in_file": len(labels),
            "num_videos_with_labels": len(matched),
            "num_videos_missing_label": len(missing_label),
            "num_labels_with_no_video": len(labels_missing_video),
        },
        "processing": {
            "num_videos_attempted": len(to_process),
            "num_videos_succeeded": len(per_video_counts),
            "num_videos_failed": len(failed),
        },
        "overall": overall,
        "per_class": per_class_stats,
        "face_count_histogram": hist,
        "failed_videos": failed,
    }

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote %s", args.output_json)
    logger.info("─── Overall ───")
    logger.info("Videos processed:       %d", overall["num_videos"])
    logger.info("Mean faces / frame:     %.2f", overall["mean_faces_per_frame"])
    logger.info("Median faces / frame:   %.2f", overall["median_faces_per_frame"])
    logger.info("Frac zero-face frames:  %.3f", overall["fraction_frames_with_zero_faces"])
    logger.info(
        "Frac videos w/ >50%% face-count variation:  %.3f",
        overall["fraction_videos_with_50pct_count_variation"],
    )
    logger.info("─── Per-class ───")
    for cls, stats in per_class_stats.items():
        logger.info(
            "  %-15s  n=%d  mean=%.2f  zero-frac=%.3f  varying=%.3f",
            cls, stats["num_videos"],
            stats["mean_faces_per_frame"],
            stats["fraction_frames_with_zero_faces"],
            stats["fraction_videos_with_50pct_count_variation"],
        )


if __name__ == "__main__":
    main()