"""Scene feature extraction.

Per clip:
  1. Sample T frames uniformly from the video.
  2. Run CLIP ViT-B/16 image encoder on each whole frame → 512-dim.
  3. Mean-pool across T → [512].
  4. Save to {output_dir}/<clip_id>.h5 with field 'features' of shape [512].

This is the "scene" branch of the model — captures global visual context
(lighting, setting, crowd density at the pixel level) without any person
cropping. Complementary to the per-person hypergraph.

Usage:
    python scripts/extract_scene_features.py \
        --video_root /path/to/datasets/GECV/GECV \
        --output_dir /path/to/datasets/gerhyperg_features/gecv/scene \
        --dataset gecv \
        --num_frames 15 \
        --device cuda

Requires:
    pip install open_clip_torch opencv-python h5py
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

# ── Locate project root ──────────────────────────────────────────────
def _find_project_root() -> Path:
    anchor = Path(__file__).resolve().parent
    for candidate in [anchor] + list(anchor.parents):
        if (candidate / "configs").is_dir() and (candidate / "requirements.txt").is_file():
            return candidate
    cwd = Path.cwd()
    if (cwd / "configs").is_dir():
        return cwd
    raise FileNotFoundError("Cannot locate project root.")

_PROJECT_ROOT = _find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.extract.discover import discover_clips

logger = logging.getLogger(__name__)

CLASSES_GECV: Tuple[str, ...] = ("positive", "negative", "neutral")
SCENE_DIM = 512


def sample_frames_uniform(video_path: Path, num_frames: int) -> List[np.ndarray]:
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
            if frames:
                frames.append(frames[-1])
                continue
            raise RuntimeError(f"Failed to read any frames from {video_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def enumerate_clips(dataset: str, video_root: Path, video_extension: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if dataset == "gecv":
        for cls in CLASSES_GECV:
            cls_dir = video_root / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing class dir: {cls_dir}")
            for vp in sorted(cls_dir.glob(f"*{video_extension}")):
                out.append((f"{cls}_{vp.stem}", vp))
    elif dataset == "vgaf":
        for vp in sorted(video_root.glob(f"*{video_extension}")):
            out.append((vp.stem, vp))
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["gecv", "vgaf"], required=True)
    parser.add_argument("--num_frames", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--batch_frames", type=int, default=15,
                        help="Forward this many frames at once through CLIP.")
    parser.add_argument("--video_extension", type=str, default=".mp4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only shard K of N, e.g. '0/2'. "
                             "Splits clips by (idx %% N).")
    parser.add_argument("--split", default="all", choices=["train", "val", "all"],
                            help="VGAF split. Ignored for GECV.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CLIP %s / %s", args.clip_model, args.clip_pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    model = model.to(args.device).eval()

    #clips = enumerate_clips(args.dataset, args.video_root, args.video_extension)
    clips = discover_clips(args.dataset, Path(args.video_root), split=args.split)
    if args.limit:
        clips = clips[: args.limit]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= k < n and n >= 1):
            raise ValueError(f"Bad --shard {args.shard!r}")
        clips = [c for i, c in enumerate(clips) if i % n == k]
        logger.info("Shard %d/%d: %d clips on this worker", k, n, len(clips))
    logger.info("Extracting scene features for %d clips → %s", len(clips), args.output_dir)

    num_done = num_skipped = num_failed = 0
    with torch.no_grad():
        #for clip_id, video_path in tqdm(clips):
        for clip_info in tqdm(clips):
            clip_id = clip_info["clip_id"]
            video_path = clip_info["video_path"]
            out_path = args.output_dir / f"{clip_id}.h5"
            if out_path.exists() and not args.overwrite:
                num_skipped += 1
                continue
            try:
                frames = sample_frames_uniform(video_path, args.num_frames)
                imgs = torch.stack([
                    preprocess(Image.fromarray(f)) for f in frames
                ]).to(args.device)                                          # [T, 3, H, W]

                # Forward in chunks if needed (most clips fit in one batch).
                feats = []
                for i in range(0, imgs.size(0), args.batch_frames):
                    chunk = imgs[i : i + args.batch_frames]
                    f = model.encode_image(chunk)                            # [chunk, 512]
                    feats.append(f.float())
                per_frame = torch.cat(feats, dim=0)                          # [T, 512]
                pooled = per_frame.mean(dim=0).cpu().numpy().astype(np.float32)  # [512]

                if pooled.shape[-1] != SCENE_DIM:
                    raise ValueError(
                        f"CLIP output dim {pooled.shape[-1]} != expected {SCENE_DIM}. "
                        f"Wrong CLIP variant?"
                    )

                with h5py.File(out_path, "w") as f:
                    f.create_dataset("features", data=pooled, compression="gzip", compression_opts=4)
                    f.attrs["clip_model"] = args.clip_model
                    f.attrs["clip_pretrained"] = args.clip_pretrained
                    f.attrs["num_frames"] = args.num_frames
                num_done += 1
            except Exception as e:
                logger.exception("Failed on %s: %s", clip_id, e)
                num_failed += 1

    logger.info("Done. extracted=%d skipped=%d failed=%d", num_done, num_skipped, num_failed)


if __name__ == "__main__":
    main()