#!/usr/bin/env python
"""
Extract TimeSformer temporal features per clip.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, TimesformerModel

# Project root on sys.path so we can import data/extract/discover.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.extract.discover import discover_clips  # noqa: E402

LOGGER = logging.getLogger("extract_temporal")



def sample_frames(video_path: Path, n_frames: int) -> np.ndarray:
    """
    Uniformly sample n_frames RGB frames from a video.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"no frames in {video_path}")

        target_indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
        out = []
        last_good = None
        for idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                last_good = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append(last_good)
            elif last_good is not None:
                out.append(last_good)
            else:
                raise RuntimeError(f"could not decode any frame from {video_path}")
        return np.stack(out, axis=0)
    finally:
        cap.release()


# feature extraction 

@torch.inference_mode()
def extract_features(
    video_path: Path,
    model: TimesformerModel,
    processor,
    device: torch.device,
    n_frames: int,
) -> np.ndarray:
    """Returns a (768,) float32 numpy array (the [CLS] token from the last layer)."""
    frames = sample_frames(video_path, n_frames=n_frames)         # (n, H, W, 3) uint8 RGB
    inputs = processor(list(frames), return_tensors="pt")          # pixel_values: (1, n, 3, 224, 224)
    pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
    outputs = model(pixel_values=pixel_values)                     # last_hidden_state: (1, T, 768)
    cls = outputs.last_hidden_state[:, 0, :].squeeze(0)            # (768,)
    return cls.float().cpu().numpy()


def write_h5(
    out_path: Path,
    features: np.ndarray,
    model_str: str,
    n_frames: int,
    metadata: dict,
) -> None:
    """Atomic write: write to .tmp, fsync, then rename."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with h5py.File(tmp_path, "w") as f:
        # Use key "features" to match visual/audio/scene convention.
        f.create_dataset("features", data=features.astype(np.float32))
        f.attrs["model"] = model_str
        f.attrs["n_frames"] = int(n_frames)
        f.attrs["dim"] = int(features.shape[0])
        f.attrs["dtype"] = "float32"
        for k, v in metadata.items():
            # Numpy / Python scalars and short strings only.
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = str(v)
    tmp_path.replace(out_path)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["vgaf", "gecv"], required=True)
    p.add_argument("--video_root", type=Path, required=True,
                   help="Dataset root passed to discover_clips, "
                        "e.g. /path/to/datasets/VGAFDATA/VGAF")
    p.add_argument("--cache_root", type=Path, required=True,
                   help="Feature cache root for THIS dataset, "
                        "e.g. /path/to/datasets/gerhyperg_features/vgaf "
                        "(output goes to <cache_root>/temporal/)")
    p.add_argument("--split", default="all",
                   help="VGAF: 'all' | 'train' | 'val'. GECV: 'all'.")
    p.add_argument("--model", default="facebook/timesformer-base-finetuned-k400")
    p.add_argument("--n_frames", type=int, default=8,
                   help="Frames per clip into TimeSformer (K400 default = 8).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, process only the first N clips (smoke test).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    LOGGER.info("device: %s", device)

    LOGGER.info("loading model: %s", args.model)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = TimesformerModel.from_pretrained(args.model).to(device).eval()
    hidden_size = model.config.hidden_size
    LOGGER.info("hidden_size=%d  n_frames=%d", hidden_size, args.n_frames)

    clips = discover_clips(args.dataset, args.video_root, split=args.split)
    if args.limit > 0:
        clips = clips[: args.limit]
    LOGGER.info("discovered %d clips for %s (split=%s)",
                len(clips), args.dataset, args.split)

    out_dir = args.cache_root / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("output dir: %s", out_dir)

    fail_log_path = out_dir / "_failures.jsonl"
    fail_log = open(fail_log_path, "a", buffering=1)

    n_done, n_skip, n_fail = 0, 0, 0
    t0 = time.time()

    try:
        for c in tqdm(clips, desc=f"temporal/{args.dataset}"):
            clip_id = c["clip_id"]
            video_path = Path(c["video_path"])
            metadata = c.get("metadata", {})
            out_path = out_dir / f"{clip_id}.h5"

            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            try:
                feat = extract_features(
                    video_path=video_path,
                    model=model,
                    processor=processor,
                    device=device,
                    n_frames=args.n_frames,
                )
                if feat.shape != (hidden_size,):
                    raise RuntimeError(
                        f"unexpected feature shape {feat.shape}, "
                        f"expected ({hidden_size},)"
                    )
                write_h5(
                    out_path=out_path,
                    features=feat,
                    model_str=args.model,
                    n_frames=args.n_frames,
                    metadata=metadata,
                )
                n_done += 1
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("FAILED %s: %s", clip_id, e)
                fail_log.write(json.dumps({
                    "clip_id": clip_id,
                    "path": str(video_path),
                    "error": str(e),
                }) + "\n")
                n_fail += 1
    finally:
        fail_log.close()

    elapsed = time.time() - t0
    LOGGER.info(
        "done.  done=%d  skip=%d  fail=%d  total=%d  elapsed=%.1fs",
        n_done, n_skip, n_fail, len(clips), elapsed,
    )
    if n_fail > 0:
        LOGGER.info("failure log: %s", fail_log_path)


if __name__ == "__main__":
    main()