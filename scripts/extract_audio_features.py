"""
Audio feature extraction.
"""

from __future__ import annotations

import sys
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import scipy.io.wavfile as wav_io
import torch
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

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
TARGET_SR = 16000
AUDIO_DIM = 1024  # HuBERT-Large hidden size



def load_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract mono 16kHz audio from a video. Returns [T_samples] float32 in [-1, 1]."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-ac", "1",                # mono
            "-ar", str(TARGET_SR),     # 16kHz
            "-acodec", "pcm_s16le",    # explicit PCM, scipy reads this
            "-vn",                     # no video
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr}")
        sr, samples = wav_io.read(tmp.name)

    if sr != TARGET_SR:
        raise RuntimeError(f"Expected {TARGET_SR}Hz from ffmpeg, got {sr}")

    if samples.ndim > 1:
        samples = samples.mean(axis=-1)

    # Normalise PCM integer → float32 in [-1, 1].
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype == np.uint8:
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        samples = samples.astype(np.float32)

    wav = torch.from_numpy(samples)

    max_len = 60 * TARGET_SR
    if wav.numel() > max_len:
        wav = wav[:max_len]
    return wav



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
        videos = []
        for e in (video_extension, ".avi", ".mkv", ".webm"):
            videos.extend(video_root.rglob(f"*{e}"))
        for vp in sorted(set(videos)):
            out.append((vp.stem, vp))
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return out


def load_hubert(model_id: str, device: str):
    """Load HuBERT, preferring the local cache to avoid repeated HF round-trips."""
    try:
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id, local_files_only=True)
        m = HubertModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
        logger.info("HuBERT loaded from local cache.")
    except (OSError, EnvironmentError, ValueError):
        logger.info("Local cache miss for %s — downloading.", model_id)
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        m = HubertModel.from_pretrained(model_id).to(device).eval()
    return fe, m


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["gecv", "vgaf"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hubert_model", type=str, default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--video_extension", type=str, default=".mp4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="all", choices=["train", "val", "all"],
                            help="VGAF split. Ignored for GECV.")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only shard K of N, e.g. '0/2'. "
                             "Splits clips by (idx %% N).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HuBERT: %s", args.hubert_model)
    feature_extractor, model = load_hubert(args.hubert_model, args.device)

    # clips = enumerate_clips(args.dataset, args.video_root, args.video_extension)
    clips = discover_clips(args.dataset, Path(args.video_root), split=args.split)
    if args.limit:
        clips = clips[: args.limit]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= k < n and n >= 1):
            raise ValueError(f"Bad --shard {args.shard!r}")
        clips = [c for i, c in enumerate(clips) if i % n == k]
        logger.info("Shard %d/%d: %d clips on this worker", k, n, len(clips))
    logger.info("Extracting audio features for %d clips → %s", len(clips), args.output_dir)

    num_done = num_skipped = num_failed = num_silent = 0
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
                wav = load_audio_from_video(video_path)
                if wav.numel() == 0 or wav.abs().max().item() < 1e-6:
                    pooled = np.zeros(AUDIO_DIM, dtype=np.float32)
                    num_silent += 1
                else:
                    inputs = feature_extractor(
                        wav.numpy(),
                        sampling_rate=TARGET_SR,
                        return_tensors="pt",
                    )
                    input_values = inputs["input_values"].to(args.device)
                    out = model(input_values).last_hidden_state         # [1, T_h, 1024]
                    pooled = out.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

                if pooled.shape[-1] != AUDIO_DIM:
                    raise ValueError(
                        f"HuBERT output dim {pooled.shape[-1]} != expected {AUDIO_DIM}. "
                        f"Wrong model variant?"
                    )

                with h5py.File(out_path, "w") as f:
                    f.create_dataset("features", data=pooled, compression="gzip", compression_opts=4)
                    f.attrs["hubert_model"] = args.hubert_model
                    f.attrs["sample_rate"] = TARGET_SR
                num_done += 1
            except Exception as e:
                logger.exception("Failed on %s: %s", clip_id, e)
                num_failed += 1

    logger.info(
        "Done. extracted=%d skipped=%d failed=%d silent=%d",
        num_done, num_skipped, num_failed, num_silent,
    )


if __name__ == "__main__":
    main()