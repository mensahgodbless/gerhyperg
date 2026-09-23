from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


GECV_CLASSES = ("positive", "neutral", "negative")


def discover_gecv_clips(
    video_root: Path,
    split: str = "all",  # GECV has no train/val — kept for API symmetry
) -> List[Dict[str, Any]]:
    
    if split not in ("all", ""):
        logger.warning("GECV has no train/val split; ignoring split=%r", split)
    video_root = Path(video_root)
    out: List[Dict[str, Any]] = []
    for cls in GECV_CLASSES:
        d = video_root / cls
        if not d.exists():
            logger.warning("GECV class dir not found: %s", d)
            continue
        for mp4 in sorted(d.glob("*.mp4")):
            out.append({
                "clip_id":    f"{cls}_{mp4.stem}",
                "video_path": str(mp4),
                "metadata": {
                    "dataset":   "gecv",
                    "class":     cls,
                    "class_idx": GECV_CLASSES.index(cls),
                },
            })
    return out



VGAF_SPLITS = ("Train", "Val")


def discover_vgaf_clips(
    video_root: Path,
    split: str = "all",
) -> List[Dict[str, Any]]:
    
    video_root = Path(video_root)

    split = split.lower()
    if split == "all":
        splits = VGAF_SPLITS
    elif split == "train":
        splits = ("Train",)
    elif split == "val":
        splits = ("Val",)
    else:
        raise ValueError(f"VGAF split must be 'train', 'val', or 'all'; got {split!r}")

    out: List[Dict[str, Any]] = []
    for s in splits:
        d = video_root / s
        if not d.exists():
            logger.warning("VGAF split dir not found: %s", d)
            continue
        for mp4 in sorted(d.glob("*.mp4")):
            out.append({
                "clip_id":    f"{s.lower()}_{mp4.stem}",
                "video_path": str(mp4),
                "metadata": {
                    "dataset": "vgaf",
                    "split":   s.lower(),
                },
            })
    return out



def discover_clips(
    dataset: str,
    video_root: Path,
    split: str = "all",
) -> List[Dict[str, Any]]:
    """Dispatch to the right per-dataset discovery function."""
    dataset = dataset.lower()
    if dataset == "gecv":
        return discover_gecv_clips(video_root, split=split)
    if dataset == "vgaf":
        return discover_vgaf_clips(video_root, split=split)
    raise ValueError(f"Unknown dataset {dataset!r}. Expected 'gecv' or 'vgaf'.")


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    gecv_root = os.environ.get("GECV_VIDEO_ROOT", "/path/to/datasets/GECV/GECV")
    vgaf_root = os.environ.get("VGAF_VIDEO_ROOT", "/path/to/datasets/VGAFDATA/VGAF")

    for name, root, splits in [
        ("gecv", gecv_root, ["all"]),
        ("vgaf", vgaf_root, ["train", "val", "all"]),
    ]:
        for sp in splits:
            try:
                clips = discover_clips(name, Path(root), split=sp)
            except (ValueError, FileNotFoundError) as e:
                print(f"[{name} {sp}] ERROR: {e}")
                continue
            print(f"[{name} {sp}] {len(clips)} clips")
            for c in clips[:3]:
                print(f"    {c['clip_id']:30s}  {c['video_path']}")
                print(f"        metadata: {c['metadata']}")
            if len(clips) > 3:
                print(f"    ... ({len(clips) - 3} more)")