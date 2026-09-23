"""
Cross-dataset evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for c in [here] + list(here.parents):
        if (c / "configs").is_dir() and (c / "requirements.txt").is_file():
            return c
    raise FileNotFoundError("project root not found")


_PROJECT_ROOT = _find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.datasets.vgaf_dataset import VGAFDataset
from data.datasets.gecv_dataset import GECVDataset, gecv_collate
from training.train_utils import build_model, evaluate
from training.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def _load_dataset(dataset: str, split: str | None,
                  cache_root_override: str | None = None,
                  use_temporal=False):
    if dataset == "vgaf":
        return VGAFDataset(
            video_root=cache_root_override or "/path/to/datasets/VGAF",
            cache_root="/path/to/datasets/features/vgaf",
            split=split or "val",
            strict=True,
            use_temporal=use_temporal,
        )
    if dataset == "gecv":
        return GECVDataset(
            video_root="/path/to/datasets/GECV",
            cache_root="/path/to/datasets/features/gecv",
            strict=True,
            use_temporal=use_temporal
        )
    raise ValueError(f"Unknown dataset {dataset!r}")


def _face_count_buckets(num_persons_per_clip: np.ndarray, edges=(2, 5, 10)):
    """Bucket clips by mean detected persons. Returns dict {bucket_name: indices}."""
    means = num_persons_per_clip
    buckets: dict[str, list[int]] = {}
    last = 0
    for e in edges:
        name = f"{last:d}-{e-1:d}" if last > 0 else f"≤{e-1:d}"
        buckets[name] = np.where((means >= last) & (means < e))[0].tolist()
        last = e
    buckets[f"≥{last:d}"] = np.where(means >= last)[0].tolist()
    return buckets


def _collect_face_counts(ds, batch_size=32) -> np.ndarray:
    """Mean num_persons per clip across T=15 frames."""
    out = []
    for i in range(len(ds)):
        sample = ds[i]
        # GECV trims to active slots; person_mask captures the trimmed reality.
        mask = sample["person_mask"]
        out.append(float(mask.float().sum(dim=1).mean().item()))
    return np.asarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config_dir", type=Path, required=True,
                        help="dir containing the resolved config.yaml for the checkpoint")
    parser.add_argument("--target_dataset", choices=["vgaf", "gecv"], required=True)
    parser.add_argument("--target_split", default=None,
                        help="for VGAF: 'train' or 'val' (default 'val')")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--drop_visual", action="store_true")
    parser.add_argument("--drop_audio", action="store_true")
    parser.add_argument("--drop_scene", action="store_true")
    parser.add_argument("--drop_temporal", action="store_true")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. -1 = CPU. Overrides the config's device.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    # Load the resolved config used to train this checkpoint — so we know
    # what backbone / pose toggle to instantiate.
    cfg_path = args.config_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml at {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)


    if args.gpu is not None:
        cfg["project"]["device"] = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    
    device = torch.device(cfg["project"]["device"])
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)
    logger.info("Target: %s (split=%s)", args.target_dataset, args.target_split)

    use_temporal = cfg["model"].get("temporal", {}).get("enabled", False)
    ds = _load_dataset(args.target_dataset, args.target_split, use_temporal=use_temporal)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"].get("num_workers", 4),
                        collate_fn=gecv_collate)

    drop_flags = {
        "drop_visual": args.drop_visual,
        "drop_audio": args.drop_audio,
        "drop_scene": args.drop_scene,
        "drop_temporal": args.drop_temporal,
    }
    metrics, preds, targets = evaluate(
        model=model, loader=loader, device=device,
        amp=cfg["train"]["amp"],
        num_classes=cfg["model"]["num_classes"],
        drop_flags=drop_flags,
    )
    logger.info("Overall — acc=%.4f f1=%.4f  (drop_flags=%s)",
                metrics["accuracy"], metrics["weighted_f1"], drop_flags)

    # Face-count stratified eval.
    logger.info("Collecting per-clip face counts …")
    face_counts = _collect_face_counts(ds)
    buckets = _face_count_buckets(face_counts)

    stratified = {}
    for name, idxs in buckets.items():
        if not idxs:
            stratified[name] = {"n": 0}
            continue
        bp = preds[idxs]
        bt = targets[idxs]
        bm = compute_all_metrics(bp, bt, num_classes=cfg["model"]["num_classes"],
                                 class_names=["positive", "neutral", "negative"])
        stratified[name] = {
            "n": len(idxs),
            "accuracy": bm["accuracy"],
            "weighted_f1": bm["weighted_f1"],
        }

    logger.info("Face-count stratified:")
    for name, m in stratified.items():
        if m["n"] == 0:
            logger.info("  %-8s  n=%3d  (empty bucket)", name, m["n"])
        else:
            logger.info("  %-8s  n=%3d  acc=%.4f  f1=%.4f",
                        name, m["n"], m["accuracy"], m["weighted_f1"])

    out_dir = args.output_dir or args.config_dir / f"cross_eval_{args.target_dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_config": str(cfg_path),
        "checkpoint": str(args.checkpoint),
        "target_dataset": args.target_dataset,
        "target_split": args.target_split,
        "drop_flags": drop_flags,
        "overall": {
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class_accuracy": metrics["per_class_accuracy"],
            "per_class_names": metrics["per_class_names"],
            "confusion_matrix": (metrics["confusion_matrix"].tolist()
                                 if hasattr(metrics["confusion_matrix"], "tolist")
                                 else metrics["confusion_matrix"]),
            "num_samples": metrics["num_samples"],
        },
        "face_count_stratified": stratified,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report: %s", out_dir / "report.json")


if __name__ == "__main__":
    main()
