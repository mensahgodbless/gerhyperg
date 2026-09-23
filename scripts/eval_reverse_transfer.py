"""
Reverse cross-dataset evaluation: GECV-trained → VGAF.
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
from data.datasets.gecv_dataset import gecv_collate
from training.train_utils import build_model, evaluate
from training.metrics import compute_all_metrics

logger = logging.getLogger(__name__)



def _load_vgaf(split: str, use_temporal: bool):
    """Target dataset is always VGAF in the reverse direction."""
    return VGAFDataset(
        video_root="/path/to/datasets/VGAFDATA/VGAF",
        cache_root="/path/to/datasets/gerhyperg_features/vgaf",
        split=split,
        strict=True,
        use_temporal=use_temporal,
    )


def _discover_folds(folds_root: Path, checkpoint_name: str) -> list[Path]:
    """
    Return sorted list of fold subdirectories under folds_root.
    """
    if not folds_root.is_dir():
        raise FileNotFoundError(f"folds_root does not exist: {folds_root}")
    candidates = []
    for child in folds_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "config.yaml").is_file() and (child / checkpoint_name).is_file():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(
            f"No fold subdirectories found under {folds_root} "
            f"(expected config.yaml + {checkpoint_name} in each fold-dir)")
    # Sort by name. Works for fold_0..fold_9 lexicographically;
    # also fine for 0..9 numerically with zero-pad, but tolerate unpadded too.
    def sort_key(p: Path):
        # Try to extract trailing digits for natural ordering
        import re
        m = re.search(r"(\d+)", p.name)
        return (int(m.group(1)) if m else 1e9, p.name)
    candidates.sort(key=sort_key)
    return candidates


def _face_count_buckets(num_persons_per_clip: np.ndarray, edges=(2, 5, 10)):
    """Bucket clips by mean detected persons. Same as forward script."""
    means = num_persons_per_clip
    buckets: dict[str, list[int]] = {}
    last = 0
    for e in edges:
        name = f"{last:d}-{e-1:d}" if last > 0 else f"≤{e-1:d}"
        buckets[name] = np.where((means >= last) & (means < e))[0].tolist()
        last = e
    buckets[f"≥{last:d}"] = np.where(means >= last)[0].tolist()
    return buckets


def _collect_face_counts(ds) -> np.ndarray:
    """Mean num_persons per clip across T=15 frames."""
    out = []
    for i in range(len(ds)):
        sample = ds[i]
        mask = sample["person_mask"]
        out.append(float(mask.float().sum(dim=1).mean().item()))
    return np.asarray(out)



# Per-fold evaluation
def _eval_one_fold(
    fold_dir: Path,
    checkpoint_name: str,
    target_split: str,
    gpu_override: int | None,
    do_face_count: bool,
    cached_face_counts: np.ndarray | None,
    cached_dataset_cache=[None],  # mutable default as lightweight cache
):
    """Evaluate the checkpoint in fold_dir on VGAF val. Returns a report dict."""
    cfg_path = fold_dir / "config.yaml"
    ckpt_path = fold_dir / checkpoint_name

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if gpu_override is not None:
        cfg["project"]["device"] = "cpu" if gpu_override < 0 else f"cuda:{gpu_override}"
    device = torch.device(cfg["project"]["device"])

    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    use_temporal = cfg["model"].get("temporal", {}).get("enabled", False)

    # Cache the dataset
    cache_key = (target_split, use_temporal)
    if cached_dataset_cache[0] is None or cached_dataset_cache[0][0] != cache_key:
        ds = _load_vgaf(target_split, use_temporal=use_temporal)
        cached_dataset_cache[0] = (cache_key, ds)
    else:
        ds = cached_dataset_cache[0][1]

    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=gecv_collate,
    )

    metrics, preds, targets = evaluate(
        model=model, loader=loader, device=device,
        amp=cfg["train"]["amp"],
        num_classes=cfg["model"]["num_classes"],
        drop_flags={"drop_visual": False, "drop_audio": False,
                    "drop_scene": False, "drop_temporal": False},
    )

    report = {
        "fold_dir": str(fold_dir),
        "checkpoint": str(ckpt_path),
        "accuracy": float(metrics["accuracy"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "per_class_accuracy": [float(x) for x in metrics["per_class_accuracy"]],
        "per_class_names": list(metrics["per_class_names"]),
        "confusion_matrix": (metrics["confusion_matrix"].tolist()
                             if hasattr(metrics["confusion_matrix"], "tolist")
                             else metrics["confusion_matrix"]),
        "num_samples": int(metrics["num_samples"]),
    }

    if do_face_count:
        # face counts are dataset-intrinsic and identical across folds,
        # so compute once and pass in
        if cached_face_counts is None:
            cached_face_counts = _collect_face_counts(ds)
        buckets = _face_count_buckets(cached_face_counts)
        stratified = {}
        for name, idxs in buckets.items():
            if not idxs:
                stratified[name] = {"n": 0}
                continue
            bm = compute_all_metrics(
                preds[idxs], targets[idxs],
                num_classes=cfg["model"]["num_classes"],
                class_names=["positive", "neutral", "negative"],
            )
            stratified[name] = {
                "n": len(idxs),
                "accuracy": float(bm["accuracy"]),
                "weighted_f1": float(bm["weighted_f1"]),
            }
        report["face_count_stratified"] = stratified

    # Tidy up GPU memory before next fold
    del model
    torch.cuda.empty_cache()

    return report, cached_face_counts



# Aggregation
def _aggregate(per_fold_reports: list[dict], class_names: list[str]) -> dict:
    """Compute mean ± std across folds. Population std (ddof=0), matching the
    forward direction's convention."""
    accs = np.array([r["accuracy"] for r in per_fold_reports])
    f1s  = np.array([r["weighted_f1"] for r in per_fold_reports])
    per_class = np.array([r["per_class_accuracy"] for r in per_fold_reports])  # (n_folds, n_classes)

    agg = {
        "num_folds": len(per_fold_reports),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),         # population, ddof=0
        "weighted_f1_mean": float(f1s.mean()),
        "weighted_f1_std": float(f1s.std()),
        "fold_accuracies": [float(a) for a in accs],
        "per_class_accuracy_mean": [float(x) for x in per_class.mean(axis=0)],
        "per_class_accuracy_std":  [float(x) for x in per_class.std(axis=0)],
        "per_class_names": class_names,
    }

    # Aggregate face-count buckets if present in every fold
    if all("face_count_stratified" in r for r in per_fold_reports):
        bucket_names = list(per_fold_reports[0]["face_count_stratified"].keys())
        face_agg = {}
        for bn in bucket_names:
            ns  = [r["face_count_stratified"][bn]["n"] for r in per_fold_reports]
            # n is dataset-intrinsic and constant across folds — sanity-check
            if len(set(ns)) != 1:
                logger.warning("Bucket %r has inconsistent n across folds: %s", bn, ns)
            n_ref = ns[0]
            if n_ref == 0:
                face_agg[bn] = {"n": 0}
                continue
            bucket_accs = np.array([r["face_count_stratified"][bn]["accuracy"]
                                    for r in per_fold_reports])
            face_agg[bn] = {
                "n": int(n_ref),
                "accuracy_mean": float(bucket_accs.mean()),
                "accuracy_std":  float(bucket_accs.std()),
            }
        agg["face_count_stratified"] = face_agg

    return agg




def main():
    parser = argparse.ArgumentParser(
        description="Reverse cross-dataset eval: GECV-trained → VGAF val.")
    parser.add_argument("--folds_root", type=Path, required=True,
                        help="Parent dir containing fold subdirectories, "
                             "each with config.yaml + best_model.pt")
    parser.add_argument("--checkpoint_name", default="best_model.pt",
                        help="Checkpoint filename inside each fold-dir.")
    parser.add_argument("--target_split", default="val",
                        choices=["val", "train"],
                        help="VGAF split to evaluate on. 'val' = 766 clips "
                             "(cleanest comparison vs in-distribution column).")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Where to write report.json. Defaults to "
                             "<folds_root>/reverse_eval_vgaf/")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. -1 = CPU. Overrides config device.")
    parser.add_argument("--no_face_count", action="store_true",
                        help="Skip face-count stratified eval.")
    parser.add_argument("--config_name", default=None,
                        help="Optional label for the config; defaults to "
                             "folds_root.name (e.g. 'gecv_hypergraph_3br').")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    fold_dirs = _discover_folds(args.folds_root, args.checkpoint_name)
    logger.info("Discovered %d folds under %s", len(fold_dirs), args.folds_root)
    for fd in fold_dirs:
        logger.info("  %s", fd.name)

    if len(fold_dirs) != 10:
        logger.warning("Expected 10 folds (GECV 10-fold CV) but found %d. "
                       "Proceeding anyway.", len(fold_dirs))

    per_fold_reports = []
    cached_face_counts = None
    for i, fd in enumerate(fold_dirs):
        logger.info("[%d/%d] Evaluating fold: %s",
                    i + 1, len(fold_dirs), fd.name)
        rep, cached_face_counts = _eval_one_fold(
            fold_dir=fd,
            checkpoint_name=args.checkpoint_name,
            target_split=args.target_split,
            gpu_override=args.gpu,
            do_face_count=not args.no_face_count,
            cached_face_counts=cached_face_counts,
        )
        per_fold_reports.append(rep)
        logger.info("  acc=%.4f  f1=%.4f", rep["accuracy"], rep["weighted_f1"])

    class_names = per_fold_reports[0]["per_class_names"]
    aggregate = _aggregate(per_fold_reports, class_names)

    logger.info("=" * 60)
    logger.info("AGGREGATE  (%s → VGAF %s)",
                args.config_name or args.folds_root.name,
                args.target_split)
    logger.info("  accuracy: %.4f ± %.4f  (n=%d folds)",
                aggregate["accuracy_mean"],
                aggregate["accuracy_std"],
                aggregate["num_folds"])
    logger.info("  weighted_f1: %.4f ± %.4f",
                aggregate["weighted_f1_mean"],
                aggregate["weighted_f1_std"])
    logger.info("  per-class accuracy (mean ± std):")
    for cls, m, s in zip(class_names,
                         aggregate["per_class_accuracy_mean"],
                         aggregate["per_class_accuracy_std"]):
        logger.info("    %-10s  %.4f ± %.4f", cls, m, s)
    if "face_count_stratified" in aggregate:
        logger.info("  face-count buckets (mean acc ± std):")
        for name, m in aggregate["face_count_stratified"].items():
            if m["n"] == 0:
                logger.info("    %-8s  n=%3d  (empty)", name, m["n"])
            else:
                logger.info("    %-8s  n=%3d  %.4f ± %.4f",
                            name, m["n"],
                            m["accuracy_mean"], m["accuracy_std"])

    out_dir = args.output_dir or (args.folds_root / "reverse_eval_vgaf")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_name": args.config_name or args.folds_root.name,
        "source_dataset": "gecv",
        "target_dataset": "vgaf",
        "target_split": args.target_split,
        "folds_root": str(args.folds_root),
        "checkpoint_name": args.checkpoint_name,
        "per_fold": per_fold_reports,
        "aggregate": aggregate,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Report: %s", out_dir / "report.json")


if __name__ == "__main__":
    main()