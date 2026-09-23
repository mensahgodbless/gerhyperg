"""
Standalone evaluation and cross-dataset transfer from saved checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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
from data.datasets.gecv_dataset import (
    GECVDataset, gecv_collate, stratified_kfold_splits,
)
from training.train_utils import build_model, evaluate, load_config


def _parse_overrides(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"Bad override: {it!r}; use key.path=value")
        key, val = it.split("=", 1)
        try:
            cast: object = int(val)
        except ValueError:
            try:
                cast = float(val)
            except ValueError:
                cast = {"true": True, "false": False}.get(val.lower(), val)
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = cast
    return out


def _build_loader(cfg, eval_dataset, clip_ids=None):
    use_temporal = cfg["model"].get("temporal", {}).get("enabled", False)
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    if eval_dataset == "vgaf":
        ds = VGAFDataset(
            video_root=cfg["data"]["video_root"],
            cache_root=cfg["data"]["cache_root"],
            split="val", strict=True, use_temporal=use_temporal,
        )
    else:
        ds = GECVDataset(
            video_root=cfg["data"]["video_root"],
            cache_root=cfg["data"]["cache_root"],
            clip_ids=clip_ids, strict=True, use_temporal=use_temporal,
        )
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=nw, collate_fn=gecv_collate)
    return ds, loader


def _discover_checkpoints(run_dir: Path):
    """VGAF runs -> seed_*/best_model.pt ; GECV runs -> fold_*/best_model.pt."""
    seeds = sorted(glob.glob(str(run_dir / "seed_*" / "best_model.pt")))
    folds = sorted(glob.glob(str(run_dir / "fold_*" / "best_model.pt")))
    return seeds or folds


def _f1_of(metrics):
    """Prefer macro-F1 (the metric the paper reports); fall back to weighted.

    NOTE: the project's current training/metrics.py computes weighted_f1 only.
    Add a macro_f1 to compute_all_metrics (see README) for this to report macro-F1.
    Returns (value, name) so the output is explicit about which F1 it is.
    """
    if "macro_f1" in metrics:
        return metrics["macro_f1"], "macro_f1"
    return metrics.get("weighted_f1"), "weighted_f1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="single checkpoint (best_model.pt)")
    ap.add_argument("--run_dir", help="run dir with seed_*/ or fold_*/ checkpoints")
    ap.add_argument("--eval_dataset", required=True, choices=["vgaf", "gecv"],
                    help="dataset to evaluate ON (the transfer target)")
    ap.add_argument("--ablation", action="append", default=[],
                    help="must match how the checkpoint was trained")
    ap.add_argument("--override", nargs="+", default=[])
    ap.add_argument("--transfer", action="store_true",
                    help="GECV target: evaluate on ALL clips (cross-dataset) "
                         "instead of per-fold held-out val")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index; -1 = CPU")
    ap.add_argument("--dump_predictions", default=None,
                    help="optional CSV of per-clip (eval,config,unit,clip_id,y_true,y_pred)")
    ap.add_argument("--out_json", default=None,
                    help="optional path to write the aggregated metrics as JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    overrides = _parse_overrides(args.override)
    if args.gpu is not None:
        overrides.setdefault("project", {})["device"] = (
            "cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    cfg = load_config(args.eval_dataset, args.ablation, overrides)
    device = torch.device(cfg["project"]["device"])
    eval_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    num_classes = cfg["model"]["num_classes"]

    if args.ckpt:
        ckpts = [Path(args.ckpt)]
    elif args.run_dir:
        ckpts = [Path(p) for p in _discover_checkpoints(Path(args.run_dir))]
        if not ckpts:
            sys.exit(f"No best_model.pt under {args.run_dir}/seed_*/ or /fold_*/")
    else:
        sys.exit("Provide --ckpt or --run_dir")

    # GECV in-distribution evaluates each fold checkpoint on its own held-out fold;
    # everything else (VGAF val, or transfer onto all GECV clips) uses one loader.
    gecv_per_fold = (args.eval_dataset == "gecv" and not args.transfer)
    folds = None
    shared_ds = shared_loader = None
    if gecv_per_fold:
        enum = GECVDataset(video_root=cfg["data"]["video_root"],
                           cache_root=cfg["data"]["cache_root"], strict=True)
        folds = stratified_kfold_splits(
            enum.all_clip_ids, enum.all_labels,
            k=cfg["data"]["num_folds"], seed=cfg["data"]["fold_seed"])
    else:
        clip_ids = None  # vgaf: full val (766); gecv transfer: all clips (408)
        shared_ds, shared_loader = _build_loader(cfg, args.eval_dataset, clip_ids)

    config_tag = "+".join(args.ablation) if args.ablation else "main"
    eval_tag = ("transfer_" if args.transfer else "") + args.eval_dataset
    accs, f1s, pcas, confs, dump_rows = [], [], [], [], []
    f1_name, class_names = "f1", None

    for i, ckpt in enumerate(ckpts):
        if gecv_per_fold:
            fold_idx = int(ckpt.parent.name.split("_")[1])
            _, val_ids = folds[fold_idx]
            ds, loader = _build_loader(cfg, "gecv", clip_ids=val_ids)
            unit = ckpt.parent.name            # fold_<i>
        else:
            ds, loader = shared_ds, shared_loader
            unit = ckpt.parent.name            # seed_<s> or fold_<i>

        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        metrics, preds, targets = evaluate(
            model, loader, device, amp=eval_amp, num_classes=num_classes)

        acc = metrics["accuracy"]
        f1, f1_name = _f1_of(metrics)
        accs.append(acc); f1s.append(f1)
        pcas.append(metrics.get("per_class_accuracy"))
        cm = metrics.get("confusion_matrix")
        confs.append(np.asarray(cm) if cm is not None else None)
        class_names = class_names or metrics.get("per_class_names")
        logger.info("[%d/%d] %s  acc=%.4f  %s=%.4f",
                    i + 1, len(ckpts), ckpt, acc, f1_name, f1)

        if args.dump_predictions:
            ids = getattr(ds, "all_clip_ids", None)
            for j, (yt, yp) in enumerate(zip(targets.tolist(), preds.tolist())):
                cid = ids[j] if ids is not None and j < len(ids) else j
                dump_rows.append([eval_tag, config_tag, unit, cid, yt, yp])

    accs, f1s = np.array(accs), np.array(f1s)
    logger.info("=" * 60)
    logger.info("%s [%s]  over %d checkpoint(s)", eval_tag, config_tag, len(accs))
    logger.info("  accuracy    %.2f ± %.2f", accs.mean() * 100, accs.std() * 100)
    logger.info("  %-11s %.2f ± %.2f", f1_name, f1s.mean() * 100, f1s.std() * 100)

    pc_mean = names = csum = rownorm = None
    # Per-class accuracy, mean across checkpoints -> Table 1 per-class columns.
    if all(p is not None for p in pcas):
        pc_mean = np.nanmean(np.array(pcas, dtype=float), axis=0) * 100
        names = class_names or [f"class_{k}" for k in range(num_classes)]
        logger.info("  per-class accuracy (mean):")
        for nm, v in zip(names, pc_mean):
            logger.info("      %-9s %.2f", nm, v)

    # Summed confusion across checkpoints -> reproduces the Fig. 2 matrices.
    if confs and all(c is not None and c.size for c in confs):
        csum = np.sum(confs, axis=0).astype(int)
        rownorm = csum / np.clip(csum.sum(axis=1, keepdims=True), 1, None) * 100
        logger.info("  confusion (summed; rows=true, cols=pred)  |  recall %%:")
        for cnt_row, rec_row in zip(csum, rownorm):
            counts = " ".join(f"{int(x):5d}" for x in cnt_row)
            recs = " ".join(f"{x:5.1f}" for x in rec_row)
            logger.info("      %s   |  %s", counts, recs)

    if args.out_json:
        out = {
            "eval": eval_tag,
            "config": config_tag,
            "n_checkpoints": int(len(accs)),
            "accuracy_mean": float(accs.mean() * 100),
            "accuracy_std": float(accs.std() * 100),
            "f1_name": f1_name,
            "f1_mean": float(f1s.mean() * 100),
            "f1_std": float(f1s.std() * 100),
            "per_class_accuracy_mean": (
                {nm: round(float(v), 2) for nm, v in zip(names, pc_mean)}
                if pc_mean is not None else None),
            "confusion_sum": (csum.tolist() if csum is not None else None),
            "recall_pct": ([[round(float(x), 2) for x in r] for r in rownorm]
                           if rownorm is not None else None),
            "checkpoints": [str(c) for c in ckpts],
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("wrote %s", args.out_json)

    if args.dump_predictions and dump_rows:
        with open(args.dump_predictions, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eval", "config", "unit", "clip_id", "y_true", "y_pred"])
            w.writerows(dump_rows)
        logger.info("wrote %s (%d rows)", args.dump_predictions, len(dump_rows))


if __name__ == "__main__":
    main()
