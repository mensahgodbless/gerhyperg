"""
Train and evaluate on GECV with 10-fold stratified CV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
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

from data.datasets.gecv_dataset import GECVDataset, gecv_collate, stratified_kfold_splits
from training.train_utils import (
    EarlyStopper, build_model, build_optimizer, build_scheduler,
    evaluate, load_config, save_run_artifacts, set_seed, train_one_epoch,
)
from training.losses import MultiBranchCELoss

logger = logging.getLogger(__name__)


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


def run_one_fold(cfg, fold_idx: int, train_ids, val_ids, output_dir: Path, seed: int) -> dict:
    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    train_ds = GECVDataset(
        video_root=cfg["data"]["video_root"],
        cache_root=cfg["data"]["cache_root"],
        clip_ids=train_ids,
        strict=True,
        use_temporal=cfg["model"].get("temporal", {}).get("enabled", False),
    )
    val_ds = GECVDataset(
        video_root=cfg["data"]["video_root"],
        cache_root=cfg["data"]["cache_root"],
        clip_ids=val_ids,
        strict=True,
        use_temporal=cfg["model"].get("temporal", {}).get("enabled", False),
    )

    logger.info("Fold %d — train=%d val=%d", fold_idx, len(train_ds), len(val_ds))

    nw = cfg["data"].get("num_workers", 4)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=gecv_collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, collate_fn=gecv_collate)

    model = build_model(cfg).to(device)

    class_weights = None
    if cfg["train"]["loss"]["use_class_weights"]:
        counts = np.bincount(train_ds.all_labels, minlength=cfg["model"]["num_classes"])
        weights = len(train_ds.all_labels) / (cfg["model"]["num_classes"] * counts.clip(min=1))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    active_branches = cfg["train"].get("active_branches", ["visual", "audio", "scene", "temporal"])

    loss_fn = MultiBranchCELoss(
        class_weights=class_weights,
        aux_weight=cfg["train"]["loss"]["aux_loss_weight"],
        active_branches=active_branches,
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if cfg["train"]["amp"] else None
    # scaler = torch.cuda.amp.GradScaler() if cfg["train"]["amp"] else None
    stopper = EarlyStopper(patience=cfg["train"]["patience"])

    rng = np.random.default_rng(seed + fold_idx)  # fold-distinct stochasticity
    history: list[dict] = []
    best_state = None

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, loss_fn=loss_fn, device=device,
            grad_clip=cfg["train"]["grad_clip"], amp=cfg["train"]["amp"],
            scaler=scaler, branch_dropout=cfg["train"].get("branch_dropout", 0.0),
            active_branches=active_branches, rng=rng, epoch=epoch,
        )
        val_metrics, _, _ = evaluate(
            model=model, loader=val_loader, device=device,
            amp=cfg["train"]["amp"], num_classes=cfg["model"]["num_classes"],
        )
        dt = time.time() - t0
        improved, should_stop = stopper.step(val_metrics["accuracy"], epoch)

        history.append({
            "epoch": epoch, "time_sec": dt, **tr,
            "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["weighted_f1"],
        })
        logger.info(
            "fold %d epoch %2d  %.1fs  train_loss=%.4f train_acc=%.4f  "
            "val_acc=%.4f f1=%.4f%s",
            fold_idx, epoch, dt, tr["train_loss"], tr["train_acc"],
            val_metrics["accuracy"], val_metrics["weighted_f1"],
            "  *best" if improved else "",
        )
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if should_stop:
            logger.info("Fold %d early stop at epoch %d (best %d).",
                        fold_idx, epoch, stopper.best_epoch)
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final_metrics, _, _ = evaluate(
        model=model, loader=val_loader, device=device,
        amp=cfg["train"]["amp"], num_classes=cfg["model"]["num_classes"],
    )

    save_run_artifacts(
        output_dir=output_dir, cfg=cfg, history=history,
        final_metrics={
            "fold": fold_idx, "best_epoch": stopper.best_epoch,
            "val_metrics": final_metrics,
        },
        model_state=best_state,
    )
    return {
        "fold": fold_idx,
        "best_epoch": stopper.best_epoch,
        "val_acc": final_metrics["accuracy"],
        "val_f1": final_metrics["weighted_f1"],
        "val_metrics": final_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="gecv")
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument("--override", nargs="+", default=[])
    parser.add_argument("--output_root", type=Path, default=_PROJECT_ROOT / "results")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="subset of fold indices (default: all)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. -1 = CPU. Overrides project.device "
                             "from the config.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    cli_overrides = _parse_overrides(args.override)
    if args.gpu is not None:
        cli_overrides.setdefault("project", {})["device"] = (
            "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
        )
    cfg = load_config(args.dataset_config, args.ablation, cli_overrides)


    seed = cfg["train"]["seeds"][0]

    # Build fold list from the dataset's clip enumeration.
    enum_ds = GECVDataset(
        video_root=cfg["data"]["video_root"],
        cache_root=cfg["data"]["cache_root"],
        strict=True,
    )
    folds = stratified_kfold_splits(
        enum_ds.all_clip_ids, enum_ds.all_labels,
        k=cfg["data"]["num_folds"], seed=cfg["data"]["fold_seed"],
    )
    selected = args.folds if args.folds is not None else list(range(len(folds)))

    name = args.output_name or (
        "gecv_" + ("_".join(args.ablation) if args.ablation else "main")
        + "_" + time.strftime("%Y%m%d_%H%M%S")
    )
    run_root = args.output_root / name
    run_root.mkdir(parents=True, exist_ok=True)
    logger.info("Run root: %s", run_root)
    logger.info("Folds to run: %s of %d", selected, len(folds))

    per_fold = []
    for fi in selected:
        train_ids, val_ids = folds[fi]
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fi, len(folds))
        logger.info("=" * 60)
        fold_dir = run_root / f"fold_{fi}"
        per_fold.append(run_one_fold(cfg, fi, train_ids, val_ids, fold_dir, seed))

    accs = [r["val_acc"] for r in per_fold]
    f1s = [r["val_f1"] for r in per_fold]
    summary = {
        "folds_run": selected,
        "num_folds_run": len(per_fold),
        "val_acc_mean": float(np.mean(accs)),
        "val_acc_std": float(np.std(accs)),
        "val_f1_mean": float(np.mean(f1s)),
        "val_f1_std": float(np.std(f1s)),
        "per_fold_acc": accs,
        "per_fold_f1": f1s,
    }
    with open(run_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("DONE — val acc %.4f ± %.4f  over %d folds",
                summary["val_acc_mean"], summary["val_acc_std"], len(per_fold))
    logger.info("Artifacts in %s", run_root)


if __name__ == "__main__":
    main()
