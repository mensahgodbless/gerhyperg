"""
Train and evaluate on VGAF Protocol A.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ── Project root ──
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
from data.datasets.gecv_dataset import gecv_collate  # same collate works for both
from training.train_utils import (
    EarlyStopper, build_model, build_optimizer, build_scheduler,
    evaluate, load_config, save_run_artifacts, set_seed, train_one_epoch,
)
from training.losses import MultiBranchCELoss
from training.splits import stratified_train_dev_split

logger = logging.getLogger(__name__)


def _parse_overrides(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"Bad override: {it!r}; use key.path=value")
        key, val = it.split("=", 1)
        # naive type cast
        try:
            cast: object = int(val)
        except ValueError:
            try:
                cast = float(val)
            except ValueError:
                cast = {"true": True, "false": False}.get(val.lower(), val)
        # nest into dict
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = cast
    return out


def run_one_seed(cfg, seed: int, output_dir: Path) -> dict:
    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    # ---- data ----
    train_ds_full = VGAFDataset(
        video_root=cfg["data"]["video_root"],
        cache_root=cfg["data"]["cache_root"],
        split="train",
        strict=True,
        use_temporal=cfg["model"].get("temporal", {}).get("enabled", False),
    )
    test_ds = VGAFDataset(
        video_root=cfg["data"]["video_root"],
        cache_root=cfg["data"]["cache_root"],
        split="val",
        strict=True,
        use_temporal=cfg["model"].get("temporal", {}).get("enabled", False),
    )

    inner_train_ids, dev_ids = stratified_train_dev_split(
        clip_ids=train_ds_full.all_clip_ids,
        labels=train_ds_full.all_labels,
        dev_ratio=cfg["data"]["dev_split_ratio"],
        seed=cfg["data"]["dev_split_seed"],
    )

    id_to_idx = {c: i for i, c in enumerate(train_ds_full.all_clip_ids)}
    inner_train = Subset(train_ds_full, [id_to_idx[c] for c in inner_train_ids])
    dev = Subset(train_ds_full, [id_to_idx[c] for c in dev_ids])

    logger.info("Splits — inner_train=%d  dev=%d  test=%d",
                len(inner_train), len(dev), len(test_ds))

    nw = cfg["data"].get("num_workers", 4)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(inner_train, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=gecv_collate, drop_last=False)
    dev_loader = DataLoader(dev, batch_size=bs, shuffle=False,
                            num_workers=nw, collate_fn=gecv_collate)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, collate_fn=gecv_collate)

    # ---- model ----
    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s backbone, %s pose, %.2fM params",
                cfg["model"]["visual"]["backbone"],
                "with" if cfg["model"]["visual"]["use_pose"] else "no",
                num_params / 1e6)

    # ---- loss ----
    class_weights = None
    if cfg["train"]["loss"]["use_class_weights"]:
        # Inverse frequency from inner_train labels (not full train) so the
        # weights reflect what the model actually sees.
        inner_labels = [train_ds_full.all_labels[id_to_idx[c]] for c in inner_train_ids]
        counts = np.bincount(inner_labels, minlength=cfg["model"]["num_classes"])
        weights = len(inner_labels) / (cfg["model"]["num_classes"] * counts.clip(min=1))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        logger.info("Class weights (CE):")
        for name, w in zip(("positive", "neutral", "negative"), weights):
            logger.info("  %-8s: %.4f", name, w)

    active_branches = list(cfg["train"].get("active_branches", ["visual", "audio", "scene", "temporal"]))
    # active_branches = cfg["train"].get("active_branches", ["visual", "audio", "scene"])
    # if cfg["model"].get("temporal", {}).get("enabled", False):
    #     active_branches.append("temporal")

    loss_fn = MultiBranchCELoss(
        class_weights=class_weights,
        aux_weight=cfg["train"]["loss"]["aux_loss_weight"],
        active_branches=active_branches,
    ).to(device)

    # ---- optimizer / scheduler ----
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if cfg["train"]["amp"] else None
    # scaler = torch.cuda.amp.GradScaler() if cfg["train"]["amp"] else None

    stopper = EarlyStopper(patience=cfg["train"]["patience"])
    history: list[dict] = []
    best_state = None

    rng = np.random.default_rng(seed)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, loss_fn=loss_fn, device=device,
            grad_clip=cfg["train"]["grad_clip"], amp=cfg["train"]["amp"],
            scaler=scaler, branch_dropout=cfg["train"].get("branch_dropout", 0.0),
            active_branches=active_branches, rng=rng, epoch=epoch,
            feature_dropout=cfg["train"].get("feature_dropout", 0.0),
        )
        dev_metrics, _, _ = evaluate(
            model=model, loader=dev_loader, device=device,
            amp=cfg["train"]["amp"], num_classes=cfg["model"]["num_classes"],
        )
        dt = time.time() - t0
        improved, should_stop = stopper.step(dev_metrics["accuracy"], epoch)

        row = {
            "epoch": epoch,
            "time_sec": dt,
            **tr,
            "dev_acc": dev_metrics["accuracy"],
            "dev_f1": dev_metrics["weighted_f1"],
        }
        history.append(row)
        logger.info(
            "epoch %2d/%d  %.1fs  train_loss=%.4f train_acc=%.4f  "
            "dev_acc=%.4f f1=%.4f%s",
            epoch, cfg["train"]["epochs"], dt,
            tr["train_loss"], tr["train_acc"],
            dev_metrics["accuracy"], dev_metrics["weighted_f1"],
            "  *best" if improved else "",
        )

        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if should_stop:
            logger.info("Early stopping triggered at epoch %d (best epoch %d).",
                        epoch, stopper.best_epoch)
            break

    if best_state is None:
        logger.warning("No improvement ever — using final epoch state.")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- final test on official Val with best dev checkpoint ----
    model.load_state_dict(best_state)
    test_metrics, test_preds, test_targets = evaluate(
        model=model, loader=test_loader, device=device,
        amp=cfg["train"]["amp"], num_classes=cfg["model"]["num_classes"],
    )
    logger.info("TEST (official Val) — acc=%.4f f1=%.4f",
                test_metrics["accuracy"], test_metrics["weighted_f1"])

    # ---- artifacts ----
    final = {
        "seed": seed,
        "best_dev_acc": stopper.best_metric,
        "best_dev_epoch": stopper.best_epoch,
        "test_metrics": test_metrics,
    }
    save_run_artifacts(
        output_dir=output_dir,
        cfg=cfg,
        history=history,
        final_metrics=final,
        model_state=best_state,
    )
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="vgaf",
                        help="dataset config name or path (default: vgaf)")
    parser.add_argument("--ablation", action="append", default=[],
                        help="ablation name (repeat for multiple). e.g. --ablation no_pose")
    parser.add_argument("--override", nargs="+", default=[],
                        help="config overrides like train.lr=1e-4")
    parser.add_argument("--output_root", type=Path, default=_PROJECT_ROOT / "results")
    parser.add_argument("--output_name", type=str, default=None,
                        help="experiment name; defaults to vgaf_<ablations>_<timestamp>")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the seed list with a single seed")
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

    seeds = [args.seed] if args.seed is not None else cfg["train"]["seeds"]

    name = args.output_name or (
        "vgaf_" + ("_".join(args.ablation) if args.ablation else "main")
        + "_" + time.strftime("%Y%m%d_%H%M%S")
    )
    run_root = args.output_root / name
    run_root.mkdir(parents=True, exist_ok=True)
    logger.info("Run root: %s", run_root)

    per_seed = []
    for s in seeds:
        seed_dir = run_root / f"seed_{s}"
        logger.info("=" * 60)
        logger.info("SEED %d  →  %s", s, seed_dir)
        logger.info("=" * 60)
        per_seed.append(run_one_seed(cfg, s, seed_dir))

    # Aggregate across seeds.
    accs = [r["test_metrics"]["accuracy"] for r in per_seed]
    f1s = [r["test_metrics"]["weighted_f1"] for r in per_seed]
    summary = {
        "seeds": seeds,
        "test_acc_mean": float(np.mean(accs)),
        "test_acc_std": float(np.std(accs)),
        "test_f1_mean": float(np.mean(f1s)),
        "test_f1_std": float(np.std(f1s)),
        "per_seed_acc": accs,
        "per_seed_f1": f1s,
    }
    import json
    with open(run_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("DONE — test acc %.4f ± %.4f  (n=%d)",
                summary["test_acc_mean"], summary["test_acc_std"], len(seeds))
    logger.info("Artifacts in %s", run_root)


if __name__ == "__main__":
    main()
