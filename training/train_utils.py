"""
Shared training utilities
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- config

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here] + list(here.parents):
        if (candidate / "configs").is_dir() and (candidate / "requirements.txt").is_file():
            return candidate
    cwd = Path.cwd()
    if (cwd / "configs").is_dir():
        return cwd
    raise FileNotFoundError("Cannot locate project root (no configs/ dir found)")


def load_config(
    dataset_config: str,
    ablation_configs: Optional[List[str]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    root = _find_project_root()

    # 1. Base config — try base.yaml then default.yaml for backward compat.
    base_path = root / "configs" / "base.yaml"
    if not base_path.exists():
        base_path = root / "configs" / "default.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"No base.yaml or default.yaml under {root/'configs'}")
    with open(base_path) as f:
        cfg = yaml.safe_load(f) or {}

    # 2. Dataset config.
    p = Path(dataset_config)
    if not p.is_absolute() and not p.exists():
        p = root / "configs" / f"{dataset_config}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config}")
    with open(p) as f:
        cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    # 3. Ablation configs (in order; later overrides earlier).
    for name in (ablation_configs or []):
        ap = Path(name)
        if not ap.is_absolute() and not ap.exists():
            ap = root / "configs" / "ablations" / f"{name}.yaml"
        if not ap.exists():
            raise FileNotFoundError(f"Ablation config not found: {name}")
        with open(ap) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    # 4. CLI overrides.
    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)

    return cfg


def save_resolved_config(cfg: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Construct GERClassifier from a resolved config.

    Dispatches on cfg['model']['visual']['backbone'] for the visual branch
    variant (hypergraph / gat / mean_pool / holistic).
    """
    # Late import to avoid pulling heavy deps at module-load time.
    from models.ger_classifier import GERClassifier

    m = cfg["model"]
    vis = m["visual"]

    per_person_dim = vis["per_person_dim"]
    if not vis.get("use_pose", True):
        per_person_dim = per_person_dim - 64
    
    temporal_cfg = m.get("temporal", {})
    use_temporal = temporal_cfg.get("enabled", False)
    temporal_dim = temporal_cfg.get("temporal_dim", 768)

    model = GERClassifier(
        per_person_dim=per_person_dim,
        audio_dim=m["audio"]["audio_dim"],
        scene_dim=m["scene"]["scene_dim"],
        temporal_dim=temporal_dim,     
        d_model=m["d_model"],
        num_classes=m["num_classes"],
        num_hyperedge_prototypes=vis.get("num_prototypes", 4),
        fusion_per_class=m["fusion"].get("per_class", False),
        dropout=m.get("dropout", 0.1),
        visual_backbone=vis["backbone"],
        gat_heads=vis.get("gat_heads", 4),
        use_pose=vis.get("use_pose", True),
        use_temporal=use_temporal, 
    )
    return model


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    tcfg = cfg["train"]
    opt_name = tcfg["optimizer"].lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )
    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Cosine schedule with linear warmup. Returns None if scheduler="none"."""
    tcfg = cfg["train"]
    sched = tcfg.get("scheduler", "none").lower()
    if sched == "none":
        return None
    if sched != "cosine":
        raise ValueError(f"Unsupported scheduler: {sched}")

    total_epochs = tcfg["epochs"]
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------- early stopping

@dataclass
class EarlyStopper:
    patience: int = 10
    best_metric: float = -float("inf")
    epochs_since_improvement: int = 0
    best_epoch: int = -1

    def step(self, current_metric: float, epoch: int) -> Tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_since_improvement = 0
            self.best_epoch = epoch
            return True, False
        self.epochs_since_improvement += 1
        return False, self.epochs_since_improvement >= self.patience


# ----------------------------------------------------- train / eval

def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _branch_dropout_flags(
    p: float,
    active_branches: Iterable[str],
    rng: np.random.Generator,
) -> Dict[str, bool]:
    """Per-step branch dropout. Inactive branches are always dropped."""
    active_set = set(active_branches)
    flags = {}
    for b in ("visual", "audio", "scene", "temporal"):
        if b not in active_set:
            flags[f"drop_{b}"] = True
        else:
            flags[f"drop_{b}"] = rng.random() < p if p > 0 else False

    # Don't drop ALL branches in a single step — leave at least one alive.
    if all(flags[f"drop_{b}"] for b in active_set):
        keep = rng.choice(list(active_set))
        flags[f"drop_{keep}"] = False
    return flags


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
    amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    branch_dropout: float,
    active_branches: Iterable[str],
    rng: np.random.Generator,
    epoch: int,
    log_every: int = 50,
    feature_dropout: float = 0.0
) -> Dict[str, float]:
    model.train()
    sums = {"total": 0.0, "main": 0.0, "n": 0}
    correct = 0

    for step, batch in enumerate(loader):
        batch = _move_batch(batch, device)

        target = batch["label"]
        bs = target.shape[0]

        per_person_features = batch["per_person_features"]
        if feature_dropout > 0 and model.training:
            # Create dropout mask: keep probability = 1 - feature_dropout
            mask = torch.rand_like(per_person_features) > feature_dropout
            per_person_features = per_person_features * mask.float()

        flags = _branch_dropout_flags(branch_dropout, active_branches, rng)

        optimizer.zero_grad(set_to_none=True)
        # with torch.cuda.amp.autocast(enabled=amp):
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            out = model(
                per_person_features=per_person_features,
                person_mask=batch["person_mask"],
                audio=batch["audio"],
                scene=batch["scene"],
                temporal=batch.get("temporal"), 
                **flags,
            )
            losses = loss_fn(out, target)
            total_loss = losses["total"]

        if amp and scaler is not None:
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        preds = out["logits"].argmax(dim=-1)
        correct += (preds == target).sum().item()
        sums["total"] += float(total_loss.item()) * bs
        sums["main"] += float(losses["main"].item()) * bs
        sums["n"] += bs

        if (step + 1) % log_every == 0:
            logger.info(
                "  epoch %d step %d/%d  loss %.4f  main %.4f",
                epoch, step + 1, len(loader),
                sums["total"] / sums["n"], sums["main"] / sums["n"],
            )

    n = max(1, sums["n"])
    return {
        "train_loss": sums["total"] / n,
        "train_main_loss": sums["main"] / n,
        "train_acc": correct / n,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    num_classes: int = 3,
    drop_flags: Optional[Dict[str, bool]] = None,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:

    from training.metrics import compute_all_metrics

    model.eval()
    flags = drop_flags or {}
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        target = batch["label"]
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(
                per_person_features=batch["per_person_features"],
                person_mask=batch["person_mask"],
                audio=batch["audio"],
                scene=batch["scene"],
                temporal=batch.get("temporal"), 
                drop_visual=flags.get("drop_visual", False),
                drop_audio=flags.get("drop_audio", False),
                drop_scene=flags.get("drop_scene", False),
                drop_temporal=flags.get("drop_temporal", False),
            )
        preds = out["logits"].argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_all_metrics(preds, targets, num_classes=num_classes,
                                  class_names=["positive", "neutral", "negative"])
    return metrics, preds, targets


# ------------------ artifact saving -----------------------
def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def save_run_artifacts(
    output_dir: Path,
    cfg: Dict[str, Any],
    history: List[Dict[str, float]],
    final_metrics: Dict[str, Any],
    model_state: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output_dir / "config.yaml")
    with open(output_dir / "history.json", "w") as f:
        json.dump(_jsonify(history), f, indent=2)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(_jsonify(final_metrics), f, indent=2)
    if model_state is not None:
        torch.save(model_state, output_dir / "best_model.pt")

# def save_run_artifacts(
#     output_dir: Path,
#     cfg: Dict[str, Any],
#     history: List[Dict[str, float]],
#     final_metrics: Dict[str, Any],
#     model_state: Optional[Dict[str, torch.Tensor]] = None,
# ) -> None:
#     """Persist the run: config, per-epoch history, final metrics, model."""
#     output_dir.mkdir(parents=True, exist_ok=True)
#     save_resolved_config(cfg, output_dir / "config.yaml")
#     with open(output_dir / "history.json", "w") as f:
#         json.dump(history, f, indent=2)

#     # Convert non-JSON-serializable numpy arrays.
#     fm_dump = copy.deepcopy(final_metrics)
#     if "confusion_matrix" in fm_dump and isinstance(fm_dump["confusion_matrix"], np.ndarray):
#         fm_dump["confusion_matrix"] = fm_dump["confusion_matrix"].tolist()
#     with open(output_dir / "metrics.json", "w") as f:
#         json.dump(fm_dump, f, indent=2)

#     if model_state is not None:
#         torch.save(model_state, output_dir / "best_model.pt")
