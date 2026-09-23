"""
Training helpers for the decomposition ablation. ADDITIVE.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

# Re-export the untouched helpers so `_b` scripts have one import site.
from training.train_utils import (  # noqa: F401
    EarlyStopper,
    build_optimizer,
    build_scheduler,
    evaluate,
    load_config,
    save_resolved_config,
    save_run_artifacts,
    set_seed,
    train_one_epoch,
)


def build_model(cfg: Dict[str, Any]) -> nn.Module:

    from models.ger_classifier_b import GERClassifierB

    m = cfg["model"]
    vis = m["visual"]

    per_person_dim = vis["per_person_dim"]
    if not vis.get("use_pose", True):
        per_person_dim = per_person_dim - 64

    temporal_cfg = m.get("temporal", {})
    use_temporal = temporal_cfg.get("enabled", False)
    temporal_dim = temporal_cfg.get("temporal_dim", 768)

    return GERClassifierB(
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
        visual_temporal_pool=vis.get("temporal_pool", "attn"),
    )
