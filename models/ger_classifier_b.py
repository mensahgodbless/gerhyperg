from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from models.ger_classifier import GERClassifier, VisualBranch
from models.encoder_variants_b import MeanPoolNoMLPEncoder

# Backbones this file adds on top of VisualBranch.BACKBONES.
_EXTRA_BACKBONES = ("mean_pool_nomlp",)
# What each extra backbone is built as in the parent before being swapped.
_PARENT_PROXY = {"mean_pool_nomlp": "mean_pool"}

TEMPORAL_POOLS = ("attn", "uniform")


class VisualBranchB(VisualBranch):
    """VisualBranch plus the ``mean_pool_nomlp`` backbone and uniform pooling."""

    BACKBONES = VisualBranch.BACKBONES + _EXTRA_BACKBONES

    def __init__(
        self,
        per_person_dim: int = 2048,
        d_model: int = 256,
        num_classes: int = 3,
        num_hyperedge_prototypes: int = 4,
        gat_heads: int = 4,
        dropout: float = 0.1,
        backbone: str = "hypergraph",
        temporal_pool: str = "attn",
    ) -> None:
        if backbone not in self.BACKBONES:
            raise ValueError(
                f"backbone must be one of {self.BACKBONES}, got {backbone!r}"
            )
        if temporal_pool not in TEMPORAL_POOLS:
            raise ValueError(
                f"temporal_pool must be one of {TEMPORAL_POOLS}, got {temporal_pool!r}"
            )

        super().__init__(
            per_person_dim=per_person_dim,
            d_model=d_model,
            num_classes=num_classes,
            num_hyperedge_prototypes=num_hyperedge_prototypes,
            gat_heads=gat_heads,
            dropout=dropout,
            backbone=_PARENT_PROXY.get(backbone, backbone),
        )

        self.backbone_name = backbone
        self.temporal_pool = temporal_pool

        if backbone == "mean_pool_nomlp":
            self.encoder = MeanPoolNoMLPEncoder(input_dim=d_model)

        if temporal_pool == "uniform":
            # Remove the learned temporal attention parameters rather than
            # leaving them unused, so param counts and state_dicts are honest.
            self.temporal_attn = None

    def forward(
        self,
        per_person_features: torch.Tensor,    # [B, T, N_max, per_person_dim]
        person_mask: torch.Tensor,            # [B, T, N_max]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_pool == "attn":
            return super().forward(per_person_features, person_mask)

        mask_bool = person_mask.bool()
        x = self.projector(per_person_features)              # [B, T, N, d_model]
        x = self.encoder(x, person_mask=person_mask)         # [B, T, d_model]

        frame_valid = mask_bool.any(dim=-1)                  # [B, T]
        valid = frame_valid.float()
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        t_weights = valid / denom                            # [B, T]
        clip_repr = (x * t_weights.unsqueeze(-1)).sum(dim=1) # [B, d_model]

        logits = self.head(clip_repr)
        return logits, clip_repr


class GERClassifierB(GERClassifier):
    """GERClassifier with the decomposition axes exposed."""

    def __init__(
        self,
        per_person_dim: int = 2048,
        audio_dim: int = 1024,
        scene_dim: int = 512,
        temporal_dim: int = 768,
        d_model: int = 256,
        num_classes: int = 3,
        num_hyperedge_prototypes: int = 4,
        gat_heads: int = 4,
        fusion_per_class: bool = False,
        dropout: float = 0.1,
        visual_backbone: str = "hypergraph",
        use_pose: bool = True,
        use_temporal: bool = False,
        visual_temporal_pool: str = "attn",
    ) -> None:
        if visual_temporal_pool not in TEMPORAL_POOLS:
            raise ValueError(
                f"visual_temporal_pool must be one of {TEMPORAL_POOLS}, "
                f"got {visual_temporal_pool!r}"
            )
        if visual_backbone == "holistic" and visual_temporal_pool != "attn":
            raise ValueError(
                "holistic collapses persons and frames in a single mean and has "
                "no temporal pooling stage, so visual_temporal_pool does not "
                "apply. Leave it at the default."
            )

        super().__init__(
            per_person_dim=per_person_dim,
            audio_dim=audio_dim,
            scene_dim=scene_dim,
            temporal_dim=temporal_dim,
            d_model=d_model,
            num_classes=num_classes,
            num_hyperedge_prototypes=num_hyperedge_prototypes,
            gat_heads=gat_heads,
            fusion_per_class=fusion_per_class,
            dropout=dropout,
            visual_backbone=_PARENT_PROXY.get(visual_backbone, visual_backbone),
            use_pose=use_pose,
            use_temporal=use_temporal,
        )

        self.visual_backbone = visual_backbone
        self.visual_temporal_pool = visual_temporal_pool

        needs_b = (
            visual_backbone in _EXTRA_BACKBONES
            or visual_temporal_pool != "attn"
        )
        if not needs_b:
            # Published configuration: parent build is used as-is, so
            # initialisation is bit-identical to ger_classifier.py.
            return

        self.visual_branch = VisualBranchB(
            per_person_dim=per_person_dim,
            d_model=d_model,
            num_classes=num_classes,
            num_hyperedge_prototypes=num_hyperedge_prototypes,
            gat_heads=gat_heads,
            dropout=dropout,
            backbone=visual_backbone,
            temporal_pool=visual_temporal_pool,
        )


# smoke test

if __name__ == "__main__":
    """Run with: python -m models.ger_classifier_b"""
    from models.ger_classifier import GERClassifier as _Parent

    B, T, N = 2, 15, 20

    CELLS = [
        ("mean_pool", "attn"),           # published Mean-pool (delegates to parent)
        ("mean_pool", "uniform"),        # minus learned temporal pooling
        ("mean_pool_nomlp", "attn"),     # minus refinement MLP
        ("mean_pool_nomlp", "uniform"),  # minus both
        ("hypergraph", "attn"),          # delegates to parent
        ("gat", "attn"),                 # delegates to parent
        ("holistic", "attn"),            # delegates to parent
    ]

    print("cell                                params    logits")
    for backbone, tpool in CELLS:
        for use_temporal in (False, True):
            model = GERClassifierB(
                visual_backbone=backbone,
                use_temporal=use_temporal,
                visual_temporal_pool=tpool,
            )
            out = model(
                torch.randn(B, T, N, 2048),
                torch.rand(B, T, N) > 0.3,
                torch.randn(B, 1024),
                torch.randn(B, 512),
                temporal=torch.randn(B, 768) if use_temporal else None,
            )
            p = sum(q.numel() for q in model.parameters())
            print(f"{backbone:16s} {tpool:8s} temporal={use_temporal!s:5s} "
                  f"{p:9d}  {tuple(out['logits'].shape)}")

    # Parity: published cells must match ger_classifier.py exactly.
    print("\nparity vs ger_classifier.py (published cells):")
    for backbone in ("hypergraph", "gat", "mean_pool", "holistic"):
        for use_temporal in (False, True):
            torch.manual_seed(0)
            a = _Parent(visual_backbone=backbone, use_temporal=use_temporal)
            torch.manual_seed(0)
            b = GERClassifierB(visual_backbone=backbone, use_temporal=use_temporal)
            sa = a.state_dict()
            sb = b.state_dict()
            same_keys = set(sa) == set(sb)
            same_vals = same_keys and all(
                torch.equal(sa[k], sb[k]) for k in sa
            )
            status = "identical" if same_vals else "DIFFERS"
            print(f"  {backbone:12s} temporal={use_temporal!s:5s} {status}")
