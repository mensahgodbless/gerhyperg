"""
Top-level group emotion recognition classifier (configurable backbone).

Architecture (decision-level fusion):
    Visual branch (one of, per cfg.model.visual.backbone):
        - hypergraph  → SoftHypergraphEncoder (default, main model)
        - gat         → GATEncoder (Wang ICPR-style pairwise baseline)
        - mean_pool   → MeanPoolEncoder (simplest aggregation baseline)
        - holistic    → HolisticVisualBranch (no per-person modelling)

    Audio branch:HuBERT 
    Scene branch: CLIP whole-frame
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.feature_projector import FeatureProjector, AudioProjector
from models.soft_hypergraph_encoder import SoftHypergraphEncoder
from models.encoder_variants import MeanPoolEncoder, GATEncoder
from models.holistic_visual_branch import HolisticVisualBranch

POSE_DIM = 64  # pose block size at the end of each per-person feature vector


# ---------------------------------------------------------- scene projector

class SceneProjector(nn.Module):
    """Project mean-pooled CLIP whole-frame features to d_model."""

    def __init__(self, scene_dim: int = 512, output_dim: int = 256) -> None:
        super().__init__()
        self.scene_dim = scene_dim
        self.output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(scene_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.scene_dim
        return self.projection(x)



# ---------------------------------------------------------- temporal projector

class TemporalProjector(nn.Module):
    """Project TimeSformer [CLS] features to d_model."""

    def __init__(self, temporal_dim: int = 768, output_dim: int = 256) -> None:
        super().__init__()
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(temporal_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.temporal_dim
        return self.projection(x)
# ----------------------------------------------------------- branches

def _make_head(d_model: int, num_classes: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, num_classes),
    )


class VisualBranch(nn.Module):
    """Per-person visual branch with configurable backbone."""

    BACKBONES = ("hypergraph", "gat", "mean_pool")  # holistic is a separate class

    def __init__(
        self,
        per_person_dim: int = 2048,
        d_model: int = 256,
        num_classes: int = 3,
        num_hyperedge_prototypes: int = 4,
        gat_heads: int = 4,
        dropout: float = 0.1,
        backbone: str = "hypergraph",
    ) -> None:
        super().__init__()
        if backbone not in self.BACKBONES:
            raise ValueError(
                f"VisualBranch backbone must be one of {self.BACKBONES}, got {backbone!r}"
            )
        self.backbone_name = backbone
        self.d_model = d_model
        self.num_classes = num_classes

        self.projector = FeatureProjector(input_dim=per_person_dim, output_dim=d_model)

        if backbone == "hypergraph":
            self.encoder = SoftHypergraphEncoder(
                input_dim=d_model, num_prototypes=num_hyperedge_prototypes,
                dropout=dropout
            )
        elif backbone == "gat":
            self.encoder = GATEncoder(input_dim=d_model, num_heads=gat_heads, dropout=dropout)
        elif backbone == "mean_pool":
            self.encoder = MeanPoolEncoder(input_dim=d_model)
        else:
            raise AssertionError(f"unreachable: {backbone}")

        self.temporal_attn = nn.Linear(d_model, 1)
        self.head = _make_head(d_model, num_classes, dropout)

    def forward(
        self,
        per_person_features: torch.Tensor,    # [B, T, N_max, per_person_dim]
        person_mask: torch.Tensor,            # [B, T, N_max]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_bool = person_mask.bool()

        x = self.projector(per_person_features)              # [B, T, N, d_model]
        x = self.encoder(x, person_mask=person_mask)         # [B, T, d_model]

        frame_valid = mask_bool.any(dim=-1)                  # [B, T]
        t_logits = self.temporal_attn(x).squeeze(-1)         # [B, T]
        t_logits = t_logits.masked_fill(~frame_valid, float("-inf"))
        t_weights = torch.softmax(t_logits, dim=1)
        t_weights = torch.nan_to_num(t_weights, nan=0.0)
        clip_repr = (x * t_weights.unsqueeze(-1)).sum(dim=1) # [B, d_model]

        logits = self.head(clip_repr)
        return logits, clip_repr


class AudioBranch(nn.Module):
    def __init__(self, audio_dim=1024, d_model=256, num_classes=3, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.projector = AudioProjector(audio_dim=audio_dim, output_dim=d_model)
        self.head = _make_head(d_model, num_classes, dropout)

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.projector(audio)
        return self.head(feat), feat


class SceneBranch(nn.Module):
    def __init__(self, scene_dim=512, d_model=256, num_classes=3, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.projector = SceneProjector(scene_dim=scene_dim, output_dim=d_model)
        self.head = _make_head(d_model, num_classes, dropout)

    def forward(self, scene: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.projector(scene)
        return self.head(feat), feat


class TemporalBranch(nn.Module):
    def __init__(self, temporal_dim=768, d_model=256, num_classes=3, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.projector = TemporalProjector(temporal_dim=temporal_dim, output_dim=d_model)
        self.head = _make_head(d_model, num_classes, dropout)

    def forward(self, temporal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.projector(temporal)
        return self.head(feat), feat

# fusion

class DecisionFusion(nn.Module):
    """Learned softmax-weighted decision fusion over branch logits."""

    def __init__(self, num_branches=3, num_classes=3, per_class=False) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        self.per_class = per_class
        if per_class:
            self.weight_logits = nn.Parameter(torch.zeros(num_branches, num_classes))
        else:
            self.weight_logits = nn.Parameter(torch.zeros(num_branches))

    def forward(self, branch_logits: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(branch_logits) == self.num_branches
        w = torch.softmax(self.weight_logits, dim=0)
        stacked = torch.stack(branch_logits, dim=0)
        if self.per_class:
            fused = (stacked * w.unsqueeze(1)).sum(dim=0)
        else:
            fused = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return fused, w


# top-level

class GERClassifier(nn.Module):
    """Per-person multimodal classifier with decision-level fusion.

    Visual backbone is selected by ``visual_backbone`` (hypergraph / gat /
    mean_pool / holistic). When ``use_pose=False`` the last 64 dims of the
    per-person features are sliced before the projector input.
    """

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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.visual_backbone = visual_backbone
        self.use_pose = use_pose
        self.use_temporal = use_temporal 

        # Build the visual branch (regular or holistic).
        if visual_backbone == "holistic":
            self.visual_branch = HolisticVisualBranch(
                per_person_dim=per_person_dim,
                d_model=d_model,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.visual_branch = VisualBranch(
                per_person_dim=per_person_dim,
                d_model=d_model,
                num_classes=num_classes,
                num_hyperedge_prototypes=num_hyperedge_prototypes,
                gat_heads=gat_heads,
                dropout=dropout,
                backbone=visual_backbone,
            )
        
        if use_temporal:
            self.temporal_branch = TemporalBranch(
                temporal_dim=temporal_dim, d_model=d_model,
                num_classes=num_classes, dropout=dropout,
            )

        self.audio_branch = AudioBranch(
            audio_dim=audio_dim, d_model=d_model,
            num_classes=num_classes, dropout=dropout,
        )
        self.scene_branch = SceneBranch(
            scene_dim=scene_dim, d_model=d_model,
            num_classes=num_classes, dropout=dropout,
        )

        n_branches = 3 + (1 if use_temporal else 0)
        self.fusion = DecisionFusion(
            num_branches=n_branches, num_classes=num_classes, per_class=fusion_per_class,
        )

    def _trim_pose(self, per_person_features: torch.Tensor) -> torch.Tensor:
        if self.use_pose:
            return per_person_features
        # Slice off the trailing POSE_DIM dimensions.
        return per_person_features[..., : per_person_features.shape[-1] - POSE_DIM]

    def forward(
        self,
        per_person_features: torch.Tensor,    # [B, T, N_max, 2048]
        person_mask: torch.Tensor,            # [B, T, N_max]
        audio: torch.Tensor,                  # [B, audio_dim]
        scene: torch.Tensor,                  # [B, scene_dim]
        temporal: torch.Tensor = None,
        drop_visual: bool = False,
        drop_audio: bool = False,
        drop_scene: bool = False,
        drop_temporal: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = per_person_features.shape[0]
        device = per_person_features.device
        nc = self.num_classes

        if self.use_temporal and temporal is None:
            # Allow drop_temporal=True with temporal=None (zero fill above);
            # disallow silent omission when the user thinks it's active.
            if not drop_temporal:
                raise ValueError(
                    "use_temporal=True but no `temporal` tensor was passed to forward()."
                )

        if drop_visual:
            visual_logits = torch.zeros(B, nc, device=device)
            visual_feat = torch.zeros(B, self.d_model, device=device)
        else:
            v_in = self._trim_pose(per_person_features)
            visual_logits, visual_feat = self.visual_branch(v_in, person_mask)

        if drop_audio:
            audio_logits = torch.zeros(B, nc, device=device)
            audio_feat = torch.zeros(B, self.d_model, device=device)
        else:
            audio_logits, audio_feat = self.audio_branch(audio)

        if drop_scene:
            scene_logits = torch.zeros(B, nc, device=device)
            scene_feat = torch.zeros(B, self.d_model, device=device)
        else:
            scene_logits, scene_feat = self.scene_branch(scene)

        if self.use_temporal:
            if drop_temporal or temporal is None:
                temporal_logits = torch.zeros(B, nc, device=device)
                temporal_feat = torch.zeros(B, self.d_model, device=device)
            else:
                temporal_logits, temporal_feat = self.temporal_branch(temporal)
        
        if self.use_temporal:
            fused_logits, fusion_weights = self.fusion(
                (visual_logits, audio_logits, scene_logits, temporal_logits)
            )
        else:
            fused_logits, fusion_weights = self.fusion(
                (visual_logits, audio_logits, scene_logits)
            )
        

        out = {
            "logits": fused_logits,
            "visual_logits": visual_logits,
            "audio_logits": audio_logits,
            "scene_logits": scene_logits,
            "fusion_weights": fusion_weights,
            "visual_feat": visual_feat,
            "audio_feat": audio_feat,
            "scene_feat": scene_feat,
        }

        if self.use_temporal:
            out["temporal_logits"] = temporal_logits
            out["temporal_feat"] = temporal_feat

        return out


# smoke test

if __name__ == "__main__":
    """Smoke test all backbone variants. Run with: python -m models.ger_classifier"""
    B, T, N = 2, 15, 20

    for backbone in ("hypergraph", "gat", "mean_pool", "holistic"):
        for use_pose in (True, False):
            for use_temporal in (False, True):
                ppd = 2048
                model = GERClassifier(
                    per_person_dim=ppd - (0 if use_pose else 64),
                    visual_backbone=backbone,
                    use_pose=use_pose,
                    use_temporal=use_temporal,
                )
                per_person = torch.randn(B, T, N, ppd)
                mask = (torch.rand(B, T, N) > 0.3)
                audio = torch.randn(B, 1024)
                scene = torch.randn(B, 512)
                temporal = torch.randn(B, 768) if use_temporal else None
                out = model(per_person, mask, audio, scene, temporal=temporal)
                #out = model(per_person, mask, audio, scene)
                params = sum(p.numel() for p in model.parameters())
                print(f"{backbone:11s}  use_pose={use_pose}  "
                    f"params={params:8d}  logits={tuple(out['logits'].shape)}")
    
