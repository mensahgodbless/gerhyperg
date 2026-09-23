"""
Holistic visual branch — Kumar ICPR 2024 style.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class HolisticVisualBranch(nn.Module):
    """Masked mean over (T, N) → projector → MLP head."""

    def __init__(
        self,
        per_person_dim: int = 2048,
        d_model: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.per_person_dim = per_person_dim
        self.d_model = d_model

        self.projection = nn.Sequential(
            nn.Linear(per_person_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        per_person_features: torch.Tensor,    # [B, T, N, D]
        person_mask: torch.Tensor,            # [B, T, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = person_mask.unsqueeze(-1).float()                          # [B, T, N, 1]
        masked = per_person_features * m                               # [B, T, N, D]
        # Masked mean over both T and N at once.
        total = masked.sum(dim=(1, 2))                                 # [B, D]
        denom = m.sum(dim=(1, 2)).clamp(min=1.0)                       # [B, 1]
        pooled = total / denom                                         # [B, D]

        clip_repr = self.projection(pooled)                            # [B, d_model]
        logits = self.head(clip_repr)                                  # [B, num_classes]
        return logits, clip_repr
