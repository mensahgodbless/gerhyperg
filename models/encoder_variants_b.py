
from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder_variants import MeanPoolEncoder


class MeanPoolNoMLPEncoder(MeanPoolEncoder):
   

    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__(input_dim=input_dim, eps=eps)
        # Drop the refinement MLP entirely. Assigning None to a registered
        # submodule name is supported by nn.Module and removes its
        # parameters from .parameters() and .state_dict().
        self.refine = None

    def forward(
        self,
        person_features: torch.Tensor,        # [B, T, N, D]
        person_mask: torch.Tensor,            # [B, T, N]
    ) -> torch.Tensor:
        # Identical to MeanPoolEncoder.forward minus the residual refine.
        m = person_mask.unsqueeze(-1).float()                       # [B, T, N, 1]
        masked = person_features * m
        denom = m.sum(dim=2).clamp(min=self.eps)                    # [B, T, 1]
        return masked.sum(dim=2) / denom                            # [B, T, D]
