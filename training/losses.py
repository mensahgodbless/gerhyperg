"""
Composite loss for multi-branch decision fusion.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBranchCELoss(nn.Module):
    """
    Cross-entropy on fused logits + auxiliary CE on each active branch.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        aux_weight: float = 0.3,
        active_branches: Iterable[str] = ("visual", "audio", "scene"),
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None  # type: ignore[assignment]
        self.aux_weight = float(aux_weight)
        self.active_branches = tuple(active_branches)
        for b in self.active_branches:
            if b not in ("visual", "audio", "scene", "temporal"):
                raise ValueError(f"Unknown branch: {b!r}")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        weight = self.class_weights if self.class_weights is not None else None

        main = F.cross_entropy(outputs["logits"], target, weight=weight)
        result = {"main": main}

        total = main
        for branch in self.active_branches:
            key = f"{branch}_logits"
            if key not in outputs:
                continue
            aux = F.cross_entropy(outputs[key], target, weight=weight)
            result[f"aux_{branch}"] = aux
            total = total + self.aux_weight * aux

        result["total"] = total
        return result
