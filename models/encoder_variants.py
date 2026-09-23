"""
Alternative visual encoders for ablation studies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolEncoder(nn.Module):
    
    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.refine = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(
        self,
        person_features: torch.Tensor,        # [B, T, N, D]
        person_mask: torch.Tensor,            # [B, T, N] bool/float
    ) -> torch.Tensor:
        m = person_mask.unsqueeze(-1).float()                       # [B, T, N, 1]
        masked = person_features * m
        denom = m.sum(dim=2).clamp(min=self.eps)                    # [B, T, 1]
        pooled = masked.sum(dim=2) / denom                          # [B, T, D]
        return pooled + self.refine(pooled)                         # residual refine


class GATEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim {input_dim} not divisible by num_heads {num_heads}")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q = nn.Linear(input_dim, input_dim)
        self.k = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, input_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

        # Frame-level pool: attention readout.
        self.pool_proj = nn.Linear(input_dim, input_dim)
        self.pool_query = nn.Linear(input_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        person_features: torch.Tensor,    # [B, T, N, D]
        person_mask: torch.Tensor,        # [B, T, N]
    ) -> torch.Tensor:
        B, T, N, D = person_features.shape
        H = self.num_heads
        Hd = self.head_dim

        # Flatten batch and time for the per-frame attention.
        x = person_features.reshape(B * T, N, D)
        mask = person_mask.reshape(B * T, N).bool()        # True = valid

        q = self.q(x).reshape(B * T, N, H, Hd).transpose(1, 2)   # [BT, H, N, Hd]
        k = self.k(x).reshape(B * T, N, H, Hd).transpose(1, 2)
        v = self.v(x).reshape(B * T, N, H, Hd).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale     # [BT, H, N, N]

        # Mask out keys that are padded persons.
        key_mask = (~mask).unsqueeze(1).unsqueeze(1)             # [BT, 1, 1, N]
        neg_inf = torch.finfo(attn_logits.dtype).min / 2
        attn_logits = attn_logits.masked_fill(key_mask, neg_inf)

        # Frames with zero valid queries → after softmax we'll mask them too,
        # but softmax over an all -inf row gives NaN. Replace those query
        # rows with uniform attention; the result is irrelevant because
        # the query position will be masked out at pooling.
        all_dead_row = (~mask).all(dim=-1, keepdim=True)         # [BT, 1]
        # Detect rows of pure -inf and fix them.
        row_dead = (attn_logits == neg_inf).all(dim=-1, keepdim=True)  # [BT, H, N, 1]
        attn_logits = torch.where(row_dead, torch.zeros_like(attn_logits), attn_logits)

        attn = F.softmax(attn_logits, dim=-1)                    # [BT, H, N, N]
        attn = self.attn_drop(attn)

        out = attn @ v                                            # [BT, H, N, Hd]
        out = out.transpose(1, 2).reshape(B * T, N, D)
        out = self.out(out)
        out = self.norm(x + out)                                  # residual

        # Frame-level attention pool over valid persons → [BT, D].
        scores = self.pool_query(torch.tanh(self.pool_proj(out))).squeeze(-1)  # [BT, N]
        scores = scores.masked_fill(~mask, neg_inf)
        pool_weights = F.softmax(scores, dim=-1)
        # Zero contribution for frames with no valid persons (rare).
        any_valid = mask.any(dim=-1, keepdim=True).float()
        pool_weights = pool_weights * any_valid
        pooled = (pool_weights.unsqueeze(-1) * out).sum(dim=1)    # [BT, D]

        return pooled.reshape(B, T, D)
