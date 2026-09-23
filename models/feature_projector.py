"""
Feature projection for pre-extracted person representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 256,
    ) -> None:
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for the linear layer."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to the model working dimension.

        Supports arbitrary leading batch dimensions — the projection is
        applied to the **last** axis only.

        Args:
            x: ``[B, T, N_max, input_dim]`` pre-extracted features during
                training, or any shape ``[*, input_dim]``.

        Returns:
            ``[B, T, N_max, output_dim]`` (or ``[*, output_dim]``)
            projected features.
        """
        assert x.shape[-1] == self.input_dim, (
            f"Expected last dim {self.input_dim}, got {x.shape[-1]}"
        )

        # Remember original shape for reshape-back
        leading_shape = x.shape[:-1]

        # Flatten to 2-D for efficient matmul
        flat = x.reshape(-1, self.input_dim)          # [*, input_dim]
        projected = self.projection(flat)              # [*, output_dim]

        # Restore leading dimensions
        out = projected.reshape(*leading_shape, self.output_dim)  # [B, T, N_max, output_dim]
        return out


class AudioProjector(nn.Module):

    def __init__(
        self,
        audio_dim: int = 1024,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.audio_dim = audio_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features.

        Args:
            x: ``[B, T, audio_dim]`` per-frame HuBERT features.

        Returns:
            ``[B, T, output_dim]`` projected audio features.
        """
        assert x.shape[-1] == self.audio_dim, (
            f"Expected last dim {self.audio_dim}, got {x.shape[-1]}"
        )
        return self.projection(x)


class TextProjector(nn.Module):
    
    def __init__(
        self,
        text_dim: int = 768,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.shape[-1] == self.text_dim, (
            f"Expected last dim {self.text_dim}, got {x.shape[-1]}"
        )
        return self.projection(x)


class MultiStreamProjector(nn.Module):
    
    def __init__(
        self,
        face_dim: int = 1408,
        body_dim: int = 512,
        bbox_dim: int = 64,
        output_dim: int = 256,
        stream_hidden: int = 0,
    ) -> None:
        
        super().__init__()
        self.face_dim = face_dim
        self.body_dim = body_dim
        self.bbox_dim = bbox_dim
        self.output_dim = output_dim
        self.total_input_dim = face_dim + body_dim + bbox_dim

        if stream_hidden > 0:
            self.face_proj = nn.Sequential(
                nn.Linear(face_dim, stream_hidden),
                nn.GELU(),
                nn.Linear(stream_hidden, output_dim // 2),
            )
            self.body_proj = nn.Sequential(
                nn.Linear(body_dim, stream_hidden),
                nn.GELU(),
                nn.Linear(stream_hidden, output_dim // 4),
            )
            self.bbox_proj = nn.Sequential(
                nn.Linear(bbox_dim, stream_hidden),
                nn.GELU(),
                nn.Linear(stream_hidden, output_dim - output_dim // 2 - output_dim // 4),
            )
        else:
            self.face_proj = nn.Linear(face_dim, output_dim // 2)
            self.body_proj = nn.Linear(body_dim, output_dim // 4)
            # Remaining dims to ensure exact output_dim
            remaining = output_dim - output_dim // 2 - output_dim // 4
            self.bbox_proj = nn.Linear(bbox_dim, remaining)

        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.shape[-1] == self.total_input_dim, (
            f"Expected last dim {self.total_input_dim}, got {x.shape[-1]}"
        )

        leading_shape = x.shape[:-1]
        flat = x.reshape(-1, self.total_input_dim)

        face_in = flat[:, : self.face_dim]                                      # [*, face_dim]
        body_in = flat[:, self.face_dim : self.face_dim + self.body_dim]        # [*, body_dim]
        bbox_in = flat[:, self.face_dim + self.body_dim :]                      # [*, bbox_dim]

        face_out = self.face_proj(face_in)   # [*, output_dim//2]
        body_out = self.body_proj(body_in)   # [*, output_dim//4]
        bbox_out = self.bbox_proj(bbox_in)   # [*, remaining]

        combined = torch.cat([face_out, body_out, bbox_out], dim=-1)  # [*, output_dim]
        combined = self.norm(combined)
        combined = self.act(combined)

        return combined.reshape(*leading_shape, self.output_dim)