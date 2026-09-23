"""
Overlapping soft hypergraph encoder for group-level feature encoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftHypergraphEncoder(nn.Module):
    """
    Fully differentiable overlapping soft hypergraph.
    """

    def __init__(
        self,
        input_dim: int,
        num_prototypes: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        
        super().__init__()
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

        # Hyperedge prototypes [M, D]
        self.hyperedge_prototypes = nn.Parameter(
            torch.randn(num_prototypes, input_dim) * 0.02
        )

        # Incidence: person → hyperedge 
        self.W_h = nn.Linear(input_dim, input_dim)

        # Person → hyperedge value projection 
        self.W_v = nn.Linear(input_dim, input_dim)

        # Message projection 
        self.W_e = nn.Linear(input_dim, input_dim)

        # Global context
        self.W_g = nn.Linear(input_dim, input_dim)
        self.W_u = nn.Linear(input_dim, input_dim)

        # Local/global gate 
        self.gate_linear = nn.Linear(input_dim, 1)

        # Attention pooling
        self.pool_proj = nn.Linear(input_dim, input_dim)
        self.pool_query = nn.Linear(input_dim, 1)

        # Layer norm for residual
        self.layer_norm = nn.LayerNorm(input_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        person_features: torch.Tensor,
        person_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode per-person features into frame-level group descriptors.
        """
        B, T, N, D = person_features.shape
        M = self.num_prototypes


        assert D == self.input_dim, (
            f"Feature dim {D} doesn't match input_dim {self.input_dim}"
        )

        
        # Expand mask for broadcasting: [B, T, N, 1]
        mask = person_mask.unsqueeze(-1).float()           # [B, T, N, 1]

        s = person_features                                 # [B, T, N, D]

        # ==============================================================
        # 1. Sigmoid incidence weights H_local [B, T, N, M]
        # ==============================================================
        # H(n, m) = σ(a_m^T · W_h(s_n))
        # where a_m = hyperedge_prototypes[m]

        h_proj = self.W_h(s)                                # [B, T, N, D]
        h_proj = self.dropout(h_proj)
        # Dot product with each prototype: [B, T, N, D] @ [M, D]^T → [B, T, N, M]
        incidence_logits = torch.einsum(
            "btnd,md->btnm", h_proj, self.hyperedge_prototypes
        )                                                    # [B, T, N, M]
        H_local = torch.sigmoid(incidence_logits)           # [B, T, N, M]

        # Zero out padded persons
        H_local = H_local * mask                            # [B, T, N, M]

        # ==============================================================
        # 2. Local hyperedge features [B, T, M, D]
        # ==============================================================
        # e_m = Σ_n H(n,m) · W_v(s_n) / (Σ_n H(n,m) + eps)

        v_proj = self.W_v(s)                                # [B, T, N, D]
        v_proj = self.dropout(v_proj)
        v_proj = v_proj * mask                              # zero padded

        # Weighted sum: [B, T, M, N] @ [B, T, N, D] → [B, T, M, D]
        H_t = H_local.permute(0, 1, 3, 2)                  # [B, T, M, N]
        weighted_sum = torch.matmul(H_t, v_proj)            # [B, T, M, D]

        # Denominator: sum of incidence per hyperedge
        H_sum = H_t.sum(dim=-1, keepdim=True)               # [B, T, M, 1]
        e_local = weighted_sum / (H_sum + self.eps)         # [B, T, M, D]

        # ==============================================================
        # 3. Row-normalise incidence [B, T, N, M]
        # ==============================================================
        # H_bar(n, m) = H(n, m) / (Σ_m' H(n, m') + eps)

        H_row_sum = H_local.sum(dim=-1, keepdim=True)       # [B, T, N, 1]
        H_bar = H_local / (H_row_sum + self.eps)            # [B, T, N, M]

        # ==============================================================
        # 4. Local messages [B, T, N, D]
        # ==============================================================
        # msg_n = Σ_m H_bar(n, m) · W_e(e_m)

        e_proj = self.W_e(e_local)                          # [B, T, M, D]
        e_proj = self.dropout(e_proj)
        # [B, T, N, M] @ [B, T, M, D] → [B, T, N, D]
        msg = torch.matmul(H_bar, e_proj)                   # [B, T, N, D]
        msg = msg * mask                                    # zero padded

        # ==============================================================
        # 5. Global context [B, T, D]
        # ==============================================================
        # e_global = masked_mean(W_g(s_n))

        g_proj = self.W_g(s) * mask                         # [B, T, N, D]
        g_proj = self.dropout(g_proj)
        valid_count = mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, T, 1, 1]
        # Sum over persons, divide by count
        e_global = g_proj.sum(dim=2) / valid_count.squeeze(-1)  # [B, T, D]

        # ==============================================================
        # 6. Person-specific gate β [B, T, N, 1]
        # ==============================================================
        beta = torch.sigmoid(self.gate_linear(s))           # [B, T, N, 1]

        # ==============================================================
        # 7. Residual update [B, T, N, D]
        # ==============================================================
        # r_n = LN(s_n + (1 - β_n) · msg_n + β_n · W_u(e_global))

        global_broadcast = self.W_u(e_global).unsqueeze(2)  # [B, T, 1, D]
        r = s + (1.0 - beta) * msg + beta * global_broadcast  # [B, T, N, D]

        # Reshape for LayerNorm, then restore
        r_flat = r.reshape(B * T * N, D)
        r_normed = self.layer_norm(r_flat).reshape(B, T, N, D)

        # Zero out padded positions after LN
        r_normed = r_normed * mask                          # [B, T, N, D]

        # ==============================================================
        # 8. Attention pooling [B, T, D]
        # ==============================================================
        # score_n = pool_query(tanh(pool_proj(r_n)))
        # a_n = softmax(score) over valid persons
        # g_t = Σ_n a_n · r_n

        pool_hidden = torch.tanh(self.pool_proj(r_normed))  # [B, T, N, D]
        pool_hidden = self.dropout(pool_hidden)
        pool_scores = self.pool_query(pool_hidden).squeeze(-1)  # [B, T, N]

        # Mask: set padded positions to large negative value
        neg_inf = torch.finfo(pool_scores.dtype).min / 2
        pool_scores = pool_scores.masked_fill(~person_mask, neg_inf)  # [B, T, N]

        # Handle frames with zero valid persons
        any_valid = person_mask.any(dim=-1, keepdim=True)    # [B, T, 1]
        attn_weights = F.softmax(pool_scores, dim=-1)        # [B, T, N]
        # Zero attention where no persons are valid
        attn_weights = attn_weights * any_valid.float()      # [B, T, N]

        # Weighted sum: [B, T, N] · [B, T, N, D] → [B, T, D]
        group_states = torch.einsum(
            "btn,btnd->btd", attn_weights, r_normed
        )                                                     # [B, T, D]

        return group_states
    

    # def forward(
    #     self,
    #     person_features: torch.Tensor,
    #     person_mask: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Encode per-person features into frame-level group descriptors.

    #     Args:
    #         person_features: ``[B, T, N_max, D]`` per-person features
    #             (from :class:`FeatureProjector`).
    #         person_mask: ``[B, T, N_max]`` bool mask — ``True`` for
    #             real persons, ``False`` for padding.

    #     Returns:
    #         ``[B, T, D]`` frame-level group descriptors.
    #     """
    #     B, T, N, D = person_features.shape
    #     M = self.num_prototypes
    #     assert D == self.input_dim, (
    #         f"Feature dim {D} doesn't match input_dim {self.input_dim}"
    #     )

    #     # Expand mask for broadcasting: [B, T, N, 1]
    #     mask = person_mask.unsqueeze(-1).float()           # [B, T, N, 1]

    #     s = person_features                                 # [B, T, N, D]

    #     # ==============================================================
    #     # 1. Sigmoid incidence weights H_local [B, T, N, M]
    #     # ==============================================================
    #     # H(n, m) = σ(a_m^T · W_h(s_n))
    #     # where a_m = hyperedge_prototypes[m]

    #     h_proj = self.W_h(s)                                # [B, T, N, D]
    #     # Dot product with each prototype: [B, T, N, D] @ [M, D]^T → [B, T, N, M]
    #     incidence_logits = torch.einsum(
    #         "btnd,md->btnm", h_proj, self.hyperedge_prototypes
    #     )                                                    # [B, T, N, M]
    #     H_local = torch.sigmoid(incidence_logits)           # [B, T, N, M]

    #     # Zero out padded persons
    #     H_local = H_local * mask                            # [B, T, N, M]

    #     # ==============================================================
    #     # 2. Local hyperedge features [B, T, M, D]
    #     # ==============================================================
    #     # e_m = Σ_n H(n,m) · W_v(s_n) / (Σ_n H(n,m) + eps)

    #     v_proj = self.W_v(s)                                # [B, T, N, D]
    #     v_proj = v_proj * mask                              # zero padded

    #     # Weighted sum: [B, T, M, N] @ [B, T, N, D] → [B, T, M, D]
    #     H_t = H_local.permute(0, 1, 3, 2)                  # [B, T, M, N]
    #     weighted_sum = torch.matmul(H_t, v_proj)            # [B, T, M, D]

    #     # Denominator: sum of incidence per hyperedge
    #     H_sum = H_t.sum(dim=-1, keepdim=True)               # [B, T, M, 1]
    #     e_local = weighted_sum / (H_sum + self.eps)         # [B, T, M, D]

    #     # ==============================================================
    #     # 3. Row-normalise incidence [B, T, N, M]
    #     # ==============================================================
    #     # H_bar(n, m) = H(n, m) / (Σ_m' H(n, m') + eps)

    #     H_row_sum = H_local.sum(dim=-1, keepdim=True)       # [B, T, N, 1]
    #     H_bar = H_local / (H_row_sum + self.eps)            # [B, T, N, M]

    #     # ==============================================================
    #     # 4. Local messages [B, T, N, D]
    #     # ==============================================================
    #     # msg_n = Σ_m H_bar(n, m) · W_e(e_m)

    #     e_proj = self.W_e(e_local)                          # [B, T, M, D]
    #     # [B, T, N, M] @ [B, T, M, D] → [B, T, N, D]
    #     msg = torch.matmul(H_bar, e_proj)                   # [B, T, N, D]
    #     msg = msg * mask                                    # zero padded

    #     # ==============================================================
    #     # 5. Global context [B, T, D]
    #     # ==============================================================
    #     # e_global = masked_mean(W_g(s_n))

    #     g_proj = self.W_g(s) * mask                         # [B, T, N, D]
    #     valid_count = mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, T, 1, 1]
    #     # Sum over persons, divide by count
    #     e_global = g_proj.sum(dim=2) / valid_count.squeeze(-1)  # [B, T, D]

    #     # ==============================================================
    #     # 6. Person-specific gate β [B, T, N, 1]
    #     # ==============================================================
    #     beta = torch.sigmoid(self.gate_linear(s))           # [B, T, N, 1]

    #     # ==============================================================
    #     # 7. Residual update [B, T, N, D]
    #     # ==============================================================
    #     # r_n = LN(s_n + (1 - β_n) · msg_n + β_n · W_u(e_global))

    #     global_broadcast = self.W_u(e_global).unsqueeze(2)  # [B, T, 1, D]
    #     r = s + (1.0 - beta) * msg + beta * global_broadcast  # [B, T, N, D]

    #     # Reshape for LayerNorm, then restore
    #     r_flat = r.reshape(B * T * N, D)
    #     r_normed = self.layer_norm(r_flat).reshape(B, T, N, D)

    #     # Zero out padded positions after LN
    #     r_normed = r_normed * mask                          # [B, T, N, D]

    #     # ==============================================================
    #     # 8. Attention pooling [B, T, D]
    #     # ==============================================================
    #     # score_n = pool_query(tanh(pool_proj(r_n)))
    #     # a_n = softmax(score) over valid persons
    #     # g_t = Σ_n a_n · r_n

    #     pool_hidden = torch.tanh(self.pool_proj(r_normed))  # [B, T, N, D]
    #     pool_scores = self.pool_query(pool_hidden).squeeze(-1)  # [B, T, N]

    #     # Mask: set padded positions to large negative value
    #     neg_inf = torch.finfo(pool_scores.dtype).min / 2
    #     pool_scores = pool_scores.masked_fill(~person_mask, neg_inf)  # [B, T, N]

    #     # Handle frames with zero valid persons
    #     any_valid = person_mask.any(dim=-1, keepdim=True)    # [B, T, 1]
    #     attn_weights = F.softmax(pool_scores, dim=-1)        # [B, T, N]
    #     # Zero attention where no persons are valid
    #     attn_weights = attn_weights * any_valid.float()      # [B, T, N]

    #     # Weighted sum: [B, T, N] · [B, T, N, D] → [B, T, D]
    #     group_states = torch.einsum(
    #         "btn,btnd->btd", attn_weights, r_normed
    #     )                                                     # [B, T, D]

    #     return group_states