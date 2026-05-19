"""Attention-gated streaming edge encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionGatedEncoder(nn.Module):
    def __init__(
        self,
        in_edge_dim: int = 83,
        hidden_dim: int = 128,
        use_layer_norm: bool = True,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout

        self.w_e = nn.Linear(in_edge_dim, in_edge_dim)
        self.w_u = nn.Linear(in_edge_dim, in_edge_dim)
        self.w_v = nn.Linear(in_edge_dim, in_edge_dim)
        self.w_a = nn.Linear(3 * in_edge_dim, 1)
        self.w_c = nn.Linear(3 * in_edge_dim, in_edge_dim)
        self.ln = nn.LayerNorm(in_edge_dim) if use_layer_norm else nn.Identity()
        self.proj = nn.Linear(in_edge_dim, 2 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def out_dim(self) -> int:
        return 2 * self.hidden_dim + self.in_edge_dim

    def forward(self, x_e: Tensor) -> Tensor:
        e_i = self.w_e(x_e)
        u_i = self.w_u(x_e)
        v_i = self.w_v(x_e)
        cat = torch.cat([u_i, v_i, e_i], dim=1)
        a_i = torch.sigmoid(self.w_a(cat))
        h_i = a_i * F.gelu(self.w_c(cat)) + x_e
        h_i = self.ln(h_i)
        h_i = F.dropout(h_i, p=self.dropout, training=self.training)
        h_proj = self.proj(h_i)
        z_edge = torch.cat([h_proj, x_e], dim=1)
        return z_edge
