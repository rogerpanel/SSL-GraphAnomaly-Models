"""E-GraphSAGE encoder with edge-aware aggregation."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EGraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int = 83,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.10,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        if aggregator not in ("mean", "sum"):
            raise ValueError(f"Unsupported aggregator: {aggregator}")
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator

        self.layers: nn.ModuleList = nn.ModuleList()
        for k in range(num_layers):
            node_in = in_node_dim if k == 0 else hidden_dim
            self.layers.append(nn.Linear(node_in + (node_in + in_edge_dim), hidden_dim))
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

    def _aggregate(
        self, msg: Tensor, dst: Tensor, num_nodes: int
    ) -> Tensor:
        out = torch.zeros(num_nodes, msg.size(1), dtype=msg.dtype, device=msg.device)
        idx = dst.unsqueeze(1).expand(-1, msg.size(1))
        out.scatter_add_(0, idx, msg)
        if self.aggregator == "mean":
            counts = torch.zeros(num_nodes, dtype=msg.dtype, device=msg.device)
            ones = torch.ones(dst.size(0), dtype=msg.dtype, device=msg.device)
            counts.scatter_add_(0, dst, ones)
            counts = counts.clamp_min(1.0).unsqueeze(1)
            out = out / counts
        return out

    def forward(
        self, x_v: Tensor, edge_index: Tensor, x_e: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x_v.size(0)
        h = x_v
        for k, lin in enumerate(self.layers):
            msg = torch.cat([h[src], x_e], dim=1)
            agg = self._aggregate(msg, dst, num_nodes)
            h_new = lin(torch.cat([h, agg], dim=1))
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h_new = F.normalize(h_new, p=2, dim=1)
            h = h_new
        z_edge = torch.cat([h[src], h[dst], x_e], dim=1)
        return z_edge, h
