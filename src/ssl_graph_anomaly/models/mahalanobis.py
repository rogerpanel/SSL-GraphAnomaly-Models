"""Diagonal Mahalanobis energy with Welford statistics."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MahalanobisEnergy(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma_inv", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros(1))

    @torch.no_grad()
    def fit(self, z: Tensor) -> None:
        if z.dim() != 2 or z.size(1) != self.dim:
            raise ValueError(f"Expected z with shape [N, {self.dim}], got {tuple(z.shape)}")
        n = z.size(0)
        mean = torch.zeros(self.dim, dtype=z.dtype, device=z.device)
        m2 = torch.zeros(self.dim, dtype=z.dtype, device=z.device)
        count = 0
        for i in range(n):
            count += 1
            x = z[i]
            delta = x - mean
            mean = mean + delta / count
            delta2 = x - mean
            m2 = m2 + delta * delta2
        var = m2 / max(count - 1, 1)
        std = torch.sqrt(var.clamp_min(0.0)) + self.eps
        self.mu = mean.to(self.mu.dtype).to(self.mu.device)
        self.sigma_inv = (1.0 / std).to(self.sigma_inv.dtype).to(self.sigma_inv.device)
        self.fitted = torch.ones(1, device=self.fitted.device)

    def forward(self, z: Tensor) -> Tensor:
        diff = (z - self.mu) * self.sigma_inv
        return (diff * diff).sum(dim=1) / self.dim
