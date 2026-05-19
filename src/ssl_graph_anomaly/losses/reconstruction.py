"""Reconstruction loss (paper Eq. 4, batch averaged)."""

from __future__ import annotations

from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor, z_recon: Tensor) -> Tensor:
        dim = z.size(-1)
        per_sample = (z - z_recon).pow(2).sum(dim=-1) / dim
        return per_sample.mean()
