"""Energy margin loss (paper Section III-D)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EnergyMarginLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, energy_clean: Tensor, energy_corrupted: Tensor) -> Tensor:
        hinge = torch.clamp(self.margin - energy_corrupted + energy_clean, min=0.0)
        return hinge.mean()
