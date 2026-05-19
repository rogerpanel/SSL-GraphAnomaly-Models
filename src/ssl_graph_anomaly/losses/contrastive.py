"""InfoNCE contrastive loss (paper Eq. 5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z_anchor: Tensor, z_positive: Tensor) -> Tensor:
        z_a = F.normalize(z_anchor, dim=-1)
        z_p = F.normalize(z_positive, dim=-1)
        logits = (z_a @ z_p.t()) / self.temperature
        targets = torch.arange(z_a.size(0), device=z_a.device)
        return F.cross_entropy(logits, targets)
