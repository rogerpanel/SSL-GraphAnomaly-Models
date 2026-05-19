"""Energy-based classification head with pushed logits."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EnergyLogitHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 34,
        hidden: Sequence[int] = (256, 128),
        mahalanobis_lambda: float = 0.10,
        energy_push_weight: float = 2.0,
        energy_mlp_hidden: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.mahalanobis_lambda = mahalanobis_lambda
        self.energy_push_weight = energy_push_weight

        self.classifier = self._build_mlp(in_dim, hidden, num_classes)
        self.energy_mlp = self._build_mlp(2 * in_dim, energy_mlp_hidden, 1)
        self.reset_parameters()

    @staticmethod
    def _build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z: Tensor,
        z_recon: Tensor,
        s_rec: Tensor,
        s_maha: Tensor,
    ) -> Dict[str, Tensor]:
        logits = self.classifier(z)
        energy_mlp_in = torch.cat([z, z_recon], dim=1)
        e_mlp = self.energy_mlp(energy_mlp_in).squeeze(-1)
        energy = e_mlp + s_rec + self.mahalanobis_lambda * s_maha

        b = energy.size(0)
        c = self.num_classes
        bias = torch.zeros(b, c, dtype=logits.dtype, device=logits.device)
        bias[:, 0] = -self.energy_push_weight * energy
        if c > 1:
            bias[:, 1:] = (self.energy_push_weight * energy).unsqueeze(1).expand(-1, c - 1)
        pushed_logits = logits + bias
        probs = F.softmax(pushed_logits, dim=1)
        return {
            "logits": logits,
            "energy": energy,
            "pushed_logits": pushed_logits,
            "probs": probs,
        }
