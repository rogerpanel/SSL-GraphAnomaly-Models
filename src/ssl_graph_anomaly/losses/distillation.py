"""Hinton soft-target distillation loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: Tensor,
        teacher_probs_or_logits: Tensor,
        teacher_is_logits: bool = False,
    ) -> Tensor:
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        if teacher_is_logits:
            teacher_probs = F.softmax(teacher_probs_or_logits / T, dim=-1)
        else:
            teacher_probs = teacher_probs_or_logits
        # KL(teacher || student) with T^2 scaling as in Hinton et al.
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return kl * (T * T)
