"""Composite SSL + supervised losses for SSL-GraphAnomaly."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ssl_graph_anomaly.losses.contrastive import InfoNCELoss
from ssl_graph_anomaly.losses.distillation import DistillationLoss
from ssl_graph_anomaly.losses.energy_margin import EnergyMarginLoss
from ssl_graph_anomaly.losses.reconstruction import ReconstructionLoss

__all__ = [
    "DistillationLoss",
    "EnergyMarginLoss",
    "InfoNCELoss",
    "ReconstructionLoss",
    "compute_total_loss",
]

_DEFAULT_WEIGHTS = {
    "reconstruction_weight": 1.0,
    "contrastive_weight": 1.0,
    "energy_margin_weight": 0.5,
    "distillation_weight": 0.25,
}


def _zero_like(ref: Tensor | None) -> Tensor:
    if ref is not None:
        return torch.zeros((), dtype=ref.dtype, device=ref.device)
    return torch.zeros(())


def compute_total_loss(
    out: dict[str, Any],
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Tensor]:
    loss_cfg: dict[str, Any] = cfg.get("loss", {}) if cfg else {}
    w_rec = float(loss_cfg.get("reconstruction_weight", _DEFAULT_WEIGHTS["reconstruction_weight"]))
    w_contr = float(loss_cfg.get("contrastive_weight", _DEFAULT_WEIGHTS["contrastive_weight"]))
    w_energy = float(loss_cfg.get("energy_margin_weight", _DEFAULT_WEIGHTS["energy_margin_weight"]))
    w_cls = float(loss_cfg.get("distillation_weight", _DEFAULT_WEIGHTS["distillation_weight"]))

    z = out.get("z")
    z_recon = out.get("z_recon")
    energy = out.get("energy")
    logits = out.get("logits")
    z_positive = batch.get("z_positive")
    energy_corrupted = batch.get("energy_corrupted")
    teacher_probs = batch.get("teacher_probs")

    ref = z if isinstance(z, Tensor) else logits if isinstance(logits, Tensor) else energy
    loss_rec = _zero_like(ref)
    loss_contr = _zero_like(ref)
    loss_energy = _zero_like(ref)
    loss_cls = _zero_like(ref)

    if isinstance(z, Tensor) and isinstance(z_recon, Tensor):
        loss_rec = ReconstructionLoss()(z, z_recon)
    if isinstance(z, Tensor) and isinstance(z_positive, Tensor):
        temperature = float(loss_cfg.get("contrastive_temperature", 0.5))
        loss_contr = InfoNCELoss(temperature=temperature)(z, z_positive)
    if isinstance(energy, Tensor) and isinstance(energy_corrupted, Tensor):
        margin = float(loss_cfg.get("energy_margin", 1.0))
        loss_energy = EnergyMarginLoss(margin=margin)(energy, energy_corrupted)
    if isinstance(logits, Tensor) and isinstance(teacher_probs, Tensor):
        T = float(loss_cfg.get("distillation_temperature", 2.0))
        loss_cls = DistillationLoss(temperature=T)(logits, teacher_probs)

    total = w_rec * loss_rec + w_contr * loss_contr + w_energy * loss_energy + w_cls * loss_cls

    return {
        "loss": total,
        "loss_rec": loss_rec,
        "loss_contr": loss_contr,
        "loss_energy": loss_energy,
        "loss_cls": loss_cls,
    }
