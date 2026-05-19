"""Tests for the four loss terms and their weighted aggregator."""

from __future__ import annotations

import pytest
import torch


def test_individual_losses_are_scalar(tiny_cfg):
    pytest.importorskip("ssl_graph_anomaly.losses")
    from ssl_graph_anomaly.losses import (
        DistillationLoss,
        EnergyMarginLoss,
        InfoNCELoss,
        ReconstructionLoss,
    )

    batch, dim = 16, 8
    num_classes = tiny_cfg["model"]["classification_head"]["num_classes"]

    z1 = torch.randn(batch, dim, requires_grad=True)
    z2 = torch.randn(batch, dim, requires_grad=True)
    z = torch.randn(batch, dim, requires_grad=True)
    z_recon = torch.randn(batch, dim, requires_grad=True)
    energy_clean = torch.randn(batch).abs()
    energy_corr = energy_clean + 0.5
    student_logits = torch.randn(batch, num_classes, requires_grad=True)
    teacher_probs = torch.softmax(torch.randn(batch, num_classes), dim=-1)

    infonce = InfoNCELoss(temperature=tiny_cfg["loss"]["infonce_temperature"])
    recon = ReconstructionLoss()
    margin = EnergyMarginLoss(margin=tiny_cfg["loss"]["margin"])
    distill = DistillationLoss(
        temperature=tiny_cfg["loss"]["distillation_temperature"],
    )

    for value in (
        infonce(z1, z2),
        recon(z, z_recon),
        margin(energy_clean, energy_corr),
        distill(student_logits, teacher_probs),
    ):
        assert isinstance(value, torch.Tensor)
        assert value.dim() == 0
        assert torch.isfinite(value)


def test_compute_total_loss_returns_loss_key(tiny_cfg):
    pytest.importorskip("ssl_graph_anomaly.losses")
    from ssl_graph_anomaly.losses import compute_total_loss

    batch, dim = 16, 8
    num_classes = tiny_cfg["model"]["classification_head"]["num_classes"]

    out = {
        "z": torch.randn(batch, dim, requires_grad=True),
        "z_recon": torch.randn(batch, dim, requires_grad=True),
        "energy": torch.randn(batch).abs(),
        "logits": torch.randn(batch, num_classes, requires_grad=True),
    }
    batch_dict = {
        "z_positive": torch.randn(batch, dim),
        "energy_corrupted": out["energy"] + 0.5,
        "teacher_probs": torch.softmax(torch.randn(batch, num_classes), dim=-1),
    }

    losses = compute_total_loss(out, batch_dict, tiny_cfg)

    assert isinstance(losses, dict)
    assert "loss" in losses
    total = losses["loss"]
    assert isinstance(total, torch.Tensor)
    assert total.dim() == 0
    assert torch.isfinite(total)
