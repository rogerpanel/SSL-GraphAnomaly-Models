"""Tests for the energy-aware logit pushing head."""

from __future__ import annotations

import pytest
import torch


def test_energy_head_pushes_attack_logits_up(tiny_cfg):
    pytest.importorskip("ssl_graph_anomaly.models.energy_head")
    from ssl_graph_anomaly.models import EnergyLogitHead

    head_cfg = tiny_cfg["model"]["classification_head"]
    enc_cfg = tiny_cfg["model"]["encoder"]
    energy_cfg = tiny_cfg["model"]["energy"]
    num_classes = head_cfg["num_classes"]
    push_weight = float(head_cfg["energy_push_weight"])
    in_dim = 2 * enc_cfg["hidden_dim"] + enc_cfg["edge_feature_dim"]

    head = EnergyLogitHead(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden=tuple(head_cfg["hidden_dims"]),
        mahalanobis_lambda=energy_cfg["mahalanobis_lambda"],
        energy_push_weight=push_weight,
        energy_mlp_hidden=tuple(energy_cfg["mlp_hidden"]),
    )
    head.eval()

    batch = 32
    z = torch.zeros(batch, in_dim)
    z_recon = torch.zeros(batch, in_dim)
    s_rec = torch.zeros(batch)
    s_maha = torch.full((batch,), 50.0)

    with torch.no_grad():
        out = head(z, z_recon, s_rec, s_maha)

    pushed = out["pushed_logits"]
    base = out["logits"]
    assert pushed.shape == (batch, num_classes)
    assert torch.isfinite(out["energy"]).all()
    assert (out["energy"] > 0).all()

    assert (pushed[:, 0] < base[:, 0]).all()
    if num_classes > 1:
        assert (pushed[:, 1:] > base[:, 1:]).all()
