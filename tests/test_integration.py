"""End-to-end smoke test exercising the full forward + backward path."""

from __future__ import annotations

import pytest
import torch


def test_forward_backward_smoke(tiny_cfg, tiny_dataset):
    pytest.importorskip("ssl_graph_anomaly.models.ssl_graph_anomaly")
    pytest.importorskip("ssl_graph_anomaly.losses")

    from ssl_graph_anomaly.losses import compute_total_loss
    from ssl_graph_anomaly.models import SSLGraphAnomaly

    model = SSLGraphAnomaly(tiny_cfg)
    model.train()

    x_e = tiny_dataset["features"]
    out = model.forward_stream(x_e)

    expected_keys = {"z", "z_recon", "s_rec", "s_maha",
                     "energy", "logits", "pushed_logits", "probs"}
    assert expected_keys.issubset(out.keys()), (
        f"missing keys: {expected_keys - set(out.keys())}"
    )

    out_pos = model.forward_stream(x_e + 0.05 * torch.randn_like(x_e))
    out_corr = model.forward_stream(x_e[torch.randperm(x_e.size(0))])
    batch = {
        "z_positive": out_pos["z"],
        "energy_corrupted": out_corr["energy"],
    }

    losses = compute_total_loss(out, batch, tiny_cfg)
    assert "loss" in losses
    loss = losses["loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    grad_norms = [
        float(p.grad.norm()) for p in model.parameters()
        if p.grad is not None
    ]
    assert grad_norms, "no parameter received a gradient"
    assert any(g > 0.0 for g in grad_norms)
