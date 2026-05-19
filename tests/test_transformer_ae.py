"""Discrepancy-aware Transformer autoencoder shape / loss tests."""

from __future__ import annotations

import pytest
import torch


def test_transformer_ae_reconstruction_shapes(tiny_cfg, tiny_dataset):
    pytest.importorskip("ssl_graph_anomaly.models.transformer_ae")
    from ssl_graph_anomaly.models import DiscrepancyTransformerAE

    tae_cfg = tiny_cfg["model"]["transformer_ae"]
    hidden = tiny_cfg["model"]["encoder"]["hidden_dim"]
    embed_dim = 2 * hidden + tiny_cfg["model"]["encoder"]["edge_feature_dim"]

    model = DiscrepancyTransformerAE(
        embed_dim=embed_dim,
        num_tokens=tae_cfg["num_tokens"],
        num_layers=tae_cfg["num_layers"],
        num_heads=tae_cfg["num_heads"],
        ff_multiplier=tae_cfg["ff_width_multiplier"],
        dropout=tae_cfg["dropout"],
    )
    batch = tiny_dataset["features"].size(0)
    z = torch.randn(batch, embed_dim, requires_grad=True)

    z_recon, s_rec = model(z)
    assert z_recon.shape == z.shape
    assert s_rec.shape == (batch,)
    assert torch.isfinite(z_recon).all()
    assert torch.isfinite(s_rec).all()
