"""Smoke tests for the two encoder variants."""

from __future__ import annotations

import pytest
import torch


def test_egraphsage_output_shape(tiny_cfg, tiny_dataset, edge_index):
    pytest.importorskip("ssl_graph_anomaly.models.egraphsage")
    from ssl_graph_anomaly.models import EGraphSAGEEncoder

    enc_cfg = tiny_cfg["model"]["encoder"]
    edge_dim = enc_cfg["edge_feature_dim"]
    hidden = enc_cfg["hidden_dim"]

    encoder = EGraphSAGEEncoder(
        in_node_dim=edge_dim,
        in_edge_dim=edge_dim,
        hidden_dim=hidden,
        num_layers=enc_cfg["num_layers"],
        dropout=enc_cfg["dropout"],
        aggregator=enc_cfg["aggregator"],
    )
    num_nodes = int(tiny_dataset["num_hosts"].item())
    x_v = torch.randn(num_nodes, edge_dim, requires_grad=True)
    x_e = tiny_dataset["features"]
    z_edge, h_node = encoder(x_v, edge_index, x_e)

    assert z_edge.dim() == 2
    assert z_edge.size(0) == x_e.size(0)
    assert z_edge.size(1) == 2 * hidden + edge_dim
    assert h_node.shape == (num_nodes, hidden)
    assert z_edge.requires_grad


def test_attention_gated_output_shape(tiny_cfg, tiny_dataset):
    pytest.importorskip("ssl_graph_anomaly.models.attention_gated")
    from ssl_graph_anomaly.models import AttentionGatedEncoder

    enc_cfg = tiny_cfg["model"]["encoder"]
    edge_dim = enc_cfg["edge_feature_dim"]
    hidden = enc_cfg["hidden_dim"]

    encoder = AttentionGatedEncoder(
        in_edge_dim=edge_dim,
        hidden_dim=hidden,
        use_layer_norm=tiny_cfg["model"]["streaming_variant"]["use_layer_norm"],
        dropout=enc_cfg["dropout"],
    )
    feats = tiny_dataset["features"].clone().requires_grad_(True)
    z = encoder(feats)

    assert z.dim() == 2
    assert z.size(0) == feats.size(0)
    assert z.size(1) == 2 * hidden + edge_dim
    assert z.requires_grad
