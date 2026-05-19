"""Tests for the Mahalanobis-energy anomaly score."""

from __future__ import annotations

import pytest
import torch


def test_mahalanobis_energy_separates_distributions():
    pytest.importorskip("ssl_graph_anomaly.models.mahalanobis")
    from ssl_graph_anomaly.models import MahalanobisEnergy

    torch.manual_seed(0)
    dim = 8
    n_fit = 2000
    n_eval = 1000

    fit_samples = torch.randn(n_fit, dim)
    energy = MahalanobisEnergy(dim=dim, eps=1.0e-6)
    energy.fit(fit_samples)

    benign = torch.randn(n_eval, dim)
    attack = torch.randn(n_eval, dim) + 3.0

    e_benign = energy(benign)
    e_attack = energy(attack)

    assert e_benign.shape == (n_eval,)
    assert e_attack.shape == (n_eval,)
    assert torch.isfinite(e_benign).all()
    assert torch.isfinite(e_attack).all()

    assert float(e_benign.mean()) < 5.0, (
        f"benign mean energy too high: {float(e_benign.mean()):.3f}"
    )
    assert float(e_attack.mean()) > 8.0, (
        f"attack mean energy too low: {float(e_attack.mean()):.3f}"
    )
    assert float(e_attack.mean()) > 3.0 * float(e_benign.mean())
