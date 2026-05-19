"""Tests for the split, Mondrian, and ACI conformal layers + drift monitor."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_split_conformal_coverage_within_band():
    pytest.importorskip("ssl_graph_anomaly.conformal")
    from ssl_graph_anomaly.conformal import SplitConformal

    rng = np.random.default_rng(0)
    cal_scores = rng.exponential(scale=1.0, size=1000)
    test_scores = rng.exponential(scale=1.0, size=5000)

    cp = SplitConformal(alpha=0.10)
    cp.calibrate(torch.as_tensor(cal_scores, dtype=torch.float32))

    threshold = cp.quantile
    coverage = float((test_scores <= threshold).mean())

    assert 0.86 <= coverage <= 0.94, (
        f"empirical coverage {coverage:.3f} not in [0.86, 0.94]"
    )


def test_mondrian_conformal_per_class_quantiles():
    pytest.importorskip("ssl_graph_anomaly.conformal")
    from ssl_graph_anomaly.conformal import MondrianConformal

    rng = np.random.default_rng(1)
    num_classes = 10
    per_class = 400
    scores = np.concatenate([
        rng.exponential(scale=1.0 + 0.2 * cls, size=per_class)
        for cls in range(num_classes)
    ]).astype(np.float32)
    labels = np.concatenate([
        np.full(per_class, cls, dtype=np.int64) for cls in range(num_classes)
    ])

    cp = MondrianConformal(alpha=0.10, num_classes=num_classes)
    cp.calibrate(scores, labels)

    quantiles = cp.quantiles
    assert len(quantiles) == num_classes
    q_arr = np.asarray([float(quantiles[c]) for c in range(num_classes)])
    rho = float(np.corrcoef(np.arange(num_classes), q_arr)[0, 1])
    assert rho > 0.5, f"per-class quantile trend too weak: rho={rho:.3f}"


def test_aci_update_path():
    pytest.importorskip("ssl_graph_anomaly.conformal")
    from ssl_graph_anomaly.conformal import AdaptiveConformalInference

    aci = AdaptiveConformalInference(alpha_star=0.10, gamma=0.01)
    starting = aci.current_alpha

    # Eq. 11: miscovered=True shrinks alpha (i.e. tightens the set toward coverage).
    for _ in range(200):
        aci.update(miscovered=True)
    shrunk = aci.current_alpha
    assert shrunk < starting

    # No miscoverage events grow alpha back toward alpha_star.
    for _ in range(500):
        aci.update(miscovered=False)
    grown = aci.current_alpha
    assert grown > shrunk


def test_drift_monitor_triggers_on_shift():
    pytest.importorskip("ssl_graph_anomaly.conformal")
    from ssl_graph_anomaly.conformal import DriftMonitor

    rng = np.random.default_rng(2)
    monitor = DriftMonitor(ks_threshold=0.10, recent_buffer=500)
    base = rng.exponential(scale=1.0, size=2000).astype(np.float32)
    monitor.set_calibration(base)

    for i, e in enumerate(rng.exponential(1.0, size=400)):
        monitor.add_observation(
            energy=float(e), is_benign_predicted=True,
            is_confirmed_benign=True, timestamp=float(i),
        )
    flag_in, _ = monitor.should_recalibrate()
    assert flag_in is False

    for i, e in enumerate(rng.exponential(scale=3.0, size=400)):
        monitor.add_observation(
            energy=float(e), is_benign_predicted=True,
            is_confirmed_benign=True, timestamp=float(400 + i),
        )
    flag_shift, info = monitor.should_recalibrate()
    assert flag_shift is True, f"drift not triggered, info={info}"
