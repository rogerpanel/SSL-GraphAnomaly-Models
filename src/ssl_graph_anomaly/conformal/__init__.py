"""Conformal prediction layers for SSL-GraphAnomaly."""

from __future__ import annotations

from typing import Any

import torch

from ssl_graph_anomaly.conformal.aci import AdaptiveConformalInference
from ssl_graph_anomaly.conformal.drift_monitor import DriftMonitor
from ssl_graph_anomaly.conformal.mondrian import MondrianConformal
from ssl_graph_anomaly.conformal.split_conformal import SplitConformal

__all__ = [
    "AdaptiveConformalInference",
    "DriftMonitor",
    "MondrianConformal",
    "SplitConformal",
    "conformal_predict",
]


def conformal_predict(
    energy: float,
    pushed_logits: torch.Tensor,
    mondrian: MondrianConformal | None,
    marginal: SplitConformal | None,
    benign_class: int = 0,
) -> dict[str, Any]:
    predicted_class = int(torch.argmax(pushed_logits).item())

    if mondrian is not None:
        prediction_set = mondrian.predict_set(float(energy), pred_class=predicted_class)
        coverage_certified = True
    elif marginal is not None:
        accepted = marginal.predict_set(float(energy))
        # Marginal certifies only benign: accepted -> {benign}, else {predicted}.
        prediction_set = [benign_class] if accepted else [predicted_class]
        coverage_certified = True
    else:
        prediction_set = [predicted_class]
        coverage_certified = False

    abstained = (len(prediction_set) > 1) and (predicted_class in prediction_set)

    return {
        "predicted_class": predicted_class,
        "prediction_set": list(prediction_set),
        "abstained": bool(abstained),
        "coverage_certified": bool(coverage_certified),
    }
