"""Evaluation, metrics, and adversarial robustness for SSL-GraphAnomaly."""
from ssl_graph_anomaly.evaluation.adversarial import run_adversarial
from ssl_graph_anomaly.evaluation.metrics import compute_metrics, evaluate_model

__all__ = ["compute_metrics", "evaluate_model", "run_adversarial"]
