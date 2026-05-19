#!/usr/bin/env python3
"""Evaluate SSL-GraphAnomaly (and optionally open baselines) on a test split.

Produces a JSON metrics report containing Macro-F1, AUROC, ECE, coverage,
set size, and latency for the trained model and each baseline.

Usage
-----
    python scripts/evaluate.py --config configs/iis3d.yaml \
        --report outputs/report.json --include_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed

LOG = get_logger(__name__)


def _evaluate_baselines(cfg: dict) -> dict:
    """Run the four open baselines + SecurityBERT if available."""
    results: dict = {}
    try:
        from ssl_graph_anomaly import baselines as B  # type: ignore
    except Exception as exc:
        LOG.warning("Baselines package not importable: %s", exc)
        return results

    methods = {
        "kitsune": getattr(B, "KitsuneBaseline", None),
        "egraphsage": getattr(B, "EGraphSAGEBaseline", None),
        "anomale": getattr(B, "AnomalEBaseline", None),
        "rtids": getattr(B, "RTIDSBaseline", None),
        "securitybert": getattr(B, "SecurityBERTBaseline", None),
    }
    for name, cls in methods.items():
        if cls is None:
            LOG.warning("Baseline %s not available; skipping.", name)
            continue
        LOG.info("Running baseline: %s", name)
        try:
            baseline = cls(cfg)
            results[name] = baseline.evaluate()
        except Exception as exc:  # pragma: no cover
            LOG.warning("Baseline %s failed: %s", name, exc)
            results[name] = {"error": str(exc)}
    return results


@click.command()
@click.option("--config", "config_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="YAML configuration file.")
@click.option("--report", "report_path",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output JSON report path.")
@click.option("--include_baselines", is_flag=True, default=False,
              help="Also evaluate the four open baselines.")
def main(config_path: Path, report_path: Path, include_baselines: bool) -> None:
    """Run the full evaluation pipeline and write a JSON metrics report."""
    cfg = load_config(str(config_path))
    seed = int(cfg.get("experiment", {}).get("seed", 17))
    set_global_seed(seed)
    LOG.info("Evaluating with config=%s", config_path)

    output_dir = Path(cfg.get("experiment", {}).get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from ssl_graph_anomaly.evaluation.metrics import evaluate_model

    LOG.info("Evaluating SSL-GraphAnomaly")
    ours = evaluate_model(cfg)
    payload: dict = {"ssl_graph_anomaly": ours}

    if include_baselines:
        LOG.info("Evaluating open baselines")
        payload["baselines"] = _evaluate_baselines(cfg)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    LOG.info("Wrote report -> %s", report_path)


if __name__ == "__main__":
    main()
