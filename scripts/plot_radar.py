#!/usr/bin/env python3
"""Render the six-axis radar plot of paper Figure 3.

Axes (clockwise from the top):
  1. Macro-F1
  2. AUROC
  3. 1 - ECE        (calibration)
  4. Coverage       (closer to 1 - alpha is better)
  5. Speed          (normalised inverse latency)
  6. Compactness    (normalised inverse model size)

Usage
-----
    python scripts/plot_radar.py --results results/baselines.json \
        --out figures/fig3_radar.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import click

from ssl_graph_anomaly.utils import get_logger

LOG = get_logger(__name__)

AXES: list[str] = [
    "Macro-F1", "AUROC", "1-ECE", "Coverage", "Speed", "Compactness",
]


def _normalise(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _extract_axes(metrics: dict, latency_ref: float,
                  params_ref: float) -> list[float]:
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    auroc = float(metrics.get("auroc", 0.0))
    ece = float(metrics.get("ece", 0.0))
    coverage = float(metrics.get("coverage", 0.0))
    latency_ms = float(metrics.get("latency_ms", latency_ref or 1.0))
    params_m = float(metrics.get("params_m", params_ref or 1.0))
    speed = _normalise(latency_ref / max(latency_ms, 1e-6), 0.0, 1.0)
    compactness = _normalise(params_ref / max(params_m, 1e-6), 0.0, 1.0)
    return [macro_f1, auroc, 1.0 - ece, coverage, speed, compactness]


@click.command()
@click.option("--results", "results_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Baseline metrics JSON.")
@click.option("--out", "out_path",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output PNG path.")
@click.option("--latency_ref", type=float, default=1.0, show_default=True,
              help="Reference latency in ms used to normalise the speed axis.")
@click.option("--params_ref", type=float, default=1.0, show_default=True,
              help="Reference parameter count (M) used to normalise compactness.")
def main(results_path: Path, out_path: Path,
         latency_ref: float, params_ref: float) -> None:
    """Read a baseline-metrics JSON and render the six-axis radar plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LOG.info("Loading metrics from %s", results_path)
    with results_path.open() as fh:
        payload = json.load(fh)

    series: dict[str, dict] = {}
    if "ssl_graph_anomaly" in payload:
        series["SSL-GraphAnomaly"] = payload["ssl_graph_anomaly"]
    for name, metrics in payload.get("baselines", {}).items():
        series[name] = metrics
    # Tolerant fallback: payload may itself be the metrics map.
    if not series and isinstance(payload, dict):
        series = {k: v for k, v in payload.items() if isinstance(v, dict)}

    if not series:
        raise click.ClickException("No metric entries found in JSON.")

    angles = [n / len(AXES) * 2.0 * math.pi for n in range(len(AXES))]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(7, 7))
    ax.set_theta_offset(math.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(AXES)
    ax.set_ylim(0.0, 1.0)
    ax.set_rlabel_position(180.0 / len(AXES))

    for name, metrics in series.items():
        if not isinstance(metrics, dict):
            continue
        values = _extract_axes(metrics, latency_ref, params_ref)
        closed = values + [values[0]]
        ax.plot(angles_closed, closed, linewidth=2.0, label=name)
        ax.fill(angles_closed, closed, alpha=0.10)

    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))
    ax.set_title("Six-axis radar (paper Fig. 3)", pad=22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    LOG.info("Radar saved -> %s", out_path)


if __name__ == "__main__":
    main()
