#!/usr/bin/env python3
"""Coverage / set-size sweep over the target miscoverage alpha.

Sweeps ``alpha in {0.01, 0.05, 0.10, 0.15, 0.20}`` for marginal and
Mondrian split-conformal regimes and records the empirical coverage
and the average prediction-set size.

Usage
-----
    python scripts/run_coverage_sweep.py --config configs/iis3d.yaml \
        --out outputs/coverage.csv
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import click

from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed

LOG = get_logger(__name__)

ALPHAS: list[float] = [0.01, 0.05, 0.10, 0.15, 0.20]
MODES: list[str] = ["marginal", "mondrian"]


def _run_one(cfg: dict, alpha: float, mode: str) -> dict[str, float]:
    """Calibrate + evaluate a single ``(alpha, mode)`` combination."""
    from ssl_graph_anomaly.training.calibrate import calibrate_conformal
    from ssl_graph_anomaly.evaluation.metrics import evaluate_model

    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("conformal", {})
    run_cfg["conformal"]["enabled"] = True
    run_cfg["conformal"]["alpha"] = float(alpha)
    run_cfg["conformal"]["mode"] = mode
    run_cfg["conformal"].setdefault("aci", {})["enabled"] = False

    try:
        calibrate_conformal(run_cfg)
        metrics = evaluate_model(run_cfg)
    except Exception as exc:  # pragma: no cover
        LOG.error("Sweep alpha=%.2f mode=%s failed: %s", alpha, mode, exc)
        metrics = {}
    return metrics


@click.command()
@click.option("--config", "config_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="YAML configuration file.")
@click.option("--out", "out_path",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output CSV path.")
@click.option("--alphas", multiple=True, type=float, default=None,
              help="Override default alpha grid.")
def main(config_path: Path, out_path: Path,
         alphas: tuple[float, ...]) -> None:
    """Sweep ``alpha`` for marginal and Mondrian split-conformal."""
    cfg = load_config(str(config_path))
    set_global_seed(int(cfg.get("experiment", {}).get("seed", 17)))

    grid = list(alphas) if alphas else ALPHAS
    LOG.info("Coverage sweep: alphas=%s modes=%s", grid, MODES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "alpha", "coverage", "set_size",
                         "macro_f1"])
        for mode in MODES:
            for alpha in grid:
                LOG.info("Running mode=%s alpha=%.3f", mode, alpha)
                metrics = _run_one(cfg, alpha, mode)
                writer.writerow([
                    mode, alpha,
                    metrics.get("coverage", float("nan")),
                    metrics.get("set_size", float("nan")),
                    metrics.get("macro_f1", float("nan")),
                ])
                fh.flush()
    LOG.info("Coverage sweep -> %s", out_path)


if __name__ == "__main__":
    main()
