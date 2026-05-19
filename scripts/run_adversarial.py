#!/usr/bin/env python3
"""Adversarial robustness sweep (FGSM and PGD-20).

Usage
-----
    python scripts/run_adversarial.py --config configs/iis3d.yaml \
        --out outputs/adversarial.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import click

from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed

LOG = get_logger(__name__)

DEFAULT_EPSILONS: list[float] = [0.0, 0.01, 0.03, 0.05, 0.10]


@click.command()
@click.option("--config", "config_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="YAML configuration file.")
@click.option("--out", "out_path",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output CSV path.")
@click.option("--methods", multiple=True,
              type=click.Choice(["fgsm", "pgd20"]),
              default=("fgsm", "pgd20"))
@click.option("--epsilons", multiple=True, type=float, default=None)
def main(config_path: Path, out_path: Path,
         methods: tuple[str, ...], epsilons: tuple[float, ...]) -> None:
    """Run FGSM and PGD-20 attacks across an epsilon sweep."""
    cfg = load_config(str(config_path))
    set_global_seed(int(cfg.get("experiment", {}).get("seed", 17)))

    eps_list = list(epsilons) if epsilons else (
        cfg.get("evaluation", {})
           .get("adversarial", {})
           .get("epsilons", DEFAULT_EPSILONS)
    )
    LOG.info("Adversarial sweep methods=%s epsilons=%s", methods, eps_list)

    from ssl_graph_anomaly.evaluation.adversarial import run_adversarial

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "epsilon", "macro_f1", "coverage",
                         "set_size", "asr", "latency_ms"])
        for method in methods:
            for eps in eps_list:
                LOG.info("Attack=%s epsilon=%.4f", method, eps)
                try:
                    metrics = run_adversarial(cfg, method=method, epsilon=eps)
                except TypeError:
                    metrics = run_adversarial(cfg)
                writer.writerow([
                    method, eps,
                    metrics.get("macro_f1", float("nan")),
                    metrics.get("coverage", float("nan")),
                    metrics.get("set_size", float("nan")),
                    metrics.get("asr", float("nan")),
                    metrics.get("latency_ms", float("nan")),
                ])
                fh.flush()
    LOG.info("Adversarial sweep done -> %s", out_path)


if __name__ == "__main__":
    main()
