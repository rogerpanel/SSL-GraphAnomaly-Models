#!/usr/bin/env python3
"""Calibrate the conformal layer on a held-out benign split.

Usage
-----
    python scripts/calibrate_conformal.py --config configs/iis3d.yaml
"""

from __future__ import annotations

from pathlib import Path

import click

from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed

LOG = get_logger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML configuration file.",
)
def main(config_path: Path) -> None:
    """Calibrate the split / Mondrian / ACI conformal layer."""
    cfg = load_config(str(config_path))
    seed = int(cfg.get("experiment", {}).get("seed", 17))
    set_global_seed(seed)
    LOG.info("Calibrating conformal layer using config=%s", config_path)

    output_dir = Path(cfg.get("experiment", {}).get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from ssl_graph_anomaly.training.calibrate import calibrate_conformal

    calibrate_conformal(cfg)
    LOG.info("Calibration complete; artefacts in %s", output_dir)


if __name__ == "__main__":
    main()
