#!/usr/bin/env python3
"""Run the self-supervised pre-training stage and (optionally) distillation.

Usage
-----
    python scripts/train_ssl.py --config configs/iis3d.yaml --stage all
    python scripts/train_ssl.py --config configs/iis3d.yaml --stage ssl
    python scripts/train_ssl.py --config configs/iis3d.yaml --stage distill
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
    help="Path to a YAML configuration file.",
)
@click.option(
    "--stage",
    type=click.Choice(["ssl", "distill", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which stage(s) of training to execute.",
)
def main(config_path: Path, stage: str) -> None:
    """Train SSL-GraphAnomaly: pre-train and (optionally) distill."""
    cfg = load_config(str(config_path))
    seed = int(cfg.get("experiment", {}).get("seed", 17))
    set_global_seed(seed)
    LOG.info("Loaded config %s (seed=%d)", config_path, seed)

    output_dir = Path(cfg.get("experiment", {}).get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Output directory: %s", output_dir)

    stage_l = stage.lower()
    if stage_l in {"ssl", "all"}:
        LOG.info("Stage 1/2: self-supervised pre-training")
        from ssl_graph_anomaly.training.pretrain import pretrain_ssl

        pretrain_ssl(cfg)
        LOG.info("Stage 1/2 complete.")

    if stage_l in {"distill", "all"}:
        LOG.info("Stage 2/2: classification-head distillation")
        from ssl_graph_anomaly.training.distill import distill_head

        distill_head(cfg)
        LOG.info("Stage 2/2 complete.")

    LOG.info("Training pipeline finished.")


if __name__ == "__main__":
    main()
