#!/usr/bin/env python3
"""Run the component-ablation matrix reproducing paper Table II.

Variants (rows of Table II):
  mlp_only           : MLP only (no GNN, no SSL, no CP)
  gnn_no_ssl_no_cp   : E-GraphSAGE encoder + softmax head, no SSL, no CP
  ssl_no_cp          : Add InfoNCE + reconstruction SSL pre-training, no CP
  maha_no_cp         : Add Mahalanobis-energy head, still no CP
  marginal_cp        : Add marginal split conformal prediction
  mondrian_cp        : Add Mondrian (class-conditional) CP
  full_aci           : Mondrian + Adaptive Conformal Inference

For each variant we train and evaluate over 5 seeds and write a CSV with
columns [variant, seed, macro_f1, coverage, set_size].

Usage
-----
    python scripts/run_ablation.py --config configs/iis3d.yaml \
        --out outputs/ablation.csv
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import click

from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed

LOG = get_logger(__name__)


VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("mlp_only",         {"gnn": False, "ssl": False, "cp": False}),
    ("gnn_no_ssl_no_cp", {"gnn": True,  "ssl": False, "cp": False}),
    ("ssl_no_cp",        {"gnn": True,  "ssl": True,  "cp": False}),
    ("maha_no_cp",       {"gnn": True,  "ssl": True,  "maha": True, "cp": False}),
    ("marginal_cp",      {"gnn": True,  "ssl": True,  "maha": True, "cp": "marginal"}),
    ("mondrian_cp",      {"gnn": True,  "ssl": True,  "maha": True, "cp": "mondrian"}),
    ("full_aci",         {"gnn": True,  "ssl": True,  "maha": True, "cp": "mondrian+aci"}),
]


def _apply_variant(cfg: dict, flags: dict) -> dict:
    """Return a deep-copied cfg modified by a variant's flag set."""
    c = copy.deepcopy(cfg)
    c.setdefault("model", {}).setdefault("encoder", {})
    c["model"]["encoder"]["type"] = "egraphsage" if flags.get("gnn") else "mlp"
    c.setdefault("loss", {})
    if not flags.get("ssl"):
        c["loss"]["reconstruction_weight"] = 0.0
        c["loss"]["contrastive_weight"] = 0.0
    c.setdefault("model", {}).setdefault("energy", {})
    c["model"]["energy"]["use_mahalanobis"] = bool(flags.get("maha", False))
    c.setdefault("conformal", {})
    cp = flags.get("cp", False)
    if cp is False:
        c["conformal"]["enabled"] = False
        c["conformal"]["aci"] = {"enabled": False}
    else:
        c["conformal"]["enabled"] = True
        if cp == "mondrian+aci":
            c["conformal"]["mode"] = "mondrian"
            c["conformal"]["aci"] = {"enabled": True,
                                     "gamma": 0.005, "warmup_steps": 2000}
        else:
            c["conformal"]["mode"] = cp
            c["conformal"]["aci"] = {"enabled": False}
    return c


def _train_and_eval(cfg: dict) -> dict[str, float]:
    """Train one variant end-to-end, return metric dict for that run."""
    from ssl_graph_anomaly.training.pretrain import pretrain_ssl
    from ssl_graph_anomaly.training.distill import distill_head
    from ssl_graph_anomaly.training.calibrate import calibrate_conformal
    from ssl_graph_anomaly.evaluation.metrics import evaluate_model

    if cfg["loss"]["reconstruction_weight"] + cfg["loss"]["contrastive_weight"] > 0:
        pretrain_ssl(cfg)
    distill_head(cfg)
    if cfg.get("conformal", {}).get("enabled", False):
        calibrate_conformal(cfg)
    return evaluate_model(cfg)


@click.command()
@click.option("--config", "config_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Base YAML configuration.")
@click.option("--out", "out_path",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output CSV path.")
@click.option("--seeds", type=int, multiple=True, default=None,
              help="Optional override list of seeds; defaults to cfg seeds.")
def main(config_path: Path, out_path: Path, seeds: tuple[int, ...]) -> None:
    """Sweep the seven ablation variants over multiple seeds."""
    cfg = load_config(str(config_path))
    base_seeds = list(seeds) if seeds else list(
        cfg.get("experiment", {}).get("seeds", [17, 42, 101, 1234, 31337])
    )
    LOG.info("Ablation will run %d variants x %d seeds = %d configurations",
             len(VARIANTS), len(base_seeds), len(VARIANTS) * len(base_seeds))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variant", "seed", "macro_f1", "coverage", "set_size"])

        for variant_name, flags in VARIANTS:
            for seed in base_seeds:
                LOG.info("== Variant %s @ seed=%d ==", variant_name, seed)
                set_global_seed(seed)
                run_cfg = _apply_variant(cfg, flags)
                run_cfg.setdefault("experiment", {})
                run_cfg["experiment"]["seed"] = seed
                run_cfg["experiment"]["output_dir"] = str(
                    Path(cfg["experiment"]["output_dir"])
                    / f"ablation/{variant_name}/seed{seed}"
                )
                try:
                    metrics = _train_and_eval(run_cfg)
                except Exception as exc:  # pragma: no cover
                    LOG.error("Variant %s seed %d failed: %s",
                              variant_name, seed, exc)
                    metrics = {}
                writer.writerow([
                    variant_name, seed,
                    metrics.get("macro_f1", float("nan")),
                    metrics.get("coverage", float("nan")),
                    metrics.get("set_size", float("nan")),
                ])
                fh.flush()
    LOG.info("Ablation finished -> %s", out_path)


if __name__ == "__main__":
    main()
