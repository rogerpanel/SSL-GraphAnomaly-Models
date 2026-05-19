"""Stage 3: fit Mondrian + marginal split conformal predictors on benign data."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_graph_anomaly.conformal import MondrianConformal, SplitConformal
from ssl_graph_anomaly.data import build_dataset
from ssl_graph_anomaly.models import SSLGraphAnomaly
from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed


def _extract_x_e(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["x_e"]
    if isinstance(batch, (tuple, list)):
        return batch[2] if len(batch) >= 3 else batch[0]
    return batch


def _make_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    ds = build_dataset(cfg, split)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["experiment"].get("num_workers", 0))
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)


@torch.no_grad()
def _collect_scores(
    model: SSLGraphAnomaly,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    energies: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for batch in tqdm(loader, desc="calibration", leave=False):
        x_e = _extract_x_e(batch).to(device).float()
        out = model.forward_stream(x_e)
        energies.append(out["energy"].detach().cpu().numpy().astype(np.float64))
        logits = out.get("pushed_logits", out.get("logits"))
        preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64))
    if not energies:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(energies, axis=0), np.concatenate(preds, axis=0)


def calibrate_conformal(cfg: dict[str, Any]) -> Path:
    logger = get_logger("ssl.calibrate")
    set_global_seed(int(cfg["experiment"]["seed"]))
    device = torch.device(cfg["experiment"].get("device", "cpu"))

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    distilled_path = ckpt_dir / "distilled.pt"

    model = SSLGraphAnomaly.load_checkpoint(str(distilled_path), map_location=device)
    model = model.to(device)

    loader = _make_loader(cfg, "calibration_benign")
    energies, preds = _collect_scores(model, loader, device)

    alpha = float(cfg["conformal"].get("alpha", 0.05))
    num_classes = int(cfg["data"]["num_classes"])

    mondrian = MondrianConformal(alpha=alpha, num_classes=num_classes)
    marginal = SplitConformal(alpha=alpha)

    if energies.shape[0] > 0:
        mondrian.calibrate(energies, preds)
        marginal.calibrate(energies)
    else:
        logger.warning("no calibration scores collected; conformal predictors are empty")

    out_path = ckpt_dir / "conformal.pkl"
    with out_path.open("wb") as fh:
        pickle.dump(
            {
                "mondrian": mondrian,
                "marginal": marginal,
                "calibration_scores": energies,
                "alpha": alpha,
            },
            fh,
        )
    logger.info("conformal predictors saved -> %s", out_path)
    return out_path


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
def cli(config_path: str) -> None:
    cfg = load_config(config_path)
    Path(cfg["experiment"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["train"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    path = calibrate_conformal(cfg)
    click.echo(str(path))


if __name__ == "__main__":
    cli()
