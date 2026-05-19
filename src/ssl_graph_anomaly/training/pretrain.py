"""Stage 1: self-supervised pretraining on benign-only flows."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import click
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_graph_anomaly.data import (
    build_dataset,
    feature_dropout,
    gaussian_jitter,
    random_replace,
)
from ssl_graph_anomaly.losses import compute_total_loss
from ssl_graph_anomaly.models import SSLGraphAnomaly
from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed


def _cosine_warmup(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


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
    return DataLoader(ds, batch_size=bs, shuffle=(split.startswith("train")), num_workers=nw)


def pretrain_ssl(cfg: dict[str, Any]) -> Path:
    logger = get_logger("ssl.pretrain")
    set_global_seed(int(cfg["experiment"]["seed"]))

    device = torch.device(cfg["experiment"].get("device", "cpu"))
    train_loader = _make_loader(cfg, "train_benign")
    val_loader = _make_loader(cfg, "val")

    model = SSLGraphAnomaly(cfg).to(device)

    ocfg = cfg["optim"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(ocfg["learning_rate"]),
        betas=(float(ocfg.get("beta1", 0.9)), float(ocfg.get("beta2", 0.999))),
        weight_decay=float(ocfg.get("weight_decay", 1e-5)),
    )

    epochs = int(cfg["train"]["ssl_epochs"])
    total_steps = max(1, epochs * max(1, len(train_loader)))
    warmup = int(ocfg.get("warmup_steps", 1000))
    scheduler = _cosine_warmup(optimizer, warmup, total_steps)

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "ssl_best.pt"
    grad_clip = float(ocfg.get("gradient_clip", 1.0))

    loss_cfg = cfg.get("loss", {})
    p_drop = float(loss_cfg.get("feature_dropout", 0.10))
    sigma = float(loss_cfg.get("gaussian_jitter_sigma", 0.05))
    p_corr = float(loss_cfg.get("mask_rate", 0.15))

    best_val = math.inf
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ssl epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            x_e = _extract_x_e(batch).to(device).float()
            x_pos = gaussian_jitter(feature_dropout(x_e, p=p_drop), sigma=sigma)
            x_corr = random_replace(x_e, p=p_corr)

            out = model.forward_stream(x_e)
            with torch.no_grad():
                out_pos = model.forward_stream(x_pos)
            out_corr = model.forward_stream(x_corr)

            batch_d = {
                "z_positive": out_pos["z"],
                "energy_corrupted": out_corr["energy"],
            }
            loss_d = compute_total_loss(out, batch_d, cfg)
            loss = loss_d["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        val_loss = _validate(model, val_loader, cfg, device, p_drop, sigma, p_corr)
        logger.info("epoch=%d val_loss=%.5f", epoch + 1, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            model.save_checkpoint(str(best_path))
            logger.info("checkpoint saved -> %s", best_path)

    if not best_path.exists():
        model.save_checkpoint(str(best_path))
    return best_path


@torch.no_grad()
def _validate(
    model: SSLGraphAnomaly,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    p_drop: float,
    sigma: float,
    p_corr: float,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x_e = _extract_x_e(batch).to(device).float()
        x_pos = gaussian_jitter(feature_dropout(x_e, p=p_drop), sigma=sigma)
        x_corr = random_replace(x_e, p=p_corr)
        out = model.forward_stream(x_e)
        out_pos = model.forward_stream(x_pos)
        out_corr = model.forward_stream(x_corr)
        batch_d = {
            "z_positive": out_pos["z"],
            "energy_corrupted": out_corr["energy"],
        }
        loss_d = compute_total_loss(out, batch_d, cfg)
        total += float(loss_d["loss"].detach().cpu())
        count += 1
    return total / max(1, count)


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--stage", type=click.Choice(["ssl", "distill"]), default="ssl")
def cli(config_path: str, stage: str) -> None:
    cfg = load_config(config_path)
    Path(cfg["experiment"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["train"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    if stage == "ssl":
        path = pretrain_ssl(cfg)
    else:
        from ssl_graph_anomaly.training.distill import distill_head

        path = distill_head(cfg)
    click.echo(str(path))


if __name__ == "__main__":
    cli()
