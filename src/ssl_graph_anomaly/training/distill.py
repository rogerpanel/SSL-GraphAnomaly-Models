"""Stage 2: head distillation against a label-smoothing surrogate teacher."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_graph_anomaly.data import build_dataset
from ssl_graph_anomaly.models import SSLGraphAnomaly
from ssl_graph_anomaly.utils import get_logger, load_config, set_global_seed


class LabelSmoothingTeacher:
    """Surrogate teacher emitting smoothed one-hot probabilities."""

    def __init__(self, num_classes: int = 34, true_prob: float = 0.9) -> None:
        self.num_classes = int(num_classes)
        self.true_prob = float(true_prob)
        self.off = (1.0 - self.true_prob) / max(1, self.num_classes - 1)

    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.full(
            (labels.size(0), self.num_classes),
            self.off,
            dtype=torch.float32,
            device=labels.device,
        )
        probs.scatter_(1, labels.view(-1, 1).long(), self.true_prob)
        return probs


def _extract(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["x_e"], batch["y"]
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 4:
            return batch[2], batch[3]
        return batch[0], batch[1]
    raise ValueError("Unsupported batch format")


def _make_loader(cfg: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    ds = build_dataset(cfg, split)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["experiment"].get("num_workers", 0))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw)


def distill_head(cfg: dict[str, Any]) -> Path:
    logger = get_logger("ssl.distill")
    set_global_seed(int(cfg["experiment"]["seed"]))
    device = torch.device(cfg["experiment"].get("device", "cpu"))

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ssl_path = ckpt_dir / "ssl_best.pt"

    model = SSLGraphAnomaly.load_checkpoint(str(ssl_path), map_location=device)
    model = model.to(device)

    # Freeze encoder + transformer AE
    for name, p in model.named_parameters():
        p.requires_grad = (
            name.startswith("head") or name.startswith("mahalanobis")
        ) or "energy_mlp" in name

    trainable = [p for p in model.parameters() if p.requires_grad]
    ocfg = cfg["optim"]
    optimizer = AdamW(
        trainable,
        lr=float(ocfg["learning_rate"]),
        weight_decay=float(ocfg.get("weight_decay", 1e-5)),
    )

    teacher = LabelSmoothingTeacher(num_classes=int(cfg["data"]["num_classes"]))
    distill_w = float(cfg["loss"].get("distillation_weight", 0.25))
    grad_clip = float(ocfg.get("gradient_clip", 1.0))
    epochs = int(cfg["train"]["distill_epochs"])

    train_loader = _make_loader(cfg, "train_benign", True)
    benign_loader = _make_loader(cfg, "train_benign", False)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"distill {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            x_e, y = _extract(batch)
            x_e = x_e.to(device).float()
            y = y.to(device).long()
            out = model.forward_stream(x_e)
            logits = out.get("pushed_logits", out.get("logits"))
            ce = F.cross_entropy(logits, y)
            t_probs = teacher(y)
            log_probs = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(log_probs, t_probs, reduction="batchmean")
            loss = ce + distill_w * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

    logger.info("fitting mahalanobis statistics on benign loader")
    model.fit_mahalanobis(benign_loader, device=device)

    out_path = ckpt_dir / "distilled.pt"
    model.save_checkpoint(str(out_path))
    logger.info("distilled checkpoint -> %s", out_path)
    return out_path


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
def cli(config_path: str) -> None:
    cfg = load_config(config_path)
    Path(cfg["experiment"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["train"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    path = distill_head(cfg)
    click.echo(str(path))


if __name__ == "__main__":
    cli()
