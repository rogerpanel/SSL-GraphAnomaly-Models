"""Adversarial robustness sweep (FGSM / PGD-20) on edge feature vectors."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_graph_anomaly.data import build_dataset
from ssl_graph_anomaly.evaluation.metrics import macro_f1
from ssl_graph_anomaly.models import SSLGraphAnomaly
from ssl_graph_anomaly.utils import get_logger

try:  # optional dependency
    import torchattacks  # type: ignore

    _HAS_TORCHATTACKS = True
except Exception:  # pragma: no cover - optional
    torchattacks = None  # type: ignore
    _HAS_TORCHATTACKS = False


class _StreamWrapper(nn.Module):
    """Adapter exposing pushed_logits = f(x_e) for adversarial libraries."""

    def __init__(self, model: SSLGraphAnomaly) -> None:
        super().__init__()
        self.model = model

    def forward(self, x_e: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_stream(x_e)
        return out.get("pushed_logits", out["logits"])


def _fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if eps <= 0.0:
        return x.detach()
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x.detach() + eps * grad.sign()).detach()


def _pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float = 0.005,
    steps: int = 20,
) -> torch.Tensor:
    if eps <= 0.0:
        return x.detach()
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
    return x_adv.detach()


def _extract(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["x_e"], batch["y"]
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 4:
            return batch[2], batch[3]
        return batch[0], batch[1]
    raise ValueError("Unsupported batch format")


def _attack(
    wrapper: nn.Module,
    method: str,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    pgd_alpha: float,
    pgd_steps: int,
) -> torch.Tensor:
    if eps <= 0.0:
        return x.detach()
    method = method.lower()
    if _HAS_TORCHATTACKS:
        try:
            if method == "fgsm":
                atk = torchattacks.FGSM(wrapper, eps=eps)
                return atk(x, y).detach()
            if method.startswith("pgd"):
                atk = torchattacks.PGD(
                    wrapper, eps=eps, alpha=pgd_alpha, steps=pgd_steps, random_start=True
                )
                return atk(x, y).detach()
        except Exception:
            pass
    if method == "fgsm":
        return _fgsm(wrapper, x, y, eps)
    if method.startswith("pgd"):
        return _pgd(wrapper, x, y, eps, alpha=pgd_alpha, steps=pgd_steps)
    raise ValueError(f"Unsupported attack method: {method}")


def run_adversarial(
    cfg: dict[str, Any],
    methods: Sequence[str] = ("fgsm", "pgd20"),
    epsilons: Iterable[float] = (0.0, 0.01, 0.03, 0.05, 0.10),
) -> pd.DataFrame:
    logger = get_logger("ssl.adversarial")
    device = torch.device(cfg["experiment"].get("device", "cpu"))
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SSLGraphAnomaly.load_checkpoint(str(ckpt_dir / "distilled.pt"), map_location=device)
    model = model.to(device).eval()
    wrapper = _StreamWrapper(model).to(device).eval()

    ds = build_dataset(cfg, "test")
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["experiment"].get("num_workers", 0))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)

    adv_cfg = cfg.get("evaluation", {}).get("adversarial", {}) or {}
    pgd_alpha = float(adv_cfg.get("pgd_alpha", 0.005))
    pgd_steps = int(adv_cfg.get("pgd_steps", 20))

    rows: list[dict[str, Any]] = []
    eps_list = list(epsilons)
    for method in methods:
        for eps in eps_list:
            y_all: list[np.ndarray] = []
            p_all: list[np.ndarray] = []
            for batch in tqdm(loader, desc=f"{method}@{eps}", leave=False):
                x_e, y = _extract(batch)
                x_e = x_e.to(device).float()
                y = y.to(device).long()
                x_adv = _attack(wrapper, method, x_e, y, float(eps), pgd_alpha, pgd_steps)
                with torch.no_grad():
                    logits = wrapper(x_adv)
                preds = torch.argmax(logits, dim=-1)
                y_all.append(y.detach().cpu().numpy().astype(np.int64))
                p_all.append(preds.detach().cpu().numpy().astype(np.int64))
            y_np = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
            p_np = np.concatenate(p_all, axis=0) if p_all else np.zeros((0,), dtype=np.int64)
            score = macro_f1(y_np, p_np)
            rows.append({"method": method, "epsilon": float(eps), "macro_f1": score})
            logger.info("method=%s eps=%.3f macro_f1=%.4f", method, float(eps), score)

    df = pd.DataFrame(rows, columns=["method", "epsilon", "macro_f1"])
    out_path = out_dir / "adversarial.csv"
    df.to_csv(out_path, index=False)
    return df
