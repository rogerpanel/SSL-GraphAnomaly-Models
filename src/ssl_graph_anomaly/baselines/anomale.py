"""Anomal-E: Deep Graph Infomax adapted to edge features (Caville et al., 2022)."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_x(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["x_e"]
    if isinstance(batch, (tuple, list)):
        return batch[2] if len(batch) >= 3 else batch[0]
    return batch


class _EdgeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class AnomalEBaseline:
    """Contrastive DGI-style edge anomaly detector with a global summary."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["experiment"].get("device", "cpu"))
        in_dim = int(cfg["data"]["num_features"])
        hidden = int(cfg["model"]["encoder"]["hidden_dim"])
        self.encoder = _EdgeEncoder(in_dim, hidden).to(self.device)
        self.bilinear = nn.Bilinear(hidden, hidden, 1).to(self.device)
        self.threshold = float(cfg.get("baselines", {}).get("anomale_threshold", 0.5))
        self.epochs = int(cfg["train"].get("baseline_epochs", 5))
        self.lr = float(cfg["optim"]["learning_rate"])

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        h_pos = self.encoder(x)
        perm = torch.randperm(x.size(0), device=x.device)
        h_neg = self.encoder(x[perm])
        summary = torch.sigmoid(h_pos.mean(dim=0, keepdim=True))
        s_pos = self.bilinear(h_pos, summary.expand_as(h_pos)).squeeze(-1)
        s_neg = self.bilinear(h_neg, summary.expand_as(h_neg)).squeeze(-1)
        labels_pos = torch.ones_like(s_pos)
        labels_neg = torch.zeros_like(s_neg)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([s_pos, s_neg], dim=0),
            torch.cat([labels_pos, labels_neg], dim=0),
        )
        return loss

    def fit(self, loader: Iterable[Any]) -> "AnomalEBaseline":
        opt = Adam(
            list(self.encoder.parameters()) + list(self.bilinear.parameters()),
            lr=self.lr,
        )
        for _ in range(self.epochs):
            self.encoder.train()
            self.bilinear.train()
            pbar = tqdm(loader, desc="anomale-train", leave=False)
            for batch in pbar:
                x = _extract_x(batch).to(self.device).float()
                if x.size(0) < 2:
                    continue
                loss = self._step(x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.detach().cpu()))
        return self

    @torch.no_grad()
    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        self.encoder.eval()
        self.bilinear.eval()
        x_t = torch.as_tensor(_to_numpy(x), dtype=torch.float32, device=self.device)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        h = self.encoder(x_t)
        perm = torch.randperm(x_t.size(0), device=x_t.device)
        corrupted_summary = torch.sigmoid(self.encoder(x_t[perm]).mean(dim=0, keepdim=True))
        logits = self.bilinear(h, corrupted_summary.expand_as(h)).squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        preds = (scores > self.threshold).astype(np.int64)
        return preds, scores
