"""RTIDS: Transformer-based real-time IDS over per-flow feature vectors."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["x_e"], batch["y"]
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 4:
            return batch[2], batch[3]
        return batch[0], batch[1]
    raise ValueError("Unsupported batch format")


class _RTIDSModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.token_proj = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, num_features + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.token_proj(x.unsqueeze(-1))
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.size(1) + 1]
        h = self.encoder(h)
        return self.head(h[:, 0])


class RTIDSBaseline:
    """Two-layer Transformer with a CLS classification head on edge features."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["experiment"].get("device", "cpu"))
        bcfg = cfg.get("baselines", {}) or {}
        d_model = int(bcfg.get("rtids_d_model", 64))
        num_layers = int(bcfg.get("rtids_num_layers", 2))
        num_heads = int(bcfg.get("rtids_num_heads", 4))
        dropout = float(bcfg.get("rtids_dropout", 0.10))
        self.model = _RTIDSModel(
            num_features=int(cfg["data"]["num_features"]),
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=int(cfg["data"]["num_classes"]),
            dropout=dropout,
        ).to(self.device)
        self.epochs = int(cfg["train"].get("baseline_epochs", 5))
        self.lr = float(cfg["optim"]["learning_rate"])
        self.weight_decay = float(cfg["optim"].get("weight_decay", 1e-5))

    def fit(self, loader: Iterable[Any]) -> "RTIDSBaseline":
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            self.model.train()
            pbar = tqdm(loader, desc="rtids-train", leave=False)
            for batch in pbar:
                x, y = _extract(batch)
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.detach().cpu()))
        return self

    @torch.no_grad()
    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        x_t = torch.as_tensor(_to_numpy(x), dtype=torch.float32, device=self.device)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        logits = self.model(x_t)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy().astype(np.int64)
        scores = probs.max(dim=-1).values.cpu().numpy().astype(np.float32)
        return preds, scores
