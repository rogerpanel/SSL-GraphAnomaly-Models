"""Supervised E-GraphSAGE baseline trained with cross-entropy on edge labels."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from ssl_graph_anomaly.models.egraphsage import EGraphSAGEEncoder


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


class _StreamlinedHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGraphSAGEBaseline:
    """Supervised E-GraphSAGE variant using a streamed edge-feature encoder."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["experiment"].get("device", "cpu"))
        in_edge_dim = int(cfg["data"]["num_features"])
        hidden = int(cfg["model"]["encoder"]["hidden_dim"])
        num_layers = int(cfg["model"]["encoder"].get("num_layers", 2))
        dropout = float(cfg["model"]["encoder"].get("dropout", 0.10))
        in_node_dim = int(cfg["model"]["encoder"].get("in_node_dim", 4))
        num_classes = int(cfg["data"]["num_classes"])

        self.encoder = EGraphSAGEEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_dim=hidden,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.in_node_dim = in_node_dim
        # Stream-only fallback: project edge feats to encoder.out_dim by concatenating
        # zero-padded node embeddings with the raw edge feature.
        self.head = _StreamlinedHead(
            in_dim=self.encoder.out_dim,
            hidden=hidden,
            num_classes=num_classes,
        ).to(self.device)

        self.epochs = int(cfg["train"].get("baseline_epochs", 5))
        self.lr = float(cfg["optim"]["learning_rate"])
        self.weight_decay = float(cfg["optim"].get("weight_decay", 1e-5))

    def _embed_edges(self, x_e: torch.Tensor) -> torch.Tensor:
        n = x_e.size(0)
        hidden = self.encoder.hidden_dim
        zero_node = torch.zeros(n * 2, self.in_node_dim, device=x_e.device, dtype=x_e.dtype)
        src = torch.arange(n, device=x_e.device, dtype=torch.long)
        dst = torch.arange(n, n * 2, device=x_e.device, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        z_edge, _ = self.encoder(zero_node, edge_index, x_e)
        return z_edge

    def fit(self, loader: Iterable[Any]) -> "EGraphSAGEBaseline":
        opt = AdamW(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        for _ in range(self.epochs):
            self.encoder.train()
            self.head.train()
            pbar = tqdm(loader, desc="egraphsage-train", leave=False)
            for batch in pbar:
                x_e, y = _extract(batch)
                x_e = x_e.to(self.device).float()
                y = y.to(self.device).long()
                z = self._embed_edges(x_e)
                logits = self.head(z)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.detach().cpu()))
        return self

    @torch.no_grad()
    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        self.encoder.eval()
        self.head.eval()
        x_t = torch.as_tensor(_to_numpy(x), dtype=torch.float32, device=self.device)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        z = self._embed_edges(x_t)
        logits = self.head(z)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy().astype(np.int64)
        scores = probs.max(dim=-1).values.cpu().numpy().astype(np.float32)
        return preds, scores
