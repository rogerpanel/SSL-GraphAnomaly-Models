"""Kitsune: ensemble-of-autoencoders anomaly detector (Mirsky et al., NDSS 2018)."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
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


class _TinyAE(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden)
        self.dec = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.enc(x))
        return self.dec(h)


class KitsuneBaseline:
    """Ensemble of cluster-wise tiny autoencoders with an RMSE-output autoencoder."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.num_features = int(cfg["data"]["num_features"])
        self.num_clusters = int(cfg.get("baselines", {}).get("kitsune_clusters", 8))
        self.hidden = int(cfg.get("baselines", {}).get("kitsune_hidden", 4))
        self.epochs = int(cfg.get("baselines", {}).get("kitsune_epochs", 5))
        self.lr = float(cfg.get("baselines", {}).get("kitsune_lr", 1e-3))
        self.device = torch.device(cfg["experiment"].get("device", "cpu"))
        self.threshold = float(cfg.get("baselines", {}).get("kitsune_threshold", 0.5))

        self._clusters: list[list[int]] = []
        self._aes: list[_TinyAE] = []
        self._output_ae: _TinyAE | None = None

    def _cluster_features(self, x: np.ndarray) -> list[list[int]]:
        if x.shape[0] < 2:
            corr = np.eye(self.num_features, dtype=np.float64)
        else:
            corr = np.corrcoef(x, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
        k = min(self.num_clusters, self.num_features)
        km = KMeans(n_clusters=k, n_init=4, random_state=0)
        labels = km.fit_predict(corr)
        clusters: list[list[int]] = [[] for _ in range(k)]
        for f_idx, c in enumerate(labels):
            clusters[int(c)].append(int(f_idx))
        return [c for c in clusters if c]

    def _collect(self, loader: Iterable[Any], limit: int = 50_000) -> np.ndarray:
        rows: list[np.ndarray] = []
        n = 0
        for batch in loader:
            x = _to_numpy(_extract_x(batch)).astype(np.float32)
            rows.append(x)
            n += x.shape[0]
            if n >= limit:
                break
        if not rows:
            return np.zeros((0, self.num_features), dtype=np.float32)
        return np.concatenate(rows, axis=0)[:limit]

    def _train_ae(self, ae: _TinyAE, data: np.ndarray) -> None:
        if data.shape[0] == 0:
            return
        ae.to(self.device).train()
        opt = Adam(ae.parameters(), lr=self.lr)
        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        bs = min(512, tensor.size(0))
        for _ in range(self.epochs):
            perm = torch.randperm(tensor.size(0), device=self.device)
            for i in range(0, tensor.size(0), bs):
                idx = perm[i : i + bs]
                xb = tensor[idx]
                recon = ae(xb)
                loss = torch.mean((recon - xb) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        ae.eval()

    def fit(self, loader: Iterable[Any]) -> "KitsuneBaseline":
        data = self._collect(loader)
        self._clusters = self._cluster_features(data) if data.size else [
            list(range(self.num_features))
        ]
        self._aes = []
        rmse_train: list[np.ndarray] = []
        for cluster in tqdm(self._clusters, desc="kitsune-cluster-aes", leave=False):
            ae = _TinyAE(in_dim=len(cluster), hidden=max(1, self.hidden))
            sub = data[:, cluster] if data.size else np.zeros((0, len(cluster)), dtype=np.float32)
            self._train_ae(ae, sub)
            self._aes.append(ae)
            if sub.size:
                with torch.no_grad():
                    t = torch.as_tensor(sub, dtype=torch.float32, device=self.device)
                    r = ae(t)
                    rmse = torch.sqrt(torch.mean((r - t) ** 2, dim=1)).cpu().numpy()
            else:
                rmse = np.zeros((0,), dtype=np.float32)
            rmse_train.append(rmse)
        if rmse_train and rmse_train[0].size > 0:
            stacked = np.stack(rmse_train, axis=1).astype(np.float32)
            self._output_ae = _TinyAE(in_dim=stacked.shape[1], hidden=max(1, self.hidden))
            self._train_ae(self._output_ae, stacked)
        return self

    def _score_batch(self, x_np: np.ndarray) -> np.ndarray:
        if not self._aes:
            return np.zeros((x_np.shape[0],), dtype=np.float32)
        rmses = []
        with torch.no_grad():
            for cluster, ae in zip(self._clusters, self._aes):
                sub = torch.as_tensor(x_np[:, cluster], dtype=torch.float32, device=self.device)
                r = ae(sub)
                rmses.append(torch.sqrt(torch.mean((r - sub) ** 2, dim=1)).cpu().numpy())
        ensemble = np.stack(rmses, axis=1)
        if self._output_ae is not None:
            with torch.no_grad():
                t = torch.as_tensor(ensemble, dtype=torch.float32, device=self.device)
                r = self._output_ae(t)
                final = torch.sqrt(torch.mean((r - t) ** 2, dim=1)).cpu().numpy()
            return final.astype(np.float32)
        return ensemble.mean(axis=1).astype(np.float32)

    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        x_np = _to_numpy(x).astype(np.float32)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        scores = self._score_batch(x_np)
        preds = (scores > self.threshold).astype(np.int64)
        return preds, scores
