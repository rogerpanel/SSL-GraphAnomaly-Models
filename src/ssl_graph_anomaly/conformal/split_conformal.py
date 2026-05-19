"""Split (marginal) conformal predictor (paper Eq. 9)."""

from __future__ import annotations

import math

import numpy as np
import torch


def _to_numpy(scores: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().numpy().astype(np.float64)
    return np.asarray(scores, dtype=np.float64)


class SplitConformal:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = float(alpha)
        self._q_hat: float | None = None
        self._n: int = 0

    def calibrate(self, scores: np.ndarray | torch.Tensor) -> None:
        s = _to_numpy(scores)
        s = s[np.isfinite(s)]
        n = int(s.shape[0])
        if n == 0:
            raise ValueError("SplitConformal.calibrate requires at least one score")
        # Finite-sample correction: rank index ceil((n+1)*(1-alpha)) (clipped to n).
        k = int(math.ceil((n + 1) * (1.0 - self.alpha)))
        k = max(1, min(k, n))
        s_sorted = np.sort(s)
        self._q_hat = float(s_sorted[k - 1])
        self._n = n

    @property
    def quantile(self) -> float:
        if self._q_hat is None:
            raise RuntimeError("SplitConformal not calibrated")
        return self._q_hat

    def predict_set(self, score: float) -> bool:
        return float(score) <= self.quantile

    def predict_sets(self, scores: np.ndarray | torch.Tensor) -> np.ndarray:
        s = _to_numpy(scores)
        return s <= self.quantile

    def coverage(
        self,
        scores: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        s = _to_numpy(scores)
        accepted = s <= self.quantile
        if labels is None:
            return float(accepted.mean())
        y = _to_numpy(labels).astype(np.int64)
        benign_mask = y == 0
        if benign_mask.sum() == 0:
            return 0.0
        return float(accepted[benign_mask].mean())
