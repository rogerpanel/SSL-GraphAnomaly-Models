"""Mondrian (per-class) conformal predictor (paper Eq. 10)."""

from __future__ import annotations

import math

import numpy as np
import torch


def _to_numpy(scores: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().numpy().astype(np.float64)
    return np.asarray(scores, dtype=np.float64)


def _quantile(s: np.ndarray, alpha: float) -> float:
    n = int(s.shape[0])
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    return float(np.sort(s)[k - 1])


class MondrianConformal:
    def __init__(
        self,
        alpha: float = 0.05,
        num_classes: int = 34,
        min_calib_size: int = 50,
    ) -> None:
        self.alpha = float(alpha)
        self.num_classes = int(num_classes)
        self.min_calib_size = int(min_calib_size)
        self._quantiles: dict[int, float] = {}
        self._marginal: float | None = None

    def calibrate(
        self,
        scores: np.ndarray | torch.Tensor,
        pred_classes: np.ndarray | torch.Tensor,
    ) -> None:
        s = _to_numpy(scores)
        c = _to_numpy(pred_classes).astype(np.int64)
        if s.shape[0] != c.shape[0]:
            raise ValueError("scores and pred_classes must have same length")
        finite = np.isfinite(s)
        s = s[finite]
        c = c[finite]
        if s.shape[0] == 0:
            raise ValueError("No finite scores for calibration")
        self._marginal = _quantile(s, self.alpha)
        self._quantiles = {}
        for cls in range(self.num_classes):
            mask = c == cls
            if int(mask.sum()) >= self.min_calib_size:
                self._quantiles[cls] = _quantile(s[mask], self.alpha)
            else:
                # Too few samples: fall back to marginal quantile.
                self._quantiles[cls] = self._marginal

    @property
    def quantiles(self) -> dict[int, float]:
        return dict(self._quantiles)

    def predict_set(self, score: float, pred_class: int) -> list[int]:
        if not self._quantiles:
            raise RuntimeError("MondrianConformal not calibrated")
        s = float(score)
        prediction_set = [cls for cls, q in self._quantiles.items() if s <= q]
        if not prediction_set:
            # Empty set: emit the single most-likely class (smallest score - q margin).
            best = min(self._quantiles.items(), key=lambda kv: s - kv[1])
            prediction_set = [best[0]]
        return prediction_set
