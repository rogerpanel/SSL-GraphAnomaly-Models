"""Adaptive Conformal Inference (paper Eq. 11)."""

from __future__ import annotations


class AdaptiveConformalInference:
    _LO = 1e-4
    _HI = 1.0 - 1e-4

    def __init__(self, alpha_star: float = 0.05, gamma: float = 0.005) -> None:
        self.alpha_star = float(alpha_star)
        self.gamma = float(gamma)
        self._alpha = float(alpha_star)

    def update(self, miscovered: bool) -> float:
        err = 1.0 if miscovered else 0.0
        new_alpha = self._alpha + self.gamma * (self.alpha_star - err)
        # Clip to (0, 1) open interval so quantile computations stay valid.
        if new_alpha < self._LO:
            new_alpha = self._LO
        elif new_alpha > self._HI:
            new_alpha = self._HI
        self._alpha = float(new_alpha)
        return self._alpha

    @property
    def current_alpha(self) -> float:
        return self._alpha

    def reset(self) -> None:
        self._alpha = float(self.alpha_star)
