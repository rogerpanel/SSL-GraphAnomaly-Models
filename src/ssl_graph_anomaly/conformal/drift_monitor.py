"""Distribution-drift and FAR-overshoot monitor for online recalibration."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.stats import ks_2samp


class DriftMonitor:
    def __init__(
        self,
        ks_threshold: float = 0.10,
        recent_buffer: int = 5000,
        sliding_window_minutes: float = 10.0,
        far_overshoot_threshold: float = 0.005,
    ) -> None:
        self.ks_threshold = float(ks_threshold)
        self.recent_buffer = int(recent_buffer)
        self.sliding_window_seconds = float(sliding_window_minutes) * 60.0
        self.far_overshoot_threshold = float(far_overshoot_threshold)
        self._recent_benign: deque[float] = deque(maxlen=self.recent_buffer)
        self._calibration: np.ndarray | None = None
        # Sliding window of (timestamp, is_false_alarm) for empirical FAR.
        self._far_events: deque[tuple[float, int]] = deque()

    def set_calibration(self, calibration_scores: np.ndarray) -> None:
        arr = np.asarray(calibration_scores, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        self._calibration = arr

    def add_observation(
        self,
        energy: float,
        is_benign_predicted: bool,
        is_confirmed_benign: bool,
        timestamp: float,
    ) -> None:
        if is_benign_predicted:
            self._recent_benign.append(float(energy))
        # A false alarm = flagged as malicious but confirmed benign.
        is_false_alarm = (not is_benign_predicted) and is_confirmed_benign
        self._far_events.append((float(timestamp), int(is_false_alarm)))
        self._evict_old(float(timestamp))

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.sliding_window_seconds
        while self._far_events and self._far_events[0][0] < cutoff:
            self._far_events.popleft()

    def _current_far(self) -> float:
        if not self._far_events:
            return 0.0
        total = len(self._far_events)
        fa = sum(ev for _, ev in self._far_events)
        return float(fa) / float(total)

    def _ks_stat(self) -> float:
        if self._calibration is None or len(self._calibration) == 0:
            return 0.0
        if len(self._recent_benign) < 2:
            return 0.0
        recent = np.asarray(self._recent_benign, dtype=np.float64)
        stat, _ = ks_2samp(recent, self._calibration)
        return float(stat)

    def should_recalibrate(self, alpha_star: float = 0.05) -> tuple[bool, dict[str, Any]]:
        ks_stat = self._ks_stat()
        current_far = self._current_far()
        far_overshoot = current_far - alpha_star
        flag = bool(
            (ks_stat > self.ks_threshold)
            or (far_overshoot > self.far_overshoot_threshold)
        )
        info: dict[str, Any] = {
            "ks_stat": ks_stat,
            "current_far": current_far,
            "far_overshoot": far_overshoot,
            "n_recent": len(self._recent_benign),
            "n_far_window": len(self._far_events),
        }
        return flag, info

    def reset(self) -> None:
        self._recent_benign.clear()
        self._far_events.clear()
