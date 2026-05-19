"""Single-flow streaming inference with conformal certification and drift monitoring."""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ssl_graph_anomaly.conformal import (
    AdaptiveConformalInference,
    DriftMonitor,
    MondrianConformal,
    SplitConformal,
    conformal_predict,
)
from ssl_graph_anomaly.data import ATTACK_TAXONOMY_34
from ssl_graph_anomaly.models import SSLGraphAnomaly

_MODEL_ID = 14
_MODEL_NAME = "SSL-GraphAnomaly"


class StreamingInference:
    """Online single/batch predictor with split-conformal + ACI + drift monitoring."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        conformal_path: str | Path,
        device: str | torch.device = "cpu",
        alpha_star: float = 0.05,
        aci_gamma: float = 0.005,
        on_drift: Callable[[Sequence[float]], None] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = SSLGraphAnomaly.load_checkpoint(
            str(checkpoint_path), map_location=self.device
        )
        self.model = self.model.to(self.device).eval()

        with Path(conformal_path).open("rb") as fh:
            blob = pickle.load(fh)
        self.mondrian: MondrianConformal | None = blob.get("mondrian")
        self.marginal: SplitConformal | None = blob.get("marginal")
        self._calibration_scores: np.ndarray = np.asarray(
            blob.get("calibration_scores", []), dtype=np.float64
        )
        self.alpha_star = float(blob.get("alpha", alpha_star))

        self.aci = AdaptiveConformalInference(alpha_star=self.alpha_star, gamma=aci_gamma)
        self.drift = DriftMonitor()
        if self._calibration_scores.size > 0:
            self.drift.set_calibration(self._calibration_scores)
        self.on_drift = on_drift

        num_classes = int(self.model.cfg["data"]["num_classes"])
        try:
            if isinstance(ATTACK_TAXONOMY_34, dict):
                self.class_names: dict[int, str] = {int(k): str(v) for k, v in ATTACK_TAXONOMY_34.items()}
            else:
                self.class_names = {i: str(v) for i, v in enumerate(ATTACK_TAXONOMY_34)}
        except Exception:
            self.class_names = {i: str(i) for i in range(num_classes)}
        self.benign_class = int(self.model.cfg["data"].get("benign_class_id", 0))

    def _class_name(self, idx: int) -> str:
        return self.class_names.get(int(idx), str(int(idx)))

    def _to_batch(self, flow_features: np.ndarray | list[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(flow_features, torch.Tensor):
            x = flow_features.detach().to(self.device).float()
        else:
            x = torch.as_tensor(np.asarray(flow_features, dtype=np.float32), device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def predict(
        self,
        flow_features: np.ndarray | list[float] | torch.Tensor,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        x = self._to_batch(flow_features)
        out = self.model.forward_stream(x)
        logits = out.get("pushed_logits", out["logits"]).detach().cpu()
        energy = out["energy"].detach().cpu().numpy().astype(np.float64)
        s_rec = out["s_rec"].detach().cpu().numpy().astype(np.float64)
        s_maha = out["s_maha"].detach().cpu().numpy().astype(np.float64)
        preds = torch.argmax(logits, dim=-1).numpy().astype(np.int64)

        results: list[dict[str, Any]] = []
        for i in range(x.size(0)):
            res = conformal_predict(
                float(energy[i]),
                logits[i],
                mondrian=self.mondrian,
                marginal=self.marginal,
                benign_class=self.benign_class,
            )
            quantiles = self.mondrian.quantiles if self.mondrian is not None else {}
            q = float(quantiles.get(int(preds[i]), float("nan"))) if quantiles else (
                self.marginal.quantile if self.marginal is not None else float("nan")
            )
            pred_set_ids = [int(c) for c in res["prediction_set"]]
            entry: dict[str, Any] = {
                "predicted_class": self._class_name(int(preds[i])),
                "predicted_class_id": int(preds[i]),
                "prediction_set": [self._class_name(c) for c in pred_set_ids],
                "anomaly_energy": float(energy[i]),
                "mahalanobis_distance": float(s_maha[i]) if s_maha.size else 0.0,
                "reconstruction_discrepancy": float(s_rec[i]) if s_rec.size else 0.0,
                "conformal_quantile": q,
                "alpha_target": float(self.aci.current_alpha),
                "coverage_certified": bool(res["coverage_certified"]),
                "abstained": bool(res["abstained"]),
                "model_id": _MODEL_ID,
                "model_name": _MODEL_NAME,
            }
            results.append(entry)
            self._update_drift(float(energy[i]), int(preds[i]))

        if isinstance(flow_features, (list, tuple, np.ndarray)) and np.asarray(flow_features).ndim == 1:
            return results[0]
        if isinstance(flow_features, torch.Tensor) and flow_features.ndim == 1:
            return results[0]
        return results

    def _update_drift(self, energy: float, predicted_class: int) -> None:
        is_benign_pred = predicted_class == self.benign_class
        # Without a label feedback channel we treat predicted-benign flows as confirmed-benign
        # for the purposes of refreshing the recent-benign buffer.
        self.drift.add_observation(
            energy=float(energy),
            is_benign_predicted=is_benign_pred,
            is_confirmed_benign=is_benign_pred,
            timestamp=time.time(),
        )
        miscovered = (not is_benign_pred)
        self.aci.update(miscovered)
        flag, _info = self.drift.should_recalibrate(alpha_star=self.alpha_star)
        if flag and self.on_drift is not None:
            recent = list(self.drift._recent_benign)  # noqa: SLF001 - intentional access
            try:
                self.on_drift(recent)
            finally:
                self.drift.reset()
