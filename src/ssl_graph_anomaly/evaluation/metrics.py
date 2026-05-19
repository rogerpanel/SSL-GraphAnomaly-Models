"""Classification, calibration, and conformal-coverage metrics."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import click
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_graph_anomaly.conformal import conformal_predict
from ssl_graph_anomaly.data import build_dataset
from ssl_graph_anomaly.models import SSLGraphAnomaly
from ssl_graph_anomaly.utils import get_logger, load_config


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def macro_f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_t = _as_numpy(y_true).astype(np.int64).ravel()
    y_p = _as_numpy(y_pred).astype(np.int64).ravel()
    if y_t.size == 0:
        return 0.0
    return float(f1_score(y_t, y_p, average="macro", zero_division=0))


def auroc_binary(y_true_binary: Iterable[int], scores: Iterable[float]) -> float:
    y = _as_numpy(y_true_binary).astype(np.int64).ravel()
    s = _as_numpy(scores).astype(np.float64).ravel()
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def expected_calibration_error(
    y_true: Iterable[int],
    probs: Iterable[Sequence[float]],
    n_bins: int = 15,
) -> float:
    y = _as_numpy(y_true).astype(np.int64).ravel()
    p = _as_numpy(probs).astype(np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if y.size == 0:
        return 0.0
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(y.size)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def coverage(
    prediction_sets: Sequence[Sequence[int]],
    y_true: Iterable[int],
) -> float:
    y = _as_numpy(y_true).astype(np.int64).ravel()
    if y.size == 0:
        return 0.0
    hits = 0
    for i, label in enumerate(y):
        if i >= len(prediction_sets):
            break
        if int(label) in set(int(c) for c in prediction_sets[i]):
            hits += 1
    return float(hits) / float(y.size)


def mean_set_size(prediction_sets: Sequence[Sequence[int]]) -> float:
    if not prediction_sets:
        return 0.0
    sizes = [len(s) for s in prediction_sets]
    return float(np.mean(sizes))


def compute_metrics(
    out_dict: dict[str, Any],
    y_true: Iterable[int],
    prediction_sets: Sequence[Sequence[int]] | None = None,
) -> dict[str, float]:
    logits = _as_numpy(out_dict["pushed_logits"])
    energy = _as_numpy(out_dict["energy"]).astype(np.float64).ravel()
    y = _as_numpy(y_true).astype(np.int64).ravel()
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    y_bin = (y != 0).astype(np.int64)

    metrics: dict[str, float] = {
        "macro_f1": macro_f1(y, preds),
        "auroc": auroc_binary(y_bin, energy),
        "ece": expected_calibration_error(y, probs),
    }
    if prediction_sets is not None:
        metrics["coverage"] = coverage(prediction_sets, y)
        metrics["mean_set_size"] = mean_set_size(prediction_sets)
    return metrics


def _extract(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        return batch["x_e"], batch.get("y")
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 4:
            return batch[2], batch[3]
        if len(batch) == 2:
            return batch[0], batch[1]
        return batch[0], None
    return batch, None


def _make_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    ds = build_dataset(cfg, split)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["experiment"].get("num_workers", 0))
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)


@torch.no_grad()
def evaluate_model(cfg: dict[str, Any]) -> dict[str, Any]:
    logger = get_logger("ssl.evaluate")
    device = torch.device(cfg["experiment"].get("device", "cpu"))
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SSLGraphAnomaly.load_checkpoint(str(ckpt_dir / "distilled.pt"), map_location=device)
    model = model.to(device).eval()

    conformal_path = ckpt_dir / "conformal.pkl"
    mondrian = None
    marginal = None
    if conformal_path.exists():
        with conformal_path.open("rb") as fh:
            blob = pickle.load(fh)
        mondrian = blob.get("mondrian")
        marginal = blob.get("marginal")

    loader = _make_loader(cfg, "test")
    all_logits: list[np.ndarray] = []
    all_energy: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    prediction_sets: list[list[int]] = []
    benign_class = int(cfg["data"].get("benign_class_id", 0))

    for batch in tqdm(loader, desc="evaluate", leave=False):
        x_e, y = _extract(batch)
        x_e = x_e.to(device).float()
        out = model.forward_stream(x_e)
        logits = out.get("pushed_logits", out.get("logits"))
        energy = out["energy"]
        all_logits.append(logits.detach().cpu().numpy())
        all_energy.append(energy.detach().cpu().numpy())
        if y is not None:
            all_y.append(_as_numpy(y).astype(np.int64).ravel())
        for i in range(logits.size(0)):
            res = conformal_predict(
                float(energy[i].detach().cpu().item()),
                logits[i].detach().cpu(),
                mondrian=mondrian,
                marginal=marginal,
                benign_class=benign_class,
            )
            prediction_sets.append([int(c) for c in res["prediction_set"]])

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, int(cfg["data"]["num_classes"])))
    energy_np = np.concatenate(all_energy, axis=0) if all_energy else np.zeros((0,))
    y_np = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=np.int64)

    metrics = compute_metrics(
        {"pushed_logits": logits_np, "energy": energy_np},
        y_np,
        prediction_sets=prediction_sets if prediction_sets else None,
    )
    metrics["num_samples"] = int(y_np.size)
    metrics["alpha_target"] = float(cfg["conformal"].get("alpha", 0.05))

    report_path = out_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("metrics -> %s", report_path)
    return metrics


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--report", "report_path", type=click.Path(), default=None)
def cli(config_path: str, report_path: str | None) -> None:
    cfg = load_config(config_path)
    Path(cfg["experiment"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(cfg)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(report_path).open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    cli()
