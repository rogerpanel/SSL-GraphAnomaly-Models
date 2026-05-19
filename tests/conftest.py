"""Shared pytest fixtures for the SSL-GraphAnomaly test suite.

Every fixture is sized to keep the entire test pass well under five
seconds on a modest CPU: ten hosts, fifty flows, eighty-three CICFlowMeter
features, and integer labels in ``[0, 33]`` matching the 34-class
attack taxonomy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch


SEED: int = 1234


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(autouse=True)
def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


@pytest.fixture()
def tiny_cfg() -> dict[str, Any]:
    """Return a minimal configuration that makes every component lightweight."""
    return {
        "experiment": {
            "name": "test",
            "seed": SEED,
            "seeds": [SEED],
            "output_dir": "outputs/tests",
            "device": "cpu",
            "precision": "fp32",
            "num_workers": 0,
        },
        "data": {
            "suite": "iis3d",
            "root": "tests/_synth",
            "num_features": 83,
            "num_classes": 34,
            "benign_class_id": 0,
            "window_seconds": 60,
            "split": {"train_benign": 0.6, "calibration_benign": 0.1,
                      "val": 0.1, "test": 0.2},
            "preprocessing": {"standardize": "zscore",
                              "clip_quantile": 0.999,
                              "log1p_features": []},
        },
        "graph": {
            "builder": "temporal_host",
            "max_edges_per_window": 200,
            "max_neighbours": 4,
            "symmetric": False,
        },
        "model": {
            "encoder": {
                "type": "egraphsage",
                "hidden_dim": 16,
                "num_layers": 2,
                "aggregator": "mean",
                "activation": "relu",
                "edge_feature_dim": 83,
                "dropout": 0.0,
            },
            "streaming_variant": {
                "enabled": True,
                "proj_dim": 16,
                "use_layer_norm": True,
            },
            "transformer_ae": {
                "num_tokens": 4,
                "num_layers": 1,
                "num_heads": 2,
                "ff_width_multiplier": 2,
                "activation": "gelu",
                "positional_encoding": "sinusoidal",
                "dropout": 0.0,
            },
            "classification_head": {
                "hidden_dims": [32, 16],
                "num_classes": 34,
                "energy_push_weight": 2.0,
            },
            "energy": {
                "use_mahalanobis": True,
                "mahalanobis_lambda": 0.1,
                "mahalanobis_eps": 1.0e-6,
                "mlp_hidden": [16, 8],
            },
        },
        "loss": {
            "reconstruction_weight": 1.0,
            "contrastive_weight": 1.0,
            "energy_margin_weight": 0.5,
            "distillation_weight": 0.25,
            "margin": 1.0,
            "infonce_temperature": 0.5,
            "distillation_temperature": 2.0,
            "mask_rate": 0.15,
            "feature_dropout": 0.10,
            "gaussian_jitter_sigma": 0.05,
        },
        "optim": {"optimizer": "adamw",
                  "learning_rate": 5.0e-4,
                  "weight_decay": 1.0e-5,
                  "beta1": 0.9, "beta2": 0.999,
                  "scheduler": "cosine", "warmup_steps": 0,
                  "gradient_clip": 1.0},
        "train": {"ssl_epochs": 1, "distill_epochs": 1, "batch_size": 16,
                  "log_every": 10, "eval_every_epochs": 1,
                  "checkpoint_dir": "outputs/tests/ckpts",
                  "early_stop_patience": 1},
        "conformal": {
            "enabled": True, "alpha": 0.1, "mode": "mondrian",
            "aci": {"enabled": True, "gamma": 0.005, "warmup_steps": 100},
            "drift": {"ks_threshold": 0.10, "recent_buffer": 500,
                      "sliding_far_window_minutes": 10,
                      "far_overshoot_threshold": 0.005},
        },
        "evaluation": {"metrics": ["macro_f1", "auroc", "ece",
                                   "coverage", "set_size", "latency_ms"],
                       "baselines": {"enabled": False,
                                     "methods": []},
                       "adversarial": {"enabled": False,
                                       "methods": ["fgsm", "pgd20"],
                                       "epsilons": [0.0, 0.01],
                                       "pgd_steps": 20,
                                       "pgd_alpha": 0.005}},
    }


@pytest.fixture()
def tiny_dataset() -> dict[str, torch.Tensor]:
    """Return a tiny synthetic dataset: 50 flows over 10 hosts."""
    num_flows = 50
    num_hosts = 10
    num_features = 83
    num_classes = 34

    rng = np.random.default_rng(SEED)
    features = rng.standard_normal((num_flows, num_features)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=num_flows).astype(np.int64)
    src = rng.integers(0, num_hosts, size=num_flows).astype(np.int64)
    dst = rng.integers(0, num_hosts, size=num_flows).astype(np.int64)
    timestamps = np.sort(rng.uniform(0.0, 60.0, size=num_flows)
                         ).astype(np.float32)

    return {
        "features": torch.from_numpy(features),
        "labels": torch.from_numpy(labels),
        "src": torch.from_numpy(src),
        "dst": torch.from_numpy(dst),
        "timestamps": torch.from_numpy(timestamps),
        "num_hosts": torch.tensor(num_hosts),
        "num_features": torch.tensor(num_features),
        "num_classes": torch.tensor(num_classes),
    }


@pytest.fixture()
def edge_index(tiny_dataset: dict[str, torch.Tensor]) -> torch.Tensor:
    """Convenience fixture exposing a [2, E] edge_index tensor."""
    return torch.stack([tiny_dataset["src"], tiny_dataset["dst"]], dim=0)
