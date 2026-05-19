"""SecurityBERT: HuggingFace transformer fine-tuned on CSV-tokenised flow features."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

try:  # optional dependency
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    _HAS_HF = True
except Exception:  # pragma: no cover - optional
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _HAS_HF = False


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


def _flow_to_csv(row: np.ndarray) -> str:
    return ",".join(f"{float(v):.5f}" for v in row.tolist())


class _FallbackTransformer(nn.Module):
    """Tiny transformer used when `transformers` is unavailable."""

    def __init__(self, num_features: int, num_classes: int, d_model: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=2 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.proj(x.unsqueeze(-1))
        cls = self.cls.expand(b, -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0])


class SecurityBERTBaseline:
    """HuggingFace transformer (or fallback) trained on CSV-tokenised flow rows."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["experiment"].get("device", "cpu"))
        self.num_classes = int(cfg["data"]["num_classes"])
        bcfg = cfg.get("baselines", {}) or {}
        self.backbone_name: str = str(bcfg.get("securitybert_backbone", "prajjwal1/bert-tiny"))
        self.max_length: int = int(bcfg.get("securitybert_max_length", 128))
        self.epochs = int(cfg["train"].get("baseline_epochs", 3))
        self.lr = float(bcfg.get("securitybert_lr", 5e-5))
        self.batch_size = int(cfg["train"].get("batch_size", 64))

        self._using_hf: bool = False
        self.tokenizer = None
        self.model: nn.Module
        if _HAS_HF:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.backbone_name, num_labels=self.num_classes
                ).to(self.device)
                self._using_hf = True
            except Exception:
                self._using_hf = False
        if not self._using_hf:
            self.model = _FallbackTransformer(
                num_features=int(cfg["data"]["num_features"]),
                num_classes=self.num_classes,
            ).to(self.device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._using_hf and self.tokenizer is not None:
            texts = [_flow_to_csv(row) for row in _to_numpy(x)]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            return out.logits
        return self.model(x)

    def fit(self, loader: Iterable[Any]) -> "SecurityBERTBaseline":
        opt = AdamW(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            self.model.train()
            pbar = tqdm(loader, desc="securitybert-train", leave=False)
            for batch in pbar:
                x, y = _extract(batch)
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                logits = self._forward(x)
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
        logits = self._forward(x_t)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy().astype(np.int64)
        scores = probs.max(dim=-1).values.cpu().numpy().astype(np.float32)
        return preds, scores
