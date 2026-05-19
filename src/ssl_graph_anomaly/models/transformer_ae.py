"""Discrepancy transformer autoencoder."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class DiscrepancyTransformerAE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        d_token = embed_dim // num_tokens
        if d_token * num_tokens < embed_dim:
            d_token += 1
        if d_token % num_heads != 0:
            d_token = ((d_token + num_heads - 1) // num_heads) * num_heads
        self.d_token = d_token
        self.token_dim = num_tokens * d_token

        self.to_tokens = nn.Linear(embed_dim, self.token_dim)
        self.from_tokens = nn.Linear(self.token_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=num_heads,
            dim_feedforward=ff_multiplier * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        pe = self._build_positional_encoding(num_tokens, d_token)
        self.register_buffer("pos_enc", pe, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _build_positional_encoding(num_tokens: int, d_token: int) -> Tensor:
        pe = torch.zeros(num_tokens, d_token)
        position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
        if d_token > 1:
            div_term = torch.exp(
                torch.arange(0, d_token, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_token)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_token > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        else:
            pe[:, 0] = torch.sin(position.squeeze(1))
        return pe.unsqueeze(0)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        b = z.size(0)
        tokens = self.to_tokens(z).view(b, self.num_tokens, self.d_token)
        tokens = tokens + self.pos_enc.to(tokens.dtype)
        out = self.transformer(tokens)
        flat = out.reshape(b, self.num_tokens * self.d_token)
        z_recon = self.from_tokens(flat)
        recon_loss_per_sample = ((z - z_recon) ** 2).sum(dim=1) / self.embed_dim
        return z_recon, recon_loss_per_sample
