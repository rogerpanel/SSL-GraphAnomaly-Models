"""End-to-end SSL-GraphAnomaly model assembly."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ssl_graph_anomaly.models.attention_gated import AttentionGatedEncoder
from ssl_graph_anomaly.models.egraphsage import EGraphSAGEEncoder
from ssl_graph_anomaly.models.energy_head import EnergyLogitHead
from ssl_graph_anomaly.models.mahalanobis import MahalanobisEnergy
from ssl_graph_anomaly.models.transformer_ae import DiscrepancyTransformerAE


class SSLGraphAnomaly(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        data = cfg["data"]
        mcfg = cfg["model"]
        enc_cfg = mcfg["encoder"]
        stream_cfg = mcfg.get("streaming_variant", {})
        ae_cfg = mcfg["transformer_ae"]
        head_cfg = mcfg["classification_head"]
        e_cfg = mcfg.get("energy", {})

        in_edge_dim = int(enc_cfg.get("edge_feature_dim", data["num_features"]))
        hidden_dim = int(enc_cfg["hidden_dim"])
        num_layers = int(enc_cfg.get("num_layers", 2))
        dropout = float(enc_cfg.get("dropout", 0.10))
        in_node_dim = int(enc_cfg.get("in_node_dim", 4))

        self.graph_encoder = EGraphSAGEEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.stream_enabled = bool(stream_cfg.get("enabled", True))
        self.stream_encoder = AttentionGatedEncoder(
            in_edge_dim=in_edge_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=bool(stream_cfg.get("use_layer_norm", True)),
            dropout=dropout,
        )

        embed_dim = self.graph_encoder.out_dim
        self.transformer_ae = DiscrepancyTransformerAE(
            embed_dim=embed_dim,
            num_tokens=int(ae_cfg.get("num_tokens", 8)),
            num_layers=int(ae_cfg.get("num_layers", 2)),
            num_heads=int(ae_cfg.get("num_heads", 4)),
            ff_multiplier=int(ae_cfg.get("ff_width_multiplier", 2)),
            dropout=float(ae_cfg.get("dropout", 0.10)),
        )

        self.mahalanobis = MahalanobisEnergy(
            dim=embed_dim,
            eps=float(e_cfg.get("mahalanobis_eps", 1e-6)),
        )
        self.use_mahalanobis = bool(e_cfg.get("use_mahalanobis", True))

        self.head = EnergyLogitHead(
            in_dim=embed_dim,
            num_classes=int(head_cfg.get("num_classes", data["num_classes"])),
            hidden=tuple(head_cfg.get("hidden_dims", (256, 128))),
            mahalanobis_lambda=float(e_cfg.get("mahalanobis_lambda", 0.10)),
            energy_push_weight=float(head_cfg.get("energy_push_weight", 2.0)),
            energy_mlp_hidden=tuple(e_cfg.get("mlp_hidden", (128, 64))),
        )
        self.embed_dim = embed_dim

    def _heads(self, z: Tensor) -> Dict[str, Tensor]:
        z_recon, s_rec = self.transformer_ae(z)
        if self.use_mahalanobis:
            s_maha = self.mahalanobis(z)
        else:
            s_maha = torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
        head_out = self.head(z, z_recon, s_rec, s_maha)
        out = {
            "z": z,
            "z_recon": z_recon,
            "s_rec": s_rec,
            "s_maha": s_maha,
        }
        out.update(head_out)
        return out

    def forward_graph(
        self, x_v: Tensor, edge_index: Tensor, x_e: Tensor
    ) -> Dict[str, Tensor]:
        z, _ = self.graph_encoder(x_v, edge_index, x_e)
        return self._heads(z)

    def forward_stream(self, x_e: Tensor) -> Dict[str, Tensor]:
        z = self.stream_encoder(x_e)
        return self._heads(z)

    @torch.no_grad()
    def fit_mahalanobis(
        self,
        loader: Iterable[Any],
        device: Optional[torch.device] = None,
    ) -> None:
        self.eval()
        zs: list[Tensor] = []
        for batch in loader:
            if isinstance(batch, dict):
                x_e = batch["x_e"]
                x_v = batch.get("x_v")
                edge_index = batch.get("edge_index")
            elif isinstance(batch, (tuple, list)):
                if len(batch) >= 3:
                    x_v, edge_index, x_e = batch[0], batch[1], batch[2]
                else:
                    x_e = batch[0]
                    x_v, edge_index = None, None
            else:
                x_e = batch
                x_v, edge_index = None, None
            if device is not None:
                x_e = x_e.to(device)
                if x_v is not None:
                    x_v = x_v.to(device)
                if edge_index is not None:
                    edge_index = edge_index.to(device)
            if x_v is not None and edge_index is not None:
                z, _ = self.graph_encoder(x_v, edge_index, x_e)
            else:
                z = self.stream_encoder(x_e)
            zs.append(z.detach())
        if not zs:
            return
        all_z = torch.cat(zs, dim=0)
        self.mahalanobis.fit(all_z)

    def save_checkpoint(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: Optional[Any] = None) -> "SSLGraphAnomaly":
        blob = torch.load(path, map_location=map_location)
        model = cls(blob["cfg"])
        model.load_state_dict(blob["state_dict"])
        return model
