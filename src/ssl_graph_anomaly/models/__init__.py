"""Neural network model layer of SSL-GraphAnomaly."""
from ssl_graph_anomaly.models.attention_gated import AttentionGatedEncoder
from ssl_graph_anomaly.models.egraphsage import EGraphSAGEEncoder
from ssl_graph_anomaly.models.energy_head import EnergyLogitHead
from ssl_graph_anomaly.models.mahalanobis import MahalanobisEnergy
from ssl_graph_anomaly.models.ssl_graph_anomaly import SSLGraphAnomaly
from ssl_graph_anomaly.models.transformer_ae import DiscrepancyTransformerAE

__all__ = [
    "AttentionGatedEncoder",
    "DiscrepancyTransformerAE",
    "EGraphSAGEEncoder",
    "EnergyLogitHead",
    "MahalanobisEnergy",
    "SSLGraphAnomaly",
]
