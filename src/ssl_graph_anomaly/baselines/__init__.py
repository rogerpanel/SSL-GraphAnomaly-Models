"""External baselines used for benchmarking SSL-GraphAnomaly."""
from ssl_graph_anomaly.baselines.anomale import AnomalEBaseline
from ssl_graph_anomaly.baselines.egraphsage import EGraphSAGEBaseline
from ssl_graph_anomaly.baselines.kitsune import KitsuneBaseline
from ssl_graph_anomaly.baselines.rtids import RTIDSBaseline
from ssl_graph_anomaly.baselines.securitybert import SecurityBERTBaseline

__all__ = [
    "AnomalEBaseline",
    "EGraphSAGEBaseline",
    "KitsuneBaseline",
    "RTIDSBaseline",
    "SecurityBERTBaseline",
]
