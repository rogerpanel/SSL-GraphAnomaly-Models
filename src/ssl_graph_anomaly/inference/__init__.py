"""Online inference and HTTP serving for SSL-GraphAnomaly."""
from ssl_graph_anomaly.inference.serve import serve_api
from ssl_graph_anomaly.inference.streaming import StreamingInference

__all__ = ["StreamingInference", "serve_api"]
