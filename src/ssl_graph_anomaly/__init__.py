"""SSL-GraphAnomaly: self-supervised graph intrusion detection with
conformal safety certification.

Reference implementation released under MIT License from
https://github.com/rogerpanel/CV/tree/main/ssl_graph_anomaly
"""

__version__ = "1.0.0"
__author__ = (
    "Roger Nick Anaedevha, "
    "Alexander Gennadievich Trofimov, "
    "Yuri Vladimirovich Borodachev"
)

from ssl_graph_anomaly.models.ssl_graph_anomaly import SSLGraphAnomaly

__all__ = ["SSLGraphAnomaly", "__version__", "__author__"]
