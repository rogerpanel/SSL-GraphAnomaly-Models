# Architecture

This document walks through every module in the order in which it
appears in the paper's pipeline (Figure 1, stages 1-7). Module paths are
relative to [`src/ssl_graph_anomaly/`](../src/ssl_graph_anomaly/).

## ASCII data-flow diagram

```
   raw CSV flows
        |
        v
+-----------------+   stage 1 -- preprocessing (z-score, clip, log1p)
| data/           |   data/preprocessing.py
| preprocessing   |
+-----------------+
        |
        v
+-----------------+   stage 2 -- temporal host-graph window
| data/           |   data/graph_builder.py
| graph_builder   |   (60-second sliding window)
+-----------------+
        |
        v
+-----------------+   stage 3 -- E-GraphSAGE / attention-gated encoder
| models/         |   models/egraphsage.py
| encoder         |   models/attention_gated.py (streaming)
+-----------------+
        |
        v   z (host embedding)
+-----------------+   stage 4 -- discrepancy-aware Transformer AE
| models/         |   models/transformer_ae.py
| transformer_ae  |
+-----------------+
        |
        v   (reconstruction, contrastive views)
+-----------------+   stage 5 -- joint SSL objective
| losses/         |   losses/__init__.py
+-----------------+   (InfoNCE + reconstruction + energy + distill)
        |
        v
+-----------------+   stage 6 -- Mahalanobis energy + logit push
| models/         |   models/mahalanobis.py
| mahalanobis     |   models/energy_head.py
| energy_head     |
+-----------------+
        |
        v
+-----------------+   stage 7 -- conformal certification + drift monitor
| conformal/      |   conformal/split.py
| split / mondrian|   conformal/mondrian.py
| aci / drift     |   conformal/aci.py
+-----------------+   conformal/drift_monitor.py
        |
        v
   prediction set, anomaly energy, abstention flag
```

## Stage 1. Data ingestion and preprocessing

* [`data/datasets.py`](../src/ssl_graph_anomaly/data/datasets.py) -
  `FlowDataset` / `build_dataset(...)` load one of the three benchmark
  suites and expose the canonical 83-feature, 34-class interface.
  `CICFLOWMETER_FEATURES` and `ATTACK_TAXONOMY_34` are the schema
  constants used everywhere downstream.
* [`data/preprocessing.py`](../src/ssl_graph_anomaly/data/preprocessing.py)
  applies z-score standardisation, quantile clipping at 0.999, and a
  `log1p` transform on the four heavy-tailed throughput / packet-count
  columns.
* [`data/augmentations.py`](../src/ssl_graph_anomaly/data/augmentations.py)
  implements the SSL augmentation policy used by stages 3 and 5:
  Gaussian feature jitter, feature dropout, masking.

## Stage 2. Temporal host-graph window

* [`data/graph_builder.py`](../src/ssl_graph_anomaly/data/graph_builder.py)
  builds a `(V, E)` graph for every 60-second window, with hosts as
  vertices and the 83-feature flow records as edge attributes. We cap
  the per-window edge count at `cfg["graph"]["max_edges_per_window"]`
  and the per-node neighbour count at
  `cfg["graph"]["max_neighbours"]`.

## Stage 3. Encoder

Two interchangeable encoder variants share the `z = encode(x)`
contract:

* [`models/egraphsage.py`](../src/ssl_graph_anomaly/src/ssl_graph_anomaly/models/egraphsage.py) -
  `EGraphSAGEEncoder` faithfully implements Lo et al. (2022) over edge
  features, with `aggregator in {mean, max, pool}`.
* [`models/attention_gated.py`](../src/ssl_graph_anomaly/src/ssl_graph_anomaly/models/attention_gated.py) -
  `AttentionGatedEncoder` is the batched streaming variant used at
  inference time when there is no graph context available.

## Stage 4. Discrepancy-aware Transformer autoencoder

* [`models/transformer_ae.py`](../src/ssl_graph_anomaly/src/ssl_graph_anomaly/models/transformer_ae.py) -
  `DiscrepancyTransformerAE` tokenises the host embedding into `K=8`
  learned tokens, runs `L=2` Transformer layers with `H=4` heads, and
  reconstructs the original embedding. The per-sample squared
  reconstruction error is exposed as a discrepancy signal feeding both
  the SSL objective and the Mahalanobis fit.

## Stage 5. Joint SSL objective

* [`losses/__init__.py`](../src/ssl_graph_anomaly/losses/__init__.py)
  exports:
  - `InfoNCELoss` (contrastive, temperature `tau=0.5`),
  - `ReconstructionLoss` (per-sample MSE),
  - `EnergyMarginLoss` (margin `m=1.0`, pushes attack energies up),
  - `DistillationLoss` (T=2 KL between student and teacher logits),
  - `compute_total_loss(components, cfg)` which combines all four with
    weights `(rec, contr, energy, distill) = (1.0, 1.0, 0.5, 0.25)`.

## Stage 6. Mahalanobis energy and logit pushing

* [`models/mahalanobis.py`](../src/ssl_graph_anomaly/src/ssl_graph_anomaly/models/mahalanobis.py) -
  `MahalanobisEnergy` fits `mu` and `Sigma^-1` on benign-only
  embeddings and returns the squared Mahalanobis distance as the
  anomaly energy.
* [`models/energy_head.py`](../src/ssl_graph_anomaly/src/ssl_graph_anomaly/models/energy_head.py) -
  `EnergyLogitHead` subtracts a weighted energy term from the
  benign-class logit and adds it to every attack-class logit, biasing
  predictions toward an attack class when energy is high.

## Stage 7. Conformal certification + drift monitor

* [`conformal/split.py`](../src/ssl_graph_anomaly/conformal/split.py) -
  `SplitConformal` implements vanilla split-conformal regression /
  classification with calibration quantile `q_{1-alpha}`.
* [`conformal/mondrian.py`](../src/ssl_graph_anomaly/conformal/mondrian.py) -
  `MondrianConformal` computes one quantile per class for
  class-conditional coverage.
* [`conformal/aci.py`](../src/ssl_graph_anomaly/conformal/aci.py) -
  `AdaptiveConformalInference` from Gibbs and Candes (2021) updates
  `alpha_t <- alpha_t + gamma * (err_t - alpha)` to keep coverage on
  target under distribution drift.
* [`conformal/drift_monitor.py`](../src/ssl_graph_anomaly/conformal/drift_monitor.py) -
  `DriftMonitor` runs a sliding Kolmogorov-Smirnov test plus a False
  Alarm Rate window check; triggers a re-calibration request.
* [`conformal/__init__.py`](../src/ssl_graph_anomaly/conformal/__init__.py)
  exposes `conformal_predict(...)` as the convenience function used by
  the inference server.

## Top-level orchestration

* [`models/ssl_graph_anomaly.py`](../src/ssl_graph_anomaly/src/ssl_graph_anomaly/models/ssl_graph_anomaly.py) -
  `SSLGraphAnomaly` ties stages 3 to 7 together, with a `mode in
  {graph, stream}` switch selecting between the GNN and the streaming
  variant.
* [`training/`](../src/ssl_graph_anomaly/training/) - SSL pre-training,
  classification-head distillation, conformal calibration.
* [`evaluation/`](../src/ssl_graph_anomaly/evaluation/) -
  Macro-F1, AUROC, ECE, coverage, set-size, latency metrics, plus
  adversarial FGSM / PGD-20 sweep.
* [`inference/serve.py`](../src/ssl_graph_anomaly/inference/serve.py) -
  FastAPI service exposing the `/api/v1/detect` endpoint described in
  the README.
* [`baselines/`](../src/ssl_graph_anomaly/baselines/) - Kitsune,
  E-GraphSAGE, Anomal-E, RTIDS, SecurityBERT wrappers used by Table III.
