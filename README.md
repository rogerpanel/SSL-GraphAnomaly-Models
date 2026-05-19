# SSL-GraphAnomaly

**Self-Supervised Graph Neural Networks for Network Intrusion Detection
with Conformal Safety Certification**

Reference implementation for the manuscript of the same title submitted to the
*IEEE Transactions on Neural Networks and Learning Systems*. This codebase is the
official reproducibility artefact released under the
[`rogerpanel/CV`](https://github.com/rogerpanel/CV) repository (folder
`ssl_graph_anomaly/`).

> Anaedevha, R. N., Trofimov, A. G., & Borodachev, Y. V. (2026).
> *Self-Supervised Graph Neural Networks for Network Intrusion Detection with
> Conformal Safety Certification.* IEEE Transactions on Neural Networks and
> Learning Systems (under review). National Research Nuclear University MEPhI,
> Moscow, Russia.

The system is the **14th registered detector** of the *RobustIDPS.ai* v3
platform (DOI [`10.5281/zenodo.19129512`](https://doi.org/10.5281/zenodo.19129512))
and is meant to be deployed alongside the twelve other models hosted in the
same repository (Neural-ODE, Hierarchical Gaussian Process, MambaShield,
Federated Graph, Stochastic Transformer, Hybrid Stochastic-LLM-Transformer,
Optimal Transport, Stochastic Games Defence, Heterogeneous Graph, Bayesian,
Encrypted-Traffic Device-ID, Boltzmann Machine).

---

## 1. What this codebase delivers

`SSL-GraphAnomaly` combines six interlocking components, each implemented as a
self-contained module so that ablations are reproducible:

| Component | Module | Paper section |
|---|---|---|
| Temporal host graph builder | `data.graph_builder` | §III-A |
| E-GraphSAGE edge-feature encoder | `models.egraphsage` | §III-B |
| Attention-gated streaming variant | `models.attention_gated` | §III-B |
| Discrepancy-aware Transformer autoencoder | `models.transformer_ae` | §III-C |
| Joint contrastive + reconstruction objective | `losses` | §III-D |
| Mahalanobis-centred anomaly energy + energy logit push | `models.mahalanobis`, `models.energy_head` | §III-E |
| Split-conformal, Mondrian, ACI certification | `conformal` | §III-F |
| Drift monitor (Kolmogorov–Smirnov + sliding FAR) | `conformal.drift_monitor` | §III-G |

The classification head exposes the **83-feature, 34-class CICFlowMeter
interface** required by the RobustIDPS.ai platform.

---

## 2. Repository layout

```
ssl_graph_anomaly/
├── README.md                 ← this file
├── LICENSE                   ← MIT, mirrors upstream
├── requirements.txt          ← exact pip-installable dependency list
├── setup.py                  ← editable install entry point
├── configs/                  ← YAML files (one per benchmark)
│   ├── default.yaml
│   ├── iis3d.yaml
│   ├── ics3d.yaml
│   └── ids_pqc.yaml
├── src/ssl_graph_anomaly/
│   ├── data/                 ← loaders, graph builder, augmentations
│   ├── models/               ← encoder, transformer, energy head
│   ├── losses/               ← InfoNCE, reconstruction, energy, KD
│   ├── conformal/            ← split, Mondrian, ACI, drift
│   ├── training/             ← pretraining, distillation, calibration
│   ├── evaluation/           ← Macro-F1/AUROC/ECE/coverage, FGSM/PGD
│   ├── baselines/            ← Kitsune, E-GraphSAGE, Anomal-E, RTIDS
│   ├── inference/            ← streaming + REST serving
│   └── utils/                ← seed, logging, I/O
├── scripts/                  ← CLI entry points
│   ├── download_datasets.py
│   ├── train_ssl.py
│   ├── calibrate_conformal.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   └── run_adversarial.py
├── notebooks/                ← walkthroughs
├── tests/                    ← unit tests for each component
├── docker/Dockerfile         ← CUDA-12.1, PyTorch-2.3 reproducible image
└── docs/                     ← extended documentation
    ├── ARCHITECTURE.md
    ├── DATASETS.md
    ├── HYPERPARAMETERS.md
    ├── REPRODUCIBILITY.md
    └── REFERENCES.md
```

---

## 3. Installation

The project targets Python ≥ 3.10 and PyTorch ≥ 2.3 with CUDA 12.1. A
CPU-only install also works (inference throughput drops to ≈ 2 000 flows/s on
an 8-core machine, as reported in the paper).

```bash
git clone https://github.com/rogerpanel/CV.git
cd CV/ssl_graph_anomaly
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -e .[full]
```

The `[full]` extras pull in `torch-geometric`, `torch-scatter`, and
`torch-sparse`. If those wheels do not match your CUDA version, follow the
[PyG installation matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

A reproducible CUDA container is provided in `docker/Dockerfile`:

```bash
docker build -t ssl-graphanomaly:latest docker/
docker run --rm --gpus all -it -v $PWD:/workspace ssl-graphanomaly:latest
```

---

## 4. Datasets

We use three publicly released aggregated benchmark suites. All carry Kaggle
DOIs and are downloadable through the helper script
`scripts/download_datasets.py` (Kaggle API credentials required).

| Suite | DOI | Underlying corpora |
|---|---|---|
| **IIS3D** — Integrated IDPS Security 3 Datasets | [`10.34740/kaggle/dsv/12479689`](https://doi.org/10.34740/kaggle/dsv/12479689) | CSE-CIC-IDS2018, CIC-IoT2023, UNSW-NB15 |
| **ICS3D** — Integrated Cloud Security 3 Datasets | [`10.34740/kaggle/dsv/12483891`](https://doi.org/10.34740/kaggle/dsv/12483891) | Microsoft Cloud telemetry, Edge-IIoTset, Kubernetes container-security events |
| **IDS-PQC** — Encrypted + Post-Quantum-Cryptography IDS | [`10.34740/kaggle/dsv/15424420`](https://doi.org/10.34740/kaggle/dsv/15424420) | NF-CSE-CIC-IDS2018-v3 + CIC-PQC-OAV-2025 (Kyber/Dilithium TLS handshakes) |

All three suites carry the harmonised **83-feature CICFlowMeter schema** and
the **34-class** attack taxonomy of Sarhan *et al.*
([NetFlow standardisation, 2022](https://doi.org/10.1186/s40537-022-00553-y)).

The full preprocessing pipeline (z-score standardisation, chronological
60/10/10/20 split with benign-only training and calibration folds) is in
`src/ssl_graph_anomaly/data/preprocessing.py` and documented in
[`docs/DATASETS.md`](docs/DATASETS.md).

---

## 5. Quick start

```bash
# 1. download datasets (~28 GB total, requires Kaggle credentials)
python scripts/download_datasets.py --suite iis3d --out data/iis3d

# 2. SSL pretraining on benign flows (30 epochs, AdamW, lr=5e-4, batch=1024)
python scripts/train_ssl.py --config configs/iis3d.yaml

# 3. fit Mahalanobis statistics + prototype distillation
python scripts/train_ssl.py --config configs/iis3d.yaml --stage distill

# 4. conformal calibration (Mondrian class-conditional, alpha=0.05)
python scripts/calibrate_conformal.py --config configs/iis3d.yaml

# 5. evaluate
python scripts/evaluate.py --config configs/iis3d.yaml --report results/iis3d.json

# 6. component ablation (Table II of the paper)
python scripts/run_ablation.py --config configs/iis3d.yaml --out results/ablation.csv

# 7. adversarial robustness sweep (FGSM, PGD-20 at eps in {0,0.01,0.03,0.05,0.10})
python scripts/run_adversarial.py --config configs/iis3d.yaml --out results/adv.csv
```

All seeds default to `{17, 42, 101, 1234, 31337}`, matching the five-seed
protocol reported in §IV of the paper.

---

## 6. Hyperparameters at a glance

| Parameter | Value | Source |
|---|---|---|
| Hidden dim `d` | 128 | §III-B |
| Encoder layers `L` | 2 | §III-B |
| Edge-feature dim `d_e` | 83 (CICFlowMeter) | §III-A |
| Number of classes `C` | 34 | §III-A |
| Transformer tokens `K` | 8 | §III-C |
| Transformer layers / heads | 2 / 4 | §III-C |
| Feed-forward width | `2d` | §III-C |
| InfoNCE temperature `τ` | 0.5 | §III-D |
| Feature dropout rate | 0.10 | §III-D |
| Gaussian jitter σ | 0.05 | §III-D |
| Masking rate | 0.15 | §III-D |
| Energy margin `m` | 1.0 | §III-D |
| Distillation temperature `T` | 2 | §III-D |
| Loss weights (rec, contr, energy, cls) | (1.0, 1.0, 0.5, 0.25) | §III-D |
| Mahalanobis weight `λ` | 0.1 | §III-E |
| Target miscoverage `α` | 0.05 | §III-F |
| ACI step size `γ` | 0.005 | §III-F |
| KS drift threshold | 0.10 | §III-G |
| Optimizer | AdamW | §IV |
| Learning rate | 5 × 10⁻⁴ | §IV |
| Batch size | 1024 | §IV |
| SSL epochs | 30 | §IV |
| Distillation epochs | 10 | §IV |
| Seeds | {17, 42, 101, 1234, 31337} | §IV |

The full sensitivity grid for the four loss weights is in
[`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).

---

## 7. Reproducing every paper figure / table

| Artefact | Command |
|---|---|
| Table II (component ablation) | `python scripts/run_ablation.py` |
| Table III (baseline comparison) | `python scripts/evaluate.py --include_baselines` |
| Fig. 3 (radar plot) | `python scripts/plot_radar.py --results results/baselines.json` |
| Fig. 4 (adversarial curves) | `python scripts/run_adversarial.py` |
| Fig. 5 (drift recovery) | `python scripts/run_drift.py --inject_at 60 --window 180` |
| Fig. 6 (coverage–efficiency) | `python scripts/run_coverage_sweep.py` |

Pre-generated JSON outputs from our reference run are checked in under
`results/` for one-click verification.

---

## 8. Integration with RobustIDPS.ai

The detector is registered as **Model #14** in the RobustIDPS.ai v3 model
registry. The REST entry point is the standard `/api/v1/detect` endpoint
returning a JSON object with:

```json
{
  "predicted_class": "DDoS-PSHACK_Flood",
  "predicted_class_id": 12,
  "prediction_set": ["BenignTraffic", "DDoS-PSHACK_Flood"],
  "anomaly_energy": 4.317,
  "mahalanobis_distance": 12.84,
  "reconstruction_discrepancy": 0.072,
  "conformal_quantile": 3.91,
  "alpha_target": 0.05,
  "coverage_certified": true,
  "abstained": false,
  "model_id": 14,
  "model_name": "SSL-GraphAnomaly"
}
```

The deployment recipe (Docker Compose + NGINX + the existing
`robustidps_web_app` frontend) is in `docs/REPRODUCIBILITY.md`.

---

## 9. Citation

If this code is useful in your own work, please cite the manuscript:

```bibtex
@article{anaedevha2026sslgraphanomaly,
  author    = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich
               and Borodachev, Yuri Vladimirovich},
  title     = {Self-Supervised Graph Neural Networks for Network Intrusion
               Detection with Conformal Safety Certification},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2026},
  note      = {Under review. Reproducibility:
               \url{https://github.com/rogerpanel/CV/tree/main/ssl_graph_anomaly}}
}
```

---

## 10. License

Released under the MIT License — see `LICENSE`.
