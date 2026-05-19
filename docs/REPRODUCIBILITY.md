# Reproducibility Guide

This guide walks through every command used to reproduce the numerical
artefacts of the paper (Tables II and III, Figures 3 through 6) and
explains how to register the trained checkpoint as Model #14 of the
RobustIDPS.ai v3 web application.

The reference wall-clock times are measured on a single
NVIDIA A100 80 GB GPU paired with a 32-core AMD EPYC 7763 host and
512 GB of DDR4 ECC memory. CPU-only execution is supported but roughly
30 times slower.

---

## 0. One-time setup

```bash
git clone https://github.com/rogerpanel/CV.git
cd CV/ssl_graph_anomaly
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -e .[full,dev]
```

The `[full]` extras pull PyTorch Geometric; `[dev]` pulls pytest,
black, ruff, and mypy. The reproducible Docker image is in
[`docker/Dockerfile`](../docker/Dockerfile).

```bash
# inside the repo root
docker build -t ssl-graphanomaly:latest -f docker/Dockerfile .
docker run --rm --gpus all -it -v $PWD:/workspace ssl-graphanomaly:latest
```

Download the three benchmark suites (Kaggle credentials required, see
[`docs/DATASETS.md`](DATASETS.md)):

```bash
python scripts/download_datasets.py --suite iis3d   --out data/iis3d
python scripts/download_datasets.py --suite ics3d   --out data/ics3d
python scripts/download_datasets.py --suite ids_pqc --out data/ids_pqc
```

Approximate disk footprint: ~28 GB across the three suites.

---

## 1. Table II - component ablation (paper section IV-B)

```bash
python scripts/run_ablation.py \
    --config configs/iis3d.yaml \
    --out results/ablation_iis3d.csv

python scripts/run_ablation.py \
    --config configs/ics3d.yaml \
    --out results/ablation_ics3d.csv

python scripts/run_ablation.py \
    --config configs/ids_pqc.yaml \
    --out results/ablation_ids_pqc.csv
```

Expected wall-clock per suite: ~6 h on one A100 (7 variants x 5 seeds,
about ten minutes each).

The output CSV has columns `variant, seed, macro_f1, coverage,
set_size`. Aggregating to `mean +/- std` across the five seeds gives
the seven rows of Table II.

---

## 2. Table III - baseline comparison (paper section IV-C)

```bash
python scripts/train_ssl.py            --config configs/iis3d.yaml --stage all
python scripts/calibrate_conformal.py  --config configs/iis3d.yaml
python scripts/evaluate.py \
    --config configs/iis3d.yaml \
    --report results/baselines_iis3d.json \
    --include_baselines
```

Expected wall-clock: ~50 min training, ~5 min calibration, ~25 min
evaluation including the four open baselines plus SecurityBERT
(~80 min total per suite). Repeat for `ics3d` and `ids_pqc`.

The resulting JSON contains a `ssl_graph_anomaly` block and a
`baselines.{kitsune, egraphsage, anomale, rtids, securitybert}` block.
Each block carries Macro-F1, AUROC, ECE, coverage, set-size, and
latency in milliseconds.

---

## 3. Figure 3 - six-axis radar

```bash
python scripts/plot_radar.py \
    --results results/baselines_iis3d.json \
    --out figures/fig3_radar_iis3d.png \
    --latency_ref 1.0 --params_ref 1.0
```

Wall-clock: <5 s. Repeat for the other two suites.

---

## 4. Figure 4 - adversarial robustness (FGSM, PGD-20)

```bash
python scripts/run_adversarial.py \
    --config configs/iis3d.yaml \
    --out results/adv_iis3d.csv
```

Expected wall-clock: ~45 min per suite (two methods x five epsilons,
PGD-20 dominates the cost). Plot with your favourite tool; the CSV
columns are `method, epsilon, macro_f1, coverage, set_size, asr,
latency_ms`.

---

## 5. Figure 5 - drift recovery

```bash
python scripts/run_drift.py \
    --config configs/iis3d.yaml \
    --inject_at 60 --window 180 \
    --out results/drift_iis3d.csv
```

Expected wall-clock: ~10 min per suite (the synthetic injection
amortises the calibration). The output CSV has columns
`minute, far_no_aci, far_with_aci`.

---

## 6. Figure 6 - coverage / set-size sweep

```bash
python scripts/run_coverage_sweep.py \
    --config configs/iis3d.yaml \
    --out results/coverage_iis3d.csv
```

Expected wall-clock: ~30 min per suite (5 alphas x 2 modes x 5 seeds).
CSV columns: `mode, alpha, coverage, set_size, macro_f1`.

---

## 7. Whole-pipeline rerun

The wrapper below runs sections 1 through 6 in sequence on all three
suites; expect ~30 hours on a single A100.

```bash
for suite in iis3d ics3d ids_pqc; do
    python scripts/run_ablation.py            --config configs/${suite}.yaml --out results/ablation_${suite}.csv
    python scripts/train_ssl.py               --config configs/${suite}.yaml --stage all
    python scripts/calibrate_conformal.py     --config configs/${suite}.yaml
    python scripts/evaluate.py                --config configs/${suite}.yaml --report results/baselines_${suite}.json --include_baselines
    python scripts/plot_radar.py              --results results/baselines_${suite}.json --out figures/fig3_radar_${suite}.png
    python scripts/run_adversarial.py         --config configs/${suite}.yaml --out results/adv_${suite}.csv
    python scripts/run_drift.py               --config configs/${suite}.yaml --inject_at 60 --window 180 --out results/drift_${suite}.csv
    python scripts/run_coverage_sweep.py      --config configs/${suite}.yaml --out results/coverage_${suite}.csv
done
```

---

## 8. RobustIDPS.ai v3 integration - Model #14

The detector is published as **Model #14** in the RobustIDPS.ai v3
model registry. The web application lives at
[`../robustidps_web_app/`](../../robustidps_web_app/) (relative to the
package root). To register a freshly trained checkpoint:

1. **Locate the checkpoint.** After step 2 above, the trained weights
   sit at
   `outputs/iis3d/ckpts/ssl_graph_anomaly_best.pt` and the conformal
   calibrator at `outputs/iis3d/calibrator.pkl`.
2. **Copy artefacts into the web app's model registry.**
   ```bash
   cp outputs/iis3d/ckpts/ssl_graph_anomaly_best.pt \
       ../robustidps_web_app/models/14_ssl_graph_anomaly/weights.pt
   cp outputs/iis3d/calibrator.pkl \
       ../robustidps_web_app/models/14_ssl_graph_anomaly/calibrator.pkl
   cp configs/iis3d.yaml \
       ../robustidps_web_app/models/14_ssl_graph_anomaly/config.yaml
   ```
3. **Update the model registry index.** Append an entry to
   `../robustidps_web_app/models/registry.json`:
   ```json
   {
     "id": 14,
     "name": "SSL-GraphAnomaly",
     "module": "ssl_graph_anomaly.inference.serve",
     "weights": "models/14_ssl_graph_anomaly/weights.pt",
     "calibrator": "models/14_ssl_graph_anomaly/calibrator.pkl",
     "config": "models/14_ssl_graph_anomaly/config.yaml",
     "endpoint": "/api/v1/detect"
   }
   ```
4. **Run the API.** From inside `ssl_graph_anomaly/`:
   ```bash
   python -m ssl_graph_anomaly.inference.serve \
       --config configs/iis3d.yaml --port 8014
   ```
   The web app reverse-proxies this on `/api/v1/models/14/detect`.
5. **Smoke-test the deployment.**
   ```bash
   curl -s http://localhost:8014/api/v1/detect \
       -H 'Content-Type: application/json' \
       -d @../robustidps_web_app/tests/sample_flow.json | jq
   ```
   The expected response payload schema is shown in section 8 of the
   [main README](../README.md).

The Docker Compose recipe combining the API and the NGINX reverse
proxy is in [`docker/docker-compose.yml`](../docker/docker-compose.yml).
