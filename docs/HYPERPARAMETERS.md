# Hyperparameter Reference and Sensitivity Grid

This document complements section 6 of the
[main README](../README.md) and contains:

1. The full hyperparameter table reproduced verbatim from the README
   plus all defaults set in [`configs/default.yaml`](../configs/default.yaml).
2. The sensitivity grid for the four loss weights
   (`reconstruction_weight`, `contrastive_weight`,
   `energy_margin_weight`, `distillation_weight`).
3. The alternate values we explored during development.

---

## 1. Reproduced README table

| Parameter | Value | Source | YAML key |
|---|---|---|---|
| Hidden dim `d` | 128 | III-B | `model.encoder.hidden_dim` |
| Encoder layers `L` | 2 | III-B | `model.encoder.num_layers` |
| Edge-feature dim `d_e` | 83 | III-A | `model.encoder.edge_feature_dim` |
| Number of classes `C` | 34 | III-A | `data.num_classes` |
| Transformer tokens `K` | 8 | III-C | `model.transformer_ae.num_tokens` |
| Transformer layers / heads | 2 / 4 | III-C | `.num_layers`, `.num_heads` |
| Feed-forward width | `2d` | III-C | `.ff_width_multiplier=2` |
| InfoNCE temperature `tau` | 0.5 | III-D | `loss.infonce_temperature` |
| Feature dropout rate | 0.10 | III-D | `loss.feature_dropout` |
| Gaussian jitter `sigma` | 0.05 | III-D | `loss.gaussian_jitter_sigma` |
| Masking rate | 0.15 | III-D | `loss.mask_rate` |
| Energy margin `m` | 1.0 | III-D | `loss.margin` |
| Distillation temperature `T` | 2 | III-D | `loss.distillation_temperature` |
| Reconstruction weight | 1.0 | III-D | `loss.reconstruction_weight` |
| Contrastive weight | 1.0 | III-D | `loss.contrastive_weight` |
| Energy-margin weight | 0.5 | III-D | `loss.energy_margin_weight` |
| Distillation weight | 0.25 | III-D | `loss.distillation_weight` |
| Mahalanobis weight `lambda` | 0.1 | III-E | `model.energy.mahalanobis_lambda` |
| Target miscoverage `alpha` | 0.05 | III-F | `conformal.alpha` |
| ACI step `gamma` | 0.005 | III-F | `conformal.aci.gamma` |
| KS drift threshold | 0.10 | III-G | `conformal.drift.ks_threshold` |
| Optimizer | AdamW | IV | `optim.optimizer` |
| Learning rate | 5e-4 | IV | `optim.learning_rate` |
| Weight decay | 1e-5 | IV | `optim.weight_decay` |
| Gradient clip | 1.0 | IV | `optim.gradient_clip` |
| Batch size | 1024 | IV | `train.batch_size` |
| SSL epochs | 30 | IV | `train.ssl_epochs` |
| Distillation epochs | 10 | IV | `train.distill_epochs` |
| Scheduler | cosine | IV | `optim.scheduler` |
| Warmup steps | 1000 | IV | `optim.warmup_steps` |
| Early stop patience | 5 | IV | `train.early_stop_patience` |
| Seeds | {17, 42, 101, 1234, 31337} | IV | `experiment.seeds` |

---

## 2. Loss-weight sensitivity grid

The four loss-weight parameters live under `cfg["loss"]`:

* `loss.reconstruction_weight` (`w_rec`)
* `loss.contrastive_weight` (`w_contr`)
* `loss.energy_margin_weight` (`w_energy`)
* `loss.distillation_weight` (`w_distill`)

We swept each axis in isolation, holding the other three at their
paper values `(w_rec, w_contr, w_energy, w_distill) = (1.0, 1.0, 0.5,
0.25)`. The grid below is the one reproduced in supplementary Table S1
of the manuscript.

| `w_rec` | `w_contr` | `w_energy` | `w_distill` | Macro-F1 (mean) | Coverage |
|---|---|---|---|---|---|
| 0.00 | 1.0 | 0.50 | 0.25 | 0.812 | 0.949 |
| 0.25 | 1.0 | 0.50 | 0.25 | 0.881 | 0.950 |
| 0.50 | 1.0 | 0.50 | 0.25 | 0.901 | 0.950 |
| 1.00 | 1.0 | 0.50 | 0.25 | **0.916** | 0.951 |
| 2.00 | 1.0 | 0.50 | 0.25 | 0.910 | 0.952 |
| 4.00 | 1.0 | 0.50 | 0.25 | 0.896 | 0.953 |
| 1.00 | 0.00 | 0.50 | 0.25 | 0.844 | 0.949 |
| 1.00 | 0.25 | 0.50 | 0.25 | 0.882 | 0.950 |
| 1.00 | 0.50 | 0.50 | 0.25 | 0.903 | 0.950 |
| 1.00 | 1.00 | 0.50 | 0.25 | **0.916** | 0.951 |
| 1.00 | 2.00 | 0.50 | 0.25 | 0.913 | 0.952 |
| 1.00 | 4.00 | 0.50 | 0.25 | 0.901 | 0.952 |
| 1.00 | 1.00 | 0.00 | 0.25 | 0.871 | 0.948 |
| 1.00 | 1.00 | 0.10 | 0.25 | 0.892 | 0.950 |
| 1.00 | 1.00 | 0.25 | 0.25 | 0.904 | 0.950 |
| 1.00 | 1.00 | 0.50 | 0.25 | **0.916** | 0.951 |
| 1.00 | 1.00 | 1.00 | 0.25 | 0.913 | 0.952 |
| 1.00 | 1.00 | 2.00 | 0.25 | 0.902 | 0.953 |
| 1.00 | 1.00 | 0.50 | 0.00 | 0.889 | 0.950 |
| 1.00 | 1.00 | 0.50 | 0.10 | 0.906 | 0.951 |
| 1.00 | 1.00 | 0.50 | 0.25 | **0.916** | 0.951 |
| 1.00 | 1.00 | 0.50 | 0.50 | 0.911 | 0.951 |
| 1.00 | 1.00 | 0.50 | 1.00 | 0.901 | 0.952 |

The four-axis maximum is reached at the paper value
`(1.0, 1.0, 0.5, 0.25)`; the surface is broad and shallow near the
optimum, so any value within `+/- 50%` of each entry retains the
ranking against the baselines in Table III.

---

## 3. Alternates explored during development

| Setting | Default | Alternates tried | Outcome |
|---|---|---|---|
| `model.encoder.aggregator` | `mean` | `max`, `pool` | `mean` matched the original E-GraphSAGE paper and was 5 to 8% faster than `pool` at equal F1. |
| `model.transformer_ae.num_tokens` | 8 | 4, 16, 32 | F1 plateaus from 8 onward; we picked the smallest plateau point. |
| `model.transformer_ae.num_heads` | 4 | 2, 8 | Heads = 4 is best for 128-d embeddings (32-d per head). |
| `loss.infonce_temperature` | 0.5 | 0.1, 0.2, 1.0 | 0.5 best; lower temperatures collapsed the representation. |
| `loss.margin` | 1.0 | 0.25, 0.5, 2.0 | 1.0 best; the curve is flat between 0.5 and 2.0. |
| `loss.mask_rate` | 0.15 | 0.05, 0.30, 0.50 | 0.15 matches GraphMAE (Hou et al. 2022). |
| `model.energy.mahalanobis_lambda` | 0.10 | 0.01, 0.05, 0.20 | Lambda = 0.10 keeps a clean separation between benign and attack energies. |
| `conformal.alpha` | 0.05 | 0.01, 0.10, 0.15 | See [`run_coverage_sweep.py`](../scripts/run_coverage_sweep.py) for the full sweep. |
| `conformal.aci.gamma` | 0.005 | 0.001, 0.01, 0.05 | Gamma = 0.005 was the smallest value that fully recovered after the 5% drift injection within 60 minutes. |
| `optim.optimizer` | `adamw` | `adam`, `sgd+momentum`, `lion` | AdamW best; Lion was 2% behind but 8% faster - kept as `experimental` only. |
| `optim.learning_rate` | 5e-4 | 1e-4, 1e-3, 3e-3 | 5e-4 best; 1e-3 unstable on UNSW-NB15. |
| `optim.scheduler` | cosine | linear, plateau | Cosine wins by 1.5% F1 over linear at 30 epochs. |
| `train.batch_size` | 1024 | 256, 512, 2048, 4096 | 1024 best given the InfoNCE negative pool size. |

For end-to-end ablations of these knobs see
[`scripts/run_ablation.py`](../scripts/run_ablation.py).
