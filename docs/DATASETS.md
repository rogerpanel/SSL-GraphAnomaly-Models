# Datasets

This page documents the three aggregated benchmark suites used in the
paper and their underlying corpora. Each suite is a Kaggle release;
download instructions sit in
[`scripts/download_datasets.py`](../scripts/download_datasets.py).

All three suites carry the harmonised **83-feature CICFlowMeter schema**
and the **34-class** attack taxonomy of Sarhan et al. (2022)
"NetFlow standardisation for network intrusion detection"
([doi:10.1186/s40537-022-00553-y](https://doi.org/10.1186/s40537-022-00553-y)).

The constants `CICFLOWMETER_FEATURES` and `ATTACK_TAXONOMY_34` are
exposed at the top of
[`src/ssl_graph_anomaly/data/__init__.py`](../src/ssl_graph_anomaly/data/__init__.py).

---

## Common preprocessing pipeline

Implemented in
[`src/ssl_graph_anomaly/data/preprocessing.py`](../src/ssl_graph_anomaly/data/preprocessing.py):

1. Parse raw CSV / Parquet flows into 83 CICFlowMeter columns.
2. Apply `log1p` to the four heavy-tailed throughput / packet-count
   features (`TotLen Fwd Pkts`, `TotLen Bwd Pkts`, `Flow Pkts/s`,
   `Flow Byts/s`).
3. Clip every feature at the 0.999 quantile of the training fold.
4. Standardise with z-score statistics fit on benign-only training
   flows.
5. Build a temporal host graph per 60-second window
   ([`data/graph_builder.py`](../src/ssl_graph_anomaly/data/graph_builder.py)).

### Split protocol

| Fold | Fraction | Contents |
|---|---|---|
| `train_benign` | 0.60 | Benign only, used for SSL pre-training. |
| `calibration_benign` | 0.10 | Benign only, used for conformal calibration. |
| `val` | 0.10 | Mixed, used for early stopping and hyper-tuning. |
| `test` | 0.20 | Mixed, the only fold used for reported numbers. |

The split is chronological: a flow's timestamp determines its fold, so
no future information leaks into training.

---

## Suite cards

### IIS3D - Integrated IDPS Security 3 Datasets

* Kaggle slug: `rogerpanel/integrated-idps-security-3-datasets`
* DOI: [`10.34740/kaggle/dsv/12479689`](https://doi.org/10.34740/kaggle/dsv/12479689)
* Underlying corpora:
  - **CSE-CIC-IDS2018** -
    [University of New Brunswick CIC](https://www.unb.ca/cic/datasets/ids-2018.html);
    Sharafaldin, Lashkari, and Ghorbani (2018).
  - **CIC-IoT2023** -
    [University of New Brunswick CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html);
    Neto et al. (2023).
  - **UNSW-NB15** -
    [UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset);
    Moustafa and Slay (2015).
* License: CC BY 4.0 on each upstream release.
* Feature schema: 83 CICFlowMeter columns (full schema in
  [`data/datasets.py:CICFLOWMETER_FEATURES`](../src/ssl_graph_anomaly/data/datasets.py)).
* Class taxonomy: 34 classes including `BenignTraffic`, the eight
  CIC-IDS2018 attack families, the seven CIC-IoT2023 attack families,
  and the nine UNSW-NB15 attack families, deduplicated and remapped
  through the Sarhan 2022 standardisation table.

### ICS3D - Integrated Cloud Security 3 Datasets

* Kaggle slug: `rogerpanel/integrated-cloud-security-3-datasets`
* DOI: [`10.34740/kaggle/dsv/12483891`](https://doi.org/10.34740/kaggle/dsv/12483891)
* Underlying corpora:
  - Microsoft Cloud telemetry (cloud-native attack logs).
  - **Edge-IIoTset** -
    [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications);
    Ferrag et al. (2022).
  - Kubernetes container-security events.
* License: CC BY 4.0.
* Feature schema: same 83-feature CICFlowMeter schema, after the
  preprocessor maps cloud-flow fields onto the canonical columns.
* Class taxonomy: same 34-class taxonomy; ICS-specific labels
  (container escape, lateral movement, secrets exfiltration) are
  remapped onto the closest IIS3D classes.

### IDS-PQC - Encrypted + Post-Quantum-Cryptography IDS

* Kaggle slug: `rogerpanel/ids-encrypted-pqc-datasets`
* DOI: [`10.34740/kaggle/dsv/15424420`](https://doi.org/10.34740/kaggle/dsv/15424420)
* Underlying corpora:
  - **NF-CSE-CIC-IDS2018-v3** -
    [University of Queensland NIDS datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/).
  - CIC-PQC-OAV-2025 (Kyber and Dilithium TLS handshakes).
* License: CC BY 4.0 on NF-CSE-CIC-IDS2018-v3; CC BY-SA 4.0 on
  CIC-PQC-OAV-2025.
* Feature schema: 83 columns; PQC handshake messages are summarised
  into the existing CICFlowMeter slots before z-scoring.
* Class taxonomy: 34 classes; PQC-specific labels (Kyber MITM,
  Dilithium signature spoofing) are folded into the
  `Generic`/`Reconnaissance` slots.

---

## Download instructions

```bash
# 1. install the Kaggle CLI and configure credentials in ~/.kaggle/kaggle.json
pip install kaggle

# 2. download (one suite at a time)
python scripts/download_datasets.py --suite iis3d   --out data/iis3d
python scripts/download_datasets.py --suite ics3d   --out data/ics3d
python scripts/download_datasets.py --suite ids_pqc --out data/ids_pqc
```

Each command unzips into the destination, prints a summary, and is
idempotent.

---

## References

The following primary sources should be cited whenever these data are
used (full BibTeX in
[`REFERENCES.md`](REFERENCES.md)):

* [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) -
  Sharafaldin et al. 2018.
* [CIC-IoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) -
  Neto et al. 2023.
* [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) -
  Moustafa and Slay 2015.
* [Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) -
  Ferrag et al. 2022.
* [NF-CSE-CIC-IDS2018-v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) -
  Sarhan et al.
* [Sarhan 2022 NetFlow standardisation](https://doi.org/10.1186/s40537-022-00553-y).
