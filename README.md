# AutoHack Benchmark — Artifact

Reproducible artifact for CAN bus intrusion detection system (IDS) analysis using the AutoHack dataset.  
This repository contains preprocessing scripts and three observation experiments evaluating RF/XGBoost-based attack classification.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19661007.svg)](https://doi.org/10.5281/zenodo.19661007)

---

## Repository Structure

```
.
├── preprocess/
│   ├── preprocess.py          # 7-feature preprocessing (per-message)
│   └── preprocess38f.py       # 38-feature preprocessing (rolling window)
├── observation_code/
│   ├── observation1.py        # Observation 1: RF baseline (Table 8, Figure 3)
│   ├── observation2.py        # Observation 2: Without UDS messages (Figure 4, Table 9)
│   └── observation3.py        # Observation 3: Multi-bus analysis (RF + XGBoost)
├── Dockerfile
├── run_all.sh
└── requirements.txt
```

---

## Dataset

The dataset is available on Zenodo:  
**DOI:** [10.5281/zenodo.19661007](https://doi.org/10.5281/zenodo.19661007)

The dataset will be downloaded and extracted automatically when running via Docker.

Expected structure after extraction:
```
Autohack2025_Dataset/
└── Interface/
    ├── train/   ← *labels.csv files
    └── test/    ← *labels.csv files
```

---

## Quick Start (Docker)

### 1. Build the image

```bash
docker build -t autohack-benchmark .
```

### 2. Run the full pipeline

The container automatically downloads the dataset from Zenodo, runs all preprocessing and observation scripts, and saves results.

```bash
docker run --rm -v "$(pwd)/Result:/artifact/Result" autohack-benchmark
```

Results are saved to `Result/` on your local machine.

### 3. Run a single script

```bash
docker run --rm \
  -v "$(pwd)/Autohack2025_Dataset:/artifact/Autohack2025_Dataset" \
  -v "$(pwd)/Result:/artifact/Result" \
  --entrypoint python autohack-benchmark \
  preprocess/preprocess.py
```

---

## Manual Setup (without Docker)

### Requirements

- Python 3.10+

```bash
pip install -r requirements.txt
```

### Run order

```bash
# 1. Place dataset at ./Autohack2025_Dataset/
# 2. Preprocessing
python preprocess/preprocess.py
python preprocess/preprocess38f.py

# 3. Observations
python observation_code/observation1.py
python observation_code/observation2.py
python observation_code/observation3.py
```

---

## Observations

| Script | Description | Outputs |
|--------|-------------|---------|
| `observation1.py` | RF baseline — per-attack Precision/Recall/F1/AUC | Table 8, Figure 3 (confusion matrix) |
| `observation2.py` | RF trained without UDS messages — false-positive analysis | Figure 4, Table 9 |
| `observation3.py` | Single-bus vs. multi-bus training with RF and XGBoost | Macro-F1 summary, feature importance |

---

## License

See [LICENSE](LICENSE) for details.
