# AutoHack: A Physically Verified Multi-Bus CAN Dataset for Intrusion Detection System Evaluation (Artifact)

This repository contains the artifact implementation for the paper **“AutoHack: A Physically Verified Multi-Bus CAN Dataset for Intrusion Detection System Evaluation.”**  
It provides the reference preprocessing and benchmark scripts used to reproduce the baseline IDS results and the main observations reported in the paper.

> **Dataset DOI**  
> The AutoHack dataset is archived on Zenodo: [10.5281/zenodo.19661007](https://doi.org/10.5281/zenodo.19661007)

---

## For Artifact Evaluators

This artifact is intended to support reproduction of the paper’s benchmark results using the AutoHack dataset.

The recommended evaluation path is:

1. Clone the repository
2. Build the Docker image (or set up the local environment)
3. Run the full pipeline
4. Inspect outputs under `Result/`

This artifact reproduces the following paper results:

- **Observation 1** → Random Forest baseline results
- **Observation 2** → Training without UDS messages
- **Observation 3** → Single-bus vs. multi-bus benchmark analysis

**Target badges.** This artifact targets the **Functional** and **Reproduced** badges in the Artifact Evaluation process.

---

# 1. Getting Started

## 1.1 System Requirements

### Hardware

The artifact was tested on a machine with the following configuration:

- **CPU:** Intel Core i9-13900K
- **RAM:** 32 GB
- **Storage:** More than 2 TB available
- **GPU:** NVIDIA GeForce RTX 2070 SUPER (8 GB) present, but **not required**

For artifact evaluation, we recommend the following minimum hardware configuration:

- **CPU:** Multi-core CPU, with **8 or more physical cores recommended**
- **RAM:** **32 GB recommended**
- **Disk:** **At least 30 GB of free space required**
- **Storage type:** **SSD strongly recommended**

The artifact is CPU-based. No GPU acceleration is required.

Observation 3 uses parallel training with `n_jobs=-1`, which means that the training process uses **all available CPU cores**.  
As a result, runtime depends strongly on the number of available CPU cores and storage performance.

### Software

The primary tested environment is:

- **Operating system:** Ubuntu 22.04 LTS
- **Python:** Python 3.11.9

> **OS support note.**
> This artifact has been tested on **Linux (Ubuntu 22.04 LTS)** and **Windows 10/11**.  
> **macOS is not actively supported** because some pinned dependencies (notably XGBoost CPU builds and certain wheel formats in `requirements.txt`) are not optimized for Apple Silicon, and the 38-feature preprocessing has not been validated on macOS.  
> **Linux or Windows execution is strongly recommended.**

---

## 1.2 Clone the Repository

Before either Docker or local execution, clone the repository to your machine.

#### Linux

```bash
git clone https://github.com/autohackdataset-cyber/AutoHack-Benchmark.git
cd AutoHack-Benchmark
```

#### Windows (PowerShell)

```powershell
git clone https://github.com/autohackdataset-cyber/AutoHack-Benchmark.git
cd AutoHack-Benchmark
```

All subsequent commands in this README assume that you start from inside the cloned repository (`AutoHack-Benchmark/`) and `cd` into the appropriate sub-folder where indicated.

---

## 1.3 Setup with Docker

> Docker handles all Python dependencies internally during the image build (`pip install -r requirements.txt` is run inside the `Dockerfile`).  
> You do **not** need to install Python packages on the host when using this path.

### Step 1. Build the Docker image

Run from the repository root.

#### Linux

```bash
cd AutoHack-Benchmark
docker build -t autohack-benchmark .
```

#### Windows (PowerShell)

```powershell
cd AutoHack-Benchmark
docker build -t autohack-benchmark .
```

### Step 2. Configure container resources (important — read before Step 3)

> **This step is required.** Running the full pipeline with Docker’s default resource limits will almost certainly cause an out-of-memory (OOM) crash during 38-feature preprocessing or Observation 3, since the pipeline loads multi-million-row tables into RAM and trains Random Forest / XGBoost models in parallel.

You need to ensure that **both** of the following are satisfied:

1. The Docker daemon / backend itself has enough RAM and CPUs allocated.
2. The `docker run` command requests those resources via `--memory`, `--cpus`, and `--shm-size`.

#### (a) Allocate resources to the Docker backend

**Linux (native Docker).**  
Docker uses host RAM and CPUs directly. No additional configuration is needed; you can skip to (b).

**Windows (Docker Desktop with WSL2 backend).**  
Docker Desktop on Windows runs containers inside a WSL2 VM, and that VM has its **own** memory limit independent of `--memory` flags.

> ⚠️ **Important for Windows users — this is the most common cause of OOM failures on this artifact.**  
> Even if the host machine has 32 GB of RAM, Docker Desktop with the WSL2 backend allocates only about **50% of host RAM (typically ~15–16 GB) by default**. Running the full pipeline under this default limit will cause severe swap thrashing and will likely OOM during 38-feature preprocessing or Observation 3, **even if you pass `--memory=28g` to `docker run`** — the container cannot exceed the WSL2 VM’s own limit.

**Step (i). Check your current WSL2 memory limit.**  
Open PowerShell and run:

```powershell
wsl -d docker-desktop -- free -h
```

Look at the `Mem: total` value in the output. For example:

```text
              total        used        free      shared  buff/cache   available
Mem:          15.5G      775.9M       14.5G        2.2M      254.2M       14.5G
```

If `Mem: total` is **less than 24 GB**, you must increase the WSL2 limit before proceeding with Step (iii).

**Step (ii). Raise the WSL2 limit via `.wslconfig`.**  
Create or edit the file `%UserProfile%\.wslconfig` (e.g. `C:\Users\<your-name>\.wslconfig`). The easiest way is to open it directly in Notepad from PowerShell:

```powershell
notepad $env:USERPROFILE\.wslconfig
```

Paste the following content and save:

```ini
[wsl2]
memory=28GB
processors=20
swap=8GB
```

Adjust the values to match your hardware: leave at least **4 GB of RAM for the Windows host**, so on a 32 GB machine use `memory=28GB`, on a 24 GB machine use `memory=20GB`, and so on. `processors` should not exceed your host CPU core count.

Then restart WSL and Docker Desktop. In PowerShell:

```powershell
wsl --shutdown
```

After a moment, relaunch Docker Desktop from the Start menu or system tray.

**Step (iii). Verify that the new limit is active.**  
Run the check command again:

```powershell
wsl -d docker-desktop -- free -h
```

`Mem: total` should now show approximately the value you set in `.wslconfig` (e.g. `28G`). Only then proceed to section (b).

#### (b) Pass resource flags to `docker run`

Use the flags below for **every** invocation of `docker run autohack-benchmark` in the following steps. The recommended values match the reference machine used in this README:

| Flag | Recommended value | Purpose |
|---|---|---|
| `--memory` | `28g` (≥ 24g) | RAM cap for the container |
| `--memory-swap` | `28g` (same as `--memory`) | Disable swap growth beyond `--memory` |
| `--cpus` | `20` (≥ 8) | Number of CPU cores the container may use |
| `--shm-size` | `2g` | Shared-memory size for sklearn parallel workers |

If you have less hardware available, scale these down — but values significantly below `--memory=24g` and `--cpus=8` are likely to OOM or run very slowly.

### Step 3. Run the full pipeline

The following command downloads the dataset from Zenodo, runs preprocessing and all observation scripts, and stores outputs in `Result/`.

#### Linux

```bash
cd AutoHack-Benchmark
docker run --rm \
  --memory=28g --memory-swap=28g --cpus=20 --shm-size=2g \
  -v "$(pwd)/Result:/artifact/Result" \
  autohack-benchmark
```

#### Windows (PowerShell)

```powershell
cd AutoHack-Benchmark
docker run --rm `
  --memory=28g --memory-swap=28g --cpus=20 --shm-size=2g `
  -v "${PWD}/Result:/artifact/Result" `
  autohack-benchmark
```

### Step 4. Check the output directory

All generated outputs are saved to:

```text
AutoHack-Benchmark/Result/
```

### Step 5. (Optional) Keep preprocessed intermediate files on the host

By default, preprocessed CSV/pickle files produced inside the container (under `preprocess/source/`) are discarded when the container exits. If you want to keep them on the host — for example, to inspect them or to skip re-running preprocessing in subsequent runs — mount an additional volume.

#### Linux

```bash
cd AutoHack-Benchmark
docker run --rm \
  --memory=28g --memory-swap=28g --cpus=20 --shm-size=2g \
  -v "$(pwd)/Result:/artifact/Result" \
  -v "$(pwd)/preprocess_cache:/artifact/preprocess/source" \
  autohack-benchmark
```

#### Windows (PowerShell)

```powershell
cd AutoHack-Benchmark
docker run --rm `
  --memory=28g --memory-swap=28g --cpus=20 --shm-size=2g `
  -v "${PWD}/Result:/artifact/Result" `
  -v "${PWD}/preprocess_cache:/artifact/preprocess/source" `
  autohack-benchmark
```

After the run, the preprocessed outputs will be available at:

```text
AutoHack-Benchmark/preprocess_cache/AutoHack/        # 8-feature preprocessing outputs
AutoHack-Benchmark/preprocess_cache/AutoHack_38f/    # 38-feature preprocessing outputs
```

---

## 1.4 Local Execution (Without Docker)

If Docker is not available, the artifact can also be executed directly on a local machine.  
**Linux and Windows are supported. macOS is not recommended** (see the OS support note in Section 1.1).

### Step 1. Install dependencies

#### Linux (Ubuntu 22.04 LTS recommended)

Run from the repository root:

```bash
cd AutoHack-Benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)

Run from the repository root in **PowerShell**:

```powershell
cd AutoHack-Benchmark
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script activation, you may need to allow it once with:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

In **Command Prompt (cmd)** instead of PowerShell:

```cmd
cd AutoHack-Benchmark
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python **3.11.9** is recommended on both platforms.

### Step 2. Download and extract the dataset

Download `Autohack2025_Dataset.zip` from Zenodo ([10.5281/zenodo.19661007](https://doi.org/10.5281/zenodo.19661007)) and extract it into the repository root so that the following structure exists:

```text
AutoHack-Benchmark/
└── Autohack2025_Dataset/
    └── Interface/
        ├── train/
        └── test/
```

#### Linux

```bash
cd AutoHack-Benchmark
wget https://zenodo.org/records/19661007/files/Autohack2025_Dataset.zip
unzip Autohack2025_Dataset.zip
```

#### Windows (PowerShell)

```powershell
cd AutoHack-Benchmark
Invoke-WebRequest -Uri "https://zenodo.org/records/19661007/files/Autohack2025_Dataset.zip" -OutFile "Autohack2025_Dataset.zip"
Expand-Archive -Path "Autohack2025_Dataset.zip" -DestinationPath "."
```

### Step 3. Run preprocessing

Enter the `preprocess/` folder and run each preprocessing script there. Output paths are computed from the script location, so results land under `AutoHack-Benchmark/preprocess/source/` regardless of where you launch from.

#### Linux

```bash
cd AutoHack-Benchmark/preprocess
python preprocess.py        # 8-feature preprocessing
python preprocess38f.py     # 38-feature preprocessing (required for Observation 3)
```

#### Windows (PowerShell)

```powershell
cd AutoHack-Benchmark\preprocess
python preprocess.py        # 8-feature preprocessing
python preprocess38f.py     # 38-feature preprocessing (required for Observation 3)
```

Preprocessed files are written under `AutoHack-Benchmark/preprocess/source/AutoHack/` and `AutoHack-Benchmark/preprocess/source/AutoHack_38f/` and are **retained on the local filesystem by default** — no additional flag is needed to keep them.

### Step 4. Run the observation scripts

Enter the `observation_code/` folder and run each observation script there.

#### Linux

```bash
cd AutoHack-Benchmark/observation_code
python observation1.py
python observation2.py
python observation3.py
```

#### Windows (PowerShell)

```powershell
cd AutoHack-Benchmark\observation_code
python observation1.py
python observation2.py
python observation3.py
```

All outputs are written under `AutoHack-Benchmark/Result/`.

### Step 5. (Optional) Run everything at once

A helper script is provided to run all steps sequentially.

#### Linux

```bash
cd AutoHack-Benchmark
bash run_all.sh
```

#### Windows (Git Bash recommended)

```bash
cd AutoHack-Benchmark
bash run_all.sh
```

If Git Bash is not installed, run the individual `python` commands from Steps 3 and 4 in PowerShell instead.

---

# 2. Directory Structure

```text
AutoHack-Benchmark/
├── preprocess/
│   ├── preprocess.py          # 8-feature preprocessing (per-message)
│   ├── preprocess38f.py       # 38-feature preprocessing (rolling-window)
│   └── source/                # Preprocessed outputs (retained on local runs)
│       ├── AutoHack/
│       └── AutoHack_38f/
├── observation_code/
│   ├── observation1.py        # Observation 1: RF baseline
│   ├── observation2.py        # Observation 2: Training without UDS messages
│   └── observation3.py        # Observation 3: Multi-bus analysis (RF + XGBoost)
├── Result/                    # Generated at runtime
│   ├── Result_ob1/            # Observation 1 outputs
│   │   ├── f1_auc_result.txt
│   │   ├── confusion_matrix.csv
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix.jpg
│   │   └── confusion_matrix.pdf
│   ├── Result_ob2/            # Observation 2 outputs
│   │   ├── f1_auc_result_without_uds.txt
│   │   ├── cm_without_uds.csv
│   │   ├── confusion_matrix_without_uds.png
│   │   ├── confusion_matrix_without_uds.jpg
│   │   └── confusion_matrix_without_uds.pdf
│   └── Result_ob3/            # Observation 3 outputs
│       ├── summary_macro_f1_001.csv
│       ├── top5_feature_importance_001.csv
│       ├── prediction_compare_RF_001.csv
│       ├── overlap_ratio_by_attack_RF_001.csv
│       ├── execution_time_001.txt
│       └── observation3_report_001.txt
├── Dockerfile
├── run_all.sh
├── requirements.txt
└── README.md
```

---

# 3. Dataset

The AutoHack dataset is available from Zenodo:

- **Dataset DOI:** [10.5281/zenodo.19661007](https://doi.org/10.5281/zenodo.19661007)

When running the artifact through Docker, the dataset is downloaded and extracted automatically.  
For local execution, download and extract the dataset manually as described in Section 1.4 Step 2.

Expected directory structure after extraction:

```text
AutoHack-Benchmark/
└── Autohack2025_Dataset/
    └── Interface/
        ├── train/
        └── test/
```

The `train/` and `test/` directories contain the labeled data used for preprocessing and benchmark evaluation.

No pretrained model weights are required.

---

# 4. Reproducing the Paper Results

The repository contains three observation scripts corresponding to the main benchmark analyses in the paper.

Reference runtimes below were measured on a machine with **20 CPU cores** and **28 GB RAM** (Docker `--cpus=20 --memory=28g`).

| Script | Purpose | Main Result Type | Approx. Runtime |
|---|---|---|---|
| Dataset download (Zenodo, 295 MB) | Fetch `Autohack2025_Dataset.zip` | raw archive | ~12 min |
| Dataset extraction (unzip) | Extract archive into repo root | raw dataset | ~0.1 min |
| `preprocess/preprocess.py` | 8-feature preprocessing | intermediate CSV / processed inputs | ~1.1 min |
| `preprocess/preprocess38f.py` | 38-feature preprocessing (prerequisite for Observation 3) | intermediate CSV / pickle files | ~75.7 min |
| `observation_code/observation1.py` | RF baseline evaluation | per-attack metrics, confusion matrix | ~7.7 min |
| `observation_code/observation2.py` | RF evaluation without UDS messages | false-positive analysis, metrics | ~4.9 min |
| `observation_code/observation3.py` | Single-bus vs. multi-bus training with RF/XGBoost | Macro-F1 summary, feature-importance analysis | ~24.7 min |
| **Full pipeline (end-to-end)** | All steps above, including dataset download | — | **~126.2 min (≈ 2h 6m )** |

**Note.** Observation 3 and 38-feature preprocessing dominate total runtime. Runtime scales with the number of CPU cores and storage I/O throughput.

---

# 5. How to Inspect the Outputs

All outputs are written to the `Result/` directory mounted from the host system.

After successful execution, reviewers should inspect the generated results as follows.  
**Note on numeric reproducibility.** Random Forest and XGBoost training are not fully deterministic across different CPU counts / library versions, so the exact numbers reviewers observe may differ slightly from the reference values below. The qualitative patterns (ordering of classes, direction of change, sign of Delta) are what should be verified.

## 5.1 Observation 1 — RF Baseline

**File to inspect:** `Result/Result_ob1/f1_auc_result.txt`

Reviewers should verify that **Spoofing and Replay show noticeably lower F1-scores** than the other attack classes. Expected output (paper reference values):

```text
Attack Type      Precision     Recall   F1-score        AUC
------------------------------------------------------------
Normal              0.9940     0.9928     0.9934     0.9952
DoS                 1.0000     1.0000     1.0000     1.0000
Spoofing            0.8328     0.7449     0.7864     0.9908
Replay              0.6498     0.7128     0.6799     0.9854
Fuzzing             0.9706     0.9943     0.9823     0.9999
UDS_Spoofing        0.9498     0.9126     0.9308     0.9992
------------------------------------------------------------
```

Key reproducibility check: **Spoofing F1 ≈ 0.79 and Replay F1 ≈ 0.68** — both substantially below the F1-scores of Normal, DoS, Fuzzing, and UDS_Spoofing (all ≥ 0.93). This confirms the baseline difficulty and supports the observation that Spoofing and Replay attacks are the hardest classes for the baseline IDS.

The confusion matrix files `confusion_matrix.{png,jpg,pdf,csv}` should show the pattern described in the paper, with **Spoofing → Normal (~7,770)** and **Replay → Normal (~8,784)** as the dominant misclassifications, and **Normal → Replay (~13,051)** as the largest false-positive bucket.

## 5.2 Observation 2 — Training without UDS Messages

### (a) Compare metric tables with and without UDS

**Files to compare:**
- `Result/Result_ob1/f1_auc_result.txt` (with UDS)
- `Result/Result_ob2/f1_auc_result_without_uds.txt` (without UDS)

Reviewers should verify that the per-attack metrics (Precision, Recall, F1-score, AUC) for **Normal, DoS, Spoofing, Replay, and Fuzzing** are **not meaningfully different** between the two files. Removing UDS messages from training should leave the other classes’ aggregate performance essentially unchanged (Macro F1 ≈ 0.86, Weighted F1 ≈ 0.98); the effect is concentrated in the confusion pattern between Normal and Fuzzing (see below).

Expected output without UDS (paper reference values):

```text
Attack Type      Precision     Recall   F1-score        AUC
------------------------------------------------------------
Normal              0.9933     0.9916     0.9925     0.9948
DoS                 0.9999     1.0000     1.0000     1.0000
Spoofing            0.8003     0.6187     0.6979     0.9898
Replay              0.6465     0.7122     0.6777     0.9846
Fuzzing             0.9074     0.9946     0.9490     0.9999
------------------------------------------------------------
Macro Avg           0.8695     0.8634     0.8634     0.9938
Weighted Avg        0.9848     0.9844     0.9844     0.9938
```

### (b) Compare confusion matrices with and without UDS

**Files to compare:**
- `Result/Result_ob1/confusion_matrix.jpg` (with UDS)
- `Result/Result_ob2/confusion_matrix_without_uds.jpg` (without UDS)

Reviewers should compare the two confusion matrices and verify that, after removing UDS messages from training, **the Normal → Fuzzing misclassification count shows a noticeably larger increase than any other class-pair**, while the remaining off-diagonal entries change only slightly.

Expected change (paper reference values):

| Misclassification | With UDS (Result_ob1) | Without UDS (Result_ob2) |
|---|---|---|
| **Normal → Fuzzing** | **966** | **5,296** |
| Normal → Spoofing | 2,346 | 1,638 |
| Normal → Replay | 13,051 | 14,440 |
| Spoofing → Normal | 7,770 | 6,631 |
| Replay → Normal | 8,784 | 10,204 |

The disproportionately large change in the Normal → Fuzzing cell — relative to the modest changes in other off-diagonal entries — is the core finding of Observation 2: UDS messages act as an implicit regularizer against Fuzzing false positives by providing non-periodic benign traffic that prevents the model from over-associating irregularity with malicious activity. The raw matrix values are in `cm_without_uds.csv`.

## 5.3 Observation 3 — Single-bus vs. Multi-bus

**Primary file to inspect:** `Result/Result_ob3/observation3_report_001.txt`

Reviewers should verify three findings.

### (a) Multi-bus training causes Macro-F1 degradation across **every** bus

Expected summary (paper reference values from Table 10):

```text
1. Macro-F1 Summary (Single-bus vs. Multi-bus)
------------------------------------------------------------------------
Model     Bus   Single-bus   Multi-bus      Delta
   RF   B-CAN       0.8411      0.6840    -0.1572
   RF   C-CAN       0.8368      0.7299    -0.1069
   RF   P-CAN       0.8880      0.7775    -0.1104
XGBoost  B-CAN      0.7879      0.7339    -0.0540
XGBoost  C-CAN      0.8613      0.7505    -0.1109
XGBoost  P-CAN      0.8236      0.7791    -0.0445
```

Check: **every Delta value is negative** for both RF and XGBoost across all three buses (B-CAN, C-CAN, P-CAN). This confirms the cross-bus domain-shift effect.

### (b) Feature importance is **not meaningfully changed** between single-bus and multi-bus

Inspect the “Top-5 Feature Importance per Experiment” section of `observation3_report_001.txt` (also saved as `top5_feature_importance_001.csv`). Reviewers should verify that the top-ranked features are substantially the same between the single-bus and multi-bus configurations. In particular, ID-aware temporal and frequency features such as **`ID_Prev_Interval`** and **`ID_Frequency`** should consistently remain among the most important features in both regimes — i.e., the degradation in (a) is **not** explained by a shift in which features the model relies on.

### (c) Overlap-ID analysis shows class-dependent patterns

Inspect Section **3.1 Overlap Ratio by Attack Class** of `observation3_report_001.txt` (also saved as `overlap_ratio_by_attack_RF_001.csv`). Expected values (paper reference values from Table 11, Random Forest):

| Class | r_test (Overlap_Ratio_All_%) | r_mis (Overlap_Ratio_MultiMisclassified_%) | Direction |
|---|---|---|---|
| Normal | 46.12% | 53.14% | ↑ |
| Spoofing | 25.47% | 25.75% | ≈ |
| Replay | 41.20% | 38.52% | ↓ |

Reviewers should confirm the following per-class pattern of `Overlap_Ratio_MultiMisclassified_%` relative to `Overlap_Ratio_All_%`:

- **Normal:** **increased** (~46% → ~53%, ↑)
- **Spoofing:** roughly **similar** (~25% ≈ ~26%, ≈)
- **Replay:** **decreased** (~41% → ~39%, ↓)

DoS, Fuzzing, and UDS are excluded from this comparison by design (DoS yields too few misclassifications, Fuzzing spans the full ID space by construction, and UDS has no overlap IDs in this dataset).

This class-dependent pattern — in particular, the **opposite directions** for Normal and Replay — supports the interpretation that overlapping CAN IDs across buses contribute differently to misclassification depending on attack semantics, and that the multi-bus setting cannot be reduced to a single uniform domain-shift effect.

---

# 6. Expected Output Summary

At a high level, the artifact produces the following types of outputs:

- preprocessed intermediate files (CSV / pickle)
- classification metrics (`f1_auc_result*.txt`)
- confusion matrices (`.csv`, `.png`, `.jpg`, `.pdf`)
- multi-bus benchmark summaries (`summary_macro_f1_001.csv`, `observation3_report_001.txt`)
- feature importance summaries (`top5_feature_importance_001.csv`)
- overlap-ID analysis (`overlap_ratio_by_attack_RF_001.csv`)

---

# 7. Notes for Reviewers

- Both **Docker-based** and **local** execution paths are supported on **Linux and Windows**; macOS is not recommended
- No GPU is required
- The artifact is **I/O-intensive** during preprocessing and **CPU-intensive** during training
- Observation 3 and the 38-feature preprocessing dominate total runtime on systems with fewer CPU cores or slower storage
- **Docker resource limits matter.** Running without `--memory`, `--cpus`, and `--shm-size` flags (or with insufficient backend allocation on Windows/macOS) will cause OOM crashes during 38-feature preprocessing or Observation 3. Follow Section 1.3 Step 2 carefully.
- Preprocessed intermediate files are retained by default on local runs; for Docker runs, mount `preprocess/source` (Section 1.3, Step 5) to keep them on the host
- Random Forest / XGBoost training is not fully deterministic across different CPU counts and library versions; reviewers should focus on qualitative patterns (class ordering, direction of change, negative Deltas) rather than exact numeric matches
- Target AE badges: **Functional** and **Reproduced**
