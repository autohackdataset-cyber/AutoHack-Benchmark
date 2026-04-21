#!/usr/bin/env bash
set -e

# =============================================================================
# run_all.sh — Download dataset from Zenodo and run the full pipeline
# =============================================================================
# Usage:
#   docker run --rm -v $(pwd)/Result:/artifact/Result <image-name>
#
# Environment variables (override defaults):
#   ZENODO_DOI   — Zenodo record ID (e.g. 12345678)
#   ZENODO_FILE  — filename to download from the record
# =============================================================================

ZENODO_DOI="${ZENODO_DOI:-19661007}"
ZENODO_FILE="${ZENODO_FILE:-Autohack2025_Dataset.zip}"
ZENODO_URL="https://zenodo.org/records/${ZENODO_DOI}/files/${ZENODO_FILE}"

# ---------------------------------------------------------------------------
# 1. Download dataset
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Step 1: Downloading dataset from Zenodo"
echo "  URL : ${ZENODO_URL}"
echo "============================================================"

if [ ! -f "${ZENODO_FILE}" ]; then
    wget --progress=bar:force:noscroll \
         -O "${ZENODO_FILE}" \
         "${ZENODO_URL}"
else
    echo "  Dataset archive already exists, skipping download."
fi

# ---------------------------------------------------------------------------
# 2. Extract dataset
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 2: Extracting dataset"
echo "============================================================"

if [ ! -d "Autohack2025_Dataset" ]; then
    unzip -q "${ZENODO_FILE}"
    echo "  Extracted to project root."
else
    echo "  Dataset directory already exists, skipping extraction."
fi

# ---------------------------------------------------------------------------
# 3. Preprocess (standard 8-feature)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 3: Preprocessing (8-feature)"
echo "============================================================"
python preprocess/preprocess.py

# ---------------------------------------------------------------------------
# 4. Preprocess (38-feature)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 4: Preprocessing (38-feature)"
echo "============================================================"
python preprocess/preprocess38f.py

# ---------------------------------------------------------------------------
# 5. Observation 1 — Baseline RF performance 
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 5: Observation 1"
echo "============================================================"
python observation_code/observation1.py

# ---------------------------------------------------------------------------
# 6. Observation 2 — Without UDS 
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 6: Observation 2"
echo "============================================================"
python observation_code/observation2.py

# ---------------------------------------------------------------------------
# 7. Observation 3 — Multi-bus analysis
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 7: Observation 3"
echo "============================================================"
python observation_code/observation3.py

echo ""
echo "============================================================"
echo "All steps completed. Results saved under Result/"
echo "============================================================"
