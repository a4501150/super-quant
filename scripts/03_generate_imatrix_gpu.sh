#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"


echo "=== Generating Importance Matrices (GPU, PyTorch) ==="

# Check calibration data exists
for DOMAIN in ${CALIBRATION_DOMAINS}; do
    CAL_FILE="${CALIBRATION_DIR}/${DOMAIN}.txt"
    if [[ ! -f "${CAL_FILE}" ]]; then
        echo "ERROR: Calibration file not found: ${CAL_FILE}"
        echo "Run: make calibrate"
        exit 1
    fi
    SIZE=$(du -sh "${CAL_FILE}" | cut -f1)
    echo "  ${DOMAIN}: ${SIZE}"
done

if [[ ! -f "${CALIBRATION_DIR}/combined.txt" ]]; then
    echo "ERROR: combined.txt not found. Run: make calibrate"
    exit 1
fi

echo ""

# ---------------------------------------------------------------
# Generate per-domain + combined imatrices in one model load
# ---------------------------------------------------------------
FORCE_FLAG=""
if [[ "${FORCE_IMATRIX:-}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

python3 "${PROJECT_DIR}/src/generate_imatrix.py" \
    --model-id "${MODEL_ID}" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --domains ${CALIBRATION_DOMAINS} \
    --output-dir "${CALIBRATION_DIR}" \
    --gguf-arch-key "${GGUF_ARCH_KEY}" \
    --llamacpp-dir "${LLAMACPP_DIR}" \
    --context-size "${IMATRIX_CONTEXT_SIZE}" \
    --dtype bfloat16 \
    ${FORCE_FLAG}

echo ""

# ---------------------------------------------------------------
# Merge per-domain imatrices with weighted blend
# ---------------------------------------------------------------
echo "--- Merging per-domain imatrices (DI-MATRIX) ---"
IMATRIX_FILES=""
for DOMAIN in ${CALIBRATION_DOMAINS}; do
    IMATRIX_FILES="${IMATRIX_FILES} ${CALIBRATION_DIR}/imatrix_${DOMAIN}.dat"
done

python3 "${PROJECT_DIR}/src/merge_imatrix.py" \
    ${IMATRIX_FILES} \
    --weights ${IMATRIX_WEIGHTS} \
    -o "${IMATRIX_MERGED}"

echo ""
echo "=== Imatrix generation complete ==="
echo "Files:"
echo "  Merged (weighted):  ${IMATRIX_MERGED}"
echo "  Combined (single):  ${IMATRIX_COMBINED}"
for DOMAIN in ${CALIBRATION_DOMAINS}; do
    echo "  Per-domain:         ${CALIBRATION_DIR}/imatrix_${DOMAIN}.dat"
done
echo ""
echo "Next: make quantize"
