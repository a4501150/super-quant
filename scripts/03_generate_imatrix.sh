#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"


echo "=== Generating Importance Matrices (DI-MATRIX) ==="

if [[ ! -f "${F16_GGUF}" ]]; then
    echo "ERROR: F16 GGUF not found at ${F16_GGUF}"
    echo "Run: make convert"
    exit 1
fi

IMATRIX_MODEL="${F16_GGUF}"

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

# Also check combined
if [[ ! -f "${CALIBRATION_DIR}/combined.txt" ]]; then
    echo "ERROR: combined.txt not found. Run: make calibrate"
    exit 1
fi

echo ""

# ---------------------------------------------------------------
# Generate per-domain imatrices
# ---------------------------------------------------------------
for DOMAIN in ${CALIBRATION_DOMAINS}; do
    CAL_FILE="${CALIBRATION_DIR}/${DOMAIN}.txt"
    OUT_FILE="${CALIBRATION_DIR}/imatrix_${DOMAIN}.dat"

    if [[ -f "${OUT_FILE}" ]]; then
        echo "Skipping ${DOMAIN} — imatrix already exists: ${OUT_FILE}"
        continue
    fi

    echo "--- Generating imatrix: ${DOMAIN} ---"
    "${LLAMA_IMATRIX}" \
        -m "${IMATRIX_MODEL}" \
        -f "${CAL_FILE}" \
        -o "${OUT_FILE}" \
        --output-format dat \
        -ngl "${GPU_LAYERS}" \
        -c "${IMATRIX_CONTEXT_SIZE}" \
        -b 512 \
        -ub 256 \
        -t "${THREADS}" \
        -fa on \
        --verbosity 1

    SIZE=$(du -sh "${OUT_FILE}" | cut -f1)
    echo "  Output: ${OUT_FILE} (${SIZE})"
    echo ""
done

# ---------------------------------------------------------------
# Generate combined imatrix (single-pass over all data)
# ---------------------------------------------------------------
if [[ ! -f "${IMATRIX_COMBINED}" ]]; then
    echo "--- Generating combined imatrix ---"
    "${LLAMA_IMATRIX}" \
        -m "${IMATRIX_MODEL}" \
        -f "${CALIBRATION_DIR}/combined.txt" \
        -o "${IMATRIX_COMBINED}" \
        --output-format dat \
        -ngl "${GPU_LAYERS}" \
        -c "${IMATRIX_CONTEXT_SIZE}" \
        -b 512 \
        -ub 256 \
        -t "${THREADS}" \
        -fa on \
        --verbosity 1

    SIZE=$(du -sh "${IMATRIX_COMBINED}" | cut -f1)
    echo "  Output: ${IMATRIX_COMBINED} (${SIZE})"
else
    echo "Skipping combined — already exists: ${IMATRIX_COMBINED}"
fi

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
