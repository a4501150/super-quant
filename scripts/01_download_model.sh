#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

echo "=== Downloading model ==="
echo "Model:  ${MODEL_ID}"
echo "Target: ${BF16_DIR}"

mkdir -p "${BF16_DIR}"

if command -v hf &>/dev/null; then
    hf download "${MODEL_ID}" --local-dir "${BF16_DIR}"
else
    echo "hf CLI not found. Run: make setup"
    exit 1
fi

echo ""
echo "--- Verifying download ---"
SAFETENSORS=$(find "${BF16_DIR}" -name "*.safetensors" | wc -l)
CONFIG="${BF16_DIR}/config.json"

if [[ ${SAFETENSORS} -eq 0 ]]; then
    echo "ERROR: No safetensors files found in ${BF16_DIR}"
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config.json not found in ${BF16_DIR}"
    exit 1
fi

TOTAL_SIZE=$(du -sh "${BF16_DIR}" | cut -f1)
echo "Downloaded ${SAFETENSORS} safetensors files (${TOTAL_SIZE} total)"
echo ""
echo "=== Download complete ==="
echo "Next: make convert"
