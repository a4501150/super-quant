#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

echo "=== Downloading model ==="
echo "Model:  ${MODEL_ID}"
echo "Cache:  ~/.cache/huggingface/hub/"

if ! command -v hf &>/dev/null; then
    echo "hf CLI not found. Run: make setup"
    exit 1
fi

SNAPSHOT_DIR=$(hf download "${MODEL_ID}")

echo ""
echo "--- Verifying download ---"
SAFETENSORS=$(find "${SNAPSHOT_DIR}" -name "*.safetensors" | wc -l)

if [[ ${SAFETENSORS} -eq 0 ]]; then
    echo "ERROR: No safetensors files found in ${SNAPSHOT_DIR}"
    exit 1
fi

if [[ ! -f "${SNAPSHOT_DIR}/config.json" ]]; then
    echo "ERROR: config.json not found in ${SNAPSHOT_DIR}"
    exit 1
fi

TOTAL_SIZE=$(du -sh "${SNAPSHOT_DIR}" | cut -f1)
echo "Downloaded ${SAFETENSORS} safetensors files (${TOTAL_SIZE} total)"
echo "Snapshot: ${SNAPSHOT_DIR}"
echo ""
echo "=== Download complete ==="
echo "Next: make convert"
