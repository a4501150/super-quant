#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

if [[ -z "${DFLASH_DRAFT_ID:-}" ]]; then
    echo "ERROR: DFLASH_DRAFT_ID not set in model config"
    exit 1
fi

echo "=== Downloading DFlash draft model ==="
echo "Model:  ${DFLASH_DRAFT_ID}"
echo "Output: ${DFLASH_DRAFT_GGUF}"

if [[ -f "${DFLASH_DRAFT_GGUF}" ]]; then
    echo "Already exists: ${DFLASH_DRAFT_GGUF} ($(du -sh "${DFLASH_DRAFT_GGUF}" | cut -f1))"
    echo "Delete it to re-download and re-convert."
    exit 0
fi

if ! command -v hf &>/dev/null; then
    echo "hf CLI not found. Run: make setup"
    exit 1
fi

SNAPSHOT_DIR=$(hf download "${DFLASH_DRAFT_ID}")
echo "Snapshot: ${SNAPSHOT_DIR}"

# DFlash draft models use a custom architecture — need the tokenizer from the target model
# Copy tokenizer if missing
if [[ ! -f "${SNAPSHOT_DIR}/tokenizer.json" ]]; then
    echo "--- Copying tokenizer from target model ---"
    TARGET_DIR=$(hf download "${MODEL_ID}" 2>/dev/null || true)
    if [[ -n "${TARGET_DIR}" ]] && [[ -f "${TARGET_DIR}/tokenizer.json" ]]; then
        cp "${TARGET_DIR}/tokenizer.json" "${SNAPSHOT_DIR}/"
        cp "${TARGET_DIR}/tokenizer_config.json" "${SNAPSHOT_DIR}/" 2>/dev/null || true
        echo "Copied tokenizer from ${MODEL_ID}"
    else
        echo "WARNING: Could not find target tokenizer. Convert may fail."
    fi
fi

echo ""
echo "--- Converting DFlash draft to GGUF ---"

CONVERT="${CONVERT_SCRIPT}"
if [[ ! -f "${CONVERT}" ]]; then
    CONVERT=$(find "${LLAMACPP_DIR}" -name "convert_hf_to_gguf.py" -type f | head -1)
fi

if [[ -z "${CONVERT}" ]] || [[ ! -f "${CONVERT}" ]]; then
    echo "ERROR: convert_hf_to_gguf.py not found"
    exit 1
fi

mkdir -p "${MODELS_DIR}"

python3 "${CONVERT}" "${SNAPSHOT_DIR}" \
    --outtype f16 \
    --outfile "${DFLASH_DRAFT_GGUF}"

if [[ ! -f "${DFLASH_DRAFT_GGUF}" ]]; then
    echo "ERROR: Conversion failed"
    exit 1
fi

DRAFT_SIZE=$(du -sh "${DFLASH_DRAFT_GGUF}" | cut -f1)
echo "  ${DFLASH_DRAFT_GGUF} (${DRAFT_SIZE})"
echo ""
echo "=== DFlash draft ready ==="
