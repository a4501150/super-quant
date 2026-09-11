#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

# Resolve HF cache snapshot path (instant if already downloaded)
BF16_DIR=$(hf download "${MODEL_ID}" 2>/dev/null | grep -oP '(?<=path=).*' || true)

if [[ -z "${BF16_DIR}" ]] || [[ ! -f "${BF16_DIR}/config.json" ]]; then
    echo "ERROR: Model not found in HF cache: ${MODEL_ID}"
    echo "Run: make download"
    exit 1
fi

echo "=== Converting to F16 GGUF ==="
echo "Source: ${BF16_DIR}"
echo "Output: ${F16_GGUF}"

# Find convert script
CONVERT="${CONVERT_SCRIPT}"
if [[ ! -f "${CONVERT}" ]]; then
    CONVERT=$(find "${LLAMACPP_DIR}" -name "convert_hf_to_gguf.py" -type f | head -1)
fi

if [[ -z "${CONVERT}" ]] || [[ ! -f "${CONVERT}" ]]; then
    echo "ERROR: convert_hf_to_gguf.py not found"
    exit 1
fi

echo "Using convert script: ${CONVERT}"

mkdir -p "${MODELS_DIR}"

# --- Text model with MTP (default: MTP tensors bundled in) ---
echo ""
echo "--- Converting text model (with MTP) ---"
python3 "${CONVERT}" "${BF16_DIR}" \
    --outtype f16 \
    --outfile "${F16_GGUF}"

if [[ ! -f "${F16_GGUF}" ]]; then
    echo "ERROR: Conversion failed — ${F16_GGUF} not created"
    exit 1
fi

F16_SIZE=$(du -sh "${F16_GGUF}" | cut -f1)
echo "  ${F16_GGUF} (${F16_SIZE})"

# --- Vision projector (mmproj) ---
echo ""
echo "--- Converting vision projector (mmproj) ---"

python3 "${CONVERT}" "${BF16_DIR}" \
    --outtype f16 \
    --mmproj \
    --outfile "${MMPROJ_GGUF}"

if [[ ! -f "${MMPROJ_GGUF}" ]]; then
    echo "ERROR: mmproj conversion failed"
    exit 1
fi

MMPROJ_SIZE=$(du -sh "${MMPROJ_GGUF}" | cut -f1)
echo "  ${MMPROJ_GGUF} (${MMPROJ_SIZE})"

# --- Smoke test (verify GGUF loads) ---
echo ""
echo "--- Smoke test ---"
if [[ -x "${LLAMA_PERPLEXITY}" ]]; then
    "${LLAMA_PERPLEXITY}" -m "${F16_GGUF}" -ngl 0 -f /dev/null --chunks 0 2>&1 | head -5 && echo "Smoke test passed (model loads)" || echo "WARNING: Smoke test failed"
else
    echo "Skipping smoke test — llama-perplexity not found. Run: make setup"
fi

echo ""
echo "=== Conversion complete ==="
echo "Next: make calibrate"
