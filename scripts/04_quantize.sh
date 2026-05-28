#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

echo "=== Quantizing Model (UD — all quants get per-tensor overrides) ==="

if [[ ! -f "${F16_GGUF}" ]]; then
    echo "ERROR: F16 GGUF not found at ${F16_GGUF}"
    echo "Run: make convert"
    exit 1
fi

# Find imatrix
IMATRIX="${IMATRIX_MERGED}"
if [[ ! -f "${IMATRIX}" ]]; then
    IMATRIX="${IMATRIX_COMBINED}"
fi
if [[ ! -f "${IMATRIX}" ]]; then
    echo "WARNING: No imatrix found. Quality will be significantly worse."
    echo "Run: make imatrix"
    IMATRIX=""
else
    echo "imatrix:   ${IMATRIX}"
fi

# Check per-tensor overrides — strip comments/blanks since llama-quantize doesn't support them
HAS_OVERRIDES=false
TENSOR_OVERRIDES_CLEAN=""
if [[ -f "${TENSOR_OVERRIDES}" ]]; then
    TENSOR_OVERRIDES_CLEAN=$(mktemp "${TMPDIR:-/tmp}/tensor_overrides.XXXXXX")
    grep -v '^\s*#' "${TENSOR_OVERRIDES}" | grep -v '^\s*$' > "${TENSOR_OVERRIDES_CLEAN}"
    HAS_OVERRIDES=true
    echo "overrides: ${TENSOR_OVERRIDES} ($(wc -l < "${TENSOR_OVERRIDES_CLEAN}") rules)"
else
    echo "overrides: none (run 'make sensitivity' to generate)"
fi
echo ""

for TYPE in ${QUANT_TYPES}; do
    OUTPUT="${MODELS_DIR}/${MODEL_NAME}-UD-${TYPE}.gguf"

    if [[ -f "${OUTPUT}" ]]; then
        echo "Skipping UD-${TYPE} — already exists"
        continue
    fi

    echo "--- UD-${TYPE} ---"

    # Build args
    ARGS=()
    if [[ -n "${IMATRIX}" ]]; then
        ARGS+=(--imatrix "${IMATRIX}")
    fi

    # Apply per-tensor overrides — required for quality quants
    if [[ "${HAS_OVERRIDES}" = true ]]; then
        QUANTIZE_HELP=$("${LLAMA_QUANTIZE}" --help 2>&1 || true)
        if echo "${QUANTIZE_HELP}" | grep -q "tensor-type-file"; then
            ARGS+=(--tensor-type-file "${TENSOR_OVERRIDES_CLEAN}")
        elif echo "${QUANTIZE_HELP}" | grep -q "output-tensor-type"; then
            ARGS+=(--output-tensor-type "${TENSOR_OVERRIDES_CLEAN}")
        else
            echo "ERROR: per-tensor overrides not supported by this llama.cpp build."
            echo "Update llama.cpp and rebuild: cd ${LLAMACPP_DIR} && git pull && cmake --preset blackwell && cmake --build build-blackwell -j\$(nproc)"
            exit 1
        fi
    fi

    "${LLAMA_QUANTIZE}" \
        "${ARGS[@]}" \
        "${F16_GGUF}" \
        "${OUTPUT}" \
        "${TYPE}"

    SIZE=$(du -sh "${OUTPUT}" | cut -f1)
    echo "  → ${OUTPUT} (${SIZE})"
    echo ""
done

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo "=== Quantization complete ==="
echo ""
echo "Models:"
for f in "${MODELS_DIR}"/${MODEL_NAME}-*.gguf; do
    if [[ -f "$f" ]] && [[ "$(basename "$f")" != *"-F16.gguf" ]]; then
        SIZE=$(du -sh "$f" | cut -f1)
        echo "  $(basename "$f"): ${SIZE}"
    fi
done
# Cleanup temp file
[[ -n "${TENSOR_OVERRIDES_CLEAN}" ]] && rm -f "${TENSOR_OVERRIDES_CLEAN}"

echo ""
echo "Next: make benchmark"
