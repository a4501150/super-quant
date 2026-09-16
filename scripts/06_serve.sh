#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"
source "${SCRIPT_DIR}/../configs/sglang.env"

export CUDA_HOME="/usr/local/cuda-${SGLANG_CUDA_VERSION}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

LLAMA_CMAKE_CACHE="${LLAMACPP_BUILD}/CMakeCache.txt"
if [[ ! -x "${LLAMA_SERVER}" || ! -f "${LLAMA_CMAKE_CACHE}" ]]; then
    echo "ERROR: llama.cpp Blackwell build is missing at ${LLAMACPP_BUILD}" >&2
    exit 1
fi
CUDA_MAJOR="${SGLANG_CUDA_VERSION%%.*}"
if ! grep -qE "^CMAKE_CUDA_COMPILER:[^=]+=${CUDA_HOME}/bin/nvcc$" "${LLAMA_CMAKE_CACHE}" ||
    ! grep -qE "^CMAKE_CUDA_ARCHITECTURES:[^=]+=${CUDA_ARCH}$" "${LLAMA_CMAKE_CACHE}" ||
    ! ldd "${LLAMA_SERVER}" | grep -q "libcudart.so.${CUDA_MAJOR}"; then
    echo "ERROR: llama.cpp must be rebuilt for CUDA ${SGLANG_CUDA_VERSION} and sm_${CUDA_ARCH}" >&2
    exit 1
fi

# Parse arguments
QUANT="${1:-UD-Q6_K}"
CTX="${2:-${NATIVE_CTX}}"
PORT="${3:-8000}"
PARALLEL="${4:-3}"
KV_TYPE_K="${5:-}"
KV_TYPE_V="${6:-}"

# Find the GGUF file
GGUF="${MODELS_DIR}/${MODEL_NAME}-${QUANT}.gguf"
if [[ ! -f "${GGUF}" ]]; then
    echo "ERROR: GGUF not found: ${GGUF}"
    echo ""
    echo "Available models:"
    for f in "${MODELS_DIR}"/${MODEL_NAME}-*.gguf; do
        [[ -f "$f" ]] && echo "  $(basename "$f" .gguf | sed "s/${MODEL_NAME}-//")"
    done
    echo ""
    echo "Usage: $0 [QUANT] [CTX_PER_SLOT] [PORT] [PARALLEL] [KV_TYPE_K] [KV_TYPE_V]"
    echo "  e.g. $0 UD-Q6_K 524288 8080 5 q8_0 turbo4"
    exit 1
fi

SIZE=$(du -sh "${GGUF}" | cut -f1)
echo "=== Launching llama-server ==="
echo "Model:    ${GGUF} (${SIZE})"
echo "Context:  ${CTX} shared pool (unified KV), ${PARALLEL} slots"
echo "KV cache: K=${KV_TYPE_K:-f16} V=${KV_TYPE_V:-f16} (unified)"
echo "Port:     ${PORT}"
echo "GPU:      ${GPU_LAYERS} layers offloaded"

# Build server args
ARGS=(
    -m "${GGUF}"
    -ngl "${GPU_LAYERS}"
    -fa on
    -c "${CTX}"
    --parallel "${PARALLEL}"
    -kvu
    --cache-ram 0
    --slot-save-path "${MODELS_DIR}/cache"
    --host 0.0.0.0
    --port "${PORT}"
    --threads "${THREADS}"
    --metrics
    --alias "${MODEL_ALIAS:-${MODEL_NAME}}"
    --jinja
    --reasoning-format "${LLAMA_REASONING_FORMAT:-deepseek}"
    --reasoning-preserve
)

# Custom chat template (per-model, optional)
if [[ -n "${CHAT_TEMPLATE_FILE:-}" ]]; then
    echo "Template: ${CHAT_TEMPLATE_FILE}"
    ARGS+=(--chat-template-file "${CHAT_TEMPLATE_FILE}")
fi

# KV cache type override (default: f16, llama.cpp native default)
if [[ -n "${KV_TYPE_K}" ]]; then
    ARGS+=(--cache-type-k "${KV_TYPE_K}")
fi
if [[ -n "${KV_TYPE_V}" ]]; then
    ARGS+=(--cache-type-v "${KV_TYPE_V}")
fi

# YaRN context extension beyond native max
if (( CTX > NATIVE_CTX )); then
    ROPE_SCALE=$(python3 -c "print(round(${CTX} / ${NATIVE_CTX}, 6))")
    echo "YaRN:     rope_scale=${ROPE_SCALE} (${NATIVE_CTX} → ${CTX})"
    ARGS+=(
        --rope-scaling yarn
        --rope-scale "${ROPE_SCALE}"
        --yarn-orig-ctx "${NATIVE_CTX}"
        --override-kv "${GGUF_ARCH_KEY}.context_length=int:${CTX}"
    )
fi

# The active model config chooses the default; a command-line SPEC_TYPE overrides it.
SPEC_TYPE="${SPEC_TYPE:-${LLAMA_SPEC_TYPE_DEFAULT:-none}}"

if [[ "${SPEC_TYPE}" == "dspark" ]]; then
    DSPARK_DRAFT="${DSPARK_DRAFT_GGUF:-}"
    if [[ -z "${DSPARK_DRAFT}" ]]; then
        echo "ERROR: SPEC_TYPE=dspark requires DSPARK_DRAFT_GGUF in the active model config."
        exit 1
    fi
    if [[ ! -f "${DSPARK_DRAFT}" ]]; then
        echo "ERROR: DSpark draft not found: ${DSPARK_DRAFT}"
        exit 1
    fi
    echo "DSpark:   ${DSPARK_DRAFT}"
    ARGS+=(
        --spec-type draft-dspark
        --spec-draft-model "${DSPARK_DRAFT}"
        --spec-draft-n-max "${LLAMA_SPEC_DRAFT_N_MAX:-7}"
        -ngld "${GPU_LAYERS}"
    )
elif [[ "${SPEC_TYPE}" == "mtp" ]]; then
    if [[ -z "${MTP_N_MAX:-}" ]]; then
        echo "ERROR: SPEC_TYPE=mtp requires MTP_N_MAX in the active model config."
        exit 1
    fi
    echo "MTP:      enabled (draft-n-max=${MTP_N_MAX})"
    ARGS+=(
        --spec-type draft-mtp
        --spec-draft-n-max "${MTP_N_MAX}"
    )
elif [[ "${SPEC_TYPE}" != "none" ]]; then
    echo "ERROR: unsupported SPEC_TYPE=${SPEC_TYPE}; expected none, dspark, or mtp."
    exit 1
fi

# Enable vision if mmproj exists
if [[ -f "${MMPROJ_GGUF}" ]]; then
    echo "Vision:   ${MMPROJ_GGUF}"
    ARGS+=(--mmproj "${MMPROJ_GGUF}")
fi

echo ""
echo "Starting server..."
echo "API:      http://localhost:${PORT}/v1/chat/completions"
echo "Health:   http://localhost:${PORT}/health"
echo "Metrics:  http://localhost:${PORT}/metrics"
echo ""

"${LLAMA_SERVER}" "${ARGS[@]}" &
SERVER_PID=$!

# Wait for server to be ready (longer timeout for large context allocation)
TIMEOUT=120
echo "Waiting for server to start (up to ${TIMEOUT}s)..."
for i in $(seq 1 "${TIMEOUT}"); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "Server is ready! (PID: ${SERVER_PID})"
        echo ""
        echo "Test:"
        echo "  curl -s http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"${MODEL_ALIAS:-${MODEL_NAME}}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
        echo ""
        echo "Press Ctrl+C to stop."
        wait ${SERVER_PID}
        exit 0
    fi
    sleep 1
done

echo "WARNING: Server did not respond within ${TIMEOUT}s. Check output above."
wait ${SERVER_PID}
