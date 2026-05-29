#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

# Parse arguments
QUANT="${1:-UD-Q6_K}"
CTX="${2:-524288}"
PORT="${3:-8080}"
PARALLEL="${4:-5}"
KV_TYPE="${5:-q8_0}"

NATIVE_CTX=262144

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
    echo "Usage: $0 [QUANT] [CTX_PER_SLOT] [PORT] [PARALLEL] [KV_TYPE]"
    echo "  e.g. $0 UD-Q6_K 524288 8080 5 turbo4"
    exit 1
fi

TOTAL_CTX=$(( CTX * PARALLEL ))
SIZE=$(du -sh "${GGUF}" | cut -f1)
echo "=== Launching llama-server ==="
echo "Model:    ${GGUF} (${SIZE})"
echo "Context:  ${CTX} per slot × ${PARALLEL} slots = ${TOTAL_CTX} total"
echo "KV cache: ${KV_TYPE} (unified)"
echo "Port:     ${PORT}"
echo "GPU:      ${GPU_LAYERS} layers offloaded"

# Build server args
ARGS=(
    -m "${GGUF}"
    -ngl "${GPU_LAYERS}"
    -fa on
    -c "${TOTAL_CTX}"
    --parallel "${PARALLEL}"
    --cache-type-k "${KV_TYPE}"
    --cache-type-v "${KV_TYPE}"
    -kvu
    --cache-ram -1
    --host 0.0.0.0
    --port "${PORT}"
    --threads "${THREADS}"
    --metrics
    --jinja
    --reasoning on
    --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
)

# YaRN context extension beyond native max
if (( CTX > NATIVE_CTX )); then
    ROPE_SCALE=$(python3 -c "print(round(${CTX} / ${NATIVE_CTX}, 6))")
    echo "YaRN:     rope_scale=${ROPE_SCALE} (${NATIVE_CTX} → ${CTX})"
    ARGS+=(
        --rope-scaling yarn
        --rope-scale "${ROPE_SCALE}"
        --yarn-orig-ctx "${NATIVE_CTX}"
        --override-kv "qwen35.context_length=int:${CTX}"
    )
fi

# Enable MTP speculative decoding if available
if [[ "${MTP_ENABLED}" = true ]]; then
    echo "MTP:      enabled (draft-n-max=${MTP_N_MAX})"
    ARGS+=(
        --spec-type draft-mtp
        --spec-draft-n-max "${MTP_N_MAX}"
    )
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
        echo "    -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
        echo ""
        echo "Press Ctrl+C to stop."
        wait ${SERVER_PID}
        exit 0
    fi
    sleep 1
done

echo "WARNING: Server did not respond within ${TIMEOUT}s. Check output above."
wait ${SERVER_PID}
