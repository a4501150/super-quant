#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

# Parse arguments
QUANT="${1:-Q5_K_M}"
CTX="${2:-32768}"
PORT="${3:-8080}"

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
    echo "Usage: $0 [QUANT_TYPE] [CONTEXT_SIZE] [PORT]"
    echo "  e.g. $0 Q5_K_M 32768 8080"
    exit 1
fi

SIZE=$(du -sh "${GGUF}" | cut -f1)
echo "=== Launching llama-server ==="
echo "Model:   ${GGUF} (${SIZE})"
echo "Context: ${CTX}"
echo "Port:    ${PORT}"
echo "GPU:     ${GPU_LAYERS} layers offloaded"

# Build server args
ARGS=(
    -m "${GGUF}"
    -ngl "${GPU_LAYERS}"
    --flash-attn
    -c "${CTX}"
    --cache-type-k q8_0
    --cache-type-v q8_0
    --host 0.0.0.0
    --port "${PORT}"
    --threads "${THREADS}"
    --metrics
    --jinja
    --chat-template-kwargs '{"enable_thinking":true}'
)

# Enable MTP speculative decoding if available
if [[ "${MTP_ENABLED}" = true ]]; then
    echo "MTP:     enabled (draft-n-max=3)"
    ARGS+=(
        --spec-type mtp
        --spec-draft-n-max 3
    )
fi

# Enable vision if mmproj exists
if [[ -f "${MMPROJ_GGUF}" ]]; then
    echo "Vision:  ${MMPROJ_GGUF}"
    ARGS+=(--mmproj "${MMPROJ_GGUF}")
fi

echo ""
echo "Starting server..."
echo "API:     http://localhost:${PORT}/v1/chat/completions"
echo "Health:  http://localhost:${PORT}/health"
echo "Metrics: http://localhost:${PORT}/metrics"
echo ""

"${LLAMA_SERVER}" "${ARGS[@]}" &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 30); do
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

echo "WARNING: Server did not respond within 30s. Check output above."
wait ${SERVER_PID}
