#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

MODEL_PATH="${NVFP4_CHECKPOINT:-${HOME}/models/nvfp4/${MODEL_NAME}-NVFP4}"
PORT="${SGLANG_PORT:-8888}"
MAX_REQUESTS="${1:-4}"
CTX="${2:-${NATIVE_CTX}}"
SERVED_NAME="${MODEL_ALIAS:-qwen3.8-27b}"

PIDFILE="${PROJECT_DIR}/.sglang.pid"
LOGFILE="${PROJECT_DIR}/.sglang.log"

VLLM_VENV="${HOME}/.venvs/vllm"

if [[ ! -f "${VLLM_VENV}/bin/python" ]]; then
    echo "ERROR: vLLM venv not found at ${VLLM_VENV}"
    echo "Install: uv venv ${VLLM_VENV} --python 3.12 && uv pip install --python ${VLLM_VENV}/bin/python vllm"
    exit 1
fi

if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Server already running (PID $OLD_PID). Run: make stop-sglang"
        exit 1
    fi
    rm -f "$PIDFILE"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    echo "Run: make quantize-nvfp4"
    exit 1
fi

echo "=== Starting vLLM Server ==="
echo "Model:   $MODEL_PATH"
echo "Port:    $PORT"
echo "Context: $CTX"
echo "Slots:   $MAX_REQUESTS"
echo "Log:     $LOGFILE"

VLLM_ATTENTION_BACKEND=TRITON_ATTN "${VLLM_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$CTX" \
    --max-num-seqs "$MAX_REQUESTS" \
    --host 0.0.0.0 \
    --port "$PORT" \
    > "$LOGFILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
echo "Server PID: $SERVER_PID"

echo "Waiting for server to start..."
for i in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server exited prematurely. Check $LOGFILE"
        tail -30 "$LOGFILE"
        rm -f "$PIDFILE"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "=== vLLM Ready ==="
        echo "API: http://localhost:${PORT}/v1/chat/completions"
        echo "Models: http://localhost:${PORT}/v1/models"
        exit 0
    fi
    if (( i % 10 == 0 )); then
        echo "  still starting... (${i}s)"
    fi
    sleep 1
done

echo "ERROR: Server did not become ready within 300s. Check $LOGFILE"
tail -30 "$LOGFILE"
exit 1
