#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

MODEL_PATH="${NVFP4_CHECKPOINT:-${HOME}/models/nvfp4/${MODEL_NAME}-NVFP4}"
DRAFT_MODEL="${DFLASH2_DRAFT_ID:-z-lab/Qwen3.8-27B-DFlash2}"
PORT="${SGLANG_PORT:-8888}"
MAX_REQUESTS="${1:-4}"
CTX="${2:-${NATIVE_CTX}}"
SERVED_NAME="${MODEL_ALIAS:-qwen3.8-27b}"

PIDFILE="${PROJECT_DIR}/.sglang.pid"
LOGFILE="${PROJECT_DIR}/.sglang.log"

SGLANG_VENV="${HOME}/.venvs/sglang"

ensure_sglang() {
    if [[ ! -f "${SGLANG_VENV}/bin/python" ]]; then
        echo "=== Installing SGLang ==="
        uv venv "${SGLANG_VENV}" --python 3.12
        uv pip install --python "${SGLANG_VENV}/bin/python" "sglang[all]"
        echo "SGLang installed to ${SGLANG_VENV}"
    fi
}

if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "SGLang already running (PID $OLD_PID). Run: make stop-sglang"
        exit 1
    fi
    rm -f "$PIDFILE"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    echo "Run: make quantize-nvfp4"
    exit 1
fi

ensure_sglang

echo "=== Starting SGLang Server ==="
echo "Model:   $MODEL_PATH"
echo "Drafter: $DRAFT_MODEL"
echo "Port:    $PORT"
echo "Context: $CTX"
echo "Slots:   $MAX_REQUESTS"
echo "Log:     $LOGFILE"

"${SGLANG_VENV}/bin/python" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    --mem-fraction-static 0.90 \
    --attention-backend fa3 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 4096 \
    --enable-mixed-chunk \
    --num-continuous-decode-steps 4 \
    --kv-cache-dtype fp8_e4m3 \
    --fp4-gemm-backend flashinfer_cutedsl \
    --cuda-graph-backend-decode breakable \
    --context-length "$CTX" \
    --max-running-requests "$MAX_REQUESTS" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-draft-tokens 8 \
    --speculative-draft-model-quantization unquant \
    --speculative-draft-attention-backend flashinfer \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --sampling-defaults model \
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
        echo "=== SGLang Ready ==="
        echo "API: http://localhost:${PORT}/v1/chat/completions"
        echo "Models: http://localhost:${PORT}/v1/models"
        echo ""
        echo "Example:"
        echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"${SERVED_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
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
