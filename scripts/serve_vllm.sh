#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"
source "${SCRIPT_DIR}/../configs/sglang.env"
source "${SCRIPT_DIR}/../configs/vllm.env"

export CUDA_HOME="/usr/local/cuda-${SGLANG_CUDA_VERSION}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] ||
    ! "${CUDA_HOME}/bin/nvcc" --version | grep -q "release ${SGLANG_CUDA_VERSION}"; then
    echo "ERROR: CUDA ${SGLANG_CUDA_VERSION} is required at ${CUDA_HOME}" >&2
    exit 1
fi

MODEL_PATH="${NVFP4_CHECKPOINT:-${HOME}/models/nvfp4/${MODEL_NAME}-NVFP4}"
PORT="${SGLANG_PORT:-8888}"
MAX_REQUESTS="${1:-4}"
CTX="${2:-${NATIVE_CTX}}"
SERVED_NAME="${MODEL_ALIAS:-qwen3.8-27b}"

PIDFILE="${PROJECT_DIR}/.vllm.pid"
LOGFILE="${PROJECT_DIR}/.vllm.log"

ensure_vllm() {
    local manifest="${VLLM_VENV}/.super-quant-vllm-env"
    local source_revision
    source_revision="$(git -C "${VLLM_SOURCE_DIR}" rev-parse HEAD)"
    if [[ ! -x "${VLLM_VENV}/bin/python" ]] ||
        [[ ! -f "${manifest}" ]] ||
        [[ "$(sed -n '1p' "${manifest}")" != "${VLLM_ENV_REVISION}" ]] ||
        [[ "$(sed -n '2p' "${manifest}")" != "${source_revision}" ]]; then
        echo "=== Building pinned vLLM environment ${VLLM_ENV_REVISION} ==="
        bash "${SCRIPT_DIR}/setup_vllm_env.sh"
    fi
}

ensure_vllm

if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "vLLM already running (PID $OLD_PID). Run: make stop-vllm"
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
