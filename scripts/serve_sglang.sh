#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"
source "${SCRIPT_DIR}/../configs/sglang.env"

CUDA_HOME="/usr/local/cuda-${SGLANG_CUDA_VERSION}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] ||
    ! "${CUDA_HOME}/bin/nvcc" --version | grep -q "release ${SGLANG_CUDA_VERSION}"; then
    echo "ERROR: CUDA ${SGLANG_CUDA_VERSION} is required at ${CUDA_HOME}" >&2
    exit 1
fi

MODEL_PATH="${NVFP4_CHECKPOINT:-${HOME}/models/nvfp4/${MODEL_NAME}-NVFP4}"
DRAFT_MODEL="${DFLASH2_DRAFT_ID:-}"
PORT="${SGLANG_PORT:-8888}"
MAX_REQUESTS="${1:-${SGLANG_MAX_RUNNING_REQUESTS:-4}}"
# mamba slots and decode graph cap follow max-running-requests
# (SGLang cookbook pins slots at 3/request without speculative decoding)
MAMBA_SLOTS="${SGLANG_MAX_MAMBA_CACHE_SIZE:-$((MAX_REQUESTS * ${SGLANG_MAMBA_SLOTS_PER_REQ:-3}))}"
CUDA_GRAPH_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-$MAX_REQUESTS}"
CTX="${2:-${NATIVE_CTX}}"
SERVED_NAME="${MODEL_ALIAS:-${MODEL_NAME}}"

PIDFILE="${PROJECT_DIR}/.sglang.pid"
LOGFILE="${PROJECT_DIR}/.sglang.log"

ensure_sglang() {
    local manifest="${SGLANG_VENV}/.super-quant-sglang-env"
    local source_revision
    source_revision="$(git -C "${SGLANG_SOURCE_DIR}" rev-parse HEAD)"
    if [[ ! -x "${SGLANG_VENV}/bin/sglang" ]] ||
        [[ ! -f "${manifest}" ]] ||
        [[ "$(sed -n '1p' "${manifest}")" != "${SGLANG_ENV_REVISION}" ]] ||
        [[ "$(sed -n '2p' "${manifest}")" != "${source_revision}" ]]; then
        echo "=== Building pinned SGLang environment ${SGLANG_ENV_REVISION} ==="
        bash "${SCRIPT_DIR}/setup_sglang_env.sh"
    fi
}

if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "SGLang already running (PID $OLD_PID). Run: make stop"
        exit 1
    fi
    rm -f "$PIDFILE"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    echo "Run: make quantize-nvfp4"
    exit 1
fi

if [[ -n "${SGLANG_PLE_SAFETENSORS_DIR:-}" ]]; then
    if [[ ! -d "${SGLANG_PLE_SAFETENSORS_DIR}" ]]; then
        echo "ERROR: PLE checkpoint not found: ${SGLANG_PLE_SAFETENSORS_DIR}" >&2
        exit 1
    fi
    if [[ ! -f "${SGLANG_PLE_SAFETENSORS_DIR}/model.safetensors.index.json" ]]; then
        echo "ERROR: PLE checkpoint index not found: ${SGLANG_PLE_SAFETENSORS_DIR}/model.safetensors.index.json" >&2
        exit 1
    fi
    export SGLANG_QWEN4_PLE_SAFETENSORS="${SGLANG_PLE_SAFETENSORS_DIR}"
fi

ensure_sglang

echo "=== Starting SGLang Server ==="
echo "Model:   $MODEL_PATH"
echo "Drafter: $DRAFT_MODEL"
echo "Port:    $PORT"
echo "Context: $CTX"
echo "Slots:   $MAX_REQUESTS (mamba slots: $MAMBA_SLOTS, graph bs cap: $CUDA_GRAPH_BS)"
echo "Log:     $LOGFILE"

ARGS=(
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --trust-remote-code
    --host 0.0.0.0
    --port "$PORT"
    --context-length "$CTX"
    --sampling-defaults "${SGLANG_SAMPLING_DEFAULTS:-model}"
)

[[ -n "${SGLANG_ATTENTION_BACKEND:-}" ]] && ARGS+=(--attention-backend "$SGLANG_ATTENTION_BACKEND")
[[ -n "${SGLANG_TP_SIZE:-}" && "${SGLANG_TP_SIZE:-1}" != "1" ]] && ARGS+=(--tp-size "$SGLANG_TP_SIZE")
[[ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]] && ARGS+=(--kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
[[ -n "${SGLANG_FP8_GEMM_BACKEND:-}" ]] && ARGS+=(--fp8-gemm-backend "$SGLANG_FP8_GEMM_BACKEND")
# Expert-axis sharding for checkpoints whose per-rank MoE intermediate fails
# the cutlass nvfp4 alignment check under pure TP (moe_intermediate/tp % 32).
[[ -n "${SGLANG_EP_SIZE:-}" ]] && ARGS+=(--ep-size "$SGLANG_EP_SIZE")
[[ -n "${SGLANG_DEBUG_DUMP_DIR:-}" ]] && ARGS+=(--debug-tensor-dump-output-folder "$SGLANG_DEBUG_DUMP_DIR")
[[ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]] && ARGS+=(--mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC")
[[ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]] && ARGS+=(--max-total-tokens "$SGLANG_MAX_TOTAL_TOKENS")
[[ -n "${SGLANG_CUDA_GRAPH_BACKEND_DECODE:-}" ]] && ARGS+=(--cuda-graph-backend-decode "$SGLANG_CUDA_GRAPH_BACKEND_DECODE")
[[ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]] && ARGS+=(--chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE" --max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-$SGLANG_CHUNKED_PREFILL_SIZE}")
[[ "${SGLANG_ENABLE_MIXED_CHUNK:-}" == "true" ]] && ARGS+=(--enable-mixed-chunk)
[[ -n "${SGLANG_NUM_CONTINUOUS_DECODE_STEPS:-}" ]] && ARGS+=(--num-continuous-decode-steps "$SGLANG_NUM_CONTINUOUS_DECODE_STEPS")
[[ -n "${MAX_REQUESTS:-}" ]] && ARGS+=(--max-running-requests "$MAX_REQUESTS")
[[ "${SGLANG_ENABLE_CACHE_REPORT:-}" == "true" ]] && ARGS+=(--enable-cache-report)
[[ "${SGLANG_DISABLE_RADIX_CACHE:-}" == "true" ]] && ARGS+=(--disable-radix-cache)
[[ "${SGLANG_ALLOW_AUTO_TRUNCATE:-}" == "true" ]] && ARGS+=(--allow-auto-truncate)
[[ "${SGLANG_ENABLE_LINEAR_REPLAYSSM_SPEC:-}" == "true" ]] && ARGS+=(--enable-linear-replayssm-spec)
[[ -n "${SGLANG_MAMBA_RADIX_CACHE_STRATEGY:-}" ]] && ARGS+=(--mamba-radix-cache-strategy "$SGLANG_MAMBA_RADIX_CACHE_STRATEGY")
[[ -n "${SGLANG_MAMBA_FULL_MEMORY_RATIO:-}" ]] && ARGS+=(--mamba-full-memory-ratio "$SGLANG_MAMBA_FULL_MEMORY_RATIO")
[[ -n "${SGLANG_PAGE_SIZE:-}" ]] && ARGS+=(--page-size "$SGLANG_PAGE_SIZE")
[[ -n "${SGLANG_MAMBA_SSM_DTYPE:-}" ]] && ARGS+=(--mamba-ssm-dtype "$SGLANG_MAMBA_SSM_DTYPE")
ARGS+=(--max-mamba-cache-size "$MAMBA_SLOTS")
[[ -n "${SGLANG_MAMBA_TRACK_INTERVAL:-}" ]] && ARGS+=(--mamba-track-interval "$SGLANG_MAMBA_TRACK_INTERVAL")
ARGS+=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_BS")
LINEAR_ATTN_DECODE_BACKEND="${SGLANG_LINEAR_ATTN_DECODE_BACKEND:-${SGLANG_LINEAR_ATTN_BACKEND:-}}"
LINEAR_ATTN_PREFILL_BACKEND="${SGLANG_LINEAR_ATTN_PREFILL_BACKEND:-${SGLANG_LINEAR_ATTN_BACKEND:-}}"
[[ -n "$LINEAR_ATTN_DECODE_BACKEND" ]] && ARGS+=(--linear-attn-decode-backend "$LINEAR_ATTN_DECODE_BACKEND")
[[ -n "$LINEAR_ATTN_PREFILL_BACKEND" ]] && ARGS+=(--linear-attn-prefill-backend "$LINEAR_ATTN_PREFILL_BACKEND")
[[ -n "${SGLANG_LINEAR_ATTN_VERIFY_BACKEND:-}" ]] && ARGS+=(--linear-attn-verify-backend "$SGLANG_LINEAR_ATTN_VERIFY_BACKEND")
[[ "${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-}" == "true" ]] && ARGS+=(--disable-flashinfer-autotune)

if [[ "${SGLANG_ENABLE_HIERARCHICAL_CACHE:-}" == "true" && "${SGLANG_DISABLE_RADIX_CACHE:-}" != "true" ]]; then
    # Keep the host MAMBA tier at least as large as the derived device slots.
    HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-}"
    if [[ -n "${SGLANG_HICACHE_MAMBA_SLOTS_PER_GB:-}" ]]; then
        [[ "$MAMBA_SLOTS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: derived MAMBA_SLOTS must be a positive integer (got: $MAMBA_SLOTS)"; exit 1; }
        [[ "$SGLANG_HICACHE_MAMBA_SLOTS_PER_GB" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SGLANG_HICACHE_MAMBA_SLOTS_PER_GB must be a positive integer (got: $SGLANG_HICACHE_MAMBA_SLOTS_PER_GB)"; exit 1; }
        MIN_HICACHE_SIZE=$(( (MAMBA_SLOTS + SGLANG_HICACHE_MAMBA_SLOTS_PER_GB - 1) / SGLANG_HICACHE_MAMBA_SLOTS_PER_GB ))
        if [[ ! "$HICACHE_SIZE" =~ ^[1-9][0-9]*$ || "$HICACHE_SIZE" -lt "$MIN_HICACHE_SIZE" ]]; then
            HICACHE_SIZE="$MIN_HICACHE_SIZE"
        fi
    fi
    ARGS+=(
        --enable-hierarchical-cache
        --hicache-storage-backend "${SGLANG_HICACHE_STORAGE_BACKEND:-file}"
        --hicache-mem-layout "${SGLANG_HICACHE_MEM_LAYOUT:-page_first}"
        --hicache-io-backend "${SGLANG_HICACHE_IO_BACKEND:-direct}"
        --hicache-write-policy "${SGLANG_HICACHE_WRITE_POLICY:-write_through}"
    )
    [[ -n "$HICACHE_SIZE" ]] && ARGS+=(--hicache-size "$HICACHE_SIZE")
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-${HOME}/models/hicache}"
    mkdir -p "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"
    echo "HiCache: backend=${SGLANG_HICACHE_STORAGE_BACKEND:-file} dir=$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR size=${HICACHE_SIZE:-ratio-default}GB"
fi
[[ "${SGLANG_DISABLE_RADIX_CACHE:-}" == "true" ]] && echo "HiCache: disabled (--disable-radix-cache)"

SPEC_ALGO="${SGLANG_SPECULATIVE_ALGORITHM:-none}"
if [[ "$SPEC_ALGO" != "none" ]] && { [[ "$SPEC_ALGO" == "NEXTN" ]] || [[ -n "$DRAFT_MODEL" ]]; }; then
    ARGS+=(
        --speculative-algorithm "$SPEC_ALGO"
        --speculative-num-draft-tokens "${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
        --speculative-draft-model-quantization "${SGLANG_SPECULATIVE_DRAFT_QUANTIZATION:-unquant}"
        --speculative-draft-attention-backend "${SGLANG_SPECULATIVE_DRAFT_ATTENTION_BACKEND:-flashinfer}"
    )
    [[ -n "$DRAFT_MODEL" ]] && ARGS+=(--speculative-draft-model-path "$DRAFT_MODEL")
    [[ -n "${SGLANG_SPECULATIVE_NUM_STEPS:-}" ]] && ARGS+=(--speculative-num-steps "$SGLANG_SPECULATIVE_NUM_STEPS")
    [[ -n "${SGLANG_SPECULATIVE_EAGLE_TOPK:-}" ]] && ARGS+=(--speculative-eagle-topk "$SGLANG_SPECULATIVE_EAGLE_TOPK")
    [[ -n "${SGLANG_SPECULATIVE_ACCEPT_THRESHOLD_SINGLE:-}" ]] && ARGS+=(--speculative-accept-threshold-single "$SGLANG_SPECULATIVE_ACCEPT_THRESHOLD_SINGLE")
    [[ -n "${SGLANG_SPECULATIVE_ACCEPT_THRESHOLD_ACC:-}" ]] && ARGS+=(--speculative-accept-threshold-acc "$SGLANG_SPECULATIVE_ACCEPT_THRESHOLD_ACC")
    [[ -n "${SGLANG_SPECULATIVE_TOKEN_MAP:-}" && -f "${SGLANG_SPECULATIVE_TOKEN_MAP}" ]] && ARGS+=(--speculative-token-map "$SGLANG_SPECULATIVE_TOKEN_MAP")
fi

# gdn-mtp-cache-mode controls HiCache persistence of the MTP draft layer's
# GDN state. Must be passed even when spec is off — SGLang defaults to 'full'
# which crashes if no MTP draft is loaded.
[[ -n "${SGLANG_GDN_MTP_CACHE_MODE:-}" ]] && ARGS+=(--gdn-mtp-cache-mode "$SGLANG_GDN_MTP_CACHE_MODE")

[[ -n "${SGLANG_REASONING_PARSER:-}" ]] && ARGS+=(--reasoning-parser "$SGLANG_REASONING_PARSER")
[[ -n "${SGLANG_TOOL_CALL_PARSER:-}" ]] && ARGS+=(--tool-call-parser "$SGLANG_TOOL_CALL_PARSER")

[[ -n "${SGLANG_PLE_SAFETENSORS_DIR:-}" ]] && echo "PLE safetensors: $SGLANG_PLE_SAFETENSORS_DIR"

# Cap kernel-JIT parallelism: FlashInfer autotune spawns one cicc per candidate
# kernel; uncapped on all cores it is a documented host-RAM OOM path.
export MAX_JOBS="${MAX_JOBS:-8}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-4}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"


"${SGLANG_VENV}/bin/sglang" serve --model-type llm "${ARGS[@]}" \
    > "$LOGFILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
echo "Server PID: $SERVER_PID"

echo "Waiting for server to start..."
for i in $(seq 1 900); do
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

echo "ERROR: Server did not become ready within 900s. Check $LOGFILE"
tail -30 "$LOGFILE"
exit 1
