#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

GGUF="${MODELS_DIR}/${MODEL_NAME}-UD-Q8_0.gguf"
DSPARK_DRAFT="${DSPARK_DRAFT_GGUF:-${MODELS_DIR}/Qwen3.8-27B-DSpark-BF16.gguf}"
RESULTS_DIR="${PROJECT_DIR}/results"
RESULT_FILE="${RESULTS_DIR}/spec_benchmark_$(date +%Y%m%d_%H%M%S).log"

PORT=8090
REPS=3
CTX=32768
KV_TYPE_K="${KV_TYPE_K:-}"
KV_TYPE_V="${KV_TYPE_V:-}"
PROMPT="Write a detailed Python implementation of a red-black tree with insert, delete, and search operations. Include type hints and docstrings for all methods."

log() { echo "$1" | tee -a "${RESULT_FILE}"; }

start_server() {
    local label="$1"
    shift
    local args=(
        -m "${GGUF}"
        -ngl "${GPU_LAYERS}"
        -fa on
        -c "${CTX}"
        --host 127.0.0.1
        --port "${PORT}"
        --threads "${THREADS}"
        --metrics
        --jinja
        "$@"
    )

    [[ -n "${KV_TYPE_K}" ]] && args+=(--cache-type-k "${KV_TYPE_K}")
    [[ -n "${KV_TYPE_V}" ]] && args+=(--cache-type-v "${KV_TYPE_V}")

    log "  Starting server: ${label}"
    "${LLAMA_SERVER}" "${args[@]}" &>/dev/null &
    SERVER_PID=$!

    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            log "  Server ready (PID: ${SERVER_PID})"
            return 0
        fi
        sleep 1
    done

    log "  ERROR: Server failed to start"
    kill ${SERVER_PID} 2>/dev/null || true
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        SERVER_PID=""
    fi
    for i in $(seq 1 10); do
        ss -tlnp 2>/dev/null | grep -q ":${PORT} " || break
        sleep 1
    done
}

run_request() {
    local rep="$1"

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "SERVER CRASHED" >&2
        return 1
    fi

    local response
    response=$(curl -sf --max-time 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(cat <<REQEOF
{
    "model": "${MODEL_NAME}",
    "messages": [{"role": "user", "content": "${PROMPT}"}],
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "stream": false
}
REQEOF
)") || { echo "CURL FAILED (server may have crashed)" >&2; return 1; }

    local tokens tg_speed pp_speed draft_n draft_accepted
    tokens=$(echo "${response}" | python3 -c "import sys,json; t=json.load(sys.stdin)['usage']; print(t.get('completion_tokens', 0))")
    tg_speed=$(echo "${response}" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(f\"{t.get('predicted_per_second', 0):.1f}\")")
    pp_speed=$(echo "${response}" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(f\"{t.get('prompt_per_second', 0):.1f}\")")
    draft_n=$(echo "${response}" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(t.get('draft_n', 0))")
    draft_accepted=$(echo "${response}" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(t.get('draft_n_accepted', 0))")

    local accept_rate="N/A"
    if [[ "${draft_n}" != "0" ]]; then
        accept_rate=$(python3 -c "print(f'{${draft_accepted}/${draft_n}*100:.1f}%')")
    fi

    echo "    rep ${rep}: ${tg_speed} t/s | ${tokens} tokens | draft ${draft_accepted}/${draft_n} (${accept_rate})" | tee -a "${RESULT_FILE}" >&2
    echo "${tg_speed}"
}

run_bench() {
    local label="$1"
    shift

    log ""
    log "--- ${label} ---"

    if ! start_server "${label}" "$@"; then
        log "  SKIPPED (server failed to start)"
        stop_server
        return
    fi

    local speeds=()
    for rep in $(seq 1 ${REPS}); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "    SERVER CRASHED — aborting config"
            break
        fi
        local speed
        speed=$(run_request "${rep}" || echo "0.0")
        if [[ "${speed}" != "0.0" ]]; then
            speeds+=("${speed}")
        else
            log "    rep ${rep}: FAILED"
        fi
    done

    if (( ${#speeds[@]} > 0 )); then
        local joined
        joined=$(IFS=,; echo "${speeds[*]}")
        local avg
        avg=$(python3 -c "s=[${joined}]; print(f'{sum(s)/len(s):.1f}')")
        log "    >>> avg: ${avg} t/s (${#speeds[@]}/${REPS} runs)"
    fi

    stop_server
}

trap stop_server EXIT

log "=== Speculative Decoding Benchmark (Server Mode) ==="
log "Target:  $(basename ${GGUF})"
log "DSpark:  $(basename ${DSPARK_DRAFT})"
log "Reps:    ${REPS}, Context: ${CTX}"
log "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
log "Mode:    --reasoning on, reasoning_effort=medium"

run_bench "1. No speculative decoding (baseline)"

run_bench "2. MTP (draft-mtp, n_max=3)" \
    --spec-type draft-mtp \
    --spec-draft-n-max 3

if [[ -f "${DSPARK_DRAFT}" ]]; then
    run_bench "3. DSpark (n_max=7)" \
        --spec-type draft-dspark \
        --spec-draft-model "${DSPARK_DRAFT}" \
        --spec-draft-n-max 7 \
        -ngld "${GPU_LAYERS}"
else
    log "WARNING: DSpark draft not found: ${DSPARK_DRAFT}"
fi

log ""
log "=== Benchmark complete ==="
log "Results: ${RESULT_FILE}"
