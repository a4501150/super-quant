#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

GGUF="${MODELS_DIR}/${MODEL_NAME}-UD-Q6_K.gguf"
RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
RESULT_FILE="${RESULTS_DIR}/concurrent_benchmark_$(date +%Y%m%d_%H%M%S).tsv"
TMPDIR_BENCH=$(mktemp -d)

PORT=8090
TOTAL_CTX=$((NATIVE_CTX / 2))
PROMPT="Write a detailed Python implementation of a red-black tree with insert, delete, and search operations. Include type hints and docstrings for all methods."
MAX_TOKENS=1024

SERVER_PID=""

log() { echo "$1" >&2; }

cleanup() {
    [[ -n "${SERVER_PID:-}" ]] && kill "${SERVER_PID}" 2>/dev/null || true
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    rm -rf "${TMPDIR_BENCH}"
}
trap cleanup EXIT

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2
}

start_server() {
    local parallel="$1"
    shift

    stop_server

    local args=(
        -m "${GGUF}"
        -ngl "${GPU_LAYERS}"
        -fa on
        -c "${TOTAL_CTX}"
        --parallel "${parallel}"
        --cache-type-k q8_0
        --cache-type-v q8_0
        -kvu
        --host 127.0.0.1
        --port "${PORT}"
        --threads "${THREADS}"
        --metrics
        --jinja
        "$@"
    )

    "${LLAMA_SERVER}" "${args[@]}" &>/dev/null &
    SERVER_PID=$!

    for _ in $(seq 1 90); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    log "ERROR: Server failed to start"
    stop_server
    return 1
}

fire_request() {
    local slot_id="$1"
    local outfile="${TMPDIR_BENCH}/resp_${slot_id}.json"
    curl -sf --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0.6,\"top_p\":0.95,\"stream\":false}" \
        -o "${outfile}" 2>/dev/null
}

run_config() {
    local label="$1"
    local parallel="$2"
    local spec_mode="$3"
    shift 3

    local extra_args=("$@")
    if [[ "${spec_mode}" == "mtp" ]]; then
        extra_args+=(--spec-type draft-mtp --spec-draft-n-max "${MTP_N_MAX}")
    elif [[ "${spec_mode}" == "dspark" ]]; then
        local draft="${DSPARK_DRAFT_GGUF:-${MODELS_DIR}/Qwen3.8-27B-DSpark-BF16.gguf}"
        extra_args+=(--spec-type draft-dspark --spec-draft-model "${draft}" --spec-draft-n-max 7 -ngld "${GPU_LAYERS}")
    fi

    log ""
    log "=== ${label} (parallel=${parallel}, spec=${spec_mode}) ==="

    if ! start_server "${parallel}" "${extra_args[@]}"; then
        log "  SKIPPED"
        return
    fi

    # Warmup
    log "  Warmup..."
    fire_request "warmup"
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "  SERVER CRASHED during warmup — skipping"
        stop_server
        return
    fi
    rm -f "${TMPDIR_BENCH}/resp_warmup.json"

    # Fire N concurrent requests
    log "  Firing ${parallel} concurrent requests..."
    rm -f "${TMPDIR_BENCH}"/resp_*.json
    local t_start
    t_start=$(date +%s.%N)

    local pids=()
    for i in $(seq 1 "${parallel}"); do
        fire_request "${i}" &
        pids+=($!)
    done

    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || ((failed++))
    done

    local t_end
    t_end=$(date +%s.%N)
    local wall_secs
    wall_secs=$(python3 -c "print(f'{${t_end} - ${t_start}:.2f}')")

    # Parse results
    local total_tokens=0
    local tps_sum=0
    local tps_count=0
    local draft_total=0
    local draft_accepted_total=0

    for i in $(seq 1 "${parallel}"); do
        local f="${TMPDIR_BENCH}/resp_${i}.json"
        if [[ ! -f "${f}" ]]; then
            log "    slot ${i}: MISSING"
            continue
        fi

        local tokens tps draft_n draft_acc
        tokens=$(python3 -c "import json; d=json.load(open('${f}')); print(d['usage'].get('completion_tokens',0))" 2>/dev/null || echo 0)
        tps=$(python3 -c "import json; d=json.load(open('${f}')); print(f\"{d.get('timings',{}).get('predicted_per_second',0):.1f}\")" 2>/dev/null || echo "0.0")
        draft_n=$(python3 -c "import json; d=json.load(open('${f}')); print(d.get('timings',{}).get('draft_n',0))" 2>/dev/null || echo 0)
        draft_acc=$(python3 -c "import json; d=json.load(open('${f}')); print(d.get('timings',{}).get('draft_n_accepted',0))" 2>/dev/null || echo 0)

        total_tokens=$((total_tokens + tokens))
        tps_sum=$(python3 -c "print(${tps_sum} + ${tps})")
        tps_count=$((tps_count + 1))
        draft_total=$((draft_total + draft_n))
        draft_accepted_total=$((draft_accepted_total + draft_acc))

        log "    slot ${i}: ${tps} t/s, ${tokens} tokens"
    done

    local req_tps_avg="0.0"
    local aggregate_tps="0.0"
    local accept_rate="N/A"

    if (( tps_count > 0 )); then
        req_tps_avg=$(python3 -c "print(f'{${tps_sum} / ${tps_count}:.1f}')")
        aggregate_tps=$(python3 -c "print(f'{${total_tokens} / ${wall_secs}:.1f}')")
    fi

    if (( draft_total > 0 )); then
        accept_rate=$(python3 -c "print(f'{${draft_accepted_total}/${draft_total}*100:.1f}%')")
    fi

    log "  >>> per-req avg: ${req_tps_avg} t/s | aggregate: ${aggregate_tps} t/s | accept: ${accept_rate} | wall: ${wall_secs}s | tokens: ${total_tokens}"

    # Append to TSV
    echo -e "${label}\t${parallel}\t${spec_mode}\t${req_tps_avg}\t${aggregate_tps}\t${accept_rate}\t${wall_secs}\t${total_tokens}" >> "${RESULT_FILE}"

    stop_server
}

# Header
log "=== Concurrent Throughput Benchmark ==="
log "Model:   $(basename "${GGUF}")"
log "Context: ${TOTAL_CTX} total (shared across slots)"
log "Prompt:  ${#PROMPT} chars, max_tokens=${MAX_TOKENS}"
log "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
log ""

echo -e "config\tparallel\tspec\treq_tps_avg\taggregate_tps\taccept_rate\twall_secs\ttotal_tokens" > "${RESULT_FILE}"

# Baseline: no speculative decoding
for p in 1 2 3 5; do
    run_config "baseline" "${p}" "none"
done

# DSpark
for p in 1 2 3 5; do
    run_config "dspark" "${p}" "dspark"
done

log ""
log "=== Benchmark Complete ==="
log "Results: ${RESULT_FILE}"
log ""

# Print summary table
log "config          parallel  spec     req_avg  agg_tps  accept   wall_s  tokens"
log "-------------- --------  ------   -------  -------  ------   ------  ------"
while IFS=$'\t' read -r cfg par spec ravg agg acc wall tok; do
    [[ "${cfg}" == "config" ]] && continue
    printf "%-14s  %8s  %-6s   %7s  %7s  %6s  %6s  %6s\n" "${cfg}" "${par}" "${spec}" "${ravg}" "${agg}" "${acc}" "${wall}" "${tok}" >&2
done < "${RESULT_FILE}"
