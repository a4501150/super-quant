#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

GGUF="${MODELS_DIR}/${MODEL_NAME}-UD-Q6_K.gguf"
RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
RESULT_FILE="${RESULTS_DIR}/kv_type_benchmark_$(date +%Y%m%d_%H%M%S).tsv"
TMPDIR_BENCH=$(mktemp -d)

PORT=8090
REPS=3
CTX=524288
PARALLEL=5
MAX_TOKENS=512

PROMPT_MATH="Prove that the sum of the first n odd numbers equals n squared. Show the proof step by step."
PROMPT_CODE="Write a Python implementation of a red-black tree with insert, delete, and search operations. Include type hints."
PROMPT_TOOL='You are a home assistant. The user says: "Turn on the living room lights and set them to 50% brightness." Respond with the appropriate function calls in JSON format.'
PROMPT_CHAT="What are the key differences between transformer and SSM architectures? Explain briefly."
PROMPT_CONCURRENT="Explain the concept of dynamic programming and provide a Python example solving the longest common subsequence problem."

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
    local label="$1"
    local kv_k="$2"
    local kv_v="$3"

    stop_server

    local args=(
        -m "${GGUF}"
        -ngl "${GPU_LAYERS}"
        -fa on
        -c "${CTX}"
        --parallel "${PARALLEL}"
        --cache-type-k "${kv_k}"
        --cache-type-v "${kv_v}"
        -kvu
        --host 127.0.0.1
        --port "${PORT}"
        --threads "${THREADS}"
        --metrics
        --jinja
        --reasoning on
        --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
    )

    # YaRN context extension
    if (( CTX > NATIVE_CTX )); then
        local rope_scale
        rope_scale=$(python3 -c "print(round(${CTX} / ${NATIVE_CTX}, 6))")
        args+=(
            --rope-scaling yarn
            --rope-scale "${rope_scale}"
            --yarn-orig-ctx "${NATIVE_CTX}"
            --override-kv "${GGUF_ARCH_KEY}.context_length=int:${CTX}"
        )
    fi

    log "  Starting server: ${label} (K=${kv_k}, V=${kv_v})"
    log "  Config: ctx=${CTX}, parallel=${PARALLEL}, unified KV, YaRN"
    "${LLAMA_SERVER}" "${args[@]}" &>/dev/null &
    SERVER_PID=$!

    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            log "  Server ready (PID: ${SERVER_PID})"
            return 0
        fi
        sleep 1
    done

    log "  ERROR: Server failed to start"
    stop_server
    return 1
}

get_vram() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

run_request() {
    local prompt="$1"
    local thinking="$2"
    local tmpreq tmpres
    tmpreq=$(mktemp)
    tmpres=$(mktemp)

    local enable_thinking="False"
    [[ "${thinking}" == "true" ]] && enable_thinking="True"

    python3 - "${MODEL_NAME}" "${prompt}" "${MAX_TOKENS}" "${enable_thinking}" "${tmpreq}" <<'PYEOF'
import json, sys
model, prompt, max_tok, thinking, outfile = sys.argv[1:6]
req = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tok),
    "temperature": 0.6,
    "top_p": 0.95,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": thinking == "True", "preserve_thinking": True}
}
with open(outfile, "w") as f:
    json.dump(req, f)
PYEOF

    curl -sf --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d @"${tmpreq}" \
        -o "${tmpres}" 2>/dev/null
    local rc=$?
    rm -f "${tmpreq}"

    if [[ ${rc} -ne 0 ]] || [[ ! -s "${tmpres}" ]]; then
        rm -f "${tmpres}"
        echo "FAIL"
        return
    fi

    python3 - "${tmpres}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
usage = d.get("usage", {})
timings = d.get("timings", {})
tokens = usage.get("completion_tokens", 0)
tps = timings.get("predicted_per_second", 0)
pp_tps = timings.get("prompt_per_second", 0)
print(f"{tps:.1f}\t{pp_tps:.1f}\t{tokens}")
PYEOF
    rm -f "${tmpres}"
}

# Single-user sequential benchmark
run_bench_single() {
    local config_label="$1"
    local vram="$2"
    local thinking="$3"
    local prompt_label="$4"
    local prompt="$5"

    local speeds=()

    for rep in $(seq 1 "${REPS}"); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "    SERVER CRASHED — aborting"
            return
        fi
        local result
        result=$(run_request "${prompt}" "${thinking}")
        if [[ "${result}" == "FAIL" ]]; then
            log "    rep ${rep}: FAILED"
            continue
        fi
        local tps pp_tps tokens
        IFS=$'\t' read -r tps pp_tps tokens <<< "${result}"
        log "    rep ${rep}: ${tps} t/s (pp: ${pp_tps} t/s), ${tokens} tok"
        speeds+=("${tps}")
    done

    if (( ${#speeds[@]} > 0 )); then
        local joined
        joined=$(IFS=,; echo "${speeds[*]}")
        local avg_tps
        avg_tps=$(python3 -c "s=[${joined}]; print(f'{sum(s)/len(s):.1f}')")
        log "    >>> avg: ${avg_tps} t/s (${#speeds[@]}/${REPS} runs)"
        echo -e "${config_label}\t${vram}\tseq-1\t${thinking}\t${prompt_label}\t${avg_tps}\t-\t-\t${#speeds[@]}" >> "${RESULT_FILE}"
    fi
}

# Concurrent benchmark: fire N requests in parallel
run_bench_concurrent() {
    local config_label="$1"
    local vram="$2"
    local n_concurrent="$3"

    log ""
    log "  --- concurrent=${n_concurrent} ---"

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "    SERVER CRASHED — skipping"
        return
    fi

    rm -f "${TMPDIR_BENCH}"/resp_*.json

    local tmpreq
    tmpreq=$(mktemp)
    python3 - "${MODEL_NAME}" "${PROMPT_CONCURRENT}" "${MAX_TOKENS}" "${tmpreq}" <<'PYEOF'
import json, sys
model, prompt, max_tok, outfile = sys.argv[1:5]
req = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tok),
    "temperature": 0.6,
    "top_p": 0.95,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}
}
with open(outfile, "w") as f:
    json.dump(req, f)
PYEOF

    local t_start
    t_start=$(date +%s.%N)

    local pids=()
    for i in $(seq 1 "${n_concurrent}"); do
        curl -sf --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d @"${tmpreq}" \
            -o "${TMPDIR_BENCH}/resp_${i}.json" 2>/dev/null &
        pids+=($!)
    done

    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || ((failed++))
    done
    rm -f "${tmpreq}"

    local t_end
    t_end=$(date +%s.%N)
    local wall_secs
    wall_secs=$(python3 -c "print(f'{${t_end} - ${t_start}:.2f}')")

    local total_tokens=0
    local tps_sum=0
    local tps_count=0

    for i in $(seq 1 "${n_concurrent}"); do
        local f="${TMPDIR_BENCH}/resp_${i}.json"
        if [[ ! -f "${f}" ]] || [[ ! -s "${f}" ]]; then
            log "    slot ${i}: MISSING/EMPTY"
            continue
        fi

        local tokens tps
        tokens=$(python3 -c "import json; d=json.load(open('${f}')); print(d['usage'].get('completion_tokens',0))" 2>/dev/null || echo 0)
        tps=$(python3 -c "import json; d=json.load(open('${f}')); print(f\"{d.get('timings',{}).get('predicted_per_second',0):.1f}\")" 2>/dev/null || echo "0.0")

        total_tokens=$((total_tokens + tokens))
        tps_sum=$(python3 -c "print(${tps_sum} + ${tps})")
        tps_count=$((tps_count + 1))

        log "    slot ${i}: ${tps} t/s, ${tokens} tokens"
    done

    if (( tps_count > 0 )); then
        local req_tps_avg aggregate_tps
        req_tps_avg=$(python3 -c "print(f'{${tps_sum} / ${tps_count}:.1f}')")
        aggregate_tps=$(python3 -c "print(f'{${total_tokens} / ${wall_secs}:.1f}')")

        # Measure VRAM under concurrent load
        local vram_concurrent
        vram_concurrent=$(get_vram)

        log "    >>> per-req: ${req_tps_avg} t/s | aggregate: ${aggregate_tps} t/s | wall: ${wall_secs}s | tokens: ${total_tokens} | VRAM: ${vram_concurrent} MiB"
        echo -e "${config_label}\t${vram_concurrent}\tconc-${n_concurrent}\ttrue\tmixed\t${req_tps_avg}\t${aggregate_tps}\t${wall_secs}\t${tps_count}" >> "${RESULT_FILE}"
    fi

    rm -f "${TMPDIR_BENCH}"/resp_*.json
}

run_config() {
    local label="$1"
    local kv_k="$2"
    local kv_v="$3"

    log ""
    log "=========================================="
    log "=== ${label} (K=${kv_k}, V=${kv_v}) ==="
    log "=========================================="

    if ! start_server "${label}" "${kv_k}" "${kv_v}"; then
        log "  SKIPPED (server failed to start — may be OOM)"
        return
    fi

    # Warmup
    log "  Warmup..."
    run_request "${PROMPT_CHAT}" "false" >/dev/null 2>&1

    # VRAM after warmup (idle, KV allocated but mostly empty)
    local vram
    vram=$(get_vram)
    log "  VRAM (idle): ${vram} MiB"

    # Phase 1: Single-user sequential (measures per-request latency)
    log ""
    log "  === Phase 1: Single-user sequential ==="

    for thinking in false true; do
        log ""
        log "  --- thinking=${thinking} ---"

        log "  [math]"
        run_bench_single "${label}" "${vram}" "${thinking}" "math" "${PROMPT_MATH}"

        log "  [code]"
        run_bench_single "${label}" "${vram}" "${thinking}" "code" "${PROMPT_CODE}"

        log "  [tool]"
        run_bench_single "${label}" "${vram}" "${thinking}" "tool" "${PROMPT_TOOL}"

        log "  [chat]"
        run_bench_single "${label}" "${vram}" "${thinking}" "chat" "${PROMPT_CHAT}"
    done

    # Phase 2: Concurrent throughput (measures aggregate throughput under load)
    log ""
    log "  === Phase 2: Concurrent throughput ==="

    for n in 1 3 5; do
        run_bench_concurrent "${label}" "${vram}" "${n}"
    done

    stop_server
}

# ============================================================

if [[ ! -f "${GGUF}" ]]; then
    echo "ERROR: Target GGUF not found: ${GGUF}"
    exit 1
fi

log "=== KV Cache Type Benchmark (Production Config) ==="
log "Model:    $(basename "${GGUF}")"
log "Config:   ctx=${CTX}, parallel=${PARALLEL}, unified KV, YaRN"
log "Reps:     ${REPS}, Max tokens: ${MAX_TOKENS}"
log "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
log ""

echo -e "config\tvram_mib\tmode\tthinking\tprompt\tavg_tps\taggregate_tps\twall_secs\truns" > "${RESULT_FILE}"

run_config "q8_0-q8_0" "q8_0" "q8_0"
run_config "q8_0-turbo4" "q8_0" "turbo4"

log ""
log "=== Benchmark Complete ==="
log "Results: ${RESULT_FILE}"
log ""

# Summary table
log "config          vram    mode    think   prompt  avg_tps  agg_tps  wall_s  runs"
log "--------------  ------  ------  ------  ------  -------  -------  ------  ----"
while IFS=$'\t' read -r cfg vram mode think prompt tps agg wall runs; do
    [[ "${cfg}" == "config" ]] && continue
    printf "%-14s  %6s  %6s  %6s  %6s  %7s  %7s  %6s  %4s\n" \
        "${cfg}" "${vram}" "${mode}" "${think}" "${prompt}" "${tps}" "${agg}" "${wall}" "${runs}" >&2
done < "${RESULT_FILE}"
