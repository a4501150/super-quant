#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

PORT="${SGLANG_PORT:-8888}"
RESULTS_DIR="${PROJECT_DIR}/results"
RESULT_FILE="${RESULTS_DIR}/sglang_benchmark_$(date +%Y%m%d_%H%M%S).json"
SERVED_NAME="${MODEL_ALIAS:-qwen3.8-27b}"

REPS=3

mkdir -p "${RESULTS_DIR}"

PROMPTS=(
    "Write a detailed Python implementation of a red-black tree with insert, delete, and search operations. Include type hints and docstrings for all methods."
    "Solve the integral of x^3 * e^(-x^2) from 0 to infinity. Show all steps."
    "Write a Rust async web server that handles /api/users CRUD with SQLite, error handling, and middleware for auth tokens."
    "Explain the architecture of a transformer model, including multi-head attention, positional encoding, and layer normalization. Use mathematical notation."
)

CONCURRENCY_LEVELS=(1 2 4 8)

log() { echo "$1" >&2; }

check_server() {
    curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1
}

run_single_request() {
    local prompt="$1"
    local response
    response=$(curl -sf --max-time 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(cat <<REQEOF
{
    "model": "${SERVED_NAME}",
    "messages": [{"role": "user", "content": "${prompt}"}],
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "stream": false
}
REQEOF
)") || { echo "{}"; return 1; }
    echo "${response}"
}

extract_metrics() {
    local response="$1"
    python3 -c "
import sys, json
r = json.loads('''${response}''')
u = r.get('usage', {})
print(json.dumps({
    'completion_tokens': u.get('completion_tokens', 0),
    'prompt_tokens': u.get('prompt_tokens', 0),
    'total_tokens': u.get('total_tokens', 0),
}))
"
}

# ---------------------------------------------------------------
# Phase 1: Single-user throughput
# ---------------------------------------------------------------
single_user_bench() {
    log ""
    log "=== Phase 1: Single-User Throughput ==="

    local results="[]"
    for prompt_idx in "${!PROMPTS[@]}"; do
        local prompt="${PROMPTS[$prompt_idx]}"
        local label="prompt_${prompt_idx}"
        log "  ${label}: ${prompt:0:60}..."

        local speeds=()
        for rep in $(seq 1 ${REPS}); do
            local start_ns end_ns response usage tokens elapsed tps
            start_ns=$(date +%s%N)
            response=$(run_single_request "${prompt}")
            end_ns=$(date +%s%N)

            tokens=$(echo "${response}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
            elapsed=$(python3 -c "print(f'{(${end_ns}-${start_ns})/1e9:.3f}')")

            if [[ "${tokens}" != "0" ]]; then
                tps=$(python3 -c "print(f'{${tokens}/${elapsed}:.1f}')")
                speeds+=("${tps}")
                log "    rep ${rep}: ${tps} t/s (${tokens} tokens, ${elapsed}s)"
            else
                log "    rep ${rep}: FAILED"
            fi
        done

        if (( ${#speeds[@]} > 0 )); then
            local joined avg
            joined=$(IFS=,; echo "${speeds[*]}")
            avg=$(python3 -c "s=[${joined}]; print(f'{sum(s)/len(s):.1f}')")
            log "    >>> avg: ${avg} t/s"
            results=$(python3 -c "
import json
r = json.loads('${results}')
r.append({'prompt_idx': ${prompt_idx}, 'avg_tps': ${avg}, 'runs': [${joined}]})
print(json.dumps(r))
")
        fi
    done
    echo "${results}"
}

# ---------------------------------------------------------------
# Phase 2: Concurrent throughput
# ---------------------------------------------------------------
concurrent_bench() {
    log ""
    log "=== Phase 2: Concurrent Throughput ==="

    local prompt="${PROMPTS[0]}"
    local results="[]"

    for n_concurrent in "${CONCURRENCY_LEVELS[@]}"; do
        log "  Concurrency: ${n_concurrent}"

        local pids=()
        local tmpdir
        tmpdir=$(mktemp -d)
        local start_ns
        start_ns=$(date +%s%N)

        for i in $(seq 1 ${n_concurrent}); do
            (
                resp=$(run_single_request "${prompt}")
                echo "${resp}" > "${tmpdir}/resp_${i}.json"
            ) &
            pids+=($!)
        done

        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        local end_ns
        end_ns=$(date +%s%N)
        local wall_s
        wall_s=$(python3 -c "print(f'{(${end_ns}-${start_ns})/1e9:.3f}')")

        local total_tokens=0
        local completed=0
        for f in "${tmpdir}"/resp_*.json; do
            [[ -f "$f" ]] || continue
            local t
            t=$(python3 -c "import json; print(json.load(open('${f}')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
            total_tokens=$((total_tokens + t))
            [[ "$t" != "0" ]] && completed=$((completed + 1))
        done

        local agg_tps per_req_tps
        agg_tps=$(python3 -c "print(f'{${total_tokens}/${wall_s}:.1f}')")
        per_req_tps=$(python3 -c "print(f'{${total_tokens}/${wall_s}/${n_concurrent}:.1f}') if ${n_concurrent} > 0 else '0'")

        log "    ${completed}/${n_concurrent} completed | ${total_tokens} tokens | ${wall_s}s wall | ${agg_tps} agg t/s | ${per_req_tps} per-req t/s"

        results=$(python3 -c "
import json
r = json.loads('${results}')
r.append({
    'concurrency': ${n_concurrent},
    'completed': ${completed},
    'total_tokens': ${total_tokens},
    'wall_seconds': ${wall_s},
    'aggregate_tps': ${agg_tps},
    'per_request_tps': ${per_req_tps},
})
print(json.dumps(r))
")

        rm -rf "${tmpdir}"
    done
    echo "${results}"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if ! check_server; then
    echo "ERROR: SGLang server not running on port ${PORT}"
    echo "Run: make serve-sglang"
    exit 1
fi

log "=== SGLang Benchmark ==="
log "Server: http://127.0.0.1:${PORT}"
log "Model:  ${SERVED_NAME}"
log "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
log ""

SINGLE_RESULTS=$(single_user_bench)
CONCURRENT_RESULTS=$(concurrent_bench)

python3 -c "
import json, sys
result = {
    'server': 'sglang',
    'model': '${SERVED_NAME}',
    'gpu': '$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)',
    'single_user': json.loads('''${SINGLE_RESULTS}'''),
    'concurrent': json.loads('''${CONCURRENT_RESULTS}'''),
}
with open('${RESULT_FILE}', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
"

log ""
log "Results saved: ${RESULT_FILE}"
