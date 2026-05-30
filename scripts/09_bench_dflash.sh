#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

GGUF="${MODELS_DIR}/${MODEL_NAME}-UD-Q6_K.gguf"
DRAFT="${DFLASH_DRAFT_GGUF}"
RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
RESULT_FILE="${RESULTS_DIR}/dflash_benchmark_$(date +%Y%m%d_%H%M%S).tsv"

PORT=8090
REPS=3
CTX=32768
MAX_TOKENS=512

PROMPT_MATH="Prove that the sum of the first n odd numbers equals n squared. Show the proof step by step."
PROMPT_CODE="Write a Python implementation of a red-black tree with insert, delete, and search operations. Include type hints."
PROMPT_TOOL='You are a home assistant. The user says: "Turn on the living room lights and set them to 50% brightness." Respond with the appropriate function calls in JSON format.'
PROMPT_CHAT="What are the key differences between transformer and SSM architectures? Explain briefly."

SERVER_PID=""

log() { echo "$1" >&2; }

cleanup() {
    [[ -n "${SERVER_PID:-}" ]] && kill "${SERVER_PID}" 2>/dev/null || true
    fuser -k "${PORT}/tcp" 2>/dev/null || true
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
    shift

    stop_server

    local args=(
        -m "${GGUF}"
        -ngl "${GPU_LAYERS}"
        -fa on
        -c "${CTX}"
        --cache-type-k q8_0
        --cache-type-v q8_0
        --host 127.0.0.1
        --port "${PORT}"
        --threads "${THREADS}"
        --metrics
        --jinja
        "$@"
    )

    log "  Starting server: ${label}"
    "${LLAMA_SERVER}" "${args[@]}" &>/dev/null &
    SERVER_PID=$!

    for _ in $(seq 1 90); do
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

    curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
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
draft_n = timings.get("draft_n", 0)
draft_acc = timings.get("draft_n_accepted", 0)
accept_rate = f"{draft_acc/draft_n*100:.1f}" if draft_n > 0 else "N/A"
avg_accepted = f"{draft_acc/max(1,tokens):.2f}" if draft_n > 0 else "N/A"
print(f"{tps:.1f}\t{tokens}\t{draft_n}\t{draft_acc}\t{accept_rate}\t{avg_accepted}")
PYEOF
    rm -f "${tmpres}"
}

run_bench() {
    local config_label="$1"
    local thinking="$2"
    local prompt_label="$3"
    local prompt="$4"

    local speeds=()
    local all_results=()

    for rep in $(seq 1 "${REPS}"); do
        local result
        result=$(run_request "${prompt}" "${thinking}")
        if [[ "${result}" == "FAIL" ]]; then
            log "    rep ${rep}: FAILED"
            continue
        fi
        local tps tokens draft_n draft_acc accept_rate avg_acc
        IFS=$'\t' read -r tps tokens draft_n draft_acc accept_rate avg_acc <<< "${result}"
        log "    rep ${rep}: ${tps} t/s, ${tokens} tok, accept=${accept_rate}% (${draft_acc}/${draft_n})"
        speeds+=("${tps}")
        all_results+=("${result}")
    done

    if (( ${#speeds[@]} > 0 )); then
        local joined
        joined=$(IFS=,; echo "${speeds[*]}")
        local avg_tps
        avg_tps=$(python3 -c "s=[${joined}]; print(f'{sum(s)/len(s):.1f}')")

        # Use last result's acceptance data for the TSV row
        local last="${all_results[-1]}"
        local tps tokens draft_n draft_acc accept_rate avg_acc
        IFS=$'\t' read -r tps tokens draft_n draft_acc accept_rate avg_acc <<< "${last}"

        log "    >>> avg: ${avg_tps} t/s (${#speeds[@]}/${REPS} runs)"
        echo -e "${config_label}\t${thinking}\t${prompt_label}\t${avg_tps}\t${accept_rate}\t${avg_acc}\t${tokens}" >> "${RESULT_FILE}"
    fi
}

run_config() {
    local label="$1"
    shift
    local server_args=("$@")

    log ""
    log "=== ${label} ==="

    if ! start_server "${label}" "${server_args[@]}"; then
        log "  SKIPPED"
        return
    fi

    # Warmup
    log "  Warmup..."
    run_request "${PROMPT_CHAT}" "false" >/dev/null 2>&1

    for thinking in false true; do
        local think_label
        [[ "${thinking}" == "true" ]] && think_label="think" || think_label="nothink"
        log ""
        log "  --- thinking=${thinking} ---"

        log "  [math]"
        run_bench "${label}" "${thinking}" "math" "${PROMPT_MATH}"

        log "  [code]"
        run_bench "${label}" "${thinking}" "code" "${PROMPT_CODE}"

        log "  [tool]"
        run_bench "${label}" "${thinking}" "tool" "${PROMPT_TOOL}"

        log "  [chat]"
        run_bench "${label}" "${thinking}" "chat" "${PROMPT_CHAT}"
    done

    stop_server
}

# ============================================================

if [[ ! -f "${GGUF}" ]]; then
    echo "ERROR: Target GGUF not found: ${GGUF}"
    exit 1
fi

if [[ ! -f "${DRAFT}" ]]; then
    echo "ERROR: DFlash draft not found: ${DRAFT}"
    echo "Run: make download-dflash"
    exit 1
fi

log "=== DFlash Benchmark ==="
log "Target:  $(basename "${GGUF}")"
log "Draft:   $(basename "${DRAFT}")"
log "Reps:    ${REPS}, Context: ${CTX}, Max tokens: ${MAX_TOKENS}"
log "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
log ""

echo -e "config\tthinking\tprompt\tavg_tps\taccept_rate\tavg_accepted\ttokens" > "${RESULT_FILE}"

# 1. Baseline (no speculative decoding)
run_config "baseline" \
    --reasoning on \
    --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'

# 2. MTP
run_config "mtp" \
    --reasoning on \
    --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
    --spec-type draft-mtp \
    --spec-draft-n-max "${MTP_N_MAX}"

# 3. DFlash — sweep n_spec to find optimal value (community says 4-8 for Qwen3.6)
for N_SPEC in 4 8 15; do
    run_config "dflash-n${N_SPEC}" \
        --reasoning on \
        --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
        --spec-type dflash \
        -md "${DRAFT}" \
        --spec-draft-n-max "${N_SPEC}" \
        --spec-draft-ngl "${GPU_LAYERS}"
done

log ""
log "=== Benchmark Complete ==="
log "Results: ${RESULT_FILE}"
log ""

# Print summary table
log "config          thinking  prompt  avg_tps  accept%  avg_acc  tokens"
log "--------------  --------  ------  -------  -------  -------  ------"
while IFS=$'\t' read -r cfg think prompt tps acc avg_acc tok; do
    [[ "${cfg}" == "config" ]] && continue
    printf "%-14s  %8s  %6s  %7s  %7s  %7s  %6s\n" "${cfg}" "${think}" "${prompt}" "${tps}" "${acc}" "${avg_acc}" "${tok}" >&2
done < "${RESULT_FILE}"
