#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

PORT="${SGLANG_PORT:-8888}"
RESULTS_DIR="${PROJECT_DIR}/results"
RESULT_FILE="${RESULTS_DIR}/sglang_benchmark_$(date +%Y%m%d_%H%M%S).json"
SERVED_NAME="${MODEL_ALIAS:-qwen3.8-27b}"

REPS="${REPS:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-8192 32768 131072}"

mkdir -p "${RESULTS_DIR}"
RUNDIR=$(mktemp -d)
DIGEST_SERVER_ACTIVE=false
cleanup() {
    local status=$?
    trap - EXIT
    if [[ "${status}" -ne 0 ]]; then
        # Keep the Phase 4 diagnostic logs next to the result JSON for post-mortem.
        local base="${RESULT_FILE%.json}" name
        for name in correct_cold_server.log correct_restore_server.log; do
            if [[ -f "${RUNDIR}/${name}" ]]; then
                cp "${RUNDIR}/${name}" "${base}.${name}" 2>/dev/null || true
            fi
        done
        local metrics answer
        for metrics in "${RUNDIR}"/*.json; do
            answer="${metrics%.json}.ans"
            if [[ -f "${answer}" ]] && python3 -c \
                'import json, sys; sys.exit(bool(json.load(open(sys.argv[1])).get("ok")))' \
                "${metrics}" 2>/dev/null; then
                cp "${answer}" "${base}.$(basename "${answer}")" 2>/dev/null || true
            fi
        done
    fi
    rm -rf "${RUNDIR}"
    if [[ "${DIGEST_SERVER_ACTIVE}" == "true" ]]; then
        log "Restoring server without page digest logging after benchmark failure..."
        bash "${SCRIPT_DIR}/stop_sglang.sh" >&2 || true
        if ! bash "${SCRIPT_DIR}/serve_sglang.sh" >&2; then
            log "ERROR: failed to restore the normal SGLang server"
        fi
    fi
    exit "${status}"
}
trap cleanup EXIT

PROMPTS=(
    "Write a detailed Python implementation of a red-black tree with insert, delete, and search operations. Include type hints and docstrings for all methods."
    "Solve the integral of x^3 * e^(-x^2) from 0 to infinity. Show all steps."
    "Write a Rust async web server that handles /api/users CRUD with SQLite, error handling, and middleware for auth tokens."
)

CONCURRENCY_LEVELS=(3 6)

log() { echo "$1" >&2; }

check_server() {
    curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1
}

run_one() {
    local prompt="$1" out_json="$2" out_text="${3:-}"
    local args=(--url "http://127.0.0.1:${PORT}/v1/chat/completions"
                --model "${SERVED_NAME}" --prompt "${prompt}" --out-json "${out_json}")
    [[ -n "${out_text}" ]] && args+=(--text-out "${out_text}")
    args+=("${@:4}")
    if ! python3 "${SCRIPT_DIR}/bench_sglang_request.py" "${args[@]}" > /dev/null; then
        if [[ ! -s "${out_json}" ]]; then
            printf '{"ok":false,"error":"benchmark client failed without metrics"}\n' > "${out_json}"
        fi
    fi
}

log_metrics_line() {
    local jf="$1" label="$2"
    python3 - "$jf" "$label" <<'PYEOF' >&2
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"    {sys.argv[2]}: FAILED ({e})")
    raise SystemExit(0)
label = sys.argv[2]
if not m.get("ok"):
    print(f"    {label}: FAILED ({m.get('error', 'no usage')})")
else:
    d = m.get("decode_tps"); p = m.get("prefill_tps")
    pre = f"{p:.0f} t/s" if p else "n/a"
    dec = f"{d:.1f} t/s" if d else "n/a"
    print(f"    {label}: decode {dec} | prefill ~{pre} (TTFT {m['ttft_s']*1000:.0f} ms, "
          f"{m['cached_tokens']}/{m['prompt_tokens']} tok cached) | "
          f"{m['completion_tokens']} out tok in {m['wall_s']:.1f}s")
PYEOF
}

# ---------------------------------------------------------------
# Phase 1: Single-user throughput
# ---------------------------------------------------------------
single_user_bench() {
    log ""
    log "=== Phase 1: Single-User Throughput ==="
    for prompt_idx in "${!PROMPTS[@]}"; do
        local prompt="${PROMPTS[$prompt_idx]}"
        log "  prompt_${prompt_idx}: ${prompt:0:60}..."
        for rep in $(seq 1 ${REPS}); do
            local jf="${RUNDIR}/single_${prompt_idx}_${rep}.json"
            local tf="${RUNDIR}/single_${prompt_idx}_${rep}.txt"
            run_one "${prompt}" "${jf}" "${tf}"
            log_metrics_line "${jf}" "rep ${rep}"
            [[ -s "${tf}" ]] && sed 's/^/      /' "${tf}" >&2
            log ""
        done
    done
}

# ---------------------------------------------------------------
# Phase 2: Concurrent throughput
# ---------------------------------------------------------------
concurrent_bench() {
    log ""
    log "=== Phase 2: Concurrent Throughput ==="
    local prompt="${PROMPTS[0]}"

    for n_concurrent in "${CONCURRENCY_LEVELS[@]}"; do
        log "  Concurrency: ${n_concurrent}"
        local pids=()
        local start_ns
        start_ns=$(date +%s%N)
        for i in $(seq 1 ${n_concurrent}); do
            (
                run_one "${prompt}" "${RUNDIR}/conc_${n_concurrent}_${i}.json"
            ) &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        local end_ns wall_s
        end_ns=$(date +%s%N)
        wall_s=$(python3 -c "print(f'{(${end_ns}-${start_ns})/1e9:.3f}')")

        python3 - "${RUNDIR}" "${n_concurrent}" "${wall_s}" <<'PYEOF' >&2
import glob, json, os, sys
rundir, n, wall = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
files = sorted(glob.glob(os.path.join(rundir, f"conc_{n}_*.json")))
runs = []
for f in files:
    try:
        m = json.load(open(f))
    except Exception:
        continue
    if m.get("ok"):
        runs.append(m)
done = len(runs)
toks = sum(r["completion_tokens"] for r in runs)
agg = toks / wall if wall > 0 else 0
ttfts = sorted(r["ttft_s"] for r in runs)
def pct(xs, q):
    return xs[min(int(len(xs) * q), len(xs) - 1)] if xs else 0
med_ttft = pct(ttfts, 0.5)
max_ttft = ttfts[-1] if ttfts else 0
cache_hits = sum(1 for r in runs if r["cached_tokens"] > 0)
print(f"    {done}/{n} completed | {toks} tokens | {wall:.1f}s wall | "
      f"{agg:.1f} agg t/s | {agg/n:.1f} per-req t/s | "
      f"TTFT p50 {med_ttft*1000:.0f} ms / max {max_ttft*1000:.0f} ms | "
      f"{cache_hits} with cache hits")
PYEOF
        log ""
    done
}

# ---------------------------------------------------------------
# Phase 3: Long-prefill probe (cold prefill + warm cache-hit pair
# per target length; thinking off, 32 forced output tokens)
# ---------------------------------------------------------------
PROBE_DIR="${HOME}/models/hicache-probe"

prefill_probe() {
    log ""
    log "=== Phase 3: Long-Prefill Probe ==="
    for target in ${PREFILL_TOKENS}; do
        local pf="${PROBE_DIR}/prompt_${target}.txt"
        if [[ ! -f "$pf" ]]; then
            mkdir -p "$PROBE_DIR"
            python3 - "$pf" "$target" <<'PYEOF'
import sys, uuid
para = ("The history of coastal cartography in the North Atlantic shows a steady "
        "improvement in the accuracy of depth soundings and shoreline geometry, "
        "driven by navigational demand and state-sponsored survey traditions. ")
text = (f"Unique marker {uuid.uuid4().hex[:8]}: the lighthouse at Skerries fades at dusk. "
        + para * (int(sys.argv[2]) // 40 + 1))
with open(sys.argv[1], "w") as f:
    f.write(text + "\n\nQuestion: what unique marker appears near the start of the text above? Reply in at most ten words.")
PYEOF
            log "  (generated fresh probe text — cold run re-prefills)"
        else
            log "  (reusing probe text — warm run exercises cache/SSD restore)"
        fi
        log "  target ${target} tokens:"
        run_one "unused" "${RUNDIR}/prefill_cold_${target}.json" "${RUNDIR}/prefill_cold_${target}.txt" --prompt-file "$pf" --max-tokens 32 --ignore-eos --no-thinking
        log_metrics_line "${RUNDIR}/prefill_cold_${target}.json" "cold"
        [[ -s "${RUNDIR}/prefill_cold_${target}.txt" ]] && sed 's/^/      /' "${RUNDIR}/prefill_cold_${target}.txt" >&2
        run_one "unused" "${RUNDIR}/prefill_warm_${target}.json" "${RUNDIR}/prefill_warm_${target}.txt" --prompt-file "$pf" --max-tokens 32 --ignore-eos --no-thinking
        log_metrics_line "${RUNDIR}/prefill_warm_${target}.json" "warm"
        [[ -s "${RUNDIR}/prefill_warm_${target}.txt" ]] && sed 's/^/      /' "${RUNDIR}/prefill_warm_${target}.txt" >&2
        log ""
    done
}

# ---------------------------------------------------------------
# Phase 4: HiCache L1->L3 correctness (cold prefill -> pre-flush revisit so
# every probe page meets the write_through_selective two-hit admission ->
# in-process cache flush -> SSD restore with marker, cache-coverage, and
# page/transfer-digest validation)
# ---------------------------------------------------------------
CORRECT_TOKENS="${CORRECT_TOKENS:-32768 131072 196608}"
CORRECT_CONC_TOKENS="${CORRECT_CONC_TOKENS:-32768 65536 98304}"
CORRECT_MIN_HIT_RATIO="${CORRECT_MIN_HIT_RATIO:-0.9}"
# Concurrent write-through can omit tail pages under pressure; require enough of
# the prefix to exercise L3 while marker and digest checks prove correctness.
CORRECT_CONC_MIN_HIT_RATIO="${CORRECT_CONC_MIN_HIT_RATIO:-0.7}"
# Session-churn probes: model the operator workflow (system prompt edited
# between sessions -> radix tree re-roots on every edit; family variants share
# a deep prefix and branch mid-tree -> sibling state must not cross-contaminate).
CORRECT_SESSION_TOKENS="${CORRECT_SESSION_TOKENS:-49152}"
CORRECT_SESSION_SYS_TOKENS="${CORRECT_SESSION_SYS_TOKENS:-4096}"
CORRECT_SESSION_FAMILIES="${CORRECT_SESSION_FAMILIES:-3}"
CORRECT_SESSION_EDITS="${CORRECT_SESSION_EDITS:-3}"
# Near the L3 eviction watermark, older edited branches can lose their tail.
# Require a substantial restored prefix; marker and digest checks stay strict.
CORRECT_SESSION_MIN_HIT="${CORRECT_SESSION_MIN_HIT:-0.25}"
CORRECT_DIGESTS="${CORRECT_DIGESTS:-true}"
CORRECT_WRITE_SETTLE_SECONDS="${CORRECT_WRITE_SETTLE_SECONDS:-10}"
# Digest hashing can keep a large write-through queue active for several minutes.
CORRECT_FLUSH_TIMEOUT="${CORRECT_FLUSH_TIMEOUT:-900}"
CORRECT_FLUSH_SETTLE_SECONDS="${CORRECT_FLUSH_SETTLE_SECONDS:-2}"

gen_correct_probe() { # $1=path $2=target tokens
    python3 - "$1" "$2" <<'PYEOF'
import sys, uuid
para = ("The history of coastal cartography in the North Atlantic shows a steady "
        "improvement in the accuracy of depth soundings and shoreline geometry, "
        "driven by navigational demand and state-sponsored survey traditions. ")
quarters = max(int(sys.argv[2]) // 160, 1)
sections = []
for _ in range(4):
    marker = uuid.uuid4().hex[:8]
    sections.append(f"Unique marker {marker}: the lighthouse at Skerries fades at dusk. "
                    + para * quarters)
text = "\n\n".join(sections)
with open(sys.argv[1], "w") as f:
    f.write(text + "\n\nQuestion: list every unique marker that appears in the text above, in the order they appear. Answer with at most twenty words.")
PYEOF
}

# Session-churn probes modeling the operator workflow: a long multi-section
# system prompt, sessions re-rooted by section edits (cumulative across edit
# rounds), and a branch family sharing one deep prefix with divergent tails.
# Expected answer order: system agent-id marker, then system revision markers
# in section order, then user body markers, then the session tail marker.
gen_session_probes() { # $1=dir $2=suffix $3=body_tokens $4=sys_tokens $5=families $6=edits
    python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PYEOF'
import os, sys, uuid
dir_, suf = sys.argv[1], sys.argv[2]
body_tokens, sys_tokens = int(sys.argv[3]), int(sys.argv[4])
fams, edits = int(sys.argv[5]), int(sys.argv[6])
def hx(): return uuid.uuid4().hex[:8]
def paras(n):
    p = ("The history of coastal cartography in the North Atlantic shows a steady "
         "improvement in the accuracy of depth soundings and shoreline geometry, "
         "driven by navigational demand and state-sponsored survey traditions. ")
    return p * n

def w(name, text):
    with open(os.path.join(dir_, name), "w") as f:
        f.write(text)

# user body: 4 marker sections + per-session tail + question
quarters = max(body_tokens // 160, 1)
body = "\n\n".join(f"Unique marker {hx()}: the lighthouse at Skerries fades at dusk. " + paras(quarters)
                   for _ in range(4))
question = ("\n\nQuestion: begin your answer with the agent ID marker from the system "
            "message, then echo every revision marker from the system message in order, "
            "then list every unique marker that appears in this message in the order "
            "they appear. Answer with at most thirty words.")

# long multi-section system prompt; edit rounds rewrite one section per round
sys_paras = max(sys_tokens // 100, 4)
titles = ["Identity", "Style guide", "Environment", "Tool usage", "Output contract"]
content = {t: paras(sys_paras) for t in titles}
def render(sys_id, revisions):
    out = []
    for t in titles:
        s = f"# Identity\nAgent ID Unique marker {sys_id}.\n{content[t]}" if t == "Identity" else f"# {t}\n{content[t]}"
        if t in revisions:
            s += f"\nRevision marker Unique marker {revisions[t]} applies to this section."
        out.append(s)
    return "\n\n".join(out)

sys_id = hx()
w(f"sess_sys_fam_{suf}.txt", render(sys_id, {}))
body_prompt = None
for i in range(1, fams + 1):
    tail = f"Session note Unique marker {hx()}: the harbor chart was redrawn at dawn."
    prompt = body + "\n\n" + tail + question
    w(f"sess_prompt_fam{i}_{suf}.txt", prompt)
    if body_prompt is None:
        body_prompt = prompt
# edit rounds: one prompt edited cumulatively, >=1 section rewritten per round,
# so section content changes at least `edits` times across the series
edit_sections = [t for t in titles if t != "Identity"]
revisions = {}
for i in range(1, edits + 1):
    revisions[edit_sections[(i - 1) % len(edit_sections)]] = hx()
    w(f"sess_sys_edit{i}_{suf}.txt", render(sys_id, dict(revisions)))
    w(f"sess_prompt_edit{i}_{suf}.txt", body_prompt)
w(f"sess_sys_fresh_{suf}.txt", render(hx(), {}))
tail = f"Session note Unique marker {hx()}: the harbor chart was redrawn at dusk."
w(f"sess_prompt_fresh_{suf}.txt", body + "\n\n" + tail + question)
PYEOF
}

# True when a failed L3 restore's only marker defect is omission, so the
# probe gets an immediate L1 replay for variance classification.
needs_replay() { # $1=metrics JSON path
    python3 "${SCRIPT_DIR}/hicache_digest_check.py" needs-replay "$1" 2>/dev/null
}

correctness_bench() {
    log ""
    log "=== Phase 4: HiCache L1->L3 Correctness ==="
    case "${CORRECT_DIGESTS}" in
        true|false) ;;
        *) log "ERROR: CORRECT_DIGESTS must be true or false"; return 1 ;;
    esac
    if [[ "${CORRECT_DIGESTS}" == "true" ]]; then
        log "  restarting server with page digest logging before cold writes..."
        DIGEST_SERVER_ACTIVE=true
        bash "${SCRIPT_DIR}/stop_sglang.sh" >&2
        # A fresh L3 namespace makes every restore read attributable to this
        # run's cold writes without clearing production's existing disk cache.
        SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${RUNDIR}/hicache-storage" \
            SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS=true \
            bash "${SCRIPT_DIR}/serve_sglang.sh" >&2
        log ""
    fi
    local -A PF=
    local target
    for target in ${CORRECT_TOKENS}; do
        PF[$target]="${RUNDIR}/correct_${target}_$$.txt"
        gen_correct_probe "${PF[$target]}" "${target}"
        log "  target ${target} tokens: cold greedy prefill (writes L1->L2->L3)"
        BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 run_one "unused" "${RUNDIR}/correct_cold_${target}.json" "" --prompt-file "${PF[$target]}" --max-tokens 48 --ignore-eos --no-thinking --full-text-out "${RUNDIR}/correct_cold_${target}.ans"
        log_metrics_line "${RUNDIR}/correct_cold_${target}.json" "cold"
        # Revisit immediately, before later large probes can evict this prefix
        # from L2. This second access qualifies selective write-through while
        # the first request remains the reported cold measurement.
        log "  target ${target} tokens: immediate revisit for selective write-through admission"
        BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 run_one "unused" "${RUNDIR}/correct_revisit_${target}.json" "" --prompt-file "${PF[$target]}" --max-tokens 48 --ignore-eos --no-thinking --full-text-out "${RUNDIR}/correct_revisit_${target}.ans" --must-match-markers
        log_metrics_line "${RUNDIR}/correct_revisit_${target}.json" "revisit"
        log ""
    done

    local -a CC_PF=() CC_TAG=()
    local i=0
    for target in ${CORRECT_CONC_TOKENS}; do
        i=$((i+1))
        CC_PF+=("${RUNDIR}/correctcc_${target}_${i}_$$.txt")
        CC_TAG+=("${target}_${i}")
        gen_correct_probe "${CC_PF[-1]}" "${target}"
    done
    log "  concurrent cold batch: ${#CC_TAG[@]} probes (${CC_TAG[*]})"
    local -a pids=()
    local start_ns end_ns
    start_ns=$(date +%s%N)
    for i in "${!CC_TAG[@]}"; do
        ( BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 run_one "unused" "${RUNDIR}/correctcc_cold_${CC_TAG[$i]}.json" "" --prompt-file "${CC_PF[$i]}" --max-tokens 48 --ignore-eos --no-thinking --full-text-out "${RUNDIR}/correctcc_cold_${CC_TAG[$i]}.ans" ) &
        pids+=($!)
    done
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    end_ns=$(date +%s%N)
    log "  concurrent cold batch wall $(python3 -c "print(f'{(${end_ns}-${start_ns})/1e9:.1f}')")s"
    for tag in "${CC_TAG[@]}"; do
        log_metrics_line "${RUNDIR}/correctcc_cold_${tag}.json" "cold ${tag}"
    done
    log ""

    log "  concurrent pre-flush revisit batch: second access for selective write-through admission"
    pids=()
    start_ns=$(date +%s%N)
    for i in "${!CC_TAG[@]}"; do
        ( BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 run_one "unused" "${RUNDIR}/correctcc_revisit_${CC_TAG[$i]}.json" "" --prompt-file "${CC_PF[$i]}" --max-tokens 48 --ignore-eos --no-thinking --must-match-markers ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    end_ns=$(date +%s%N)
    log "  concurrent revisit batch wall $(python3 -c "print(f'{(${end_ns}-${start_ns})/1e9:.1f}')")s"
    for tag in "${CC_TAG[@]}"; do
        log_metrics_line "${RUNDIR}/correctcc_revisit_${tag}.json" "revisit ${tag}"
    done
    log ""

    # Session churn: long multi-section system prompt edited between sessions
    # (radix tree re-roots at every edit) plus a branch family that shares one
    # deep prefix and diverges at the tail (sibling sessions branch mid-tree).
    local -a SE_TAG=()
    local -A SE_SYS=()
    for i in $(seq 1 ${CORRECT_SESSION_FAMILIES}); do
        SE_TAG+=("fam${i}")
        SE_SYS["fam${i}"]="${RUNDIR}/sess_sys_fam_$$.txt"
    done
    for i in $(seq 1 ${CORRECT_SESSION_EDITS}); do
        SE_TAG+=("edit${i}")
        SE_SYS["edit${i}"]="${RUNDIR}/sess_sys_edit${i}_$$.txt"
    done
    SE_TAG+=("fresh")
    SE_SYS[fresh]="${RUNDIR}/sess_sys_fresh_$$.txt"
    gen_session_probes "${RUNDIR}" "$$" "${CORRECT_SESSION_TOKENS}" "${CORRECT_SESSION_SYS_TOKENS}" "${CORRECT_SESSION_FAMILIES}" "${CORRECT_SESSION_EDITS}"
    log "  session-churn cold batch: ${#SE_TAG[@]} sessions (${SE_TAG[*]})"
    local sys_f pf
    for tag in "${SE_TAG[@]}"; do
        sys_f="${SE_SYS[$tag]}"
        pf="${RUNDIR}/sess_prompt_${tag}_$$.txt"
        BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 run_one "unused" "${RUNDIR}/session_cold_${tag}.json" "" --prompt-file "${pf}" --system-file "${sys_f}" --max-tokens 80 --ignore-eos --no-thinking --full-text-out "${RUNDIR}/session_cold_${tag}.ans"
        log_metrics_line "${RUNDIR}/session_cold_${tag}.json" "cold ${tag}"
        if needs_replay "${RUNDIR}/session_cold_${tag}.json"; then
            log "  session ${tag}: marker omission on cold generation; immediate replay"
            BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 \
                run_one "unused" "${RUNDIR}/session_cold_replay_${tag}.json" "" --prompt-file "${pf}" --system-file "${sys_f}" --max-tokens 80 --ignore-eos --no-thinking \
                --must-match-markers
            log_metrics_line "${RUNDIR}/session_cold_replay_${tag}.json" "cold replay ${tag}"
        fi
    done
    log ""

    log "  waiting ${CORRECT_WRITE_SETTLE_SECONDS}s for write-through storage..."
    sleep "${CORRECT_WRITE_SETTLE_SECONDS}"
    # /flush_cache drops L1/L2 radix and recurrent state but leaves the L3
    # storage backend intact, so the next requests must restore from SSD.
    log "  flushing L1/L2 in process so warm runs must restore from L3 (SSD)..."
    log "  waiting up to ${CORRECT_FLUSH_TIMEOUT}s for requests and write-through storage to become idle..."
    curl --fail-with-body -sS -X POST \
        "http://127.0.0.1:${PORT}/flush_cache?timeout=${CORRECT_FLUSH_TIMEOUT}" >&2
    sleep "${CORRECT_FLUSH_SETTLE_SECONDS}"
    if [[ "${CORRECT_DIGESTS}" == "true" ]]; then
        # Digest hashing can keep the write-through queue draining for minutes;
        # snapshot the cold log only after the flush proves storage idle so
        # every cold write digest is inside the copy and no restore read is.
        cp "${PROJECT_DIR}/.sglang.log" "${RUNDIR}/correct_cold_server.log"
    fi
    log ""

    for target in ${CORRECT_TOKENS}; do
        log "  target ${target} tokens: L3 restore run (must match expected markers)"
        BENCH_MIN_CACHE_HIT_RATIO="${CORRECT_MIN_HIT_RATIO}" BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 \
            run_one "unused" "${RUNDIR}/correct_l3_${target}.json" "" --prompt-file "${PF[$target]}" --max-tokens 48 --ignore-eos --no-thinking \
            --full-text-out "${RUNDIR}/correct_l3_${target}.ans" --must-match-markers
        log_metrics_line "${RUNDIR}/correct_l3_${target}.json" "L3 restore"
        log "  target ${target} tokens: L1 hit run after restore (must match expected markers)"
        BENCH_MIN_CACHE_HIT_RATIO="${CORRECT_MIN_HIT_RATIO}" BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 \
            run_one "unused" "${RUNDIR}/correct_l1_${target}.json" "" --prompt-file "${PF[$target]}" --max-tokens 48 --ignore-eos --no-thinking \
            --full-text-out "${RUNDIR}/correct_l1_${target}.ans" --must-match-markers
        log_metrics_line "${RUNDIR}/correct_l1_${target}.json" "L1 hit"
        if needs_replay "${RUNDIR}/correct_l1_${target}.json"; then
            log "  target ${target} tokens: marker omission on L1 hit; immediate L1 replay"
            BENCH_MIN_CACHE_HIT_RATIO="${CORRECT_MIN_HIT_RATIO}" BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 \
                run_one "unused" "${RUNDIR}/correct_l1_replay_${target}.json" "" --prompt-file "${PF[$target]}" --max-tokens 48 --ignore-eos --no-thinking \
                --must-match-markers
            log_metrics_line "${RUNDIR}/correct_l1_replay_${target}.json" "L1 replay"
        fi
        log ""
    done

    log "  concurrent L3 restore batch: ${#CC_TAG[@]} probes (${CC_TAG[*]})"
    pids=()
    start_ns=$(date +%s%N)
    for i in "${!CC_TAG[@]}"; do
        ( BENCH_MIN_CACHE_HIT_RATIO="${CORRECT_CONC_MIN_HIT_RATIO}" BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 run_one "unused" "${RUNDIR}/correctcc_l3_${CC_TAG[$i]}.json" "" --prompt-file "${CC_PF[$i]}" --max-tokens 48 --ignore-eos --no-thinking --must-match-markers ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    end_ns=$(date +%s%N)
    log "  concurrent restore batch wall $(python3 -c "print(f'{(${end_ns}-${start_ns})/1e9:.1f}')")s"
    for tag in "${CC_TAG[@]}"; do
        log_metrics_line "${RUNDIR}/correctcc_l3_${tag}.json" "L3 restore ${tag}"
    done
    log ""

    # A first marker omission may be batch-sensitive generation, not cache
    # corruption: replay immediately (state is now in L1) and let the final
    # result classify it against the digest evidence.
    for i in "${!CC_TAG[@]}"; do
        tag="${CC_TAG[$i]}"
        if needs_replay "${RUNDIR}/correctcc_l3_${tag}.json"; then
            log "  concurrent ${tag}: marker omission on first L3 restore; immediate L1 replay"
            BENCH_MIN_CACHE_HIT_RATIO="${CORRECT_CONC_MIN_HIT_RATIO}" BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 \
                run_one "unused" "${RUNDIR}/correctcc_replay_${tag}.json" "" --prompt-file "${CC_PF[$i]}" --max-tokens 48 --ignore-eos --no-thinking --must-match-markers
            log_metrics_line "${RUNDIR}/correctcc_replay_${tag}.json" "L1 replay ${tag}"
        fi
    done
    log ""

    log "  session-churn restore runs: ${#SE_TAG[@]} sessions (${SE_TAG[*]})"
    for tag in "${SE_TAG[@]}"; do
        sys_f="${SE_SYS[$tag]}"
        pf="${RUNDIR}/sess_prompt_${tag}_$$.txt"
        BENCH_MIN_CACHE_HIT_RATIO="${CORRECT_SESSION_MIN_HIT}" BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 \
            run_one "unused" "${RUNDIR}/session_l3_${tag}.json" "" --prompt-file "${pf}" --system-file "${sys_f}" --max-tokens 80 --ignore-eos --no-thinking \
            --full-text-out "${RUNDIR}/session_l3_${tag}.ans" --must-match-markers
        log_metrics_line "${RUNDIR}/session_l3_${tag}.json" "L3 restore ${tag}"
        if needs_replay "${RUNDIR}/session_l3_${tag}.json"; then
            log "  session ${tag}: marker omission on first L3 restore; immediate L1 replay"
            BENCH_MIN_CACHE_HIT_RATIO="${CORRECT_SESSION_MIN_HIT}" BENCH_TEMPERATURE=0.0 BENCH_TOP_K=1 \
                run_one "unused" "${RUNDIR}/session_replay_${tag}.json" "" --prompt-file "${pf}" --system-file "${sys_f}" --max-tokens 80 --ignore-eos --no-thinking \
                --must-match-markers
            log_metrics_line "${RUNDIR}/session_replay_${tag}.json" "L1 replay ${tag}"
        fi
    done
    log ""

    # Snapshot the restore-phase log before any restart truncates it; it is the
    # digest-check input and the failure post-mortem copy.
    if [[ -f "${PROJECT_DIR}/.sglang.log" ]]; then
        cp "${PROJECT_DIR}/.sglang.log" "${RUNDIR}/correct_restore_server.log"
    fi

    # Corruption detector: compare post-flush read hashes to the cold server's
    # write hashes for the same storage key and component, and require every
    # KV/mamba host<->device transfer digest to be exact (kv_k, kv_v,
    # kv_scale_k, and kv_scale_v included). Also reject inconsistent repeated
    # writes or reads. Reads of old shared template pages can be unpaired;
    # newly generated UUID probe pages must produce paired keys.
    if [[ "${CORRECT_DIGESTS}" == "true" ]]; then
        if python3 "${SCRIPT_DIR}/hicache_digest_check.py" check \
            "${RUNDIR}/correct_cold_server.log" \
            "${RUNDIR}/correct_restore_server.log" \
            "${RUNDIR}/hicache_digest_check.json"; then
            log "  digest check passed (storage pairing complete, transfer digests exact)"
        else
            log "  WARNING: digest check failed; failures recorded in hicache_digest_check.json"
        fi
        log ""
        log "  restarting server without page digest logging..."
        bash "${SCRIPT_DIR}/stop_sglang.sh" >&2
        bash "${SCRIPT_DIR}/serve_sglang.sh" >&2
        DIGEST_SERVER_ACTIVE=false
        log ""
    fi
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if ! check_server; then
    echo "ERROR: SGLang server not running on port ${PORT}"
    echo "Run: MODEL_DIR=${MODEL_DIR} make serve"
    exit 1
fi

log "=== SGLang Benchmark ==="
log "Server: http://127.0.0.1:${PORT}"
log "Model:  ${SERVED_NAME}"
log "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
log ""

# Public modes group related checks; individual phases remain available for diagnosis.
PHASES="${BENCH_PHASES:-performance correctness}"
for phase in ${PHASES//,/ }; do
    case "$phase" in
        performance)
            single_user_bench
            concurrent_bench
            prefill_probe
            ;;
        correctness) correctness_bench ;;
        single) single_user_bench ;;
        conc) concurrent_bench ;;
        prefill) prefill_probe ;;
        *)
            echo "ERROR: unknown BENCH_PHASES entry '${phase}'" >&2
            echo "Use performance, correctness, single, conc, or prefill." >&2
            exit 1
            ;;
    esac
done

python3 - "${RUNDIR}" "${RESULT_FILE}" "${SERVED_NAME}" "${SCRIPT_DIR}" <<'PYEOF'
import glob, json, os, re, sys
rundir, out_file, model, scripts_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, scripts_dir)
import hicache_digest_check as diag

def load(tag, pi, rep):
    f = os.path.join(rundir, f"{tag}_{pi}_{rep}.json")
    if not os.path.isfile(f):
        return None
    try:
        m = json.load(open(f))
    except Exception:
        return None
    return m if m.get("ok") else None

single = []
for pi in range(3):
    reps = [r for r in (load("single", pi, i) for i in range(1, 10)) if r]
    if not reps:
        continue
    dec = [r["decode_tps"] for r in reps if r.get("decode_tps")]
    pre = [r["prefill_tps"] for r in reps if r.get("prefill_tps")]
    single.append({
        "prompt_idx": pi,
        "avg_decode_tps": round(sum(dec) / len(dec), 1) if dec else None,
        "avg_prefill_tps": round(sum(pre) / len(pre), 1) if pre else None,
        "avg_ttft_s": round(sum(r["ttft_s"] for r in reps) / len(reps), 3),
        "runs": reps,
    })

conc = []
levels = sorted({int(re.match(r"conc_(\d+)_", os.path.basename(f)).group(1))
                 for f in glob.glob(os.path.join(rundir, "conc_*.json"))
                 if re.match(r"conc_(\d+)_", os.path.basename(f))})
for n in levels:
    runs = []
    for f in glob.glob(os.path.join(rundir, f"conc_{n}_*.json")):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        if m.get("ok"):
            runs.append(m)
    if not runs:
        continue
    ttfts = sorted(r["ttft_s"] for r in runs)
    conc.append({
        "concurrency": n,
        "completed": len(runs),
        "total_tokens": sum(r["completion_tokens"] for r in runs),
        "ttft_p50_s": ttfts[len(ttfts) // 2],
        "ttft_max_s": ttfts[-1],
        "cache_hit_requests": sum(1 for r in runs if r["cached_tokens"] > 0),
        "runs": runs,
    })

def rd(name):
    try:
        return json.load(open(os.path.join(rundir, name)))
    except Exception:
        return None

probe = []
for f in sorted(glob.glob(os.path.join(rundir, "prefill_cold_*.json"))):
    tgt = re.match(r"prefill_cold_(\d+)", os.path.basename(f)).group(1)
    probe.append({"target_tokens": int(tgt),
                  "cold": rd(f"prefill_cold_{tgt}.json"),
                  "warm": rd(f"prefill_warm_{tgt}.json")})

correctness = []
for f in sorted(glob.glob(os.path.join(rundir, "correct_cold_*.json"))):
    m = re.fullmatch(r"correct_cold_(\d+)\.json", os.path.basename(f))
    if not m:
        continue
    tgt = m.group(1)
    correctness.append({"target_tokens": int(tgt),
                        "cold": rd(f"correct_cold_{tgt}.json"),
                        "cold_revisit": rd(f"correct_revisit_{tgt}.json"),
                        "l3_restore": rd(f"correct_l3_{tgt}.json"),
                        "l1_hit": rd(f"correct_l1_{tgt}.json"),
                        "l1_replay": rd(f"correct_l1_replay_{tgt}.json")})
concurrent_correctness = []
for f in sorted(glob.glob(os.path.join(rundir, "correctcc_cold_*.json"))):
    m = re.fullmatch(r"correctcc_cold_(\d+_\d+)\.json", os.path.basename(f))
    if not m:
        continue
    mt = m.group(1)
    concurrent_correctness.append({"probe": mt,
                                   "cold": rd(f"correctcc_cold_{mt}.json"),
                                   "cold_revisit": rd(f"correctcc_revisit_{mt}.json"),
                                   "l3_restore": rd(f"correctcc_l3_{mt}.json"),
                                   "l1_replay": rd(f"correctcc_replay_{mt}.json")})
session_churn = []
for f in sorted(glob.glob(os.path.join(rundir, "session_cold_*.json"))):
    m = re.fullmatch(r"session_cold_((?:fam|edit)\d+|fresh)\.json", os.path.basename(f))
    if not m:
        continue
    tag = m.group(1)
    session_churn.append({"session": tag,
                          "cold": rd(f"session_cold_{tag}.json"),
                          "cold_replay": rd(f"session_cold_replay_{tag}.json"),
                          "l3_restore": rd(f"session_l3_{tag}.json"),
                          "l1_replay": rd(f"session_replay_{tag}.json")})

# Only omission-only responses with exact digests and a complete same-prompt
# replay can be classified as generation variance. Cold omissions also need a
# successful restore. L1 omissions remain failures until state use is attested.
digests = rd("hicache_digest_check.json")
variances = []
variance_files = set()

def complete(metrics):
    return bool(metrics) and metrics.get("ok") and metrics.get("matched") is True

def note_variance(section, probe, phase, metrics_file, metrics, replay):
    variance = diag.classify_generation_variance(metrics, replay, digests, phase)
    if variance:
        variance = {"section": section, "probe": probe,
                    "metrics_file": metrics_file, **variance}
        variances.append(variance)
        variance_files.add(metrics_file)
    return variance

for entry in correctness:
    tgt = entry["target_tokens"]
    if complete(entry["cold_revisit"]) and complete(entry["l3_restore"]):
        entry["cold_variance"] = note_variance(
            "correctness", f"target_{tgt}", "cold generation",
            f"correct_cold_{tgt}.json", entry["cold"], entry["cold_revisit"])
    entry["variance"] = note_variance(
        "correctness", f"target_{tgt}", "L3 restore",
        f"correct_l3_{tgt}.json", entry["l3_restore"], entry["l1_hit"])
for entry in concurrent_correctness:
    mt = entry["probe"]
    if complete(entry["cold_revisit"]) and complete(entry["l3_restore"]):
        entry["cold_variance"] = note_variance(
            "concurrent_correctness", mt, "cold generation",
            f"correctcc_cold_{mt}.json", entry["cold"], entry["cold_revisit"])
    entry["variance"] = note_variance(
        "concurrent_correctness", mt, "L3 restore",
        f"correctcc_l3_{mt}.json", entry["l3_restore"], entry["l1_replay"])
for entry in session_churn:
    tag = entry["session"]
    if complete(entry["l3_restore"]):
        entry["cold_variance"] = note_variance(
            "session_churn", tag, "cold generation",
            f"session_cold_{tag}.json", entry["cold"], entry["cold_replay"])
    entry["variance"] = note_variance(
        "session_churn", tag, "L3 restore",
        f"session_l3_{tag}.json", entry["l3_restore"], entry["l1_replay"])

failures = []
for f in sorted(glob.glob(os.path.join(rundir, "*.json"))):
    name = os.path.basename(f)
    try:
        metrics = json.load(open(f))
    except Exception as exc:
        failures.append({"file": name, "error": str(exc)})
        continue
    if name in variance_files:
        continue
    if not metrics.get("ok"):
        failures.append({
            "file": name,
            "error": metrics.get("error", "request failed"),
        })

result = {
    "server": "sglang",
    "harness": 6,
    "model": model,
    "single_user": single,
    "concurrent": conc,
    "prefill_probe": probe,
    "correctness": correctness,
    "concurrent_correctness": concurrent_correctness,
    "session_churn": session_churn,
    "digest_check": digests,
    "variances": variances,
    "failures": failures,
}
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "single_user"} | {"single_user_summary": [{"prompt_idx": s["prompt_idx"], "avg_decode_tps": s["avg_decode_tps"], "avg_prefill_tps": s["avg_prefill_tps"], "avg_ttft_s": s["avg_ttft_s"]} for s in single], "variance_count": len(variances)}, indent=2), file=sys.stderr)
if failures:
    raise SystemExit(1)
PYEOF
