#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

LLAMA_BENCH="${LLAMACPP_BUILD}/bin/llama-bench"

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.json"
mkdir -p "${RESULTS_DIR}"

echo "=== Benchmarking Quantized Models ==="
echo "Results: ${RESULTS_FILE}"

# Collect all GGUF files to benchmark
GGUF_FILES=()
for f in "${MODELS_DIR}"/${MODEL_NAME}-*.gguf; do
    [[ -f "$f" ]] || continue
    [[ "$(basename "$f")" == *"-mmproj-"* ]] && continue
    GGUF_FILES+=("$f")
done

if [[ ${#GGUF_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No GGUF files found in ${MODELS_DIR}"
    echo "Run: make quantize"
    exit 1
fi

echo "Found ${#GGUF_FILES[@]} models to benchmark"
echo ""

# Temp files
BENCH_JSON="${RESULTS_DIR}/.bench_throughput_${TIMESTAMP}.json"
PPL_JSON="${RESULTS_DIR}/.bench_ppl_${TIMESTAMP}.json"
F16_LOGITS="${RESULTS_DIR}/.f16_logits_${TIMESTAMP}.bin"
PPL_CHUNKS=32

# Perplexity test file
PPL_FILE="${CALIBRATION_DIR}/perplexity_test.txt"
if [[ ! -f "${PPL_FILE}" ]]; then
    head -c 500000 "${CALIBRATION_DIR}/combined.txt" > "${PPL_FILE}"
fi

echo '{}' > "${PPL_JSON}"

# ---------------------------------------------------------------
# Phase 1: Throughput via llama-bench
# ---------------------------------------------------------------
MODEL_LIST=$(printf "%s," "${GGUF_FILES[@]}")
MODEL_LIST="${MODEL_LIST%,}"

echo "=== Phase 1/3: Throughput (pp=512, tg=128, 3 reps) ==="
echo ""

# JSON to stdout (captured), progress/warnings to stderr (displayed)
"${LLAMA_BENCH}" \
    -m "${MODEL_LIST}" \
    -ngl "${GPU_LAYERS}" \
    -fa 1 \
    -t "${THREADS}" \
    -p 512 \
    -n 128 \
    -r 3 \
    -o json \
    --progress \
    > "${BENCH_JSON}"

# Print a quick readable summary from the saved JSON
python3 -c "
import json, os
with open('${BENCH_JSON}') as f: data = json.load(f)
models = {}
for e in data:
    name = e.get('model_type', 'unknown')
    size = round(e['model_size'] / 1073741824, 1)
    models.setdefault(name, {'size': size})
    if e['n_prompt'] > 0 and e['n_gen'] == 0:
        models[name]['pp'] = f\"{e['avg_ts']:.1f} ±{e['stddev_ts']:.1f}\"
    elif e['n_gen'] > 0 and e['n_prompt'] == 0:
        models[name]['tg'] = f\"{e['avg_ts']:.1f} ±{e['stddev_ts']:.1f}\"
print(f\"{'Model':<35} {'Size':>8} {'pp t/s':>16} {'tg t/s':>16}\")
print('-' * 80)
for name, m in models.items():
    print(f\"{name:<35} {m['size']:>6.1f}G {m.get('pp','N/A'):>16} {m.get('tg','N/A'):>16}\")
"
echo ""

# ---------------------------------------------------------------
# Phase 2: Perplexity via llama-perplexity
# ---------------------------------------------------------------
echo "=== Phase 2/3: Perplexity (${PPL_CHUNKS} chunks) ==="
echo ""

for GGUF in "${GGUF_FILES[@]}"; do
    NAME=$(basename "${GGUF}" .gguf)
    echo "  ${NAME}..."
    PPL_OUTPUT=$("${LLAMA_PERPLEXITY}" \
        -m "${GGUF}" \
        -f "${PPL_FILE}" \
        -ngl "${GPU_LAYERS}" \
        --chunks "${PPL_CHUNKS}" \
        2>&1 || true)
    PPL=$(echo "${PPL_OUTPUT}" | grep -oP 'Final estimate: PPL = \K[\d.]+' || echo "N/A")
    echo "    PPL = ${PPL}"

    python3 -c "
import json
f = '${PPL_JSON}'
with open(f) as fh: data = json.load(fh)
data.setdefault('${NAME}', {})['ppl'] = '${PPL}'
with open(f, 'w') as fh: json.dump(data, fh)
"
done
echo ""

# ---------------------------------------------------------------
# Phase 3: KL Divergence vs F16
# ---------------------------------------------------------------
echo "=== Phase 3/3: KL Divergence vs F16 ==="
echo ""

# Save F16 reference logits
echo "  Saving F16 reference logits..."
"${LLAMA_PERPLEXITY}" \
    -m "${F16_GGUF}" \
    -f "${PPL_FILE}" \
    -ngl "${GPU_LAYERS}" \
    --chunks "${PPL_CHUNKS}" \
    --save-all-logits "${F16_LOGITS}" \
    2>&1 | tail -1 || true
echo ""

for GGUF in "${GGUF_FILES[@]}"; do
    NAME=$(basename "${GGUF}" .gguf)
    [[ "${NAME}" == *"-F16" ]] && continue

    echo "  ${NAME} vs F16..."
    KL_OUTPUT=$("${LLAMA_PERPLEXITY}" \
        -m "${GGUF}" \
        -f "${PPL_FILE}" \
        -ngl "${GPU_LAYERS}" \
        --chunks "${PPL_CHUNKS}" \
        --kl-divergence \
        --kl-divergence-base "${F16_LOGITS}" \
        2>&1 || true)

    # Format: "Mean    KLD:   0.007326 ±   0.002427"
    #         "Maximum KLD:  12.480855"
    #         "99.9%   KLD:   0.971475"
    KL_MEAN=$(echo "${KL_OUTPUT}" | grep -oP '^Mean\s+KLD:\s+\K[\d.]+' || echo "N/A")
    KL_MAX=$(echo "${KL_OUTPUT}" | grep -oP '^Maximum KLD:\s+\K[\d.]+' || echo "N/A")
    KL_P999=$(echo "${KL_OUTPUT}" | grep -oP '^99\.9%\s+KLD:\s+\K[\d.]+' || echo "N/A")
    echo "    KL: mean=${KL_MEAN} max=${KL_MAX} p99.9=${KL_P999}"

    python3 -c "
import json
f = '${PPL_JSON}'
with open(f) as fh: data = json.load(fh)
d = data.setdefault('${NAME}', {})
d['kl_mean'] = '${KL_MEAN}'
d['kl_max'] = '${KL_MAX}'
d['kl_p999'] = '${KL_P999}'
with open(f, 'w') as fh: json.dump(data, fh)
"
done
echo ""

# ---------------------------------------------------------------
# Merge all results into final JSON
# ---------------------------------------------------------------
python3 "${PROJECT_DIR}/src/merge_benchmark.py" \
    "${BENCH_JSON}" "${PPL_JSON}" "${RESULTS_FILE}"

rm -f "${BENCH_JSON}" "${PPL_JSON}" "${F16_LOGITS}"

echo "=== All benchmarks complete ==="
echo "Results saved: ${RESULTS_FILE}"
echo ""
echo "Next: make compare"
