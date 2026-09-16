#!/usr/bin/env bash
# Full prod-3 pipeline, one command: fused-kernel ext build -> preflight ->
# chained AWQ|GPTQ run. Detached via setsid; log: /home/ray/quantize_prod5.log
set -uo pipefail
REPO=/home/ray/super-quant
LOG=/home/ray/quantize_prod5.log
PY=$REPO/.venv/bin/python
export TORCH_EXTENSIONS_DIR=/home/ray/.cache/torch_extensions
export PATH=/home/ray/super-quant/.venv/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0

echo "=== [runner] apt cuda-nvcc-13-4 (for modelopt fused quant kernels)" >> "$LOG"
sudo apt-get update -qq >> /home/ray/apt_ext.log 2>&1 \
  && sudo apt-get install -y --no-install-recommends cuda-nvcc-13-4 cuda-cudart-dev-13-4 >> /home/ray/apt_ext.log 2>&1 \
  || echo "!!! APT_FAILED (run continues on Triton-only path)" >> "$LOG"

echo "=== [runner] JIT-build modelopt CUDA extensions (base/fp8/mx)" >> "$LOG"
$PY -c "from modelopt.torch.quantization.extensions import get_cuda_ext, get_cuda_ext_fp8, get_cuda_ext_mx
get_cuda_ext(True); get_cuda_ext_fp8(True); get_cuda_ext_mx(True); print('EXT_OK')" >> /home/ray/ext_build.log 2>&1 \
  || echo "!!! EXT_BUILD_FAILED (quantizer falls back to eager/Triton per-format)" >> "$LOG"

echo "=== [runner] preflight" >> "$LOG"
cd "$REPO"
bash scripts/run_flashnext_preflight.sh >> "$LOG" 2>&1 || { echo "PREFLIGHT_FAILED" >> "$LOG"; exit 1; }

echo "=== [runner] launch chained AWQ -> GPTQ" >> "$LOG"
CALIB_PHASE=awq CHAIN_INPROC=1 \
STAGE_DIR=/home/ray/stage-prod \
CKPT_DIR=/home/ray/ckpt-prod \
UPLOAD_PREFIX=gs://ads-billing-models/super-quant/qwen38-flash-next-w4a4/builds/20260914-flashnext-prod \
BF16_DIR=/tmp/flashnext-bf16 \
KEEP_BF16=1 LOG="$LOG" \
bash scripts/run_flashnext_fsdp2.sh
