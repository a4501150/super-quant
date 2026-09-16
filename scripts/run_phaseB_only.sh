#!/usr/bin/env bash
# One-shot phase B launcher (AWQ state already banked; mirrors the chained
# launcher's second command, plus the staged-skip-file fix).
set -uo pipefail
cd /home/ray/super-quant
LOG="${LOG:-/home/ray/quantize_phaseB.log}"

# Stage dir self-containedness fix: the staged index keeps source shard
# names for skipped tensors (PLE/vision/MTP); the FSDP2 loader opens every
# file the index names. Copy any referenced-but-missing source shards.
.venv/bin/python - <<'PY'
import json, os, shutil
stage = "/home/ray/stage-prod"
src = "/tmp/flashnext-bf16"
wm = json.load(open(os.path.join(stage, "model.safetensors.index.json")))["weight_map"]
missing = sorted({f for f in wm.values() if not os.path.exists(os.path.join(stage, f))})
for f in missing:
    shutil.copy2(os.path.join(src, f), os.path.join(stage, f))
print(f"staged-skip-fix: copied {missing}" if missing else "staged-skip-fix: nothing missing")
PY
[ $? -eq 0 ] || { echo "SKIPFIX_FAILED" >> $LOG; exit 1; }

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python scripts/gpu_util_guard.py --period-ms 2 >> /home/ray/guard.log 2>&1 &
G=$!

.venv/bin/torchrun --standalone --nproc_per_node=8 \
  src/quantize_mopt.py \
  --config configs/Qwen3.8-Flash-Next/modelopt.json \
  --calibration-dir /tmp/qwen38-flash-next-w4a4/calibration/Qwen3.8-Flash-Next \
  --output-dir /tmp/flashnext-nvfp4 \
  --checkpoint-dir /home/ray/ckpt-prod \
  --upload-prefix gs://ads-billing-models/super-quant/qwen38-flash-next-w4a4/builds/20260914-flashnext-prod \
  --calib-phase gptq --stage-dir /home/ray/stage-prod --resume-stage /home/ray/stage-prod \
  >> $LOG 2>&1
RC=$?
kill $G 2>/dev/null
echo "QUANTIZE_EXITED rc=$RC" >> $LOG
exit $RC
