#!/usr/bin/env bash
# Production launcher: FSDP2 + torchrun data-parallel NVFP4 calibration of
# Qwen3.8 Flash Next on 8 local GPUs (ModelOpt hf_ptq distributed pattern).
#
# Run detached on the worker, e.g.:
#   nohup bash scripts/run_flashnext_fsdp2.sh > /home/ray/launcher.log 2>&1 &
# Extra args pass through to quantize_mopt.py (e.g. --token-budget 32768).
set -uo pipefail

REPO="${REPO:-/home/ray/super-quant}"
CONFIG="${CONFIG:-configs/Qwen3.8-Flash-Next/modelopt.json}"
CALIB_DIR="${CALIB_DIR:-/tmp/qwen38-flash-next-w4a4/calibration/Qwen3.8-Flash-Next}"
OUT_DIR="${OUT_DIR:-/tmp/flashnext-nvfp4}"
CKPT_DIR="${CKPT_DIR:-/home/ray/ckpt}"
UPLOAD_PREFIX="${UPLOAD_PREFIX:-gs://ads-billing-models/super-quant/qwen38-flash-next-w4a4/builds/20260913-flashnext-prod}"
BF16_DIR="${BF16_DIR:-/tmp/flashnext-bf16}"
LOG="${LOG:-/home/ray/quantize.log}"
NPROC="${NPROC:-8}"

cd "$REPO"

# Stage the PLE table + aux tensors to tmpfs (idempotent, size-verified):
# all ranks mmap ONE physical copy of the n-gram table; reattach reads from
# here because the janitor removes BF16_DIR before export.
.venv/bin/python scripts/ple_shm_prep.py --model-dir "$BF16_DIR" \
  || { echo "PREP_FAILED" >> "$LOG"; exit 1; }

# GPU utilization guard: GPUIdleDownscaling reclaimed the preflight at 0.98%
# average utilization; FSDP2 + CPU-PLE passes are dispatch-bound and keep
# SMs idle, so launch a low-cost kernel heartbeat for the whole run.
.venv/bin/python scripts/gpu_util_guard.py --period-ms 2 \
  >> /home/ray/guard.log 2>&1 &
GUARD=$!
trap 'kill "$GUARD" 2>/dev/null' EXIT

# Disk janitor (opt-in): it existed for the Pro-6000-class ~463 GiB disks.
# On the H100 node RAID (5.9 TiB) BF16 + stage + export all fit side by side,
# and deleting costs a 331 GiB re-pull on any retry. Default: keep. The
# "=== ModelOpt quantize" marker is printed once the FSDP2 loader has
# returned on rank 0; the loader is a chain of cross-rank broadcasts, so
# every rank is done reading the checkpoint at that point (sleep covers a
# lagging rank's last read).
JANITOR=""
if [ "${KEEP_BF16:-1}" != 1 ]; then
  (
    until grep -q '=== ModelOpt quantize' "$LOG" 2>/dev/null; do sleep 5; done
    sleep 30
    if [ -d "$BF16_DIR" ]; then
      rm -rf "$BF16_DIR" && echo "[janitor] removed $BF16_DIR" >> "$LOG"
    fi
  ) &
  JANITOR=$!
fi

# NCCL env: hf_ptq relies on the 2 h PG timeout set inside the runner; the
# async error handler makes a dead/straggler rank abort the job quickly
# instead of hanging to the timeout. NCCL_SOCKET_IFNAME only needs setting
# if bootstrap picks the wrong interface (single-node runs are fine on the
# default route; override by exporting it before launch if needed).
# These MUST be exports, not an assignment prefix on the torchrun command:
# when the STAGE_ARGS block below was inserted between the prefix and the
# command, the line continuation ran into a `#` and bash turned the whole
# prefix into a plain shell assignment — the 2026-09-14 prod run then ran
# without expandable_segments and fragmented to OOM (16.9 GiB stranded
# reserved-unallocated against a 15.2 GiB allocation).
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# (No PYTHONWARNINGS here: the torch_dtype deprecation noise is filtered
# in-process at the top of quantize_mopt.py — the env module filter never
# matched because the warning is emitted from transformers' frame.)

# Optional stage checkpoint (post-quantize mutated-weights write): export
# failures then cost an export retry (--resume-stage $STAGE_DIR), not a full
# calibration replay. Empty = off (run-2 behavior).
STAGE_ARGS=()
if [ -n "${STAGE_DIR:-}" ]; then
  STAGE_ARGS+=(--stage-dir "$STAGE_DIR")
fi

# Phase checkpoint (see quantize_mopt.py --calib-phase): CALIB_PHASE=awq runs
# only the recipe's AWQ leg and stops after mtq_state + staged weights land.
# CHAIN_GPTQ=1 then starts a SECOND torchrun for the GPTQ leg over that
# staged state: the costly AWQ result is banked independently, and GPTQ +
# export get a fresh allocator (no clip-phase fragmentation carried over).
PHASE_ARGS=()
if [ -n "${CALIB_PHASE:-}" ]; then
  PHASE_ARGS+=(--calib-phase "$CALIB_PHASE")
fi

.venv/bin/torchrun --standalone --nproc_per_node="$NPROC" \
    src/quantize_mopt.py \
    --config "$CONFIG" \
    --calibration-dir "$CALIB_DIR" \
    --output-dir "$OUT_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --upload-prefix "$UPLOAD_PREFIX" \
    "${STAGE_ARGS[@]}" "${PHASE_ARGS[@]}" "$@" \
    >> "$LOG" 2>&1
RC=$?

if [ "$RC" -ne 0 ] || [ "${CHAIN_GPTQ:-0}" != 1 ]; then
  kill "$JANITOR" 2>/dev/null
  echo "QUANTIZE_EXITED rc=$RC" >> "$LOG"
  exit "$RC"
fi

if [ -z "${STAGE_DIR:-}" ]; then
  echo "CHAIN_GPTQ requires STAGE_DIR (phase hand-off is via staged weights)" >> "$LOG"
  kill "$JANITOR" 2>/dev/null
  echo "QUANTIZE_EXITED rc=2" >> "$LOG"
  exit 2
fi
# Mirror the staged weights to GCS while GPTQ runs (2026-09-14: a pod
# reschedule after the AWQ bank destroyed the only local copy of 234 GiB
# of staged weights; the upload watcher had not completed any file).
# Backgrounded so the ~30-60 min upload overlaps the GPTQ phase.
if [ -n "${UPLOAD_PREFIX:-}" ]; then
  setsid nohup gcloud storage cp -r -n "$STAGE_DIR" "$UPLOAD_PREFIX/stage-prod/" \
    >> /home/ray/stage_upload.log 2>&1 &
fi

# The staged dir is self-contained since the staging fix (89a0258): drop the
# 331 GiB BF16 clone now that phase A banked, so the pod's ephemeral-storage
# quota survives GPTQ + export spill (2026-09-14: pod EVICTED for exceeding
# the 800Gi container limit right at the phase boundary).
# Opt-in only (KEEP_BF16=0): with the pod's ephemeral LIMIT at 3Ti the peak
# (~1.2Ti: BF16 + staged + export) fits, and the local BF16 doubles as the
# reference checkpoint for the NVFP4-vs-BF16 holdout eval. Authoritative
# copy stays in the GCS archive either way.
if [ "${KEEP_BF16:-1}" = 0 ] && [ -d "$BF16_DIR" ]; then
  rm -rf "$BF16_DIR" && echo "[chain] freed $BF16_DIR (staged dir self-contained)" >> "$LOG"
fi

echo "=== chain phase gptq: resume from $STAGE_DIR ===" >> "$LOG"
.venv/bin/torchrun --standalone --nproc_per_node="$NPROC" \
    src/quantize_mopt.py \
    --config "$CONFIG" \
    --calibration-dir "$CALIB_DIR" \
    --output-dir "$OUT_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --upload-prefix "$UPLOAD_PREFIX" \
    --calib-phase gptq --stage-dir "$STAGE_DIR" --resume-stage "$STAGE_DIR" "$@" \
    >> "$LOG" 2>&1
RC=$?
kill "$JANITOR" 2>/dev/null
echo "QUANTIZE_EXITED rc=$RC" >> "$LOG"
exit "$RC"
