#!/usr/bin/env bash
# Worker preflight for a fresh Hendrix pod (2026-09-14: pod reschedule
# destroyed /home/ray incl. the built venv, /tmp staging, and calibration;
# restart without these checks wastes ~40 min dying at model load).
# Run from the repo root: bash scripts/run_flashnext_preflight.sh
set -uo pipefail
REPO="${REPO:-/home/ray/super-quant}"
BF16="${BF16:-/tmp/flashnext-bf16}"
CALIB_TG="gs://ads-billing-models/super-quant/qwen38-flash-next-w4a4/inputs/Qwen3.8-Flash-Next-calibration-v1.tar.gz"
CALIB_ROOT=/tmp/qwen38-flash-next-w4a4
ARCHIVE="gs://ads-billing-models/super-quant/qwen38-flash-next-w4a4/archive/20260912-13-ddp-calibration/builds/20260912-linearized-bf16"
fail() { echo "!!! PREFLIGHT_FAILED: $*" >&2; exit 1; }

# GPU idle guard first: download/prep phases sit below the 2.5% idle-
# downscale threshold and Hendrix reclaims the worker group at 1 h.
cd "$REPO"
.venv/bin/python scripts/gpu_util_guard.py --period-ms 2 >> /home/ray/guard.log 2>&1 &
GUARD=$!
trap 'kill "$GUARD" 2>/dev/null' EXIT

# tmpfs: the PLE table is 51 GiB mmap-shared from /dev/shm across ranks.
SHM_KIB=$(df -k /dev/shm | awk 'NR==2{print $2}')
[ "$SHM_KIB" -ge $((60 * 1024 * 1024)) ] || fail "/dev/shm only ${SHM_KIB} KiB; pod needs >=60 GiB tmpfs"

# venv: cu13.4 torch + repo wheelhouse per spot-ml CLAUDE.md launch stack.
.venv/bin/python -c "import torch, modelopt" 2>/dev/null \
  || fail ".venv missing/broken on this pod; rebuild cu134 stack (see CLAUDE.md 'Quantization stack' + wheelhouse/qwen38-w4a4/) before relaunch"

# staged BF16 checkpoint (glob form avoids gcloud's trailing-slash nesting trap)
if [ ! -f "$BF16/model.safetensors.index.json" ]; then
  mkdir -p "$BF16"
  gcloud storage cp "${ARCHIVE}/*" "$BF16/" || fail "BF16 pull"
fi
[ -f "$BF16/linearization-manifest.json" ] || fail "linearization manifest missing"
# local count: gcloud storage ls rejects file:// paths (2026-09-14 prod3: the
# old gcloud-ls form errored, making the count check silently vacuous)
echo "bf16 ok: $(find "$BF16" -maxdepth 1 -name '*.safetensors' | wc -l) shards"

# calibration corpus
if [ ! -d "$CALIB_ROOT/calibration/Qwen3.8-Flash-Next" ]; then
  mkdir -p "$CALIB_ROOT"
  gcloud storage cp "$CALIB_TG" /tmp/ && tar -xzf /tmp/Qwen3.8-Flash-Next-calibration-v1.tar.gz -C "$CALIB_ROOT" || fail "calibration pull"
fi
# tarball top level is `Qwen3.8-Flash-Next/` directly (v1); launcher expects a
# calibration/ level (old pods were laid out by hand). Normalize.
if [ ! -d "$CALIB_ROOT/calibration/Qwen3.8-Flash-Next" ] && [ -d "$CALIB_ROOT/Qwen3.8-Flash-Next" ]; then
  mkdir -p "$CALIB_ROOT/calibration" && mv "$CALIB_ROOT/Qwen3.8-Flash-Next" "$CALIB_ROOT/calibration/"
fi
ls "$CALIB_ROOT/calibration/Qwen3.8-Flash-Next" >/dev/null || fail "calibration dir missing"

# PLE tmpfs staging (idempotent, size-verified)
.venv/bin/python scripts/ple_shm_prep.py --model-dir "$BF16" || fail "ple_shm_prep"

# staged-dir self-containedness prerequisite for --resume-stage runs
[ -f "$BF16/model-00006-of-00052.safetensors" ] || fail "skip-shard source files missing"

echo "=== PREFLIGHT_DONE ==="