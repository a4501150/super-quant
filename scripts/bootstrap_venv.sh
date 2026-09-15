#!/usr/bin/env bash
# Rebuild the cu134 quantization venv on a fresh pod (2026-09-14: pod
# reschedule destroyed /home/ray incl. the old hand-bootstrapped venv;
# this script IS the bootstrap now, kept in the repo for the next time).
set -euo pipefail
REPO=/home/ray/super-quant
WH=gs://ads-billing-models/wheelhouse
echo "=== calib layout fix"
CR=/tmp/qwen38-flash-next-w4a4
if [ -d "$CR/Qwen3.8-Flash-Next" ] && [ ! -d "$CR/calibration/Qwen3.8-Flash-Next" ]; then
  mkdir -p "$CR/calibration" && mv "$CR/Qwen3.8-Flash-Next" "$CR/calibration/"
fi
echo "=== uv sync (pinned pyproject/uv.lock stack)"
cd "$REPO"
uv sync --frozen
echo "=== fork wheels"
gcloud storage cp "$WH/modelopt/sq-0.46.0+sq8/nvidia_modelopt-0.46.0+sq8-py3-none-any.whl" /tmp/
gcloud storage cp "$WH/qwen38-w4a4/torch2.15cu134-py312-fatbin/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl" /tmp/
uv pip install --python "$REPO/.venv/bin/python" ninja flash-linear-attention einops torchvision pillow \
# fused quant extensions need ninja; GDN layers fall back to slow reference kernels without fla
uv pip install --python "$REPO/.venv/bin/python" \
  /tmp/nvidia_modelopt-0.46.0+sq8-py3-none-any.whl \
  /tmp/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl
echo "=== import smoke"
.venv/bin/python - <<'PY'
import torch, modelopt, causal_conv1d_cuda
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
print("smoke ok:", torch.__version__, modelopt.__version__, "gpus", torch.cuda.device_count())
PY
echo "=== BOOTSTRAP_DONE"
