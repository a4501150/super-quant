#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/sglang.env"
source "${SCRIPT_DIR}/../configs/vllm.env"

CUDA_HOME="/usr/local/cuda-${SGLANG_CUDA_VERSION}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_MAIN_CUDA_VERSION="${SGLANG_CUDA_VERSION}"
export VLLM_PRECOMPILED_WHEEL_VARIANT="${SGLANG_CUDA_TAG}"
export VLLM_USE_PRECOMPILED=1
export MAX_JOBS="${MAX_JOBS:-2}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] ||
    ! "${CUDA_HOME}/bin/nvcc" --version | grep -q "release ${SGLANG_CUDA_VERSION}"; then
    echo "ERROR: CUDA ${SGLANG_CUDA_VERSION} is required at ${CUDA_HOME}" >&2
    exit 1
fi
if [[ ! -f "${VLLM_SOURCE_DIR}/pyproject.toml" ]]; then
    echo "ERROR: vLLM source not found at ${VLLM_SOURCE_DIR}" >&2
    exit 1
fi

VLLM_SOURCE_REVISION="$(git -C "${VLLM_SOURCE_DIR}" rev-parse HEAD)"
if [[ "$(git -C "${VLLM_SOURCE_DIR}" describe --tags --exact-match 2>/dev/null || true)" != "${VLLM_SOURCE_REF}" ]]; then
    echo "ERROR: ${VLLM_SOURCE_DIR} is not checked out at ${VLLM_SOURCE_REF}" >&2
    exit 1
fi

for pidfile in "${SCRIPT_DIR}/../.sglang.pid" "${SCRIPT_DIR}/../.vllm.pid"; do
    if [[ -f "${pidfile}" ]]; then
        server_pid="$(<"${pidfile}")"
        if kill -0 "${server_pid}" 2>/dev/null; then
            echo "ERROR: stop SGLang and vLLM before rebuilding the vLLM environment." >&2
            exit 1
        fi
    fi
done

VENV_ROOT="$(dirname "${VLLM_VENV}")"
VENV_RELEASE="${VENV_ROOT}/vllm-${VLLM_ENV_REVISION}-${VLLM_SOURCE_REVISION:0:12}"
MANIFEST="${VENV_RELEASE}/.super-quant-vllm-env"

if [[ -x "${VENV_RELEASE}/bin/python" && -f "${MANIFEST}" ]] &&
    [[ "$(sed -n '1p' "${MANIFEST}")" == "${VLLM_ENV_REVISION}" ]] &&
    [[ "$(sed -n '2p' "${MANIFEST}")" == "${VLLM_SOURCE_REVISION}" ]]; then
    echo "vLLM environment ${VLLM_ENV_REVISION} is already installed."
else
    rm -rf "${VENV_RELEASE}"
    uv venv "${VENV_RELEASE}" --python 3.12
    python_bin="${VENV_RELEASE}/bin/python"

    uv pip install --python "${python_bin}" \
        --editable "${VLLM_SOURCE_DIR}" \
        --torch-backend auto \
        --index-strategy unsafe-best-match

    uv pip install --python "${python_bin}" \
        --prerelease allow --index "${SGLANG_TORCH_INDEX}" --reinstall \
        "torch==${SGLANG_TORCH_VERSION}" \
        "torchvision==${SGLANG_TORCHVISION_VERSION}" \
        "torchaudio==${SGLANG_TORCHAUDIO_VERSION}" \
        "torchcodec==${VLLM_TORCHCODEC_VERSION}"

    EXPECTED_CUDA="${SGLANG_CUDA_VERSION}" \
    EXPECTED_CUDA_TAG="${SGLANG_CUDA_TAG}" \
    EXPECTED_TORCHCODEC="${VLLM_TORCHCODEC_VERSION}" \
    VIRTUAL_ENV="${VENV_RELEASE}" uv run --active --no-project python - <<'PY'
from importlib.metadata import version
import os

import torch
import torchaudio
import torchvision
import vllm
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch

assert torch.version.cuda == os.environ["EXPECTED_CUDA"], torch.version.cuda
assert os.environ["EXPECTED_CUDA_TAG"] in torch.__version__, torch.__version__
assert os.environ["EXPECTED_CUDA_TAG"] in torchaudio.__version__, torchaudio.__version__
assert os.environ["EXPECTED_CUDA_TAG"] in torchvision.__version__, torchvision.__version__
assert version("torchcodec") == os.environ["EXPECTED_TORCHCODEC"], version("torchcodec")
assert torch.cuda.get_device_capability() == (12, 0)

x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
weight = torch.ones(128, device="cuda", dtype=torch.float16)
out = torch.empty_like(x)
torch.ops._C.rms_norm(out, x, weight, 1e-6)
torch.cuda.synchronize()
reference = torch.nn.functional.rms_norm(x, (128,), weight, 1e-6)
torch.testing.assert_close(out, reference, rtol=2e-3, atol=2e-3)
print(
    f"Validated vllm={vllm.__version__} torch={torch.__version__} "
    f"torchvision={torchvision.__version__} torchaudio={torchaudio.__version__}"
)
print("Validated vLLM stable-libtorch CUDA RMSNorm operator")
PY

    VENV_RELEASE="${VENV_RELEASE}" \
    EXPECTED_CUDA_MAJOR="${SGLANG_CUDA_VERSION%%.*}" \
    "${python_bin}" - <<'PY'
import os
import re
import subprocess
from pathlib import Path

root = Path(os.environ["VENV_RELEASE"]) / "lib"
expected_major = os.environ["EXPECTED_CUDA_MAJOR"]
bad = []
for path in root.rglob("*.so*"):
    if not path.is_file():
        continue
    result = subprocess.run(
        ["readelf", "-d", path], capture_output=True, text=True, check=False
    )
    runtime_majors = set(re.findall(r"libcudart\.so\.(\d+)", result.stdout))
    if runtime_majors and runtime_majors != {expected_major}:
        bad.append(f"{path}: {sorted(runtime_majors)}")
if bad:
    raise SystemExit("Unexpected CUDA runtime links found:\n" + "\n".join(bad))
print(f"Validated: extension runtime links use CUDA {expected_major}")
PY

    printf '%s\n%s\n' \
        "${VLLM_ENV_REVISION}" "${VLLM_SOURCE_REVISION}" > "${MANIFEST}"
fi

mkdir -p "${VENV_ROOT}"
if [[ -e "${VLLM_VENV}" && ! -L "${VLLM_VENV}" ]]; then
    rm -rf "${VLLM_VENV}"
else
    rm -f "${VLLM_VENV}"
fi
ln -s "${VENV_RELEASE}" "${VLLM_VENV}"

echo "vLLM environment: ${VLLM_VENV} -> ${VENV_RELEASE}"
