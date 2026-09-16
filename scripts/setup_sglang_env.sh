#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/sglang.env"

CUDA_HOME="/usr/local/cuda-${SGLANG_CUDA_VERSION}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "ERROR: CUDA ${SGLANG_CUDA_VERSION} is not installed at ${CUDA_HOME}" >&2
    exit 1
fi
if ! "${CUDA_HOME}/bin/nvcc" --version | grep -q "release ${SGLANG_CUDA_VERSION}"; then
    echo "ERROR: ${CUDA_HOME}/bin/nvcc is not CUDA ${SGLANG_CUDA_VERSION}" >&2
    exit 1
fi
if [[ ! -f "${SGLANG_SOURCE_DIR}/python/pyproject.toml" ]]; then
    echo "ERROR: SGLang source not found at ${SGLANG_SOURCE_DIR}" >&2
    exit 1
fi
SGLANG_SOURCE_REVISION="$(git -C "${SGLANG_SOURCE_DIR}" rev-parse HEAD)"

PIDFILE="${SCRIPT_DIR}/../.sglang.pid"
if [[ -f "${PIDFILE}" ]]; then
    server_pid="$(<"${PIDFILE}")"
    if kill -0 "${server_pid}" 2>/dev/null; then
        echo "ERROR: stop the SGLang server before rebuilding its environment." >&2
        exit 1
    fi
fi

VENV_ROOT="$(dirname "${SGLANG_VENV}")"
VENV_RELEASE="${VENV_ROOT}/sglang-${SGLANG_ENV_REVISION}-${SGLANG_SOURCE_REVISION:0:12}"
MANIFEST="${VENV_RELEASE}/.super-quant-sglang-env"

if [[ -x "${VENV_RELEASE}/bin/python" && -f "${MANIFEST}" ]] &&
    [[ "$(sed -n '1p' "${MANIFEST}")" == "${SGLANG_ENV_REVISION}" ]] &&
    [[ "$(sed -n '2p' "${MANIFEST}")" == "${SGLANG_SOURCE_REVISION}" ]]; then
    echo "SGLang environment ${SGLANG_ENV_REVISION} is already installed."
else
    build_root="$(mktemp -d)"
    trap 'rm -rf "${build_root}"' EXIT

    mkdir -p "${build_root}/sglang"
    cp -a "${SGLANG_SOURCE_DIR}/python" "${build_root}/sglang/python"
    cp -a "${SGLANG_SOURCE_DIR}/rust" "${build_root}/sglang/rust"
    cp -a "${SGLANG_SOURCE_DIR}/proto" "${build_root}/sglang/proto"
    # Copy to each destination separately; one cp command treats the first
    # destination as another source and fails on a clean rebuild.
    cp "${SGLANG_SOURCE_DIR}/README.md" "${SGLANG_SOURCE_DIR}/LICENSE" \
        "${build_root}/sglang/"
    cp "${SGLANG_SOURCE_DIR}/README.md" "${SGLANG_SOURCE_DIR}/LICENSE" \
        "${build_root}/sglang/python/"

    pyproject="${build_root}/sglang/python/pyproject.toml"
    sed -i \
        -e "s|^  \"torch==.*|  \"torch==${SGLANG_TORCH_VERSION}\",|" \
        -e "s|^  \"torchaudio==.*|  \"torchaudio==${SGLANG_TORCHAUDIO_VERSION}\",|" \
        -e "s|^  \"torchvision.*|  \"torchvision==${SGLANG_TORCHVISION_VERSION}\",|" \
        -e "s|^  \"torchcodec==.*|  \"torchcodec==${SGLANG_TORCHCODEC_VERSION}\",|" \
        -e '/^  "sglang-kernel==.*",$/d' \
        -e '/^  "sgl-deep-gemm==.*",$/d' \
        -e '/^  "sgl-deep-ep==.*",$/d' \
        "${pyproject}"

    for dependency in \
        "torch==${SGLANG_TORCH_VERSION}" \
        "torchvision==${SGLANG_TORCHVISION_VERSION}" \
        "torchaudio==${SGLANG_TORCHAUDIO_VERSION}" \
        "torchcodec==${SGLANG_TORCHCODEC_VERSION}"; do
        if ! grep -qF "\"${dependency}\"" "${pyproject}"; then
            echo "ERROR: failed to pin ${dependency} in the SGLang dependency copy" >&2
            exit 1
        fi
    done

    if [[ "${SGLANG_CUDA_VERSION}" != "13.4" ]]; then
        echo "ERROR: SGLang requires CUDA 13.4" >&2
        exit 1
    fi

    native_wheel_dir="${SGLANG_NATIVE_WHEEL_DIR:-${HOME}/.cache/super-quant/wheels/${SGLANG_CUDA_TAG}-${SGLANG_SOURCE_REVISION:0:12}}"
    shopt -s nullglob
    kernel_wheels=("${native_wheel_dir}"/sglang_kernel-${SGLANG_KERNEL_VERSION}-*.whl)
    deep_gemm_wheels=("${native_wheel_dir}"/sgl_deep_gemm-${SGLANG_DEEP_GEMM_VERSION}-*.whl)
    deep_ep_wheels=("${native_wheel_dir}"/sgl_deep_ep-${SGLANG_DEEP_EP_VERSION}-*.whl)
    shopt -u nullglob
    if (( ${#kernel_wheels[@]} != 1 || ${#deep_gemm_wheels[@]} != 1 || ${#deep_ep_wheels[@]} != 1 )); then
        echo "ERROR: expected one sglang-kernel, sgl-deep-gemm, and sgl-deep-ep wheel in ${native_wheel_dir}" >&2
        exit 1
    fi
    native_wheels=("${kernel_wheels[0]}" "${deep_gemm_wheels[0]}" "${deep_ep_wheels[0]}")

    requirements="${build_root}/requirements.txt"
    uv pip compile "${pyproject}" \
        --python-version 3.12 \
        --prerelease allow \
        --extra-index-url "${SGLANG_TORCH_INDEX}" \
        --index-strategy unsafe-best-match \
        --no-emit-package sglang \
        --output-file "${requirements}"

    rm -rf "${VENV_RELEASE}"
    uv venv "${VENV_RELEASE}" --python 3.12
    python_bin="${VENV_RELEASE}/bin/python"
    uv pip install --python "${python_bin}" \
        --prerelease allow \
        --extra-index-url "${SGLANG_TORCH_INDEX}" \
        --index-strategy unsafe-best-match \
        --requirements "${requirements}"
    uv pip install --python "${python_bin}" --no-deps "${native_wheels[@]}"
    uv pip install --python "${python_bin}" \
        --no-build-isolation --no-deps --editable "${SGLANG_SOURCE_DIR}/python"
    uv pip install --python "${python_bin}" "pytest==${SGLANG_PYTEST_VERSION}"

    EXPECTED_CUDA="${SGLANG_CUDA_VERSION}" \
    EXPECTED_CUDA_MAJOR="${cuda_major}" \
    EXPECTED_CUDA_TAG="${SGLANG_CUDA_TAG}" \
    EXPECTED_TORCH="${SGLANG_TORCH_VERSION}" \
    EXPECTED_TORCHVISION="${SGLANG_TORCHVISION_VERSION}" \
    EXPECTED_TORCHAUDIO="${SGLANG_TORCHAUDIO_VERSION}" \
    EXPECTED_TORCHCODEC="${SGLANG_TORCHCODEC_VERSION}" \
    EXPECTED_KERNEL="${SGLANG_KERNEL_VERSION}" \
    EXPECTED_DEEP_GEMM="${SGLANG_DEEP_GEMM_VERSION}" \
    EXPECTED_DEEP_EP="${SGLANG_DEEP_EP_VERSION}" \
    "${python_bin}" - <<'PY'
import importlib
import importlib.metadata
import os

import torch
import torchaudio
import torchvision
from torch_memory_saver.utils import get_binary_path_from_package

expected = {
    "torch": os.environ["EXPECTED_TORCH"],
    "torchvision": os.environ["EXPECTED_TORCHVISION"],
    "torchaudio": os.environ["EXPECTED_TORCHAUDIO"],
    "torchcodec": os.environ["EXPECTED_TORCHCODEC"],
    "sglang-kernel": os.environ["EXPECTED_KERNEL"],
    "sgl-deep-gemm": os.environ["EXPECTED_DEEP_GEMM"],
    "sgl-deep-ep": os.environ["EXPECTED_DEEP_EP"],
}
for package, version in expected.items():
    actual = importlib.metadata.version(package)
    assert actual == version, (package, actual, version)
assert torch.version.cuda == os.environ["EXPECTED_CUDA"], torch.version.cuda
assert os.environ["EXPECTED_CUDA_TAG"] in torch.__version__, torch.__version__
assert os.environ["EXPECTED_CUDA_TAG"] in torchaudio.__version__, torchaudio.__version__
assert os.environ["EXPECTED_CUDA_TAG"] in torchvision.__version__, torchvision.__version__
assert torch.cuda.get_device_capability() == (12, 0)
for module in ("flashinfer", "deep_ep", "deep_gemm", "sgl_kernel", "sglang"):
    importlib.import_module(module)
for stem in (
    "torch_memory_saver_hook_mode_torch",
    "torch_memory_saver_hook_mode_preload",
):
    selected = get_binary_path_from_package(stem)
    assert f"_cu{os.environ['EXPECTED_CUDA_MAJOR']}." in selected.name, selected
x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
y = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
z = x @ y
torch.cuda.synchronize()
assert torch.isfinite(z).all()
print(
    f"Validated torch={torch.__version__} torchvision={torchvision.__version__} "
    f"torchaudio={torchaudio.__version__} device={torch.cuda.get_device_name()}"
)
PY

    VENV_RELEASE="${VENV_RELEASE}" EXPECTED_CUDA_MAJOR="${cuda_major}" \
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
    if not runtime_majors or runtime_majors == {expected_major}:
        continue
    if path.name.startswith("torch_memory_saver_hook_mode_"):
        continue
    bad.append(f"{path}: {sorted(runtime_majors)}")
if bad:
    raise SystemExit("Unexpected CUDA runtime links found:\n" + "\n".join(bad))
print(f"Validated: extension runtime links use CUDA {expected_major}")
PY

    printf '%s\n%s\n' \
        "${SGLANG_ENV_REVISION}" "${SGLANG_SOURCE_REVISION}" > "${MANIFEST}"
fi

mkdir -p "${VENV_ROOT}"
if [[ -e "${SGLANG_VENV}" && ! -L "${SGLANG_VENV}" ]]; then
    rm -rf "${SGLANG_VENV}"
else
    rm -f "${SGLANG_VENV}"
fi
ln -s "${VENV_RELEASE}" "${SGLANG_VENV}"

echo "SGLang environment: ${SGLANG_VENV} -> ${VENV_RELEASE}"
