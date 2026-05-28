#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

echo "=== Super-Quant Setup ==="
echo "llama.cpp dir: ${LLAMACPP_DIR}"
echo "CUDA arch:     sm_${CUDA_ARCH} (Blackwell)"

# ---------------------------------------------------------------
# 1. Install + pin CUDA 12.8 toolkit
#
# CUDA 12.8 is NVIDIA's official recommendation for llama.cpp on
# Blackwell (sm_120). It's not a workaround — it's actually faster:
#   - MMQ kernels work (CUDA 13.0-13.1 crash, segfault)
#   - IQ quant kernels compile correctly (CUDA 13.2 produces gibberish)
#   - cuBLAS has Blackwell-optimized routines
#   - Generates native cubin (no PTX JIT overhead)
#
# The toolkit (nvcc, cuBLAS, headers) is separate from the driver.
# Driver 596.36 stays untouched — it supports CUDA 12.x just fine.
#
# Refs:
#   https://forums.developer.nvidia.com/t/321330 (NVIDIA migration guide)
#   https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-cuda-toolkit-trap-llama-cpp
# ---------------------------------------------------------------
CUDA_VERSION="12.8"
CUDA_TOOLKIT="/usr/local/cuda-${CUDA_VERSION}"
CUDA_INSTALLER_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run"
CUDA_INSTALLER="/tmp/cuda_12.8.1_linux.run"

if [[ -d "${CUDA_TOOLKIT}" ]] && "${CUDA_TOOLKIT}/bin/nvcc" --version &>/dev/null; then
    echo "CUDA ${CUDA_VERSION} toolkit already installed at ${CUDA_TOOLKIT}"
else
    echo "CUDA ${CUDA_VERSION} toolkit not found. Installing..."
    echo "(This only installs the compiler toolkit — your GPU driver stays untouched.)"
    echo ""

    if [[ ! -f "${CUDA_INSTALLER}" ]]; then
        echo "Downloading CUDA ${CUDA_VERSION} toolkit..."
        wget -q --show-progress -O "${CUDA_INSTALLER}" "${CUDA_INSTALLER_URL}"
    fi

    echo "Installing to ${CUDA_TOOLKIT} (needs sudo for /usr/local)..."
    sudo sh "${CUDA_INSTALLER}" \
        --toolkit \
        --silent \
        --toolkitpath="${CUDA_TOOLKIT}" \
        --no-man-page \
        --no-opengl-libs \
        --no-drm

    if ! "${CUDA_TOOLKIT}/bin/nvcc" --version &>/dev/null; then
        echo "ERROR: CUDA installation failed. Try manually:"
        echo "  sudo sh ${CUDA_INSTALLER} --toolkit --toolkitpath=${CUDA_TOOLKIT}"
        exit 1
    fi

    echo "CUDA ${CUDA_VERSION} installed successfully."
    rm -f "${CUDA_INSTALLER}"
fi

# Pin CUDA 12.8 system-wide in ~/.bashrc
BASHRC="${HOME}/.bashrc"
MARKER="# CUDA 12.8 toolkit (pinned by super-quant)"
if ! grep -qF "${MARKER}" "${BASHRC}" 2>/dev/null; then
    cat >> "${BASHRC}" <<EOF

${MARKER}
export CUDA_HOME="${CUDA_TOOLKIT}"
export PATH="${CUDA_TOOLKIT}/bin:\${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLKIT}/lib64:\${LD_LIBRARY_PATH:-}"
EOF
    echo "Pinned CUDA ${CUDA_VERSION} in ${BASHRC}"
else
    echo "CUDA ${CUDA_VERSION} already pinned in ${BASHRC}"
fi

# Set for the rest of this script (bashrc has a non-interactive guard,
# so `source ~/.bashrc` won't work here — just export directly)
export CUDA_HOME="${CUDA_TOOLKIT}"
export PATH="${CUDA_TOOLKIT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLKIT}/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA toolkit: ${CUDA_HOME} ($(nvcc --version | grep -oP 'V[\d.]+'))"

# ---------------------------------------------------------------
# 2. Python environment (uv)
# ---------------------------------------------------------------
echo ""
echo "--- Setting up Python environment ---"

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --project "${PROJECT_DIR}"

echo "Python $(uv run --project "${PROJECT_DIR}" python3 --version) | uv $(uv --version)"

# ---------------------------------------------------------------
# 3. Verify llama.cpp is built
# ---------------------------------------------------------------
echo ""
echo "--- Checking llama.cpp ---"

MISSING=0
for bin in llama-quantize llama-imatrix llama-server llama-cli llama-perplexity; do
    if [[ -x "${LLAMACPP_BUILD}/bin/${bin}" ]]; then
        echo "  OK: ${bin}"
    else
        echo "  MISSING: ${bin}"
        MISSING=1
    fi
done

if [[ ${MISSING} -eq 1 ]]; then
    echo ""
    echo "ERROR: llama.cpp is not built. Build it first:"
    echo ""
    echo "  cd ${LLAMACPP_DIR}"
    echo "  cmake --preset blackwell"
    echo "  cmake --build build-blackwell -j\$(nproc)"
    echo ""
    exit 1
fi

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "ERROR: convert_hf_to_gguf.py not found at ${CONVERT_SCRIPT}"
    exit 1
fi

echo ""
echo "=== Setup complete ==="
echo "CUDA toolkit:  ${CUDA_HOME}"
echo "Python:        uv managed"
echo "llama.cpp:     ${LLAMACPP_BUILD}/"
echo ""
echo "Next: build llama.cpp if not done, then make download"
