#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"
source "${SCRIPT_DIR}/../configs/sglang.env"

echo "=== Super-Quant Setup ==="
echo "llama.cpp dir: ${LLAMACPP_DIR}"
echo "CUDA arch:     sm_${CUDA_ARCH} (Blackwell)"

# ---------------------------------------------------------------
# 1. Verify and pin the shared CUDA toolkit.
# ---------------------------------------------------------------
CUDA_VERSION="${SGLANG_CUDA_VERSION}"
CUDA_TOOLKIT="/usr/local/cuda-${CUDA_VERSION}"

if [[ ! -x "${CUDA_TOOLKIT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA ${CUDA_VERSION} toolkit is required at ${CUDA_TOOLKIT}."
    echo "Install that toolkit without replacing the NVIDIA display driver."
    exit 1
fi
if ! "${CUDA_TOOLKIT}/bin/nvcc" --version | grep -q "release ${CUDA_VERSION}"; then
    echo "ERROR: ${CUDA_TOOLKIT}/bin/nvcc is not CUDA ${CUDA_VERSION}."
    exit 1
fi
echo "CUDA ${CUDA_VERSION} toolkit found at ${CUDA_TOOLKIT}"

BASHRC="${HOME}/.bashrc"
MARKER="# CUDA ${CUDA_VERSION} toolkit (pinned by super-quant)"
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

# Set the selected toolkit for the rest of this script.
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
    echo "  cmake -S . -B ${LLAMACPP_BUILD} -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build ${LLAMACPP_BUILD} -j\$(nproc)"
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
