# super-quant

`super-quant` is a config-driven quantization and serving pipeline for hybrid GatedDeltaNet/attention language models. The checked-in deployment profile is tuned for an NVIDIA RTX PRO 6000 Blackwell GPU, but model selection and model-specific behavior come from `configs/<model>/model.env`.

The repository has two main roles:

- Build calibrated GGUF models for llama.cpp.
- Run NVFP4 checkpoints with SGLang or vLLM.

## Current configurations

| Configuration | Current use |
| --- | --- |
| `Qwen3.8-Flash-Next` | Default production serving with SGLang, PLE SSD offload, and HiCache |
| `Qwen3.8-27B-AEON` | GGUF, AWQ-UD, NVFP4, llama.cpp, and SGLang workflows |
| `Qwen3.6-27B-AEON` | GGUF workflows and published artifacts |

These are implementations of the same model-config interface, not hardcoded pipeline targets. `configs/model.env` selects `Qwen3.8-Flash-Next` by default, so `make serve` starts SGLang with that model. Override the selection for one command with `MODEL_DIR=<name>`.

## Requirements

- Linux or WSL2 on an SM120 Blackwell GPU.
- CUDA 13.4 at `/usr/local/cuda-13.4`.
- Python 3.12 or newer, managed by `uv`.
- At least 64 GB of system memory for SGLang or vLLM startup. Configure swap on memory-constrained WSL2 systems because FlashInfer JIT compilation can exhaust RAM.
- llama.cpp at `~/src/llama.cpp`.
- SGLang source at `~/src/sglang` for SGLang serving.
- vLLM source at `~/src/vllm` for vLLM serving.

The configured SGLang environment also requires locally built CUDA 13.4 native wheels under `~/.cache/super-quant/wheels/cu134-<sglang-revision>`. `make setup-sglang` reports the exact path if the wheels are missing.

## Setup

Install the project dependencies and validate the shared CUDA toolkit and llama.cpp build:

```bash
make setup
```

`make setup` pins CUDA 13.4 in `~/.bashrc`; it does not build llama.cpp. If the configured build is absent, build it with CUDA 13.4 and SM120 support:

```bash
cmake -S ~/src/llama.cpp \
  -B ~/src/llama.cpp/build-cu134 \
  -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ~/src/llama.cpp/build-cu134 -j"$(nproc)"
```

Stop any active server, then build or validate the isolated serving environments as needed:

```bash
make setup-sglang
make setup-vllm
```

The launchers also check environment manifests and rebuild stale environments.

## Production serving

Start the configured SGLang server:

```bash
make serve
```

The current `Qwen3.8-Flash-Next` configuration uses:

- FlashInfer attention and GDN prefill, decode, and verify.
- CUTLASS FP8 matrix multiplication, with the NVFP4 backend selected automatically.
- FP8 E4M3 KV cache.
- Breakable CUDA graphs.
- A 262,144-token context limit and six running requests.
- PLE mmap offload from `~/models/ple/Qwen3.8-Flash-Next-NVFP4`.
- A 12 GB HiCache host tier and a file-backed SSD tier capped at 50 GB.

The SSD tier starts eviction at 90% of its cap and keeps 50 GB of disk space free. Its default directory is `~/models/hicache`.

SGLang and vLLM are independent servers with separate process state:

```bash
make stop       # stop SGLang (.sglang.pid)
make stop-vllm  # stop vLLM (.vllm.pid)
```

Start the alternative servers with:

```bash
make serve-vllm
MODEL_DIR=Qwen3.8-27B-AEON SPEC_TYPE=none make serve-llama
```

SGLang logs to `.sglang.log`; vLLM logs to `.vllm.log`. They normally use the same configured port and GPU, so do not start both with the same resources. llama.cpp runs in the foreground and must be stopped separately. Omit `SPEC_TYPE=none` only when the active model config defines an installed speculative draft.

## GGUF pipeline

All stages use the active `MODEL_DIR`. Select any model configuration that defines the GGUF and calibration fields required by the stages you run. For example:

```bash
export MODEL_DIR=Qwen3.8-27B-AEON
```

A complete fresh baseline run is:

```bash
make setup
make download
make convert
make calibrate
make imatrix
make sensitivity
make quantize
make bench-llamacpp
make compare
```

The stages produce:

1. A source-precision GGUF with MTP tensors.
2. Model-scoped multi-domain JSONL calibration records and deterministic text renders.
3. A disjoint structured holdout for sensitivity and perplexity checks.
4. Per-domain importance matrices and a weighted merged matrix.
5. Sensitivity-driven, model-specific tensor overrides.
6. UD-prefixed GGUF quants that use both the importance matrix and overrides.
7. Throughput, perplexity, and output-distribution comparison results.

JSONL is the canonical calibration format. It retains complete conversations, message roles, tool definitions, source revisions, and stable record IDs. The `.txt` files are deterministic renders of the same records for llama.cpp tools. A manifest records the build policy, source counts, tokenizer revision, deduplication counts, and artifact hashes under `calibration/<MODEL_DIR>/`.

`make all` runs the normal pipeline with the existing tensor override file. It does not regenerate sensitivity data. Run `make sensitivity` when the model or override policy changes.

### AWQ-UD pipeline

AWQ channel pre-scaling uses separate checkpoints, matrices, sensitivity results, overrides, and GGUF names. A compatible model provides `configs/<model>/quantize.json`, which selects its source-precision checkpoint, calibration policy, module targets, and exclusions. Run it on a host with enough GPU and system memory:

```bash
uv sync --extra quantize
MODEL_DIR=<model> make calibrate
MODEL_DIR=<model> make awq-all
```

Do not use an already quantized checkpoint as the source, and do not mix AWQ artifacts with baseline weights. AWQ changes the channel basis used by later calibration stages. The checked-in AWQ recipes select approximately 128K effective tokens across weighted domains and pack only complete conversations; overlength conversations are reported and skipped rather than truncated.

### Native NVFP4 pipeline

Create an AWQ plus GPTQ NVFP4 checkpoint for any model with a quantization recipe on a sufficiently large host:

```bash
uv sync --extra quantize
MODEL_DIR=<model> make calibrate
MODEL_DIR=<model> make quantize-nvfp4
```

The checked-in native recipes use approximately 1M effective weight-calibration tokens for the dense 27B models and 2M for Flash-Next. Static FP8 KV-cache scales use a separate 262K-token pass with explicit position offsets through the 262,144-token context range. Flash-Next also runs a real router preflight, writes `expert_coverage.json`, and fails if any routed expert does not meet its coverage policy. Convert the checkpoint to GGUF only when llama.cpp compatibility is required:

```bash
MODEL_DIR=<model> make convert-nvfp4
```

The current WSL workstation cannot quantize the 176B Qwen3.8-Flash-Next checkpoint with AWQ or GPTQ because the job exceeds its practical VRAM and system-memory capacity. This is a host limit, not a pipeline or model-config restriction; larger machines can run the same targets with an appropriate configuration.

Native compressed-tensor serving and converted GGUF serving are different inference paths and should be benchmarked separately.

## Benchmarks and tests

```bash
make bench-llamacpp  # GGUF quality and llama.cpp throughput
make bench-sglang    # SGLang performance and HiCache correctness
make compare         # result summary
make test            # unit tests
```

A normal `bench-sglang` run executes both public modes. Use `BENCH_PHASES=performance` or `BENCH_PHASES=correctness` to limit it. The individual `single`, `conc`, and `prefill` phases remain available for focused diagnosis. Run `uv run python scripts/bench_sglang_request.py --help` for request-harness options.

Benchmark JSON files are stored in `results/`. The CUDA 13.4 production qualification is recorded in:

- `results/sglang_benchmark_20260911_114909.json`
- `results/sglang_benchmark_20260911_115654.json`

The correctness run compared 17,366 restored page digests with no write conflicts, read conflicts, or mismatches.

## Model configuration

Every Make target resolves the active model through `configs/model.env` and then loads `configs/<model>/model.env`. A model file defines only the features that model uses:

- Common identity: `MODEL_ID`, `MODEL_NAME`, and `NATIVE_CTX`.
- GGUF stages: `GGUF_ARCH_KEY`, `QUANT_TYPES`, calibration domains, and imatrix settings.
- Native quantization: checkpoint and output paths used by AWQ, GPTQ, or NVFP4. Structured recipe policy lives in the optional `configs/<model>/quantize.json` file.
- Serving: model checkpoint, alias, context, backend, cache, and optional speculative-decoding settings.

A model does not need variables or a recipe for unsupported stages. If a selected target needs missing model policy, add it to that model's config directory instead of adding model-name conditions to the Makefile or scripts.

Machine paths and shared build settings live in `configs/model.env`. CUDA and serving-environment pins live in `configs/sglang.env` and `configs/vllm.env`.

Generated calibration data is stored in `calibration/<MODEL_DIR>/` and ignored by Git. Generated models are outside the repository:

- GGUF: `~/models/gguf`
- NVFP4: `~/models/nvfp4`
- Hugging Face downloads: `~/.cache/huggingface/hub`
- HiCache SSD data: `~/models/hicache`
- PLE mmap data: `~/models/ple`

Run `make help` for the main target list.

## License

Apache-2.0
