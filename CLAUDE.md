# super-quant

Config-driven quantization and serving pipeline for hybrid language models. The checked-in deployment profile targets SM120 Blackwell, and the default configuration serves `Qwen3.8-Flash-Next` with SGLang.

## Error-prevention notes

- CUDA 13.4 is the shared toolkit for the project environment, SGLang, vLLM, and llama.cpp. Do not downgrade to CUDA 13.0-13.1, which crashed SM120 MMQ kernels, or CUDA 13.2, which miscompiled IQ dequantization kernels.
- Every pipeline and serving target must select behavior through `MODEL_DIR` and feature variables in `configs/<model>/model.env`. Do not add model-name branches to the Makefile or scripts. The default `Qwen3.8-Flash-Next` config currently defines production SGLang serving; choose a config with GGUF fields for GGUF stages.
- This WSL workstation cannot run AWQ or GPTQ quantization for the 176B Qwen3.8-Flash-Next checkpoint because the job exceeds practical VRAM and system-memory capacity. This is a local resource limit, not a reason to disable those model-agnostic targets.
- Shared AWQ and GPTQ scripts interpret `configs/<model>/quantize.json`. Keep architecture targets and exclusions in that recipe, and reject already-compressed sources instead of re-quantizing them.
- SGLang, vLLM, and the project use separate environments because their dependency constraints conflict. Do not install serving packages into the project `.venv`.
- SGLang native wheels are keyed by CUDA tag and SGLang source revision under `~/.cache/super-quant/wheels/`. Rebuild the wheels before changing the SGLang revision.
- Stop the production server before an eight-job native CUDA build. The 2026-09-16 OOM came from compiling while the model still held its large host-memory allocation; do not reduce production performance settings or permanently throttle builds because of that overlap.
- The qualified SM120 SGLang stack uses CUDA 13.4 nightly PyTorch and revision-matched native wheels. Stable PyTorch only publishes older CUDA runtimes; do not substitute its cu130 wheels for this production environment.
- Keep each qualified, dated SGLang production branch unchanged. For an upstream update, create a new dated branch from `origin/main`, apply only the local patches that upstream still lacks, then rebuild the revision-keyed wheels and repeat qualification.
- SGLang uses `.sglang.pid`/`.sglang.log`; vLLM uses `.vllm.pid`/`.vllm.log`. They are independent processes but normally compete for the same configured port and GPU.
- `Qwen3.8-Flash-Next` reads its 95.4 GiB BF16 PLE table directly from the selected safetensors checkpoint. Checkpoint prefetch must skip the direct PLE source files instead of warming the full table, and the direct source must not be combined with SGLang's built-in pinned/file PLE offload.
- FlashInfer GDN prefill on SM120 requires CUDA 13 or newer. Production uses FlashInfer for GDN prefill, decode, and verify.
- The production HiCache log reports the 15 GB L2 host tier. The L3 file tier is capped separately at 50 GB, starts eviction near 45 GB, and uses selective write-through to avoid persisting one-use branches.
- To empty all HiCache tiers without restarting an idle SGLang server, call `POST /flush_cache?timeout=<seconds>` and then `POST /hicache/storage-backend/clear`, and use this sequence in benchmarks only when a restart serves solely to clear cache state rather than change launch settings or reset process state.
- HiCache L3 restores through L2 before data reaches the GPU. Missing L3 suffix pages cause recomputation, not foreign state restoration.
- Byte-identical restored pages can still produce different tokens. GDN and CUTLASS NVFP4 MoE execution is batch-sensitive near tied logits. Treat digest conflicts, foreign markers, or reordered markers as cache corruption; a marker omission without a digest mismatch is a generation-path difference.
- RecoverSSM work must finish before Mamba state copy, donation, quantization, free, or slot reuse. Preserve this ordering when changing cache code.
- The Qwen3.6/3.8 27B AEON models are hybrid architectures: 48 of 64 layers use GatedDeltaNet recurrence and 16 use full attention. Quantization error in recurrence tensors compounds across token positions.
- Per-tensor override patterns are regular expressions. Anchor and escape tensor names, for example `^blk\.3\.attn_output\.weight$=f16`; an unescaped dot or unanchored name can match unrelated tensors.
- Calibration records, tensor overrides, and importance matrices are model-specific. Never reuse an artifact from another model or AWQ channel basis; generated calibration data belongs under `calibration/<MODEL_DIR>`.
- JSONL is the canonical calibration source. Preserve every conversation as one complete role-aware record; never split or truncate a conversation to meet a token target, and do not replace the multi-domain set with Wikipedia-only data.
- Routed-MoE calibration must measure real router selections and fail its coverage policy. Do not enable synthetic all-expert calibration to hide uncovered experts.
- MTP tensors must be included during GGUF conversion. Current llama.cpp uses `--spec-type draft-mtp`, not `--spec-type mtp`.
- MTP helps single-request latency but reduces concurrent throughput for these models. Do not enable it by default for concurrent service.
- TurboQuant KV cache types are not usable here because their internal rotation conflicts with the graph-level Hadamard rotation. Use supported standard KV types.
- Native NVFP4 and NVFP4 converted to GGUF are not equivalent. GGUF conversion dequantizes FP8 layers and does not preserve the native compressed-tensor inference path.
- `max_shard_size="4GB"` is required when saving NVFP4 checkpoints. Larger default shards can exhaust system memory.
- The NVFP4 checkpoint needs `preprocessor_config.json`; the quantization save path does not add it automatically.
- `make all` uses the existing tensor override file and does not run sensitivity analysis. Run `make sensitivity` after a model or policy change.
