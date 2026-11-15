# super-quant

Config-driven quantization and serving pipeline for hybrid language models. The checked-in deployment profile targets SM120 Blackwell, and the default configuration serves `Qwen3.8-Flash-Next` with Infernix (`make serve-infernix`).

## Error-prevention notes

- CUDA 13.4 is the shared toolkit for the project environment. Do not downgrade to CUDA 13.0-13.1, which crashed SM120 kernels on this GPU, or CUDA 13.2, which miscompiled kernels.
- Every pipeline and serving target must select behavior through `MODEL_DIR` and feature variables in `configs/<model>/model.env`. Do not add model-name branches to the Makefile or scripts. The default `Qwen3.8-Flash-Next` config currently defines production Infernix serving; other configs define quantization recipes.
- This workstation cannot run AWQ or GPTQ quantization for the 176B Qwen3.8-Flash-Next checkpoint because the job exceeds practical VRAM and system-memory capacity. This is a local resource limit, not a reason to disable those model-agnostic targets.
- Shared AWQ and GPTQ scripts interpret `configs/<model>/quantize.json`. Keep architecture targets and exclusions in that recipe, and reject already-compressed sources instead of re-quantizing them.
- Infernix uses `.infernix.pid` and a dated log per start under `logs/infernix/`; `.infernix.log` at the project root is a symlink to the newest one, and starts prune logs older than `INFERNIX_LOG_KEEP_DAYS` (default 30).
- The Qwen3.6/3.8 27B AEON models are hybrid architectures: 48 of 64 layers use GatedDeltaNet recurrence and 16 use full attention. Quantization error in recurrence tensors compounds across token positions.
- Calibration records are model-specific. Never reuse an artifact from another model; generated calibration data belongs under `calibration/<MODEL_DIR>`.
- JSONL is the canonical calibration source. Preserve every conversation as one complete role-aware record; never split or truncate a conversation to meet a token target, and do not replace the multi-domain set with Wikipedia-only data.
- Routed-MoE calibration must measure real router selections and fail its coverage policy. Do not enable synthetic all-expert calibration to hide uncovered experts.
- MTP helps single-request latency but reduces concurrent throughput for these models. Do not enable it by default for concurrent service.
- `max_shard_size="4GB"` is required when saving NVFP4 checkpoints. Larger default shards can exhaust system memory.
- The NVFP4 checkpoint needs `preprocessor_config.json`; the quantization save path does not add it automatically.
