# super-quant

Multi-model GGUF quantization pipeline for hybrid SSM/attention models. Produces Unsloth Dynamic-style (UD) quants with per-tensor overrides, GPU-native importance matrices, and GPU-native sensitivity analysis.

Built for llama.cpp on NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM). Currently targeting Qwen3.x-27B.

## Quantized Models

| Model | HuggingFace |
|-------|-------------|
| Qwen3.8-27B-AEON-Ultimate-Uncensored | [lambsea/Qwen3.8-27B-AEON-Ultimate-Uncensored-UD-GGUF](https://huggingface.co/lambsea/Qwen3.8-27B-AEON-Ultimate-Uncensored-UD-GGUF) |
| Qwen3.6-27B-AEON-Ultimate-Uncensored | [lambsea/Qwen3.6-27B-AEON-Ultimate-Uncensored-UD-GGUF](https://huggingface.co/lambsea/Qwen3.6-27B-AEON-Ultimate-Uncensored-UD-GGUF) |

## Pipeline

### Baseline (UD)

```
make all
```

Runs the full pipeline: download, convert, calibrate, imatrix, sensitivity, quantize, benchmark.

```
download   → HF model weights to local cache
convert    → F16 GGUF (with MTP tensors) + mmproj GGUF (vision encoder)
calibrate  → Multi-domain calibration data from 13 HF datasets
imatrix    → GPU-native importance matrix generation (PyTorch, 65k context)
sensitivity → GPU-native per-tensor KL divergence probing
quantize   → UD quants with 685 per-tensor overrides + imatrix
benchmark  → Throughput + perplexity + KL divergence vs F16
```

### AWQ Pre-Scaled (AWQ-UD)

```
make awq-all
```

Runs the AWQ pre-scaling pipeline, then the full GGUF pipeline on the AWQ-scaled weights:

```
awq-prescale  → AWQ channel scaling on BF16 model (lossless, saves plain BF16 checkpoint)
awq-convert   → AWQ-scaled BF16 → F16 GGUF
awq-imatrix   → Regenerate imatrix from AWQ-scaled model
awq-sensitivity → Regenerate sensitivity from AWQ-scaled model
awq-quantize  → UD quants from AWQ F16 GGUF + AWQ imatrix + AWQ overrides
```

AWQ (Activation-Aware Weight Quantization) redistributes weight magnitudes per channel to reduce outlier sensitivity during quantization. The transformation is lossless at BF16. Combined with K-quant block quantization, imatrix importance weighting, and UD per-tensor overrides, this produces better reasoning coherence than NVFP4 AWQ+GPTQ at similar bit widths.

Each step can run independently via `make <step>`.

## What Makes These Quants Different

### SSM Recurrence Preservation

Qwen3.x-27B is a hybrid GatedDeltaNet + attention model. 48 of 64 layers use a recurrent SSM where quantization error compounds across token positions. All SSM recurrence tensors (`ssm_alpha`, `ssm_beta`, `ssm_out`) are preserved at source precision (F16).

### GPU-Native Sensitivity Analysis

Each tensor group is probed by simulating Q4_0 quantization in-place on GPU and measuring KL divergence vs unquantized logits. No disk I/O, no subprocess calls. Runs in ~7 minutes vs 30-60 minutes with the llama.cpp subprocess approach.

### GPU-Native Imatrix Generation

Replaces `llama-imatrix` with a PyTorch forward-hook accumulator that keeps all computation on GPU. Zero PCIe D2H copies during generation. Processes 3M tokens at 65k context in ~27 minutes on a single GPU.

### Multi-Domain Calibration

Calibrated on 4 domains (general, code, reasoning, agentic) from 13 HF datasets rather than Wikipedia-only. Special tokens are stripped automatically. Samples are kept whole, never truncated mid-conversation. Per-domain imatrices are merged with equal weights (DI-MATRIX approach).

### Per-Tensor Overrides

685 explicit overrides per model. Every non-FFN tensor has an assigned precision based on measured sensitivity. No dependence on llama-quantize's internal promotion rules.

### AWQ Channel Pre-Scaling

AWQ redistributes weight magnitudes across input channels before quantization. Outlier channels that dominate a quantization block's range are scaled down, with compensating scales absorbed into adjacent LayerNorm parameters. The BF16 model is functionally identical, but quantizes with less error. The AWQ-UD pipeline applies this pre-scaling, then runs the full GGUF pipeline (imatrix, sensitivity, quantize) on the transformed weights.

## Switching Models

1. Create `configs/<model>/model.env` with model-specific settings
2. Set `MODEL_DIR` in `configs/model.env`
3. Run `make sensitivity` to generate tensor overrides for the new model
4. Run `make quantize` and `make benchmark`

See `configs/Qwen3.8-27B-AEON/model.env` for an example.

## Latest Results (Qwen3.8-27B-AEON)

| Quant | Size | tg t/s | PPL | KL mean | KL max | KL p99.9 |
|-------|------|--------|-----|---------|--------|----------|
| F16 | 50.9 GB | 30.0 | 5.7102 | — | — | — |
| UD-Q8_0 | 34.7 GB | 39.9 | 5.7181 | 0.0020 | 1.83 | 0.25 |
| **UD-Q6_K** | **30.6 GB** | **48.7** | **5.7288** | **0.0042** | **7.20** | **0.38** |
| UD-Q5_K_M | 28.7 GB | 53.0 | 5.7243 | 0.0087 | 6.20 | 0.98 |
| UD-IQ4_XS | 25.9 GB | 55.3 | 5.7288 | 0.0240 | 9.37 | 3.26 |

Benchmarked on NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), llama.cpp fork ([a4501150/llama.cpp](https://github.com/a4501150/llama.cpp)), pp=512, tg=128.

## Project Structure

```
configs/
  model.env                      # Active model selector
  Qwen3.8-27B-AEON/model.env     # Model-specific config
  Qwen3.8-27B-AEON/tensor_overrides.txt  # 685 per-tensor overrides
src/
  awq_prescale.py                # AWQ-only channel pre-scaling (saves BF16)
  quantize_nvfp4.py              # AWQ+GPTQ NVFP4 quantization
  generate_imatrix.py            # GPU-native imatrix generator (PyTorch)
  sensitivity_analysis.py        # GPU-native per-tensor KL probing
  generate_hybrid_overrides.py   # Sensitivity → tensor override file
  prepare_calibration.py         # Multi-domain calibration data builder
  merge_imatrix.py               # DI-MATRIX weighted merge
  compare_results.py             # Benchmark comparison table
  model_utils.py                 # Shared model loading and tensor mapping
scripts/
  00_setup.sh                    # Build llama.cpp + install deps
  01_download_model.sh           # Download from HuggingFace
  02_convert_to_gguf.sh          # HF → F16 GGUF + mmproj
  03_generate_imatrix_gpu.sh     # GPU imatrix wrapper
  04_quantize.sh                 # Quantize with overrides + imatrix
  05_benchmark.sh                # Throughput + PPL + KL benchmarks
  06_serve.sh                    # Launch llama-server
```

## Requirements

- NVIDIA GPU with 48+ GB VRAM (96 GB recommended for 27B models at 65k context)
- CUDA 12.9 toolkit (12.8 also works for llama.cpp)
- Python 3.12+ via uv
- llama.cpp (built by `make setup`)

## License

Apache-2.0
