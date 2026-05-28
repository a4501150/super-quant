---
license: apache-2.0
base_model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16
tags:
  - gguf
  - qwen3
  - qwen3.6
  - quantized
  - llama-cpp
  - imatrix
  - per-tensor
  - mtp
  - vision
  - blackwell
quantized_by: lambsea
pipeline_tag: text-generation
---

# Qwen3.6-27B-AEON-Ultimate-Uncensored — GGUF (UD Quants)

**Unsloth Dynamic-style (UD)** GGUF quantizations of [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16).

Every quant uses **per-tensor overrides** (sensitivity-driven) + **importance matrix** (multi-domain calibration). MTP speculative decoding and vision (mmproj) are preserved.

## Quant Comparison

| File | Quant | Size | tg t/s | PPL | KL mean | KL max | KL p99.9 | BPW |
|------|-------|------|--------|-----|---------|--------|----------|-----|
| [F16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) | F16 | 50.9 GB | 31.8 | 2.6022 | — | — | — | 16.00 |
| UD-Q8_0 | Q8_0 | 30.6 GB | 48.0 | 2.5951 | 0.0039 | 8.46 | 0.52 | 8.14 |
| **UD-Q6_K** | **Q6_K** | **26.4 GB** | **52.9** | **2.5980** | **0.0069** | **10.09** | **0.78** | **6.57** |
| UD-Q5_K_M | Q5_K_M | 24.5 GB | 56.3 | 2.6029 | 0.0093 | 11.75 | 1.42 | 5.69 |
| UD-IQ4_XS | IQ4_XS | 21.8 GB | 62.0 | 2.5730 | 0.0219 | 16.13 | 3.51 | 4.25 |

> **Recommended: UD-Q6_K** — best quality/size ratio. KL p99.9=0.78 is better than a plain Q8_0 (0.97) at 30% smaller size.

Benchmarked on NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), llama.cpp b9375, pp=512, tg=128, 3 reps, 32 chunks for PPL/KL.

## What Makes These Different

### Per-Tensor Sensitivity Analysis

Instead of applying one quant type uniformly, we probed each tensor group individually by quantizing only that group to Q4_0 while keeping everything else at F16, then measuring KL divergence against pure F16 logits. Tensor groups are then assigned precision tiers based on measured sensitivity:

| Precision | Tensor Groups | Why |
|-----------|--------------|-----|
| F32 | `ssm_alpha`, `ssm_beta` (96 tensors) | Critical SSM state parameters — regression-confirmed |
| F16 | `attn_qkv`, `attn_gate` (96 tensors), MTP layer (15 tensors) | Highest measured sensitivity (KL > 0.008) + MTP accuracy |
| Q8_0 | `ssm_out`, `attn_v`, `attn_q` (80 tensors) | High sensitivity (KL 0.004-0.008) |
| Q6_K | `attn_output`, `attn_k`, `ffn_down` edge (45 tensors) | Moderate sensitivity (KL 0.002-0.004) |
| Base | FFN middle layers, embeddings, etc. (229 tensors) | Low sensitivity — base quant + imatrix handles well |

### Multi-Domain Importance Matrix

Calibrated on a balanced mix of general text, code, reasoning, and agentic (tool-calling) samples (~750K tokens per domain) rather than Wikipedia-only. This prevents perplexity benchmark overfitting while maintaining real-world instruction-following quality.

### MTP + Vision Preserved

- **MTP (Multi-Token Prediction)**: The MTP draft head (blk.64) is pinned at F16 to preserve speculative decoding accuracy. Use `--spec-type mtp --spec-draft-n-max 3` for ~1.5-2x faster generation.
- **Vision**: The mmproj file contains the full vision encoder (334 tensors, 885 MB). Use `--mmproj` flag with llama-server for image/video understanding.

## Files

| File | Description | Size |
|------|-------------|------|
| `Qwen3.6-27B-AEON-UD-Q8_0.gguf` | Highest quality quantization | 30.6 GB |
| `Qwen3.6-27B-AEON-UD-Q6_K.gguf` | **Recommended** — best quality/size | 26.4 GB |
| `Qwen3.6-27B-AEON-UD-Q5_K_M.gguf` | Balanced | 24.5 GB |
| `Qwen3.6-27B-AEON-UD-IQ4_XS.gguf` | Smallest, for constrained VRAM | 21.8 GB |
| `Qwen3.6-27B-AEON-mmproj-F16.gguf` | Vision encoder (use with `--mmproj`) | 885 MB |
| `imatrix_merged.dat` | Importance matrix for requantization | 13 MB |

## Usage

### llama-server (recommended)

```bash
# Q6_K with YaRN 512k context, 5 concurrent slots, MTP + vision
llama-server \
    -m Qwen3.6-27B-AEON-UD-Q6_K.gguf \
    --mmproj Qwen3.6-27B-AEON-mmproj-F16.gguf \
    -ngl 99 \
    --flash-attn \
    -c 2621440 \
    --parallel 5 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    -kvu \
    --cache-ram -1 \
    --rope-scaling yarn \
    --rope-scale 2.0 \
    --yarn-orig-ctx 262144 \
    --override-kv "qwen35.context_length=int:524288" \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --jinja \
    --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
    --host 0.0.0.0 --port 8080
```

> **Note:** `--spec-type draft-mtp` requires llama.cpp b9375+. All flags above work with stock llama.cpp. Our [fork](https://github.com/a4501150/llama.cpp) adds DFlash and TurboQuant KV cache support (`turbo2`/`turbo3`/`turbo4`).

### llama-cli

```bash
llama-cli \
    -m Qwen3.6-27B-AEON-UD-Q6_K.gguf \
    -ngl 99 \
    --flash-attn \
    -c 524288 \
    --rope-scaling yarn \
    --rope-scale 2.0 \
    --yarn-orig-ctx 262144 \
    --jinja \
    --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
```

### Chat Template Notes

- `enable_thinking` activates the model's reasoning mode (chain-of-thought in `<think>` blocks).
- `preserve_thinking` retains reasoning blocks in conversation history, preventing "amnesia" during multi-turn and tool-calling loops. Defaults to false in the stock template.
- **No spaces** after colons in the JSON — Qwen3.6's template parser is whitespace-sensitive.

## Architecture

Qwen3.6-27B is a **hybrid SSM-attention** model:
- 64 transformer layers + 1 MTP layer (blk.0–64)
- 48 SSM layers (GatedDeltaNet, no KV cache) + 16 full attention layers + 1 MTP attention
- 27B total parameters, 24 attention heads, 4 KV heads, head dim 256
- Vocab: 248,320 tokens, max context: 262,144 tokens

## Quantization Pipeline

Built with [super-quant](https://github.com/lambsea/super-quant):

1. Convert HF → F16 GGUF (with MTP) + mmproj GGUF (vision)
2. Multi-domain calibration data preparation
3. Per-domain importance matrix generation + weighted merge
4. Per-tensor sensitivity analysis (KL divergence probing)
5. Quantize with hybrid-optimal per-tensor overrides + imatrix
6. Benchmark: throughput + perplexity + KL divergence vs F16

## Links

- **HuggingFace**: [lambsea/Qwen3.6-27B-AEON-Ultimate-Uncensored-UD-GGUF](https://huggingface.co/lambsea/Qwen3.6-27B-AEON-Ultimate-Uncensored-UD-GGUF)
- **GitHub (quantization pipeline)**: [a4501150/super-quant](https://github.com/a4501150/super-quant)
- **Base model**: [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)

## Credits

- Base model: [AEON-7](https://huggingface.co/AEON-7)
- Architecture: [Qwen Team](https://huggingface.co/Qwen)
- Quantization: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Sensitivity methodology inspired by [APEX quant research](https://github.com/APEX-Quantization)
