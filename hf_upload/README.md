---
license: apache-2.0
base_model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16
tags:
  - gguf
  - qwen3
  - quantized
  - llama-cpp
  - imatrix
  - mtp
  - vision
pipeline_tag: text-generation
---

# Qwen3.6-27B-AEON UD GGUF

GGUF quantizations of `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`, built with the `super-quant` multi-domain calibration pipeline.

Each `UD-` model uses an importance matrix and 685 anchored per-tensor overrides. The override file preserves selected non-FFN, recurrent-state, and MTP tensors at F16 while the requested base quantization is applied to the remaining weights.

## Files

| File | Size | Intended use |
| --- | ---: | --- |
| `Qwen3.6-27B-AEON-UD-Q8_0.gguf` | 34.74 GiB | Highest GGUF precision |
| `Qwen3.6-27B-AEON-UD-Q6_K.gguf` | 30.57 GiB | Recommended quality and size balance |
| `Qwen3.6-27B-AEON-UD-Q5_K_M.gguf` | 28.68 GiB | Smaller balanced model |
| `Qwen3.6-27B-AEON-UD-IQ4_XS.gguf` | 25.93 GiB | Lowest memory use |
| `Qwen3.6-27B-AEON-mmproj-F16.gguf` | 0.86 GiB | Vision projector |
| `imatrix_merged.dat` | 13.0 MiB | Importance matrix for requantization |

## Measured results

Results were measured on an NVIDIA RTX PRO 6000 Blackwell with llama.cpp. Generation throughput used a 512-token prompt and 128 generated tokens. Perplexity and KL divergence used 32 chunks from the calibration test corpus.

| Quant | Generation tokens/s | Perplexity | Mean KL divergence |
| --- | ---: | ---: | ---: |
| F16 | 30.9 | 5.7215 | — |
| UD-Q8_0 | 44.7 | 5.7055 | 0.002855 |
| UD-Q6_K | 49.1 | 5.7046 | 0.004479 |
| UD-Q5_K_M | 46.9 | 5.7589 | 0.011128 |
| UD-IQ4_XS | 56.5 | 5.7630 | 0.023572 |

These measurements compare formats on one machine. They are not a general quality ranking for all prompts or runtimes.

## llama-server

```bash
llama-server \
  -m Qwen3.6-27B-AEON-UD-Q6_K.gguf \
  --mmproj Qwen3.6-27B-AEON-mmproj-F16.gguf \
  -ngl 99 \
  -fa on \
  -c 262144 \
  --parallel 1 \
  --jinja \
  --reasoning on \
  --reasoning-preserve \
  --host 0.0.0.0 \
  --port 8080
```

For MTP speculative decoding, use a llama.cpp build that supports the included MTP tensors and add:

```bash
--spec-type draft-mtp --spec-draft-n-max 3
```

MTP is most useful for one active request. Measure it before use with concurrent traffic because its extra graph and state work can reduce aggregate throughput.

## Build method

1. Convert the source checkpoint to source-precision GGUF with MTP and vision tensors.
2. Build complete calibration samples from general, code, reasoning, and tool-use domains.
3. Generate one importance matrix per domain and merge them.
4. Measure tensor-group sensitivity against source-precision logits.
5. Quantize with the merged importance matrix and model-specific overrides.
6. Compare throughput, perplexity, and output distributions against the source-precision GGUF.

The pipeline source and exact configuration are in this repository under `configs/Qwen3.6-27B-AEON`.
