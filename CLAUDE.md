# super-quant

Multi-model GGUF quantization pipeline deployed via llama.cpp on RTX PRO 6000 Blackwell (96GB VRAM). Currently targeting Qwen3.8-27B.

## Gotchas

- **CUDA 12.8 toolkit is pinned system-wide** — `make setup` auto-installs it to `/usr/local/cuda-12.8` and appends to `~/.bashrc`. The driver (596.36, reports "CUDA 13.2") stays untouched — driver and toolkit are separate things. CUDA 13.x toolkits have multiple bugs on Blackwell: 13.0-13.1 segfault MMQ kernels, 13.2 miscompiles IQ dequant kernels. CUDA 12.8 is NVIDIA's official recommendation for sm_120. See: https://forums.developer.nvidia.com/t/321330
- **Qwen3.x chat template is whitespace-sensitive** — `'{"enable_thinking":true}'` works, `'{"enable_thinking": true}'` silently fails (space after colon breaks it). Upstream llama.cpp now handles this via `--reasoning on` and `--reasoning-preserve` flags instead of `--chat-template-kwargs`.
- **MTP GGUFs are different from standard GGUFs** — MTP tensors (draft head weights) must be included at convert time. `convert_hf_to_gguf.py` from llama.cpp b9180+ includes them by default. Standard GGUFs without MTP tensors cannot use `--spec-type draft-mtp`.
- **`--spec-type mtp` was renamed to `--spec-type draft-mtp`** in llama.cpp around May 2026. Build b9375+ uses the new name.
- **MTP is lossless but kills concurrent throughput** — single-user: 91 t/s with draft-n-max=2 (1.8x over 50 t/s baseline). But MTP saturates GPU compute at 1 slot. At 5 concurrent users: baseline 173 t/s aggregate vs MTP 77 t/s (2.3x worse). Crossover at 2 users. The overhead is baked into the context (3x RS buffer for rollback snapshots), not per-decode — you can't adaptively disable it at runtime. For concurrent workloads, don't use `--spec-type draft-mtp`. EOS suppression inside `<think>` blocks is handled automatically by the reasoning budget sampler when `--reasoning on` is active — no per-request `thinking_budget_tokens` needed.
- **Qwen3.8 hybrid arch**: 16 of 64 layers use full attention with KV cache; the other 48 use GatedDeltaNet (SSM, no KV cache). KV cache quantization still matters for the 16 attention layers.
- **Turbo KV cache is broken upstream** — turbo4 quantize applies internal FWHT rotation that conflicts with the graph-level Hadamard rotation, producing garbled output. All turbo KV types (turbo2/3/4, TCQ variants) are affected. Use q8_0/q8_0.
- **Wikipedia-only calibration data overfits perplexity benchmarks** but produces worse real-world output for instruct models. The pipeline uses multi-domain calibration (chat/code/math/tool-calling/Chinese/long-context) deliberately — don't "simplify" back to wikitext.
- **Calibration uses token budgets with complete samples** — each domain group has a token target (~750K-1M). Samples are shuffled then collected until the budget is reached. The last sample is always included whole even if it overshoots. Never truncate mid-conversation. Special tokens (from the target model's tokenizer) are stripped automatically — some HF datasets embed `<|endoftext|>` or similar tokens that would distort the imatrix.
- **IQ quants only exist up to ~4.5 bpw** (IQ4_XS, IQ4_NL). There is no IQ5/IQ6/IQ8 — K-quants (Q5_K_M, Q6_K, Q8_0) are the only option at 5+ bits.
- **imatrix .dat format** is not self-describing — the legacy format has no magic or version header. `merge_imatrix.py` reads/writes it. The PyTorch generator (`src/generate_imatrix.py`) writes the same format.
- **imatrix generator needs system RAM for mmap** — `from_pretrained` mmaps safetensor shards via the kernel. If system RAM < largest shard (e.g. 31GB RAM, 49GB shard), the kernel rejects the mmap under `overcommit_memory=0`. The generator falls back to streaming shards to GPU via raw file I/O. Fix permanently with `sudo sysctl -w vm.overcommit_memory=1`.
- **Per-tensor overrides use regex matching** — `llama-quantize --tensor-type-file` treats tensor names as regex patterns via `std::regex_search` (substring match). Dots match any character and short names match longer ones (e.g. `output.weight` matches `blk.3.attn_output.weight`). Always use anchored, escaped patterns: `^blk\.3\.attn_output\.weight$=f16`. The sensitivity script generates these automatically.
- **Per-tensor overrides** (`configs/<model>/tensor_overrides.txt`) are model-specific — current Qwen3.8 file has 65 layers (48 SSM + 17 attn), all SSM recurrence tensors (alpha/beta/out) at F16 (source precision), MTP layer (blk.64) at F16. Small SSM state tensors (ssm_a, ssm_conv1d, ssm_dt, ssm_norm) stay F32 via llama-quantize's internal rules. Each model gets its own overrides in its config subdir.
- **Q4_K is an alias for Q4_K_M** which has internal rules promoting certain tensor categories (attn_output → Q5_K, attn_v → Q5_K in early layers, ffn_down → Q6_K in edge layers). The sensitivity script uses Q4_0 (no internal rules) to get uniform probing across all tensor groups.
- **Switching models**: change `MODEL_DIR` in `configs/model.env`, create `configs/<model>/model.env` with model-specific settings (MODEL_ID, NATIVE_CTX, GGUF_ARCH_KEY, quant types, DSpark drafter path), then run `make sensitivity` to generate tensor overrides.
- **Downloaded models use HF cache** — `01_download_model.sh` stores weights in `~/.cache/huggingface/hub/`, not in the project. Other tools can reuse them.
- **Generated GGUFs live in `~/models/gguf/`** (override with `SUPER_QUANT_MODELS` env var). Shared across projects, not in the repo.

## Architecture decisions

- **DI-MATRIX**: separate imatrices per domain (general/code/reasoning/agentic) merged with weighted blend, rather than a single imatrix from combined data. Both are generated — benchmark to determine which is better for this model.
- **PyTorch GPU imatrix generator** replaces `llama-imatrix`. Uses forward hooks to accumulate squared activations on GPU — no PCIe D2H copies during generation. Supports multi-GPU via `device_map="auto"`. Uses `gguf-py` `TensorNameMap` for model-agnostic HF→GGUF name mapping.
- **All quants are UD (Unsloth Dynamic-style)**: every quant gets per-tensor overrides + imatrix, not just the low-bit ones. The `UD-` prefix on output filenames reflects this.
- **Unsloth Dynamic-style quants** use `--tensor-type-file` flag (or `--output-tensor-type` in older builds). `04_quantize.sh` errors out if neither is supported — quality quants require per-tensor overrides.

## Benchmark results

Results are in `results/`. Key files:
- `results/benchmark_2026-08-17_001719.json` — Qwen3.8-27B-AEON quant comparison (PPL/KL/throughput)
- `results/dspark_benchmark_20260817.json` — DSpark vs MTP vs baseline single-user speculative decoding
- `results/concurrent_benchmark_20260817_025314.tsv` — DSpark vs baseline concurrent throughput (1-5 users)
- `results/sensitivity.json` — per-tensor group KL divergence from Q4_0 probing

Single-user DSpark (UD-Q8_0, reasoning_effort=medium): 1.9x code, 3.0x math, 2.2x reasoning, 1.2x creative vs baseline ~44 t/s.

Concurrent DSpark (UD-Q6_K, 1-5 users): 1.47x per-req at 1 user (72 vs 49 t/s), 1.36x at 5 users (45 vs 33 t/s). Aggregate throughput stays above baseline at every concurrency level (178 vs 156 t/s at 5 users). MTP drops below baseline at 2+ users.

## Deployment config

Default: `make serve` → UD-Q6_K, DSpark speculative decoding, 512k context, 5 parallel slots, q8_0 KV cache, unified KV, YaRN, vision.

- **DSpark speculative decoding** uses `RadixArk/Qwen3.8-27B-DSpark` (1.36B params, 2.6 GB BF16 GGUF). An extension of DFlash that adds a low-rank Markov head for better draft quality. Cross-attends to target model hidden states at layers 4/16/28/40/52. Block size 7, meaning 7 draft tokens per round. Scales to concurrent users — aggregate throughput stays above baseline at 1-5 users (unlike MTP which collapses at 2+ users).
- **DSpark acceptance varies by content type** — math/reasoning: 40-58% acceptance, 3.7-5.1 mean tokens per round. Creative writing: 13% acceptance, 1.9 mean tokens per round. The drafter excels at structured/predictable content.
- **Reasoning is per-request** — the serve script does not set `--reasoning on` or `--chat-template-kwargs`. The harness controls reasoning via per-request `reasoning_effort` (`xhigh`/`medium`/`low`/`none`). Qwen3.8 recommended sampling: temp=0.6, top_k=20, top_p=0.95.
- **YaRN auto-activates** when CTX > NATIVE_CTX (set per model in `configs/<model>/model.env`). `--override-kv ${GGUF_ARCH_KEY}.context_length=int:CTX` bypasses server-context.cpp cap bug (llama.cpp #22140).
- **Unified KV** (`-kvu`) — single shared KV buffer across all slots, better memory pooling with lazy allocation.
- **q8_0 KV cache** — hybrid SSM means only 16/64 layers have KV cache, so q8_0 is cheap. Turbo KV types are broken (see gotcha above).
- **MTP fallback** — use `SPEC_TYPE=mtp make serve` for MTP. Better on creative/agentic content but kills concurrent throughput (saturates GPU at 1 slot, 2.3x worse at 5 concurrent users).

## Paths

- llama.cpp: `/home/jinyang/src/llama.cpp` (upstream `ggml-org/llama.cpp`, no fork patches)
- Generated GGUFs: `~/models/gguf/` (override: `SUPER_QUANT_MODELS` env var)
- Downloaded models: `~/.cache/huggingface/hub/` (standard HF cache)
- Per-model configs: `configs/<MODEL_DIR>/` (tensor overrides, model.env)
- GPU: NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM, sm_120

## Commands

```bash
make help                                   # show all targets
make all                                    # full pipeline
make serve                                  # Q6_K, DSpark, 5 slots, q8_0 KV
make serve SPEC_TYPE=mtp                    # MTP instead of DSpark
make serve SPEC_TYPE=none                   # no speculative decoding
```
