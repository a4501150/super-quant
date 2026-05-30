# super-quant

Multi-model GGUF quantization pipeline deployed via llama.cpp on RTX PRO 6000 Blackwell (96GB VRAM). Currently targeting Qwen3.6-27B.

## Gotchas

- **CUDA 12.8 toolkit is pinned system-wide** — `make setup` auto-installs it to `/usr/local/cuda-12.8` and appends to `~/.bashrc`. The driver (596.36, reports "CUDA 13.2") stays untouched — driver and toolkit are separate things. CUDA 13.x toolkits have multiple bugs on Blackwell: 13.0-13.1 segfault MMQ kernels, 13.2 miscompiles IQ dequant kernels. CUDA 12.8 is NVIDIA's official recommendation for sm_120. See: https://forums.developer.nvidia.com/t/321330
- **Qwen3.6 chat template is whitespace-sensitive** — `'{"enable_thinking":true}'` works, `'{"enable_thinking": true}'` silently fails (space after colon breaks it).
- **MTP GGUFs are different from standard GGUFs** — MTP tensors (draft head weights) must be included at convert time. `convert_hf_to_gguf.py` from llama.cpp b9180+ includes them by default. Standard GGUFs without MTP tensors cannot use `--spec-type draft-mtp`.
- **`--spec-type mtp` was renamed to `--spec-type draft-mtp`** in llama.cpp around May 2026. Build b9375+ uses the new name.
- **MTP is lossless but kills concurrent throughput** — single-user: 91 t/s with draft-n-max=2 (1.8x over 50 t/s baseline). But MTP saturates GPU compute at 1 slot. At 5 concurrent users: baseline 173 t/s aggregate vs MTP 77 t/s (2.3x worse). Crossover at 2 users. The overhead is baked into the context (3x RS buffer for rollback snapshots), not per-decode — you can't adaptively disable it at runtime. For concurrent workloads, don't use `--spec-type draft-mtp`. EOS suppression inside `<think>` blocks is handled automatically by the reasoning budget sampler when `--reasoning on` is active — no per-request `thinking_budget_tokens` needed.
- **Qwen3.6 hybrid arch**: only 8/32 layers use full attention with KV cache; the other 24 use GatedDeltaNet (SSM, no KV cache). This means KV cache quantization (`--cache-type-k q8_0`) has minimal impact — there's very little KV to compress. TurboQuant similarly has ~1-3% overhead only.
- **Wikipedia-only calibration data overfits perplexity benchmarks** but produces worse real-world output for instruct models. The pipeline uses multi-domain calibration (chat/code/math/tool-calling/Chinese/long-context) deliberately — don't "simplify" back to wikitext.
- **Calibration uses token budgets with complete samples** — each domain group has a token target (~750K-1M). Samples are shuffled then collected until the budget is reached. The last sample is always included whole even if it overshoots. Never truncate mid-conversation.
- **IQ quants only exist up to ~4.5 bpw** (IQ4_XS, IQ4_NL). There is no IQ5/IQ6/IQ8 — K-quants (Q5_K_M, Q6_K, Q8_0) are the only option at 5+ bits.
- **imatrix .dat format** is not self-describing — files from different llama.cpp versions may be incompatible. Regenerate imatrix if you update llama.cpp.
- **Per-tensor overrides use regex matching** — `llama-quantize --tensor-type-file` treats tensor names as regex patterns via `std::regex_search` (substring match). Dots match any character and short names match longer ones (e.g. `output.weight` matches `blk.3.attn_output.weight`). Always use anchored, escaped patterns: `^blk\.3\.attn_output\.weight$=f16`. The sensitivity script generates these automatically.
- **Per-tensor overrides** (`configs/<model>/tensor_overrides.txt`) are model-specific — current Qwen3.6 file has 65 layers (48 SSM + 17 attn), SSM alpha/beta at F32, MTP layer (blk.64) at F16. Each model gets its own overrides in its config subdir.
- **Q4_K is an alias for Q4_K_M** which has internal rules promoting certain tensor categories (attn_output → Q5_K, attn_v → Q5_K in early layers, ffn_down → Q6_K in edge layers). The sensitivity script uses Q4_0 (no internal rules) to get uniform probing across all tensor groups.
- **Switching models**: change `MODEL_DIR` in `configs/model.env`, create `configs/<model>/model.env` with model-specific settings (MODEL_ID, NATIVE_CTX, GGUF_ARCH_KEY, quant types, MTP config), then run `make sensitivity` to generate tensor overrides.
- **Downloaded models use HF cache** — `01_download_model.sh` stores weights in `~/.cache/huggingface/hub/`, not in the project. Other tools can reuse them.
- **Generated GGUFs live in `~/models/gguf/`** (override with `SUPER_QUANT_MODELS` env var). Shared across projects, not in the repo.

## Architecture decisions

- **DI-MATRIX**: separate imatrices per domain (general/code/reasoning/agentic) merged with weighted blend, rather than a single imatrix from combined data. Both are generated — benchmark to determine which is better for this model.
- **All quants are UD (Unsloth Dynamic-style)**: every quant gets per-tensor overrides + imatrix, not just the low-bit ones. The `UD-` prefix on output filenames reflects this.
- **Unsloth Dynamic-style quants** use `--tensor-type-file` flag (or `--output-tensor-type` in older builds). `04_quantize.sh` errors out if neither is supported — quality quants require per-tensor overrides.

## Benchmark results

### Baseline (2026-05-28, hand-crafted overrides, pre-regex-fix)

These results had `output.weight` regex-matching all `attn_output.weight` tensors, silently forcing them to q8_0. MTP layer (blk.64) was unprotected. ssm_alpha/beta at f32, but attn_qkv/attn_gate unprotected.

| Quant | Size | tg t/s | PPL | KL mean | KL max | KL p99.9 |
|-------|------|--------|-----|---------|--------|----------|
| F16 | 50.9G | 31.5 | 2.6022 | — | — | — |
| UD-Q8_0 | 26.6G | 52.3 | 2.5997 | 0.0073 | 12.48 | 0.97 |
| UD-Q6_K | 22.2G | 59.5 | 2.5925 | 0.0073 | 10.12 | 0.95 |
| UD-Q5_K_M | 20.2G | 64.1 | 2.6072 | 0.0167 | 14.20 | 3.59 |
| UD-IQ4_XS | 17.3G | 69.0 | 2.5702 | 0.0335 | 22.88 | 8.74 |

### Hybrid-optimal (2026-05-28, sensitivity + APEX + tuned thresholds)

332/546 tensors overridden (61%). Combines sensitivity-discovered protections (attn_qkv, attn_gate → f16) with APEX-validated assignments (ssm_alpha/beta → f32) and lets FFN middle layers use base quant. Generated by `src/generate_hybrid_overrides.py`.

| Quant | Size | tg t/s | PPL | KL mean | KL max | KL p99.9 |
|-------|------|--------|-----|---------|--------|----------|
| F16 | 50.9G | 31.8 | 2.6022 | — | — | — |
| UD-Q8_0 | 30.6G | 48.0 | 2.5951 | 0.0039 | 8.46 | 0.52 |
| UD-Q6_K | 26.4G | 52.9 | 2.5980 | 0.0069 | 10.09 | 0.78 |
| UD-Q5_K_M | 24.5G | 56.3 | 2.6029 | 0.0093 | 11.75 | 1.42 |
| UD-IQ4_XS | 21.8G | 62.0 | 2.5730 | 0.0219 | 16.13 | 3.51 |

Q6_K remains the sweet spot — 26.4G with p99.9=0.78, better than baseline Q8_0 (p99.9=0.97) at similar size. Q8_0 achieves p99.9=0.52 (best ever) but at 30.6G.

**Key improvements over baseline:**
- Q8_0: KL p99.9 0.97→0.52 (47% reduction), KL max 12.48→8.46 (32% reduction)
- Q6_K: KL p99.9 0.95→0.78 (18% reduction)
- All quants have lower KL max than baseline

**Sensitivity analysis findings (Q4_0 probe, 16 chunks):**
- `attn_qkv` (48 SSM attention tensors, KL 0.0105) — most sensitive, was completely unprotected in hand-crafted overrides
- `attn_gate` (48 tensors, KL 0.0100) — also newly discovered as sensitive
- `ssm_alpha/beta` — Q4_0 probe showed low sensitivity (KL ~0.0014) but removing f32 protection caused Q6_K regression. Kept at f32.
- FFN middle layers — letting base quant handle these (instead of forcing q8_0) restores size differentiation without meaningful KL impact

## Deployment config

Default: `make serve` → UD-Q6_K, 512k context per slot, 5 parallel slots, q8_0 KV cache, unified KV, YaRN, DFlash (or MTP fallback), vision.

- **Thinking mode uses both flags** — `--reasoning on` sets the server-level reasoning format, `--chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'` passes Qwen's recommended template variables explicitly. Both are passed in `06_serve.sh` and `07_bench_spec.sh`.
- **YaRN auto-activates** when CTX > NATIVE_CTX (set per model in `configs/<model>/model.env`). `--override-kv ${GGUF_ARCH_KEY}.context_length=int:CTX` bypasses server-context.cpp cap bug (llama.cpp #22140).
- **Unified KV** (`-kvu`) — single shared KV buffer across all slots, better memory pooling with lazy allocation.
- **q8_0 KV cache** — hybrid SSM means only 16/64 layers have KV cache, so q8_0 is cheap enough. No need for turbo quant.
- **DFlash speculative decoding** uses a separate 5-layer diffusion drafter (`z-lab/Qwen3.6-27B-DFlash`, 3.2GB). Unlike MTP, DFlash doesn't add compute to the target model's forward pass — it cross-attends to hidden states captured passively into a ring buffer. This means DFlash should NOT hurt concurrent throughput. The drafter proposes up to 16 tokens per round. Upstream benchmarks show 7+ avg accepted tokens on math/code/tool-calling content. Our fork has 3 fixes: (1) `speculative.cpp` guards against auto-enabling `draft-simple` when `dflash` is active; (2) drafter context gets `n_seq_max >= n_parallel`; (3) `server-context.cpp` defensively clears `spec_i_batch`.
- **DFlash vs MTP for serving**: DFlash is preferred for concurrent workloads (doesn't saturate GPU at 1 slot like MTP does). `make serve` auto-selects DFlash if the draft model is present, falling back to MTP. Override with `SPEC_TYPE=mtp` or `SPEC_TYPE=none`. Use `make bench-dflash` to compare baseline/MTP/DFlash across thinking and non-thinking workloads.

## Paths

- llama.cpp fork: `/home/jinyang/src/llama.cpp`
- Generated GGUFs: `~/models/gguf/` (override: `SUPER_QUANT_MODELS` env var)
- Downloaded models: `~/.cache/huggingface/hub/` (standard HF cache)
- Per-model configs: `configs/<MODEL_DIR>/` (tensor overrides, model.env)
- GPU: NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM, sm_120

## Commands

```bash
make help                                   # show all targets
make all                                    # full pipeline
make serve                                  # Q6_K, DFlash (auto), 5 slots
make serve SPEC_TYPE=mtp                    # force MTP instead of DFlash
make serve SPEC_TYPE=none                   # no speculative decoding
make bench-dflash                           # baseline vs MTP vs DFlash comparison
make download-dflash                        # download + convert DFlash draft model
```
