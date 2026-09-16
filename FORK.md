# ModelOpt fork

We run NVIDIA Model-Optimizer from our fork instead of the PyPI wheel plus
bootstrap-time `patch -p1` into site-packages, so every deviation from upstream
is a reviewed, dated commit.

- Fork: <https://github.com/a4501150/Model-Optimizer>, branch `sq/0.46`
- Base: upstream tag `0.46.0` = `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`
  (matches PyPI `nvidia-modelopt==0.46.0`, the version used by the
  2026-09-13 FSDP2 production builds)

## Commit stack on top of the base

| Commit | Change | Upstream link |
|---|---|---|
| `021ef83e` | Fix GPTQ Hessian coordinates after AWQ smoothing (cherry-pick of open PR #2401, `-x`) | [NVIDIA/Model-Optimizer#2401](https://github.com/NVIDIA/Model-Optimizer/pull/2401) |
| `b9b34f9e` | Memoize `_get_module_name` results on the module (validated via `get_submodule`); removes the quadratic `named_modules()` rebuild per GPTQ-solve / export-requantize weight update on 512-expert MoE graphs | — (candidate PR; supersedes the `_install_fast_module_names` monkeypatch in `src/quantize_mopt.py`) |
| `4bf9800e` | Cumulative `[gptq] layers solved:` progress lines with rate | — (parity with the production log parser) |
| `c794d21d` | Gated `empty_cache()` at weight-window exit (only when ≥2 GiB reclaimable): full-layer DTensor replicas linger under `expandable_segments` and the next window OOMs | — (2026-09-13 export OOM hygiene) |
| `142b8b3d` | `GPTQHelper.HESSIAN_OFFLOAD_FRACTION` class knob replacing the hardcoded 0.65 (placement decided at allocation, before the pass that grows usage); default unchanged. (Earlier `73f82b46` revision misplaced the two `@contextmanager` decorators onto the new helper and broke the writeback context; fixed + force-pushed before the `+sq3` wheel.) | — |
| `2dbe3170` | Offload each processed FSDP unit's export outputs (quantized weights + scale buffers) to host memory before resharding, plus final-unit hand-off the boundary hack never resharded | — (2026-09-14 export OOM, holders named via `_record_memory_history` snapshot: 19.6 GiB nvfp4 quantize outputs + 4.7 GiB tail gather buffers + 2.5 GiB fp8 scale temporaries) |
| `ba303c0c` | `fsdp2_aware_weight_update`: pre-bind `fsdp_param_mapping` so an OOM in setup no longer turns `finally` into `UnboundLocalError` masking the real error; `_release_cached_cuda_blocks()` now always runs in `finally` | — |
| `2b796b13` | `from_config(dtype=)` instead of deprecated `torch_dtype=` | — (closes the queued +sq4 warning item) |
| `4c645a54` | Keep export-output values on rank 0 only (meta elsewhere): all ranks requantize (unshard is collective) but only rank 0 saves; per-rank host copies duplicated the payload | — (2026-09-14 rerun: +sq4 fixed VRAM — export held flat at ~32 GiB — but the 8x host duplication + 103 GiB PLE tmpfs OOMKilled the 900 Gi container cgroup) |
| `dce79364` | Re-wrap `QTensorWrapper` for the non-rank-0 meta downgrade: `Parameter.set_data` refuses subclass→plain-meta swaps, so +sq5's non-zero ranks died at the first FSDP boundary while rank 0 spin-waited on NCCL ~2h before the barrier surfaced it | — (2026-09-15 run-4 export) |

| `7aabed14` | Non-rank-0 downgrade never raises (meta swap -> QTensorWrapper rewrap -> new_meta -> keep-CUDA with class-name print) and export except prints the real error before the finally-barrier | — (2026-09-15 run 5: first non-fused unit's qtensor-subclass params threw; ranks parked at the barrier, cause hidden until 2h timeout) |

## Wheel

Built pure-Python wheel, version pinned so it is unambiguously ours. Current
deployed build is `+sq8` (installed via `scripts/bootstrap_venv.sh`;
`+sq1`…`+sq7` are superseded — `+sq2` carried the broken decorator placement):

```
gs://ads-billing-models/wheelhouse/modelopt/sq-0.46.0+sq8/nvidia_modelopt-0.46.0+sq8-py3-none-any.whl
```

Rebuild (deterministic from the branch):

```bash
git clone -b sq/0.46 https://github.com/a4501150/Model-Optimizer && cd Model-Optimizer
SETUPTOOLS_SCM_PRETEND_VERSION=0.46.0+sq8 uv build --wheel
```

## Known cosmetic / queued fork items

- Grouped-GEMM MoE expert path (deferred 2026-09-14: block-forward surgery
  under the QuantModule wrappers, tolerance-verify needed; fused QDQ via
  compiled extensions is live and carries the launch-overhead win).
- `awq_lite` NaN pre-quant scales on dead channels (found 2026-09-14, phase-B
  NaN crash): per-input-channel scale `amax_x^a / amax_w^(1-a)` is 0/0=NaN for
  channels with zero activation or weight amax over the corpus; the NaN lands
  in the quantizer state and poisons any later fake-quant forward (measured
  385/2560 channels, layer-0 GDN in_proj_qkv). Fix upstream: guard zero amax
  -> scale 1.0 in the lite search. Driver-side workaround banked:
  `_sanitize_quantizer_state` in `src/quantize_mopt.py` neutralizes NaN pqs/amax
  after restore — keep it regardless (protects against any stale state).

## Status / migration

The 2026-09-13 worker stack is still wheel + bootstrap PR-patch + the driver
monkeypatches in `src/quantize_mopt.py`. Migration = install this wheel and
drop (1) the PR-patch block of the bootstrap and (2) the monkeypatches that
the wheel now covers (`_get_module_name` fast map; the GPTQ progress wrapper
can stay as a no-op or go). Import smoke of the wheel against the cu134 torch
venv must run **after** a live calibration exits — replacing site-packages
under a running process is unsound.

## Layerwise mode audit (2026-09-14)

ModelOpt 0.46 `LayerwiseConfig` (recipe per-method `layerwise: {enable: true}`,
alias `use_sequential`) was trialed via the driver's `--layerwise` flag on the
FSDP2 worker. Two findings:

1. Per-layer **checkpointing** (`checkpoint_dir`) hard-refuses multi-process
   jobs (`layerwise_calib._CheckpointState`); the sweep itself is supported.
   We run without it; `--stage-dir` + `mtq_state.pt` remain our resume story.
2. The sweep died at layer 2/48: the per-layer replay re-enters the embedding
   input stack and raises `aten.embedding.default got mixed torch.Tensor and
   DTensor`. The mode assumes stock HF causal-LM input plumbing; our PLE
   n-gram bridge + VLM wrapper violate that. Not a quick fix — layerwise is
   off the critical path until upstream (or a fork patch) handles
   model-specific input stacks.

Export-phase memory (unrelated to layerwise): `_process_quantized_modules`
already reshards at FSDP-module boundaries, yet allocated still grew
~1.6 GiB/layer to OOM at ~69 GiB (first observation 2026-09-14 prod run:
`SQQ_MEM_HISTORY=1` watchdog snapshot named the holders exactly — the full
quantized tensors `_export_quantized_weight` registers on each processed
module inside the unshard window, plus the never-resharded tail layer's
gather buffers). Fixed in `2dbe3170`: processed units hand their export
outputs to host memory at each boundary (host has 1.7 TiB free; nothing
reads them on-device again). First observation of the fix is the ongoing
prod re-run.

## Split-phase calibration + fragmentation lesson (2026-09-14)

The first full-budget production run died inside `awq_clip`'s caching
traversal (not export) with a pure fragmentation fingerprint: 16.9 GiB
reserved-but-unallocated stranded next to a 15.16 GiB allocation failing
against 14.48 GiB contiguous-free. Root cause was NOT ModelOpt: the launcher
passed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as a command
assignment prefix, and a later edit inserted a comment block between the
`\`-continuation and `torchrun` — bash degraded the prefix to non-exported
shell variables, so the ranks ran without the flag. Fixed in `71ca10f`
(real `export`s; keep launcher env as exports, never as an assignment
prefix, so no comment can silently swallow it).

Same commit adds phase checkpointing (`--calib-phase awq|gptq` + the
launcher's `CHAIN_GPTQ=1`): quantize() has no in-call resume, so the AWQ
leg runs as process 1 and banks mtq_state + staged weights before any
export risk; the GPTQ leg + export run as process 2 with a fresh allocator
via the (previously unexercised) resume-over-staged-weights path. If GPTQ
or export fails, AWQ — the slow stage — is banked. The BF16 janitor is now
opt-in (`KEEP_BF16=0`): on the H100 node RAID it only buys a 331 GiB
re-pull on retries. The export reference-leak analysis above stays open —
first observation deferred until a calibration completes.

Boundary rule for future fixes: anything that changes behavior *inside*
modelopt lands here as a commit; anything that orchestrates modelopt for this
checkpoint (PLE sharded-mmap shim, bucketed forward loop, dataset builder,
loader, GPU-util guard) stays in this repo. When upstream merges #2401 and we
rebase onto a tag that contains it, the cherry-pick commit drops out.
