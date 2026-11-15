# super-quant

`super-quant` is a config-driven quantization and serving pipeline for hybrid GatedDeltaNet/attention language models. The checked-in deployment profile is tuned for an NVIDIA RTX PRO 6000 Blackwell GPU, but model selection and model-specific behavior come from `configs/<model>/model.env`.

The repository has two main roles:

- Build reproducible calibration corpora and AWQ+GPTQ NVFP4 checkpoints with llm-compressor.
- Serve every model with Infernix (the engine in `~/src/ninferpro`) and benchmark the serving stack.

## Current configurations

| Configuration | Current use |
| --- | --- |
| `Qwen3.8-Flash-Next` | Default production serving with Infernix, NVFP4 |
| `Qwen3.8-27B-AEON` | 27B model with an NVFP4 quantization recipe |
| `Qwen3.6-27B-AEON` | 27B model with an NVFP4 quantization recipe |

These are implementations of the same model-config interface, not hardcoded pipeline targets. `configs/model.env` selects `Qwen3.8-Flash-Next` by default, so `make serve-infernix` serves that model. Override the selection for one command with `MODEL_DIR=<name>`.

## Requirements

- Linux on an SM120 Blackwell GPU.
- CUDA 13.4 at `/usr/local/cuda-13.4`.
- Python 3.12 or newer, managed by `uv`.
- The Infernix engine built at `~/src/ninferpro` (see `~/src/ninferpro/README.md` for engine details).

## Setup

Install the project dependencies and validate the shared CUDA toolkit:

```bash
make setup
```

`make setup` pins CUDA 13.4 in `~/.bashrc` and runs `uv sync`. Quantization additionally requires `uv sync --extra quantize`.

## Production serving

Production serving is Infernix (the Infernix engine from `~/src/ninferpro`), started in the background from the active model config:

```bash
make serve-infernix
make stop-infernix
```

Serving behavior — engine binary, artifact, draft model, port, KV dtype, context, concurrency, and expert placement — is defined by the `INFERNIX_*` variables in `configs/<model>/model.env`. See `~/src/ninferpro/README.md` for engine internals. `make infernix-placement`, `make infernix-gaming`, and `make infernix-full` re-place experts between restarts.

## Calibration corpus

All stages use the active `MODEL_DIR`. Select any model configuration that defines the fields required by the stages you run. For example:

```bash
export MODEL_DIR=Qwen3.8-27B-AEON
```

JSONL is the canonical calibration format. It retains complete conversations, message roles, tool definitions, source revisions, and stable record IDs. Calibration has two stages:

1. `calibration/corpus-v1/` is a shared tokenizer-neutral parent corpus. Its four domain JSONL files preserve complete conversations and unsplit documents from the pinned source revisions. Its manifest contains source and extraction policy, counts, and artifact hashes, but no model, tokenizer, token count, or rendered text.
2. `calibration/<MODEL_DIR>/` is a model-scoped build selected from that parent with the target tokenizer. Its `.txt` files are deterministic text renders, and its manifest records the parent digest, tokenizer revision, token budgets, deduplication, and artifact hashes.

Build the parent once, then derive any configured model build from it:

```bash
make calibrate-corpus
MODEL_DIR=<model> make calibrate
```

`make recalibrate` only rebuilds the model-scoped selection. `make recalibrate-corpus` explicitly reloads the pinned source datasets and replaces the shared parent. Package a validated parent reproducibly with `make package-calibration-corpus`; override `CORPUS_ARCHIVE` to select another output path. Interrupted parent extraction uses `calibration/.corpus-source-cache/`, while a valid parent lets all model builds run without the original Hugging Face datasets.

Each model must generate or download its own model-scoped calibration build. Do not copy a completed calibration directory, text render, holdout, or manifest to another model. The shared parent corpus is the reusable source; model-specific selection and rendering artifacts are not reusable.

A verified Qwen3.8-Flash-Next calibration build is available as a release asset. Extract it from the repository root before running quantization on another host:

```bash
curl -L https://github.com/a4501150/super-quant/releases/download/calibration-qwen3.8-flash-next-v1/Qwen3.8-Flash-Next-calibration-v1.tar.gz -o Qwen3.8-Flash-Next-calibration-v1.tar.gz
tar -xzf Qwen3.8-Flash-Next-calibration-v1.tar.gz -C calibration
```

Its SHA-256 is `9cc7338e5a8495a87b71eddb5449cd0cf8d880fe5d11daceac01577e3f29c698`.

The shared tokenizer-neutral corpus is available as a separate release asset. It contains 52,913 complete records across the four domains and extracts to `calibration/corpus-v1/`:

```bash
curl -L https://github.com/a4501150/super-quant/releases/download/calibration-corpus-v1/calibration-corpus-v1.tar.gz -o calibration-corpus-v1.tar.gz
tar -xzf calibration-corpus-v1.tar.gz -C calibration
```

Its SHA-256 is `c905a8725d19bdfcd1c57e9f89ffcb129b642a82adde77334b862ad791a93991`.

## Native NVFP4 quantization

Create an AWQ plus GPTQ NVFP4 checkpoint for any model with a quantization recipe on a sufficiently large host:

```bash
uv sync --extra quantize
MODEL_DIR=<model> make calibrate
MODEL_DIR=<model> make quantize-nvfp4
```

The recipe lives in `configs/<model>/quantize.json` and selects the source-precision checkpoint, calibration policy, AWQ channel-scaling targets, NVFP4 groups, and exclusions. The checked-in native recipes use approximately 1M effective weight-calibration tokens for the dense 27B models and 2M for Flash-Next. Static FP8 KV-cache scales use a separate 262K-token pass with explicit position offsets through the 262,144-token context range. Flash-Next also runs a real router preflight, writes `expert_coverage.json`, and fails if any routed expert does not meet its coverage policy.

Do not use an already quantized checkpoint as the source. The local workstation cannot quantize the 176B Qwen3.8-Flash-Next checkpoint with AWQ or GPTQ because the job exceeds its practical VRAM and system-memory capacity. This is a host limit, not a pipeline or model-config restriction; larger machines can run the same target with an appropriate configuration.

Checkpoints are written to `~/models/nvfp4/<MODEL_NAME>-NVFP4`.

## Benchmarks and tests

```bash
make test  # unit tests
```

Benchmark JSON and Markdown files are stored in `results/`.

### Infernix benches

`scripts/bench_infernix.py` is the T3 promotion bench (`make t3` in ninferpro): one server boot, placements switched at runtime, speeds parsed from the server's own `req#N done` log lines. The speculative drafter is not hard-coded: acceptance parsing reads the drafter name from the log line, and the `--spec` boot flag defaults to `INFERNIX_SPEC` from the model config (`dflash2` today).

`scripts/bench_infernix_frontier.py` measures the serving frontier: one `infernix-serve` boot, then a sweep of client concurrency levels (`--concurrency 1,2,4,6,8` by default), each running a realistic agent-traffic mix for a fixed number of seconds (`--seconds`, default 300). By default the traffic is ninferpro's `bench/agentic_ab` workload - the deterministic plan shaped from ~3,300 production requests (long resumed sessions, tool loops, compaction, subagent fan-outs, retries and aborts), replayed closed-loop with the model's own output fed back and capped at the level's in-flight limit; `--traffic synthetic` runs one long-context session beside N-1 short ones instead. Each level reports total output tok/s, per-user decode tok/s (median/p10), TTFT p50/p90, speculative acceptance and prefix-cache hit rate, all read from the server's `--request-log-jsonl` machine log (never client timing), plus the Pareto frontier over (median per-user tok/s, total tok/s) and the best point meeting `--sla-tok-s` (default 30 tok/s per user, median). Results go to `results/infernix_frontier_<ninferpro-commit>.{json,md}`; rows are written incrementally per level.

```bash
python3 scripts/bench_infernix_frontier.py \
    --binary ~/src/ninferpro/build/apps/infernix-serve --port 8411
# quick smoke of the pipeline (short budget, shrunk plan, capped turns):
python3 scripts/bench_infernix_frontier.py --binary ... --port 8411 \
    --traffic agentic --concurrency 1,4 --seconds 60 --scale 0.3 --max-tokens 512
```

The script holds the shared gpu lock for the whole life of its server (from before the boot until it has killed it by PID and the process has exited), so other agents' GPU runs cannot collide with the live model; keep runs short. It always kills its server by PID on exit. Unit tests: `python3 -m unittest tests.test_bench_infernix_frontier`.

## Model configuration

Every Make target resolves the active model through `configs/model.env` and then loads `configs/<model>/model.env`. A model file defines only the features that model uses:

- Common identity: `MODEL_ID`, `MODEL_NAME`, and `NATIVE_CTX`.
- Native quantization: checkpoint and output paths used by AWQ, GPTQ, or NVFP4. Structured recipe policy lives in the optional `configs/<model>/quantize.json` file.
- Serving: model checkpoint, alias, context, cache, and optional speculative-decoding settings.

A model does not need variables or a recipe for unsupported stages. If a selected target needs missing model policy, add it to that model's config directory instead of adding model-name conditions to the Makefile or scripts.

Machine paths and shared build settings live in `configs/model.env`.

Generated calibration data is stored in `calibration/<MODEL_DIR>/` and ignored by Git. Generated models are outside the repository:

- NVFP4: `~/models/nvfp4`
- Hugging Face downloads: `~/.cache/huggingface/hub`

Run `make help` for the main target list.

## License

Apache-2.0
