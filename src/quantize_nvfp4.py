#!/usr/bin/env python3
"""Build a recipe-driven AWQ plus GPTQ NVFP4 checkpoint."""

import argparse
import json
import os
import threading
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
from compressed_tensors.quantization.quant_scheme import (
    NVFP4,
    QuantizationArgs,
    QuantizationScheme,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from model_utils import (
    _pin_ple_lookup_tables_to_cpu,
    build_awq_modifier,
    build_calibration_dataset,
    distributed_dataset_partition,
    enable_parallel_onload,
    keep_ple_lookup_tables_on_cpu,
    load_calibration_records,
    load_quantization_model,
    load_quantization_recipe,
    load_quantization_tokenizer,
    measure_expert_coverage,
    save_quantization_processor,
    validate_awq_mapping_targets,
    validate_quantization_targets,
)


def build_quantization_scheme(group):
    if group["format"] == "nvfp4":
        scheme = QuantizationScheme(targets=group["targets"], **NVFP4)
        scheme.weights.observer = "imatrix_mse"
        scheme.weights.observer_kwargs = {"strict": False}
        return scheme
    return QuantizationScheme(
        targets=group["targets"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            symmetric=True,
            strategy="channel",
            dynamic=False,
            actorder="static",
            observer="memoryless_minmax",
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type="float",
            symmetric=True,
            strategy="token",
            dynamic=True,
        ),
    )


def build_weight_recipe(settings):
    config_groups = {
        group["name"]: build_quantization_scheme(group) for group in settings["groups"]
    }
    gptq = settings["gptq"]
    return [
        build_awq_modifier(settings),
        GPTQModifier(
            config_groups=config_groups,
            ignore=settings["ignore"],
            kv_cache_scheme=None,
            block_size=gptq["block_size"],
            dampening_frac=gptq["dampening_frac"],
            actorder=gptq["actorder"],
            offload_hessians=gptq["offload_hessians"],
        ),
    ]


def build_kv_modifier(settings):
    if settings["kv_cache"] != "fp8":
        return None
    return QuantizationModifier(
        targets=[],
        kv_cache_scheme=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=False,
            symmetric=True,
        ),
    )


def distributed_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def distributed_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def print_primary(message=""):
    if distributed_rank() == 0:
        print(message, flush=True)


def initialize_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"{torch.accelerator.current_accelerator().type}:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    backend = "nccl" if device.type == "cuda" else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(hours=2),
    )
    torch.distributed.barrier()


def synchronized_phase(run):
    distributed = distributed_world_size() > 1
    if distributed:
        torch.distributed.barrier()
    started = time.monotonic()
    run()
    if distributed:
        torch.distributed.barrier()
    elapsed = time.monotonic() - started
    if distributed:
        device = torch.device("cuda", torch.cuda.current_device())
        value = torch.tensor(elapsed, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        elapsed = value.item()
    return elapsed


class GpuTelemetry:
    def __init__(self, path, interval_seconds=10):
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = None
        self.phase = "startup"

    def set_phase(self, phase):
        self.phase = phase

    def _sample(self):
        while not self.stop_event.is_set():
            sample = {"time": time.time(), "phase": self.phase, "gpus": {}}
            for index in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(index)
                    utilization = torch.cuda.utilization(index)
                except (OSError, RuntimeError, ValueError):
                    free, total, utilization = None, None, None
                sample["gpus"][str(index)] = {
                    "free_bytes": free,
                    "total_bytes": total,
                    "utilization_percent": utilization,
                }
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(sample, sort_keys=True) + "\n")
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self):
        if distributed_rank() == 0 and torch.cuda.is_available():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_seconds + 1)


def report_device_map(model):
    from compressed_tensors.offload import get_device_map

    counts = {}
    for onload_device, offload_device in get_device_map(model).values():
        key = f"{onload_device}->{offload_device}"
        counts[key] = counts.get(key, 0) + 1
    print_primary(f"Distributed ranks: {distributed_world_size()}")
    print_primary(
        f"Model device-map module counts: {json.dumps(counts, sort_keys=True)}"
    )


def run_calibration_pass(model, tokenizer, recipe, dataset, sequence_length, runtime):
    local_dataset = distributed_dataset_partition(dataset)
    print_primary(
        f"Calibration rows: {len(dataset):,} global; "
        f"{len(local_dataset):,} per-rank slice on rank 0"
    )
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=recipe,
        dataset=local_dataset,
        data_collator="truncation",
        batch_size=runtime["batch_size"],
        max_seq_length=sequence_length,
        num_calibration_samples=len(local_dataset),
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=False,
        pipeline="sequential",
        tie_word_embeddings=False,
        dataloader_num_workers=runtime["dataloader_num_workers"],
        sequential_targets_per_subgraph=runtime["sequential_targets_per_subgraph"],
        sequential_prefetch=runtime["sequential_prefetch"],
        enable_compile=runtime["enable_compile"],
    )


def apply_probe(settings, token_budget):
    calibrations = [settings["weight_calibration"]]
    if settings["kv_calibration"] is not None:
        calibrations.append(settings["kv_calibration"])
    for calibration in calibrations:
        required_sequences = max(len(calibration["domains"]), distributed_world_size())
        minimum = calibration["sequence_length"] * required_sequences
        if token_budget < minimum:
            raise ValueError(
                f"Probe requires at least {minimum:,} tokens for "
                f"{required_sequences} sequences at sequence length "
                f"{calibration['sequence_length']:,}"
            )
        calibration["token_budget"] = token_budget
    if coverage := settings.get("expert_coverage"):
        coverage["token_budget"] = token_budget


def runtime_estimate(name, elapsed, measured_tokens, full_tokens):
    return {
        "phase": name,
        "elapsed_seconds": elapsed,
        "measured_tokens": measured_tokens,
        "full_tokens": full_tokens,
        "linear_estimate_seconds": elapsed * full_tokens / measured_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="AWQ+GPTQ NVFP4 quantization")
    parser.add_argument("--config", required=True, help="Per-model quantize.json")
    parser.add_argument(
        "--calibration-dir", required=True, help="Structured calibration directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output checkpoint directory"
    )
    parser.add_argument(
        "--model-id", help="Source-precision model path or Hugging Face model ID"
    )
    parser.add_argument(
        "--offload-dir",
        help="Local disk cache for distributed models that exceed the mmap limit",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Calibration batch size on each rank"
    )
    parser.add_argument(
        "--probe-tokens",
        type=int,
        help="Run a reduced profiling pass and skip checkpoint saving",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip phases already recorded in the output directory's "
            "phase_state.json; the weight-phase checkpoint is the output "
            "directory itself"
        ),
    )
    args = parser.parse_args()

    initialize_distributed()
    try:
        recipe = load_quantization_recipe(args.config)
        if args.model_id:
            recipe["source"]["model_id"] = args.model_id
        settings = recipe["nvfp4"]
        runtime = recipe.get(
            "runtime",
            {
                "batch_size": 1,
                "dataloader_num_workers": 0,
                "sequential_targets_per_subgraph": 1,
                "sequential_prefetch": False,
                "enable_compile": False,
            },
        )
        if args.batch_size is not None:
            if args.batch_size <= 0:
                raise ValueError("--batch-size must be positive")
            runtime["batch_size"] = args.batch_size

        full_weight_tokens = settings["weight_calibration"]["token_budget"]
        full_kv_tokens = (
            settings["kv_calibration"]["token_budget"]
            if settings["kv_calibration"] is not None
            else None
        )
        if args.probe_tokens is not None:
            if args.probe_tokens <= 0:
                raise ValueError("--probe-tokens must be positive")
            apply_probe(settings, args.probe_tokens)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = load_quantization_tokenizer(recipe)
        weight_calibration = settings["weight_calibration"]
        domains = list(weight_calibration["domains"])
        if settings["kv_calibration"] is not None:
            domains.extend(
                domain
                for domain in settings["kv_calibration"]["domains"]
                if domain not in domains
            )
        records_by_domain = load_calibration_records(args.calibration_dir, domains)
        print_primary("=== Loading and packing weight calibration data ===")
        weight_dataset, weight_report = build_calibration_dataset(
            args.calibration_dir,
            tokenizer,
            weight_calibration,
            records_by_domain=records_by_domain,
        )
        print_primary(
            f"Packed {weight_report['selected_tokens']:,} effective tokens into "
            f"{len(weight_dataset):,} weight-calibration sequences"
        )

        phase_state_path = output_dir / "phase_state.json"
        completed_phases = []
        if args.resume and phase_state_path.is_file():
            completed_phases = list(
                json.loads(phase_state_path.read_text(encoding="utf-8"))[
                    "completed_phases"
                ]
            )
            print_primary(
                "=== Resuming; already-complete phases: "
                f"{', '.join(completed_phases)} ==="
            )

        def mark_phase_complete(phase):
            if distributed_world_size() > 1:
                torch.distributed.barrier()
            if distributed_rank() == 0:
                completed_phases.append(phase)
                phase_state_path.write_text(
                    json.dumps(
                        {"completed_phases": completed_phases},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        model_recipe = recipe
        resumed_from_checkpoint = "awq_gptq" in completed_phases
        if resumed_from_checkpoint:
            model_recipe = deepcopy(recipe)
            model_recipe["source"]["model_id"] = str(output_dir)
        print_primary(f"\n=== Loading model: {model_recipe['source']['model_id']} ===")
        model = load_quantization_model(
            model_recipe,
            offload_dir=args.offload_dir,
            allow_quantized=resumed_from_checkpoint,
        )
        report_device_map(model)
        validate_awq_mapping_targets(model, settings["mappings"])
        counts = validate_quantization_targets(
            model, settings["groups"], settings["ignore"]
        )
        for group_name, count in counts.items():
            print_primary(f"Matched {group_name} modules: {count}")
        onload_workers = runtime.get("parallel_onload_workers", 0)
        if onload_workers > 0:
            wrapped = enable_parallel_onload(model, workers=onload_workers)
            print_primary(
                f"Parallel onload active on {wrapped} modules "
                f"({onload_workers} worker threads per rank)"
            )

        # Dispatch leaves caches on cpu; the sequential pipeline only fixes
        # devices per subgraph, and trace_subgraphs touches model.device
        # before any subgraph runs. Promote once here so every entry path
        # (fresh, coverage-skipped resume) starts CUDA-placed; PLE tables
        # stay pinned in host RAM.
        from compressed_tensors.offload import set_onload_device
        from llmcompressor.utils import get_main_device

        set_onload_device(model, get_main_device())
        _pin_ple_lookup_tables_to_cpu(model)

        timings = []
        telemetry_path = output_dir / "gpu_telemetry.jsonl"
        with GpuTelemetry(telemetry_path) as telemetry:
            if coverage := settings.get("expert_coverage"):
                if "expert_coverage" in completed_phases:
                    print_primary("\n=== Expert coverage already measured ===")
                else:
                    print_primary("\n=== Measuring real routed-expert coverage ===")
                    telemetry.set_phase("expert_coverage")
                    with keep_ple_lookup_tables_on_cpu(model):
                        measure_expert_coverage(
                            model,
                            args.calibration_dir,
                            tokenizer,
                            weight_calibration,
                            coverage,
                            records_by_domain=records_by_domain,
                            checkpoint_path=(
                                output_dir / "coverage_partial.json"
                            ),
                            on_domain_complete=mark_phase_complete,
                        )
                    mark_phase_complete("expert_coverage")

            if "awq_gptq" in completed_phases:
                print_primary("\n=== AWQ + GPTQ checkpoint present; skipping ===")
            else:
                print_primary("\n=== Running distributed AWQ + GPTQ calibration ===")
                telemetry.set_phase("awq_gptq")
                with keep_ple_lookup_tables_on_cpu(model):
                    weight_elapsed = synchronized_phase(
                        lambda: run_calibration_pass(
                            model,
                            tokenizer,
                            build_weight_recipe(settings),
                            weight_dataset,
                            weight_calibration["sequence_length"],
                            runtime,
                        )
                    )
                timings.append(
                    runtime_estimate(
                        "awq_gptq",
                        weight_elapsed,
                        weight_report["selected_tokens"],
                        full_weight_tokens,
                    )
                )
                print_primary(
                    f"\n=== Saving weight-phase checkpoint to {output_dir} ==="
                )
                model.save_pretrained(
                    output_dir,
                    save_compressed=True,
                    safe_serialization=True,
                    max_shard_size=recipe["save"]["shard_size"],
                )
                mark_phase_complete("awq_gptq")

            kv_modifier = build_kv_modifier(settings)
            if kv_modifier is not None and "complete" in completed_phases:
                print_primary("\n=== FP8 KV-cache scales already saved ===")
                kv_modifier = None
            if kv_modifier is not None:
                kv_calibration = settings["kv_calibration"]
                print_primary("\n=== Loading long-position KV calibration data ===")
                kv_dataset, kv_report = build_calibration_dataset(
                    args.calibration_dir,
                    tokenizer,
                    kv_calibration,
                    records_by_domain=records_by_domain,
                )
                print_primary(
                    f"Packed {kv_report['selected_tokens']:,} effective tokens into "
                    f"{len(kv_dataset):,} KV-calibration sequences"
                )
                print_primary("\n=== Calibrating static FP8 KV-cache scales ===")
                telemetry.set_phase("fp8_kv")
                with keep_ple_lookup_tables_on_cpu(model):
                    kv_elapsed = synchronized_phase(
                        lambda: run_calibration_pass(
                            model,
                            tokenizer,
                            [kv_modifier],
                            kv_dataset,
                            kv_calibration["sequence_length"],
                            runtime,
                        )
                    )
                timings.append(
                    runtime_estimate(
                        "fp8_kv",
                        kv_elapsed,
                        kv_report["selected_tokens"],
                        full_kv_tokens,
                    )
                )

        if distributed_rank() == 0:
            report = {
                "world_size": distributed_world_size(),
                "runtime": runtime,
                "probe_tokens": args.probe_tokens,
                "timings": timings,
                "estimate_warning": (
                    "Linear token scaling excludes model load, coverage, save, and upload."
                ),
            }
            (output_dir / "runtime_estimate.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)

        if args.probe_tokens is not None:
            print_primary("\n=== Probe complete; checkpoint saving skipped ===")
        elif "complete" in completed_phases:
            print_primary("\n=== Checkpoint already complete; skipping final save ===")
        else:
            print_primary(
                f"\n=== Saving compressed checkpoint to {args.output_dir} ==="
            )
            model.save_pretrained(
                args.output_dir,
                save_compressed=True,
                safe_serialization=True,
                max_shard_size=recipe["save"]["shard_size"],
            )
            if distributed_world_size() > 1:
                torch.distributed.barrier()
            if distributed_rank() == 0:
                tokenizer.save_pretrained(args.output_dir)
                save_quantization_processor(recipe, args.output_dir)
            mark_phase_complete("complete")
            if distributed_rank() == 0:
                print("\n=== Done ===")
                print(f"Checkpoint: {args.output_dir}")
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
