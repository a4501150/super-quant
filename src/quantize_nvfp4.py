#!/usr/bin/env python3
"""Build a recipe-driven AWQ plus GPTQ NVFP4 checkpoint."""

import argparse
from pathlib import Path

from compressed_tensors.quantization.quant_scheme import (
    NVFP4,
    QuantizationArgs,
    QuantizationScheme,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from model_utils import (
    build_awq_modifier,
    build_calibration_dataset,
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


def run_calibration_pass(model, tokenizer, recipe, dataset, sequence_length):
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=recipe,
        dataset=dataset,
        data_collator="truncation",
        batch_size=1,
        max_seq_length=sequence_length,
        num_calibration_samples=len(dataset),
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=False,
        pipeline="independent",
    )


def main():
    parser = argparse.ArgumentParser(description="AWQ+GPTQ NVFP4 quantization")
    parser.add_argument("--config", required=True, help="Per-model quantize.json")
    parser.add_argument(
        "--calibration-dir", required=True, help="Structured calibration directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output checkpoint directory"
    )
    args = parser.parse_args()

    recipe = load_quantization_recipe(args.config)
    settings = recipe["nvfp4"]
    weight_calibration = settings["weight_calibration"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_quantization_tokenizer(recipe)
    domains = list(weight_calibration["domains"])
    if settings["kv_calibration"] is not None:
        domains.extend(
            domain
            for domain in settings["kv_calibration"]["domains"]
            if domain not in domains
        )
    records_by_domain = load_calibration_records(args.calibration_dir, domains)
    print("=== Loading and packing weight calibration data ===")
    weight_dataset, weight_report = build_calibration_dataset(
        args.calibration_dir,
        tokenizer,
        weight_calibration,
        records_by_domain=records_by_domain,
    )
    print(
        f"Packed {weight_report['selected_tokens']:,} effective tokens into "
        f"{len(weight_dataset):,} weight-calibration sequences"
    )

    print(f"\n=== Loading model: {recipe['source']['model_id']} ===")
    model = load_quantization_model(recipe)
    validate_awq_mapping_targets(model, settings["mappings"])
    counts = validate_quantization_targets(
        model, settings["groups"], settings["ignore"]
    )
    for group_name, count in counts.items():
        print(f"Matched {group_name} modules: {count}")

    if coverage := settings.get("expert_coverage"):
        print("\n=== Measuring real routed-expert coverage ===")
        measure_expert_coverage(
            model,
            args.calibration_dir,
            tokenizer,
            weight_calibration,
            coverage,
            records_by_domain=records_by_domain,
        )

    print("\n=== Running AWQ + GPTQ weight calibration ===")
    run_calibration_pass(
        model,
        tokenizer,
        build_weight_recipe(settings),
        weight_dataset,
        weight_calibration["sequence_length"],
    )

    kv_modifier = build_kv_modifier(settings)
    if kv_modifier is not None:
        kv_calibration = settings["kv_calibration"]
        print("\n=== Loading long-position KV calibration data ===")
        kv_dataset, kv_report = build_calibration_dataset(
            args.calibration_dir,
            tokenizer,
            kv_calibration,
            records_by_domain=records_by_domain,
        )
        print(
            f"Packed {kv_report['selected_tokens']:,} effective tokens into "
            f"{len(kv_dataset):,} KV-calibration sequences"
        )
        print("\n=== Calibrating static FP8 KV-cache scales ===")
        run_calibration_pass(
            model,
            tokenizer,
            [kv_modifier],
            kv_dataset,
            kv_calibration["sequence_length"],
        )

    print(f"\n=== Saving compressed checkpoint to {args.output_dir} ===")
    model.save_pretrained(
        args.output_dir,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size=recipe["save"]["shard_size"],
    )
    tokenizer.save_pretrained(args.output_dir)
    save_quantization_processor(recipe, args.output_dir)

    print("\n=== Done ===")
    print(f"Checkpoint: {args.output_dir}")
    print(
        f"Convert to GGUF: convert_hf_to_gguf.py {args.output_dir} "
        "--fp8-as-q8 --outfile model.gguf"
    )


if __name__ == "__main__":
    main()
