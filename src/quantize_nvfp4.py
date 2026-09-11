#!/usr/bin/env python3
"""AWQ+GPTQ NVFP4 mixed-precision quantization using llm-compressor.

Combines:
- AWQ pre-scaling (zero inference cost) on MLP inputs
- GPTQ optimal rounding with imatrix_mse observer
- NVFP4 W4A4 for MLP layers 0-55
- FP8 dynamic W8A8 for attention, GDN linear attention, MLP layers 56-63
- Static FP8 KV cache calibration for the 16 attention layers
- Our multi-domain calibration data (general/code/reasoning/agentic)

Output: compressed-tensors HF checkpoint consumable by vLLM/SGLang and
convertible to GGUF via convert_hf_to_gguf.py --fp8-as-q8.
"""

import argparse
from pathlib import Path

from compressed_tensors.quantization.quant_scheme import (
    NVFP4,
    QuantizationArgs,
    QuantizationScheme,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

from model_utils import (
    build_awq_modifier,
    build_calibration_dataset,
    load_calibration_texts,
    load_quantization_model,
    load_quantization_recipe,
    load_quantization_tokenizer,
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


def build_recipe(settings):
    config_groups = {
        group["name"]: build_quantization_scheme(group) for group in settings["groups"]
    }
    kv_cache_scheme = None
    if settings["kv_cache"] == "fp8":
        kv_cache_scheme = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=False,
            symmetric=True,
        )
    gptq = settings["gptq"]
    return [
        build_awq_modifier(settings),
        GPTQModifier(
            config_groups=config_groups,
            ignore=settings["ignore"],
            kv_cache_scheme=kv_cache_scheme,
            block_size=gptq["block_size"],
            dampening_frac=gptq["dampening_frac"],
            actorder=gptq["actorder"],
            offload_hessians=gptq["offload_hessians"],
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="AWQ+GPTQ NVFP4 quantization")
    parser.add_argument("--config", required=True, help="Per-model quantize.json")
    parser.add_argument(
        "--calibration-dir", required=True, help="Calibration text directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output checkpoint directory"
    )
    args = parser.parse_args()

    recipe = load_quantization_recipe(args.config)
    settings = recipe["nvfp4"]
    calibration = settings["calibration"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_quantization_tokenizer(recipe)
    print("=== Loading calibration data ===")
    samples = load_calibration_texts(args.calibration_dir, calibration["domains"])
    print(f"Total samples: {len(samples)}")

    print(
        f"\n=== Tokenizing {calibration['num_samples']} samples "
        f"(max {calibration['max_length']} tokens) ==="
    )
    dataset = build_calibration_dataset(
        samples,
        tokenizer,
        calibration["num_samples"],
        calibration["max_length"],
    )
    print(f"Tokenized: {len(dataset)} samples")

    print(f"\n=== Loading model: {recipe['source']['model_id']} ===")
    model = load_quantization_model(recipe)
    validate_awq_mapping_targets(model, settings["mappings"])
    counts = validate_quantization_targets(
        model, settings["groups"], settings["ignore"]
    )
    for group_name, count in counts.items():
        print(f"Matched {group_name} modules: {count}")

    print("\n=== Running AWQ + GPTQ ===")
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=build_recipe(settings),
        dataset=dataset,
        data_collator="truncation",
        batch_size=1,
        max_seq_length=calibration["max_length"],
        num_calibration_samples=len(dataset),
        shuffle_calibration_samples=False,
        pipeline="independent",
    )

    print(f"\n=== Saving compressed checkpoint to {args.output_dir} ===")
    model.save_pretrained(
        args.output_dir,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size=recipe["save"]["shard_size"],
    )
    tokenizer.save_pretrained(args.output_dir)

    print("\n=== Done ===")
    print(f"Checkpoint: {args.output_dir}")
    print(
        f"Convert to GGUF: convert_hf_to_gguf.py {args.output_dir} "
        "--fp8-as-q8 --outfile model.gguf"
    )


if __name__ == "__main__":
    main()
