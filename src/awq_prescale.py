#!/usr/bin/env python3
"""AWQ-only channel pre-scaling for GGUF quantization.

Applies AWQ per-channel scaling to redistribute weight magnitudes, then
saves as plain BF16.  The output is an ordinary HF checkpoint (no
compressed-tensors metadata) that feeds into the standard GGUF pipeline:
convert -> imatrix -> sensitivity -> llama-quantize.

AWQ scale search needs a temporary fake quantizer (W4A16_ASYM) to evaluate
candidate scales.  All quantization state is stripped before saving.
"""

import argparse
from pathlib import Path

import torch
from compressed_tensors.quantization.quant_metadata import QuantizationMetadata
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

from model_utils import (
    build_awq_modifier,
    build_calibration_dataset,
    load_quantization_model,
    load_quantization_recipe,
    load_quantization_tokenizer,
    validate_awq_mapping_targets,
    validate_quantization_targets,
)


def build_recipe(settings):
    return [
        build_awq_modifier(settings),
        QuantizationModifier(
            scheme=settings["temporary_scheme"],
            targets=settings["targets"],
            ignore=settings["ignore"],
        ),
    ]


def strip_quantization(model):
    """Remove all quantization state so the model saves as plain BF16."""
    model.apply(QuantizationMetadata.clear_quantization)

    for module in model.modules():
        for attr in ("quantization_status", "quantization_enabled"):
            if hasattr(module, attr):
                delattr(module, attr)

    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "quantization_config"):
            delattr(cfg, "quantization_config")


def verify_clean(model):
    """Verify that no quantization artifacts remain."""
    issues = []

    for name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != torch.bfloat16:
            issues.append(f"  {name}: dtype={param.dtype}")

    for name in list(model.state_dict().keys()):
        for bad in (
            "weight_scale",
            "weight_zero_point",
            "input_scale",
            "output_scale",
            "weight_global_scale",
        ):
            if bad in name:
                issues.append(f"  unexpected param: {name}")

    for module in model.modules():
        if hasattr(module, "quantization_scheme"):
            issues.append(f"  {type(module).__name__} has quantization_scheme")

    if issues:
        print("WARNING: quantization artifacts remain:")
        for i in issues:
            print(i)
        return False

    print("Verification passed: no quantization artifacts found.")
    return True


def main():
    parser = argparse.ArgumentParser(description="AWQ pre-scaling for GGUF pipeline")
    parser.add_argument("--config", required=True, help="Per-model quantize.json")
    parser.add_argument(
        "--calibration-dir", required=True, help="Structured calibration directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output checkpoint directory"
    )
    args = parser.parse_args()

    recipe = load_quantization_recipe(args.config)
    settings = recipe["awq_prescale"]
    calibration = settings["calibration"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_quantization_tokenizer(recipe)
    print("=== Loading and packing structured calibration data ===")
    dataset, calibration_report = build_calibration_dataset(
        args.calibration_dir, tokenizer, calibration
    )
    print(
        f"Packed {calibration_report['selected_tokens']:,} effective tokens into "
        f"{len(dataset):,} sequences"
    )

    print(f"\n=== Loading model: {recipe['source']['model_id']} ===")
    model = load_quantization_model(recipe)
    validate_awq_mapping_targets(model, settings["mappings"])
    counts = validate_quantization_targets(
        model,
        [{"name": "awq_prescale", "targets": settings["targets"]}],
        settings["ignore"],
    )
    print(f"Matched AWQ target modules: {counts['awq_prescale']}")

    print(
        f"\n=== Running AWQ scale search "
        f"({settings['temporary_scheme']} temporary quantizer) ==="
    )
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=build_recipe(settings),
        dataset=dataset,
        data_collator="truncation",
        batch_size=1,
        max_seq_length=calibration["sequence_length"],
        num_calibration_samples=len(dataset),
        shuffle_calibration_samples=False,
        pipeline="independent",
    )

    print("\n=== Stripping quantization state ===")
    strip_quantization(model)
    if not verify_clean(model):
        raise RuntimeError("AWQ checkpoint still contains quantization artifacts")

    print(f"\n=== Saving AWQ-scaled BF16 checkpoint to {args.output_dir} ===")
    model.save_pretrained(
        args.output_dir,
        save_compressed=False,
        safe_serialization=True,
        max_shard_size=recipe["save"]["shard_size"],
    )
    tokenizer.save_pretrained(args.output_dir)

    print("\n=== Done ===")
    print(f"AWQ-scaled BF16 checkpoint: {args.output_dir}")
    print("Next: convert to F16 GGUF, regenerate imatrix, then quantize")


if __name__ == "__main__":
    main()
