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
import os
import re
import sys
from pathlib import Path

import datasets
import torch
from compressed_tensors.quantization.quant_metadata import QuantizationMetadata
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 1024
SHARD_SIZE = "4GB"

LINEAR_TARGETS = [
    r"re:.*layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
    r"re:.*layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*layers\.\d+\.mlp\.(gate|up|down)_proj$",
]

IGNORE_LIST = [
    "lm_head",
    r"re:.*embed_tokens$",
    r"re:visual.*",
    r"re:model\.visual.*",
    r"re:.*mtp.*",
    r"re:.*linear_attn\.(in_proj_b|in_proj_a|conv1d|A_log|dt_bias|norm)$",
    r"re:.*layernorm.*",
    r"re:.*_norm$",
    r"re:.*\.norm\.weight$",
]


def load_calibration_texts(calibration_dir: str, domains: list[str]) -> list[str]:
    samples = []
    for domain in domains:
        path = Path(calibration_dir) / f"{domain}.txt"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping domain '{domain}'")
            continue
        raw = path.read_text(encoding="utf-8")
        chunks = re.split(r"\n{2,}", raw)
        domain_samples = [c.strip() for c in chunks if len(c.strip()) > 100]
        print(f"  {domain}: {len(domain_samples)} samples from {path}")
        samples.extend(domain_samples)
    return samples


def build_dataset(samples: list[str], tokenizer, num_samples: int, max_length: int):
    import random
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:num_samples]

    tokenized = []
    for text in samples:
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        tokenized.append({k: v.squeeze(0).tolist() for k, v in inputs.items()})

    return datasets.Dataset.from_dict(
        {k: [item[k] for item in tokenized] for k in tokenized[0]}
    )


def data_collator(batch):
    assert len(batch) == 1
    return {k: torch.as_tensor(v).unsqueeze(0) for k, v in batch[0].items()}


def build_recipe():
    return [
        AWQModifier(
            mappings=None,
            offload_device=torch.device("cpu"),
            duo_scaling=True,
            n_grid=20,
        ),
        QuantizationModifier(
            scheme="W4A16_ASYM",
            targets=LINEAR_TARGETS,
            ignore=IGNORE_LIST,
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
        for bad in ("weight_scale", "weight_zero_point", "input_scale",
                     "output_scale", "weight_global_scale"):
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
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--calibration-dir", required=True, help="Path to calibration text files")
    parser.add_argument("--output-dir", required=True, help="Output checkpoint directory")
    parser.add_argument("--domains", nargs="+", default=["general", "code", "reasoning", "agentic"])
    parser.add_argument("--num-samples", type=int, default=NUM_CALIBRATION_SAMPLES)
    parser.add_argument("--max-length", type=int, default=MAX_SEQUENCE_LENGTH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Loading model: {args.model_id} ===")
    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"\n=== Loading calibration data ===")
    samples = load_calibration_texts(args.calibration_dir, args.domains)
    print(f"Total samples: {len(samples)}")

    print(f"\n=== Tokenizing {args.num_samples} samples (max {args.max_length} tokens) ===")
    dataset = build_dataset(samples, tokenizer, args.num_samples, args.max_length)
    print(f"Tokenized: {len(dataset)} samples")

    print(f"\n=== Running AWQ scale search (W4A16_ASYM temporary quantizer) ===")
    oneshot(
        model=model,
        processor=tokenizer,
        recipe=build_recipe(),
        dataset=dataset,
        data_collator=data_collator,
        batch_size=1,
        max_seq_length=args.max_length,
        num_calibration_samples=len(dataset),
        shuffle_calibration_samples=False,
        pipeline="independent",
    )

    print(f"\n=== Stripping quantization state ===")
    strip_quantization(model)
    verify_clean(model)

    print(f"\n=== Saving AWQ-scaled BF16 checkpoint to {args.output_dir} ===")
    model.save_pretrained(
        args.output_dir,
        save_compressed=False,
        safe_serialization=True,
        max_shard_size=SHARD_SIZE,
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n=== Done ===")
    print(f"AWQ-scaled BF16 checkpoint: {args.output_dir}")
    print(f"Next: convert to F16 GGUF, regenerate imatrix, then quantize")


if __name__ == "__main__":
    main()
