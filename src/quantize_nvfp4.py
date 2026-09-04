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
import os
import re
import sys
from pathlib import Path

import datasets
import torch
from compressed_tensors.quantization.quant_scheme import (
    FP8_DYNAMIC,
    NVFP4,
    QuantizationArgs,
    QuantizationScheme,
)
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
from llmcompressor.utils import load_context

NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 1024
SHARD_SIZE = "4GB"


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
    awq_mappings = [
        AWQMapping(
            smooth_layer=r"re:.*post_attention_layernorm$",
            balance_layers=[
                r"re:.*mlp\.gate_proj$",
                r"re:.*mlp\.up_proj$",
            ],
        ),
        AWQMapping(
            smooth_layer=r"re:.*mlp\.up_proj$",
            balance_layers=[
                r"re:.*mlp\.down_proj$",
            ],
        ),
    ]

    nvfp4_mlp = QuantizationScheme(
        targets=[
            r"re:.*layers\.([0-9]|[1-4][0-9]|5[0-5])"
            r"\.mlp\.(gate|up|down)_proj$"
        ],
        **NVFP4,
    )
    nvfp4_mlp.weights.observer = "imatrix_mse"
    nvfp4_mlp.weights.observer_kwargs = {"strict": False}

    fp8_sensitive = QuantizationScheme(
        targets=[
            r"re:.*self_attn\.(q|k|v|o)_proj$",
            r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
            r"re:.*layers\.(5[6-9]|6[0-3])"
            r"\.mlp\.(gate|up|down)_proj$",
        ],
        **FP8_DYNAMIC,
    )

    kv_cache_scheme = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        dynamic=False,
        symmetric=True,
    )

    return [
        AWQModifier(
            mappings=awq_mappings,
            duo_scaling=True,
            n_grid=20,
        ),
        GPTQModifier(
            config_groups={
                "nvfp4_mlp": nvfp4_mlp,
                "fp8_sensitive": fp8_sensitive,
            },
            ignore=[
                "lm_head",
                r"re:.*embed_tokens$",
                r"re:visual.*",
                r"re:model\.visual.*",
                r"re:.*mtp.*",
                r"re:.*linear_attn\.(in_proj_b|in_proj_a|conv1d|A_log|dt_bias|norm)$",
                r"re:.*layernorm.*",
                r"re:.*_norm$",
                r"re:.*\.norm\.weight$",
            ],
            kv_cache_scheme=kv_cache_scheme,
            block_size=128,
            dampening_frac=0.01,
            actorder="static",
            offload_hessians=False,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="AWQ+GPTQ NVFP4 quantization")
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
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

    print(f"\n=== Running AWQ + GPTQ ===")
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

    print(f"\n=== Saving compressed checkpoint to {args.output_dir} ===")
    model.save_pretrained(
        args.output_dir,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size=SHARD_SIZE,
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n=== Done ===")
    print(f"Checkpoint: {args.output_dir}")
    print(f"Convert to GGUF: convert_hf_to_gguf.py {args.output_dir} --fp8-as-q8 --outfile model.gguf")


if __name__ == "__main__":
    main()
