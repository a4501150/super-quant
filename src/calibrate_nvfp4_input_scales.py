#!/usr/bin/env python3
"""Calibrate input_global_scale for W4A16 NVFP4 MoE checkpoints.

Runs calibration samples through the model, measures per-expert activation
amax at each MoE projection (gate/up/down), then saves the scales into the
checkpoint as input_global_scale tensors and updates config.json to declare
input_activations for the NVFP4 group.

Usage:
    uv run src/calibrate_nvfp4_input_scales.py \
        --checkpoint ~/.cache/huggingface/hub/models--orcarouter--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/<hash> \
        --num-samples 64 \
        --seq-len 512
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def find_moe_layers(model) -> dict[str, torch.nn.Module]:
    """Find all MoE expert projection layers and return {name: module}."""
    projections = {}
    for name, module in model.named_modules():
        if re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)$", name):
            projections[name] = module
    return projections


def collect_amax(
    model,
    tokenizer,
    num_samples: int,
    seq_len: int,
    calibration_texts: list[str],
) -> dict[str, float]:
    """Run calibration and collect per-projection activation amax values."""
    projections = find_moe_layers(model)
    logger.info("Found %d MoE expert projections", len(projections))

    amax_dict: dict[str, float] = defaultdict(float)
    hooks = []

    def make_hook(proj_name):
        def hook_fn(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            val = x.abs().max().item()
            if val > amax_dict[proj_name]:
                amax_dict[proj_name] = val
        return hook_fn

    for name, module in projections.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    samples_run = 0
    with torch.no_grad():
        for text in calibration_texts:
            if samples_run >= num_samples:
                break
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
            )
            input_ids = tokens["input_ids"].to(model.device)
            if input_ids.shape[1] < 4:
                continue
            model(input_ids)
            samples_run += 1
            if samples_run % 8 == 0:
                logger.info("  %d/%d samples", samples_run, num_samples)

    for h in hooks:
        h.remove()

    logger.info("Ran %d calibration samples", samples_run)
    return dict(amax_dict)


def compute_input_global_scales(
    amax_dict: dict[str, float],
    num_layers: int,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    """Compute input_global_scale from amax values.

    For experts that were not activated during calibration, use the
    per-layer mean of activated experts.
    """
    proj_types = ["gate_proj", "up_proj", "down_proj"]
    scales = {}

    for layer_idx in range(num_layers):
        for proj in proj_types:
            layer_vals = []
            for expert_idx in range(num_experts):
                key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}"
                if key in amax_dict and amax_dict[key] > 0:
                    layer_vals.append(amax_dict[key])

            if layer_vals:
                fallback = sum(layer_vals) / len(layer_vals)
            else:
                fallback = 1.0
                logger.warning(
                    "Layer %d %s: no experts activated, using scale=1.0",
                    layer_idx, proj,
                )

            for expert_idx in range(num_experts):
                key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}"
                amax = amax_dict.get(key, 0.0)
                if amax <= 0:
                    amax = fallback
                scale = amax / 448.0
                scale_key = f"{key}.input_global_scale"
                scales[scale_key] = torch.tensor([scale], dtype=torch.float32)

    return scales


def save_scales_to_checkpoint(
    scales: dict[str, torch.Tensor],
    checkpoint: Path,
):
    """Save input_global_scale tensors as a new safetensors shard."""
    shard_name = "model-input-scales.safetensors"
    shard_path = checkpoint / shard_name
    save_file(scales, str(shard_path))
    logger.info("Saved %d scale tensors to %s", len(scales), shard_path)

    idx_path = checkpoint / "model.safetensors.index.json"
    with open(idx_path) as f:
        index = json.load(f)

    for key in scales:
        index["weight_map"][key] = shard_name

    total_bytes = sum(t.numel() * t.element_size() for t in scales.values())
    if "total_size" in index.get("metadata", {}):
        index["metadata"]["total_size"] = str(
            int(index["metadata"]["total_size"]) + total_bytes
        )

    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=False)
    logger.info("Updated %s with %d new keys", idx_path, len(scales))


def update_config(checkpoint: Path):
    """Add input_activations config to the NVFP4 group in config.json."""
    cfg_path = checkpoint / "config.json"
    with open(cfg_path) as f:
        config = json.load(f)

    qc = config.get("quantization_config", {})
    for group_name, group in qc.get("config_groups", {}).items():
        fmt = group.get("format", "")
        if "nvfp4" in fmt and group.get("input_activations") is None:
            group["input_activations"] = {
                "actorder": None,
                "block_structure": None,
                "dynamic": "local",
                "group_size": 16,
                "num_bits": 4,
                "observer": "static_minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.float8_e4m3fn",
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
            }
            logger.info("Added input_activations to %s", group_name)

    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Updated %s", cfg_path)


def get_calibration_texts(num_samples: int) -> list[str]:
    """Generate diverse calibration prompts."""
    prompts = [
        "Explain the theory of general relativity in simple terms.",
        "Write a Python function to find the longest common subsequence.",
        "What are the main differences between TCP and UDP protocols?",
        "Solve: If f(x) = x^3 - 6x^2 + 11x - 6, find all roots.",
        "Describe the process of photosynthesis step by step.",
        "Implement a red-black tree insertion algorithm in C++.",
        "What caused the fall of the Roman Empire?",
        "Prove that the square root of 2 is irrational.",
        "Write a REST API endpoint for user authentication in Node.js.",
        "Explain quantum entanglement and its implications for computing.",
        "How does the human immune system respond to a viral infection?",
        "Write a SQL query to find the top 10 customers by revenue.",
        "What are the key principles of machine learning?",
        "Explain the difference between supervised and unsupervised learning.",
        "Write a recursive solution for the Tower of Hanoi problem.",
        "Describe the water cycle and its importance for ecosystems.",
        "Implement a basic neural network from scratch in Python.",
        "What are the main challenges in natural language processing?",
        "Explain how blockchain consensus mechanisms work.",
        "Write a shell script that monitors disk usage and sends alerts.",
        "Describe the process of protein folding and why it matters.",
        "Implement a concurrent hash map in Rust.",
        "What are the principles of thermodynamics?",
        "Write a MapReduce algorithm for word frequency counting.",
        "Explain the P vs NP problem in computer science.",
        "How does CRISPR gene editing technology work?",
        "Write a parser for mathematical expressions using recursive descent.",
        "Describe the architecture of a modern CPU pipeline.",
        "Implement a B-tree with split and merge operations.",
        "Explain the CAP theorem and its implications for distributed systems.",
        "What are the main types of chemical bonds?",
        "Write a Dijkstra's shortest path algorithm implementation.",
    ]
    while len(prompts) < num_samples:
        prompts.extend(prompts)
    return prompts[:num_samples]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Calibrate input_global_scale for W4A16 NVFP4 MoE"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    cfg_path = args.checkpoint / "config.json"
    with open(cfg_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)
    num_layers = tc["num_hidden_layers"]
    num_experts = tc["num_experts"]
    logger.info("Model: %d layers, %d experts", num_layers, num_experts)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))

    logger.info("Loading model (this will use GPU + CPU offload)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    calibration_texts = get_calibration_texts(args.num_samples)
    logger.info("Running calibration with %d samples, seq_len=%d...",
                args.num_samples, args.seq_len)
    amax_dict = collect_amax(model, tokenizer, args.num_samples, args.seq_len,
                             calibration_texts)

    activated = sum(1 for v in amax_dict.values() if v > 0)
    total = num_layers * num_experts * 3
    logger.info("Activated %d/%d expert projections (%.1f%%)",
                activated, total, 100 * activated / total)

    del model
    torch.cuda.empty_cache()

    logger.info("Computing input_global_scale values...")
    scales = compute_input_global_scales(amax_dict, num_layers, num_experts)
    logger.info("Computed %d scale tensors", len(scales))

    vals = [t.item() for t in scales.values()]
    logger.info("Scale stats: min=%.4f, max=%.4f, mean=%.4f",
                min(vals), max(vals), sum(vals) / len(vals))

    logger.info("Saving to checkpoint...")
    save_scales_to_checkpoint(scales, args.checkpoint)
    update_config(args.checkpoint)

    logger.info("Done. Checkpoint is now W4A4 NVFP4-compatible.")


if __name__ == "__main__":
    main()
