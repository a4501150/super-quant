#!/usr/bin/env python3
"""
GPU-native per-tensor sensitivity analysis.

For each tensor group, simulates Q4_0 quantization in-place on GPU and
measures KL divergence vs unquantized logits. No disk I/O, no GGUF
writing, no subprocess calls. Runs ~5-10x faster than the llama.cpp
subprocess approach.

Usage:
    make sensitivity

Output:
    results/sensitivity.json      — raw KL scores per group
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from model_utils import (
    discover_linear_modules,
    get_n_layers,
    load_model,
    print_gpu_info,
    resolve_arch,
    tokenize_and_chunk,
)

EDGE_FRACTION = 0.12

SKIP_SUFFIXES = {
    "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
    "ssm_conv1d.weight", "ssm_norm.weight", "ssm_dt.bias", "ssm_a",
    "attn_norm.weight", "post_attention_norm.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "output_norm.weight",
}


def quantize_q4_0_roundtrip(weight: torch.Tensor) -> torch.Tensor:
    """Simulate Q4_0 quantization noise on a weight tensor."""
    orig_shape = weight.shape
    flat = weight.reshape(-1, 32).float()
    absmax = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scale = absmax / 7.0
    quantized = (flat / scale).round().clamp(-8, 7)
    return (quantized * scale).reshape(orig_shape).to(weight.dtype)


def build_groups(
    mapped: dict[str, tuple[torch.nn.Linear, str]], n_layers: int,
) -> tuple[dict[str, list[str]], list[str]]:
    """Group mapped tensors by suffix pattern, split FFN into edge/middle."""
    edge_n = max(1, int(n_layers * EDGE_FRACTION))

    mtp_tensors = []
    skip_tensors = []
    base_groups: dict[str, list[str]] = defaultdict(list)

    for gguf_name in mapped:
        suffix = re.sub(r"^blk\.\d+\.", "", gguf_name)

        m = re.match(r"^blk\.(\d+)\.", gguf_name)
        if m and "nextn." in gguf_name:
            mtp_tensors.append(gguf_name)
            continue
        block = int(m.group(1)) if m else -1
        if block >= n_layers:
            mtp_tensors.append(gguf_name)
            continue

        if suffix in SKIP_SUFFIXES:
            skip_tensors.append(gguf_name)
            continue

        base_groups[suffix].append(gguf_name)

    FFN_SUFFIXES = {"ffn_down.weight", "ffn_gate.weight", "ffn_up.weight"}
    groups: dict[str, list[str]] = {}
    for suffix, names in base_groups.items():
        if suffix in FFN_SUFFIXES and n_layers > 0:
            edge, middle = [], []
            for n in names:
                lm = re.match(r"^blk\.(\d+)\.", n)
                layer = int(lm.group(1)) if lm else 0
                if layer < edge_n or layer >= n_layers - edge_n:
                    edge.append(n)
                else:
                    middle.append(n)
            if edge:
                groups[f"{suffix} [edge]"] = edge
            if middle:
                groups[f"{suffix} [middle]"] = middle
        else:
            groups[suffix] = names

    return groups, skip_tensors + mtp_tensors


def main():
    parser = argparse.ArgumentParser(
        description="GPU-native per-tensor sensitivity analysis"
    )
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--test-file", required=True, help="Calibration text for KL measurement")
    parser.add_argument("--gguf-arch-key", required=True, help="GGUF architecture key (e.g. qwen35, llama)")
    parser.add_argument("--llamacpp-dir", default="/home/jinyang/src/llama.cpp",
                        help="Path to llama.cpp tree (for gguf-py)")
    parser.add_argument("--output-json", default="results/sensitivity.json")
    parser.add_argument("--chunks", type=int, default=16, help="Number of calibration chunks")
    parser.add_argument("--context-size", type=int, default=4096, help="Tokens per chunk")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    print("=== GPU Per-Tensor Sensitivity Analysis ===")
    print(f"Model:      {args.model_id}")
    print(f"Test data:  {args.test_file}")
    print(f"Chunks:     {args.chunks}")
    print(f"Context:    {args.context_size}")
    print_gpu_info()
    print()

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code,
    )

    n_layers = get_n_layers(args.model_id, args.trust_remote_code)

    print(f"Loading model ({n_layers} layers)...")
    t0 = time.time()
    model = load_model(args.model_id, torch_dtype, args.trust_remote_code)
    model.eval()
    text_cfg = getattr(model.config, "text_config", model.config)
    text_cfg.use_cache = False
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    print("Building tensor name map...")
    tmap = resolve_arch(args.gguf_arch_key, args.llamacpp_dir, n_layers)
    mapped, _ = discover_linear_modules(model, tmap)
    print(f"  Mapped nn.Linear modules: {len(mapped)}")
    print()

    groups, skipped = build_groups(mapped, n_layers)
    print(f"Tensor groups to probe: {len(groups)}")
    print(f"Skipped tensors (SSM/norms/MTP): {len(skipped)}")
    for suffix, names in sorted(groups.items()):
        print(f"  {suffix:<30} {len(names)} tensors")
    print()

    with open(args.test_file, "r", encoding="utf-8") as f:
        text = f.read()
    all_chunks = tokenize_and_chunk(text, tokenizer, args.context_size)
    chunks = all_chunks[: args.chunks]
    input_device = next(model.parameters()).device
    print(f"Using {len(chunks)} chunks of {args.context_size} tokens")
    print()

    print("--- Probing tensor groups ---")
    header = f"{'Group':<30} {'KL mean':>10} {'KL max':>10} {'KL p99.9':>10}"
    print(header)
    print("-" * len(header))

    results = {}
    t_total = time.time()

    for group_name, gguf_names in sorted(groups.items()):
        modules = [mapped[n][0] for n in gguf_names]

        kl_all = []
        for chunk in tqdm(chunks, desc=f"  {group_name[:25]}", leave=False):
            input_ids = torch.tensor([chunk], dtype=torch.long, device=input_device)

            with torch.inference_mode():
                ref_logits = model(input_ids=input_ids, use_cache=False).logits

            saved_weights = [m.weight.data.clone() for m in modules]
            for m in modules:
                m.weight.data = quantize_q4_0_roundtrip(m.weight.data)

            with torch.inference_mode():
                test_logits = model(input_ids=input_ids, use_cache=False).logits

            for m, w in zip(modules, saved_weights):
                m.weight.data = w
            del saved_weights

            ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
            test_lp = F.log_softmax(test_logits.float(), dim=-1)
            kl = (ref_lp.exp() * (ref_lp - test_lp)).sum(dim=-1)
            kl_all.append(kl.squeeze(0))
            del ref_logits, test_logits, ref_lp, test_lp

        kl_cat = torch.cat(kl_all)
        kl_mean = kl_cat.mean().item()
        kl_max = kl_cat.max().item()
        kl_p999 = kl_cat.quantile(0.999).item()
        del kl_all, kl_cat

        results[group_name] = {
            "kl_mean": kl_mean,
            "kl_max": kl_max,
            "kl_p999": kl_p999,
            "n_tensors": len(gguf_names),
        }

        print(f"  {group_name:<30} {kl_mean:>10.6f} {kl_max:>10.4f} {kl_p999:>10.4f}")

    elapsed = time.time() - t_total
    print()
    print(f"Probing complete in {elapsed:.1f}s")
    print()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    report = {
        "probe_quant": "Q4_0",
        "chunks": len(chunks),
        "context_size": args.context_size,
        "groups": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Written: {args.output_json}")

    print()
    print("=== Sensitivity analysis complete ===")
    print("Next: review results, then make quantize")


if __name__ == "__main__":
    main()
