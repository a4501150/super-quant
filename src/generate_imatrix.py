#!/usr/bin/env python3
"""
GPU-native imatrix generator using PyTorch forward hooks.

Replaces llama-imatrix for CUDA targets. llama-imatrix does a synchronous
cudaMemcpy D2H for every tracked tensor at every chunk; this tool keeps all
accumulation on GPU and transfers only the final statistics.

Outputs the legacy llama.cpp .dat format consumed by merge_imatrix.py and
llama-quantize --imatrix.
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
import torch
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


class ImatrixAccumulator:
    """GPU-resident accumulator for squared activation magnitudes."""

    def __init__(self):
        self.sum_sq: dict[str, torch.Tensor] = {}
        self.count: dict[str, int] = {}

    def register(self, gguf_name: str, in_features: int, device: torch.device):
        self.sum_sq[gguf_name] = torch.zeros(in_features, dtype=torch.float32, device=device)
        self.count[gguf_name] = 0

    def reset(self):
        for name in self.sum_sq:
            self.sum_sq[name].zero_()
            self.count[name] = 0

    def accumulate(self, gguf_name: str, x: torch.Tensor, in_features: int):
        flat = x.detach().reshape(-1, in_features)
        n_tokens = flat.shape[0]
        self.sum_sq[gguf_name].add_(flat.float().pow(2).sum(dim=0))
        self.count[gguf_name] += n_tokens

    def get_entries(self) -> dict[str, dict]:
        entries = {}
        for name in sorted(self.sum_sq):
            n_tokens = self.count[name]
            if n_tokens == 0:
                continue
            values = (self.sum_sq[name] / n_tokens).cpu().numpy().astype(np.float32)
            entries[name] = {
                "n_chunks": n_tokens,
                "n_values": values.size,
                "values": values,
            }
        return entries


def write_imatrix_dat(path: str, entries: dict[str, dict]):
    """Write legacy llama.cpp imatrix .dat format."""
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(entries)))
        for name in sorted(entries):
            data = entries[name]
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("i", data["n_chunks"]))
            f.write(struct.pack("i", data["n_values"]))
            f.write(data["values"].tobytes())


def process_calibration_file(
    input_path: str,
    model: torch.nn.Module,
    tokenizer,
    accumulator: ImatrixAccumulator,
    context_size: int,
    limit_tokens: int | None = None,
) -> int:
    """Run forward passes over a calibration file and accumulate importance."""
    input_device = next(model.parameters()).device
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = tokenize_and_chunk(text, tokenizer, context_size)
    total_tokens = sum(len(c) for c in chunks)

    if limit_tokens:
        limited_chunks = []
        running = 0
        for c in chunks:
            limited_chunks.append(c)
            running += len(c)
            if running >= limit_tokens:
                break
        chunks = limited_chunks
        total_tokens = sum(len(c) for c in chunks)

    print(f"    Tokens: {total_tokens:,} in {len(chunks)} chunks of up to {context_size}")

    processed = 0
    for chunk in tqdm(chunks, desc="    Chunks", leave=False):
        input_ids = torch.tensor([chunk], dtype=torch.long, device=input_device)
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
        processed += len(chunk)

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="GPU-native imatrix generator (PyTorch, replaces llama-imatrix)"
    )
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--calibration-dir", help="Directory with domain .txt files")
    parser.add_argument("--domains", nargs="+", default=["general", "code", "reasoning", "agentic"])
    parser.add_argument("--output-dir", help="Output directory for .dat files")
    parser.add_argument("--input", help="Single calibration .txt file (alternative to --calibration-dir)")
    parser.add_argument("--output", help="Single output .dat file (with --input)")
    parser.add_argument("--gguf-arch-key", required=True, help="GGUF architecture key (e.g. qwen35, llama)")
    parser.add_argument("--llamacpp-dir", default="/home/jinyang/src/llama.cpp",
                        help="Path to llama.cpp tree (for gguf-py)")
    parser.add_argument("--context-size", type=int, default=32768, help="Tokens per chunk")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-tokens", type=int, default=None, help="Cap tokens per job (for validation)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .dat files")
    args = parser.parse_args()

    if args.input and args.output:
        jobs = [{"name": "single", "input": args.input, "output": args.output}]
    elif args.calibration_dir and args.output_dir:
        jobs = []
        for domain in args.domains:
            jobs.append({
                "name": domain,
                "input": os.path.join(args.calibration_dir, f"{domain}.txt"),
                "output": os.path.join(args.output_dir, f"imatrix_{domain}.dat"),
            })
        combined_path = os.path.join(args.calibration_dir, "combined.txt")
        if os.path.isfile(combined_path):
            jobs.append({
                "name": "combined",
                "input": combined_path,
                "output": os.path.join(args.output_dir, "imatrix_combined.dat"),
            })
    else:
        print("ERROR: Provide either --input/--output or --calibration-dir/--output-dir")
        sys.exit(1)

    pending_jobs = []
    for job in jobs:
        if not os.path.isfile(job["input"]):
            print(f"ERROR: Calibration file not found: {job['input']}")
            sys.exit(1)
        if os.path.isfile(job["output"]) and not args.force:
            print(f"Skipping {job['name']} — already exists: {job['output']}")
            continue
        pending_jobs.append(job)

    if not pending_jobs:
        print("All imatrix files already exist. Use --force to rebuild.")
        return

    print("=== GPU Imatrix Generator ===")
    print(f"Model:        {args.model_id}")
    print(f"Architecture: {args.gguf_arch_key}")
    print(f"Context:      {args.context_size}")
    print(f"Dtype:        {args.dtype}")
    print_gpu_info()
    print(f"Jobs:         {len(pending_jobs)}")
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
    print(f"  Layers: {n_layers}")
    print()

    print("Building tensor name map...")
    tmap = resolve_arch(args.gguf_arch_key, args.llamacpp_dir, n_layers)

    mapped, unmapped = discover_linear_modules(model, tmap)
    print(f"  Mapped nn.Linear modules: {len(mapped)}")
    if unmapped:
        print(f"  Unmapped nn.Linear modules: {len(unmapped)}")
        for name in unmapped[:5]:
            print(f"    {name}")
        if len(unmapped) > 5:
            print(f"    ... and {len(unmapped) - 5} more")
    print()

    if not mapped:
        print("ERROR: No nn.Linear modules mapped to GGUF tensor names")
        sys.exit(1)

    accumulator = ImatrixAccumulator()
    for gguf_name, (module, _hf_name) in mapped.items():
        weight_device = module.weight.device
        accumulator.register(gguf_name, module.in_features, weight_device)

    hooks = []
    for gguf_name, (module, _hf_name) in mapped.items():
        in_features = module.in_features
        hook = module.register_forward_pre_hook(
            lambda _mod, args, _gn=gguf_name, _if=in_features: accumulator.accumulate(_gn, args[0], _if)
        )
        hooks.append(hook)

    for job in pending_jobs:
        accumulator.reset()

        print(f"--- {job['name']} ---")
        print(f"  Input:  {job['input']}")
        print(f"  Output: {job['output']}")

        t0 = time.time()
        n_tokens = process_calibration_file(
            job["input"], model, tokenizer, accumulator,
            args.context_size, args.limit_tokens,
        )
        elapsed = time.time() - t0

        entries = accumulator.get_entries()
        os.makedirs(os.path.dirname(job["output"]) or ".", exist_ok=True)
        write_imatrix_dat(job["output"], entries)

        size_mb = os.path.getsize(job["output"]) / 1024 / 1024
        print(f"    {len(entries)} tensors, {n_tokens:,} tokens, {elapsed:.1f}s, {size_mb:.1f} MB")
        print()

    for hook in hooks:
        hook.remove()

    print("=== Imatrix generation complete ===")
    for job in pending_jobs:
        if os.path.isfile(job["output"]):
            size_mb = os.path.getsize(job["output"]) / 1024 / 1024
            print(f"  {job['output']}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
