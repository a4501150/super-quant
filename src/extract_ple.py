#!/usr/bin/env python3
"""Extract PLE n-gram embedding table from a Qwen4-Exp checkpoint into a flat
binary file for SGLang mmap serving (SGLANG_QWEN4_PLE_MMAP).

The checkpoint stores the table as 128 safetensor shards
(ngram_embedding.shard_N.weight), each [~2.5M, 160] in BF16 or FP8.
This script concatenates them row-by-row into a single flat file that the
Qwen4ExpMmapEmbedding loader reads via os.pread / numpy.memmap.

Usage:
    uv run src/extract_ple.py \
        --checkpoint ~/.cache/huggingface/hub/models--orcarouter--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/<hash> \
        --output ~/models/ple/Qwen3.8-Flash-Next-NVFP4
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open


def find_ple_shards(index_path: Path) -> tuple[list[str], str]:
    """Return (sorted shard keys, safetensors filename) from the index."""
    with open(index_path) as f:
        index = json.load(f)
    wm = index["weight_map"]
    pattern = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
    shards = {}
    shard_file = None
    for key, fname in wm.items():
        m = pattern.search(key)
        if m:
            shards[int(m.group(1))] = key
            shard_file = fname
    if not shards:
        print("No PLE ngram_embedding shards found in index.", file=sys.stderr)
        sys.exit(1)
    sorted_keys = [shards[i] for i in sorted(shards.keys())]
    return sorted_keys, shard_file


def find_weight_scale(checkpoint: Path, shard_keys: list[str]) -> float:
    """Find the weight_scale buffer value from the checkpoint."""
    prefix = shard_keys[0].rsplit(".ngram_embedding.", 1)[0]
    scale_name = f"{prefix}.ngram_embedding.weight_scale"

    with open(checkpoint / "model.safetensors.index.json") as f:
        index = json.load(f)
    wm = index["weight_map"]

    if scale_name in wm:
        fname = wm[scale_name]
        with safe_open(str(checkpoint / fname), framework="pt") as f:
            return f.get_tensor(scale_name).float().item()

    for fname in sorted(set(wm.values())):
        fpath = checkpoint / fname
        if not fpath.exists():
            continue
        with safe_open(str(fpath), framework="pt") as f:
            if scale_name in f.keys():
                return f.get_tensor(scale_name).float().item()

    print(f"weight_scale not found, defaulting to 1.0", file=sys.stderr)
    return 1.0


def main():
    parser = argparse.ArgumentParser(description="Extract PLE table for mmap serving")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the HF checkpoint snapshot directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for ple.bin and ple.json")
    parser.add_argument("--dtype", choices=["bf16", "f8_e4m3"], default=None,
                        help="Output dtype. Default: same as checkpoint.")
    args = parser.parse_args()

    index_path = args.checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"Index not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    shard_keys, shard_filename = find_ple_shards(index_path)
    num_shards = len(shard_keys)
    print(f"Found {num_shards} PLE shards in {shard_filename}")

    shard_path = args.checkpoint / shard_filename
    print(f"Opening {shard_path} ({os.path.getsize(shard_path) / 1e9:.1f} GB)...")

    with safe_open(str(shard_path), framework="pt") as sf:
        first = sf.get_tensor(shard_keys[0])
        src_dtype = first.dtype
        rows_per_shard = first.shape[0]
        dim = first.shape[1]
        del first

    if args.dtype == "f8_e4m3":
        out_dtype = torch.float8_e4m3fn
        dtype_str = "F8_E4M3"
    elif args.dtype == "bf16":
        out_dtype = torch.bfloat16
        dtype_str = "BF16"
    else:
        out_dtype = src_dtype
        dtype_str = "F8_E4M3" if src_dtype == torch.float8_e4m3fn else "BF16"

    total_rows = num_shards * rows_per_shard
    itemsize = torch.empty(0, dtype=out_dtype).element_size()
    row_bytes = dim * itemsize
    total_bytes = total_rows * row_bytes

    print(f"Source dtype: {src_dtype}, output dtype: {out_dtype}")
    print(f"Table: {total_rows} rows x {dim} dim x {itemsize} B = {total_bytes / 1e9:.2f} GB")

    args.output.mkdir(parents=True, exist_ok=True)
    bin_name = f"ple.{dtype_str.lower().replace('_', '')}.bin"
    bin_path = args.output / bin_name

    print(f"Writing {bin_path}...")
    with open(bin_path, "wb") as out_f:
        with safe_open(str(shard_path), framework="pt") as sf:
            for i, key in enumerate(shard_keys):
                tensor = sf.get_tensor(key)
                if tensor.dtype != out_dtype:
                    tensor = tensor.to(out_dtype)
                out_f.write(tensor.view(torch.uint8).numpy().tobytes())
                if (i + 1) % 16 == 0 or i == num_shards - 1:
                    pct = (i + 1) / num_shards * 100
                    written = (i + 1) * rows_per_shard * row_bytes
                    print(f"  {i + 1}/{num_shards} shards ({pct:.0f}%, {written / 1e9:.1f} GB)")

    actual_size = os.path.getsize(bin_path)
    if actual_size != total_bytes:
        print(f"ERROR: expected {total_bytes} bytes, got {actual_size}", file=sys.stderr)
        sys.exit(1)

    weight_scale = find_weight_scale(args.checkpoint, shard_keys)

    metadata = {
        "file": bin_name,
        "rows": total_rows,
        "dim": dim,
        "dtype": dtype_str,
        "weight_scale": weight_scale,
    }
    json_path = args.output / "ple.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone.")
    print(f"  Binary: {bin_path} ({actual_size / 1e9:.2f} GB)")
    print(f"  Metadata: {json_path}")
    print(f"  weight_scale: {weight_scale}")
    print(f"\nTo serve:")
    print(f"  SGLANG_QWEN4_PLE_MMAP={args.output} python -m sglang.launch_server ...")


if __name__ == "__main__":
    main()
