#!/usr/bin/env python3
"""
Merge multiple importance matrix (imatrix) .dat files with weighted blending.

DI-MATRIX approach: generate separate imatrices per domain (general, code,
reasoning, agentic), then merge with weights reflecting desired domain balance.

The imatrix .dat format (llama.cpp):
  - Header: 1 int (n_entries)
  - Per entry:
    - name_len (int), name (bytes)
    - n_chunks (int), n_values (int)
    - values (n_values floats)
"""

import argparse
import struct
import sys
from pathlib import Path


def read_imatrix(path: str) -> dict:
    entries = {}
    with open(path, "rb") as f:
        n_entries = struct.unpack("i", f.read(4))[0]
        for _ in range(n_entries):
            name_len = struct.unpack("i", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            n_chunks = struct.unpack("i", f.read(4))[0]
            n_values = struct.unpack("i", f.read(4))[0]
            values = list(struct.unpack(f"{n_values}f", f.read(n_values * 4)))
            entries[name] = {
                "n_chunks": n_chunks,
                "n_values": n_values,
                "values": values,
            }
    return entries


def write_imatrix(path: str, entries: dict):
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(entries)))
        for name, data in entries.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("i", data["n_chunks"]))
            f.write(struct.pack("i", data["n_values"]))
            f.write(struct.pack(f"{data['n_values']}f", *data["values"]))


def merge_imatrices(imatrix_paths: list[str], weights: list[float]) -> dict:
    assert len(imatrix_paths) == len(weights), "Must have same number of paths and weights"
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    imatrices = []
    for path in imatrix_paths:
        print(f"  Reading: {path}")
        imatrices.append(read_imatrix(path))

    all_names = set()
    for im in imatrices:
        all_names.update(im.keys())

    merged = {}
    for name in sorted(all_names):
        present = [(i, im[name]) for i, im in enumerate(imatrices) if name in im]
        if not present:
            continue

        n_values = present[0][1]["n_values"]
        total_chunks = sum(im[name]["n_chunks"] for im in imatrices if name in im)

        merged_values = [0.0] * n_values
        for idx, data in present:
            w = weights[idx]
            for j in range(n_values):
                merged_values[j] += data["values"][j] * w

        merged[name] = {
            "n_chunks": total_chunks,
            "n_values": n_values,
            "values": merged_values,
        }

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple imatrix .dat files with weighted blending (DI-MATRIX)"
    )
    parser.add_argument("inputs", nargs="+", help="Input imatrix .dat files")
    parser.add_argument("-o", "--output", required=True, help="Output merged imatrix .dat")
    parser.add_argument(
        "--weights", nargs="+", type=float,
        help="Blend weights per input (default: equal weights)",
    )
    args = parser.parse_args()

    if args.weights:
        if len(args.weights) != len(args.inputs):
            print(f"ERROR: {len(args.weights)} weights for {len(args.inputs)} inputs")
            sys.exit(1)
        weights = args.weights
    else:
        weights = [1.0] * len(args.inputs)

    print(f"Merging {len(args.inputs)} imatrices:")
    for path, w in zip(args.inputs, weights):
        print(f"  {path} (weight: {w:.3f})")

    merged = merge_imatrices(args.inputs, weights)
    write_imatrix(args.output, merged)

    print(f"\nMerged {len(merged)} tensor entries → {args.output}")
    print(f"Output size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
