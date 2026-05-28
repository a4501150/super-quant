#!/usr/bin/env python3
"""
Per-tensor-group KL divergence sensitivity analysis.

For each tensor group in the model, quantizes only that group to Q4_K (keeping
everything else at F16) and measures KL divergence vs pure F16 logits. Groups
with high KL divergence are sensitive and need higher precision in the final
quantization.

Usage:
    make sensitivity

Output:
    configs/tensor_overrides.txt  — per-tensor quant assignments
    results/sensitivity.json      — raw KL scores per group
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

SKIP_SUFFIXES = {
    "attn_norm.weight",
    "post_attention_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "output_norm.weight",
    "ssm_norm.weight",
    "ssm_dt.bias",
    "ssm_a",
}

PROBE_QUANT = "Q4_0"
KL_CHUNKS = 16
EDGE_FRACTION = 0.12


def get_layer_count(tensor_names: list[str]) -> int:
    """Find the highest block number in tensor names."""
    max_layer = -1
    for name in tensor_names:
        m = re.match(r"^blk\.(\d+)\.", name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    return max_layer + 1 if max_layer >= 0 else 0


def split_edge_middle(
    tensor_names: list[str], n_layers: int,
) -> tuple[list[str], list[str]]:
    """Split block tensors into edge (first/last ~12%) and middle."""
    edge_n = max(1, int(n_layers * EDGE_FRACTION))
    edge, middle = [], []
    for name in tensor_names:
        m = re.match(r"^blk\.(\d+)\.", name)
        if not m:
            edge.append(name)
            continue
        layer = int(m.group(1))
        if layer < edge_n or layer >= n_layers - edge_n:
            edge.append(name)
        else:
            middle.append(name)
    return edge, middle


def discover_tensor_groups(
    model_path: str, position_aware: bool = True,
) -> tuple[dict[str, list[str]], list[str]]:
    """Read GGUF tensor names and group by suffix pattern.

    Returns (groups, skip_tensors) where skip_tensors are norms/biases
    that should always stay at f32.

    If position_aware, FFN groups are split into edge/middle subgroups
    to detect U-shaped sensitivity (edge layers more sensitive).
    """
    result = subprocess.run(
        [
            "uv", "run", "python3", "-c",
            f"import sys; sys.path.insert(0, '/home/jinyang/src/llama.cpp/gguf-py');"
            f"import gguf; reader = gguf.GGUFReader('{model_path}');"
            f"[print(t.name) for t in reader.tensors]",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to read tensor names: {result.stderr}")
        sys.exit(1)

    tensor_names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
    n_layers = get_layer_count(tensor_names)

    mtp_blocks = set()
    for name in tensor_names:
        if "nextn." in name:
            m = re.match(r"^blk\.(\d+)\.", name)
            if m:
                mtp_blocks.add(int(m.group(1)))

    skip_tensors = []
    base_groups: dict[str, list[str]] = defaultdict(list)
    for name in tensor_names:
        m = re.match(r"^blk\.(\d+)\.", name)
        if m and int(m.group(1)) in mtp_blocks:
            skip_tensors.append(name)
            continue
        suffix = re.sub(r"^blk\.\d+\.", "", name)
        if suffix in SKIP_SUFFIXES:
            skip_tensors.append(name)
            continue
        base_groups[suffix].append(name)

    if not position_aware or n_layers == 0:
        return dict(base_groups), skip_tensors

    FFN_SUFFIXES = {"ffn_down.weight", "ffn_gate.weight", "ffn_up.weight"}
    groups: dict[str, list[str]] = {}
    for suffix, names in base_groups.items():
        if suffix in FFN_SUFFIXES:
            edge, middle = split_edge_middle(names, n_layers)
            if edge:
                groups[f"{suffix} [edge]"] = edge
            if middle:
                groups[f"{suffix} [middle]"] = middle
        else:
            groups[suffix] = names

    return groups, skip_tensors


def save_f16_logits(
    llama_perplexity: str, model: str, test_file: str,
    logits_path: str, gpu_layers: int, chunks: int,
) -> bool:
    """Run F16 model and save reference logits."""
    result = subprocess.run(
        [
            llama_perplexity,
            "-m", model,
            "-f", test_file,
            "-ngl", str(gpu_layers),
            "--chunks", str(chunks),
            "--save-all-logits", logits_path,
        ],
        capture_output=True, text=True, timeout=600,
    )
    return os.path.exists(logits_path)


def measure_kl(
    llama_perplexity: str, model: str, test_file: str,
    logits_path: str, gpu_layers: int, chunks: int,
) -> dict[str, float | None]:
    """Measure KL divergence of a quantized model vs F16 logits."""
    try:
        result = subprocess.run(
            [
                llama_perplexity,
                "-m", model,
                "-f", test_file,
                "-ngl", str(gpu_layers),
                "--chunks", str(chunks),
                "--kl-divergence",
                "--kl-divergence-base", logits_path,
            ],
            capture_output=True, text=True, timeout=600,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("    WARNING: KL measurement timed out")
        return {"mean": None, "max": None, "p999": None}

    kl_mean = kl_max = kl_p999 = None
    for line in output.split("\n"):
        m = re.match(r"^Mean\s+KLD:\s+([\d.]+)", line)
        if m:
            kl_mean = float(m.group(1))
        m = re.match(r"^Maximum KLD:\s+([\d.]+)", line)
        if m:
            kl_max = float(m.group(1))
        m = re.match(r"^99\.9%\s+KLD:\s+([\d.]+)", line)
        if m:
            kl_p999 = float(m.group(1))

    return {"mean": kl_mean, "max": kl_max, "p999": kl_p999}


def quantize_with_overrides(
    llama_quantize: str, model: str, output: str, override_file: str,
    probe_quant: str,
) -> bool:
    """Quantize model with one group at probe_quant and everything else at F16.

    llama-quantize won't demote tensors below the base type, so we use
    probe_quant as the base and override all non-target tensors to F16.
    """
    result = subprocess.run(
        [
            llama_quantize,
            "--tensor-type-file", override_file,
            model, output, probe_quant,
        ],
        capture_output=True, text=True, timeout=1200,
    )
    return result.returncode == 0 and os.path.exists(output)


def assign_quant_level(kl_mean: float) -> str | None:
    """Map KL mean to a quant level. None = use base quant type."""
    if kl_mean > 0.05:
        return "f32"
    if kl_mean > 0.01:
        return "f16"
    if kl_mean > 0.005:
        return "q8_0"
    if kl_mean > 0.002:
        return "q6_k"
    return None


def generate_overrides(
    ranked: list[tuple[str, list[str], dict]],
    skip_tensors: list[str],
    output_path: str,
):
    """Write tensor_overrides.txt from ranked sensitivity results."""
    lines = []
    lines.append("# Auto-generated by sensitivity_analysis.py")
    lines.append("# Tensor groups ranked by KL divergence sensitivity (highest first)")
    lines.append("# Groups with [edge]/[middle] reflect U-shaped layer sensitivity")
    lines.append("#")

    mtp_blocks = set()
    for n in skip_tensors:
        if "nextn." in n:
            m = re.match(r"^blk\.(\d+)\.", n)
            if m:
                mtp_blocks.add(int(m.group(1)))
    mtp_tensors = [
        n for n in skip_tensors
        if any(n.startswith(f"blk.{b}.") for b in mtp_blocks)
    ]
    if mtp_tensors:
        lines.append("# MTP layer — pinned at f16 to preserve speculative decoding accuracy")
        for name in sorted(mtp_tensors):
            escaped = re.escape(name)
            lines.append(f"^{escaped}$=f16")
        lines.append("")

    for group_name, tensor_names, kl in ranked:
        kl_mean = kl.get("mean")
        if kl_mean is None:
            continue
        quant = assign_quant_level(kl_mean)
        if quant is None:
            continue

        kl_max = kl.get("max")
        max_str = f"{kl_max:.4f}" if isinstance(kl_max, float) else "N/A"
        lines.append(f"# {group_name}: KL mean={kl_mean:.6f} max={max_str} -> {quant}")
        for name in sorted(tensor_names):
            escaped = re.escape(name)
            lines.append(f"^{escaped}$={quant}")
        lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Per-tensor KL divergence sensitivity analysis"
    )
    parser.add_argument("--model", required=True, help="Path to F16 GGUF model")
    parser.add_argument("--test-file", required=True, help="Text file for KL measurement")
    parser.add_argument("--llama-perplexity", required=True)
    parser.add_argument("--llama-quantize", required=True)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--output", default="configs/tensor_overrides.txt")
    parser.add_argument("--output-json", default="results/sensitivity.json")
    parser.add_argument(
        "--probe-quant", default=PROBE_QUANT,
        help=f"Quant type to probe each group with (default: {PROBE_QUANT})",
    )
    parser.add_argument(
        "--chunks", type=int, default=KL_CHUNKS,
        help=f"Number of chunks for KL measurement (default: {KL_CHUNKS})",
    )
    args = parser.parse_args()

    chunks = args.chunks

    print("=== Per-Tensor Sensitivity Analysis ===")
    print(f"Model:      {args.model}")
    print(f"Test data:  {args.test_file}")
    print(f"Probe:      {args.probe_quant}")
    print(f"Chunks:     {chunks}")
    print()

    # Step 1: Discover tensor groups
    print("--- Discovering tensor groups ---")
    groups, skip_tensors = discover_tensor_groups(args.model)
    print(f"Found {len(groups)} tensor groups ({sum(len(v) for v in groups.values())} tensors)")
    print(f"  ({len(skip_tensors)} norm/bias tensors skipped, kept at f32)")
    for suffix, names in sorted(groups.items()):
        print(f"  {suffix:<30} {len(names)} tensors")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        f16_logits = os.path.join(tmpdir, "f16_logits.bin")

        # Step 2: Save F16 reference logits
        print("--- Saving F16 reference logits ---")
        if not save_f16_logits(
            args.llama_perplexity, args.model, args.test_file,
            f16_logits, args.gpu_layers, chunks,
        ):
            print("ERROR: Failed to save F16 reference logits")
            sys.exit(1)
        logits_size = os.path.getsize(f16_logits) / (1024 ** 3)
        print(f"  Saved: {f16_logits} ({logits_size:.1f} GB)")
        print()

        # Step 3: Probe each tensor group
        print("--- Probing tensor groups ---")
        print(f"{'Group':<30} {'KL mean':>10} {'KL max':>10} {'KL p999':>10} {'Level':>8}")
        print("-" * 75)

        all_group_tensors = {n for names in groups.values() for n in names}

        results = {}
        for suffix, tensor_names in sorted(groups.items()):
            target_set = set(tensor_names)
            override_path = os.path.join(tmpdir, "override.txt")
            with open(override_path, "w") as f:
                for name in sorted(all_group_tensors - target_set):
                    escaped = re.escape(name)
                    f.write(f"^{escaped}$=f16\n")
                for name in sorted(skip_tensors):
                    escaped = re.escape(name)
                    f.write(f"^{escaped}$=f32\n")

            probe_gguf = os.path.join(tmpdir, "probe.gguf")
            if not quantize_with_overrides(
                args.llama_quantize, args.model, probe_gguf, override_path,
                args.probe_quant,
            ):
                print(f"  {suffix:<30} {'SKIP':>10} (quantize failed)")
                continue

            # Measure KL
            kl = measure_kl(
                args.llama_perplexity, probe_gguf, args.test_file,
                f16_logits, args.gpu_layers, chunks,
            )

            # Clean up probe GGUF immediately
            if os.path.exists(probe_gguf):
                os.remove(probe_gguf)

            kl_mean = kl.get("mean")
            kl_max = kl.get("max")
            kl_p999 = kl.get("p999")
            level = assign_quant_level(kl_mean) if kl_mean is not None else "N/A"

            results[suffix] = {
                "tensor_names": tensor_names,
                "kl_mean": kl_mean,
                "kl_max": kl_max,
                "kl_p999": kl_p999,
                "assigned_quant": level,
                "n_tensors": len(tensor_names),
            }

            mean_str = f"{kl_mean:.6f}" if kl_mean is not None else "N/A"
            max_str = f"{kl_max:.4f}" if kl_max is not None else "N/A"
            p999_str = f"{kl_p999:.4f}" if kl_p999 is not None else "N/A"
            level_str = level if level else "base"
            print(f"  {suffix:<30} {mean_str:>10} {max_str:>10} {p999_str:>10} {level_str:>8}")

    print()

    # Step 4: Rank by sensitivity and generate overrides
    ranked = sorted(
        [
            (suffix, data["tensor_names"], {"mean": data["kl_mean"], "max": data["kl_max"], "p999": data["kl_p999"]})
            for suffix, data in results.items()
            if data["kl_mean"] is not None
        ],
        key=lambda x: x[2]["mean"],
        reverse=True,
    )

    print("--- Generating tensor overrides ---")
    if os.path.exists(args.output):
        from datetime import datetime
        import shutil
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{args.output}.{ts}.bak"
        shutil.copy2(args.output, backup)
        print(f"  Backed up existing: {backup}")
    generate_overrides(ranked, skip_tensors, args.output)
    print(f"  Written: {args.output}")

    n_overridden = sum(
        len(names)
        for _, names, kl in ranked
        if kl["mean"] is not None and assign_quant_level(kl["mean"]) is not None
    )
    print(f"  {n_overridden} tensors overridden, rest use base quant type")

    # Step 5: Save JSON report
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    report = {
        "probe_quant": args.probe_quant,
        "chunks": chunks,
        "groups": {
            suffix: {k: v for k, v in data.items() if k != "tensor_names"}
            for suffix, data in results.items()
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report:  {args.output_json}")

    print()
    print("=== Sensitivity analysis complete ===")
    print("Review the overrides, then: make quantize")


if __name__ == "__main__":
    main()
