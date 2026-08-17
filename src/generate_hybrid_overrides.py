#!/usr/bin/env python3
"""
Generate hybrid tensor overrides by combining sensitivity data with
forced assignments (APEX research, MTP protection).

Reads results/sensitivity.json + GGUF tensor names. Outputs anchored
regex patterns for llama-quantize --tensor-type-file.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
import shutil

EDGE_FRACTION = 0.12

# SSM recurrence tensors — preserve at source precision (BF16 -> f16)
FORCED_F16_SSM = {
    "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
    "ssm_conv1d.weight", "ssm_norm.weight", "ssm_dt.bias", "ssm_a",
}

# Small tensors that must stay at source precision (norms, biases)
FORCED_F16_SMALL = {
    "attn_norm.weight", "post_attention_norm.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "output_norm.weight",
}

# Groups where sensitivity probe is unreliable — let base quant handle
BASE_GROUPS = {
    "ffn_down.weight [middle]",
    "ffn_gate.weight [middle]",
    "ffn_up.weight [middle]",
    "ffn_gate.weight [edge]",
    "ffn_up.weight [edge]",
    "output.weight",
    "token_embd.weight",
}

THRESHOLDS = [
    (0.008, "f16"),
    (0.004, "q8_0"),
    (0.002, "q6_k"),
]


def get_layer_count(tensor_names):
    max_layer = -1
    for name in tensor_names:
        m = re.match(r"^blk\.(\d+)\.", name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    return max_layer + 1 if max_layer >= 0 else 0


def read_tensor_names(model_path):
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
    return [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]


def build_groups(tensor_names):
    n_layers = get_layer_count(tensor_names)
    edge_n = max(1, int(n_layers * EDGE_FRACTION))

    mtp_blocks = set()
    for name in tensor_names:
        if "nextn." in name:
            m = re.match(r"^blk\.(\d+)\.", name)
            if m:
                mtp_blocks.add(int(m.group(1)))

    skip_tensors = []
    mtp_tensors = []
    base_groups = defaultdict(list)

    for name in tensor_names:
        m = re.match(r"^blk\.(\d+)\.", name)
        if m and int(m.group(1)) in mtp_blocks:
            mtp_tensors.append(name)
            continue

        suffix = re.sub(r"^blk\.\d+\.", "", name)
        if suffix in FORCED_F16_SSM or suffix in FORCED_F16_SMALL:
            skip_tensors.append(name)
            continue

        base_groups[suffix].append(name)

    FFN_SUFFIXES = {"ffn_down.weight", "ffn_gate.weight", "ffn_up.weight"}
    groups = {}
    for suffix, names in base_groups.items():
        if suffix in FFN_SUFFIXES and n_layers > 0:
            edge, middle = [], []
            for n in names:
                lm = re.match(r"^blk\.(\d+)\.", n)
                if not lm:
                    edge.append(n)
                    continue
                layer = int(lm.group(1))
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

    forced_f16_tensors = []
    for name in tensor_names:
        suffix = re.sub(r"^blk\.\d+\.", "", name)
        m = re.match(r"^blk\.(\d+)\.", name)
        if m and int(m.group(1)) in mtp_blocks:
            continue
        if suffix in FORCED_F16_SSM or suffix in FORCED_F16_SMALL:
            forced_f16_tensors.append(name)
        elif not re.match(r"^blk\.\d+\.", name) and suffix in FORCED_F16_SMALL:
            forced_f16_tensors.append(name)

    return groups, skip_tensors, mtp_tensors, forced_f16_tensors


def assign_level(kl_mean, group_name):
    if group_name in BASE_GROUPS:
        return None
    if kl_mean is None:
        return None
    for threshold, level in THRESHOLDS:
        if kl_mean > threshold:
            return level
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate hybrid tensor overrides")
    parser.add_argument("--sensitivity", default="results/sensitivity.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="configs/tensor_overrides.txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.sensitivity) as f:
        sensitivity = json.load(f)

    tensor_names = read_tensor_names(args.model)
    groups, skip_tensors, mtp_tensors, forced_f16 = build_groups(tensor_names)

    lines = []
    lines.append("# Hybrid tensor overrides — sensitivity data + APEX research + MTP protection")
    lines.append(f"# Generated by generate_hybrid_overrides.py from {args.sensitivity}")
    lines.append("#")

    lines.append("")
    lines.append("# SSM recurrence + small tensors — f16 (preserve source BF16 precision)")
    for name in sorted(forced_f16):
        lines.append(f"^{re.escape(name)}$=f16")

    lines.append("")
    lines.append("# MTP layer — f16 (speculative decoding accuracy)")
    for name in sorted(mtp_tensors):
        lines.append(f"^{re.escape(name)}$=f16")

    ranked = []
    for group_name, tensor_list in sorted(groups.items()):
        sens = sensitivity.get("groups", {}).get(group_name, {})
        kl_mean = sens.get("kl_mean")
        kl_max = sens.get("kl_max")
        level = assign_level(kl_mean, group_name)
        ranked.append((group_name, tensor_list, kl_mean, kl_max, level))

    ranked.sort(key=lambda x: x[2] if x[2] is not None else -1, reverse=True)

    for group_name, tensor_list, kl_mean, kl_max, level in ranked:
        if level is None:
            continue
        mean_str = f"{kl_mean:.6f}" if kl_mean is not None else "N/A"
        max_str = f"{kl_max:.4f}" if kl_max is not None else "N/A"
        lines.append("")
        lines.append(f"# {group_name}: KL mean={mean_str} max={max_str} -> {level}")
        for name in sorted(tensor_list):
            lines.append(f"^{re.escape(name)}$={level}")

    content = "\n".join(lines) + "\n"

    override_count = sum(1 for l in lines if l and not l.startswith("#"))
    total_tensors = sum(len(v) for v in groups.values())

    print(f"Override summary:")
    print(f"  f16:  {len(forced_f16)} tensors (SSM recurrence + norms)")
    print(f"  f16:  {len(mtp_tensors)} tensors (MTP)")
    for level_name in ["f16", "q8_0", "q6_k"]:
        count = sum(len(tl) for _, tl, _, _, lv in ranked if lv == level_name)
        groups_at = [gn for gn, _, _, _, lv in ranked if lv == level_name]
        if count:
            print(f"  {level_name:5}: {count} tensors ({', '.join(groups_at)})")
    base_count = sum(len(tl) for _, tl, _, _, lv in ranked if lv is None)
    print(f"  base: {base_count} tensors")
    print(f"  Total overrides: {override_count}")
    print()

    if args.dry_run:
        print("--- DRY RUN (not writing) ---")
        print(content)
        return

    if os.path.exists(args.output):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{args.output}.{ts}.bak"
        shutil.copy2(args.output, backup)
        print(f"Backed up: {backup}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(content)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
