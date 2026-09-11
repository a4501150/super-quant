#!/usr/bin/env python3
"""
Parse benchmark results and print a formatted comparison table.
"""

import argparse
import glob
import json
import os
import sys


def load_latest_results(results_dir: str) -> dict:
    pattern = os.path.join(results_dir, "benchmark_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No benchmark results found in {results_dir}")
        sys.exit(1)
    latest = files[-1]
    print(f"Loading: {latest}\n")
    with open(latest) as f:
        return json.load(f)


def strip_model_prefix(name: str, model_name: str) -> str:
    prefix = f"{model_name}-"
    return name[len(prefix):] if name.startswith(prefix) else name


def fmt_speed(val, stddev=None):
    if val == "N/A" or val is None:
        return "N/A"
    s = f"{val}"
    if stddev and float(stddev) > 0:
        s += f" ±{stddev}"
    return s


def print_table(benchmarks: list[dict], model_name: str):
    if not benchmarks:
        print("No benchmarks found.")
        return

    benchmarks.sort(key=lambda b: float(b.get("size_gb", 0)), reverse=True)

    has_ppl = any(b.get("ppl", "N/A") != "N/A" for b in benchmarks)
    has_kl = any(b.get("kl_mean", "N/A") != "N/A" for b in benchmarks)

    header = f"{'Quant':<20} {'Size':>8} {'pp t/s':>14} {'tg t/s':>14}"
    if has_ppl:
        header += f" {'PPL':>8}"
    if has_kl:
        header += f" {'KL mean':>8} {'KL max':>8} {'KL p999':>8}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for b in benchmarks:
        name = strip_model_prefix(b.get("name", "unknown"), model_name)
        size = f"{b['size_gb']} GB"
        pp = fmt_speed(b.get("pp_tokens_per_sec", "N/A"), b.get("pp_stddev"))
        tg = fmt_speed(b.get("tg_tokens_per_sec", "N/A"), b.get("tg_stddev"))
        line = f"{name:<20} {size:>8} {pp:>14} {tg:>14}"
        if has_ppl:
            line += f" {b.get('ppl', 'N/A'):>8}"
        if has_kl:
            line += f" {b.get('kl_mean', 'N/A'):>8} {b.get('kl_max', 'N/A'):>8} {b.get('kl_p999', 'N/A'):>8}"
        print(line)

    print(sep)


def print_markdown(benchmarks: list[dict], model_name: str):
    if not benchmarks:
        return

    benchmarks.sort(key=lambda b: float(b.get("size_gb", 0)), reverse=True)

    has_ppl = any(b.get("ppl", "N/A") != "N/A" for b in benchmarks)
    has_kl = any(b.get("kl_mean", "N/A") != "N/A" for b in benchmarks)

    header = "| Quant | Size | pp t/s | tg t/s |"
    sep = "|-------|------|--------|--------|"
    if has_ppl:
        header += " PPL |"
        sep += "-----|"
    if has_kl:
        header += " KL mean | KL max | KL p999 |"
        sep += "---------|--------|---------|"
    print(header)
    print(sep)

    for b in benchmarks:
        name = strip_model_prefix(b.get("name", "unknown"), model_name)
        size = f"{b['size_gb']} GB"
        pp = fmt_speed(b.get("pp_tokens_per_sec", "N/A"), b.get("pp_stddev"))
        tg = fmt_speed(b.get("tg_tokens_per_sec", "N/A"), b.get("tg_stddev"))
        line = f"| {name} | {size} | {pp} | {tg} |"
        if has_ppl:
            line += f" {b.get('ppl', 'N/A')} |"
        if has_kl:
            line += f" {b.get('kl_mean', 'N/A')} | {b.get('kl_max', 'N/A')} | {b.get('kl_p999', 'N/A')} |"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare quantization benchmark results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", ""),
                        help="Model name prefix to strip from display (or set MODEL_NAME env var)")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown table")
    args = parser.parse_args()

    results = load_latest_results(args.results_dir)
    benchmarks = results.get("benchmarks", [])

    if args.markdown:
        print_markdown(benchmarks, args.model_name)
    else:
        print_table(benchmarks, args.model_name)


if __name__ == "__main__":
    main()
