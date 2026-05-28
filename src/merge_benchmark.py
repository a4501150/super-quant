#!/usr/bin/env python3
"""
Merge llama-bench throughput JSON with perplexity/KL JSON into unified results.
"""

import json
import os
import sys


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <bench.json> <ppl_kl.json> <output.json>")
        sys.exit(1)

    bench_file, ppl_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(bench_file) as f:
        bench = json.load(f)

    with open(ppl_file) as f:
        ppl_kl = json.load(f)

    models: dict[str, dict] = {}
    for entry in bench:
        fname = os.path.basename(entry.get("model_filename", ""))
        name = fname.removesuffix(".gguf")
        if name not in models:
            models[name] = {
                "name": name,
                "file": fname,
                "model_type": entry.get("model_type", ""),
                "size_gb": round(entry.get("model_size", 0) / 1073741824, 2),
            }
        if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
            models[name]["pp_tokens_per_sec"] = round(entry.get("avg_ts", 0), 1)
            models[name]["pp_stddev"] = round(entry.get("stddev_ts", 0), 1)
        elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
            models[name]["tg_tokens_per_sec"] = round(entry.get("avg_ts", 0), 1)
            models[name]["tg_stddev"] = round(entry.get("stddev_ts", 0), 1)

    for name, metrics in ppl_kl.items():
        if name in models:
            models[name].update(metrics)

    result = {"benchmarks": list(models.values())}
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
