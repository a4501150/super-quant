#!/usr/bin/env python3
"""Convert a fused Qwen4Exp checkpoint to per-expert 2D weights."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from llmcompressor.utils import load_context

from model_utils import (
    get_quantization_model_class,
    inspect_source_config,
    load_quantization_recipe,
    load_quantization_tokenizer,
    save_quantization_processor,
    use_streaming_moe_linearization,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_linearized_checkpoint(output_dir):
    index_path = output_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    keys = set(index["weight_map"])
    linearized = {
        key
        for key in keys
        if ".experts." in key
        and any(
            f".{projection}.weight" in key
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    }
    fused = {
        key
        for key in keys
        if ".experts.gate_up_proj" in key or ".experts.down_proj" in key
    }
    if not linearized:
        raise RuntimeError("Converted checkpoint has no per-expert 2D weights")
    if fused:
        raise RuntimeError(
            f"Converted checkpoint still has {len(fused)} fused expert tensors"
        )
    return len(keys), len(linearized)


def main():
    parser = argparse.ArgumentParser(
        description="Linearize a fused Qwen4Exp BF16 checkpoint"
    )
    parser.add_argument("--config", required=True, help="Per-model quantize.json")
    parser.add_argument("--model-id", required=True, help="Fused source checkpoint")
    parser.add_argument("--output-dir", required=True, help="Linearized checkpoint")
    parser.add_argument("--gpu-memory", default="80GiB")
    parser.add_argument("--cpu-memory", default="104GiB")
    parser.add_argument("--shard-size", default="5GB")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_quantization_recipe(args.config)
    recipe["source"]["model_id"] = args.model_id
    source_config = inspect_source_config(recipe)
    if source_config.get("model_type") != "qwen4_exp":
        raise ValueError("Streaming MoE conversion requires a qwen4_exp source")
    model_class = get_quantization_model_class(recipe)
    max_memory = {
        index: args.gpu_memory for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = args.cpu_memory

    started = time.monotonic()
    with use_streaming_moe_linearization(), load_context(model_class):
        model = model_class.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=recipe["source"]["trust_remote_code"],
        )
    model.config.super_quant_linearized_moe = True
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.shard_size,
    )
    tokenizer = load_quantization_tokenizer(recipe)
    tokenizer.save_pretrained(output_dir)
    save_quantization_processor(recipe, output_dir)

    tensor_count, expert_tensor_count = validate_linearized_checkpoint(output_dir)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "source": args.model_id,
        "elapsed_seconds": time.monotonic() - started,
        "tensor_count": tensor_count,
        "expert_2d_tensor_count": expert_tensor_count,
        "files": hashes,
    }
    (output_dir / "linearization-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
