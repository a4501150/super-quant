#!/usr/bin/env python3
"""
Multi-domain calibration dataset builder for GGUF imatrix quantization.

Downloads datasets across 8 categories, tokenizes, and outputs one file per
domain group sized to a token budget. Each sample is kept complete — never
truncated mid-conversation or mid-reasoning chain.
"""

import argparse
import os
import re
import random

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

DOMAIN_TOKEN_TARGETS = {
    "general": 1_000_000,
    "code": 750_000,
    "reasoning": 750_000,
    "agentic": 500_000,
}

CATEGORIES = {
    "chat_en": {
        "description": "Multi-turn chat (English)",
        "sources": [
            {"dataset": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "text_field": "messages"},
            {"dataset": "teknium/OpenHermes-2.5", "split": "train", "text_field": "conversations", "streaming": True, "max_samples": 500000},
        ],
        "domain_group": "general",
    },
    "chat_zh": {
        "description": "Multi-turn chat (Chinese)",
        "sources": [
            {"dataset": "FreedomIntelligence/ShareGPT-CN", "split": "train", "text_field": "conversations"},
            {"dataset": "m-a-p/COIG-CQIA", "config": "zhihu", "split": "train", "text_field": "instruction"},
        ],
        "domain_group": "general",
    },
    "code": {
        "description": "Code (Python/JS/Rust/C++/etc)",
        "sources": [
            {"dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K", "split": "train", "text_field": "instruction,response"},
        ],
        "domain_group": "code",
    },
    "reasoning": {
        "description": "Math / Reasoning / Chain-of-thought",
        "sources": [
            {"dataset": "nvidia/OpenMathInstruct-2", "split": "train", "text_field": "problem,generated_solution", "streaming": True, "max_samples": 500000},
            {"dataset": "open-r1/OpenR1-Math-220k", "split": "train", "text_field": "problem,solution"},
        ],
        "domain_group": "reasoning",
    },
    "tool_calling": {
        "description": "Tool calling / Function calling",
        "sources": [
            {"dataset": "glaiveai/glaive-function-calling-v2", "split": "train", "text_field": "system,chat"},
            {"dataset": "Salesforce/xlam-function-calling-60k", "split": "train", "text_field": "query,answers"},
        ],
        "domain_group": "agentic",
    },
    "agentic": {
        "description": "Agentic workflows (multi-step tool use)",
        "sources": [
            {"dataset": "NousResearch/hermes-function-calling-v1", "split": "train", "text_field": "conversations"},
        ],
        "domain_group": "agentic",
    },
    "long_context": {
        "description": "Long-context passages (2K+ tokens)",
        "sources": [
            {"dataset": "Yukang/LongAlpaca-12k", "split": "train", "text_field": "instruction,output"},
            {"dataset": "emozilla/pg19", "split": "train", "text_field": "text", "streaming": True, "max_samples": 200},
        ],
        "domain_group": "general",
    },
    "prose": {
        "description": "English diverse prose",
        "sources": [
            {"dataset": "froggeric/imatrix", "split": "train", "text_field": "text"},
        ],
        "domain_group": "general",
    },
}


def extract_text(row: dict, text_field: str) -> str | None:
    if "," in text_field:
        parts = []
        for f in text_field.split(","):
            val = row.get(f.strip(), "")
            if val:
                parts.append(str(val) if not isinstance(val, str) else val)
        return "\n\n".join(parts) if parts else None

    val = row.get(text_field)
    if val is None:
        return None

    if isinstance(val, str):
        return val if val.strip() else None

    if isinstance(val, list):
        parts = []
        for msg in val:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if content:
                    parts.append(f"{role}: {content}" if role else str(content))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts) if parts else None

    return str(val)


def load_source(source: dict) -> list[str]:
    ds_name = source["dataset"]
    split = source.get("split", "train")
    text_field = source["text_field"]
    streaming = source.get("streaming", False)
    max_samples = source.get("max_samples")

    print(f"  Loading {ds_name} (split={split})...")

    config = source.get("config")

    try:
        ds = load_dataset(ds_name, config, split=split, streaming=streaming)
    except Exception as e:
        print(f"  WARNING: Failed to load {ds_name}: {e}")
        return []

    texts = []
    count = 0

    for row in tqdm(ds, desc=f"  {ds_name.split('/')[-1]}", leave=False):
        text = extract_text(row, text_field)
        if not text or len(text.strip()) < 50:
            continue
        texts.append(text)
        count += 1
        if max_samples and count >= max_samples:
            break

    print(f"    → {len(texts)} complete samples")
    return texts


def select_to_token_budget(texts: list[str], token_target: int, tokenizer) -> tuple[list[str], int]:
    """Select complete samples up to token budget. Never truncates a sample."""
    random.shuffle(texts)
    selected = []
    total_tokens = 0

    for text in texts:
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        selected.append(text)
        total_tokens += n_tokens
        if total_tokens >= token_target:
            break

    return selected, total_tokens


def build_calibration(output_dir: str, model_id: str, token_targets: dict[str, int], force: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    domain_groups = sorted(set(c["domain_group"] for c in CATEGORIES.values()))
    expected_files = [f"{d}.txt" for d in domain_groups] + ["combined.txt"]

    if not force and all(os.path.exists(os.path.join(output_dir, f)) for f in expected_files):
        sizes = {f: os.path.getsize(os.path.join(output_dir, f)) for f in expected_files}
        if all(s > 1000 for s in sizes.values()):
            print("Calibration files already exist — skipping. Use --force to rebuild.")
            for fname, size in sizes.items():
                print(f"  {fname}: {size / 1024 / 1024:.1f} MB")
            return

    print("=== Building calibration dataset ===")
    print(f"Tokenizer: {model_id}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"Token targets: {token_targets}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print()

    domain_texts: dict[str, list[str]] = {d: [] for d in domain_groups}

    for cat_name, cfg in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"  {cfg['description']}")
        print(f"{'='*60}")

        for source in cfg["sources"]:
            texts = load_source(source)
            domain_texts[cfg["domain_group"]].extend(texts)

    special_tokens = set(tokenizer.all_special_tokens)
    if special_tokens:
        pattern = re.compile("|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)))
        stripped_total = 0
        for domain in domain_texts:
            cleaned = []
            for text in domain_texts[domain]:
                new_text = pattern.sub("", text)
                if new_text != text:
                    stripped_total += 1
                new_text = new_text.strip()
                if len(new_text) >= 50:
                    cleaned.append(new_text)
            domain_texts[domain] = cleaned
        if stripped_total:
            print(f"\n  Stripped special tokens from {stripped_total} samples ({', '.join(sorted(special_tokens))})")

    print(f"\n{'='*60}")
    print("  Selecting samples to token budgets")
    print(f"{'='*60}")

    domain_selected: dict[str, list[str]] = {}
    domain_token_counts: dict[str, int] = {}

    for domain, texts in domain_texts.items():
        target = token_targets.get(domain, 750_000)
        print(f"\n  {domain}: {len(texts)} raw samples, target {target:,} tokens")
        selected, n_tokens = select_to_token_budget(texts, target, tokenizer)
        domain_selected[domain] = selected
        domain_token_counts[domain] = n_tokens
        print(f"    → {len(selected)} samples, {n_tokens:,} tokens")

    for domain, texts in domain_selected.items():
        outpath = os.path.join(output_dir, f"{domain}.txt")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts))
        size_mb = os.path.getsize(outpath) / 1024 / 1024
        n_tokens = domain_token_counts[domain]
        print(f"\nWrote {outpath}: {len(texts)} samples, {n_tokens:,} tokens, {size_mb:.1f} MB")

    all_texts = []
    for texts in domain_selected.values():
        all_texts.extend(texts)
    random.shuffle(all_texts)

    combined_path = os.path.join(output_dir, "combined.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_texts))
    size_mb = os.path.getsize(combined_path) / 1024 / 1024
    total_tokens = sum(domain_token_counts.values())
    print(f"\nWrote {combined_path}: {len(all_texts)} samples, {total_tokens:,} tokens, {size_mb:.1f} MB")

    print("\n=== Calibration dataset ready ===")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".txt"):
            fpath = os.path.join(output_dir, fname)
            size = os.path.getsize(fpath)
            print(f"  {fname}: {size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Build multi-domain calibration dataset for GGUF imatrix")
    parser.add_argument("--output-dir", default="calibration", help="Output directory")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID"),
                        help="Model ID for tokenizer (or set MODEL_ID env var)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output files exist")
    parser.add_argument("--tokens-per-domain", type=int, default=None,
                        help="Override token target for all domains")
    args = parser.parse_args()

    random.seed(args.seed)

    token_targets = dict(DOMAIN_TOKEN_TARGETS)
    if args.tokens_per_domain:
        for domain in token_targets:
            token_targets[domain] = args.tokens_per_domain

    build_calibration(args.output_dir, args.model_id, token_targets, force=args.force)


if __name__ == "__main__":
    main()
