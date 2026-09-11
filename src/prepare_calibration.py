#!/usr/bin/env python3
"""Build reproducible structured calibration data and GGUF text renders."""

import argparse
import functools
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model_utils import CALIBRATION_SCHEMA_VERSION

SCHEMA_VERSION = CALIBRATION_SCHEMA_VERSION
EXTRACTOR_VERSION = 1
RENDER_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_MAX_DOCUMENT_TOKENS = 8192
DEFAULT_HOLDOUT_TOKENS = 25_000
DEFAULT_MAX_SOURCE_RECORDS = 5_000
DEFAULT_MIN_SOURCE_RECORDS = 100
DEFAULT_SOURCE_SHUFFLE_BUFFER = 10_000
MIN_CONTENT_CHARS = 50

DOMAIN_TOKEN_TARGETS = {
    "general": 1_000_000,
    "code": 750_000,
    "reasoning": 750_000,
    "agentic": 500_000,
}

SOURCES = (
    {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "split": "train_sft",
        "category": "chat_en",
        "domain": "general",
        "extractor": "messages",
        "field": "messages",
    },
    {
        "dataset": "teknium/OpenHermes-2.5",
        "revision": "b82037821055c377bed0d495e72e46de3bc72e84",
        "split": "train",
        "category": "chat_en",
        "domain": "general",
        "extractor": "messages",
        "field": "conversations",
        "system_fields": ["system_prompt", "custom_instruction"],
        "streaming": True,
        "max_samples": 5_000,
    },
    {
        "dataset": "FreedomIntelligence/ShareGPT-CN",
        "revision": "9a2dc206b29883ee428f64438a2cacc32cf1ce55",
        "split": "train",
        "category": "chat_zh",
        "domain": "general",
        "extractor": "messages",
        "field": "conversations",
    },
    {
        "dataset": "m-a-p/COIG-CQIA",
        "revision": "8b55868c6168adf86c30e7ca0f782cca1c514297",
        "config": "zhihu",
        "split": "train",
        "category": "chat_zh",
        "domain": "general",
        "extractor": "prompt_response",
        "prompt_fields": ["instruction", "input"],
        "response_fields": ["output", "response", "answer"],
    },
    {
        "dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "revision": "b0079beaa0361d82412520b873715bee59cc7dd4",
        "split": "train",
        "category": "code",
        "domain": "code",
        "extractor": "prompt_response",
        "prompt_fields": ["instruction"],
        "response_fields": ["response"],
    },
    {
        "dataset": "nvidia/OpenMathInstruct-2",
        "revision": "469216e3f46f4dacf476b382e192485ea51a143e",
        "split": "train",
        "category": "reasoning",
        "domain": "reasoning",
        "extractor": "prompt_response",
        "prompt_fields": ["problem"],
        "response_fields": ["generated_solution"],
        "streaming": True,
        "max_samples": 5_000,
    },
    {
        "dataset": "open-r1/OpenR1-Math-220k",
        "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
        "split": "train",
        "category": "reasoning",
        "domain": "reasoning",
        "extractor": "prompt_response",
        "prompt_fields": ["problem"],
        "response_fields": ["solution"],
    },
    {
        "dataset": "glaiveai/glaive-function-calling-v2",
        "revision": "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
        "split": "train",
        "category": "tool_calling",
        "domain": "agentic",
        "extractor": "tagged_tool_chat",
        "system_field": "system",
        "chat_field": "chat",
    },
    {
        "dataset": "Salesforce/xlam-function-calling-60k",
        "revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
        "split": "train",
        "category": "tool_calling",
        "domain": "agentic",
        "extractor": "prompt_response",
        "system_fields": ["tools"],
        "prompt_fields": ["query"],
        "response_fields": ["answers"],
    },
    {
        "dataset": "NousResearch/hermes-function-calling-v1",
        "revision": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
        "split": "train",
        "category": "agentic",
        "domain": "agentic",
        "extractor": "messages",
        "field": "conversations",
        "system_fields": ["tools"],
        "merge_leading_system_messages": True,
    },
    {
        "dataset": "Yukang/LongAlpaca-12k",
        "revision": "46dce924ed8786979556018e191c0f557d8f4aa2",
        "split": "train",
        "category": "long_context",
        "domain": "general",
        "extractor": "prompt_response",
        "prompt_fields": ["instruction", "input"],
        "response_fields": ["output"],
        "max_samples": 1_000,
        "shuffle_buffer": 500,
    },
    {
        "dataset": "emozilla/pg19",
        "revision": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
        "split": "train",
        "category": "long_context",
        "domain": "general",
        "extractor": "document",
        "field": "text",
        "streaming": True,
        "max_samples": 20,
        "minimum_records": 20,
        "shuffle_buffer": 40,
    },
    {
        "dataset": "froggeric/imatrix",
        "revision": "4cb52cf0af4ba963fca22a083efac534d13b873d",
        "split": "train",
        "category": "prose",
        "domain": "general",
        "extractor": "document",
        "field": "text",
    },
)

ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "function": "tool",
    "gpt": "assistant",
    "human": "user",
    "observation": "tool",
    "system": "system",
    "tool": "tool",
    "user": "user",
}
TAGGED_ROLE = re.compile(
    r"(?im)^(?:<\|)?(system|user|human|assistant|bot|gpt|tool|function|observation)"
    r"(?:\|>)?(?:\s+response)?\s*:\s*"
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role not in ROLE_ALIASES:
        raise ValueError(f"unsupported message role: {value!r}")
    return ROLE_ALIASES[role]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True).strip()


def _first_text(row: dict[str, Any], fields: Iterable[str]) -> str:
    return next((text for field in fields if (text := _text(row.get(field)))), "")


def _system_messages(
    row: dict[str, Any], source: dict[str, Any]
) -> list[dict[str, str]]:
    content = "\n\n".join(
        text
        for field in source.get("system_fields", ())
        if (text := _text(row.get(field)))
    )
    return [{"role": "system", "content": content}] if content else []


def _merge_leading_system_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    system_count = 0
    for message in messages:
        if message["role"] != "system":
            break
        system_count += 1
    if system_count <= 1:
        return messages
    content = "\n\n".join(message["content"] for message in messages[:system_count])
    return [{"role": "system", "content": content}, *messages[system_count:]]


def extract_messages(
    row: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any] | None:
    raw_messages = row.get(source["field"])
    if not isinstance(raw_messages, list):
        raise TypeError(f"{source['field']} is not a message list")
    messages = _system_messages(row, source)
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise TypeError(f"invalid message value: {raw!r}")
        content = _first_text(raw, ("content", "value", "text"))
        if not content:
            continue
        role_value = next(
            (raw.get(field) for field in ("role", "from", "speaker") if raw.get(field)),
            None,
        )
        messages.append({"role": normalize_role(role_value), "content": content})
    if source.get("merge_leading_system_messages"):
        messages = _merge_leading_system_messages(messages)
    return {"kind": "messages", "messages": messages} if messages else None


def extract_prompt_response(
    row: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any] | None:
    prompt_parts = [
        text for field in source["prompt_fields"] if (text := _text(row.get(field)))
    ]
    response = _first_text(row, source["response_fields"])
    if not prompt_parts or not response:
        return None
    return {
        "kind": "messages",
        "messages": [
            *_system_messages(row, source),
            {"role": "user", "content": "\n\n".join(prompt_parts)},
            {"role": "assistant", "content": response},
        ],
    }


def extract_document(
    row: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any] | None:
    text = _text(row.get(source["field"]))
    return {"kind": "document", "text": text} if text else None


def extract_tagged_tool_chat(
    row: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any] | None:
    messages = []
    system = _text(row.get(source["system_field"]))
    if system:
        messages.append({"role": "system", "content": system})
    chat = _text(row.get(source["chat_field"]))
    matches = list(TAGGED_ROLE.finditer(chat))
    if not matches:
        raise ValueError("tagged tool chat has no recognized role markers")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
        content = chat[match.end() : end].strip()
        if content:
            messages.append(
                {"role": normalize_role(match.group(1)), "content": content}
            )
    return {"kind": "messages", "messages": messages} if messages else None


EXTRACTORS = {
    "document": extract_document,
    "messages": extract_messages,
    "prompt_response": extract_prompt_response,
    "tagged_tool_chat": extract_tagged_tool_chat,
}


@functools.cache
def _special_token_pattern(tokens: tuple[str, ...]) -> re.Pattern[str] | None:
    if not tokens:
        return None
    return re.compile("|".join(re.escape(token) for token in tokens))


def strip_special_tokens(record: dict[str, Any], special_tokens: Iterable[str]) -> bool:
    tokens = tuple(
        sorted((token for token in special_tokens if token), key=len, reverse=True)
    )
    pattern = _special_token_pattern(tokens)
    if pattern is None:
        return False
    changed = False
    fields = record["messages"] if record["kind"] == "messages" else [record]
    key = "content" if record["kind"] == "messages" else "text"
    for field in fields:
        cleaned = pattern.sub("", field[key]).strip()
        changed |= cleaned != field[key]
        field[key] = cleaned
    return changed


def canonical_payload(record: dict[str, Any]) -> str:
    if record["kind"] == "document":
        content = record["text"]
    else:
        content = "\n".join(
            f"{message['role']}\0{message['content']}" for message in record["messages"]
        )
    return unicodedata.normalize("NFKC", content).strip()


def stable_record_id(record: dict[str, Any], segment: int = 0) -> str:
    source = record["source"]
    identity = {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "config": source.get("config"),
        "split": source["split"],
        "segment": segment,
        "payload": canonical_payload(record),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return digest[:24]


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _record_token_count(record: dict[str, Any], tokenizer) -> int:
    return record.get("_rendered_tokens") or _token_count(
        tokenizer, render_record(record)
    )


def _split_units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units = []
    for paragraph in paragraphs:
        units.extend(
            part.strip() for part in SENTENCE_BOUNDARY.split(paragraph) if part.strip()
        )
    return units


def segment_document(
    record: dict[str, Any], tokenizer, max_tokens: int
) -> list[dict[str, Any]]:
    if record["kind"] != "document":
        record["id"] = stable_record_id(record)
        return [record]
    token_count = _token_count(tokenizer, record["text"])
    if token_count <= max_tokens:
        record["id"] = stable_record_id(record)
        record["_rendered_tokens"] = token_count
        return [record]
    units = _split_units(record["text"])
    segments: list[str] = []
    current: list[str] = []
    current_tokens = 0
    separator_tokens = _token_count(tokenizer, "\n\n")
    for unit in units:
        token_ids = tokenizer.encode(unit, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            if current:
                segments.append("\n\n".join(current))
                current = []
                current_tokens = 0
            segments.extend(
                tokenizer.decode(token_ids[start : start + max_tokens]).strip()
                for start in range(0, len(token_ids), max_tokens)
            )
        elif (
            current and current_tokens + separator_tokens + len(token_ids) > max_tokens
        ):
            segments.append("\n\n".join(current))
            current = [unit]
            current_tokens = len(token_ids)
        else:
            current.append(unit)
            current_tokens += len(token_ids) + (
                separator_tokens if current_tokens else 0
            )
    if current:
        segments.append("\n\n".join(current))
    output = []
    for index, text in enumerate(segments):
        child = {**record, "text": text}
        child["id"] = stable_record_id(child, index)
        child["_rendered_tokens"] = _token_count(tokenizer, text)
        output.append(child)
    return output


def _near_key(text: str) -> str:
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _near_fingerprint(text: str) -> frozenset[int]:
    words = _near_key(text).split()
    count = max(1, len(words) - 3)
    if count <= 128:
        indices = range(count)
    else:
        indices = {index * (count - 1) // 127 for index in range(128)}
    hashes = {
        int.from_bytes(
            hashlib.blake2b(
                " ".join(words[index : index + 4]).encode(), digest_size=8
            ).digest()
        )
        for index in indices
    }
    return frozenset(sorted(hashes)[:16])


def deduplicate_records(
    records: Iterable[dict[str, Any]], near_similarity: float = 0.8
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted = []
    exact_seen: set[bytes] = set()
    buckets: dict[int, list[int]] = defaultdict(list)
    fingerprints: list[frozenset[int]] = []
    removed: dict[str, Any] = {
        "exact": 0,
        "near": 0,
        "by_source": defaultdict(lambda: {"exact": 0, "near": 0}),
    }
    for record in records:
        source_name = record["source"]["dataset"]
        normalized = _near_key(canonical_payload(record))
        exact = hashlib.sha256(normalized.encode()).digest()
        if exact in exact_seen:
            removed["exact"] += 1
            removed["by_source"][source_name]["exact"] += 1
            continue
        fingerprint = _near_fingerprint(normalized)
        candidates = {index for feature in fingerprint for index in buckets[feature]}
        if any(
            len(fingerprint & fingerprints[index])
            / min(len(fingerprint), len(fingerprints[index]))
            >= near_similarity
            for index in candidates
        ):
            removed["near"] += 1
            removed["by_source"][source_name]["near"] += 1
            continue
        exact_seen.add(exact)
        index = len(fingerprints)
        fingerprints.append(fingerprint)
        for feature in fingerprint:
            buckets[feature].append(index)
        accepted.append(record)
    removed["by_source"] = dict(sorted(removed["by_source"].items()))
    return accepted, removed


def render_record(record: dict[str, Any]) -> str:
    if record["kind"] == "document":
        return record["text"].strip()
    return "\n\n".join(
        f"{message['role'].capitalize()}:\n{message['content'].strip()}"
        for message in record["messages"]
    )


def _take_to_token_budget(
    records: Iterable[dict[str, Any]], token_target: int, tokenizer
) -> tuple[list[dict[str, Any]], int]:
    if token_target < 0:
        raise ValueError("token target cannot be negative")
    if token_target == 0:
        return [], 0
    selected = []
    total = 0
    for record in records:
        selected.append(record)
        total += _record_token_count(record, tokenizer)
        if total >= token_target:
            return selected, total
    raise ValueError(f"only {total:,} tokens available for target {token_target:,}")


def select_to_token_budget(
    records: Iterable[dict[str, Any]], token_target: int, tokenizer, seed: int
) -> tuple[list[dict[str, Any]], int]:
    candidates = sorted(records, key=lambda record: record["id"])
    random.Random(seed).shuffle(candidates)
    return _take_to_token_budget(candidates, token_target, tokenizer)


def select_holdout(
    records: Iterable[dict[str, Any]], token_target: int, tokenizer, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    candidates = sorted(records, key=lambda record: record["id"])
    random.Random(seed).shuffle(candidates)
    holdout, total = _take_to_token_budget(
        reversed(candidates), token_target, tokenizer
    )
    holdout_ids = {record["id"] for record in holdout}
    training = [record for record in candidates if record["id"] not in holdout_ids]
    return training, holdout, total


def extract_record(
    row: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any] | None:
    payload = EXTRACTORS[source["extractor"]](row, source)
    if payload is None:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": source["domain"],
        "category": source["category"],
        "source": {
            key: source[key]
            for key in ("dataset", "revision", "config", "split")
            if key in source
        },
        **payload,
    }


def _source_cache_policy(
    source: dict[str, Any], cache_context: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "source": source,
        **cache_context,
    }


def _source_cache_paths(cache_dir: Path, policy: dict[str, Any]) -> tuple[Path, Path]:
    key = hashlib.sha256(
        json.dumps(policy, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return cache_dir / f"{key}.jsonl", cache_dir / f"{key}.manifest.json"


def _load_source_cache(
    cache_dir: Path, policy: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    data_path, manifest_path = _source_cache_paths(cache_dir, policy)
    try:
        data = data_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["policy"] != policy or manifest["sha256"] != _sha256(data):
            return None
        records = [json.loads(line) for line in data.splitlines() if line]
        if len(records) != manifest["stats"]["segments"]:
            return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return records, manifest["stats"]


def _write_source_cache(
    cache_dir: Path,
    policy: dict[str, Any],
    records: list[dict[str, Any]],
    stats: dict[str, int],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, manifest_path = _source_cache_paths(cache_dir, policy)
    data = _jsonl_bytes(records)
    temporary_data = None
    temporary_manifest = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".source.", dir=cache_dir, delete=False
        ) as temporary:
            temporary.write(data)
            temporary_data = Path(temporary.name)
        temporary_data.replace(data_path)
        manifest = {"policy": policy, "stats": stats, "sha256": _sha256(data)}
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".source-manifest.", dir=cache_dir, delete=False
        ) as temporary:
            temporary.write(_json_bytes(manifest))
            temporary_manifest = Path(temporary.name)
        temporary_manifest.replace(manifest_path)
    finally:
        for path in (temporary_data, temporary_manifest):
            if path is not None:
                path.unlink(missing_ok=True)


def load_source(
    source: dict[str, Any],
    tokenizer,
    special_tokens: Iterable[str],
    max_document_tokens: int,
    *,
    cache_dir: Path | None = None,
    cache_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    special_tokens = tuple(special_tokens)
    max_samples = source.get("max_samples", DEFAULT_MAX_SOURCE_RECORDS)
    minimum_records = source.get("minimum_records", DEFAULT_MIN_SOURCE_RECORDS)
    if max_samples and max_samples < minimum_records:
        raise ValueError(
            f"source {source['dataset']} caps accepted records at {max_samples}, "
            f"below its minimum of {minimum_records}"
        )
    if (cache_dir is None) != (cache_context is None):
        raise ValueError("cache directory and context must be provided together")
    if cache_context is not None and (
        cache_context.get("special_tokens") != sorted(set(special_tokens))
        or cache_context.get("max_document_tokens") != max_document_tokens
    ):
        raise ValueError("source cache context does not match extraction parameters")
    cache_policy = None
    if cache_dir is not None and cache_context is not None:
        cache_policy = _source_cache_policy(source, cache_context)
        if cached := _load_source_cache(cache_dir, cache_policy):
            records, stats = cached
            print(
                f"  Using cached {source['dataset']} "
                f"({source['split']}@{source['revision'][:12]}): "
                f"{stats['accepted']:,} records, {stats['segments']:,} segments, "
                f"{stats['tokens']:,} tokens"
            )
            return records, stats

    print(
        f"  Loading {source['dataset']} ({source['split']}@{source['revision'][:12]})..."
    )
    try:
        dataset = load_dataset(
            source["dataset"],
            source.get("config"),
            split=source["split"],
            revision=source["revision"],
            streaming=source.get("streaming", True),
        )
    except Exception as error:
        raise RuntimeError(
            f"failed to load required source {source['dataset']}: {error}"
        ) from error

    if hasattr(dataset, "shuffle"):
        source_seed = source.get(
            "seed",
            cache_context["seed"] if cache_context is not None else DEFAULT_SEED,
        )
        dataset = dataset.shuffle(
            seed=source_seed,
            buffer_size=source.get("shuffle_buffer", DEFAULT_SOURCE_SHUFFLE_BUFFER),
        )
    records = []
    stats = {"rows": 0, "accepted": 0, "empty": 0, "special_tokens_stripped": 0}
    for row in tqdm(dataset, desc=f"  {source['dataset'].split('/')[-1]}", leave=False):
        stats["rows"] += 1
        try:
            record = extract_record(row, source)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"schema mismatch in {source['dataset']} row {stats['rows']}: {error}"
            ) from error
        if record is None or len(canonical_payload(record)) < MIN_CONTENT_CHARS:
            stats["empty"] += 1
            continue
        if strip_special_tokens(record, special_tokens):
            stats["special_tokens_stripped"] += 1
        if len(canonical_payload(record)) < MIN_CONTENT_CHARS:
            stats["empty"] += 1
            continue
        records.extend(segment_document(record, tokenizer, max_document_tokens))
        stats["accepted"] += 1
        if max_samples and stats["accepted"] >= max_samples:
            break
    if stats["accepted"] < minimum_records:
        raise ValueError(
            f"required source {source['dataset']} produced {stats['accepted']} valid "
            f"records; minimum is {minimum_records}"
        )
    stats["segments"] = len(records)
    for record in records:
        record["_rendered_tokens"] = _token_count(tokenizer, render_record(record))
    stats["tokens"] = sum(record["_rendered_tokens"] for record in records)
    print(
        f"    {stats['accepted']:,} records, {stats['segments']:,} segments, "
        f"{stats['tokens']:,} tokens"
    )
    if cache_dir is not None and cache_policy is not None:
        _write_source_cache(cache_dir, cache_policy, records, stats)
    return records, stats


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(record) for record in records)


def _render_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return (
        "\n\n---\n\n".join(render_record(record) for record in records) + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_policy(
    model_id: str,
    tokenizer_revision: str | None,
    token_targets: dict[str, int],
    holdout_tokens: int,
    max_document_tokens: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "render_version": RENDER_VERSION,
        "model_id": model_id,
        "tokenizer_revision": tokenizer_revision,
        "token_targets": token_targets,
        "holdout_tokens_per_domain": holdout_tokens,
        "max_document_tokens": max_document_tokens,
        "seed": seed,
        "default_max_source_records": DEFAULT_MAX_SOURCE_RECORDS,
        "default_min_source_records": DEFAULT_MIN_SOURCE_RECORDS,
        "default_source_shuffle_buffer": DEFAULT_SOURCE_SHUFFLE_BUFFER,
        "default_streaming": True,
        "sources": list(SOURCES),
    }


def _valid_existing_build(output_dir: Path, policy: dict[str, Any]) -> bool:
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["policy"] != policy:
            return False
        for name, metadata in manifest["files"].items():
            data = (output_dir / name).read_bytes()
            if len(data) != metadata["bytes"] or _sha256(data) != metadata["sha256"]:
                return False
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError):
        return False
    return True


def build_calibration(
    output_dir: str | os.PathLike[str],
    model_id: str,
    token_targets: dict[str, int],
    *,
    force: bool = False,
    seed: int = DEFAULT_SEED,
    holdout_tokens: int = DEFAULT_HOLDOUT_TOKENS,
    max_document_tokens: int = DEFAULT_MAX_DOCUMENT_TOKENS,
) -> dict[str, Any] | None:
    if not model_id:
        raise ValueError("model ID is required")
    output_path = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    policy = _build_policy(
        model_id,
        tokenizer_revision,
        token_targets,
        holdout_tokens,
        max_document_tokens,
        seed,
    )
    if not force and _valid_existing_build(output_path, policy):
        print(f"Calibration build is current: {output_path}")
        return None

    source_cache_dir = output_path.parent / ".source-cache"
    cache_context = {
        key: value
        for key, value in policy.items()
        if key
        not in {
            "schema_version",
            "extractor_version",
            "token_targets",
            "holdout_tokens_per_domain",
            "sources",
        }
    }
    cache_context["special_tokens"] = sorted(set(tokenizer.all_special_tokens))
    all_records = []
    source_stats = []
    for source in SOURCES:
        records, stats = load_source(
            source,
            tokenizer,
            tokenizer.all_special_tokens,
            max_document_tokens,
            cache_dir=source_cache_dir,
            cache_context=cache_context,
        )
        all_records.extend(records)
        source_stats.append(
            {
                "source": {
                    key: source[key]
                    for key in ("dataset", "revision", "config", "split")
                    if key in source
                },
                "stats": stats,
            }
        )

    records, dedup_stats = deduplicate_records(all_records)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[record["domain"]].append(record)
    unexpected_domains = set(by_domain) - set(token_targets)
    if unexpected_domains:
        raise ValueError(
            "No token target configured for domains: "
            + ", ".join(sorted(unexpected_domains))
        )

    selected_by_domain = {}
    holdout_by_domain = {}
    domain_stats = {}
    for index, (domain, target) in enumerate(sorted(token_targets.items())):
        try:
            training, holdout, holdout_count = select_holdout(
                by_domain[domain], holdout_tokens, tokenizer, seed + index
            )
        except ValueError as error:
            raise ValueError(f"{domain} holdout selection failed: {error}") from error
        try:
            selected, selected_count = select_to_token_budget(
                training, target, tokenizer, seed + 1000 + index
            )
        except ValueError as error:
            raise ValueError(
                f"{domain} calibration selection failed: {error}"
            ) from error
        selected_by_domain[domain] = selected
        holdout_by_domain[domain] = holdout
        domain_stats[domain] = {
            "available_records": len(by_domain[domain]),
            "selected_records": len(selected),
            "selected_tokens": selected_count,
            "holdout_records": len(holdout),
            "holdout_tokens": holdout_count,
        }

    combined = [
        record
        for domain in sorted(selected_by_domain)
        for record in selected_by_domain[domain]
    ]
    random.Random(seed).shuffle(combined)
    holdout = [
        record
        for domain in sorted(holdout_by_domain)
        for record in holdout_by_domain[domain]
    ]
    random.Random(seed + 1).shuffle(holdout)
    if {record["id"] for record in combined} & {record["id"] for record in holdout}:
        raise RuntimeError("calibration and holdout records overlap")
    for record in (*combined, *holdout):
        record.pop("_rendered_tokens", None)
    del all_records, records, by_domain

    artifacts: dict[str, bytes] = {}
    for domain, domain_records in selected_by_domain.items():
        artifacts[f"{domain}.jsonl"] = _jsonl_bytes(domain_records)
        artifacts[f"{domain}.txt"] = _render_bytes(domain_records)
    artifacts["combined.jsonl"] = _jsonl_bytes(combined)
    artifacts["combined.txt"] = _render_bytes(combined)
    artifacts["holdout.jsonl"] = _jsonl_bytes(holdout)
    artifacts["holdout.txt"] = _render_bytes(holdout)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "tokenizer": {"model_id": model_id, "revision": tokenizer_revision},
        "sources": source_stats,
        "deduplication": dedup_stats,
        "domains": domain_stats,
        "files": {
            name: {"bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(artifacts.items())
        },
    }
    artifacts["manifest.json"] = _json_bytes(manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        for name, data in artifacts.items():
            (stage / name).write_bytes(data)
        if output_path.exists():
            backup_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_path.name}.backup.", dir=output_path.parent
                )
            )
            backup = backup_root / output_path.name
            output_path.replace(backup)
            try:
                stage.replace(output_path)
            except BaseException:
                backup.replace(output_path)
                shutil.rmtree(backup_root)
                raise
            shutil.rmtree(backup_root)
        else:
            stage.replace(output_path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    print(f"Wrote reproducible calibration build: {output_path}")
    for domain, stats in domain_stats.items():
        print(
            f"  {domain}: {stats['selected_records']:,} records, "
            f"{stats['selected_tokens']:,} calibration tokens, "
            f"{stats['holdout_tokens']:,} holdout tokens"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="calibration")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tokens-per-domain", type=int)
    parser.add_argument(
        "--holdout-tokens-per-domain", type=int, default=DEFAULT_HOLDOUT_TOKENS
    )
    parser.add_argument(
        "--max-document-tokens", type=int, default=DEFAULT_MAX_DOCUMENT_TOKENS
    )
    args = parser.parse_args()

    token_targets = dict(DOMAIN_TOKEN_TARGETS)
    if args.tokens_per_domain:
        token_targets = dict.fromkeys(token_targets, args.tokens_per_domain)
    build_calibration(
        args.output_dir,
        args.model_id,
        token_targets,
        force=args.force,
        seed=args.seed,
        holdout_tokens=args.holdout_tokens_per_domain,
        max_document_tokens=args.max_document_tokens,
    )


if __name__ == "__main__":
    main()
