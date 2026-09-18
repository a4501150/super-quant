#!/usr/bin/env python3
"""Build reproducible tokenizer-neutral corpora and model calibration renders.

The build is split into two layers:

1. A tokenizer-free parent corpus (``build_corpus``) that pins the source set
   and stores complete, unsplit records per domain. It carries no model,
   tokenizer, or token-count data, so one parent can serve every target model.
2. A model-specific build (``build_calibration``) that loads a validated
   parent corpus, applies the target tokenizer, segments documents, and emits
   deduplicated, budgeted calibration and holdout artifacts.
"""

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
CORPUS_SCHEMA_VERSION = 1
EXTRACTOR_VERSION = 1
RENDER_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_MAX_DOCUMENT_TOKENS = 8192
DEFAULT_HOLDOUT_TOKENS = 25_000
DEFAULT_MAX_SOURCE_RECORDS = 5_000
DEFAULT_MIN_SOURCE_RECORDS = 100
DEFAULT_SOURCE_SHUFFLE_BUFFER = 10_000
MIN_CONTENT_CHARS = 50
CORPUS_SOURCE_CACHE_DIRNAME = ".corpus-source-cache"
SOURCE_KEYS = ("dataset", "revision", "config", "split")

# Explicit, versioned cleanup policy for the parent corpus. Fixed source
# control tokens that must never reach a parent record go in "stripped_tokens";
# the list is empty, so the parent preserves source text verbatim and all
# tokenizer-derived special-token stripping happens in the model build.
CORPUS_CLEANUP_POLICY: dict[str, Any] = {"version": 1, "stripped_tokens": []}

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


class CorpusError(RuntimeError):
    """Raised when a present corpus directory fails validation."""


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


def stable_record_id(
    record: dict[str, Any], segment: int = 0, payload: str | None = None
) -> str:
    """Id over the record's source identity and canonical payload.

    Pass an already-normalized ``payload`` to avoid re-normalizing a record
    that the caller has just canonicalized."""
    source = record["source"]
    identity = {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "config": source.get("config"),
        "split": source["split"],
        "segment": segment,
        "payload": canonical_payload(record) if payload is None else payload,
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
        "source": _source_descriptor(source),
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
        if len(records) != manifest["stats"]["records"]:
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
    *,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract one source into complete, unsplit parent-corpus records.

    No tokenizer is involved: extraction, shuffle seeds, caps, and minimums
    come only from the pinned source definition and the corpus-level defaults,
    so the per-source cache under ``corpus_dir.parent/.corpus-source-cache``
    is model-neutral and shared by every target model.
    """
    max_samples = source.get("max_samples", DEFAULT_MAX_SOURCE_RECORDS)
    minimum_records = source.get("minimum_records", DEFAULT_MIN_SOURCE_RECORDS)
    if max_samples and max_samples < minimum_records:
        raise ValueError(
            f"source {source['dataset']} caps accepted records at {max_samples}, "
            f"below its minimum of {minimum_records}"
        )
    cache_policy = None
    if cache_dir is not None:
        cache_policy = _source_cache_policy(source, _corpus_cache_context(seed))
        if cached := _load_source_cache(cache_dir, cache_policy):
            records, stats = cached
            print(
                f"  Using cached {source['dataset']} "
                f"({source['split']}@{source['revision'][:12]}): "
                f"{stats['accepted']:,} records, "
                f"{stats['characters']:,} characters"
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
        source_seed = source.get("seed", seed)
        dataset = dataset.shuffle(
            seed=source_seed,
            buffer_size=source.get("shuffle_buffer", DEFAULT_SOURCE_SHUFFLE_BUFFER),
        )
    cleanup_tokens = CORPUS_CLEANUP_POLICY["stripped_tokens"]
    records = []
    stats = {"rows": 0, "accepted": 0, "empty": 0, "cleanup_stripped": 0}
    characters = 0
    for row in tqdm(dataset, desc=f"  {source['dataset'].split('/')[-1]}", leave=False):
        stats["rows"] += 1
        try:
            record = extract_record(row, source)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"schema mismatch in {source['dataset']} row {stats['rows']}: {error}"
            ) from error
        if record is None:
            stats["empty"] += 1
            continue
        if cleanup_tokens and strip_special_tokens(record, cleanup_tokens):
            stats["cleanup_stripped"] += 1
        # One normalization covers the minimum-length check, the stable id,
        # and the character stat for this record.
        payload = canonical_payload(record)
        if len(payload) < MIN_CONTENT_CHARS:
            stats["empty"] += 1
            continue
        # Parent records stay complete and unsplit; ids cover the whole
        # payload so they are stable regardless of any target tokenizer.
        record["id"] = stable_record_id(record, payload=payload)
        records.append(record)
        characters += len(payload)
        stats["accepted"] += 1
        if max_samples and stats["accepted"] >= max_samples:
            break
    if stats["accepted"] < minimum_records:
        raise ValueError(
            f"required source {source['dataset']} produced {stats['accepted']} valid "
            f"records; minimum is {minimum_records}"
        )
    stats["records"] = len(records)
    stats["characters"] = characters
    print(
        f"    {stats['accepted']:,} records, {stats['characters']:,} characters"
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


def _corpus_cache_context(seed: int) -> dict[str, Any]:
    """Policy fields shared by the per-source extract caches and the corpus
    manifest policy, so a cache entry is invalidated exactly when the
    corpus-level extraction inputs change."""
    return {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "cleanup_policy": CORPUS_CLEANUP_POLICY,
        "seed": seed,
    }


def _corpus_policy(seed: int) -> dict[str, Any]:
    """Tokenizer-neutral policy for the parent corpus, keyed independently
    of any model, tokenizer, or document-token configuration."""
    return {
        **_corpus_cache_context(seed),
        "default_max_source_records": DEFAULT_MAX_SOURCE_RECORDS,
        "default_min_source_records": DEFAULT_MIN_SOURCE_RECORDS,
        "default_source_shuffle_buffer": DEFAULT_SOURCE_SHUFFLE_BUFFER,
        "default_streaming": True,
        "sources": list(SOURCES),
    }


def _corpus_domains() -> tuple[str, ...]:
    return tuple(dict.fromkeys(source["domain"] for source in SOURCES))


def _source_descriptor(source: dict[str, Any]) -> dict[str, Any]:
    """The pinned subset of a source definition that identifies its records."""
    return {key: source[key] for key in SOURCE_KEYS if key in source}


def _source_identity(source: dict[str, Any]) -> tuple[Any, ...]:
    descriptor = _source_descriptor(source)
    return tuple(descriptor.get(key) for key in SOURCE_KEYS)


def _validate_corpus_record(record: Any, domain: str, location: str) -> None:
    if not isinstance(record, dict):
        raise CorpusError(f"corpus record {location} is not an object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise CorpusError(
            f"corpus record {location} has unsupported schema "
            f"{record.get('schema_version')!r}"
        )
    if record.get("domain") != domain:
        raise CorpusError(
            f"corpus record {location} has domain {record.get('domain')!r}, "
            f"expected {domain!r}"
        )
    if not isinstance(record.get("category"), str) or not record["category"]:
        raise CorpusError(f"corpus record {location} is missing its category")
    if not isinstance(record.get("source"), dict):
        raise CorpusError(f"corpus record {location} is missing its source")
    if "_rendered_tokens" in record:
        raise CorpusError(f"corpus record {location} carries tokenizer token counts")
    kind = record.get("kind")
    if kind == "document":
        if not isinstance(record.get("text"), str) or not record["text"]:
            raise CorpusError(f"corpus record {location} has no document text")
    elif kind == "messages":
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages or not all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and message["role"]
            and isinstance(message.get("content"), str)
            and message["content"]
            for message in messages
        ):
            raise CorpusError(f"corpus record {location} has invalid messages")
    else:
        raise CorpusError(f"corpus record {location} has invalid kind {kind!r}")
    if record.get("id") != stable_record_id(record):
        raise CorpusError(f"corpus record {location} id does not match its payload")


def _validated_corpus_manifest(
    corpus_dir: Path, seed: int
) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    """Read and integrity-check a corpus, reading each file exactly once.

    Returns the parsed manifest, the raw manifest bytes, and the verified
    per-domain JSONL blobs keyed by file name."""
    try:
        data = (corpus_dir / "manifest.json").read_bytes()
        manifest = json.loads(data)
        policy = manifest["policy"]
        files = manifest["files"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise CorpusError(
            f"invalid corpus manifest in {corpus_dir}: {error}"
        ) from error
    if policy != _corpus_policy(seed):
        raise CorpusError(
            f"corpus policy mismatch for {corpus_dir}; rebuild it with "
            "--force-corpus"
        )
    expected_files = {f"{domain}.jsonl" for domain in _corpus_domains()}
    if set(files) != expected_files:
        raise CorpusError(
            f"corpus {corpus_dir} file set {sorted(files)} does not match the "
            f"expected {sorted(expected_files)}"
        )
    blobs: dict[str, bytes] = {}
    for name, metadata in files.items():
        try:
            blob = (corpus_dir / name).read_bytes()
            expected_bytes = metadata["bytes"]
            expected_sha256 = metadata["sha256"]
        except (KeyError, OSError, TypeError) as error:
            raise CorpusError(
                f"corpus file metadata for {corpus_dir / name} is invalid"
            ) from error
        if len(blob) != expected_bytes or _sha256(blob) != expected_sha256:
            raise CorpusError(f"corpus file {corpus_dir / name} failed integrity checks")
        blobs[name] = blob
    return manifest, data, blobs


def _corpus_records(
    dir_path: Path, blobs: dict[str, bytes]
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Validate manifest-verified domain blobs record by record, yielding
    (pinned-source index, record) pairs in stored order."""
    source_order = {
        _source_identity(source): index for index, source in enumerate(SOURCES)
    }
    for domain in _corpus_domains():
        name = f"{domain}.jsonl"
        for number, line in enumerate(blobs[name].splitlines(), 1):
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise CorpusError(
                    f"corpus file {dir_path / name}:{number}: invalid JSON"
                ) from error
            _validate_corpus_record(record, domain, f"{name}:{number}")
            identity = _source_identity(record["source"])
            if identity not in source_order:
                raise CorpusError(
                    f"corpus contains an unknown source: {identity}"
                )
            yield source_order[identity], record


def load_corpus(
    corpus_dir: str | os.PathLike[str], *, seed: int = DEFAULT_SEED
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Validate a parent corpus and load its records.

    Returns the manifest, the records in source order, and the manifest's
    SHA-256. A present-but-invalid corpus raises ``CorpusError``.
    """
    dir_path = Path(corpus_dir)
    manifest, manifest_bytes, blobs = _validated_corpus_manifest(dir_path, seed)
    entries = sorted(_corpus_records(dir_path, blobs), key=lambda entry: entry[0])
    records = [record for _index, record in entries]
    return manifest, records, _sha256(manifest_bytes)


def load_corpus_artifacts(
    corpus_dir: str | os.PathLike[str], *, seed: int = DEFAULT_SEED
) -> tuple[dict[str, Any], str, dict[str, bytes]]:
    """Validate a parent corpus without retaining its parsed records.

    Returns the manifest, the manifest's SHA-256, and the verified payload
    bytes — the manifest and every domain JSONL file, keyed by file name —
    ready to publish or package. Every corpus file is read exactly once and
    every record is validated before returning.
    """
    dir_path = Path(corpus_dir)
    manifest, manifest_bytes, blobs = _validated_corpus_manifest(dir_path, seed)
    for _entry in _corpus_records(dir_path, blobs):
        pass
    payload = {**blobs, "manifest.json": manifest_bytes}
    return manifest, _sha256(manifest_bytes), payload


def build_corpus(
    corpus_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any] | None:
    """Build the tokenizer-free parent corpus unless it is already current.

    A present-but-invalid corpus raises; use ``force`` to rebuild it.
    """
    dir_path = Path(corpus_dir)
    if dir_path.exists() and not force:
        load_corpus(dir_path, seed=seed)
        print(f"Corpus build is current: {dir_path}")
        return None

    policy = _corpus_policy(seed)
    cache_dir = dir_path.parent / CORPUS_SOURCE_CACHE_DIRNAME
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_stats = []
    for source in SOURCES:
        records, stats = load_source(source, seed=seed, cache_dir=cache_dir)
        by_domain[source["domain"]].extend(records)
        source_stats.append({"source": _source_descriptor(source), "stats": stats})

    artifacts: dict[str, bytes] = {}
    for domain in _corpus_domains():
        artifacts[f"{domain}.jsonl"] = _jsonl_bytes(by_domain.get(domain, []))
    manifest = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "policy": policy,
        "sources": source_stats,
        "files": {
            name: {"bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(artifacts.items())
        },
    }
    artifacts["manifest.json"] = _json_bytes(manifest)

    _stage_and_publish(dir_path, artifacts)
    print(f"Wrote tokenizer-neutral corpus build: {dir_path}")
    for domain in _corpus_domains():
        print(f"  {domain}: {len(by_domain.get(domain, [])):,} records")
    return manifest


def _publish_directory(stage: Path, destination: Path) -> None:
    if destination.exists():
        backup_root = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.backup.", dir=destination.parent)
        )
        backup = backup_root / destination.name
        destination.replace(backup)
        try:
            stage.replace(destination)
        except BaseException:
            backup.replace(destination)
            shutil.rmtree(backup_root)
            raise
        shutil.rmtree(backup_root)
    else:
        stage.replace(destination)


def _stage_and_publish(destination: Path, artifacts: dict[str, bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for name, data in artifacts.items():
            (stage / name).write_bytes(data)
        _publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _build_policy(
    model_id: str,
    tokenizer_revision: str | None,
    token_targets: dict[str, int],
    holdout_tokens: int,
    max_document_tokens: int,
    seed: int,
    corpus_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "render_version": RENDER_VERSION,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "model_id": model_id,
        "tokenizer_revision": tokenizer_revision,
        "token_targets": token_targets,
        "holdout_tokens_per_domain": holdout_tokens,
        "max_document_tokens": max_document_tokens,
        "seed": seed,
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


def _target_tokenizer_records(
    parent_records: list[dict[str, Any]], tokenizer, max_document_tokens: int
) -> tuple[list[dict[str, Any]], int]:
    """Adapt parent records to the target tokenizer: strip special tokens,
    drop records that became too short, segment documents, and recompute ids
    and rendered token counts for the possibly rewritten payloads."""
    dropped_short = 0
    records: list[dict[str, Any]] = []
    for record in parent_records:
        strip_special_tokens(record, tokenizer.all_special_tokens)
        if len(canonical_payload(record)) < MIN_CONTENT_CHARS:
            dropped_short += 1
            continue
        if record["kind"] == "document":
            records.extend(segment_document(record, tokenizer, max_document_tokens))
        else:
            record["id"] = stable_record_id(record)
            record["_rendered_tokens"] = _token_count(
                tokenizer, render_record(record)
            )
            records.append(record)
    return records, dropped_short


def build_calibration(
    output_dir: str | os.PathLike[str],
    model_id: str,
    token_targets: dict[str, int],
    corpus_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    seed: int = DEFAULT_SEED,
    holdout_tokens: int = DEFAULT_HOLDOUT_TOKENS,
    max_document_tokens: int = DEFAULT_MAX_DOCUMENT_TOKENS,
) -> dict[str, Any] | None:
    if not model_id:
        raise ValueError("model ID is required")
    output_path = Path(output_dir)
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        build_corpus(corpus_path, seed=seed)
    corpus_manifest, parent_records, corpus_digest = load_corpus(
        corpus_path, seed=seed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    policy = _build_policy(
        model_id,
        tokenizer_revision,
        token_targets,
        holdout_tokens,
        max_document_tokens,
        seed,
        corpus_digest,
    )
    if not force and _valid_existing_build(output_path, policy):
        print(f"Calibration build is current: {output_path}")
        return None

    records, dropped_short = _target_tokenizer_records(
        parent_records, tokenizer, max_document_tokens
    )
    del parent_records
    if dropped_short:
        print(
            f"  Dropped {dropped_short:,} records below {MIN_CONTENT_CHARS} "
            "characters after target special-token stripping"
        )

    records, dedup_stats = deduplicate_records(records)
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
    del records, by_domain

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
        "corpus": {
            "manifest_sha256": corpus_digest,
            "sources": corpus_manifest["sources"],
        },
        "deduplication": dedup_stats,
        "domains": domain_stats,
        "files": {
            name: {"bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(artifacts.items())
        },
    }
    artifacts["manifest.json"] = _json_bytes(manifest)

    _stage_and_publish(output_path, artifacts)

    print(f"Wrote reproducible calibration build: {output_path}")
    for domain, stats in domain_stats.items():
        print(
            f"  {domain}: {stats['selected_records']:,} records, "
            f"{stats['selected_tokens']:,} calibration tokens, "
            f"{stats['holdout_tokens']:,} holdout tokens"
        )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="model-specific calibration output directory",
    )
    parser.add_argument(
        "--corpus-dir",
        default="calibration/corpus-v1",
        help="tokenizer-neutral parent corpus directory",
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="build or validate the parent corpus without a target model",
    )
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force", action="store_true", help="rebuild the model calibration output"
    )
    parser.add_argument(
        "--force-corpus",
        action="store_true",
        help="rebuild the tokenizer-neutral parent corpus",
    )
    parser.add_argument("--tokens-per-domain", type=int)
    parser.add_argument(
        "--holdout-tokens-per-domain", type=int, default=DEFAULT_HOLDOUT_TOKENS
    )
    parser.add_argument(
        "--max-document-tokens", type=int, default=DEFAULT_MAX_DOCUMENT_TOKENS
    )
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus_dir)
    if args.corpus_only:
        build_corpus(corpus_path, force=args.force_corpus, seed=args.seed)
        return
    if not args.model_id:
        parser.error("--model-id is required unless --corpus-only is used")
    if not args.output_dir:
        parser.error("--output-dir is required unless --corpus-only is used")
    if args.force_corpus:
        # An absent corpus is built by build_calibration; only an explicit
        # rebuild needs to happen here.
        build_corpus(corpus_path, force=True, seed=args.seed)

    token_targets = dict(DOMAIN_TOKEN_TARGETS)
    if args.tokens_per_domain:
        token_targets = dict.fromkeys(token_targets, args.tokens_per_domain)
    build_calibration(
        args.output_dir,
        args.model_id,
        token_targets,
        corpus_path,
        force=args.force,
        seed=args.seed,
        holdout_tokens=args.holdout_tokens_per_domain,
        max_document_tokens=args.max_document_tokens,
    )


if __name__ == "__main__":
    main()
