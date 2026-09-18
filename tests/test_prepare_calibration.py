import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from typing import ClassVar
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prepare_calibration
from prepare_calibration import CorpusError

DOMAINS = ("general", "code", "reasoning", "agentic")
SHORT_DOC = "<special>" * 15 + "too short after stripping."


class FakeTokenizer:
    def __init__(self, special_tokens=("<special>",), revision="tokenizer-revision"):
        self.all_special_tokens = list(special_tokens)
        self.init_kwargs = {"_commit_hash": revision}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def source(**overrides):
    value = {
        "dataset": "example/data",
        "revision": "a" * 40,
        "split": "train",
        "category": "chat",
        "domain": "general",
        "minimum_records": 1,
    }
    value.update(overrides)
    return value


def doc_source(name, domain):
    return source(
        dataset=f"example/{name}",
        extractor="document",
        field="text",
        domain=domain,
    )


def chat_source(name, domain="general"):
    return source(
        dataset=f"example/{name}",
        extractor="messages",
        field="messages",
        domain=domain,
    )


def doc_text(name, index):
    """A multi-paragraph document whose words are unique to (name, index)."""
    slug = name.replace("/", "-")
    sentences = " ".join(f"{slug}x{index}y{j}." for j in range(40))
    return f"Opening paragraph for {name} {index}.\n\n{sentences}"


def chat_row(tag):
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Explain the {tag} topic with full details and examples please."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"The {tag} topic works because every token is accounted for. "
                    "<special>"
                ),
            },
        ]
    }


def default_sources():
    # The second general source is deliberately interleaved with other
    # domains so corpus loads must reorder stored domain files by SOURCES.
    return (
        doc_source("general-docs", "general"),
        chat_source("general-chat", "general"),
        doc_source("code-docs", "code"),
        doc_source("general-long", "general"),
        doc_source("reasoning-docs", "reasoning"),
        doc_source("agentic-docs", "agentic"),
    )


def default_rows():
    return {
        "example/general-docs": [
            {"text": doc_text("general-docs", 0)},
            {"text": SHORT_DOC},
            {"text": doc_text("general-docs", 1) + " <tokenspec> trailing note."},
        ],
        "example/general-chat": [chat_row("general chat")],
        "example/general-long": [
            {"text": doc_text("general-long", index)} for index in range(2)
        ],
        "example/code-docs": [
            {"text": doc_text("code-docs", index)} for index in range(2)
        ],
        "example/reasoning-docs": [
            {"text": doc_text("reasoning-docs", index)} for index in range(2)
        ],
        "example/agentic-docs": [
            {"text": doc_text("agentic-docs", index)} for index in range(2)
        ],
    }


def record(text, index=0, kind="document"):
    value = {
        "schema_version": prepare_calibration.SCHEMA_VERSION,
        "domain": "general",
        "category": "prose",
        "source": {
            "dataset": "example/data",
            "revision": "a" * 40,
            "split": "train",
        },
        "kind": kind,
    }
    if kind == "document":
        value["text"] = text
    else:
        value["messages"] = text
    value["id"] = prepare_calibration.stable_record_id(value, index)
    return value


def read_jsonl(path):
    # Bytes splitlines so JSON-escaped separators like U+2028 inside a line
    # never split a record, matching the corpus loader's byte-level parsing.
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def counting_bytes_reader():
    """Return per-path read counts and a Path.read_bytes replacement."""
    counts = defaultdict(int)
    original = Path.read_bytes

    def read(path):
        counts[path] += 1
        return original(path)

    return counts, read


def dataset_order_key(item):
    if item["kind"] == "messages":
        return ("MSG",)
    line = item["text"].split("\n", 1)[0]
    if line.startswith("Opening paragraph for "):
        parts = line.split(" ")
        return ("DOC", parts[3], int(parts[4].rstrip(".")))
    return ("SHORT",)


def validate_artifact_hashes(test, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    test.assertEqual(
        set(manifest["files"]) | {"manifest.json"},
        {p.name for p in directory.iterdir()},
    )
    for name, metadata in manifest["files"].items():
        blob = (directory / name).read_bytes()
        test.assertEqual(len(blob), metadata["bytes"], name)
        test.assertEqual(prepare_calibration._sha256(blob), metadata["sha256"], name)
    return manifest


def corpus_snapshot(corpus_dir):
    return {path.name: path.read_bytes() for path in corpus_dir.iterdir()}


def output_blob(directory):
    return b"".join(
        path.read_bytes()
        for path in directory.iterdir()
        if path.suffix in {".jsonl", ".txt"}
    )


def stripped_survivors(parents, special_tokens):
    """Mirror the model-build strip/drop/id-recompute step for assertions."""
    survivors = []
    for parent in parents:
        stripped = copy.deepcopy(parent)
        prepare_calibration.strip_special_tokens(stripped, special_tokens)
        if len(prepare_calibration.canonical_payload(stripped)) < 50:
            continue
        stripped["id"] = prepare_calibration.stable_record_id(stripped)
        survivors.append(stripped)
    return survivors


class PipelineTestBase(unittest.TestCase):
    SOURCES: ClassVar = default_sources()
    ROWS: ClassVar = default_rows()
    TARGETS: ClassVar = {"general": 300, "code": 100, "reasoning": 100, "agentic": 50}

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base = Path(self.temp_dir.name)
        self.corpus_dir = self.base / "corpus"

    def build_parent(
        self,
        *,
        sources=None,
        rows=None,
        force=False,
        seed=prepare_calibration.DEFAULT_SEED,
        corpus_dir=None,
        offline=False,
    ):
        sources = self.SOURCES if sources is None else sources
        rows = self.ROWS if rows is None else rows
        calls = []
        if offline:
            loader = mock.Mock(side_effect=OSError("offline"))
        else:

            def load(name, config=None, **kwargs):
                calls.append(name)
                return rows[name]

            loader = mock.Mock(side_effect=load)
        with (
            mock.patch.object(prepare_calibration, "SOURCES", sources),
            mock.patch.object(prepare_calibration, "load_dataset", loader),
        ):
            manifest = prepare_calibration.build_corpus(
                self.corpus_dir if corpus_dir is None else corpus_dir,
                force=force,
                seed=seed,
            )
        return manifest, loader, calls

    def run_model(
        self,
        output,
        model_id="example/model",
        *,
        sources=None,
        targets=None,
        tokenizers=None,
        force=False,
        seed=prepare_calibration.DEFAULT_SEED,
        holdout_tokens=25,
        max_document_tokens=100,
    ):
        sources = self.SOURCES if sources is None else sources
        if tokenizers is None:
            tokenizers = {model_id: FakeTokenizer()}
        with (
            mock.patch.object(prepare_calibration, "SOURCES", sources),
            mock.patch.object(
                prepare_calibration.AutoTokenizer,
                "from_pretrained",
                side_effect=lambda name, **kwargs: tokenizers[name],
            ),
        ):
            return prepare_calibration.build_calibration(
                output,
                model_id,
                dict(self.TARGETS if targets is None else targets),
                self.corpus_dir,
                force=force,
                seed=seed,
                holdout_tokens=holdout_tokens,
                max_document_tokens=max_document_tokens,
            )

    def load_corpus(self, corpus_dir=None, seed=prepare_calibration.DEFAULT_SEED):
        with mock.patch.object(prepare_calibration, "SOURCES", self.SOURCES):
            return prepare_calibration.load_corpus(
                self.corpus_dir if corpus_dir is None else corpus_dir, seed=seed
            )

    def rewrite_domain_file(self, records, domain="general", corpus_dir=None):
        corpus_dir = self.corpus_dir if corpus_dir is None else corpus_dir
        data = prepare_calibration._jsonl_bytes(records)
        (corpus_dir / f"{domain}.jsonl").write_bytes(data)
        manifest_path = corpus_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][f"{domain}.jsonl"] = {
            "bytes": len(data),
            "sha256": prepare_calibration._sha256(data),
        }
        manifest_path.write_bytes(prepare_calibration._json_bytes(manifest))


class ExtractorTest(unittest.TestCase):
    def test_message_extractor_normalizes_roles_and_preserves_blank_lines(self):
        row = {
            "conversation": [
                {"from": "human", "value": "First paragraph.\n\nSecond paragraph."},
                {"from": "gpt", "value": "Answer"},
                {"from": "observation", "value": "Tool output"},
            ]
        }
        result = prepare_calibration.extract_record(
            row, source(extractor="messages", field="conversation")
        )
        self.assertEqual(
            [message["role"] for message in result["messages"]],
            ["user", "assistant", "tool"],
        )
        self.assertIn("\n\n", result["messages"][0]["content"])

    def test_message_extractor_merges_leading_system_content(self):
        result = prepare_calibration.extract_record(
            {
                "tools": [{"name": "weather"}],
                "conversation": [
                    {"from": "system", "value": "Follow policy."},
                    {"from": "human", "value": "Find weather."},
                ],
            },
            source(
                extractor="messages",
                field="conversation",
                system_fields=["tools"],
                merge_leading_system_messages=True,
            ),
        )
        self.assertEqual(
            [message["role"] for message in result["messages"]],
            ["system", "user"],
        )
        self.assertIn("weather", result["messages"][0]["content"])
        self.assertIn("Follow policy.", result["messages"][0]["content"])

    def test_prompt_response_extractor_keeps_pair_together(self):
        result = prepare_calibration.extract_record(
            {"instruction": "Solve it", "input": "Use Python", "answer": [1, 2]},
            source(
                extractor="prompt_response",
                prompt_fields=["instruction", "input"],
                response_fields=["answer"],
            ),
        )
        self.assertEqual(result["kind"], "messages")
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][1]["content"], "[1, 2]")

    def test_external_tool_definitions_become_a_system_message(self):
        result = prepare_calibration.extract_record(
            {
                "query": "Find the weather",
                "answers": [{"name": "weather", "arguments": {"city": "Oslo"}}],
                "tools": [{"name": "weather", "description": "Get weather"}],
            },
            source(
                extractor="prompt_response",
                system_fields=["tools"],
                prompt_fields=["query"],
                response_fields=["answers"],
            ),
        )
        self.assertEqual(
            [message["role"] for message in result["messages"]],
            ["system", "user", "assistant"],
        )
        self.assertIn("weather", result["messages"][0]["content"])

    def test_document_extractor_keeps_paragraphs(self):
        result = prepare_calibration.extract_record(
            {"body": "Paragraph one.\n\nParagraph two."},
            source(extractor="document", field="body"),
        )
        self.assertEqual(result["text"], "Paragraph one.\n\nParagraph two.")

    def test_tagged_tool_chat_preserves_system_and_tool_roles(self):
        result = prepare_calibration.extract_record(
            {
                "system": "Use tools.",
                "chat": "USER: Find it\nASSISTANT: Calling\nFUNCTION RESPONSE: result",
            },
            source(
                extractor="tagged_tool_chat",
                system_field="system",
                chat_field="chat",
            ),
        )
        self.assertEqual(
            [message["role"] for message in result["messages"]],
            ["system", "user", "assistant", "tool"],
        )

    def test_unknown_role_and_invalid_schema_fail(self):
        with self.assertRaisesRegex(ValueError, "unsupported message role"):
            prepare_calibration.normalize_role("moderator")
        with self.assertRaisesRegex(TypeError, "not a message list"):
            prepare_calibration.extract_record(
                {"messages": "user: flattened"},
                source(extractor="messages", field="messages"),
            )


class RecordPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_conversation_is_never_segmented(self):
        messages = [
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "y" * 100},
        ]
        value = record(messages, kind="messages")
        result = prepare_calibration.segment_document(value, self.tokenizer, 10)
        self.assertEqual(result, [value])

    def test_long_document_segments_at_token_limit(self):
        value = record("A" * 15 + "\n\n" + "B" * 15)
        segments = prepare_calibration.segment_document(value, self.tokenizer, 10)
        self.assertEqual(
            "".join(segment["text"] for segment in segments), "A" * 15 + "B" * 15
        )
        self.assertTrue(all(len(segment["text"]) <= 10 for segment in segments))
        self.assertEqual(len({segment["id"] for segment in segments}), len(segments))

    def test_special_tokens_are_removed_from_each_message(self):
        value = record(
            [
                {"role": "user", "content": "before<special>after"},
                {"role": "assistant", "content": "done<special>"},
            ],
            kind="messages",
        )
        self.assertTrue(prepare_calibration.strip_special_tokens(value, ["<special>"]))
        self.assertEqual(value["messages"][0]["content"], "beforeafter")
        self.assertNotIn("<special>", value["messages"][1]["content"])

    def test_ids_are_stable_and_include_source_revision(self):
        value = record("stable content")
        first = prepare_calibration.stable_record_id(value)
        self.assertEqual(
            first, prepare_calibration.stable_record_id(copy.deepcopy(value))
        )
        changed = copy.deepcopy(value)
        changed["source"]["revision"] = "b" * 40
        self.assertNotEqual(first, prepare_calibration.stable_record_id(changed))

    def test_exact_duplicates_are_removed_deterministically(self):
        first = record("The same normalized text.")
        duplicate = record("The same normalized text!!!", 1)
        unique = record("A completely different document about another topic.", 2)
        accepted, removed = prepare_calibration.deduplicate_records(
            [first, duplicate, unique]
        )
        self.assertEqual([item["id"] for item in accepted], [first["id"], unique["id"]])
        self.assertEqual(
            removed,
            {
                "exact": 1,
                "near": 0,
                "by_source": {"example/data": {"exact": 1, "near": 0}},
            },
        )

    def test_near_duplicates_are_removed(self):
        first = record(" ".join(["calibration"] * 100))
        near = record(" ".join(["calibration"] * 99 + ["calibrations"]), 1)
        accepted, removed = prepare_calibration.deduplicate_records([first, near])
        self.assertEqual(accepted, [first])
        self.assertEqual(removed["near"], 1)

    def test_zero_token_holdout_selects_no_records(self):
        records = [record("complete sample " + "x" * 60)]
        training, holdout, tokens = prepare_calibration.select_holdout(
            records, 0, self.tokenizer, 7
        )
        self.assertEqual(training, records)
        self.assertEqual(holdout, [])
        self.assertEqual(tokens, 0)

    def test_holdout_is_disjoint_and_selection_is_deterministic(self):
        records = [record(f"sample {index} " + "x" * 60, index) for index in range(8)]
        training, holdout, _ = prepare_calibration.select_holdout(
            records, 60, self.tokenizer, 7
        )
        selected_a, tokens_a = prepare_calibration.select_to_token_budget(
            training, 120, self.tokenizer, 11
        )
        selected_b, tokens_b = prepare_calibration.select_to_token_budget(
            training, 120, self.tokenizer, 11
        )
        self.assertEqual(
            [item["id"] for item in selected_a], [item["id"] for item in selected_b]
        )
        self.assertEqual(tokens_a, tokens_b)
        self.assertFalse(
            {item["id"] for item in selected_a} & {item["id"] for item in holdout}
        )

    def test_render_preserves_roles_and_internal_blank_lines(self):
        value = record(
            [
                {"role": "system", "content": "Policy"},
                {"role": "user", "content": "Part one\n\nPart two"},
                {"role": "assistant", "content": "Done"},
            ],
            kind="messages",
        )
        rendered = prepare_calibration.render_record(value)
        self.assertIn("System:\nPolicy", rendered)
        self.assertIn("User:\nPart one\n\nPart two", rendered)
        self.assertIn("Assistant:\nDone", rendered)


class SourceCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cache_dir = Path(self.temp_dir.name)
        self.source = source(extractor="document", field="text")

    def test_extracted_records_are_complete_unsplit_and_tokenizer_free(self):
        text = doc_text("cache-docs", 0)
        rows = [{"text": text}, {"text": ""}]
        with mock.patch.object(
            prepare_calibration, "load_dataset", return_value=rows
        ):
            records, stats = prepare_calibration.load_source(
                self.source, cache_dir=self.cache_dir
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], text)
        self.assertNotIn("_rendered_tokens", records[0])
        self.assertEqual(
            set(records[0]),
            {"schema_version", "domain", "category", "source", "kind", "text", "id"},
        )
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["empty"], 1)
        self.assertEqual(stats["records"], 1)
        self.assertEqual(
            stats["characters"],
            len(prepare_calibration.canonical_payload(records[0])),
        )

    def test_completed_source_is_reused_without_loading_dataset(self):
        rows = [{"text": doc_text("cache-docs", 0)}]
        with mock.patch.object(
            prepare_calibration, "load_dataset", return_value=rows
        ):
            first = prepare_calibration.load_source(
                self.source, cache_dir=self.cache_dir
            )
        offline = mock.Mock(side_effect=OSError("offline"))
        with mock.patch.object(prepare_calibration, "load_dataset", offline):
            second = prepare_calibration.load_source(
                self.source, cache_dir=self.cache_dir
            )
        self.assertEqual(first, second)
        offline.assert_not_called()

    def test_corrupt_cache_is_rebuilt(self):
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": doc_text("cache-docs", 0)}],
        ):
            prepare_calibration.load_source(self.source, cache_dir=self.cache_dir)
        next(self.cache_dir.glob("*.jsonl")).write_text("corrupt\n")
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": doc_text("cache-docs", 1)}],
        ) as loader:
            records, _ = prepare_calibration.load_source(
                self.source, cache_dir=self.cache_dir
            )
        loader.assert_called_once()
        self.assertIn("cache-docsx1y0", records[0]["text"])

    def test_malformed_cache_manifest_is_rebuilt(self):
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": doc_text("cache-docs", 0)}],
        ):
            prepare_calibration.load_source(self.source, cache_dir=self.cache_dir)
        next(self.cache_dir.glob("*.manifest.json")).write_bytes(b"\xff")
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": doc_text("cache-docs", 1)}],
        ) as loader:
            records, _ = prepare_calibration.load_source(
                self.source, cache_dir=self.cache_dir
            )
        loader.assert_called_once()
        self.assertIn("cache-docsx1y0", records[0]["text"])

    def test_seed_change_creates_a_separate_cache_entry(self):
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": doc_text("cache-docs", 0)}],
        ):
            prepare_calibration.load_source(
                self.source, seed=42, cache_dir=self.cache_dir
            )
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": doc_text("cache-docs", 1)}],
        ) as loader:
            prepare_calibration.load_source(
                self.source, seed=43, cache_dir=self.cache_dir
            )
        loader.assert_called_once()
        self.assertEqual(len(list(self.cache_dir.glob("*.manifest.json"))), 2)
        offline = mock.Mock(side_effect=OSError("offline"))
        with mock.patch.object(prepare_calibration, "load_dataset", offline):
            for seed in (42, 43):
                prepare_calibration.load_source(
                    self.source, seed=seed, cache_dir=self.cache_dir
                )
        offline.assert_not_called()

    def test_interrupted_manifest_swap_does_not_create_valid_cache(self):
        policy = prepare_calibration._source_cache_policy(
            self.source,
            {
                "corpus_schema_version": prepare_calibration.CORPUS_SCHEMA_VERSION,
                "extractor_version": prepare_calibration.EXTRACTOR_VERSION,
                "cleanup_policy": prepare_calibration.CORPUS_CLEANUP_POLICY,
                "seed": prepare_calibration.DEFAULT_SEED,
            },
        )
        data_path, manifest_path = prepare_calibration._source_cache_paths(
            self.cache_dir, policy
        )
        records = [record("atomic cache record with enough characters to validate.")]
        stats = {"rows": 1, "accepted": 1, "empty": 0, "records": 1, "characters": 56}
        original_replace = Path.replace

        def fail_manifest_swap(path, target):
            if Path(target) == manifest_path:
                raise OSError("simulated manifest swap failure")
            return original_replace(path, target)

        with (
            mock.patch.object(Path, "replace", fail_manifest_swap),
            self.assertRaisesRegex(OSError, "simulated manifest swap failure"),
        ):
            prepare_calibration._write_source_cache(
                self.cache_dir, policy, records, stats
            )
        # The data file landed but its manifest did not, so the entry is invalid.
        self.assertTrue(data_path.exists())
        self.assertFalse(manifest_path.exists())
        self.assertIsNone(
            prepare_calibration._load_source_cache(self.cache_dir, policy)
        )
        self.assertFalse(list(self.cache_dir.glob(".source*")))

    def test_required_source_load_failure_is_fatal(self):
        with (
            mock.patch.object(
                prepare_calibration,
                "load_dataset",
                side_effect=OSError("offline"),
            ),
            self.assertRaisesRegex(RuntimeError, "required source example/data"),
        ):
            prepare_calibration.load_source(self.source)

    def test_source_cap_below_minimum_fails_before_loading(self):
        constrained = {**self.source, "max_samples": 1, "minimum_records": 2}
        with (
            mock.patch.object(prepare_calibration, "load_dataset") as loader,
            self.assertRaisesRegex(ValueError, "below its minimum"),
        ):
            prepare_calibration.load_source(constrained)
        loader.assert_not_called()

    def test_source_minimum_contribution_is_enforced(self):
        constrained = {**self.source, "minimum_records": 2}
        with (
            mock.patch.object(
                prepare_calibration,
                "load_dataset",
                return_value=[{"text": doc_text("cache-docs", 0)}],
            ),
            self.assertRaisesRegex(ValueError, "minimum is 2"),
        ):
            prepare_calibration.load_source(constrained)


class ParentCorpusTest(PipelineTestBase):
    def setUp(self):
        super().setUp()
        self.manifest, _, _ = self.build_parent()

    def test_parent_emits_only_domain_files_and_a_manifest(self):
        self.assertEqual(
            {path.name for path in self.corpus_dir.iterdir()},
            {f"{domain}.jsonl" for domain in DOMAINS} | {"manifest.json"},
        )
        validate_artifact_hashes(self, self.corpus_dir)
        manifest_text = (self.corpus_dir / "manifest.json").read_text()
        for tokenizer_detail in (
            "tokenizer",
            "model_id",
            "_rendered_tokens",
            "max_document_tokens",
            "holdout",
        ):
            self.assertNotIn(tokenizer_detail, manifest_text)
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["policy"], self.manifest["policy"])
        self.assertEqual(
            [entry["source"]["dataset"] for entry in manifest["sources"]],
            [entry["source"]["dataset"] for entry in self.manifest["sources"]],
        )

    def test_parent_preserves_source_text_verbatim(self):
        blob = (self.corpus_dir / "general.jsonl").read_bytes()
        self.assertIn(b"<special>", blob)
        self.assertIn(b"<tokenspec>", blob)
        _, records, _ = self.load_corpus()
        chat = next(item for item in records if item["kind"] == "messages")
        self.assertEqual(
            [message["content"] for message in chat["messages"]],
            [message["content"] for message in chat_row("general chat")["messages"]],
        )
        documents = {
            item["text"] for item in records if item["kind"] == "document"
        }
        for row in self.ROWS["example/code-docs"]:
            self.assertIn(row["text"], documents)
        for item in records:
            self.assertNotIn("_rendered_tokens", item)

    def test_parent_loader_preserves_unicode_line_separator_inside_json(self):
        _, records, _ = self.load_corpus()
        general = [item for item in records if item["domain"] == "general"]
        document = next(item for item in general if item["kind"] == "document")
        document["text"] += "\u2028still the same JSONL record"
        document["id"] = prepare_calibration.stable_record_id(document)
        self.rewrite_domain_file(general)

        _, loaded, _ = self.load_corpus()
        rewritten = next(item for item in loaded if item["id"] == document["id"])
        self.assertIn("\u2028still the same JSONL record", rewritten["text"])

    def test_parent_build_is_deterministic_across_fresh_caches(self):
        first = corpus_snapshot(self.corpus_dir)
        second_dir = self.base / "second" / "corpus"
        manifest, _, calls = self.build_parent(corpus_dir=second_dir)
        self.assertIsNotNone(manifest)
        self.assertEqual(sorted(calls), sorted(self.ROWS))
        self.assertEqual(corpus_snapshot(second_dir), first)

    def test_cli_corpus_only_defaults_to_the_calibration_corpus_dir(self):
        previous_cwd = os.getcwd()
        os.chdir(self.base)
        try:
            with (
                mock.patch.object(prepare_calibration, "SOURCES", self.SOURCES),
                mock.patch.object(
                    prepare_calibration,
                    "load_dataset",
                    side_effect=lambda name, config=None, **kwargs: self.ROWS[name],
                ),
            ):
                prepare_calibration.main(["--corpus-only"])
        finally:
            os.chdir(previous_cwd)
        default_dir = self.base / "calibration" / "corpus-v1"
        self.assertEqual(corpus_snapshot(default_dir), corpus_snapshot(self.corpus_dir))

    def test_cli_model_build_requires_explicit_output_directory(self):
        with self.assertRaises(SystemExit):
            prepare_calibration.main(
                ["--corpus-dir", str(self.corpus_dir), "--model-id", "example/model"]
            )

    def test_shared_source_cache_allows_offline_corpus_rebuild(self):
        snapshot = corpus_snapshot(self.corpus_dir)
        shutil.rmtree(self.corpus_dir)
        manifest, loader, calls = self.build_parent(offline=True)
        loader.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIsNotNone(manifest)
        self.assertEqual(corpus_snapshot(self.corpus_dir), snapshot)

    def test_records_follow_pinned_source_order_not_storage_order(self):
        _, records, _ = self.load_corpus()
        self.assertEqual(
            [dataset_order_key(item) for item in records],
            [
                ("DOC", "general-docs", 0),
                ("SHORT",),
                ("DOC", "general-docs", 1),
                ("MSG",),
                ("DOC", "code-docs", 0),
                ("DOC", "code-docs", 1),
                ("DOC", "general-long", 0),
                ("DOC", "general-long", 1),
                ("DOC", "reasoning-docs", 0),
                ("DOC", "reasoning-docs", 1),
                ("DOC", "agentic-docs", 0),
                ("DOC", "agentic-docs", 1),
            ],
        )

        # Store the general domain in scrambled order; the loaded order must
        # still follow the pinned SOURCES sequence.
        path = self.corpus_dir / "general.jsonl"
        lines = [line for line in path.read_bytes().splitlines() if line]
        self.rewrite_domain_file(list(reversed([json.loads(line) for line in lines])))
        _, records, _ = self.load_corpus()
        self.assertEqual(
            [dataset_order_key(item) for item in records],
            [
                ("DOC", "general-docs", 1),
                ("SHORT",),
                ("DOC", "general-docs", 0),
                ("MSG",),
                ("DOC", "code-docs", 0),
                ("DOC", "code-docs", 1),
                ("DOC", "general-long", 1),
                ("DOC", "general-long", 0),
                ("DOC", "reasoning-docs", 0),
                ("DOC", "reasoning-docs", 1),
                ("DOC", "agentic-docs", 0),
                ("DOC", "agentic-docs", 1),
            ],
        )

    def test_parent_never_deduplicates(self):
        shared = doc_text("dup-docs", 0)
        sources = (
            doc_source("dup-general", "general"),
            doc_source("dup-code", "code"),
        )
        rows = {
            "example/dup-general": [{"text": shared}, {"text": shared}],
            "example/dup-code": [{"text": shared}],
        }
        corpus_dir = self.base / "dup-corpus"
        _, _, calls = self.build_parent(
            sources=sources, rows=rows, corpus_dir=corpus_dir
        )
        self.assertEqual(len(calls), 2)
        general = read_jsonl(corpus_dir / "general.jsonl")
        code = read_jsonl(corpus_dir / "code.jsonl")
        self.assertEqual(len(general), 2)
        self.assertEqual(general[0]["id"], general[1]["id"])
        self.assertEqual(len(code), 1)
        self.assertEqual(code[0]["text"], shared)


class CorpusIntegrityTest(PipelineTestBase):
    def setUp(self):
        super().setUp()
        self.build_parent()

    def test_corrupt_domain_file_raises_corpus_error(self):
        with (self.corpus_dir / "code.jsonl").open("ab") as handle:
            handle.write(b"x")
        with self.assertRaisesRegex(CorpusError, "code.jsonl failed integrity"):
            self.load_corpus()
        with self.assertRaisesRegex(CorpusError, "code.jsonl"):
            self.build_parent()
        with self.assertRaisesRegex(CorpusError, "code.jsonl"):
            self.run_model(self.base / "build")

    def test_corrupt_manifest_raises_corpus_error(self):
        (self.corpus_dir / "manifest.json").write_bytes(b"\xff")
        with self.assertRaisesRegex(CorpusError, "invalid corpus manifest"):
            self.load_corpus()

    def test_missing_domain_file_raises_corpus_error(self):
        (self.corpus_dir / "general.jsonl").unlink()
        with self.assertRaisesRegex(CorpusError, "general.jsonl"):
            self.load_corpus()

    def test_unexpected_manifest_file_set_raises(self):
        manifest_path = self.corpus_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["files"]["general.jsonl"]
        manifest_path.write_bytes(prepare_calibration._json_bytes(manifest))
        with self.assertRaisesRegex(CorpusError, "does not match the expected"):
            self.load_corpus()

    def test_parent_record_with_token_counts_is_rejected(self):
        records = read_jsonl(self.corpus_dir / "general.jsonl")
        records[0]["_rendered_tokens"] = 123
        self.rewrite_domain_file(records)
        with self.assertRaisesRegex(CorpusError, "carries tokenizer token counts"):
            self.load_corpus()

    def test_record_with_mismatched_domain_is_rejected(self):
        records = read_jsonl(self.corpus_dir / "general.jsonl")
        records[0]["domain"] = "code"
        self.rewrite_domain_file(records)
        with self.assertRaisesRegex(CorpusError, "expected 'general'"):
            self.load_corpus()

    def test_record_from_an_unknown_source_is_rejected(self):
        ghost = {
            "schema_version": prepare_calibration.SCHEMA_VERSION,
            "domain": "general",
            "category": "prose",
            "source": {
                "dataset": "ghost/dataset",
                "revision": "c" * 40,
                "split": "train",
            },
            "kind": "document",
            "text": "ghost document body with enough characters to validate fine.",
        }
        ghost["id"] = prepare_calibration.stable_record_id(ghost)
        records = read_jsonl(self.corpus_dir / "general.jsonl")
        self.rewrite_domain_file([ghost, *records])
        with self.assertRaisesRegex(CorpusError, "unknown source"):
            self.load_corpus()

    def test_corpus_policy_seed_mismatch_demands_rebuild(self):
        with self.assertRaisesRegex(CorpusError, "corpus policy mismatch"):
            self.load_corpus(seed=99)

    def test_corpus_validation_reads_each_file_exactly_once(self):
        names = ("manifest.json", *(f"{domain}.jsonl" for domain in DOMAINS))
        counts, read = counting_bytes_reader()
        with (
            mock.patch.object(prepare_calibration, "SOURCES", self.SOURCES),
            mock.patch.object(Path, "read_bytes", read),
        ):
            loaded_manifest, records, manifest_digest = self.load_corpus()
            manifest, digest, payload = (
                prepare_calibration.load_corpus_artifacts(self.corpus_dir)
            )
        # One read per file per validation call, counting both entry points.
        for name in names:
            self.assertEqual(counts[self.corpus_dir / name], 2, name)
        self.assertEqual(manifest, loaded_manifest)
        self.assertEqual(digest, manifest_digest)
        self.assertEqual(manifest_digest, prepare_calibration._sha256(
            payload["manifest.json"]
        ))
        self.assertEqual(set(payload), set(names))
        for domain in DOMAINS:
            self.assertIn(f"{domain}.jsonl", payload)
        self.assertEqual(len(records), 12)

    def test_load_corpus_artifacts_rejects_a_corrupt_corpus(self):
        (self.corpus_dir / "code.jsonl").write_bytes(b"corrupt\n")
        with (
            mock.patch.object(prepare_calibration, "SOURCES", self.SOURCES),
            self.assertRaisesRegex(CorpusError, "code.jsonl failed integrity"),
        ):
            prepare_calibration.load_corpus_artifacts(self.corpus_dir)

    def test_load_corpus_artifacts_rejects_a_record_from_an_unknown_source(self):
        ghost = {
            "schema_version": prepare_calibration.SCHEMA_VERSION,
            "domain": "general",
            "category": "prose",
            "source": {
                "dataset": "ghost/dataset",
                "revision": "c" * 40,
                "split": "train",
            },
            "kind": "document",
            "text": "ghost document body with enough characters to validate fine.",
        }
        ghost["id"] = prepare_calibration.stable_record_id(ghost)
        records = read_jsonl(self.corpus_dir / "general.jsonl")
        self.rewrite_domain_file([ghost, *records])
        with (
            mock.patch.object(prepare_calibration, "SOURCES", self.SOURCES),
            self.assertRaisesRegex(CorpusError, "unknown source"),
        ):
            prepare_calibration.load_corpus_artifacts(self.corpus_dir)

    def test_force_rebuild_recovers_a_corrupt_corpus_offline(self):
        (self.corpus_dir / "general.jsonl").write_text("corruption\n")
        manifest, loader, _ = self.build_parent(force=True, offline=True)
        self.assertIsNotNone(manifest)
        loader.assert_not_called()
        _, records, _ = self.load_corpus()
        self.assertTrue(records)
        validate_artifact_hashes(self, self.corpus_dir)

    def test_interrupted_corpus_swap_restores_previous_corpus(self):
        snapshot = corpus_snapshot(self.corpus_dir)
        original_replace = Path.replace

        def fail_stage_swap(path, target):
            if Path(target) == self.corpus_dir and path.name.startswith(".corpus."):
                raise OSError("simulated corpus swap failure")
            return original_replace(path, target)

        with (
            mock.patch.object(Path, "replace", fail_stage_swap),
            self.assertRaisesRegex(OSError, "simulated corpus swap failure"),
        ):
            self.build_parent(force=True)
        self.assertEqual(corpus_snapshot(self.corpus_dir), snapshot)
        self.assertFalse(
            [
                path
                for path in self.base.iterdir()
                if path.is_dir() and path.name.startswith(".corpus.backup.")
            ]
        )


class ModelBuildTest(PipelineTestBase):
    def setUp(self):
        super().setUp()
        self.build_parent()

    def test_build_is_deterministic_and_manifest_hashes_validate(self):
        first = self.base / "first"
        second = self.base / "second"
        manifest_a = self.run_model(first)
        manifest_b = self.run_model(second)
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(
            {path.name: path.read_bytes() for path in first.iterdir()},
            {path.name: path.read_bytes() for path in second.iterdir()},
        )
        manifest = validate_artifact_hashes(self, first)
        self.assertTrue(
            prepare_calibration._valid_existing_build(first, manifest["policy"])
        )
        allowed = {
            "schema_version",
            "id",
            "domain",
            "category",
            "source",
            "kind",
            "text",
            "messages",
        }
        for line in (first / "combined.jsonl").read_text().splitlines():
            self.assertFalse(set(json.loads(line)) - allowed)
        self.assertEqual(
            manifest["corpus"]["sources"][0]["source"],
            {
                "dataset": "example/general-docs",
                "revision": "a" * 40,
                "split": "train",
            },
        )
        # The manifest records corpus identity, not host-local paths.
        self.assertEqual(set(manifest["corpus"]), {"manifest_sha256", "sources"})
        self.assertEqual(manifest["tokenizer"], {"model_id": "example/model", "revision": "tokenizer-revision"})
        corpus_digest = prepare_calibration._sha256(
            (self.corpus_dir / "manifest.json").read_bytes()
        )
        self.assertEqual(manifest["policy"]["corpus_manifest_sha256"], corpus_digest)
        self.assertEqual(manifest["corpus"]["manifest_sha256"], corpus_digest)

    def test_current_build_is_reused_without_rebuilding(self):
        output = self.base / "build"
        self.run_model(output)
        snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
        offline = mock.Mock(side_effect=OSError("offline"))
        with mock.patch.object(prepare_calibration, "load_dataset", offline):
            self.assertIsNone(self.run_model(output))
        offline.assert_not_called()
        self.assertEqual(
            {path.name: path.read_bytes() for path in output.iterdir()}, snapshot
        )

    def test_corrupt_artifact_invalidates_build_and_is_repaired(self):
        output = self.base / "build"
        manifest = self.run_model(output)
        (output / "combined.txt").write_text("corrupt")
        self.assertFalse(
            prepare_calibration._valid_existing_build(output, manifest["policy"])
        )
        rebuilt = self.run_model(output)
        self.assertIsNotNone(rebuilt)
        validate_artifact_hashes(self, output)

    def test_failed_atomic_swap_restores_previous_build(self):
        output = self.base / "build"
        self.run_model(output)
        previous = {path.name: path.read_bytes() for path in output.iterdir()}
        original_replace = Path.replace

        def fail_stage_swap(path, target):
            if path.name.startswith(".build.") and Path(target) == output:
                raise OSError("simulated swap failure")
            return original_replace(path, target)

        with (
            mock.patch.object(Path, "replace", fail_stage_swap),
            self.assertRaisesRegex(OSError, "simulated swap failure"),
        ):
            self.run_model(output, force=True)
        self.assertEqual(
            {path.name: path.read_bytes() for path in output.iterdir()}, previous
        )

    def test_corpus_rebuild_invalidates_the_model_build(self):
        output = self.base / "build"
        first = self.run_model(output)
        changed_rows = {
            **self.ROWS,
            "example/general-long": [
                {"text": doc_text("general-long", 0)},
                {"text": doc_text("general-long", 7)},
            ],
        }
        # The source cache pins content by policy, so drop it to change rows.
        shutil.rmtree(self.base / prepare_calibration.CORPUS_SOURCE_CACHE_DIRNAME)
        self.build_parent(rows=changed_rows, force=True)
        rebuilt = self.run_model(output)
        self.assertIsNotNone(rebuilt)
        self.assertNotEqual(
            first["policy"]["corpus_manifest_sha256"],
            rebuilt["policy"]["corpus_manifest_sha256"],
        )
        validate_artifact_hashes(self, output)

    def test_no_token_target_for_a_corpus_domain_fails_clearly(self):
        with self.assertRaisesRegex(
            ValueError,
            "No token target configured for domains: agentic, code, reasoning",
        ):
            self.run_model(self.base / "build", targets={"general": 100})

    def test_target_domain_without_records_fails_clearly(self):
        sources = (doc_source("only-general", "general"),)
        rows = {"example/only-general": [{"text": doc_text("only-general", 0)}]}
        self.build_parent(sources=sources, rows=rows, force=True)
        with self.assertRaisesRegex(
            ValueError,
            "code calibration selection failed: only 0 tokens available",
        ):
            self.run_model(
                self.base / "build",
                sources=sources,
                targets={"general": 100, "code": 100},
                holdout_tokens=0,
            )

    def test_insufficient_token_budget_fails_clearly(self):
        with self.assertRaisesRegex(
            ValueError,
            r"general calibration selection failed: only [\d,]+ tokens available "
            r"for target 10,000,000",
        ):
            self.run_model(
                self.base / "build",
                targets={**self.TARGETS, "general": 10_000_000},
            )

    def test_target_tokenizer_strips_special_tokens_and_drops_short_records(self):
        sources = (doc_source("strip-docs", "general"),)
        rows = {
            "example/strip-docs": [
                {"text": doc_text("strip-docs", 0) + " <special> embedded."},
                {"text": SHORT_DOC},
                {"text": doc_text("strip-docs", 1)},
            ]
        }
        corpus = self.base / "strip-corpus"
        self.build_parent(sources=sources, rows=rows, corpus_dir=corpus)
        with mock.patch.object(prepare_calibration, "SOURCES", sources):
            _, parents, _ = prepare_calibration.load_corpus(corpus)
        survivors = stripped_survivors(parents, ["<special>"])
        self.assertEqual(len(survivors), 2)
        total_tokens = sum(
            len(prepare_calibration.render_record(item)) for item in survivors
        )
        output = self.base / "strip-build"
        with (
            mock.patch.object(prepare_calibration, "SOURCES", sources),
            mock.patch.object(
                prepare_calibration.AutoTokenizer,
                "from_pretrained",
                return_value=FakeTokenizer(),
            ),
        ):
            manifest = prepare_calibration.build_calibration(
                output,
                "example/model",
                {"general": total_tokens},
                corpus,
                holdout_tokens=0,
                max_document_tokens=100_000,
            )
        selected = read_jsonl(output / "combined.jsonl")
        self.assertEqual(
            sorted(item["id"] for item in selected),
            sorted(item["id"] for item in survivors),
        )
        self.assertNotIn(b"<special>", output_blob(output))
        self.assertNotIn(b"too short", output_blob(output))
        self.assertEqual(manifest["domains"]["general"]["selected_records"], 2)
        self.assertEqual(manifest["domains"]["general"]["holdout_records"], 0)

    def test_oversized_documents_are_resegmented_for_the_target_tokenizer(self):
        output = self.base / "build"
        manifest = self.run_model(output, max_document_tokens=100)
        _, parents, _ = self.load_corpus()
        parent_ids = {item["id"] for item in parents}
        documents = [
            item
            for name in ("general.jsonl", "code.jsonl", "reasoning.jsonl", "agentic.jsonl")
            for item in read_jsonl(output / name)
            if item["kind"] == "document"
        ]
        self.assertTrue(documents)
        for item in documents:
            self.assertLessEqual(len(item["text"]), 100)
            self.assertNotIn("_rendered_tokens", item)
            # Every parent document exceeds the 100-token segment limit, so
            # all output documents carry fresh segment ids.
            self.assertNotIn(item["id"], parent_ids)
        for domain, target in self.TARGETS.items():
            self.assertGreaterEqual(
                manifest["domains"][domain]["selected_tokens"], target
            )

    def test_model_output_never_touches_the_parent_corpus(self):
        snapshot = corpus_snapshot(self.corpus_dir)
        self.run_model(self.base / "build")
        self.assertEqual(corpus_snapshot(self.corpus_dir), snapshot)


class SharedParentTest(PipelineTestBase):
    SOURCES: ClassVar = (doc_source("shared-docs", "general"),)
    ROWS: ClassVar = {
        "example/shared-docs": [
            {"text": doc_text("shared-docs", 0) + " <special> embedded."},
            {"text": doc_text("shared-docs", 1) + " <tokenspec> note."},
            {"text": SHORT_DOC},
        ]
    }

    def setUp(self):
        super().setUp()
        self.build_parent()
        self.snapshot = corpus_snapshot(self.corpus_dir)

    def test_one_valid_parent_serves_two_models_while_offline(self):
        tokenizers = {
            "model-a": FakeTokenizer(["<special>"], "rev-a"),
            "model-b": FakeTokenizer(["<tokenspec>"], "rev-b"),
        }
        # Target exactly the surviving tokens of model-a. Model-b keeps the
        # short record that model-a drops but loses more elsewhere, so its
        # budget cannot be satisfied without selecting every record too.
        _, parents, _ = self.load_corpus()
        target_a = sum(
            len(prepare_calibration.render_record(item))
            for item in stripped_survivors(parents, ["<special>"])
        )
        output_a = self.base / "model-a"
        output_b = self.base / "model-b"
        offline = mock.Mock(side_effect=OSError("offline"))
        with mock.patch.object(prepare_calibration, "load_dataset", offline):
            manifest_a = self.run_model(
                output_a,
                "model-a",
                tokenizers=tokenizers,
                targets={"general": target_a},
                holdout_tokens=0,
                max_document_tokens=100_000,
            )
            manifest_b = self.run_model(
                output_b,
                "model-b",
                tokenizers=tokenizers,
                targets={"general": target_a},
                holdout_tokens=0,
                max_document_tokens=100_000,
            )
            self.assertIsNone(
                self.run_model(
                    output_a,
                    "model-a",
                    tokenizers=tokenizers,
                    targets={"general": target_a},
                    holdout_tokens=0,
                    max_document_tokens=100_000,
                )
            )
        offline.assert_not_called()
        self.assertEqual(corpus_snapshot(self.corpus_dir), self.snapshot)
        self.assertEqual(
            manifest_a["policy"]["corpus_manifest_sha256"],
            manifest_b["policy"]["corpus_manifest_sha256"],
        )
        self.assertEqual(manifest_a["tokenizer"]["revision"], "rev-a")
        self.assertEqual(manifest_b["tokenizer"]["revision"], "rev-b")

        blob_a = output_blob(output_a)
        blob_b = output_blob(output_b)
        # Each model strips only its own tokenizer's special tokens.
        self.assertNotIn(b"<special>", blob_a)
        self.assertIn(b"<tokenspec>", blob_a)
        self.assertIn(b"<special>", blob_b)
        self.assertNotIn(b"<tokenspec>", blob_b)
        # Short-record dropping is target-tokenizer dependent: model-a strips
        # the special-token-heavy record below the minimum, model-b keeps it.
        self.assertNotIn(b"too short after stripping", blob_a)
        self.assertIn(b"too short after stripping", blob_b)


if __name__ == "__main__":
    unittest.main()
