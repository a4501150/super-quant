import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prepare_calibration


class FakeTokenizer:
    all_special_tokens: ClassVar = ["<special>"]
    init_kwargs: ClassVar = {"_commit_hash": "tokenizer-revision"}

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
        self.tokenizer = FakeTokenizer()
        self.source = source(extractor="document", field="text")
        self.context = {
            "render_version": prepare_calibration.RENDER_VERSION,
            "model_id": "example/model",
            "tokenizer_revision": "tokenizer-revision",
            "special_tokens": ["<special>"],
            "max_document_tokens": 100,
            "seed": 42,
        }

    def load(self, context=None):
        return prepare_calibration.load_source(
            self.source,
            self.tokenizer,
            self.tokenizer.all_special_tokens,
            max_document_tokens=100,
            cache_dir=self.cache_dir,
            cache_context=context or self.context,
        )

    def test_completed_source_is_reused_without_loading_dataset(self):
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": "cached source " + "x" * 80}],
        ):
            first = self.load()
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            side_effect=OSError("offline"),
        ) as loader:
            second = self.load()
        self.assertEqual(first, second)
        loader.assert_not_called()

    def test_corrupt_cache_is_rebuilt(self):
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": "original source " + "x" * 80}],
        ):
            self.load()
        next(self.cache_dir.glob("*.jsonl")).write_text("corrupt\n")
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": "rebuilt source " + "y" * 80}],
        ) as loader:
            records, _ = self.load()
        loader.assert_called_once()
        self.assertIn("rebuilt source", records[0]["text"])

    def test_malformed_cache_manifest_is_rebuilt(self):
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": "original source " + "x" * 80}],
        ):
            self.load()
        next(self.cache_dir.glob("*.manifest.json")).write_bytes(b"\xff")
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": "rebuilt source " + "y" * 80}],
        ) as loader:
            records, _ = self.load()
        loader.assert_called_once()
        self.assertIn("rebuilt source", records[0]["text"])

    def test_policy_change_uses_a_different_cache_entry(self):
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": "first policy " + "x" * 80}],
        ):
            self.load()
        changed_context = {**self.context, "seed": 43}
        with mock.patch.object(
            prepare_calibration,
            "load_dataset",
            return_value=[{"text": "second policy " + "y" * 80}],
        ) as loader:
            records, _ = self.load(changed_context)
        loader.assert_called_once()
        self.assertIn("second policy", records[0]["text"])
        self.assertEqual(len(list(self.cache_dir.glob("*.manifest.json"))), 2)

    def test_interrupted_manifest_swap_does_not_create_valid_cache(self):
        policy = prepare_calibration._source_cache_policy(self.source, self.context)
        _, manifest_path = prepare_calibration._source_cache_paths(
            self.cache_dir, policy
        )
        records = [record("atomic cache record")]
        stats = {"segments": 1}
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
        self.assertIsNone(
            prepare_calibration._load_source_cache(self.cache_dir, policy)
        )
        self.assertFalse(list(self.cache_dir.glob(".source*")))


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.tokenizer = FakeTokenizer()
        self.records = [
            record(f"document {index} " + chr(65 + index) * 80, index)
            for index in range(6)
        ]
        self.source = source(extractor="document", field="text")

    def run_build(self, output, *, force=False):
        with (
            mock.patch.object(prepare_calibration, "SOURCES", (self.source,)),
            mock.patch.object(
                prepare_calibration.AutoTokenizer,
                "from_pretrained",
                return_value=self.tokenizer,
            ),
            mock.patch.object(
                prepare_calibration,
                "load_source",
                return_value=(copy.deepcopy(self.records), {"accepted": 6}),
            ),
        ):
            return prepare_calibration.build_calibration(
                output,
                "example/model",
                {"general": 100},
                force=force,
                holdout_tokens=50,
                max_document_tokens=100,
            )

    def test_build_is_deterministic_and_manifest_hashes_validate(self):
        first = Path(self.temp_dir.name) / "first"
        second = Path(self.temp_dir.name) / "second"
        manifest_a = self.run_build(first)
        manifest_b = self.run_build(second)
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(
            manifest_a["sources"][0]["source"],
            {
                "dataset": "example/data",
                "revision": "a" * 40,
                "split": "train",
            },
        )
        self.assertEqual(
            {path.name: path.read_bytes() for path in first.iterdir()},
            {path.name: path.read_bytes() for path in second.iterdir()},
        )
        self.assertTrue(
            prepare_calibration._valid_existing_build(first, manifest_a["policy"])
        )
        self.assertFalse(
            set(json.loads((first / "combined.jsonl").read_text().splitlines()[0]))
            - {
                "schema_version",
                "id",
                "domain",
                "category",
                "source",
                "kind",
                "text",
            }
        )

    def test_failed_atomic_swap_restores_previous_build(self):
        output = Path(self.temp_dir.name) / "build"
        self.run_build(output)
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
            self.run_build(output, force=True)
        self.assertEqual(
            {path.name: path.read_bytes() for path in output.iterdir()}, previous
        )

    def test_corrupt_artifact_invalidates_existing_build(self):
        output = Path(self.temp_dir.name) / "build"
        manifest = self.run_build(output)
        (output / "combined.txt").write_text("corrupt")
        self.assertFalse(
            prepare_calibration._valid_existing_build(output, manifest["policy"])
        )

    def test_required_source_load_failure_is_fatal(self):
        with (
            mock.patch.object(
                prepare_calibration,
                "load_dataset",
                side_effect=OSError("offline"),
            ),
            self.assertRaisesRegex(RuntimeError, "required source example/data"),
        ):
            prepare_calibration.load_source(
                self.source, self.tokenizer, [], max_document_tokens=100
            )

    def test_source_cap_below_minimum_fails_before_loading(self):
        constrained = {
            **self.source,
            "max_samples": 1,
            "minimum_records": 2,
        }
        with (
            mock.patch.object(prepare_calibration, "load_dataset") as loader,
            self.assertRaisesRegex(ValueError, "below its minimum"),
        ):
            prepare_calibration.load_source(
                constrained, self.tokenizer, [], max_document_tokens=100
            )
        loader.assert_not_called()

    def test_source_minimum_contribution_is_enforced(self):
        constrained = {**self.source, "minimum_records": 2}
        with (
            mock.patch.object(
                prepare_calibration,
                "load_dataset",
                return_value=[{"text": "x" * 80}],
            ),
            self.assertRaisesRegex(ValueError, "minimum is 2"),
        ):
            prepare_calibration.load_source(
                constrained, self.tokenizer, [], max_document_tokens=100
            )


if __name__ == "__main__":
    unittest.main()
