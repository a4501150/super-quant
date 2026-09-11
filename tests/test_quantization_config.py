import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_utils
import quantize_nvfp4


class MockTokenizer:
    eos_token_id = 0
    pad_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % 97 + 1 for character in text]

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        text = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return {"input_ids": self.encode(text)}


class MockRouter(torch.nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, hidden_states):
        rows = hidden_states.shape[0]
        indices = torch.arange(rows, device=hidden_states.device).remainder(
            self.num_experts
        )[:, None]
        return torch.zeros(rows, self.num_experts), torch.ones(rows, 1), indices


class MockRoutedMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = MockRouter()


class MockRoutedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MockRoutedMlp()


class MockRoutedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(128, 4)
        self.layers = torch.nn.ModuleList([MockRoutedBlock(), MockRoutedBlock()])
        self.config = SimpleNamespace(num_experts=4, num_experts_per_tok=1)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False):
        del attention_mask, position_ids, use_cache
        hidden_states = self.embed_tokens(input_ids).reshape(-1, 4)
        for layer in self.layers:
            layer.mlp.gate(hidden_states)
        return hidden_states


class MockMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.up_proj = torch.nn.Linear(4, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 4, bias=False)


class MockBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MockMlp()


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([MockBlock(), MockBlock()])


class QuantizationRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe_path = ROOT / "configs" / "Qwen3.8-27B-AEON" / "quantize.json"
        cls.recipe_data = json.loads(cls.recipe_path.read_text())

    def write_recipe(self, recipe):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "quantize.json"
        path.write_text(json.dumps(recipe))
        return path

    def test_all_checked_in_recipes_validate(self):
        paths = sorted((ROOT / "configs").glob("*/quantize.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                model_utils.load_quantization_recipe(path)

    def test_flash_next_uses_source_precision_and_excludes_ple(self):
        path = ROOT / "configs" / "Qwen3.8-Flash-Next" / "quantize.json"
        recipe = model_utils.load_quantization_recipe(path)
        self.assertEqual(recipe["source"]["model_id"], "Qwen/Qwen3.8-Flash-Next")
        self.assertIn("re:.*\\.ple\\..*", recipe["nvfp4"]["ignore"])
        self.assertTrue(recipe["nvfp4"]["gptq"]["offload_hessians"])

    def test_unknown_key_is_rejected(self):
        recipe = copy.deepcopy(self.recipe_data)
        recipe["source"]["architecture"] = "qwen"
        with self.assertRaisesRegex(ValueError, "unknown keys: architecture"):
            model_utils.load_quantization_recipe(self.write_recipe(recipe))

    def test_duplicate_group_target_is_rejected(self):
        recipe = copy.deepcopy(self.recipe_data)
        duplicate = recipe["nvfp4"]["groups"][0]["targets"][0]
        recipe["nvfp4"]["groups"][1]["targets"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "multiple groups"):
            model_utils.load_quantization_recipe(self.write_recipe(recipe))

    def test_source_quantization_is_rejected_before_loading(self):
        recipe = model_utils.load_quantization_recipe(self.recipe_path)
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "quantization_status": "compressed",
            }
        }
        with (
            mock.patch.object(
                model_utils.PretrainedConfig,
                "get_config_dict",
                return_value=(config, {}),
            ),
            self.assertRaisesRegex(ValueError, "already quantized"),
        ):
            model_utils.inspect_source_config(recipe)

    def test_nested_source_quantization_is_rejected(self):
        recipe = model_utils.load_quantization_recipe(self.recipe_path)
        config = {
            "text_config": {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed",
                }
            }
        }
        with (
            mock.patch.object(
                model_utils.PretrainedConfig,
                "get_config_dict",
                return_value=(config, {}),
            ),
            self.assertRaisesRegex(ValueError, "already quantized in text_config"),
        ):
            model_utils.inspect_source_config(recipe)

    def test_boolean_token_budget_is_rejected(self):
        recipe = copy.deepcopy(self.recipe_data)
        recipe["nvfp4"]["weight_calibration"]["token_budget"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            model_utils.load_quantization_recipe(self.write_recipe(recipe))

    def test_budget_must_allocate_each_domain(self):
        recipe = copy.deepcopy(self.recipe_data)
        recipe["nvfp4"]["weight_calibration"]["token_budget"] = 3
        with self.assertRaisesRegex(ValueError, "at least one token per domain"):
            model_utils.load_quantization_recipe(self.write_recipe(recipe))

    def test_domain_weights_must_cover_domains(self):
        recipe = copy.deepcopy(self.recipe_data)
        del recipe["nvfp4"]["weight_calibration"]["domain_weights"]["agentic"]
        with self.assertRaisesRegex(ValueError, "domain_weights"):
            model_utils.load_quantization_recipe(self.write_recipe(recipe))

    def test_image_text_save_copies_preprocessor_config(self):
        recipe = copy.deepcopy(self.recipe_data)
        with (
            tempfile.TemporaryDirectory() as source_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            source_path = Path(source_dir)
            expected = '{"processor_class": "TestProcessor"}\n'
            (source_path / "preprocessor_config.json").write_text(expected)
            recipe["source"]["model_id"] = source_dir
            recipe["source"]["task"] = "image-text-to-text"
            model_utils.save_quantization_processor(recipe, output_dir)
            self.assertEqual(
                (Path(output_dir) / "preprocessor_config.json").read_text(), expected
            )

    def test_task_selects_image_text_auto_model(self):
        recipe = model_utils.load_quantization_recipe(self.recipe_path)
        self.assertIs(
            model_utils.get_quantization_model_class(recipe),
            model_utils.AutoModelForImageTextToText,
        )

    def test_nvfp4_recipe_separates_weights_and_static_fp8_kv(self):
        recipe = model_utils.load_quantization_recipe(self.recipe_path)
        settings = recipe["nvfp4"]
        gptq = quantize_nvfp4.build_weight_recipe(settings)[1]
        kv_modifier = quantize_nvfp4.build_kv_modifier(settings)
        self.assertIsNone(gptq.kv_cache_scheme)
        self.assertEqual(
            set(gptq.config_groups), {group["name"] for group in settings["groups"]}
        )
        self.assertEqual(kv_modifier.targets, [])
        self.assertIsNotNone(kv_modifier.kv_cache_scheme)
        self.assertEqual(kv_modifier.kv_cache_scheme.num_bits, 8)
        self.assertEqual(kv_modifier.kv_cache_scheme.type, "float")
        self.assertFalse(kv_modifier.kv_cache_scheme.dynamic)

    def test_kv_position_offsets_must_fit_context(self):
        recipe = copy.deepcopy(self.recipe_data)
        recipe["nvfp4"]["kv_calibration"]["position_offsets"][-1] = 260_000
        with self.assertRaisesRegex(ValueError, "exceed max_position"):
            model_utils.load_quantization_recipe(self.write_recipe(recipe))


class CalibrationPackingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name)
        self.tokenizer = MockTokenizer()

    def write_domain(self, domain, records):
        path = self.path / f"{domain}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def make_record(self, record_id, domain, *, messages=None, text=None):
        record = {
            "schema_version": 1,
            "id": record_id,
            "domain": domain,
            "category": "test",
            "source": {
                "dataset": "test/data",
                "revision": "a" * 40,
                "split": "train",
            },
            "kind": "messages" if messages is not None else "document",
        }
        if messages is not None:
            record["messages"] = messages
        else:
            record["text"] = text
        return record

    def test_structured_loader_preserves_complete_conversation(self):
        messages = [
            {"role": "user", "content": "first\n\nsecond"},
            {"role": "assistant", "content": "complete answer"},
        ]
        self.write_domain(
            "general", [self.make_record("one", "general", messages=messages)]
        )
        loaded = model_utils.load_calibration_records(self.path, ["general"])
        self.assertEqual(loaded["general"][0]["messages"], messages)

    def test_structured_loader_rejects_malformed_conversation_at_source_line(self):
        malformed = self.make_record("bad", "general", messages=[])
        self.write_domain("general", [malformed])
        with self.assertRaisesRegex(ValueError, r"general\.jsonl:1"):
            model_utils.load_calibration_records(self.path, ["general"])

    def test_overlength_conversation_is_skipped_not_truncated(self):
        messages = [
            {"role": "user", "content": "x" * 80},
            {"role": "assistant", "content": "y" * 80},
        ]
        value = self.make_record("long", "general", messages=messages)
        units, stats = model_utils._calibration_units([value], self.tokenizer, 32)
        self.assertEqual(units, [])
        self.assertEqual(stats["skipped_conversations"], 1)
        self.assertGreater(stats["skipped_conversation_tokens"], 32)

    def test_documents_can_be_segmented_but_conversations_stay_whole(self):
        document = self.make_record("doc", "general", text="d" * 70)
        conversation = self.make_record(
            "chat",
            "general",
            messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        units, stats = model_utils._calibration_units(
            [document, conversation], self.tokenizer, 32
        )
        self.assertEqual(stats["document_segments"], 3)
        self.assertEqual(sum(unit["id"].startswith("chat") for unit in units), 1)
        self.assertTrue(all(len(unit["token_ids"]) <= 32 for unit in units))

    def test_kv_dataset_reaches_configured_long_positions(self):
        self.write_domain(
            "general",
            [self.make_record("long-doc", "general", text="k" * 160)],
        )
        settings = {
            "domains": ["general"],
            "token_budget": 128,
            "sequence_length": 64,
            "domain_weights": {"general": 1.0},
            "minimum_achieved_ratio": 1.0,
            "position_offsets": [0, 192],
            "max_position": 256,
        }
        dataset, _ = model_utils.build_calibration_dataset(
            self.path, self.tokenizer, settings
        )
        self.assertEqual(dataset[0]["position_ids"], list(range(64)))
        self.assertEqual(dataset[1]["position_ids"], list(range(192, 256)))

    def test_real_router_coverage_reports_each_layer_and_domain(self):
        self.write_domain(
            "general",
            [self.make_record("coverage", "general", text="r" * 96)],
        )
        calibration = {
            "domains": ["general"],
            "token_budget": 64,
            "sequence_length": 32,
            "domain_weights": {"general": 1.0},
            "minimum_achieved_ratio": 1.0,
        }
        policy = {
            "router_patterns": ["re:.*mlp\\.gate$"],
            "output_index": 2,
            "num_experts_config_key": "num_experts",
            "top_k_config_key": "num_experts_per_tok",
            "token_budget": 64,
            "minimum_tokens_per_expert": 1,
            "maximum_uncovered_experts": 0,
        }
        report = model_utils.measure_expert_coverage(
            MockRoutedModel(), self.path, self.tokenizer, calibration, policy
        )
        self.assertEqual(len(report["layers"]), 2)
        self.assertTrue(
            all(layer["uncovered"] == 0 for layer in report["layers"].values())
        )
        self.assertEqual(report["domain_assignments"]["general"], 128)
        self.assertEqual(
            json.loads((self.path / "expert_coverage.json").read_text()), report
        )

    def test_real_router_coverage_threshold_fails_closed(self):
        self.write_domain(
            "general",
            [self.make_record("coverage", "general", text="r" * 64)],
        )
        calibration = {
            "domains": ["general"],
            "token_budget": 32,
            "sequence_length": 32,
            "domain_weights": {"general": 1.0},
            "minimum_achieved_ratio": 1.0,
        }
        policy = {
            "router_patterns": ["re:.*mlp\\.gate$"],
            "output_index": 2,
            "num_experts_config_key": "num_experts",
            "top_k_config_key": "num_experts_per_tok",
            "token_budget": 32,
            "minimum_tokens_per_expert": 100,
            "maximum_uncovered_experts": 0,
        }
        with self.assertRaisesRegex(ValueError, "coverage did not meet policy"):
            model_utils.measure_expert_coverage(
                MockRoutedModel(), self.path, self.tokenizer, calibration, policy
            )

    def test_token_budget_packing_is_fixed_length_and_deterministic(self):
        for domain, character in (("general", "g"), ("code", "c")):
            self.write_domain(
                domain,
                [
                    self.make_record(f"{domain}-{index}", domain, text=character * 30)
                    for index in range(5)
                ],
            )
        settings = {
            "domains": ["general", "code"],
            "token_budget": 120,
            "sequence_length": 64,
            "domain_weights": {"general": 0.5, "code": 0.5},
            "minimum_achieved_ratio": 1.0,
        }
        first, report_a = model_utils.build_calibration_dataset(
            self.path, self.tokenizer, settings, seed=3
        )
        second, report_b = model_utils.build_calibration_dataset(
            self.path, self.tokenizer, settings, seed=3
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(report_a, report_b)
        self.assertTrue(all(len(row) == 64 for row in first["input_ids"]))
        self.assertGreaterEqual(report_a["selected_tokens"], 120)
        self.assertEqual(report_a["packing"]["packed_tokens"], 120)


class QuantizationExecutionTest(unittest.TestCase):
    def test_weight_and_kv_calibration_run_as_separate_passes(self):
        recipe_path = ROOT / "configs" / "Qwen3.8-27B-AEON" / "quantize.json"
        recipe = model_utils.load_quantization_recipe(recipe_path)
        weight_dataset = [{"input_ids": [1]}]
        kv_dataset = [{"input_ids": [2]}]
        model = mock.Mock()
        tokenizer = mock.Mock()
        with (
            tempfile.TemporaryDirectory() as output_dir,
            mock.patch.object(
                sys,
                "argv",
                [
                    "quantize_nvfp4.py",
                    "--config",
                    str(recipe_path),
                    "--calibration-dir",
                    output_dir,
                    "--output-dir",
                    output_dir,
                ],
            ),
            mock.patch.object(
                quantize_nvfp4,
                "load_quantization_recipe",
                return_value=recipe,
            ),
            mock.patch.object(
                quantize_nvfp4,
                "load_quantization_tokenizer",
                return_value=tokenizer,
            ),
            mock.patch.object(
                quantize_nvfp4,
                "load_calibration_records",
                return_value={
                    domain: []
                    for domain in recipe["nvfp4"]["weight_calibration"]["domains"]
                },
            ),
            mock.patch.object(
                quantize_nvfp4,
                "build_calibration_dataset",
                side_effect=[
                    (weight_dataset, {"selected_tokens": 10}),
                    (kv_dataset, {"selected_tokens": 20}),
                ],
            ),
            mock.patch.object(
                quantize_nvfp4, "load_quantization_model", return_value=model
            ),
            mock.patch.object(quantize_nvfp4, "validate_awq_mapping_targets"),
            mock.patch.object(
                quantize_nvfp4,
                "validate_quantization_targets",
                return_value={"weights": 1},
            ),
            mock.patch.object(quantize_nvfp4, "save_quantization_processor"),
            mock.patch.object(quantize_nvfp4, "oneshot") as oneshot,
        ):
            quantize_nvfp4.main()

        self.assertEqual(oneshot.call_count, 2)
        weight_call, kv_call = oneshot.call_args_list
        self.assertIs(weight_call.kwargs["dataset"], weight_dataset)
        self.assertIsNone(weight_call.kwargs["recipe"][1].kv_cache_scheme)
        self.assertFalse(weight_call.kwargs["moe_calibrate_all_experts"])
        self.assertIs(kv_call.kwargs["dataset"], kv_dataset)
        self.assertEqual(len(kv_call.kwargs["recipe"]), 1)
        self.assertIsNotNone(kv_call.kwargs["recipe"][0].kv_cache_scheme)
        self.assertEqual(kv_call.kwargs["recipe"][0].targets, [])
        model.save_pretrained.assert_called_once()


class QuantizationTargetTest(unittest.TestCase):
    def test_target_counts_are_reported(self):
        groups = [
            {
                "name": "mlp",
                "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
            }
        ]
        counts = model_utils.validate_quantization_targets(MockModel(), groups, [])
        self.assertEqual(counts, {"mlp": 6})

    def test_target_that_matches_nothing_is_rejected(self):
        groups = [{"name": "attention", "targets": ["re:.*self_attn.*$"]}]
        with self.assertRaisesRegex(ValueError, "matched no modules"):
            model_utils.validate_quantization_targets(MockModel(), groups, [])

    def test_module_in_multiple_groups_is_rejected(self):
        groups = [
            {"name": "all_linear", "targets": ["Linear"]},
            {"name": "mlp", "targets": ["re:.*mlp\\.gate_proj$"]},
        ]
        with self.assertRaisesRegex(ValueError, "multiple quantization groups"):
            model_utils.validate_quantization_targets(MockModel(), groups, [])

    def test_awq_mapping_targets_are_validated(self):
        mappings = [
            {
                "smooth_layer": "re:.*mlp\\.gate_proj$",
                "balance_layers": ["re:.*mlp\\.up_proj$"],
            }
        ]
        model_utils.validate_awq_mapping_targets(MockModel(), mappings)
        mappings[0]["balance_layers"] = ["re:.*self_attn.*$"]
        with self.assertRaisesRegex(ValueError, "AWQ mappings matched no modules"):
            model_utils.validate_awq_mapping_targets(MockModel(), mappings)


if __name__ == "__main__":
    unittest.main()
