import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_utils
import quantize_nvfp4


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

    def test_boolean_sample_count_is_rejected(self):
        recipe = copy.deepcopy(self.recipe_data)
        recipe["nvfp4"]["calibration"]["num_samples"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            model_utils.load_quantization_recipe(self.write_recipe(recipe))

    def test_task_selects_image_text_auto_model(self):
        recipe = model_utils.load_quantization_recipe(self.recipe_path)
        self.assertIs(
            model_utils.get_quantization_model_class(recipe),
            model_utils.AutoModelForImageTextToText,
        )

    def test_nvfp4_recipe_enables_static_fp8_kv_calibration(self):
        recipe = model_utils.load_quantization_recipe(self.recipe_path)
        gptq = quantize_nvfp4.build_recipe(recipe["nvfp4"])[1]
        self.assertIsNotNone(gptq.kv_cache_scheme)
        self.assertEqual(gptq.kv_cache_scheme.num_bits, 8)
        self.assertEqual(gptq.kv_cache_scheme.type, "float")
        self.assertFalse(gptq.kv_cache_scheme.dynamic)


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
