"""
Shared model loading and tensor mapping utilities.

Used by generate_imatrix.py and sensitivity_analysis.py.
"""

import contextlib
import glob
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)

CALIBRATION_SCHEMA_VERSION = 1

QUANTIZATION_TASKS = {
    "causal-lm": AutoModelForCausalLM,
    "image-text-to-text": AutoModelForImageTextToText,
}

SAFETENSOR_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def load_safetensor_to_device(path: str, device: str) -> dict[str, torch.Tensor]:
    """Load a safetensor file directly to device without mmap."""
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
        data_offset = 8 + header_size

        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = SAFETENSOR_DTYPES[meta["dtype"]]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            f.seek(data_offset + start)
            raw = f.read(end - start)
            tensors[name] = (
                torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).to(device)
            )

    return tensors


def auto_max_memory() -> dict:
    """Build max_memory dict from available GPUs."""
    if not torch.cuda.is_available():
        return {}
    mem = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        mem[i] = f"{int(total * 0.92 / 1024**3)}GiB"
    return mem


def load_model(
    model_id: str, torch_dtype: torch.dtype, trust_remote_code: bool = False
) -> torch.nn.Module:
    """Load model to GPU(s), bypassing CPU mmap when system RAM is limited."""
    max_memory = auto_max_memory()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        return model
    except RuntimeError as e:
        if "mmap" not in str(e) and "allocate memory" not in str(e):
            raise

    print(
        "  Standard loading failed (system RAM < model size), streaming weights to GPU..."
    )
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_path = snapshot_download(model_id)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        )

    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shard_files:
        print("ERROR: No .safetensors files found")
        sys.exit(1)

    device = "cuda:0"
    expected_keys = set(model.state_dict().keys())
    all_tensors = {}
    for shard_path in shard_files:
        size_gb = os.path.getsize(shard_path) / 1024**3
        print(f"  Loading {os.path.basename(shard_path)} ({size_gb:.1f} GB)...")
        shard_tensors = load_safetensor_to_device(shard_path, device)
        for name, tensor in shard_tensors.items():
            all_tensors[name] = tensor.to(torch_dtype)
        del shard_tensors

    # Auto-detect and strip multimodal wrapper prefix
    overlap = len(set(all_tensors) & expected_keys)
    if overlap < len(expected_keys) // 2:
        prefix_map = {"model.language_model.": "model."}
        for old_prefix, new_prefix in prefix_map.items():
            remapped = {}
            for k, v in all_tensors.items():
                if k.startswith(old_prefix):
                    remapped[new_prefix + k[len(old_prefix) :]] = v
                elif not k.startswith("model.visual."):
                    remapped[k] = v
            new_overlap = len(set(remapped) & expected_keys)
            if new_overlap > overlap:
                skipped = len(all_tensors) - len(remapped)
                print(
                    f"  Remapped keys: stripped '{old_prefix}' prefix ({new_overlap} matches, {skipped} vision keys skipped)"
                )
                all_tensors = remapped
                break

    model.load_state_dict(all_tensors, strict=False, assign=True)
    matched = len(set(all_tensors) & expected_keys)
    print(f"  Loaded {matched}/{len(expected_keys)} expected tensors")
    del all_tensors
    torch.cuda.empty_cache()

    meta_count = 0
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.device.type == "meta":
                new = torch.zeros(param.shape, dtype=param.dtype, device=device)
                setattr(module, name, torch.nn.Parameter(new, requires_grad=False))
                meta_count += 1
        for name, buf in module.named_buffers(recurse=False):
            if buf.device.type == "meta":
                module.register_buffer(
                    name, torch.zeros(buf.shape, dtype=buf.dtype, device=device)
                )
                meta_count += 1
    if meta_count:
        print(f"  Materialized {meta_count} remaining meta tensors")
    return model


def _validate_object(
    value: Any,
    path: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
    unknown = value.keys() - required - optional
    if unknown:
        raise ValueError(f"{path} has unknown keys: {', '.join(sorted(unknown))}")
    return value


def _validate_string_list(value: Any, path: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{path} must be a list of non-empty strings")
    if not value and not allow_empty:
        raise ValueError(f"{path} must not be empty")
    for item in value:
        if item.startswith("re:"):
            try:
                re.compile(item.removeprefix("re:"))
            except re.error as exc:
                raise ValueError(
                    f"{path} contains invalid regex {item!r}: {exc}"
                ) from exc


def _validate_calibration(
    value: Any, path: str, *, long_positions: bool = False
) -> None:
    required = {
        "domains",
        "token_budget",
        "sequence_length",
        "domain_weights",
        "minimum_achieved_ratio",
    }
    if long_positions:
        required.update({"position_offsets", "max_position"})
    data = _validate_object(value, path, required=required)
    _validate_string_list(data["domains"], f"{path}.domains")
    if len(data["domains"]) != len(set(data["domains"])):
        raise ValueError(f"{path}.domains must be unique")
    for key in ("token_budget", "sequence_length"):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError(f"{path}.{key} must be a positive integer")
    if data["token_budget"] < len(data["domains"]):
        raise ValueError(
            f"{path}.token_budget must provide at least one token per domain"
        )
    weights = data["domain_weights"]
    if not isinstance(weights, dict) or set(weights) != set(data["domains"]):
        raise ValueError(f"{path}.domain_weights must define each configured domain")
    if any(
        type(weight) not in {int, float} or weight <= 0 for weight in weights.values()
    ):
        raise ValueError(f"{path}.domain_weights values must be positive numbers")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"{path}.domain_weights must sum to 1")
    ratio = data["minimum_achieved_ratio"]
    if type(ratio) not in {int, float} or not 0 < ratio <= 1:
        raise ValueError(f"{path}.minimum_achieved_ratio must be in (0, 1]")
    if long_positions:
        offsets = data["position_offsets"]
        if (
            not isinstance(offsets, list)
            or not offsets
            or any(type(offset) is not int or offset < 0 for offset in offsets)
        ):
            raise ValueError(f"{path}.position_offsets must be non-negative integers")
        if offsets != sorted(set(offsets)):
            raise ValueError(f"{path}.position_offsets must be sorted and unique")
        if type(data["max_position"]) is not int or data["max_position"] <= 0:
            raise ValueError(f"{path}.max_position must be a positive integer")
        if offsets[-1] + data["sequence_length"] > data["max_position"]:
            raise ValueError(f"{path}.position_offsets exceed max_position")


def _validate_awq_mappings(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be null or a non-empty list")
    for index, item in enumerate(value):
        mapping = _validate_object(
            item,
            f"{path}[{index}]",
            required={"smooth_layer", "balance_layers"},
        )
        if not isinstance(mapping["smooth_layer"], str) or not mapping["smooth_layer"]:
            raise ValueError(f"{path}[{index}].smooth_layer must be a non-empty string")
        _validate_string_list(
            mapping["balance_layers"], f"{path}[{index}].balance_layers"
        )


def _validate_awq_common(
    data: dict[str, Any], path: str, calibration_key: str = "calibration"
) -> None:
    _validate_calibration(data[calibration_key], f"{path}.{calibration_key}")
    _validate_awq_mappings(data["mappings"], f"{path}.mappings")
    if not isinstance(data["duo_scaling"], bool):
        raise TypeError(f"{path}.duo_scaling must be a boolean")
    if type(data["n_grid"]) is not int or data["n_grid"] <= 0:
        raise ValueError(f"{path}.n_grid must be a positive integer")


def load_quantization_recipe(path: str | os.PathLike[str]) -> dict[str, Any]:
    recipe_path = Path(path)
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Quantization recipe not found: {recipe_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {recipe_path}: {exc}") from exc

    root = _validate_object(
        recipe,
        "recipe",
        required={"source", "save", "awq_prescale", "nvfp4"},
        optional={"runtime"},
    )
    source = _validate_object(
        root["source"],
        "recipe.source",
        required={"model_id", "task", "trust_remote_code"},
    )
    if not isinstance(source["model_id"], str) or not source["model_id"]:
        raise ValueError("recipe.source.model_id must be a non-empty string")
    if not isinstance(source["task"], str) or source["task"] not in QUANTIZATION_TASKS:
        choices = ", ".join(sorted(QUANTIZATION_TASKS))
        raise ValueError(f"recipe.source.task must be one of: {choices}")
    if not isinstance(source["trust_remote_code"], bool):
        raise TypeError("recipe.source.trust_remote_code must be a boolean")

    save = _validate_object(root["save"], "recipe.save", required={"shard_size"})
    if not isinstance(save["shard_size"], str) or not save["shard_size"]:
        raise ValueError("recipe.save.shard_size must be a non-empty string")

    if "runtime" in root:
        runtime = _validate_object(
            root["runtime"],
            "recipe.runtime",
            required={
                "batch_size",
                "dataloader_num_workers",
                "sequential_targets_per_subgraph",
                "sequential_prefetch",
                "enable_compile",
            },
            optional={"parallel_onload_workers", "materialize_caches"},
        )
        if runtime.get("parallel_onload_workers") == "auto":
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            # Two threads per rank per usable core-pair, capped so 8 ranks
            # never spawn more copy threads than the node has cores at all.
            runtime["parallel_onload_workers"] = max(
                2, min(16, (os.cpu_count() or 8) // (2 * world_size))
            )
        for key in (
            "batch_size",
            "dataloader_num_workers",
            "sequential_targets_per_subgraph",
            "parallel_onload_workers",
        ):
            minimum = 1 if key in ("batch_size", "sequential_targets_per_subgraph") else 0
            if key in runtime and (
                type(runtime[key]) is not int or runtime[key] < minimum
            ):
                raise ValueError(f"recipe.runtime.{key} has an invalid value")
        for key in ("sequential_prefetch", "enable_compile"):
            if not isinstance(runtime[key], bool):
                raise TypeError(f"recipe.runtime.{key} must be a boolean")

    awq = _validate_object(
        root["awq_prescale"],
        "recipe.awq_prescale",
        required={
            "calibration",
            "mappings",
            "duo_scaling",
            "n_grid",
            "temporary_scheme",
            "targets",
            "ignore",
        },
    )
    _validate_awq_common(awq, "recipe.awq_prescale")
    if not isinstance(awq["temporary_scheme"], str) or not awq["temporary_scheme"]:
        raise ValueError(
            "recipe.awq_prescale.temporary_scheme must be a non-empty string"
        )
    _validate_string_list(awq["targets"], "recipe.awq_prescale.targets")
    _validate_string_list(awq["ignore"], "recipe.awq_prescale.ignore", allow_empty=True)

    nvfp4 = _validate_object(
        root["nvfp4"],
        "recipe.nvfp4",
        required={
            "weight_calibration",
            "kv_calibration",
            "mappings",
            "duo_scaling",
            "n_grid",
            "groups",
            "ignore",
            "kv_cache",
            "gptq",
        },
        optional={"expert_coverage"},
    )
    _validate_awq_common(nvfp4, "recipe.nvfp4", "weight_calibration")
    _validate_string_list(nvfp4["ignore"], "recipe.nvfp4.ignore", allow_empty=True)
    if not isinstance(nvfp4["groups"], list) or not nvfp4["groups"]:
        raise ValueError("recipe.nvfp4.groups must be a non-empty list")
    group_names = set()
    group_patterns = set()
    for index, item in enumerate(nvfp4["groups"]):
        group = _validate_object(
            item,
            f"recipe.nvfp4.groups[{index}]",
            required={"name", "format", "targets"},
        )
        if not isinstance(group["name"], str) or not group["name"]:
            raise ValueError(f"recipe.nvfp4.groups[{index}].name must be non-empty")
        if group["name"] in group_names:
            raise ValueError(f"Duplicate NVFP4 group name: {group['name']}")
        group_names.add(group["name"])
        if group["format"] not in {"nvfp4", "fp8"}:
            raise ValueError(
                f"recipe.nvfp4.groups[{index}].format must be nvfp4 or fp8"
            )
        _validate_string_list(group["targets"], f"recipe.nvfp4.groups[{index}].targets")
        duplicates = group_patterns.intersection(group["targets"])
        if duplicates:
            raise ValueError(
                "NVFP4 target patterns occur in multiple groups: "
                + ", ".join(sorted(duplicates))
            )
        group_patterns.update(group["targets"])

    if nvfp4["kv_cache"] not in {None, "fp8"}:
        raise ValueError("recipe.nvfp4.kv_cache must be null or 'fp8'")
    if nvfp4["kv_cache"] == "fp8":
        _validate_calibration(
            nvfp4["kv_calibration"],
            "recipe.nvfp4.kv_calibration",
            long_positions=True,
        )
    elif nvfp4["kv_calibration"] is not None:
        raise ValueError("recipe.nvfp4.kv_calibration must be null without kv_cache")
    if "expert_coverage" in nvfp4:
        coverage = _validate_object(
            nvfp4["expert_coverage"],
            "recipe.nvfp4.expert_coverage",
            required={
                "router_patterns",
                "output_index",
                "num_experts_config_key",
                "top_k_config_key",
                "token_budget",
                "minimum_tokens_per_expert",
                "maximum_uncovered_experts",
            },
        )
        _validate_string_list(
            coverage["router_patterns"],
            "recipe.nvfp4.expert_coverage.router_patterns",
        )
        for key in (
            "output_index",
            "token_budget",
            "minimum_tokens_per_expert",
            "maximum_uncovered_experts",
        ):
            if type(coverage[key]) is not int or coverage[key] < 0:
                raise ValueError(
                    f"recipe.nvfp4.expert_coverage.{key} must be a non-negative integer"
                )
        if coverage["token_budget"] == 0:
            raise ValueError(
                "recipe.nvfp4.expert_coverage.token_budget must be positive"
            )
        for key in ("num_experts_config_key", "top_k_config_key"):
            if not isinstance(coverage[key], str) or not coverage[key]:
                raise ValueError(
                    f"recipe.nvfp4.expert_coverage.{key} must be a non-empty string"
                )
    gptq = _validate_object(
        nvfp4["gptq"],
        "recipe.nvfp4.gptq",
        required={"block_size", "dampening_frac", "actorder", "offload_hessians"},
    )
    if type(gptq["block_size"]) is not int or gptq["block_size"] <= 0:
        raise ValueError("recipe.nvfp4.gptq.block_size must be a positive integer")
    if (
        type(gptq["dampening_frac"]) not in {int, float}
        or not 0 <= gptq["dampening_frac"] <= 1
    ):
        raise ValueError("recipe.nvfp4.gptq.dampening_frac must be between 0 and 1")
    if not isinstance(gptq["actorder"], str) or not gptq["actorder"]:
        raise ValueError("recipe.nvfp4.gptq.actorder must be a non-empty string")
    if not isinstance(gptq["offload_hessians"], bool):
        raise TypeError("recipe.nvfp4.gptq.offload_hessians must be a boolean")
    return root


def get_quantization_model_class(recipe: dict[str, Any]):
    return QUANTIZATION_TASKS[recipe["source"]["task"]]


def inspect_source_config(
    recipe: dict[str, Any], allow_quantized: bool = False
) -> dict[str, Any]:
    source = recipe["source"]
    config, _ = PretrainedConfig.get_config_dict(
        source["model_id"], trust_remote_code=source["trust_remote_code"]
    )
    configs = [("config", config)]
    if isinstance(config.get("text_config"), dict):
        configs.append(("text_config", config["text_config"]))
    for config_name, model_config in configs:
        quantization = model_config.get("quantization_config")
        if quantization and not allow_quantized:
            method = quantization.get("quant_method", "unknown")
            status = quantization.get("quantization_status", "configured")
            raise ValueError(
                f"Source model is already quantized in {config_name} "
                f"({method}, status={status}): {source['model_id']}. "
                "Use a source-precision checkpoint."
            )
    return config


def streaming_linearize_moe(model):
    import gc

    import tqdm
    from llmcompressor.modeling.moe import linearize
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

    names = [name for name, _module in linearize.get_non_linearized_moes(model)]
    for name in tqdm.tqdm(names, desc="Streaming linearize experts"):
        module = model.get_submodule(name)
        config = getattr(module, "config", model.config)
        linear_experts_class = LinearExperts2D.get_linear_experts_cls(
            module.__class__
        )
        linear_experts = linear_experts_class.from_experts_module(module, config)
        model.set_submodule(name, linear_experts)
        del module, linear_experts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model


@contextlib.contextmanager
def use_streaming_moe_linearization():
    import llmcompressor.modeling.moe.linearize as linearize

    original_linearize_moe = linearize.linearize_moe
    linearize.linearize_moe = streaming_linearize_moe
    try:
        yield
    finally:
        linearize.linearize_moe = original_linearize_moe


@contextlib.contextmanager
def qwen4_exp_linearized_load_mapping(source_config):
    if not source_config.get("super_quant_linearized_moe", False):
        yield
        return
    if source_config.get("model_type") != "qwen4_exp":
        raise ValueError(
            "super_quant_linearized_moe is only supported for qwen4_exp"
        )

    from llmcompressor.modeling.moe import conversion_mappings
    from transformers.core_model_loading import WeightRenaming

    model_type = "qwen4_exp"
    import_paths = (
        "transformers.models.qwen4_exp.configuration_qwen4_exp.Qwen4ExpTextConfig",
        "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextExperts",
    )
    mappings = (
        [],
        [
            WeightRenaming(
                source_patterns=r"\.experts\.(\d+)\.gate_proj\.",
                target_patterns=r".experts.\1.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"\.experts\.(\d+)\.up_proj\.",
                target_patterns=r".experts.\1.up_proj.",
            ),
            WeightRenaming(
                source_patterns=r"\.experts\.(\d+)\.down_proj\.",
                target_patterns=r".experts.\1.down_proj.",
            ),
        ],
    )
    old_import_paths = conversion_mappings.ARCH_TO_IMPORT_PATHS.get(model_type)
    old_mappings = conversion_mappings.ARCH_TO_2D_MAPPINGS.get(model_type)
    original_get_mapping = conversion_mappings.get_checkpoint_conversion_mapping

    def get_checkpoint_conversion_mapping(model_type):
        return original_get_mapping(model_type) or []

    conversion_mappings.ARCH_TO_IMPORT_PATHS[model_type] = import_paths
    conversion_mappings.ARCH_TO_2D_MAPPINGS[model_type] = mappings
    conversion_mappings.get_checkpoint_conversion_mapping = (
        get_checkpoint_conversion_mapping
    )
    try:
        yield
    finally:
        conversion_mappings.get_checkpoint_conversion_mapping = original_get_mapping
        if old_import_paths is None:
            conversion_mappings.ARCH_TO_IMPORT_PATHS.pop(model_type, None)
        else:
            conversion_mappings.ARCH_TO_IMPORT_PATHS[model_type] = old_import_paths
        if old_mappings is None:
            conversion_mappings.ARCH_TO_2D_MAPPINGS.pop(model_type, None)
        else:
            conversion_mappings.ARCH_TO_2D_MAPPINGS[model_type] = old_mappings


@contextlib.contextmanager
def preserve_meta_device_map_for_non_source_rank():
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed or torch.distributed.get_rank() == 0:
        yield
        return

    import transformers.core_model_loading as core_model_loading
    import transformers.modeling_utils as modeling_utils
    from safetensors.torch import _getdtype

    original_get_device_map = modeling_utils._get_device_map
    original_materialize_copy = core_model_loading._materialize_copy

    def get_device_map(model, device_map, max_memory, hf_quantizer):
        if device_map == "meta":
            return {"": torch.device("meta")}
        return original_get_device_map(model, device_map, max_memory, hf_quantizer)

    def materialize_copy_on_meta(tensor, device=None, dtype=None):
        shape = getattr(tensor, "shape", None)
        if shape is None:
            shape = torch.Size(tensor.get_shape())
        tensor_dtype = getattr(tensor, "dtype", None)
        if not isinstance(tensor_dtype, torch.dtype):
            tensor_dtype = _getdtype(tensor.get_dtype())
        return torch.empty(shape, dtype=dtype or tensor_dtype, device="meta")

    modeling_utils._get_device_map = get_device_map
    core_model_loading._materialize_copy = materialize_copy_on_meta
    try:
        yield
    finally:
        core_model_loading._materialize_copy = original_materialize_copy
        modeling_utils._get_device_map = original_get_device_map


def load_quantization_model(
    recipe: dict[str, Any],
    offload_dir: str | os.PathLike[str] | None = None,
    allow_quantized: bool = False,
) -> torch.nn.Module:
    from llmcompressor.utils import load_context

    source_config = inspect_source_config(recipe, allow_quantized=allow_quantized)
    source = recipe["source"]
    model_class = get_quantization_model_class(recipe)
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto_offload" if distributed else "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": source["trust_remote_code"],
    }
    if distributed and offload_dir is not None:
        offload_path = Path(offload_dir)
        offload_path.mkdir(parents=True, exist_ok=True)
        load_kwargs.update(
            offload_folder=str(offload_path),
            max_memory={"cpu": "104GiB"},
        )
    with (
        preserve_meta_device_map_for_non_source_rank(),
        qwen4_exp_linearized_load_mapping(source_config),
        load_context(model_class),
    ):
        model = model_class.from_pretrained(source["model_id"], **load_kwargs)
    return model


def load_quantization_tokenizer(recipe: dict[str, Any]):
    source = recipe["source"]
    return AutoTokenizer.from_pretrained(
        source["model_id"], trust_remote_code=source["trust_remote_code"]
    )


def save_quantization_processor(
    recipe: dict[str, Any], output_dir: str | os.PathLike[str]
) -> None:
    if recipe["source"]["task"] != "image-text-to-text":
        return
    from huggingface_hub import hf_hub_download

    source = recipe["source"]
    source_path = Path(source["model_id"])
    if source_path.is_dir():
        config_path = source_path / "preprocessor_config.json"
    else:
        config_path = Path(
            hf_hub_download(source["model_id"], "preprocessor_config.json")
        )
    (Path(output_dir) / "preprocessor_config.json").write_bytes(
        config_path.read_bytes()
    )


def build_awq_modifier(settings: dict[str, Any]):
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.transform.awq.mappings import AWQMapping

    mappings = settings["mappings"]
    if mappings is not None:
        mappings = [
            AWQMapping(
                smooth_layer=mapping["smooth_layer"],
                balance_layers=mapping["balance_layers"],
            )
            for mapping in mappings
        ]
    return AWQModifier(
        mappings=mappings,
        duo_scaling=settings["duo_scaling"],
        n_grid=settings["n_grid"],
    )


def _validate_calibration_record_payload(
    record: dict[str, Any], path: Path, line_number: int
) -> None:
    location = f"{path}:{line_number}"
    if record.get("kind") == "document":
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Document record has no text in {location}")
        return
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Conversation record has no messages in {location}")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"Invalid message {index} in {location}")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Invalid message role at message {index} in {location}")
        if (
            not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(f"Empty message content at message {index} in {location}")


def load_calibration_records(
    calibration_dir: str | os.PathLike[str], domains: list[str]
) -> dict[str, list[dict[str, Any]]]:
    records_by_domain = {}
    seen_ids = set()
    for domain in domains:
        path = Path(calibration_dir) / f"{domain}.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise ValueError(f"Calibration records not found: {path}") from exc
        records = []
        for line_number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if record.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported calibration schema in {path}:{line_number}: "
                    f"{record.get('schema_version')!r}"
                )
            if record.get("domain") != domain:
                raise ValueError(
                    f"Domain mismatch in {path}:{line_number}: {record.get('domain')!r}"
                )
            if record.get("kind") not in {"messages", "document"}:
                raise ValueError(f"Invalid record kind in {path}:{line_number}")
            _validate_calibration_record_payload(record, path, line_number)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Missing record ID in {path}:{line_number}")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate calibration record ID: {record_id}")
            seen_ids.add(record_id)
            records.append(record)
        if not records:
            raise ValueError(f"No calibration records in {path}")
        records_by_domain[domain] = records
        print(f"  {domain}: {len(records):,} structured records from {path}")
    return records_by_domain


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.squeeze(0).tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _tokenize_calibration_record(record: dict[str, Any], tokenizer) -> list[int]:
    if record["kind"] == "document":
        return tokenizer.encode(record["text"], add_special_tokens=False)
    inputs = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        truncation=False,
    )
    return _as_token_ids(inputs["input_ids"])


def _domain_token_quotas(calibration: dict[str, Any]) -> dict[str, int]:
    domains = calibration["domains"]
    remaining = calibration["token_budget"] - len(domains)
    quotas = {
        domain: 1 + int(remaining * calibration["domain_weights"][domain])
        for domain in domains[:-1]
    }
    quotas[domains[-1]] = calibration["token_budget"] - sum(quotas.values())
    return quotas


def _document_token_segments(text: str, tokenizer, sequence_length: int):
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    current = []
    for paragraph in (part.strip() for part in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        token_ids = tokenizer.encode(paragraph, add_special_tokens=False)
        if len(token_ids) > sequence_length:
            if current:
                yield current
                current = []
            for start in range(0, len(token_ids), sequence_length):
                yield token_ids[start : start + sequence_length]
            continue
        addition = [*separator, *token_ids] if current else token_ids
        if current and len(current) + len(addition) > sequence_length:
            yield current
            current = token_ids
        else:
            current.extend(addition)
    if current:
        yield current


def _calibration_units(
    records: list[dict[str, Any]], tokenizer, sequence_length: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    units = []
    stats = {
        "input_records": len(records),
        "document_segments": 0,
        "skipped_conversations": 0,
        "skipped_conversation_tokens": 0,
    }
    for record in records:
        if record["kind"] == "messages":
            token_ids = _tokenize_calibration_record(record, tokenizer)
            if len(token_ids) > sequence_length:
                stats["skipped_conversations"] += 1
                stats["skipped_conversation_tokens"] += len(token_ids)
                continue
            units.append({"id": record["id"], "token_ids": token_ids})
            continue
        for index, segment in enumerate(
            _document_token_segments(record["text"], tokenizer, sequence_length)
        ):
            units.append({"id": f"{record['id']}:{index}", "token_ids": segment})
            stats["document_segments"] += 1
    return units, stats


def _pack_calibration_units(
    units: list[dict[str, Any]], sequence_length: int, separator_id: int, pad_id: int
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    rows = []
    current = []
    packed_tokens = 0
    separators = 0
    for unit in units:
        token_ids = unit["token_ids"]
        separator = [separator_id] if current else []
        if current and len(current) + len(separator) + len(token_ids) > sequence_length:
            padding = sequence_length - len(current)
            rows.append(
                {
                    "input_ids": current + [pad_id] * padding,
                    "attention_mask": [1] * len(current) + [0] * padding,
                }
            )
            current = []
            separator = []
        current.extend(separator)
        current.extend(token_ids)
        packed_tokens += len(token_ids)
        separators += len(separator)
    if current:
        padding = sequence_length - len(current)
        rows.append(
            {
                "input_ids": current + [pad_id] * padding,
                "attention_mask": [1] * len(current) + [0] * padding,
            }
        )
    padding_tokens = len(rows) * sequence_length - packed_tokens - separators
    return rows, {
        "sequences": len(rows),
        "packed_tokens": packed_tokens,
        "separator_tokens": separators,
        "padding_tokens": padding_tokens,
    }


def build_calibration_dataset(
    calibration_dir: str | os.PathLike[str],
    tokenizer,
    calibration: dict[str, Any],
    *,
    seed: int = 42,
    records_by_domain: dict[str, list[dict[str, Any]]] | None = None,
):
    import datasets

    if records_by_domain is None:
        records_by_domain = load_calibration_records(
            calibration_dir, calibration["domains"]
        )
    sequence_length = calibration["sequence_length"]
    quotas = _domain_token_quotas(calibration)
    selected_units = []
    report = {
        "token_budget": calibration["token_budget"],
        "sequence_length": sequence_length,
        "domains": {},
    }
    for domain_index, domain in enumerate(calibration["domains"]):
        candidates = sorted(records_by_domain[domain], key=lambda record: record["id"])
        random.Random(seed + domain_index).shuffle(candidates)
        selected = []
        selected_tokens = 0
        stats = {
            "input_records": 0,
            "document_segments": 0,
            "skipped_conversations": 0,
            "skipped_conversation_tokens": 0,
        }
        examined_units = 0
        for record in candidates:
            units, record_stats = _calibration_units(
                [record], tokenizer, sequence_length
            )
            for key in stats:
                stats[key] += record_stats[key]
            examined_units += len(units)
            for unit in units:
                selected.append(unit)
                selected_tokens += len(unit["token_ids"])
                if selected_tokens >= quotas[domain]:
                    break
            if selected_tokens >= quotas[domain]:
                break
        achieved_ratio = selected_tokens / quotas[domain]
        if achieved_ratio < calibration["minimum_achieved_ratio"]:
            raise ValueError(
                f"{domain} calibration reached {selected_tokens:,}/{quotas[domain]:,} "
                f"tokens ({achieved_ratio:.1%}); minimum is "
                f"{calibration['minimum_achieved_ratio']:.1%}"
            )
        report["domains"][domain] = {
            **stats,
            "examined_units": examined_units,
            "quota_tokens": quotas[domain],
            "selected_units": len(selected),
            "selected_tokens": selected_tokens,
            "achieved_ratio": achieved_ratio,
        }
        selected_units.extend(selected)

    selected_units.sort(key=lambda unit: unit["id"])
    random.Random(seed + 10_000).shuffle(selected_units)
    separator_id = tokenizer.eos_token_id
    if separator_id is None:
        raise ValueError("Tokenizer must define eos_token_id for sample packing")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = separator_id
    rows, packing = _pack_calibration_units(
        selected_units, sequence_length, separator_id, pad_id
    )
    position_offsets = calibration.get("position_offsets")
    if position_offsets:
        for index, row in enumerate(rows):
            offset = position_offsets[index % len(position_offsets)]
            row["position_ids"] = list(range(offset, offset + sequence_length))
        packing["position_offsets"] = position_offsets
    report["packing"] = packing
    report["selected_tokens"] = sum(
        domain["selected_tokens"] for domain in report["domains"].values()
    )
    report["effective_ratio"] = report["selected_tokens"] / calibration["token_budget"]
    print(json.dumps(report, indent=2, sort_keys=True))
    return (
        datasets.Dataset.from_dict(
            {key: [row[key] for row in rows] for key in rows[0]}
        ),
        report,
    )


def distributed_dataset_partition(dataset, allow_empty=False):
    """Return this rank's deterministic slice of a calibration dataset."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return dataset
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rows = len(dataset)
    base, extra = divmod(rows, world_size)
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    if start == end and not allow_empty:
        raise ValueError(
            f"Calibration dataset has {rows} rows for {world_size} ranks"
        )
    return dataset.select(range(start, end))


_onload_pool = None


def _wrap_forward_with_prefetch(forward_fn, caches):
    import functools

    from compressed_tensors.offload.cache import OffloadCache

    @functools.wraps(forward_fn)
    def prefetching_forward(*args, **kwargs):
        # Tracing inspects offloaded tensors without moving them; skip prefetch
        # whenever onloading is disabled (access would return meta tensors anyway).
        if OffloadCache.onloading_disabled:
            return forward_fn(*args, **kwargs)
        if any(
            isinstance(arg, torch.fx.Proxy) for arg in (*args, *kwargs.values())
        ):
            # fx.symbolic_trace calls wrapped forwards with Proxy arguments;
            # onloading real weights then races subgraph device placement.
            return forward_fn(*args, **kwargs)
        added = []
        futures = []
        for cache in caches:
            for offloaded in cache.offloaded_values.values():
                if (
                    offloaded is None
                    or offloaded in OffloadCache.keep_onloaded_values
                ):
                    continue
                futures.append(_onload_pool.submit(cache.onload, offloaded))
                added.append(offloaded)
        for offloaded, future in zip(added, futures):
            OffloadCache.keep_onloaded_values[offloaded] = future.result()
        try:
            return forward_fn(*args, **kwargs)
        finally:
            # Outside the pipeline's keep-resident phase, the onloaded entries
            # must not outlive the forward or VRAM grows every call.
            if not OffloadCache.offloading_disabled:
                for offloaded in added:
                    OffloadCache.keep_onloaded_values.pop(offloaded, None)

    # compressed-tensors' unwrap_offload_forward re-reads forward through
    # ``__func__``; a bare function would crash it during quantization.
    prefetching_forward.__func__ = prefetching_forward
    return prefetching_forward


def enable_parallel_onload(model, workers: int) -> int:
    """Prefetch each offloaded subtree's parameters concurrently before forward.

    compressed-tensors onloads lazily per parameter access on the calling
    thread, so an experts forward copies its weights serially and every rank
    saturates exactly one core while GPUs wait. ``onload`` on the disk and
    CPU caches is a pure read with its own file handle and no collectives, so
    entries are safe to fetch from a shared thread pool.

    Only the topmost offloaded module per branch is wrapped, and its batch
    covers every cache in the subtree: leaf Linear caches hold one weight
    each, so per-module batching has nothing to parallelize. A wrapped layer
    warms all descendant weights in one concurrent burst; nested forwards
    then hit the keep-onloaded cache.
    """
    global _onload_pool
    from concurrent.futures import ThreadPoolExecutor

    from compressed_tensors.offload.cache import OffloadCache

    if _onload_pool is None or _onload_pool._max_workers < workers:
        if _onload_pool is not None:
            _onload_pool.shutdown(wait=True)
        _onload_pool = ThreadPoolExecutor(max_workers=workers)

    def caches_of(module):
        return [
            cache
            for cache in (module._parameters, module._buffers)
            if isinstance(cache, OffloadCache)
        ]

    wrapped = 0

    def visit(module):
        nonlocal wrapped
        caches = caches_of(module)
        if caches:
            subtree_caches = list(caches)
            for descendant in module.modules():
                if descendant is not module:
                    subtree_caches.extend(caches_of(descendant))
            module.forward = _wrap_forward_with_prefetch(
                module.forward, subtree_caches
            )
            wrapped += 1
        else:
            for child in module.children():
                visit(child)

    visit(model)
    return wrapped


def _pin_ple_lookup_tables_to_cpu(model):
    """Keep out-of-scope heavy weights CPU-resident during calibration.

    Covers the PLE lookup tables and the whole vision tower: text-only
    training never forward-passes vision, and its dispatched cache entries
    were observed to poison the first tracing-triggered onload (meta/
    file-backed stores fault the H2D copy engine with a sticky illegal
    access). Pinning caches to CPU keeps them reachable for state_dict
    passthrough while keeping every CUDA copy on the text path.
    """
    from compressed_tensors.offload.cache import OffloadCache

    pinned = []
    suffix = "ple.ple_embedding.ngram_embedding"
    for name, module in model.named_modules():
        if not (
            name.endswith(suffix)
            or name == "model.visual"
            or name.startswith("model.visual.")
        ):
            continue
        for cache in (module._parameters, module._buffers):
            if isinstance(cache, OffloadCache):
                cache.onload_device = torch.device("cpu")
        pinned.append(name)
    return pinned


def classify_offload_caches(model):
    """Inventory dispatched cache stores: meta / file-view / private RAM.

    File-view entries alias safetensors shards (or any external file) and
    meta entries hold no bytes at all; both have been implicated in sticky
    H2D illegal-access faults during calibration. Returns a stats dict.
    """
    from bisect import bisect_right
    from compressed_tensors.offload.cache import OffloadCache

    maps = []
    try:
        with open("/proc/self/maps") as handle:
            for line in handle:
                parts = line.split(maxsplit=5)
                if len(parts) < 6 or not parts[1].startswith("r"):
                    continue
                try:
                    lo, hi = (int(x, 16) for x in parts[0].split("-"))
                except ValueError:
                    continue
                path = parts[5].strip()
                if path.startswith("/"):
                    maps.append((lo, path))
    except OSError:
        pass
    maps.sort()
    starts = [lo for lo, _ in maps]

    stats = {"meta": 0, "file_view": 0, "private": 0, "none": 0}
    file_backed_names = []
    for mod_name, module in model.named_modules():
        for holder in (module._parameters, module._buffers):
            if not isinstance(holder, OffloadCache):
                continue
            for key, tensor in holder.offloaded_values.items():
                full = f"{mod_name}.{key}" if mod_name else str(key)
                if tensor is None:
                    stats["none"] += 1
                    continue
                if tensor.device.type == "meta":
                    stats["meta"] += 1
                    file_backed_names.append((full, "meta"))
                    continue
                addr = tensor.data_ptr()
                idx = bisect_right(starts, addr) - 1
                if idx >= 0 and addr >= maps[idx][0]:
                    # address falls inside a file mapping (heuristic: next
                    # mapping start bounds it; good enough for a report)
                    stats["file_view"] += 1
                    file_backed_names.append((full, maps[idx][1]))
                else:
                    stats["private"] += 1
    stats["names"] = file_backed_names
    return stats


def materialize_offload_caches(
    model,
    cache_dir,
    source_dir,
    private_limit: int = 2**20,
):
    """Rewrite dispatched cache stores as memory we own.

    Post-dispatch entries may be file views into the safetensors shards or
    meta placeholders; the copy engine faults reading those during
    calibration (sticky illegal access). After this call every entry is
    either a private clone (entries below ``private_limit`` bytes) or a
    shared file mapping under ``cache_dir`` on node-local storage, populated
    once by rank 0 and mapped read-mostly by every rank. Meta entries are
    read from the source checkpoint via its shard index; tensors absent from
    the checkpoint (loaded as placeholders, e.g. unused vision keys) are
    zero-filled and returned in the ``missing`` report list.
    """
    from safetensors.torch import safe_open
    from compressed_tensors.offload.cache import OffloadCache

    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    rank = torch.distributed.get_rank() if distributed else 0

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    index_path = Path(source_dir) / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]

    entries = []
    for mod_name, module in model.named_modules():
        for holder in (module._parameters, module._buffers):
            if not isinstance(holder, OffloadCache):
                continue
            for key, tensor in list(holder.offloaded_values.items()):
                if tensor is None:
                    continue
                full = f"{mod_name}.{key}" if mod_name else str(key)
                entries.append((holder, key, full, tensor))

    missing: list[str] = []
    shards_cache: dict[str, Any] = {}

    def read_source(full: str) -> torch.Tensor:
        shard = weight_map[full]
        if shard not in shards_cache:
            shards_cache[shard] = safe_open(
                str(Path(source_dir) / shard), framework="pt"
            )
        return shards_cache[shard].get_tensor(full)

    def attach(full: str, tensor: torch.Tensor) -> torch.Tensor:
        nbytes = max(tensor.numel() * tensor.element_size(), 1)
        path = cache_dir / (full.replace(".", "_") + ".bin")
        storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=nbytes)
        flat = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(storage)
        return flat[: nbytes // tensor.element_size()].view(tensor.shape)

    stats = {"private": 0, "shared_file": 0, "meta_fixed": 0}
    for holder, key, full, tensor in entries:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes < private_limit:
            if tensor.device.type == "meta":
                new = torch.zeros(tensor.shape, dtype=tensor.dtype)
                if full in weight_map:
                    new.copy_(read_source(full))
                else:
                    missing.append(full)
                stats["meta_fixed"] += 1
            else:
                new = tensor.detach().to("cpu").clone()
                stats["private"] += 1
        elif rank == 0 or not distributed:
            dst = attach(full, tensor)
            if tensor.device.type == "meta":
                if full in weight_map:
                    dst.copy_(read_source(full))
                else:
                    dst.zero_()
                    missing.append(full)
                stats["meta_fixed"] += 1
            else:
                dst.copy_(tensor.detach().to("cpu"))
            stats["shared_file"] += 1
            holder.offloaded_values[key] = dst
            continue
        else:
            # Non-zero ranks attach after the barrier below; no data writes.
            if tensor.device.type == "meta" and full not in weight_map:
                missing.append(full)
            stats["shared_file"] += 1
        holder.offloaded_values[key] = new if nbytes < private_limit else tensor

    if distributed:
        torch.distributed.barrier()
        if rank != 0:
            for holder, key, full, tensor in entries:
                if (
                    tensor.numel() * tensor.element_size() >= private_limit
                    and tensor.numel() > 0
                ):
                    holder.offloaded_values[key] = attach(full, tensor)

    return {"entries": len(entries), "missing": missing, **stats}


@contextlib.contextmanager
def keep_ple_lookup_tables_on_cpu(model):
    """Keep large PLE lookups in RAM while active subgraphs run on the GPU."""
    from compressed_tensors import offload
    from compressed_tensors.utils import patch_attr
    from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline
    from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline

    set_onload_device = offload.set_onload_device

    def set_onload_device_except_ple(target, device):
        result = set_onload_device(target, device)
        _pin_ple_lookup_tables_to_cpu(target)
        return result

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch_attr(offload, "set_onload_device", set_onload_device_except_ple)
        )
        stack.enter_context(
            patch_attr(
                sequential_pipeline,
                "set_onload_device",
                set_onload_device_except_ple,
            )
        )
        stack.enter_context(
            patch_attr(
                data_free_pipeline,
                "set_onload_device",
                set_onload_device_except_ple,
            )
        )
        yield


def _reduce_expert_coverage_across_ranks(local_counts):
    """Sum router coverage counters across ranks on a CPU-side Gloo group.

    The default process group is NCCL and is sized for calibration-phase
    stalls. Ranks can need much longer to stream their final coverage
    sequences from disk, and this reduction involves no tensors on the GPU,
    so it runs on a separate group with an order-of-magnitude longer timeout.
    """
    world_size = torch.distributed.get_world_size()
    group = torch.distributed.new_group(
        backend="gloo", timeout=timedelta(hours=8)
    )
    rank = torch.distributed.get_rank()
    # torch.distributed.gather_object rejects a non-None output list on
    # non-destination ranks.
    gathered = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(local_counts, gathered, dst=0, group=group)
    merged = None
    if rank == 0:
        merged = {
            name: {
                domain: sum(entry[name][domain] for entry in gathered)
                for domain in domain_counts
            }
            for name, domain_counts in gathered[0].items()
        }
    payload = [merged]
    torch.distributed.broadcast_object_list(payload, src=0, group=group)
    return payload[0]


def measure_expert_coverage(
    model: torch.nn.Module,
    calibration_dir: str | os.PathLike[str],
    tokenizer,
    weight_calibration: dict[str, Any],
    policy: dict[str, Any],
    *,
    records_by_domain: dict[str, list[dict[str, Any]]] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    on_domain_complete=None,
) -> dict[str, Any]:
    """Measure routed-expert coverage, one calibration domain at a time.

    With ``checkpoint_path`` set, merged per-domain counts are rewritten
    after each domain and domains already present there are skipped, so a
    crash redoes only the in-flight domain. ``on_domain_complete`` is called
    on every rank after each domain so callers can checkpoint phase state.
    """
    from compressed_tensors.offload import set_onload_device
    from compressed_tensors.utils.match import is_match
    from llmcompressor.utils import get_main_device

    set_onload_device(model, get_main_device())
    text_config = getattr(model.config, "text_config", model.config)
    num_experts = getattr(text_config, policy["num_experts_config_key"])
    top_k = getattr(text_config, policy["top_k_config_key"])
    routers = {
        name: module
        for name, module in model.named_modules()
        if any(is_match(name, module, pattern) for pattern in policy["router_patterns"])
    }
    if not routers:
        raise ValueError("Expert coverage router patterns matched no modules")

    counts = {
        name: {domain: None for domain in weight_calibration["domains"]}
        for name in routers
    }
    state: dict[str, Any] = {"domain": None, "mask": None}

    def make_hook(name):
        def hook(_module, _inputs, output):
            if state["domain"] is None or state["mask"] is None:
                raise RuntimeError(f"Router {name} ran without active coverage context")
            if not isinstance(output, (tuple, list)):
                raise TypeError(f"Router {name} must return a tuple or list")
            if policy["output_index"] >= len(output):
                raise ValueError(
                    f"Router {name} has no output index {policy['output_index']}"
                )
            selected = output[policy["output_index"]]
            if not isinstance(selected, torch.Tensor) or selected.ndim < 2:
                raise TypeError(
                    f"Router {name} selected-expert output must be a tensor"
                )
            selected = selected.reshape(-1, selected.shape[-1])
            if selected.shape[-1] != top_k:
                raise ValueError(
                    f"Router {name} returned top-{selected.shape[-1]}, expected {top_k}"
                )
            mask = state["mask"].to(selected.device)
            if selected.shape[0] != mask.numel():
                raise ValueError(
                    f"Router {name} returned {selected.shape[0]} token rows for "
                    f"an attention mask with {mask.numel()} tokens"
                )
            selected = selected[mask].reshape(-1).to(dtype=torch.int64)
            batch_counts = torch.bincount(selected, minlength=num_experts).cpu()
            if batch_counts.shape[0] != num_experts:
                raise ValueError(f"Router {name} selected an out-of-range expert")
            domain = state["domain"]
            if counts[name][domain] is None:
                counts[name][domain] = batch_counts
            else:
                counts[name][domain] += batch_counts

        return hook

    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in routers.items()
    ]
    model.eval()
    input_device = get_main_device()
    coverage_calibration = {
        **weight_calibration,
        "token_budget": policy["token_budget"],
    }
    domain_budgets = _domain_token_quotas(coverage_calibration)
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    rank = torch.distributed.get_rank() if distributed else 0
    partial: dict[str, Any] = {"token_budget": policy["token_budget"], "domains": {}}
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        loaded = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if loaded.get("token_budget") == policy["token_budget"]:
            partial = loaded
    try:
        for domain_index, domain in enumerate(weight_calibration["domains"]):
            if domain in partial["domains"]:
                print(
                    f"[rank {rank}] coverage domain={domain} restored from checkpoint",
                    flush=True,
                )
                continue
            domain_budget = domain_budgets[domain]
            domain_calibration = {
                **weight_calibration,
                "domains": [domain],
                "token_budget": domain_budget,
                "domain_weights": {domain: 1.0},
            }
            dataset, _ = build_calibration_dataset(
                calibration_dir,
                tokenizer,
                domain_calibration,
                seed=42 + domain_index,
                records_by_domain=records_by_domain,
            )
            # A small domain can legitimately hold fewer sequences than
            # ranks; a rank then contributes no coverage rows for it.
            dataset = distributed_dataset_partition(dataset, allow_empty=True)
            state["domain"] = domain
            for sequence_index, row in enumerate(dataset, start=1):
                state["mask"] = torch.tensor(
                    row["attention_mask"], dtype=torch.bool, device=input_device
                )
                inputs = {
                    key: torch.tensor(
                        value, dtype=torch.long, device=input_device
                    ).unsqueeze(0)
                    for key, value in row.items()
                }
                try:
                    with torch.inference_mode():
                        model(**inputs, use_cache=False)
                finally:
                    state["mask"] = None
                print(
                    f"[rank {rank}] coverage domain={domain} "
                    f"sequence {sequence_index}/{len(dataset)} complete",
                    flush=True,
                )
            state["domain"] = None
            local_domain = {
                name: {
                    domain: domain_counts[domain].cpu()
                    if domain_counts[domain] is not None
                    else torch.zeros(num_experts, dtype=torch.int64)
                }
                for name, domain_counts in counts.items()
            }
            if distributed:
                local_domain = _reduce_expert_coverage_across_ranks(local_domain)
            partial["domains"][domain] = {
                name: domain_counts[domain].tolist()
                for name, domain_counts in local_domain.items()
            }
            if checkpoint_path is not None and rank == 0:
                path = Path(checkpoint_path)
                temporary_path = path.with_name(path.name + ".tmp")
                temporary_path.write_text(
                    json.dumps(partial, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(path)
            if on_domain_complete is not None:
                on_domain_complete(f"coverage:{domain}")
    finally:
        state["domain"] = None
        state["mask"] = None
        for handle in handles:
            handle.remove()

    per_domain = {
        domain: {
            name: partial["domains"].get(domain, {}).get(name, [0] * num_experts)
            for name in counts
        }
        for domain in weight_calibration["domains"]
    }
    report = {
        "policy": policy,
        "num_experts": num_experts,
        "top_k": top_k,
        "layers": {},
        "domain_assignments": {
            domain: sum(sum(values) for values in per_domain[domain].values())
            for domain in weight_calibration["domains"]
        },
    }
    failures = []
    for name in counts:
        total = [
            sum(
                per_domain[domain][name][index]
                for domain in weight_calibration["domains"]
            )
            for index in range(num_experts)
        ]
        sorted_counts = sorted(total)
        uncovered = sum(1 for count in total if count == 0)
        minimum = sorted_counts[0]
        layer_report = {
            "minimum": minimum,
            "median": statistics.median(sorted_counts),
            "p05": sorted_counts[max(0, (num_experts * 5 + 99) // 100 - 1)],
            "uncovered": uncovered,
            "domain_assignments": {
                domain: sum(per_domain[domain][name])
                for domain in weight_calibration["domains"]
            },
        }
        report["layers"][name] = layer_report
        if (
            uncovered > policy["maximum_uncovered_experts"]
            or minimum < policy["minimum_tokens_per_expert"]
        ):
            failures.append(f"{name}: minimum={minimum}, uncovered={uncovered}")
    rendered_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if rank == 0:
        print(rendered_report, end="")
        report_path = Path(calibration_dir) / "expert_coverage.json"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".expert_coverage.",
                suffix=".json",
                dir=report_path.parent,
                delete=False,
            ) as temporary:
                temporary.write(rendered_report)
                temporary_path = Path(temporary.name)
            temporary_path.replace(report_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    if failures:
        raise ValueError(
            "Routed-expert coverage did not meet policy:\n  " + "\n  ".join(failures)
        )
    return report


def validate_awq_mapping_targets(
    model: torch.nn.Module, mappings: list[dict[str, Any]] | None
) -> None:
    if mappings is None:
        return

    from compressed_tensors.utils.match import is_match

    module_list = list(model.named_modules())
    missing = []
    for index, mapping in enumerate(mappings):
        patterns = [mapping["smooth_layer"], *mapping["balance_layers"]]
        for pattern in patterns:
            if not any(is_match(name, module, pattern) for name, module in module_list):
                missing.append(f"mapping {index}: {pattern}")
    if missing:
        raise ValueError("AWQ mappings matched no modules:\n  " + "\n  ".join(missing))


def validate_quantization_targets(
    model: torch.nn.Module,
    groups: list[dict[str, Any]],
    ignore: list[str],
) -> dict[str, int]:
    from compressed_tensors.utils.match import is_match

    counts = {group["name"]: 0 for group in groups}
    pattern_counts = {
        (group["name"], target): 0 for group in groups for target in group["targets"]
    }
    for module_name, module in model.named_modules():
        matched_groups = []
        for group in groups:
            matched_patterns = [
                target
                for target in group["targets"]
                if is_match(module_name, module, target, ignore)
            ]
            if matched_patterns:
                matched_groups.append(group["name"])
                counts[group["name"]] += 1
                for target in matched_patterns:
                    pattern_counts[(group["name"], target)] += 1
        if len(matched_groups) > 1:
            raise ValueError(
                f"Module {module_name} matches multiple quantization groups: "
                + ", ".join(matched_groups)
            )
    missing = [
        f"{group_name}: {target}"
        for (group_name, target), count in pattern_counts.items()
        if count == 0
    ]
    if missing:
        raise ValueError(
            "Quantization targets matched no modules:\n  " + "\n  ".join(missing)
        )
    return counts


def resolve_arch(gguf_arch_key: str, llamacpp_dir: str, n_layers: int):
    """Build HF->GGUF tensor name map from gguf-py."""
    gguf_py_path = os.path.join(llamacpp_dir, "gguf-py")
    if not os.path.isdir(gguf_py_path):
        print(f"ERROR: gguf-py not found at {gguf_py_path}")
        sys.exit(1)

    sys.path.insert(0, gguf_py_path)
    from gguf import MODEL_ARCH, get_tensor_name_map

    arch_map = {name.lower(): arch for name, arch in MODEL_ARCH.__members__.items()}
    arch_key = gguf_arch_key.lower().replace("-", "_")
    if arch_key not in arch_map:
        print(
            f"ERROR: Unknown arch '{gguf_arch_key}'. Available: {', '.join(sorted(arch_map))}"
        )
        sys.exit(1)

    arch = arch_map[arch_key]
    tmap = get_tensor_name_map(arch, n_layers)
    return tmap


def discover_linear_modules(
    model, tmap
) -> tuple[dict[str, tuple[torch.nn.Linear, str]], list[str]]:
    """Find nn.Linear modules that map to GGUF tensor names."""
    mapped = {}
    unmapped = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        gguf_base = tmap.get_name(name)
        if gguf_base is None:
            unmapped.append(name)
            continue

        gguf_name = gguf_base + ".weight"
        mapped[gguf_name] = (module, name)

    return mapped, unmapped


def get_n_layers(model_id: str, trust_remote_code: bool = False) -> int:
    """Get the number of hidden layers from a model's config."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    text_config = getattr(config, "text_config", config)
    return text_config.num_hidden_layers


def tokenize_and_chunk(text: str, tokenizer, context_size: int) -> list[list[int]]:
    """Tokenize text and split into non-overlapping chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), context_size):
        chunk = tokens[i : i + context_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def print_gpu_info():
    """Print GPU information."""
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"GPUs:         {n_gpus}")
    if n_gpus:
        for i in range(n_gpus):
            name = torch.cuda.get_device_properties(i).name
            vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  [{i}] {name} ({vram:.0f} GB)")
