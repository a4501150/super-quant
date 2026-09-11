"""
Shared model loading and tensor mapping utilities.

Used by generate_imatrix.py and sensitivity_analysis.py.
"""

import glob
import json
import os
import random
import re
import sys
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


def _validate_calibration(value: Any, path: str) -> None:
    data = _validate_object(
        value,
        path,
        required={"domains", "num_samples", "max_length"},
    )
    _validate_string_list(data["domains"], f"{path}.domains")
    for key in ("num_samples", "max_length"):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError(f"{path}.{key} must be a positive integer")


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


def _validate_awq_common(data: dict[str, Any], path: str) -> None:
    _validate_calibration(data["calibration"], f"{path}.calibration")
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
            "calibration",
            "mappings",
            "duo_scaling",
            "n_grid",
            "groups",
            "ignore",
            "kv_cache",
            "gptq",
        },
    )
    _validate_awq_common(nvfp4, "recipe.nvfp4")
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


def inspect_source_config(recipe: dict[str, Any]) -> dict[str, Any]:
    source = recipe["source"]
    config, _ = PretrainedConfig.get_config_dict(
        source["model_id"], trust_remote_code=source["trust_remote_code"]
    )
    configs = [("config", config)]
    if isinstance(config.get("text_config"), dict):
        configs.append(("text_config", config["text_config"]))
    for config_name, model_config in configs:
        quantization = model_config.get("quantization_config")
        if quantization:
            method = quantization.get("quant_method", "unknown")
            status = quantization.get("quantization_status", "configured")
            raise ValueError(
                f"Source model is already quantized in {config_name} "
                f"({method}, status={status}): {source['model_id']}. "
                "Use a source-precision checkpoint."
            )
    return config


def load_quantization_model(recipe: dict[str, Any]) -> torch.nn.Module:
    from llmcompressor.utils import load_context

    inspect_source_config(recipe)
    source = recipe["source"]
    model_class = get_quantization_model_class(recipe)
    with load_context(model_class):
        return model_class.from_pretrained(
            source["model_id"],
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=source["trust_remote_code"],
        )


def load_quantization_tokenizer(recipe: dict[str, Any]):
    source = recipe["source"]
    return AutoTokenizer.from_pretrained(
        source["model_id"], trust_remote_code=source["trust_remote_code"]
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


def load_calibration_texts(calibration_dir: str, domains: list[str]) -> list[str]:
    samples = []
    for domain in domains:
        path = Path(calibration_dir) / f"{domain}.txt"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping domain '{domain}'")
            continue
        chunks = re.split(r"\n{2,}", path.read_text(encoding="utf-8"))
        domain_samples = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
        print(f"  {domain}: {len(domain_samples)} samples from {path}")
        samples.extend(domain_samples)
    if not samples:
        raise ValueError("No calibration samples were loaded")
    return samples


def build_calibration_dataset(samples, tokenizer, num_samples: int, max_length: int):
    import datasets

    shuffled = list(samples)
    random.Random(42).shuffle(shuffled)
    tokenized = []
    for text in shuffled[:num_samples]:
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        tokenized.append(
            {key: value.squeeze(0).tolist() for key, value in inputs.items()}
        )
    if not tokenized:
        raise ValueError("No calibration samples remained after tokenization")
    return datasets.Dataset.from_dict(
        {key: [item[key] for item in tokenized] for key in tokenized[0]}
    )


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
