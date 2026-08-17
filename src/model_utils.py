"""
Shared model loading and tensor mapping utilities.

Used by generate_imatrix.py and sensitivity_analysis.py.
"""

import glob
import json
import os
import re
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

SAFETENSOR_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
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
            tensors[name] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).to(device)

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


def load_model(model_id: str, torch_dtype: torch.dtype, trust_remote_code: bool = False) -> torch.nn.Module:
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

    print("  Standard loading failed (system RAM < model size), streaming weights to GPU...")
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_path = snapshot_download(model_id)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

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
                    remapped[new_prefix + k[len(old_prefix):]] = v
                elif not k.startswith("model.visual."):
                    remapped[k] = v
            new_overlap = len(set(remapped) & expected_keys)
            if new_overlap > overlap:
                skipped = len(all_tensors) - len(remapped)
                print(f"  Remapped keys: stripped '{old_prefix}' prefix ({new_overlap} matches, {skipped} vision keys skipped)")
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
                module.register_buffer(name, torch.zeros(buf.shape, dtype=buf.dtype, device=device))
                meta_count += 1
    if meta_count:
        print(f"  Materialized {meta_count} remaining meta tensors")
    return model


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
        print(f"ERROR: Unknown arch '{gguf_arch_key}'. Available: {', '.join(sorted(arch_map))}")
        sys.exit(1)

    arch = arch_map[arch_key]
    tmap = get_tensor_name_map(arch, n_layers)
    return tmap


def discover_linear_modules(model, tmap) -> tuple[dict[str, tuple[torch.nn.Linear, str]], list[str]]:
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
