#!/usr/bin/env python3
"""Build a ModelOpt NVFP4-experts / FP8-attention checkpoint.

Two placement modes, selected by ``modelopt.device_map`` in the config:

* ``"cuda"`` / ``"auto"`` — single-process calibration (no DDP, no torchrun):
  the model is placed with an HF ``device_map`` — one device, or sharded
  across a node's GPUs.
* ``"fsdp2"`` — the NVIDIA blessed distributed path (mirrors
  ``examples/hf_ptq`` in Model-Optimizer): launch with ``torchrun
  --nproc_per_node=8``, decoder layers are FSDP2-sharded via
  ``parallel_load_and_prepare_fsdp2`` (round-robin parallel safetensors
  reads + broadcasts), calibration is data-parallel over per-rank slices of
  the packed dataset, and ``mtq.quantize`` performs the amax all-reduces
  internally. All checkpoint/file I/O is rank-0 only.

Stage checkpoints land in ``--checkpoint-dir`` and, when ``--upload-prefix``
is set, are mirrored to GCS after each stage so a late failure never loses
the calibration work:

  1. dataset_report.json  coverage per domain + packing parameters
  2. mtq_state.pt         calibrated quantizer state (mto.save); re-export
                          is possible without recalibrating
  3. manifest.json        census of enabled quantizers + sha256 of shards
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import traceback
import warnings

# Queued for the +sq4 wheel (FORK.md): modelopt's loader still passes the
# pre-rename ``torch_dtype=`` kwarg to HF factories; transformers 5 warns
# once per rank at model load. Cosmetic rename, filter it in-process (the
# PYTHONWARNINGS module-scope filter never matched — the warning is emitted
# from inside transformers' frame).
warnings.filterwarnings("ignore", message=r".*torch_dtype is deprecated.*",
                        category=UserWarning)
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import build_calibration_dataset, load_calibration_records


def load_model(model_id: str, task: str, device_map: str, trust: bool):
    """Task-aware load: causal-LM checkpoints and VLM wrappers (text-only use)."""
    if "image" in task or "vision" in task:
        from transformers import AutoModelForImageTextToText

        loader = AutoModelForImageTextToText
    else:
        loader = AutoModelForCausalLM
    kwargs = {}
    if device_map == "auto":
        # device_map=auto plans every free byte of each GPU, but loading the
        # linearized checkpoint merges expert tensors on the target device
        # (torch.stack in core_model_loading), spiking ~2 GiB above plan and
        # OOMing. Cap each device below capacity to leave merge/activation
        # headroom; the >72 GiB PLE table then offloads to CPU on its own.
        kwargs["max_memory"] = {
            "cpu": "700GiB",
            **{i: "72GiB" for i in range(torch.cuda.device_count())},
        }
    return loader.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=trust,
        **kwargs,
    )


def input_device(model) -> torch.device:
    """Device where input_ids must land under a sharded device_map."""
    return model.get_input_embeddings().weight.device


@dataclass
class DistributedState:
    """Minimal view of the torchrun process group (single-process defaults)."""

    rank: int = 0
    world_size: int = 1
    device: Any = None
    is_main: bool = True


def setup_fsdp2() -> DistributedState:
    """Join the torchrun process group, mirroring examples/hf_ptq.

    The 2 h collective timeout is set at process-group creation (PyTorch has
    no per-call barrier timeout): rank 0's export write, manifest hashing and
    GCS sync run while the peers wait at the teardown barrier, which can
    exceed NCCL's 30-min default.
    """
    from datetime import timedelta

    from modelopt.torch.utils import distributed as dist_utils

    dist_utils.setup(timeout=timedelta(hours=2))
    # Every rank packs the identical corpus deterministically and shards it
    # afterwards; isolate the HF datasets disk cache per rank so concurrent
    # cached-map writes don't race on the same fingerprint path.
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        tempfile.gettempdir(), f"hf_datasets_cache_rank{dist_utils.rank()}"
    )
    return DistributedState(
        rank=dist_utils.rank(),
        world_size=dist_utils.size(),
        device=torch.device(f"cuda:{dist_utils.local_rank()}"),
        is_main=dist_utils.rank() == 0,
    )


def _checkpoint_uses_vlm_layout(model_id: str) -> bool:
    """True when the safetensors index keys live under ``model.language_model.*``."""
    index = Path(model_id) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    keys = json.loads(index.read_text()).get("weight_map", {})
    return any(k.startswith("model.language_model.") for k in keys)


def load_model_fsdp2(model_id: str, trust: bool, state: DistributedState):
    """ModelOpt's FSDP2 loader: round-robin parallel safetensors reads +
    broadcasts, ``fully_shard`` on every decoder layer and the root.

    Checkpoint keys absent from the text-only causal-LM class built from the
    pre-fetched ``hf_config`` (vision/MTP aux weights) are logged and skipped
    by the loader.

    Workaround for a loader gap with VLM-layout checkpoints: the loader
    computes HF's conversion plan *after* ``fsdp2_wrap``, at which point
    transformers' rename engine silently stops stripping the
    ``model.language_model.`` prefix (verified: pre-wrap the plan maps
    ``model.language_model.layers.N.*`` to ``model.layers.N.*``; post-wrap it
    leaves the key unrenamed, so every decoder-layer key classifies as
    "skipped" and the first broadcast group is empty -> RuntimeError).
    Injecting the equivalent legacy regex rename (the mechanism
    ``_resolve_target`` applies before the converter path) restores the
    pre-wrap resolution.
    """
    from transformers import AutoConfig

    from modelopt.torch.utils.plugins import model_load_utils as mu

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    # Drop the n-gram PLE table from the calibration model. It is a single
    # 51.2B-param (95 GiB BF16) tensor inside decoder layer 2: too big to
    # broadcast-stage and too big to un-shard per rank on 80 GiB cards. The
    # table is on the BF16 keep-list (never quantized), so skipping its
    # construction (text_config.ple_layer_ids -> []) changes no quantizer
    # state; the forward guards on ``self.ple is not None`` and skips the
    # residual injection consistently, and reattach_skipped_weights() copies
    # the tensors + the original ple_layer_ids back into the export so the
    # SERVED model keeps PLE. Calibration-time caveat: layers 2..N see
    # hidden states without the layer-2 PLE injection; branch magnitude is
    # measured post-hoc, see scripts/measure_ple_residual.py.
    tc = getattr(hf_config, "text_config", hf_config)
    ple_layers = list(getattr(tc, "ple_layer_ids", None) or [])
    if ple_layers:
        tc.ple_layer_ids = []
        print(
            f"dropped PLE table for calibration (ple_layer_ids={ple_layers}); "
            "reattached verbatim at export",
            flush=True,
        )
    # broadcast_chunk_size=1: the loader default (8 decoder layers per
    # broadcast collective) staged the whole group's full weights in one
    # CUDA buffer (95 GiB on an 80 GiB card, on top of the ~44 GiB FSDP2
    # shard) and OOMed on the first group. One layer per collective keeps the
    # transient at ~12 GiB.

    original_plan = mu._conversion_plan
    patched = _checkpoint_uses_vlm_layout(model_id)
    if patched:

        def _conversion_plan_with_vlm_rename(model):
            plan = original_plan(model)
            if plan is not None and not plan["legacy_renames"]:
                plan = {
                    **plan,
                    "legacy_renames": {r"^model\.language_model\.": "model."},
                }
            return plan

        mu._conversion_plan = _conversion_plan_with_vlm_rename

    try:
        model = mu.parallel_load_and_prepare_fsdp2(
            model_id,
            state.device,
            state.rank,
            state.world_size,
            trust_remote_code=trust,
            attn_implementation=None,
            hf_config=hf_config,
            broadcast_chunk_size=1,
        )
    finally:
        if patched:
            mu._conversion_plan = original_plan
    if getattr(model.config, "architectures", None) is None:
        # AutoConfig leaves architectures unset (from_pretrained fills it);
        # modelopt's export probes it in is_multimodal_model and crashed on
        # None. Restore the checkpoint's own value so the exported config
        # advertises the same class vLLM must load.
        model.config.architectures = list(
            getattr(hf_config, "architectures", None) or [type(model).__name__]
        )
    return model, ple_layers


def reattach_skipped_weights(model_id: str, export_dir: str, ple_layer_ids: list[int]) -> None:
    """Copy checkpoint tensors the FSDP2 load skipped into the exported
    checkpoint (PLE n-gram table, vision tower, MTP) and restore
    ``ple_layer_ids`` in the exported config.

    Pure tensor surgery: these weights are outside the quant map, so the
    bytes from the source checkpoint are exactly what serving needs. Each
    source shard is streamed one at a time; the 95 GiB PLE tensor lives in a
    shard of its own.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_map = json.loads(
        (Path(model_id) / "model.safetensors.index.json").read_text()
    )["weight_map"]
    out_index_path = Path(export_dir) / "model.safetensors.index.json"
    out_index = json.loads(out_index_path.read_text())
    missing = sorted(set(src_map) - set(out_index["weight_map"]))
    print(f"reattaching {len(missing)} skipped tensors from {model_id}", flush=True)

    # The janitor removes the staged checkpoint before export; the aux shards
    # were copied to the tmpfs staging dir by scripts/ple_shm_prep.py, so
    # prefer those when present.
    shm_manifest = Path("/dev/shm/flashnext-ple/manifest.json")
    shm_shards = (
        {s: e["path"] for s, e in json.loads(shm_manifest.read_text())["shards"].items()}
        if shm_manifest.is_file()
        else {}
    )

    by_shard: dict[str, list[str]] = {}
    for k in missing:
        by_shard.setdefault(src_map[k], []).append(k)
    for shard, keys in sorted(by_shard.items()):
        tensors = {}
        source = Path(shm_shards.get(shard) or (Path(model_id) / shard))
        with safe_open(source, framework="pt", device="cpu") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
        name = f"reattached-{shard}"
        save_file(tensors, str(Path(export_dir) / name), metadata={"format": "pt"})
        for k in keys:
            out_index["weight_map"][k] = name
        del tensors

    out_index_path.write_text(json.dumps(out_index, indent=2))
    if ple_layer_ids:
        cfg_path = Path(export_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text())
        (cfg.get("text_config") or cfg)["ple_layer_ids"] = ple_layer_ids
        cfg_path.write_text(json.dumps(cfg, indent=2))


_DTYPE_BY_ST = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
}


def inject_ple_modules(
    model, model_id: str, ple_layers: list[int], shm_dir: str = "/dev/shm/flashnext-ple"
) -> None:
    """Re-attach PLE layers the FSDP2 loader skipped (see load_model_fsdp2).

    The n-gram table stays CPU-resident and is *mmapped* from the shared tmpfs
    shard: all ranks read one physical copy (no 8x95 GiB CPU copies, invisible
    to FSDP2 which wrapped the layers before injection). A forward bridge
    moves activations to CPU, runs the module's real math, and returns the
    residual branch to the caller's CUDA device — exact numerics, a few
    hundred MB of transfers and a couple of CPU-seconds per batch.

    After injection the config's ``ple_layer_ids`` is restored, so the parent
    forward passes ``ple_input_ids`` (it gates on the non-empty list) and the
    exported config advertises PLE as present.
    """
    import mmap
    import struct
    import types

    import numpy as np
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    meta = json.loads((Path(shm_dir) / "manifest.json").read_text())
    hdrs: dict[str, dict] = {}

    def header(shard: str) -> dict:
        if shard not in hdrs:
            p = Path(meta["shards"][shard]["path"])
            with p.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdrs[shard] = {"j": json.loads(f.read(n)), "ds": 8 + n, "p": str(p)}
        return hdrs[shard]

    def shm_view(shard: str, key: str) -> torch.Tensor:
        h = header(shard)
        info = h["j"][key]
        s, e = info["data_offsets"]
        shape = tuple(info["shape"])
        raw = np.memmap(h["p"], dtype=np.uint16, mode="r", shape=((e - s) // 2,), offset=h["ds"] + s)
        return (
            torch.from_numpy(raw).view(_DTYPE_BY_ST[info["dtype"]]).reshape(shape)
        )

    tc = getattr(model.config, "text_config", model.config)
    for i, one_based in enumerate(ple_layers):
        layer_idx = one_based - 1
        layer = model.model.layers[layer_idx]
        with init_empty_weights():
            ple = Qwen4ExpTextPLELayer(tc, layer_idx, i)
        prefix = f"model.language_model.layers.{layer_idx}.ple."
        import re as _re

        # The linearizer splits the 51.2B n-gram table into row-block shards
        # named "...ngram_embedding.shard_<k>.weight" (concat on dim 0 = the
        # single nn.Embedding weight). Group shards by their de-sharded attr
        # path + row index, concat in order, then set that one attribute.
        def deshard(n: str) -> tuple[str, int]:
            m = _re.search(r"\.shard_(\d+)\.", n)
            if not m:
                return n, 0
            return n[: m.start()] + n[m.end() - 1:], int(m.group(1))

        grouped: dict[str, list[tuple[int, torch.Tensor]]] = {}
        for key, shard in meta["keys"].items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            attr, idx = deshard(name)
            path = Path(meta["shards"][shard]["path"])
            if "ngram_embedding" in key:
                t = shm_view(shard, key)
            else:
                with safe_open(path, framework="pt", device="cpu") as f:
                    t = f.get_tensor(key)
            grouped.setdefault(attr, []).append((idx, t))

        n_loaded = 0
        keep: list = []  # hold np.memmap owners alive for the views' lifetime
        for attr, parts in grouped.items():
            parts.sort(key=lambda p: p[0])
            if len(parts) == 1:
                parent_mod = ple
                for part in attr.split(".")[:-1]:
                    parent_mod = getattr(parent_mod, part)
                keep.append(parts[0][1])
                setattr(parent_mod, attr.split(".")[-1],
                        nn.Parameter(parts[0][1], requires_grad=False))
                n_loaded += 1
                continue
            # Sharded n-gram table: replace the nn.Embedding with a lookup
            # over the mmap views. A dim-0 concat would materialize 51.2 GiB
            # of *private* RAM per rank (8 ranks > 900 GiB pod cgroup: OOM);
            # the views keep the single shared page-cache copy.
            wts = [p[1] for p in parts]
            keep.extend(wts)

            class _ShardedTable(nn.Module):
                def __init__(self, ws):
                    super().__init__()
                    self.ws = ws
                    self.rows = ws[0].shape[0]
                    self._w = ws[0]

                @property
                def weight(self):  # parent reads .weight.device for placement
                    return self._w

                def forward(self, ids):
                    flat = ids.reshape(-1)
                    sid = torch.div(flat, self.rows, rounding_mode="floor")
                    idx = flat - sid * self.rows
                    out = torch.empty(flat.shape[0], self._w.shape[1],
                                      dtype=self._w.dtype)
                    order = torch.argsort(sid)
                    pos = 0
                    for s, c in enumerate(
                            torch.bincount(sid, minlength=len(self.ws)).tolist()):
                        if c:
                            sel = order[pos:pos + c]
                            out[sel] = self.ws[s].index_select(0, idx[sel])
                            pos += c
                    return out.reshape(*ids.shape, self._w.shape[1])

                def to(self, *a, **k):  # views are CPU-pinned; nothing to move
                    return self

            pos = 0
            holder_attr, table_attr = attr.rsplit(".", 2)[0], attr.rsplit(".", 2)[1]
            holder = ple
            for part in holder_attr.split("."):
                holder = getattr(holder, part)
            setattr(holder, table_attr, _ShardedTable(wts))
            n_loaded += 1
        ple._ple_views_keep = keep
        assert n_loaded, f"no staged PLE tensors for layer {layer_idx}"
        ple.eval()

        orig_forward = ple.forward
        from_cpu = lambda x: x.cpu() if torch.is_tensor(x) else x  # noqa: E731

        def bridge(*args, _fn=orig_forward, **kwargs):
            dev = next(
                (a.device for a in args if torch.is_tensor(a) and a.is_cuda), None
            )
            out = _fn(
                *(from_cpu(a) for a in args),
                **{k: from_cpu(v) for k, v in kwargs.items()},
            )
            if isinstance(out, torch.Tensor):
                out = out.to(dev) if dev is not None else out
            return out

        ple.forward = types.MethodType(lambda _m, *a, _fn=orig_forward, **k: bridge(*a, **k), ple)
        layer.ple = ple
        print(
            f"PLE re-attached at layer {layer_idx} ({n_loaded} tensors, "
            "table mmap-shared from tmpfs, forward CPU-bridged)",
            flush=True,
    )
    tc.ple_layer_ids = list(ple_layers)


def build_forward_loop(model, dataset, tokenizer, batch_size, device=None):
    pad_id = tokenizer.pad_token_id

    def collate(rows):
        # Batches are length-bucketed by forward_loop, so no padding here;
        # use the packer's own mask (it marks tail pads and separators).
        input_ids = torch.tensor([row["input_ids"] for row in rows])
        attention_mask = torch.tensor([row["attention_mask"] for row in rows])
        # Under FSDP2 the embedding weight is a DTensor; pass this rank's
        # local CUDA device explicitly instead.
        dev = device if device is not None else input_device(model)
        return {
            "input_ids": input_ids.to(dev),
            "attention_mask": attention_mask.to(dev),
        }

    def forward_loop(calibrate_model):
        import time as _t
        from itertools import groupby

        # Length-bucketed iteration (long_sequences puts 32K/16K rows ahead of
        # the 2048 rows on every rank): one uniform-length batch group at a
        # time, so batches need no padding and each rank's per-bucket batch
        # count — hence its collective schedule — is a rank-invariant fact.
        lengths = [len(x) for x in dataset["input_ids"]]
        order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
        t0 = _t.monotonic()
        done = 0
        for neg_len, idxs in groupby(order, key=lambda i: -lengths[i]):
            loader = DataLoader(
                dataset.select(list(idxs)),
                batch_size=batch_size,
                collate_fn=collate,
            )
            for batch in loader:
                calibrate_model(**batch)
                done += 1
                print(f"[calib] batch {done} len={-neg_len} "
                      f"shape={tuple(batch['input_ids'].shape)} "
                      f"elapsed={_t.monotonic() - t0:.1f}s", flush=True)
        print(f"[calib] forward loop done: {done} batches, "
              f"{_t.monotonic() - t0:.1f}s total", flush=True)

    return forward_loop


def quantizer_census(model) -> dict[str, int]:
    """Enabled-quantizer counts keyed by component class (checklist item 1)."""
    buckets: Counter[str] = Counter()
    for name, module in model.named_modules():
        wq = getattr(module, "weight_quantizer", None)
        if wq is None or not wq.is_enabled:
            continue
        for cls in ("mlp.experts", "shared_expert", "self_attn", "linear_attn"):
            if cls in name:
                buckets[cls] += 1
                break
        else:
            buckets["other"] += 1
    return dict(buckets)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _phase_config(recipe_yaml: str, phase: str, out_dir: str):
    """Load a recipe variant restricted to one algorithm family.

    ``--calib-phase awq|gptq`` splits the chain into two quantize calls so
    ``mto.save`` + staged weights capture the AWQ result before the GPTQ
    phase (or its fragmentation) can cost a full replay. Entries are matched
    by method-name prefix (awq_full -> 'awq'). load_config stays modelopt's,
    not yaml.safe_load, so the ExMy aliases survive; the variant is published
    atomically (same pattern as the layerwise injection).
    """
    import yaml

    from modelopt.torch.opt.config_loader import load_config

    raw = yaml.safe_load(Path(recipe_yaml).read_text())
    kept = [e for e in raw.get("algorithm", []) if str(e.get("method", "")).startswith(phase)]
    if not kept:
        raise SystemExit(f"recipe {recipe_yaml} has no {phase!r} algorithm entries")
    raw["algorithm"] = kept
    path = Path(out_dir) / f"recipe_phase_{phase}.yaml"
    tmp = f"{path}.{os.getpid()}"
    Path(tmp).write_text(yaml.safe_dump(raw))
    os.replace(tmp, path)
    return load_config(str(path))


class Checkpointer:
    """Writes stage files locally and mirrors the whole dir to GCS per stage."""

    def __init__(self, directory: str | None, upload_prefix: str | None, is_main: bool = True):
        self.dir = Path(directory) if directory else None
        self.prefix = upload_prefix.rstrip("/") if upload_prefix else None
        self.is_main = is_main
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, obj: Any) -> None:
        if self.dir is None or not self.is_main:
            return
        path = self.dir / name
        if isinstance(obj, str):
            path.write_text(obj)
        else:
            path.write_text(json.dumps(obj, indent=2, sort_keys=True))
        print(f"=== CKPT {name} ok ===")

    def save_state(self, model) -> None:
        if self.dir is None or not self.is_main:
            return
        import modelopt.torch.opt as mto

        mto.save(model, str(self.dir / "mtq_state.pt"))
        print("=== CKPT mtq_state.pt ok ===")

    def sync(self, stage: str) -> None:
        if self.dir is None or self.prefix is None or not self.is_main:
            return
        # Retry, and never fatal: a transient GCS 5xx must not discard a
        # calibration that already completed (check=True made the upload the
        # most failure-prone step of the whole run). Final failure prints a
        # loud marker; operator re-runs `gcloud storage cp -r -n` by hand.
        # -n (resume-friendly) for the big weight trees, but small bank
        # files MUST overwrite: prod3 found the re-run silently kept the
        # previous attempt's dataset_report/mtq_state in GCS, so a restore
        # after another pod loss would have picked stale state.
        clobber = stage in ("dataset", "state", "calibration")
        for attempt in (1, 2, 3):
            rc = subprocess.run(
                ["gcloud", "storage", "cp", "-r", *(["-n"] if not clobber else []),
                 f"{self.dir}/.", f"{self.prefix}/"]
            ).returncode
            if rc == 0:
                print(f"=== CKPT synced to {self.prefix} after {stage} ===")
                return
            print(f"[sync] attempt {attempt} rc={rc} after {stage}", flush=True)
            import time

            time.sleep(30 * attempt)
        print(
            f"!!! SYNC_FAILED to {self.prefix} after {stage} -- re-run: "
            f"gcloud storage cp -r -n {self.dir}/. {self.prefix}/ !!!",
            flush=True,
        )


def main():
    # A rank died inside export with no traceback (fatal signal, not a cgroup
    # OOM event). Dump Python stacks on SIGSEGV/SIGABRT so the next attempt is
    # diagnosable from the log alone.
    import faulthandler

    faulthandler.enable()
    parser = argparse.ArgumentParser(description="ModelOpt NVFP4 quantization")
    parser.add_argument("--config", required=True, help="Per-model modelopt-*.json")
    parser.add_argument(
        "--calibration-dir", required=True, help="Structured calibration directory"
    )
    parser.add_argument("--output-dir", required=True, help="Export directory")
    parser.add_argument(
        "--checkpoint-dir", help="Stage checkpoint directory (default: none)"
    )
    parser.add_argument(
        "--upload-prefix",
        help="gs:// prefix to mirror stage checkpoints after every stage",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        help="Override weight_calibration.token_budget (smoke runs)",
    )
    parser.add_argument(
        "--stage-dir",
        help="FSDP2 only: after quantize, write the AWQ/GPTQ-mutated BF16 "
        "weights here so an export failure costs an export retry, not a "
        "calibration replay",
    )
    parser.add_argument(
        "--resume-stage",
        help="Load mutated weights + mtq_state.pt and go straight to export "
        "(requires --checkpoint-dir with the state saved by a prior "
        "--stage-dir run)",
    )
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Run each recipe algorithm layer-by-layer (modelopt 0.46 "
        "layerwise mode): one QDQ-propagating sweep per method instead of "
        "full-model passes, per-layer checkpoints under --checkpoint-dir.",
    )
    parser.add_argument(
        "--calib-phase",
        choices=["all", "awq", "gptq"],
        default="all",
        help="Split the recipe algorithm chain across process launches: "
        "'awq' runs only the AWQ leg then banks mtq_state + staged weights "
        "and exits before export; 'gptq' (with --resume-stage) restores that "
        "state over the staged weights, runs the GPTQ leg, re-banks, and "
        "exports. modelopt offers no resume inside quantize(), so the phase "
        "boundary is where checkpointing becomes possible.",
    )
    args = parser.parse_args()
    if args.resume_stage and not (args.checkpoint_dir and os.environ.get("RANK")):
        raise SystemExit("--resume-stage requires --checkpoint-dir and a torchrun launch")
    if args.calib_phase == "gptq" and not args.resume_stage:
        raise SystemExit("--calib-phase gptq requires --resume-stage (AWQ hand-off)")
    if args.calib_phase != "all" and not (args.checkpoint_dir and args.stage_dir):
        raise SystemExit("--calib-phase requires --checkpoint-dir and --stage-dir")
    if args.layerwise and not args.checkpoint_dir:
        raise SystemExit("--layerwise requires --checkpoint-dir (per-layer resume state)")

    recipe: dict[str, Any] = json.loads(Path(args.config).read_text())
    settings = recipe["modelopt"]
    device_map = settings.get("device_map", "cuda")
    use_fsdp2 = device_map == "fsdp2"
    if use_fsdp2 and os.environ.get("RANK") is None:
        raise SystemExit(
            "device_map=fsdp2 requires launching with torchrun "
            "(e.g. torchrun --nproc_per_node=8)"
        )

    state = (
        setup_fsdp2()
        if use_fsdp2
        else DistributedState(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    )
    try:
        run(recipe, args, state, use_fsdp2)
    except BaseException:
        if use_fsdp2:
            # Peers are likely parked in a collective this rank will never
            # rejoin: print the traceback and exit now (hf_ptq's abort
            # pattern); torchrun reaps the rest.
            traceback.print_exc()
            from modelopt.torch.utils import distributed as dist_utils

            dist_utils.abort()
        raise
    finally:
        if use_fsdp2:
            # Barrier + destroy: non-main ranks wait here while rank 0 writes
            # the manifest and mirrors it to GCS, inside the 2 h PG timeout.
            from modelopt.torch.utils import distributed as dist_utils

            dist_utils.cleanup()


def _sanitize_quantizer_state(model, say) -> None:
    """Neutralize NaN entries in the restored quantizer state (2026-09-14).

    awq_lite computes per-input-channel pre-quant scales
    ``s = amax_x^a / amax_w^(1-a)``; channels dead across the whole corpus
    (zero activation or weight amax) yield 0/0 = NaN, so the AWQ leg banks
    NaN scales (measured: 385/2560 channels of layer-0 in_proj_qkv). Without
    this, the first calibrated forward multiplies hidden states by NaN and
    GPTQ dies in the amax calibrator's assert -- and a NaN input scale is
    unservable in the exported checkpoint anyway. Dead channels carry no
    signal, so the neutral scale 1.0 is exact for them; amax NaNs fall back
    to the max of the finite entries (conservative, keeps static FP8 scales
    valid).
    """
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
        TensorQuantizer,
    )

    n_pqs = n_amax = 0
    offenders: list[str] = []
    for path, m in model.named_modules():
        if not isinstance(m, TensorQuantizer):
            continue
        pqs = getattr(m, "_pre_quant_scale", None)
        if torch.is_tensor(pqs) and pqs.is_floating_point():
            k = int(torch.isnan(pqs).sum())
            if k:
                n_pqs += k
                offenders.append(path)
                pqs.nan_to_num_(nan=1.0)
        amax = getattr(m, "_amax", None)
        if (
            torch.is_tensor(amax)
            and not amax.is_meta
            and amax.is_floating_point()
        ):
            k = int(torch.isnan(amax).sum())
            if k:
                n_amax += k
                finite = amax[~torch.isnan(amax)]
                rep = (
                    float(finite.max())
                    if finite.numel()
                    else 1.0
                )
                amax.nan_to_num_(nan=rep)
    say(
        f"restored-state sanitizer: {n_pqs} NaN pre_quant_scale channels -> 1.0 "
        f"({len(offenders)} quantizers; e.g. {offenders[:3]}), "
        f"{n_amax} NaN amax -> finite max"
    )


def _debug_phase_b(model, stage_dir: str) -> None:
    """Phase-B forensics (env DEBUG_PHASEB=1, crash 2026-09-14 23:07:56:
    GPTQ's first forward hit ``AssertionError: detected nan values in amax``
    with all staged shard bytes verified finite). Two questions to answer:

    1. Did the FSDP2 loader take the STAGED (AWQ-folded) values, or did the
       checkpoint's per-expert keys -- which the staged index still lists
       alongside our runtime-named fused keys -- overwrite the fused expert
       params with unfolded originals? Compares loaded params (after
       all-gather) against the staged safetensors bytes for a sample.
       A mismatch while mtq_state carries pre_quant_scale means activations
       get multiplied by AWQ scales that the weights never divided back out
       -> bf16 overflow -> the observed NaN.
    2. Which module first OUTPUTS NaN in the calibration forward: forward
       hooks on every leaf module print innermost-first, so the first line
       names the generator, not a downstream victim.
    """
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor

    rank = int(os.environ.get("RANK", "0"))

    # --- 1) staged-weight integrity probe -------------------------------
    weight_map = json.loads(
        (Path(stage_dir) / "model.safetensors.index.json").read_text()
    )["weight_map"]
    params = dict(model.named_parameters())
    probes = [n for n in params if n.endswith("experts.gate_up_proj")][:1]
    probes += [n for n in params if n.endswith("mlp.gate.weight")][:1]
    for name in probes:
        if name not in weight_map:
            print(f"[phaseb-probe][r{rank}] {name}: NOT IN staged index", flush=True)
            continue
        p = params[name]
        full = p.full_tensor() if isinstance(p, DTensor) else p.detach()
        with safe_open(Path(stage_dir) / weight_map[name], framework="pt") as sf:
            ref = sf.get_tensor(name)
        full_cpu = full.detach().to("cpu").to(torch.float32)
        ref_cpu = ref.to(torch.float32)
        same = torch.equal(full_cpu, ref_cpu.to(full_cpu.dtype))
        if same:
            print(
                f"[phaseb-probe][r{rank}] {name}: EXACT match vs staged shard",
                flush=True,
            )
        else:
            diff = (full_cpu - ref_cpu).abs()
            denom = ref_cpu.abs().clamp_min(1e-12)
            print(
                f"[phaseb-probe][r{rank}] {name}: MISMATCH vs staged shard "
                f"frac_diff={(diff > 0).float().mean():.3f} "
                f"max_abs={diff.max():.4g} max_rel={(diff / denom).max():.4g} "
                f"(param maxabs={full_cpu.abs().max():.4g}, "
                f"staged maxabs={ref_cpu.abs().max():.4g})",
                flush=True,
            )
            del diff
        del full, full_cpu, ref, ref_cpu

    # --- 2) NaN generator trace -----------------------------------------
    # Hook EVERY module (parents too: the GDN block and the hyper-connection
    # mix run their numerics in bare functionals between quantized leaves, so
    # a leaf-only trace cannot see where those first turn NaN). Each module
    # reports once, classified by whether its inputs were already NaN: the
    # first ``input_clean=True`` line is the generator; later lines are
    # victims.
    reported: set[str] = set()

    def _nan_any(obj) -> bool:
        if torch.is_tensor(obj):
            return obj.is_floating_point() and bool(torch.isnan(obj).any())
        if isinstance(obj, (tuple, list)):
            return any(_nan_any(v) for v in obj)
        return False

    def make_hook(path: str):
        def hook(_mod, inputs, output):
            if path in reported or not _nan_any(output):
                return
            reported.add(path)
            tens = output[0] if isinstance(output, (tuple, list)) else output
            print(
                f"[phaseb-nan][r{rank}] first NaN output: {path} "
                f"({type(_mod).__name__}, shape={tuple(tens.shape)}, "
                f"input_clean={not any(_nan_any(i) for i in inputs)})",
                flush=True,
            )

        return hook

    n_hooks = 0
    for path, mod in model.named_modules():
        if path:
            mod.register_forward_hook(make_hook(path))
            n_hooks += 1
    print(
        f"[phaseb-probe][r{rank}] NaN hooks on {n_hooks} modules", flush=True
    )

    # --- 3) runtime state of the first suspect quantizers ---------------
    mods = dict(model.named_modules())
    for qname in (
        "model.layers.0.linear_attn.in_proj_qkv.input_quantizer",
        "model.layers.0.linear_attn.out_proj.input_quantizer",
    ):
        q = mods.get(qname)
        if q is None:
            continue
        pqs = getattr(q, "pre_quant_scale", None)
        pqs_info = (
            "None"
            if pqs is None
            else (
                f"shape={tuple(pqs.shape)} maxabs={pqs.abs().max().item():.4g} "
                f"inf={int(torch.isinf(pqs).sum())} "
                f"nan={int(torch.isnan(pqs).sum())} dtype={pqs.dtype}"
            )
        )
        try:
            amax_info = f"{q.amax.abs().max().item():.4g}"
        except Exception:  # noqa: BLE001
            amax_info = "unset"
        print(
            f"[phaseb-probe][r{rank}] {qname}: enable_pqs={q._enable_pre_quant_scale} "
            f"pqs={pqs_info} amax={amax_info} fake_quant={q.fake_quant} "
            f"if_quant={q._if_quant} if_calib={q._if_calib} "
            f"dynamic={q._dynamic} block_sizes={q.block_sizes} "
            f"backend={q.backend}",
            flush=True,
        )


def _wrap_gptq_progress() -> None:
    """Add a solved-layer counter to ModelOpt's per-layer GPTQ MSE print.

    Upstream prints one ``[name] Relative MSE error`` line per solved linear
    with no position in the sequence — a multi-minute silent hole on this
    model. Monkeypatch (not a site-packages edit) so it survives pod
    rebuilds and lives with the driver.
    """
    import time as _t

    from modelopt.torch.quantization.utils.calib_utils import GPTQHelper

    if getattr(GPTQHelper._print_mse_error, "_progress_wrapped", False):
        return
    orig = GPTQHelper._print_mse_error
    ctr = {"n": 0, "t0": _t.monotonic()}

    def timed(self, hessian):
        orig(self, hessian)
        ctr["n"] += 1
        el = _t.monotonic() - ctr["t0"]
        print(f"[gptq] layers solved: {ctr['n']} "
              f"({ctr['n'] / max(el, 1e-9) * 60:.1f}/min, {el:.0f}s elapsed)",
              flush=True)

    timed._progress_wrapped = True
    GPTQHelper._print_mse_error = timed


def _install_fast_module_names(model) -> None:
    """Serve module lookups from one precomputed id->name map.

    modelopt's ``fsdp2_aware_weight_update`` calls
    ``core_utils._get_module_name`` once per exported expert weight; with no
    caller-supplied map each call rebuilds ``named_modules()`` and linearly
    scans it — quadratic for this 512-expert MoE (~50k updates × ~200k-node
    walks; observed as a >15-minute silent, single-core export phase).
    Rebind the symbol in every module that imported it by name.
    """
    import sys

    import modelopt.torch.quantization.utils.core_utils as cu

    names = {id(m): n for n, m in model.named_modules()}
    orig = cu._get_module_name

    def fast(module, root_model, name_to_module=None):
        n = names.get(id(module))
        return n if n is not None else orig(module, root_model, name_to_module)

    for mod in list(sys.modules.values()):
        # __dict__ lookup, NOT getattr: transformers' lazy-module __getattr__
        # executes arbitrary submodule imports (2026-09-14 prod3: probing the
        # lazy `transformers` module imported image_processing_aria -> died on
        # missing torchvision 40 min into AWQ, pre-bank).
        if mod is not None and mod.__dict__.get("_get_module_name", None) is orig:
            mod._get_module_name = fast


def _pre_export_memory(limit_gib: float = 40.0) -> None:
    """Free every reclaimable byte, then prove what is still live on GPU.

    The 2026-09-13 production run reached export with ~69 GiB of allocated
    tensors per rank (FSDP2 shards alone are ~30 GiB), so the first unshard
    inside ``requantize_resmooth_fused_llm_layers`` died with CUDA OOM.
    GPTQ frees each handle (Hessian, h_inv, fp32 weight copy) right after
    solving and calls ``empty_cache``, so the holders were unidentified;
    every run past that point dumps an allocator snapshot
    (``/home/ray/preexport_snap_r*.pickle``, viewable on
    pytorch.org/memory_viz) when live memory exceeds ``limit_gib``.
    """
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )
    alloc = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    print(
        f"[mem] rank {rank} pre-export: allocated {alloc:.1f} GiB, "
        f"reserved {reserved:.1f} GiB",
        flush=True,
    )
    if alloc > limit_gib:
        path = f"/home/ray/preexport_snap_r{rank}.pickle"
        torch.cuda.memory._dump_snapshot(path)
        print(f"[mem] rank {rank} snapshot -> {path}", flush=True)


def stage_mutated_weights(
    model, source_model_id: str, stage_dir: str, state: DistributedState
) -> tuple[int, int]:
    """Write the AWQ/GPTQ-mutated weights as a loadable partial checkpoint.

    Without this, the mutations live only in the sharded model until export,
    so any export-phase failure (the 2026-09-13 CUDA OOM cost exactly this)
    discards the whole calibration. Tensors are materialized one at a time —
    ``DTensor.full_tensor()`` is an all-gather every rank must join in the
    same order — and rank 0 accumulates on CPU and flushes ~4 GiB safetensors
    shards, so extra peak GPU memory is one full tensor per rank.

    Skipped PLE/vision/MTP tensors stay absent on purpose: the index keeps
    the source weight_map entries for them, which
    ``reattach_skipped_weights`` resolves via the tmpfs manifest. Config and
    tokenizer JSONs are copied alongside so the directory is a self-contained
    ``--resume-stage`` input (no 331 GiB re-pull needed for an export retry).

    Returns (num_files, staged_bytes).
    """
    import shutil

    from modelopt.torch.utils import distributed as dist_utils
    from safetensors.torch import save_file
    from torch.distributed.tensor import DTensor

    stage = Path(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    src = Path(source_model_id)

    names = [n for n, _ in model.named_parameters()]
    params = dict(model.named_parameters())
    staged: dict[str, str] = {}
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = flush_bytes = total_bytes = 0
    parts: list[str] = []
    uncollected = 0

    # Overlapped staging (2026-09-14; replaced the serialized gather->copy->
    # write loop after its ~40 MB/s staging pass cost ~2 h per calibration):
    # rank 0 stages each gathered tensor through a PINNED double buffer with
    # a non-blocking copy, so tensor N's GPU->CPU transfer overlaps tensor
    # N+1's all-gather; full 4 GiB shards go to a background writer thread
    # (max 2 in flight), so NVMe writes overlap the collectives. The
    # all-gather ORDER is untouched -- every rank still calls full_tensor()
    # in the identical sequence, the only collective-correctness constraint.
    # Byte-exactness: tensors move as opaque uint8 views, never through a
    # dtype round-trip, so output bytes are identical to the old path.
    import threading

    slots: list[torch.Tensor] = []  # pinned byte buffers, grown on demand
    inflight: list[list] = []  # [event, name, view] awaiting copy-done
    writers: list[threading.Thread] = []
    write_errors: list[BaseException] = []

    def pinned_slot(nbytes: int) -> torch.Tensor:
        idx = len(inflight) % 2  # parity of outstanding copies
        if len(slots) <= idx:
            slots.append(torch.empty(max(nbytes, 2**30), dtype=torch.uint8, pin_memory=True))
        elif slots[idx].numel() < nbytes:
            slots[idx] = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        return slots[idx]

    def harvest_all() -> None:
        """Complete every pending D2H copy; append its data to the buffer.

        The clone() detaches bytes from the recyclable pinned slot (a plain
        CPU->CPU copy at RAM speed, no PCIe); without it, the slot's next
        occupant would overwrite a not-yet-written shard member.
        """
        nonlocal buf_bytes
        while inflight:
            ev, nm, view = inflight.pop(0)
            ev.synchronize()
            final = view.clone()
            buf[nm] = final
            buf_bytes += final.numel() * final.element_size()

    def flush_shard(force: bool = False) -> None:
        # flush_bytes is tracked on EVERY rank from gathered tensor sizes, so
        # the barrier below is entered by all ranks in the same order. The
        # 2026-09-14 prod run deadlocked here when only rank 0 (which alone
        # tracked the threshold) called barrier() while the others issued
        # the next all_gather: collective mismatch, silent until the 2 h PG
        # timeout. Payload/serialization remains rank-0-only.
        nonlocal buf_bytes, flush_bytes, total_bytes
        if not (flush_bytes >= 4 * 2**30 or force):
            return
        flush_bytes = 0
        harvest_all()
        dist_utils.barrier()  # everyone contributed this shard's tensors
        if state.is_main and buf:
            part = f"model-{len(parts) + 1:05d}-of-STAGECOUNT.safetensors"
            parts.append(part)
            payload = dict(buf)
            while len(writers) >= 2 and not write_errors:
                writers.pop(0).join()
            if write_errors:
                raise write_errors[0]

            def _write(payload=payload, part=part):
                try:
                    save_file(payload, stage / part)
                except BaseException as exc:  # re-raised on the main thread
                    write_errors.append(exc)

            t = threading.Thread(target=_write, daemon=True)
            t.start()
            writers.append(t)
            for k in payload:
                staged[k] = part
            shard_bytes = sum(v.numel() * v.element_size() for v in payload.values())
            total_bytes += shard_bytes
            print(
                f"[stage] shard {len(parts)} handed off "
                f"({shard_bytes / 2**30:.1f} GiB, "
                f"{total_bytes / 2**30:.1f} GiB total)",
                flush=True,
            )
        buf.clear()
        buf_bytes = 0

    for i, name in enumerate(names):
        p = params[name]
        full = p.full_tensor() if isinstance(p, DTensor) else p.detach()
        # All ranks see the full gathered tensor, so all ranks agree on where
        # shard boundaries (and their barriers) fall, without extra comms.
        flush_bytes += full.numel() * full.element_size()
        if state.is_main:
            src_t = full.detach().contiguous()
            nbytes = src_t.numel() * src_t.element_size()
            slot = pinned_slot(nbytes)
            view = slot[:nbytes].view(src_t.dtype).view(src_t.shape)
            if src_t.is_cuda:
                slot[:nbytes].copy_(
                    src_t.view(torch.uint8).reshape(-1), non_blocking=True
                )
                ev = torch.cuda.Event()
                ev.record()  # current stream of this rank's current device
                inflight.append([ev, name, view])
            else:
                harvest_all()
                slot[:nbytes].copy_(src_t.view(torch.uint8).reshape(-1))
                final = view.clone()
                buf[name] = final
                buf_bytes += nbytes
            del src_t
        del full
        # One tensor of lag is the overlap: a slot may only be reused once
        # its copy completed, which the pinned_slot parity enforces by
        # syncing the older event before the next copy issues.
        if len(inflight) >= 2:
            ev, nm, view = inflight.pop(0)
            ev.synchronize()
            final = view.clone()
            buf[nm] = final
            buf_bytes += final.numel() * final.element_size()
        uncollected += 1
        flush_shard()
        if uncollected >= 512:
            harvest_all()
            dist_utils.barrier()  # cap CPU churn between ranks
            uncollected = 0

    flush_shard(force=True)
    while writers:
        writers.pop(0).join()
    if write_errors:
        raise write_errors[0]

    num = len(parts)
    if state.is_main:
        renames = {
            old: old.replace("-of-STAGECOUNT", f"-of-{num:05d}") for old in parts
        }
        for old, new in renames.items():
            (stage / old).rename(stage / new)
        weight_map = json.loads(
            (src / "model.safetensors.index.json").read_text()
        )["weight_map"]
        weight_map.update({k: renames[v] for k, v in staged.items()})
        (stage / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map})
        )
        # Self-contained staged dir: the index keeps SOURCE shard names for
        # skipped tensors (PLE/vision/MTP) and the FSDP2 loader opens every
        # file the index names (2026-09-14: chained phase B died on
        # FileNotFoundError for model-00006; the one-shot phase B script had
        # this fix inline, the chain path did not).
        for f in sorted(set(weight_map.values())):
            if not (stage / f).exists() and (src / f).exists():
                shutil.copy2(src / f, stage / f)
        for meta in sorted(src.iterdir()):
            if meta.suffix in (".json", ".jinja", ".txt", ".model") and not (
                stage / meta.name
            ).exists():
                shutil.copy2(meta, stage / meta.name)
    return num, total_bytes


def run(recipe: dict[str, Any], args, state: DistributedState, use_fsdp2: bool) -> None:
    source = recipe["source"]
    settings = recipe["modelopt"]
    batch_size = int(settings.get("batch_size", 1))
    ckpt = Checkpointer(
        args.checkpoint_dir, args.upload_prefix, is_main=state.is_main
    )

    def say(msg: str) -> None:
        if state.is_main:
            print(msg, flush=True)

    resume = bool(args.resume_stage)
    tokenizer_dir = args.resume_stage if resume else source["model_id"]
    say(f"=== Loading tokenizer{', staged (resume)' if resume else ' + calibration'}: {tokenizer_dir} ===")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=source["trust_remote_code"]
    )
    if resume and args.calib_phase == "all":
        say(f"=== Resume export from staged weights: {args.resume_stage} ===")
    else:
        # Runs on fresh launches AND on the gptq phase: resume skips the
        # model-side calibration record, but the GPTQ leg still needs the
        # corpus for its Hessian pass.
        calibration = settings["weight_calibration"]
        if args.token_budget:
            calibration = {**calibration, "token_budget": args.token_budget}
        records = load_calibration_records(args.calibration_dir, calibration["domains"])
        dataset, report = build_calibration_dataset(
            args.calibration_dir, tokenizer, calibration, records_by_domain=records
        )
        say(
            f"Packed {report['token_budget']:,}-token budget into "
            f"{len(dataset):,} sequences of {report['sequence_length']}"
        )
        if use_fsdp2:
            # All ranks built the identical packed dataset; shard AFTER packing.
            # Truncate to a multiple of world_size * batch_size so every rank
            # runs exactly the same number of forward iterations — modelopt's
            # dp collectives (amax all-reduce) require matched iteration counts.
            strip = state.world_size * batch_size
            if len(dataset) < strip:
                # Tiny (smoke) budgets: fall back to equal sequence counts per
                # rank with one partial batch each — matched iteration counts
                # still hold.
                strip = state.world_size
            usable = len(dataset) - len(dataset) % strip
            if usable != len(dataset):
                dataset = dataset.select(range(usable))
            dataset = dataset.select(range(state.rank, usable, state.world_size))
            say(
                f"Calibration sharded across {state.world_size} ranks "
                f"({usable:,} -> {len(dataset)} sequences/rank)"
            )
        ckpt.write("dataset_report.json", report)
        ckpt.sync("dataset")

    say(
        f"=== Loading model: {tokenizer_dir} "
        f"(device_map={settings.get('device_map', 'cuda')}) ==="
    )
    ple_layers: list[int] = []
    if use_fsdp2:
        # FSDP2 gathers full params for forward/export; keep eager, as hf_ptq
        # does, so compile does not fight the per-layer all-gathers.
        torch.compiler.set_stance("force_eager")
        model, ple_layers = load_model_fsdp2(
            tokenizer_dir, source["trust_remote_code"], state
        )
        forward_device = state.device
        if ple_layers:
            # PLE must participate in calibration forwards: re-attach it
            # CPU-resident (table mmap-shared from /dev/shm) post-FSDP-wrap.
            # On resume, model_id is the stage dir; its index keeps the source
            # shard-name entries, so the tmpfs manifest resolves them.
            inject_ple_modules(model, tokenizer_dir, ple_layers)
    else:
        model = load_model(
            source["model_id"],
            source["task"],
            settings.get("device_map", "cuda"),
            source["trust_remote_code"],
        )
        forward_device = None
    model.eval()

    recipe_sha = sha256(Path(settings["recipe_yaml"]))
    if resume:
        # Replay the quantizer structure recorded by mto.save(). The mutated
        # weights are already in place as FSDP2 shards in this process, so
        # restore_from_modelopt_state is used directly rather than mto.restore,
        # whose full-state_dict weight load expects an unsharded replica.
        from modelopt.torch.opt.conversion import restore_from_modelopt_state

        objs = torch.load(
            Path(args.checkpoint_dir) / "mtq_state.pt", map_location="cpu"
        )
        model = restore_from_modelopt_state(model, modelopt_state=objs["modelopt_state"])
        _install_fast_module_names(model)
        model.eval()
        _sanitize_quantizer_state(model, say)
        say("modelopt state restored over staged weights")
        if os.environ.get("DEBUG_PHASEB"):
            _debug_phase_b(model, tokenizer_dir)
        calibrated = False
        if args.calib_phase == "gptq":
            # Phase hand-off: AWQ leg already banked by the prior launch; the
            # restored quantizers carry its scales/clip alphas, so GPTQ here
            # sees exactly the state a single-process awq->gptq chain would.
            import modelopt.torch.quantization as mtq

            _wrap_gptq_progress()
            say("\n=== ModelOpt quantize (phase gptq over restored AWQ state) ===")
            mtq.quantize(
                model,
                _phase_config(settings["recipe_yaml"], "gptq", args.checkpoint_dir),
                build_forward_loop(
                    model, dataset, tokenizer, batch_size, device=forward_device
                ),
            )
            calibrated = True
    else:
        # modelopt's loader, not yaml.safe_load: it converts the ExMy aliases
        # (e4m3/e2m1) to the (E, M) tuples the fake-quant kernels require and
        # validates against the schema declared in the recipe header comment.
        from modelopt.torch.opt.config_loader import load_config

        recipe_path = settings["recipe_yaml"]
        if args.layerwise:
            # load_config only takes a path, so inject the nested
            # ``layerwise`` block (QuantizeAlgorithmConfig.layerwise, alias
            # use_sequential) into a copy of the recipe. qdq_from_prev
            # propagates each layer's calibrated outputs into the next
            # layer's inputs. No checkpoint_dir: upstream rejects layerwise
            # resume state under multi-process FSDP2 (layerwise_calib
            # _CheckpointState), and --stage-dir already covers coarse
            # resume at this budget.
            import yaml

            raw = yaml.safe_load(Path(recipe_path).read_text())
            for entry in raw.get("algorithm", []):
                entry["layerwise"] = {
                    "enable": True,
                    "get_qdq_activations_from_prev_layer": True,
                }
            recipe_path = str(Path(args.checkpoint_dir) / "recipe_layerwise.yaml")
            # Atomic publish: every rank writes then renames the same bytes,
            # so a reader never observes a half-written variant.
            tmp = f"{recipe_path}.{os.getpid()}"
            Path(tmp).write_text(yaml.safe_dump(raw))
            os.replace(tmp, recipe_path)
            say(f"layerwise recipe variant written to {recipe_path}")

        # In-process chain (CHAIN_INPROC=1 with phase awq): run the GPTQ leg
        # on the LIVE model right after AWQ instead of re-entering via
        # --resume-stage. The 2026-09-14 post-mortem: the restored phase-B saw
        # 36,564 NaN pre-quant-scale channels banked by awq_lite AND a
        # double-applied pre-scale (inf -> NaN in the first forward); the
        # mid-chain checkpoint semantics around folded vs unfolded scales are
        # not dependable, while the live state is exactly what a single-process
        # awq->gptq chain sees. The sanitizer between the legs is what the
        # stock recipe-entries-in-one-call path lacks (it would re-run gptq
        # over the NaN scales, same crash), so chain here explicitly.
        chain = args.calib_phase == "awq" and os.environ.get("CHAIN_INPROC") == "1"
        if args.calib_phase == "awq":
            cfg = _phase_config(settings["recipe_yaml"], "awq", args.checkpoint_dir)
            say(f"phase recipe (awq leg only) published under {args.checkpoint_dir}")
        else:
            cfg = load_config(recipe_path)

        say("\n=== ModelOpt quantize (calibrates with the corpus forward loop) ===")
        # Long rows (32K) blow up under eager attention (s^2 score materialization);
        # SDPA/flash keeps them linear. Verify the resolved implementation here,
        # before the first forward, because the loader never prints it.
        say(
            f"attention implementation: {getattr(model.config, '_attn_implementation', None)!r}"
            f" / text: {getattr(getattr(model.config, 'text_config', None), '_attn_implementation', None)!r}"
        )
        import modelopt.torch.quantization as mtq

        _wrap_gptq_progress()
        mtq.quantize(
            model,
            cfg,
            build_forward_loop(model, dataset, tokenizer, batch_size, device=forward_device),
        )
        if chain:
            _sanitize_quantizer_state(model, say)
            say(
                "\n=== chain (in-process): gptq leg over live AWQ state ==="
            )
            mtq.quantize(
                model,
                _phase_config(settings["recipe_yaml"], "gptq", args.checkpoint_dir),
                build_forward_loop(
                    model, dataset, tokenizer, batch_size, device=forward_device
                ),
            )
        calibrated = True
    if calibrated:
        say(f"Quantized module census: {quantizer_census(model)}")
        if use_fsdp2:
            # Quantizer-state record (amax, clip alphas, resmodes): lets an
            # export retry via --resume-stage rebuild the quantizer structure
            # without recalibrating. FSDP2-safe because rank-local quantizer
            # state is collective-synced.
            if state.is_main:
                ckpt.save_state(model)
                say("mtq_state saved under checkpoint dir (audit + resume record)")
                # Upload the state NOW, not only with the post-staging sync:
                # a pod reschedule during the ~20 min staging write (seen
                # 2026-09-14) otherwise leaves the calibration with no GCS
                # copy at all. Cheap (~min) insurance against the exact
                # event that already cost one rerun.
                ckpt.sync("state")
            _install_fast_module_names(model)
            if args.stage_dir:
                say(f"\n=== Staging mutated weights to {args.stage_dir} ===")
                num, gib = stage_mutated_weights(
                    model, source["model_id"], args.stage_dir, state
                )
                say(f"Staged {num} safetensors shards, {gib / 2**30:.1f} GiB")
                ckpt.sync("staged")
        else:
            ckpt.save_state(model)
            ckpt.sync("calibration")
    if args.calib_phase == "awq" and not chain:
        # Phase A stops here on purpose: the export OOM history is downstream
        # of the split point, so phase A must not depend on it. In-process
        # chain (CHAIN_INPROC=1) proceeds through the gptq leg into export;
        # the bank above still landed the (now gptq-updated) state + staged
        # weights in GCS before any export risk.
        say(
            "\n=== Phase awq banked: mtq_state + staged weights synced; export "
            "deferred to the gptq phase ==="
        )
        return
    if use_fsdp2:
        # Common path: both fresh and resumed runs want identical pre-export
        # cleanup/reporting — this is the phase it was added to protect.
        _pre_export_memory()

    say(f"\n=== Exporting HF checkpoint to {args.output_dir} ===")
    from modelopt.torch.export import export_hf_checkpoint

    # Run 2 exported from a clean 29.4 GiB start and still died at 68.9 GiB
    # OOM inside an export unshard: the growth happens inside the export loop
    # (~1.6 GiB per processed layer, died after ~25/48). A time series plus a
    # snapshot at the 55 GiB crossing attributes the holders.
    stop_watch = threading.Event()

    def _export_watch() -> None:
        rank = torch.distributed.get_rank()
        crossed = False
        while not stop_watch.wait(5.0):
            gib = torch.cuda.memory_allocated() / 2**30
            print(f"[mem] export live {gib:.1f} GiB", flush=True)
            if not crossed and gib > 55:
                path = f"/home/ray/export_snap_r{rank}.pickle"
                torch.cuda.memory._dump_snapshot(path)
                print(f"[mem] export snapshot -> {path}", flush=True)
                crossed = True

    if use_fsdp2:
        # Allocation stack traces so the watchdog snapshot names the holders of
        # the per-layer growth instead of just its size. ~KB CPU per entry.
        if os.environ.get("SQQ_MEM_HISTORY", "1") == "1":
            try:
                torch.cuda.memory._record_memory_history(max_entries=120_000)
            except Exception as exc:  # noqa: BLE001
                say(f"[mem] history recording unavailable: {exc}")
        threading.Thread(target=_export_watch, daemon=True).start()
    # All ranks participate in the FSDP2 all-gathers that gather the full
    # state dict to rank-0 CPU; file writes happen on rank 0 only. no_grad,
    # not inference_mode: state_dict() -> param.detach() breaks on inference
    # tensors (see hf_ptq.export_quantized).
    try:
        with torch.no_grad():
            export_hf_checkpoint(model, export_dir=args.output_dir)
    finally:
        stop_watch.set()
    if state.is_main:
        tokenizer.save_pretrained(args.output_dir)
        if use_fsdp2:
            # PLE/vision/MTP tensors the loader skipped + original
            # ple_layer_ids, back into the export so the served model is
            # complete. Before the manifest so the hashes cover them.
            reattach_skipped_weights(
                source["model_id"], args.output_dir, ple_layers
            )

        out = Path(args.output_dir)
        manifest = {
            "recipe": str(settings["recipe_yaml"]),
            "recipe_sha256": recipe_sha,
            "census": quantizer_census(model),
            "files": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(out.glob("*"))
                if p.is_file()
            },
        }
        ckpt.write("manifest.json", json.dumps(manifest, indent=2))
        ckpt.sync("export")
    say(f"\n=== Done: {args.output_dir} ===")


if __name__ == "__main__":
    main()
