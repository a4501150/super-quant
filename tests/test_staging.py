"""Functional test for overlapped stage_mutated_weights (world=1, gloo).

Verifies: shard splitting at 4 GiB, byte-exact round-trip (bf16 + f32),
writer-thread completion, index write, meta copy.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29611")
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
sys.path.insert(0, "/home/ray/super-quant/src")

import torch
import torch.distributed as dist
from safetensors.torch import load_file

import quantize_mopt as qm

dist.init_process_group("gloo")
dev = "cuda:0"

model = torch.nn.Module()
p1 = torch.randn(2**30, dtype=torch.bfloat16, device=dev)          # 2 GiB
p2 = torch.arange(2**30, dtype=torch.float32, device=dev)           # 4 GiB
p3 = torch.randn(1000, dtype=torch.float16, device=dev)             # tiny
for sub, p in (("layerA", p1), ("layerB", p2), ("layerC", p3)):
    mod = torch.nn.Module()
    mod.weight = torch.nn.Parameter(p, requires_grad=False)
    setattr(model, sub, mod)

src = tempfile.mkdtemp(prefix="sqtest_src_")
stage = tempfile.mkdtemp(prefix="sqtest_stage_")
with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
    json.dump({"weight_map": {"legacy.key": "model-00001-of-00005.safetensors"}}, f)
with open(os.path.join(src, "config.json"), "w") as f:
    f.write('{"test": 1}')

num, total = qm.stage_mutated_weights(
    model, src, stage, SimpleNamespace(is_main=True, rank=0, world_size=1, device=dev)
)
print(f"num_shards={num} total_bytes={total}")
assert num == 2, f"expected 2 shards (6.0009 GiB over a 4 GiB threshold), got {num}"
assert total == 2**31 + 2**32 + 1000 * 2, "total_bytes mismatch"

got = {}
for shard in sorted(os.listdir(stage)):
    if shard.endswith(".safetensors"):
        for k, v in load_file(os.path.join(stage, shard)).items():
            got[k] = v
assert set(got) == {"layerA.weight", "layerB.weight", "layerC.weight"}, sorted(got)
assert torch.equal(got["layerA.weight"], p1.cpu()), "bf16 bytes differ"
assert torch.equal(got["layerB.weight"], p2.cpu().float()), "f32 bytes differ"
assert torch.equal(got["layerC.weight"], p3.cpu()), "f16 bytes differ"

idx = json.load(open(os.path.join(stage, "model.safetensors.index.json")))
assert "legacy.key" in idx["weight_map"], "source weight_map entry dropped"
assert idx["metadata"]["total_size"] == total
assert os.path.exists(os.path.join(stage, "config.json")), "meta copy missing"
for part in idx["weight_map"].values():
    assert "STAGECOUNT" not in part, f"un-renamed shard name {part}"

print("PASS: bytes exact, shards=2, index + meta ok")
dist.destroy_process_group()
