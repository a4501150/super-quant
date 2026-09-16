"""Functional test for overlapped stage_mutated_weights (world=1, gloo).

Run this file directly on a CUDA host with ModelOpt installed. It verifies shard
splitting at 4 GiB, byte-exact round-trip, writer completion, index output, and
metadata copying.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29611")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file

    import quantize_mopt as qm

    dist.init_process_group("gloo")
    dev = "cuda:0"

    try:
        model = torch.nn.Module()
        p1 = torch.randn(2**30, dtype=torch.bfloat16, device=dev)  # 2 GiB
        p2 = torch.arange(2**30, dtype=torch.float32, device=dev)  # 4 GiB
        p3 = torch.randn(1000, dtype=torch.float16, device=dev)
        for sub, param in (("layerA", p1), ("layerB", p2), ("layerC", p3)):
            mod = torch.nn.Module()
            mod.weight = torch.nn.Parameter(param, requires_grad=False)
            setattr(model, sub, mod)

        with tempfile.TemporaryDirectory(prefix="sqtest_src_") as src:
            with tempfile.TemporaryDirectory(prefix="sqtest_stage_") as stage:
                with open(
                    os.path.join(src, "model.safetensors.index.json"), "w"
                ) as f:
                    json.dump(
                        {
                            "weight_map": {
                                "legacy.key": "model-00001-of-00005.safetensors"
                            }
                        },
                        f,
                    )
                with open(os.path.join(src, "config.json"), "w") as f:
                    f.write('{"test": 1}')

                num, total = qm.stage_mutated_weights(
                    model,
                    src,
                    stage,
                    SimpleNamespace(is_main=True, rank=0, world_size=1, device=dev),
                )
                print(f"num_shards={num} total_bytes={total}")
                assert num == 2, (
                    f"expected 2 shards (6.0009 GiB over a 4 GiB threshold), got {num}"
                )
                assert total == 2**31 + 2**32 + 1000 * 2, "total_bytes mismatch"

                got = {}
                for shard in sorted(os.listdir(stage)):
                    if shard.endswith(".safetensors"):
                        for key, value in load_file(os.path.join(stage, shard)).items():
                            got[key] = value
                assert set(got) == {
                    "layerA.weight",
                    "layerB.weight",
                    "layerC.weight",
                }, sorted(got)
                assert torch.equal(got["layerA.weight"], p1.cpu()), "bf16 bytes differ"
                assert torch.equal(
                    got["layerB.weight"], p2.cpu().float()
                ), "f32 bytes differ"
                assert torch.equal(got["layerC.weight"], p3.cpu()), "f16 bytes differ"

                with open(os.path.join(stage, "model.safetensors.index.json")) as f:
                    index = json.load(f)
                assert "legacy.key" in index["weight_map"], (
                    "source weight_map entry dropped"
                )
                assert index["metadata"]["total_size"] == total
                assert os.path.exists(os.path.join(stage, "config.json")), (
                    "meta copy missing"
                )
                for part in index["weight_map"].values():
                    assert "STAGECOUNT" not in part, f"un-renamed shard name {part}"

                print("PASS: bytes exact, shards=2, index + meta ok")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
