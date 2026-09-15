#!/usr/bin/env python3
"""Stage the PLE n-gram table to /dev/shm once, for all ranks to mmap.

The 51.2B-param n-gram embedding (95 GiB BF16) is too large to load per FSDP2
rank, but it must still run during calibration forward. The linearized
checkpoint stores it in a safetensors shard of its own, so prep is a copy of
that shard into the tmpfs, plus a manifest of tensor offsets read from the
safetensors header. At runtime each rank mmaps the same tmpfs file: safetensors
and torch.frombuffer over MAP_SHARED page-cache pages means all 8 ranks read
ONE physical copy (tmpfs counts once against the pod memory limit).

Idempotent: verifies by size; pass --force to re-copy.
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

HEADER_MAX = 1 << 28


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > HEADER_MAX:
            raise ValueError(f"suspicious safetensors header length {n} in {path}")
        return json.loads(f.read(n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="staged BF16 checkpoint dir")
    ap.add_argument("--shm-dir", default="/dev/shm/flashnext-ple")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    shm = Path(args.shm_dir)
    src_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    # PLE tables (huge, mmap-shared) plus the small aux tensors (vision/MTP)
    # that the FSDP2 loader skips: the janitor deletes the staged checkpoint
    # before export runs, so reattach_skipped_weights() must find them here.
    ple_keys = sorted(
        k for k in src_map if ".ple." in k or ".visual." in k or ".mtp" in k
    )
    if not ple_keys:
        sys.exit("no .ple./.visual./.mtp keys in the checkpoint index")

    shm.mkdir(parents=True, exist_ok=True)
    # Group by shard, keep shard names so the runner's safe_open path also
    # works for the small non-table PLE tensors.
    meta = {"model_dir": str(model_dir), "shards": {}}
    for shard in sorted({src_map[k] for k in ple_keys}):
        src = model_dir / shard
        dst = shm / shard
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or args.force:
            print(f"copying {shard} ({src.stat().st_size / 2**30:.1f} GiB) -> shm")
            shutil.copyfile(src, dst.with_suffix(dst.suffix + ".tmp"))
            dst.with_suffix(dst.suffix + ".tmp").rename(dst)
        hdr = read_header(dst)
        meta["shards"][shard] = {
            "path": str(dst),
            "data_start": struct.unpack("<Q", dst.open("rb").read(8))[0]
            + len(json.dumps({k: hdr[k] for k in hdr if k != "__metadata__"}).encode()),
        }
    meta["keys"] = {k: src_map[k] for k in ple_keys}
    # Recompute data_start properly: header json length is exact, not re-dumped.
    for shard, entry in meta["shards"].items():
        p = Path(entry["path"])
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            f.read(n)
            entry["data_start"] = 8 + n
    out = shm / "manifest.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"PLE shm manifest -> {out} ({len(ple_keys)} tensors)")


if __name__ == "__main__":
    main()
