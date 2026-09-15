#!/usr/bin/env python3
"""Strip the legacy-namespace BF16 carry-over from a prod export.

Our export driver re-attaches the original linearized-BF16 tensors
(legacy ``model.*`` names) next to the serving-namespace quantized tree
(``model.language_model.*``). Only the legacy PLE n-gram shards are still
needed from that set; the per-expert BF16 originals and leftover fused
``experts.gate_up_proj``/``down_proj`` BF16 copies are dead weight the
engines never load. Full retained files are hard-linked into the output
dir; files with a mix are rewritten without the dead tensors, and the
index + total_size are rebuilt.

Usage: slim_nvfp4.py SRC_DIR DST_DIR [--keep-regex REGEX]
"""
import argparse
import json
import re
import struct
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_KEEP = r"^model\.language_model\.layers\.\d+\.ple\."


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--keep-regex", default=DEFAULT_KEEP,
                    help="legacy-prefix tensors to keep (default: PLE)")
    a = ap.parse_args()
    keep = re.compile(a.keep_regex)

    src, dst = a.src, a.dst
    dst.mkdir(parents=True, exist_ok=True)
    index = json.load(open(src / "model.safetensors.index.json"))
    wm = index["weight_map"]

    # The engine's loader strips the VLM prefix (``model.language_model.X``
    # -> ``model.X``) and serves quantized experts from the per-expert
    # ``.weight_packed``/``.weight_scale`` set. Dead for serving are the
    # pre-quant BF16 originals: per-expert ``.proj.weight`` under either
    # prefix, and fused BF16 ``experts.gate_up_proj``/``down_proj``
    # leftovers (which would otherwise hit the fused-quant loader path).
    hdr_cache = {}

    def dead(t, meta):
        if ".experts." not in t and "experts." not in t:
            return False
        if keep.match(t):
            return False
        if meta["dtype"] != "BF16":
            return False
        return (t.endswith("_proj.weight")
                or t.endswith("experts.gate_up_proj")
                or t.endswith("experts.down_proj"))

    for f in sorted(src.glob("*.safetensors")):
        hdr_cache[f.name] = read_header(f)
    drop = {t for f in hdr_cache for t, m in hdr_cache[f].items()
            if dead(t, m)}
    print(f"[slim] dropping {len(drop)} tensors of {len(wm)}")

    drop_by_file = {}
    for t in drop:
        f = wm.get(t)
        if f:
            drop_by_file.setdefault(f, set()).add(t)
    all_files = {f for f in wm.values()} | set(drop_by_file)
    for f in sorted(drop_by_file):
        names = set(hdr_cache[f])
        kept = names - drop_by_file[f]
        p = src / f
        if kept == names:
            continue
        if not kept:
            print(f"[slim] drop whole file {f}")
            continue
        print(f"[slim] rewrite {f}: keep {len(kept)}/{len(names)}")
        with safe_open(p, framework="pt") as sf:
            tensors = {k: sf.get_tensor(k) for k in kept}
        save_file(tensors, str(dst / f), metadata={"format": "pt"})
        del tensors
    for f in sorted(all_files - set(drop_by_file)):
        (dst / f).hardlink_to(src / f)

    # rebuild index
    new_wm = {}
    total = 0
    ES = {"BF16": 2, "F8_E4M3": 1, "F32": 4, "U8": 1, "I64": 8, "F16": 2,
          "I32": 4, "I8": 1, "F64": 8, "BOOL": 1, "UI8": 1}
    for f in all_files:
        dropped = drop_by_file.get(f, set())
        kept_names = set(hdr_cache[f]) - dropped
        if not (dst / f).exists() and not kept_names:
            continue
        if not (dst / f).exists():
            raise SystemExit(f"missing output file {f}")
        for t in kept_names:
            new_wm[t] = f
            m = hdr_cache[f][t]
            b = ES.get(m["dtype"], 0)
            for d in m["shape"]:
                b *= d
            total += b
    if set(new_wm) != set(wm) - drop:
        raise SystemExit("index rebuild lost/gained tensors")
    index["weight_map"] = new_wm
    index["metadata"]["total_size"] = total
    json.dump(index, open(dst / "model.safetensors.index.json", "w"), indent=1)
    for m in ("config.json", "generation_config.json", "hf_quant_config.json",
              "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "preprocessor_config.json"):
        if (src / m).exists():
            (dst / m).hardlink_to(src / m)
    print(f"[slim] done: {len(new_wm)} tensors, total_size {total/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
