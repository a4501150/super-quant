#!/usr/bin/env python3
"""Match the reference NVFP4 release's tensor structure, key for key.

Two structural divergences that break vLLM serving while dequantized
values are all correct (garbage text + NaN logits):

1. weight_global_scale stored as scalar () — the offload pipeline writes
   scalars, modelopt/orca write a 1-element tensor (1,); widen to (1,).
2. FP8 weight_scale in float32 — cast to bfloat16 like the reference.

Run after requant_fp8_scales.py. Idempotent.

Usage: fix_global_scale_shape.py --tree /tmp/hf_upload
"""
import argparse, glob, json, os, struct
from safetensors import safe_open
from safetensors.torch import save_file
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
a = ap.parse_args()
n_fix = 0
for f in sorted(glob.glob(a.tree + "/model*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n)).get("__metadata__")
    final = {}
    changed = False
    with safe_open(f, framework="pt") as t:
        for k in t.keys():
            v = t.get_tensor(k)
            if k.endswith("weight_global_scale") and v.shape == torch.Size([]):
                v = v.reshape(1)
                changed = True
                n_fix += 1
            elif k.endswith(".weight_scale") and v.dtype == torch.float32:
                v = v.to(torch.bfloat16)
                changed = True
            final[k] = v
    if changed:
        save_file(final, f + ".new", metadata=meta)
        os.replace(f + ".new", f)
        print("[fixshape]", os.path.basename(f), flush=True)
print("[fixshape] DONE globals_scaled widened:", n_fix, flush=True)
