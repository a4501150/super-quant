#!/usr/bin/env python3
"""Fix inverted NVFP4 weight_global_scale in a serving tree.

Root cause: our export stored weight_global_scale as the DEQUANT-side value
(small, ~1/amax-scaled), but vLLM's CT NVFP4 path computes the kernel alpha
as 1.0/weight_global_scale — i.e. the checkpoint must store the QUANT-side
global multiplier (modelopt/NVIDIA convention, same as the orca tree:
block scales already live in the gs-scaled space and saturate near 448).
With the reciprocal stored, every expert tensor is rescaled by gs^2
(~1e-8..1e8 per tensor) -> garbage logits and NaN logprobs at serve time,
even though dequantizing with a *multiply* (the wrong direction) hides the
bug in offline checks because the two inversions cancel.

Codes and block scales are untouched; only the scalar tensor is inverted.
"""
import glob, json, os, struct
from safetensors import safe_open
from safetensors.torch import save_file
import torch

tree = "/tmp/flashnext-serving"
n_fixed = 0
n_seen = 0
report = {}
for f in sorted(glob.glob(tree + "/model*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n)).get("__metadata__")
    with safe_open(f, framework="pt") as t:
        keys = list(t.keys())
        touched = False
        final = {}
        for k in keys:
            v = t.get_tensor(k)
            if k.endswith(".weight_global_scale"):
                n_seen += 1
                assert v.dtype in (torch.float32, torch.bfloat16) and v.numel() == 1, \
                    (k, v.dtype, tuple(v.shape))
                old = v.float().item()
                v = torch.tensor([1.0 / old], dtype=v.dtype)
                n_fixed += 1
                touched = True
                if len(report) < 3:
                    report[k] = (old, 1.0 / old)
            final[k] = v
    if touched:
        save_file(final, f + ".new", metadata=meta)
        os.replace(f + ".new", f)
        print(f"[invert] {os.path.basename(f)} done", flush=True)
print(f"[invert] seen={n_seen} fixed={n_fixed} samples={json.dumps(report)}")
print("[invert] DONE")
