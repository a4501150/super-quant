#!/usr/bin/env python3
"""Repair FP8 per-channel weight_scale in a serving tree without rebuilding.

Symptom it fixes: tree boots and loads, but the model emits garbage/NaN
logits. Diagnosis that identifies it: dequantizing float8_e4m3fn tensors
with their stored scales diverges from BF16 ground truth by O(1) relative
error, while a two-factor fit (output-channel scale x input-channel
diagonal) converges to the FP8 noise floor — i.e. the codes are good
quantizations of AWQ-smoothed targets but the scales shipped inconsistent
with them (NVFP4 expert blocks are unaffected).

Repair: re-quantize every float8_e4m3fn weight straight from the BF16
source tree with the standard CT recipe (scale = per-out-channel amax/448,
RNE). Uniform pass; already-correct FP8 tensors are harmlessly rewritten.
Archive shards stream one at a time from GCS, so peak disk stays ~1 shard.

Usage: requant_fp8_scales.py --tree /tmp/hf_upload --src-gs gs://.../20260912-linearized-bf16 --index /tmp/bf16_index.json
"""
import argparse, glob, json, os, struct, subprocess, sys
from safetensors import safe_open
from safetensors.torch import save_file
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True, help="serving tree to patch in place")
ap.add_argument("--src-gs", required=True, help="GS URI of the BF16 source build")
ap.add_argument("--index", required=True, help="local model.safetensors.index.json of the source")
a = ap.parse_args()
idx = json.load(open(a.index))["weight_map"]

plan = {}
for f in sorted(glob.glob(a.tree + "/model*.safetensors")):
    with safe_open(f, framework="pt") as t:
        keys = [k for k in t.keys() if k.endswith(".weight") and
                t.get_tensor(k).dtype == torch.float8_e4m3fn]
    if keys:
        plan[f] = keys
total = sum(len(v) for v in plan.values())
print(f"[requant] {total} fp8 tensors across {len(plan)} shards", flush=True)
missing = [k for ks in plan.values() for k in ks
           if not (idx.get(k) or idx.get(k + ".weight"))]
if missing:
    sys.exit(f"{len(missing)} fp8 keys absent from source index, e.g. {missing[:3]}")

cur = None
def ref(key):
    global cur
    shard = idx.get(key) or idx.get(key + ".weight")
    if shard != cur:
        lp = f"/tmp/bf16_{shard}"
        if not os.path.exists(lp):
            subprocess.run(["gcloud", "storage", "cp", "-n",
                            f"{a.src_gs.rstrip('/')}/{shard}", lp], check=True)
        if cur and os.path.exists(f"/tmp/bf16_{cur}"):
            os.remove(f"/tmp/bf16_{cur}")
        cur = shard
    with safe_open(f"/tmp/bf16_{shard}", framework="pt") as t:
        key = key if key in t.keys() else key + ".weight"
        return t.get_tensor(key).float()

done = 0
for f in sorted(plan):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n)).get("__metadata__")
    final = {}
    with safe_open(f, framework="pt") as t:
        for k in t.keys():
            v = t.get_tensor(k)
            if k.endswith(".weight_scale") and v.ndim == 1 and \
                    v.dtype in (torch.float32, torch.bfloat16):
                # The export leaves orphaned 1-D scales in a shard that
                # does not hold the weight; vLLM streams shard files
                # directly (ignores the index) and its QKV scale params
                # assert 2-D, so a stray flat scale crashes model load.
                # (fix_scales.py equivalent, applied in passing.)
                v = v.reshape(-1, 1)
            final[k] = v
    for k in plan[f]:
        w = ref(k)
        assert w.shape == final[k].shape, (k, tuple(w.shape), tuple(final[k].shape))
        s = w.abs().amax(dim=1, keepdim=True) / 448.0
        final[k] = (w / s).to(torch.float8_e4m3fn)
        sk = k[:-len(".weight")] + ".weight_scale"
        if sk in final:  # scale tensor lives with the weight
            final[sk] = s.float()
        done += 1
    save_file(final, f + ".new", metadata=meta)
    os.replace(f + ".new", f)
    print(f"[requant] {os.path.basename(f)} ok total={done}/{total}", flush=True)
print("[requant] DONE", flush=True)
