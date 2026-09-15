#!/usr/bin/env python3
"""Mirror the orca serving release and overlay our calibrated tensors.

Our export's file layout (legacy non-VLM prefix, ModelOpt tensor names,
pre-quant BF16 re-attach under model.language_model.*) is not what
sglang serves cleanly. Instead of translating it, we adopt the orcarouter
release of the same base as the layout: the output directory reproduces
orca's shard files and tensor names, and

  - every tensor we have (resolved by module after name mapping) comes
    from our checkpoint — experts (our ``.weight`` -> ``.weight_packed``,
    our ``.weight_scale_2`` -> ``.weight_global_scale``), FP8
    attention/shared weights with their channel scales, AWQ-touched
    norms/gates, embed, lm_head;
  - anything we lack is copied verbatim from orca (vision tower, PLE
    shards, MTP head, bookkeeping);
  - orca's stacked ``experts.gate_up_proj``/``experts.down_proj`` copies
    are dropped ONLY where we ship per-expert tensors for that mlp (the
    quantized main stack); the MTP layer's BF16 stacked experts have no
    per-expert counterpart on our side and are copied verbatim, keeping
    orca's tensor-name set 100% covered;
  - where our export carries both the quantized ``model.layers.*`` name
    and the pre-quant BF16 re-attach under ``model.language_model.*``,
    the QUANTIZED tensor wins — the old first-wins canon map could pick
    the BF16 carry and silently overwrite orca's FP8 attention weights;
  - QSA ``self_attn.indexer.*`` is served BF16 (matching orca): where our
    tree quantized it, the BF16 re-attach value replaces the FP8 overlay
    and both quant manifests stop declaring those layers;
  - our k_scale/v_scale/input_scale tensors (which have no orca-space
    counterpart) ride along in one extra shard;
  - config files stay ours (VLM config adopted from the BF16 base + our
    hf_quant_config.json with the static FP8 KV declaration).

Usage: serving_repack.py ORCA_DIR OUR_SLIM_DIR DST_DIR
"""
import argparse
import json
import struct
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

ES = {"BF16": 2, "F8_E4M3": 1, "F32": 4, "U8": 1, "I64": 8, "F16": 2,
      "I32": 4, "I8": 1, "F64": 8, "BOOL": 1, "UI8": 1}


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def canon(name):
    return name.replace("model.language_model.", "model.", 1)


def our_counterpart(orca_name):
    """Our-side (canonical) tensor name for an orca name, or itself."""
    c = canon(orca_name)
    for suffix, ours in ((".weight_packed", ".weight"),
                         (".weight_global_scale", ".weight_scale_2")):
        if c.endswith(suffix):
            return c[: -len(suffix)] + ours
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orca", type=Path)
    ap.add_argument("slim", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)

    owm = json.load(open(a.orca / "model.safetensors.index.json"))["weight_map"]
    swm = json.load(open(a.slim / "model.safetensors.index.json"))["weight_map"]
    scanon = {}
    for t in swm:
        if not t.startswith("model.language_model."):
            scanon.setdefault(canon(t), t)   # quantized tree wins
    for t in swm:
        scanon.setdefault(canon(t), t)       # BF16 re-attach only as fallback
    our_mlp = {canon(t).rsplit(".experts.", 1)[0] for t in scanon
               if ".experts." in canon(t)}

    # resolve every orca tensor: our value (under orca's name) or orca's
    pick_ours = {}   # orca_name -> our original name
    pick_orca = {}   # orca_name -> orca file
    n_stacked = 0
    for t in owm:
        c = canon(t)
        if (c.endswith("experts.gate_up_proj") or c.endswith("experts.down_proj")) \
                and c.rsplit(".experts.", 1)[0] in our_mlp:
            n_stacked += 1   # our per-expert tensors replace this fused view
            continue
        ours = our_counterpart(t)
        if ours in scanon and swm[scanon[ours]]:
            pick_ours[t] = scanon[ours]
        else:
            pick_orca[t] = owm[t]
    covered = {canon(our_counterpart(t)) for t in pick_ours}
    extras = [t for c, t in scanon.items()
              if c.endswith((".input_scale", ".k_scale", ".v_scale"))
              and c not in covered]
    print(f"[repack] orca layout {len(owm)} | ours {len(pick_ours)} | "
          f"orca-copied {len(pick_orca)} | dropped stacked {n_stacked} | "
          f"carried scales {len(extras)}")

    o_hdrs = {f: read_header(a.orca / f) for f in set(owm.values())}
    s_hdrs = {f: read_header(a.slim / f) for f in set(swm.values())}

    # QSA indexers steer the sparse-attention top-k; quant noise there
    # shifts retrieval, so ship them BF16 like orca. The pre-quant BF16
    # re-attach in our slim carries the unquantized value (provenance:
    # both bases descend from the same uncensored checkpoint).
    n_unquant = 0
    for t in list(pick_ours):
        if ".indexer." not in canon(t):
            continue
        cur = pick_ours[t]
        if s_hdrs[swm[cur]][cur]["dtype"] == "BF16":
            continue
        if t in swm and s_hdrs[swm[t]][t]["dtype"] == "BF16":
            pick_ours[t] = t
            n_unquant += 1
    extras = [e for e in extras if ".indexer." not in canon(e)]

    def osz(name):  # size of a picked-ours tensor
        e = s_hdrs[swm[name]][name]
        b = ES.get(e["dtype"], 0)
        for d in e["shape"]:
            b *= d
        return b

    # group by file to minimize shard opens
    orca_by_file = {}
    for t, f in pick_orca.items():
        orca_by_file.setdefault(f, []).append(t)
    ours_by_ourfile = {}
    for t, s in pick_ours.items():
        ours_by_ourfile.setdefault(swm[s], []).append((t, s))
    # destination file name = orca's file; our overlays for that file are
    # gathered across ALL our files, so stage them per destination file
    dest_overlay = {}  # orca_file -> {orca_name: (our_file, our_name)}
    our_by_canon_file = {}
    for t, s in pick_ours.items():
        of = owm[t]
        dest_overlay.setdefault(of, {})[t] = (swm[s], s)

    new_wm, total = {}, 0
    for f in sorted(set(owm.values())):
        tensors = {}
        if f in orca_by_file:
            with safe_open(a.orca / f, framework="pt") as sf:
                for t in orca_by_file[f]:
                    tensors[t] = sf.get_tensor(t)
            for t in orca_by_file[f]:
                total += osz_orca(o_hdrs[f][t])
                new_wm[t] = f
        if f in dest_overlay:
            cache = {}
            for t, (of, on) in dest_overlay[f].items():
                if of not in cache:
                    cache[of] = safe_open(a.slim / of, framework="pt")
                tensors[t] = cache[of].get_tensor(on)
                total += osz(on)
                new_wm[t] = f
            for sf in cache.values():
                pass  # safetensors files need no explicit close here
        if tensors:
            save_file(tensors, str(a.dst / f), metadata={"format": "pt"})

    if extras:
        tensors = {}
        by_file = {}
        for t in extras:
            by_file.setdefault(swm[t], []).append(t)
        for f, ts in by_file.items():
            with safe_open(a.slim / f, framework="pt") as sf:
                for t in ts:
                    tensors[t] = sf.get_tensor(t)
        save_file(tensors, str(a.dst / "scales.safetensors"),
                  metadata={"format": "pt"})
        for t, tensor in tensors.items():
            total += tensor.numel() * tensor.element_size()
            new_wm[t] = "scales.safetensors"

    index = json.load(open(a.orca / "model.safetensors.index.json"))
    index["weight_map"] = new_wm
    index["metadata"]["total_size"] = total
    json.dump(index, open(a.dst / "model.safetensors.index.json", "w"), indent=1)
    for m in ("config.json", "generation_config.json", "hf_quant_config.json",
              "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "preprocessor_config.json", "video_preprocessor_config.json",
              "vocab.json", "merges.txt"):
        src, dst = a.slim / m, a.dst / m
        if src.exists():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.hardlink_to(src)
    if n_unquant:
        # mirror manifests must stop declaring the indexer layers we un-quantized
        for src_name, section in (("hf_quant_config.json", "quantization"),
                                  ("config.json", "quantization_config")):
            src = a.slim / src_name
            if not src.exists():
                continue
            doc = json.load(open(src))
            ql = doc[section]["quantized_layers"]
            for k in [k for k in ql if "indexer" in k]:
                del ql[k]
            if src_name == "config.json":
                doc[section].setdefault("ignore", []).append(r"re:.*indexer.*")
            json.dump(doc, open(a.dst / src_name, "w"), indent=2)
        print(f"[repack] QSA indexers kept BF16: {n_unquant} swapped, "
              "quantized_layers entries dropped")
    print(f"[repack] done: {len(new_wm)} tensors, total_size {total/2**30:.1f} GiB")


def osz_orca(e):
    b = ES.get(e["dtype"], 0)
    for d in e["shape"]:
        b *= d
    return b


if __name__ == "__main__":
    main()
