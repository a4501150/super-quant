#!/usr/bin/env python3
"""NVFP4-vs-BF16 agreement benchmark (quant_agreement).

Reproduces the published comparison protocol:
  - corpus: thinking-mode output self-distilled from the BF16 base,
    1-2 seed prompts per category (coding/math/reasoning/writing);
  - top-1: raw argmax agreement with BF16 under teacher forcing over the
    corpus; buckets by BF16 top1-top2 logprob margin: near-tie <0.5,
    moderate 0.5-2, confident 2-5, certain >5;
  - divmed: median first-divergence token index over N free greedy
    generations vs the BF16 outputs;
  - tok/s: served decode throughput (sglang.bench_serving, recorded
    separately by the driver script).

Talks to the serving engine natively: SGLang via /generate
(return_logprob / input_top_logprobs), vLLM via /v1/completions
(prompt_logprobs, return_token_ids). Both are normalized to the same
per-position [[logprob, token_id, None], ...] layout; field names are
probed once and reported on mismatch.

Usage:
  quant_agreement.py gen    --engine vllm --port 8000 --run run_bf16  # BF16 only
  quant_agreement.py score  --engine vllm --port 8000 --run run_bf16 --label ours
  quant_agreement.py table  run_bf16
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:{port}"
BUCKETS = [("near-tie", lambda m: m < 0.5), ("moderate", lambda m: 0.5 <= m < 2),
           ("confident", lambda m: 2 <= m < 5), ("certain", lambda m: m >= 5)]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(value, path):
    with open(path, "w") as f:
        json.dump(value, f)


def post(port, path, payload, timeout=3600):
    req = urllib.request.Request(
        API.format(port=port) + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# seed prompts: 2 per category. Long-open-ended so thinking-mode outputs
# run long enough to reach the ~140K-token corpus target of the protocol.
SEEDS = [
    ("coding", "Design and implement a rate-limiter library (token bucket, sliding window, and GCRA variants) in Python. Compare the three algorithms for correctness under concurrent async access, discuss trade-offs, and write a test suite that would catch a subtle off-by-one in each."),
    ("coding", "You are reviewing a pull request that replaces an LRU cache with an LFU cache in a KV store. Enumerate every behavior change this could cause (eviction fairness, memory growth, lock contention, cold-start patterns), and give concrete scenarios where the new cache performs worse."),
    ("math", "Prove that there are infinitely many primes p ≡ 3 (mod 4). Then generalize: for which residue classes a mod m can you give an Euclid-style proof, and where does the method break? Connect the answer to Dirichlet's theorem."),
    ("math", "A random walk on Z^2 starts at the origin. Estimate, with full reasoning, the expected number of visits to the origin before the walk first exits the disk of radius R, as R → ∞. Justify each approximation and bound the error of your heuristic."),
    ("reasoning", "A company observes that its median customer lifetime value rose 12% after it deliberately shut down its two worst-performing regions. Explain every way this observation could mislead a board slide, and design the analysis that would settle each doubt."),
    ("reasoning", "Six people, three projects, unknown preferences. A set of public statements is given, some of which are verifiably lies (each liar always lies, each truth-teller always tells the truth). Work through the logic exhaustively, model each assignment, and present the unique consistent schedule — or prove none exists."),
    ("writing", "Write a long-form essay comparing how oral epic traditions and early print culture shaped ideas of authorship, ending with three testable claims about how AI-generated text will change attribution norms."),
    ("writing", "Rewrite the same incident — a bridge collapsing during its own dedication ceremony — four times: as a wire report, a grieving first-person account, an engineering root-cause memo, and a satirical op-ed. Then analyze what each version had to suppress."),
]
GREEDY_TEMPLATES = [
    "Give a {n}-step plan to: {task}", "Explain {topic} to a beginner.",
    "List five common mistakes in {topic} and how to avoid them.",
    "Compare and contrast: {a} vs {b}.",
]
TASKS = ["debug a flaky CI", "migrate a monolith to services",
         "prepare a backyard camping trip", "negotiate a salary raise",
         "reduce household food waste", "learn basic music theory",
         "audit a small ecommerce checkout", "recover from a failed raid"]
TOPICS = ["gradient descent", "container networking", "auction theory",
          "the placebo effect", "B+ trees", "options pricing",
          "cache coherence", "the tragedy of the commons"]
PAIRS = [("TCP", "UDP"), ("monoids", "categories"), ("SQLite", "Postgres"),
         ("gRPC", "REST"), ("goroutines", "threads"), ("PCA", "t-SNE"),
         ("LRU", "LFU"), ("NFPA", "NVFP4")]


def greedy_prompts():
    out, n = [], 0
    for t in TASKS:
        for tpl in GREEDY_TEMPLATES[:2]:
            out.append(tpl.format(n=5, task=t, topic=TOPICS[n % len(TOPICS)],
                                  a=PAIRS[n % len(PAIRS)][0],
                                  b=PAIRS[n % len(PAIRS)][1]))
            n += 1
    for topic in TOPICS:
        for tpl in GREEDY_TEMPLATES[1:]:
            out.append(tpl.format(n=5, task="X", topic=topic,
                                  a=PAIRS[n % len(PAIRS)][0],
                                  b=PAIRS[n % len(PAIRS)][1]))
            n += 1
    for a, b in PAIRS:
        for _ in range(20):
            out.append(GREEDY_TEMPLATES[3].format(a=a, b=b, n=5,
                                                  topic=topic, task="X")
                       + f" (perspective #{len(out) % 10})")
    return out


def chat(port, prompt, max_tokens=32768):
    r = post(port, "/v1/chat/completions", {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6, "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }, timeout=7200)
    ch = r["choices"][0]
    m = ch["message"]
    text = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
    # finish_reason == "length" marks a cap-pinned doc; the corpus table
    # should say so rather than let readers infer it from token counts.
    return text, ch.get("finish_reason"), r.get("usage", {}).get("completion_tokens")


def logprob_scoring(port, text, engine):
    """Teacher-forced per-position (top1_id, margin) from prompt logprobs."""
    if engine == "vllm":
        r = post(port, "/v1/completions", {
            "model": "default", "prompt": text,
            "max_tokens": 1, "temperature": 0, "prompt_logprobs": 2,
        }, timeout=7200)
        pl = r["choices"][0].get("prompt_logprobs")
        if pl is None:
            sys.exit(f"unexpected completion keys: {sorted(r['choices'][0])}")
        out = []
        for row in pl:
            # position 0 has no context; vLLM reports it as null
            if not row:
                out.append([])
                continue
            ent = sorted(((float(v["logprob"]), int(tid))
                          for tid, v in row.items()), reverse=True)
            out.append([[lp, tid, None] for lp, tid in ent])
        return out
    r = post(port, "/generate", {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True, "logprob_start_len": 0,
        "top_logprobs_num": 2,
    }, timeout=7200)
    mi = r["meta_info"]
    top = mi.get("input_top_logprobs") or mi.get("input_token_logprobs")
    if top is None:
        sys.exit(f"unexpected meta_info keys: {sorted(mi)}")
    return top


def score_corpus(port, corpus, engine):
    """Teacher-force every doc separately and concatenate the position
    rows. One request per doc keeps each prompt under max-model-len even
    with 64K-token outputs and keeps arm score files index-aligned by
    construction (all arms score the same corpus.json)."""
    rows = []
    for c in corpus:
        rows.extend(logprob_scoring(port, c["prompt"] + "\n" + c["text"],
                                    engine))
    return rows


def greedy_ids(port, prompt, n, engine):
    """Greedy continuation token ids, up to n tokens."""
    if engine == "vllm":
        r = post(port, "/v1/completions", {
            "model": "default", "prompt": prompt, "temperature": 0,
            "max_tokens": n, "return_token_ids": True})
        ids = r["choices"][0].get("token_ids")
        if ids is None:
            sys.exit(f"vllm completion keys: {sorted(r['choices'][0])}")
        return ids
    r = post(port, "/generate", {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": n},
        "return_logprob": True, "top_logprobs_num": 1})
    return [e[1] for e in r["meta_info"]["output_token_logprobs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "score", "table"])
    ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--run", default="run_bf16")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="gen-phase per-stream output cap")
    a = ap.parse_args()
    run = Path(a.run); run.mkdir(exist_ok=True)

    if a.phase == "gen":  # run this against the BF16 server
        # All streams run concurrently: wall time is the longest single
        # generation, not the sum (sglang continuous batching absorbs the
        # aggregate decode).
        from concurrent.futures import ThreadPoolExecutor

        def gen_seed(i_cat_p):
            i, (cat, p) = i_cat_p
            t0 = time.time()
            out, fr, ctok = chat(a.port, p, max_tokens=a.max_tokens)
            print(f"[gen] {cat}: {len(out)} chars finish={fr} "
                  f"completion_tokens={ctok}", flush=True)
            return i, {"category": cat, "prompt": p, "text": out,
                       "finish_reason": fr, "completion_tokens": ctok,
                       "secs": time.time() - t0}

        if (run / "corpus.json").exists():
            corpus = load_json(run / "corpus.json")
            print(f"[gen] reusing corpus: {len(corpus)} rows", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=len(SEEDS)) as ex:
                rows = list(ex.map(gen_seed, enumerate(SEEDS)))
            corpus = [r for _, r in sorted(rows)]
            dump_json(corpus, run / "corpus.json")
        gp = greedy_prompts()
        assert len(gp) >= 200, len(gp)

        def gen_greedy(p):
            return {"prompt": p, "ids": greedy_ids(a.port, p, 512, a.engine)}

        with ThreadPoolExecutor(max_workers=16) as ex:
            greedy = list(ex.map(gen_greedy, gp[:200]))
        dump_json(greedy, run / "greedy_bf16.json")
        sc = score_corpus(a.port, corpus, a.engine)
        dump_json(sc, run / "score_bf16.json")
        print(f"[gen] corpus positions scored: {len(sc)}")
        return

    if a.phase == "score":
        label = a.label or sys.exit("--label required")
        corpus = load_json(run / "corpus.json")
        sc = score_corpus(a.port, corpus, a.engine)
        dump_json(sc, run / f"score_{label}.json")
        # divmed vs BF16 greedy (concurrent streams; divs stay index-aligned)
        from concurrent.futures import ThreadPoolExecutor

        greedy = load_json(run / "greedy_bf16.json")

        def one_div(g):
            ids = greedy_ids(a.port, g["prompt"], len(g["ids"]) or 1,
                             a.engine)
            return next((i for i, (x, y) in enumerate(zip(g["ids"], ids))
                         if x != y), min(len(g["ids"]), len(ids)))

        with ThreadPoolExecutor(max_workers=16) as ex:
            divs = list(ex.map(one_div, greedy))
        dump_json(divs, run / f"div_{label}.json")
        print(f"[score] {label}: {len(sc)} positions, "
              f"divmed={statistics.median(divs)}")
        return

    # table phase
    run = Path(a.run)
    bf16 = load_json(run / "score_bf16.json")

    def top(e):  # -> (logprob, token_id) or None for the no-context row
        return None if not e or e[0][0] is None else (e[0][0], e[0][1])

    rows, marg = [], {}
    for i, e in enumerate(bf16):
        t = top(e)
        if t is None:
            continue
        second = e[1][0] if len(e) > 1 and e[1][0] is not None else -1e9
        rows.append(i)
        marg[i] = t[0] - second
    print(f"[table] scored positions: {len(rows)}")
    for f in sorted(run.glob("score_*.json")):
        label = f.stem.removeprefix("score_")
        if label == "bf16":
            continue
        arm = load_json(f)
        agree = {n: 0 for n, _ in BUCKETS}
        tot = {n: 0 for n, _ in BUCKETS}
        hits = 0
        for i in rows:
            if i >= len(arm):
                print(f"[table] {label}: tokenization drift ({len(rows)} vs {len(arm)} positions), truncating")
                break
            a = top(arm[i])
            if a is None:
                continue
            m = marg[i]
            name = next(n for n, f2 in BUCKETS if f2(m))
            tot[name] += 1
            if a[1] == bf16[i][0][1]:
                hits += 1
            else:
                agree[name] += 1
        div = run / f"div_{label}.json"
        dm = int(statistics.median(load_json(div))) if div.exists() else None

        def pct(k, agree=agree, tot=tot):
            return 100.0 * agree[k] / tot[k] if tot[k] else 0.0

        print(f"{label}: top1_agree={100.0 * hits / len(rows):.2f}%  "
              + "  ".join(f"{k[:4]}_disagree={pct(k):.2f}%" for k, _ in BUCKETS)
              + f"  divmed={dm}")


if __name__ == "__main__":
    main()
