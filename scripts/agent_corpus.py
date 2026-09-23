#!/usr/bin/env python3
"""Self-distill a thinking-mode corpus through a minimal agentic harness.

Single-shot chat requests leave an agent-trained model without the loop it
was trained for: thinking drifts into tool calls that cannot exist. This
script runs each seed task as a real tool loop against the served model
(system prompt + one ``run_bash`` tool executed locally, results fed
back), which is the deployment shape we care about, at ~the cost of the
plain-corpus pass. The corpus row's ``text`` is the serialized transcript
(assistant text plus fed-back tool outputs); downstream scoring treats it
as opaque reference text exactly like quant_agreement.py's corpus.

No output caps are sent: max_tokens is omitted so each request may run to
the server's native context, and the only loop bound is max_turns (a
runaway guard, not a budget).

Usage: agent_corpus.py --port 8000 --out run_bf16
Requires the server to run with --enable-auto-tool-choice
--tool-call-parser qwen3_coder.
"""
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "http://127.0.0.1:{port}/v1/chat/completions"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(value, path):
    with open(path, "w") as f:
        json.dump(value, f)


SYSTEM = (
    "You are an engineering agent. You accomplish tasks by writing and "
    "running shell commands (typically python3 -c, or heredoc python) with "
    "the run_bash tool, inspecting their output, and iterating until the "
    "result is verified. Think first, then act."
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Run a bash command in a fresh sandbox; stdout/stderr "
                       "go to output. Python via `python3` is available.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "timeout": {"type": "integer",
                            "description": "seconds to wait; default 30, "
                                           "capped at 600"},
            },
            "required": ["cmd"],
        },
    },
}]

# Tools are exercised where the deployment uses them: the coding seeds run
# the agentic loop; everything else is plain single-turn thinking with no
# tools in the request (so thinking cannot lean on a harness that isn't
# there). Corpus therefore covers both served modes. Tuple flag: agentic.
SEEDS = [
    ("coding", "Implement a rate-limiter library (token bucket, sliding window, GCRA) in Python with a test suite; run the tests with run_bash, fix failures, and only summarize once they all pass.", True),
    ("coding", "Write an LFU cache with O(1) operations plus a micro-benchmark against a dict-LRU baseline; run both with run_bash and report measured behavior on a Zipf workload.", True),
    ("math", "Prove that √2 is irrational. Generalize: for which positive integers n is √n irrational — prove your classification. Then determine for which pairs (a,b) of positive integers √a + √b is rational, and prove it.", False),
    ("math", "Walk through the covering-congruence obstruction to a Euclidean proof for primes a (mod m): construct, by hand, why residues with a = 1 (mod 8) defeat the classic argument, and state what analytic input replaces it.", False),
    ("reasoning", "A company observes its median customer LTV rose 12% after it deliberately shut down its two worst-performing regions. Explain every way this could mislead a board slide, and design the analysis that would settle each doubt.", False),
    ("reasoning", "Six people, three projects, statements from known liars (always lie) and truth-tellers (always tell the truth). Work through the logic exhaustively, model each assignment, and present the unique consistent schedule — or prove none exists.", False),
    ("writing", "Write a long-form essay comparing how oral epic traditions and early print culture shaped ideas of authorship, ending with three testable claims about how AI-generated text will change attribution norms.", False),
    ("writing", "Rewrite the same incident — a bridge collapsing during its own dedication ceremony — four times: as a wire report, a grieving first-person account, an engineering root-cause memo, and a satirical op-ed. Then analyze what each version had to suppress.", False),
]


def exec_bash(cmd, cwd, timeout=30):
    timeout = min(int(timeout), 600)
    # own process group: subprocess's timeout only kills the direct child,
    # so bash-spawned python grandchildren escaped as CPU-burning orphans.
    p = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return out + (("\nSTDERR:\n" + err) if err else "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        return f"ERROR: {timeout}s timeout"
    except Exception as e:  # noqa: BLE001
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.communicate()
        return f"ERROR: {e}"


def post(port, payload, timeout=3600):
    req = urllib.request.Request(
        API.format(port=port), data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_task(port, cat, task, agentic, max_turns=None, dump_to=None):
    # max_turns=None: agentic rollouts end only by model stop or context
    # exhaustion; a turn cap would clip the trajectory mid-loop and cost
    # the closing summary turn (positions stay valid, the doc loses its
    # natural end — not worth it for an 8-doc corpus).
    if not agentic:
        # plain mode: no tools, no harness system prompt — the shape a
        # thinking request has when nobody is executing anything.
        r = post(port, {
            "model": "default",
            "messages": [{"role": "user", "content": task}],
            "temperature": 0.6, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": True}})
        ch = r["choices"][0]
        m = ch["message"]
        text = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
        row = {"category": cat, "prompt": task, "agentic": False,
               "text": text, "turns": 1,
               "finish_reason": ch.get("finish_reason"),
               "completion_tokens": r.get("usage", {}).get("completion_tokens")}
        if dump_to:
            dump_json(row, dump_to)
        return row
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": task}]
    sandbox = Path(f"/tmp/agent_sandbox_{abs(hash(task)) % 2**32}")
    sandbox.mkdir(exist_ok=True)
    transcript, toks, turns, fr = [], 0, 0, None
    while max_turns is None or turns < max_turns:
        r = post(port, {
            "model": "default", "messages": msgs, "tools": TOOLS,
            "tool_choice": "auto", "temperature": 0.6, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": True}})
        ch = r["choices"][0]
        m = ch["message"]
        fr = ch.get("finish_reason") or fr
        ct = r.get("usage", {}).get("completion_tokens", 0)
        toks += ct
        text = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
        calls = m.get("tool_calls") or []
        if text.strip():
            transcript.append(text)
        msgs.append(m)
        turns += 1
        if not calls:
            break
        for c in calls:
            args = json.loads(c["function"]["arguments"])
            cmd = args["cmd"]
            out = exec_bash(cmd, str(sandbox), args.get("timeout", 30))
            transcript.append(f"TOOL run_bash({cmd})\n{out}")
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": out})
    row = {"category": cat, "prompt": task, "agentic": True,
           "text": "\n".join(transcript), "turns": turns,
           "finish_reason": fr, "completion_tokens": toks}
    if dump_to:  # per-doc partial: peekable mid-run, salvageable on hang
        dump_json(row, dump_to)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="run_bf16")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None,
                    help="comma-separated seed indices to (re)run in place, "
                         "e.g. 0,1 — corpus.json is re-assembled from disk")
    ap.add_argument("--assemble", action="store_true",
                    help="only rebuild corpus.json from doc_<i>.json")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    if not a.assemble:
        idxs = ([int(x) for x in a.only.split(",")] if a.only
                else list(range(len(SEEDS))))
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(
                lambda i: run_task(a.port, *SEEDS[i],
                                   dump_to=str(out / f"doc_{i}.json")),
                idxs))
    rows = [load_json(out / f"doc_{i}.json")
            for i in range(len(SEEDS))]   # assemble: merged view of disk
    dump_json(rows, out / "corpus.json")
    for r in rows:
        print(f"[agent] {r['category']:>9} finish={r['finish_reason']} "
              f"toks={r['completion_tokens']} turns={r['turns']}")
    print(f"[agent] corpus written: {len(rows)} docs in "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
