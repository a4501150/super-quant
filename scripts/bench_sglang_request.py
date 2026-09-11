#!/usr/bin/env python3
"""Streamed SGLang request with TTFT / prefill / decode metrics.

Prints one JSON line. Sampling params read from BENCH_* env vars.
"""

import argparse
import json
import os
import re
import sys
import time
from urllib.request import Request, urlopen

EXPECTED_MARKER_RE = re.compile(
    r"Unique marker ([0-9a-f]{8})(?![0-9a-z])", re.IGNORECASE
)
RESPONSE_MARKER_RE = re.compile(r"(?<![0-9a-z])([0-9a-f]{8})(?![0-9a-z])")


def expected_marker_sequence(system, prompt):
    source = f"{system}\n{prompt}" if system else prompt
    return [marker.lower() for marker in EXPECTED_MARKER_RE.findall(source)]


def response_marker_sequence(response_text, expected_count):
    return RESPONSE_MARKER_RE.findall(response_text.lower())[:expected_count]


def main():
    ap = argparse.ArgumentParser(
        description="Send one streamed SGLang chat request and record latency metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Environment:
  BENCH_TEMPERATURE, BENCH_TOP_K, BENCH_TOP_P
  BENCH_REASONING_EFFORT, BENCH_ENABLE_THINKING
  BENCH_MIN_CACHE_HIT_RATIO  Fail if cached_tokens / prompt_tokens is lower.

Every standalone eight-hex 'Unique marker <hex>' in the prompt and system
content must appear in the response, in order. --must-match-markers enforces
that sequence without comparing optional prose. --must-match compares response
text with a file after whitespace normalization through the last marker.

Examples:
  uv run python scripts/bench_sglang_request.py \\
    --url http://127.0.0.1:8000/v1/chat/completions \\
    --model qwen3.8-flash-next --prompt 'Explain radix caches.'

  BENCH_MIN_CACHE_HIT_RATIO=0.95 uv run python scripts/bench_sglang_request.py \\
    --url http://127.0.0.1:8000/v1/chat/completions \\
    --model qwen3.8-flash-next --prompt-file prompt.txt --max-tokens 32
""",
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt")
    ap.add_argument(
        "--prompt-file", help="read prompt from file (avoids ARG_MAX for long prompts)"
    )
    system_group = ap.add_mutually_exclusive_group()
    system_group.add_argument("--system", help="system message content")
    system_group.add_argument("--system-file", help="read system message from file")
    ap.add_argument("--text-out", help="write truncated thinking/response text here")
    ap.add_argument("--out-json", help="write metrics JSON to this file")
    ap.add_argument("--full-text-out", help="write the full response text here")
    match_group = ap.add_mutually_exclusive_group()
    match_group.add_argument(
        "--must-match",
        help="fail unless the response matches this file (whitespace-normalized, up to the last marker)",
    )
    match_group.add_argument(
        "--must-match-markers",
        action="store_true",
        help="fail unless the response starts with the request's expected marker sequence",
    )
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--no-thinking", action="store_true")
    a = ap.parse_args()
    if a.prompt_file:
        with open(a.prompt_file) as f:
            a.prompt = f.read()
    elif not a.prompt:
        ap.error("--prompt or --prompt-file required")
    if a.system_file:
        with open(a.system_file) as f:
            a.system = f.read()

    expected_markers = expected_marker_sequence(a.system, a.prompt)
    if a.must_match_markers and not expected_markers:
        ap.error("--must-match-markers requires at least one Unique marker <eight-hex>")
    min_cache_hit_ratio = float(os.environ.get("BENCH_MIN_CACHE_HIT_RATIO", "0"))
    thinking = (
        os.environ.get("BENCH_ENABLE_THINKING", "true") == "true" and not a.no_thinking
    )
    messages = [] if a.system is None else [{"role": "system", "content": a.system}]
    messages.append({"role": "user", "content": a.prompt})
    body = {
        "model": a.model,
        "messages": messages,
        "temperature": float(os.environ.get("BENCH_TEMPERATURE", "1.0")),
        "top_k": int(os.environ.get("BENCH_TOP_K", "20")),
        "top_p": float(os.environ.get("BENCH_TOP_P", "0.95")),
        "reasoning_effort": os.environ.get("BENCH_REASONING_EFFORT", "medium"),
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if a.max_tokens:
        body["max_tokens"] = a.max_tokens
    if a.ignore_eos:
        body["ignore_eos"] = True
    req = Request(
        a.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.monotonic()
    ttft = None
    usage = {}
    reasoning, content = [], []
    with urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                rc = delta.get("reasoning_content")
                cc = delta.get("content")
                if rc or cc:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    if rc:
                        reasoning.append(rc)
                    if cc:
                        content.append(cc)
    wall = time.monotonic() - t0
    if ttft is None:
        ttft = wall

    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    computed = max(pt - cached, 0)
    response_text = "".join(reasoning) + "".join(content)
    rt = response_text.lower()
    response_markers = response_marker_sequence(response_text, len(expected_markers))
    missing_markers = [m for m in expected_markers if m not in rt]
    first_pos = [rt.find(m) for m in expected_markers]
    markers_out_of_order = (
        len(expected_markers) > 1
        and all(p >= 0 for p in first_pos)
        and first_pos != sorted(first_pos)
    )
    marker_valid = not missing_markers and not markers_out_of_order
    matched = None
    matched_strict = None
    mismatch_desc = ""
    if a.must_match_markers:
        got_sig = response_marker_sequence(response_text, len(expected_markers))
        matched = got_sig == expected_markers
        if not matched:
            mismatch_desc = (
                "response marker sequence mismatch: "
                f"expected {str(expected_markers)[:120]!r}, "
                f"got {str(got_sig)[:120]!r}"
            )
    elif a.must_match:
        with open(a.must_match) as f:
            expected_text = f.read()
        matched_strict = response_text == expected_text

        def answer_sig(text):
            norm = " ".join(text.lower().split())
            if expected_markers:
                idx = norm.find(expected_markers[-1])
                if idx >= 0:
                    return norm[: idx + len(expected_markers[-1])]
            return norm

        got_sig = answer_sig(response_text)
        exp_sig = answer_sig(expected_text)
        matched = got_sig == exp_sig
        if not matched:
            mismatch_desc = (
                f"response diverges from {a.must_match}: "
                f"expected {exp_sig[:120]!r}, got {got_sig[:120]!r}"
            )
    cache_hit_ratio = cached / pt if pt else 0
    cache_valid = cache_hit_ratio >= min_cache_hit_ratio

    out = {
        "ok": ct > 0 and marker_valid and cache_valid and matched is not False,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "completion_tokens": ct,
        "ttft_s": round(ttft, 3),
        "wall_s": round(wall, 3),
        "prefill_tps": round(computed / ttft, 1) if ttft > 0.01 and computed else None,
        "decode_tps": round((ct - 1) / (wall - ttft), 1)
        if ct > 1 and wall > ttft
        else None,
        "expected_markers": expected_markers,
        "response_markers": response_markers,
        "matched": matched,
        "matched_strict": matched_strict,
        "marker_valid": marker_valid,
        "cache_hit_ratio": round(cache_hit_ratio, 4),
        "min_cache_hit_ratio": min_cache_hit_ratio,
    }
    # TTFT includes fixed queue/tokenize overhead; the derived rate is
    # only meaningful once the prompt dwarfs that overhead.
    if computed < 256:
        out["prefill_tps"] = None
    if not marker_valid:
        if missing_markers:
            out["error"] = "markers missing from response: " + ", ".join(
                missing_markers
            )
        else:
            out["error"] = "markers appear out of prompt order"
    elif matched is False:
        out["error"] = mismatch_desc
    elif not cache_valid:
        out["error"] = (
            f"cache-hit ratio {cache_hit_ratio:.4f} is below required "
            f"{min_cache_hit_ratio:.4f}"
        )
    print(json.dumps(out))
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(out, f)

    if a.text_out:
        with open(a.text_out, "w") as f:
            rtxt = " ".join("".join(reasoning).split())
            ctxt = " ".join("".join(content).split())
            if rtxt:
                f.write("--- thinking ---\n" + rtxt[:500] + " ...\n\n")
            if ctxt:
                f.write("--- response ---\n" + ctxt[:1000] + " ...\n")
    if a.full_text_out:
        with open(a.full_text_out, "w") as f:
            f.write(response_text)

    if not out["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI must return failures as JSON
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)
