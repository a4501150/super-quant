#!/usr/bin/env python3
"""Focused intelligence benchmark: coding + math + general.

Sends problems to an OpenAI-compatible API, auto-grades responses.
Usage:
    python src/eval_problems.py --port 8888 --model qwen3.8-27b --label aeon
    python src/eval_problems.py --port 8888 --model qwen3.8-27b --label official
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    id: str
    category: str  # coding | math | general
    prompt: str
    test_code: str = ""       # for coding: pytest code
    expected: str = ""        # for math/general: expected answer
    grader: str = "exact"     # exact | contains | code | format
    aliases: list[str] = field(default_factory=list)


PROBLEMS = [
    # ── Coding (8) ────────────────────────────────────────────────────
    Problem(
        id="code_01_two_sum",
        category="coding",
        prompt=(
            "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` "
            "that returns the indices of the two numbers that add up to `target`. "
            "Each input has exactly one solution. Do not use the same element twice. "
            "Return the answer as a list of two indices in any order."
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            assert sorted(two_sum([2,7,11,15], 9)) == [0,1]
            assert sorted(two_sum([3,2,4], 6)) == [1,2]
            assert sorted(two_sum([3,3], 6)) == [0,1]
            assert sorted(two_sum([1,5,3,7], 8)) == [1,2]
            assert sorted(two_sum([-1,0,1,2], 1)) == [0,3]
        """),
    ),
    Problem(
        id="code_02_longest_palindrome",
        category="coding",
        prompt=(
            "Write a Python function `longest_palindrome(s: str) -> str` "
            "that returns the longest palindromic substring of `s`. "
            "If there are multiple with the same length, return any one."
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            r = longest_palindrome("babad")
            assert r in ("bab", "aba"), f"got {r!r}"
            assert longest_palindrome("cbbd") == "bb"
            assert longest_palindrome("a") == "a"
            assert longest_palindrome("racecar") == "racecar"
            assert len(longest_palindrome("abcddcbaXY")) >= 8
        """),
    ),
    Problem(
        id="code_03_merge_k_sorted",
        category="coding",
        prompt=(
            "Write a Python function `merge_k_sorted(lists: list[list[int]]) -> list[int]` "
            "that merges k sorted lists into one sorted list. "
            "Each inner list is sorted in ascending order."
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            assert merge_k_sorted([[1,4,5],[1,3,4],[2,6]]) == [1,1,2,3,4,4,5,6]
            assert merge_k_sorted([]) == []
            assert merge_k_sorted([[]]) == []
            assert merge_k_sorted([[1],[2],[3]]) == [1,2,3]
            assert merge_k_sorted([[5,10,15],[1,2,3],[7,8]]) == [1,2,3,5,7,8,10,15]
        """),
    ),
    Problem(
        id="code_04_lru_cache",
        category="coding",
        prompt=(
            "Implement a Python class `LRUCache` with:\n"
            "- `__init__(self, capacity: int)` — initialize with positive capacity\n"
            "- `get(self, key: int) -> int` — return value if key exists, else -1\n"
            "- `put(self, key: int, value: int) -> None` — update or insert. "
            "If capacity exceeded, evict the least recently used key.\n"
            "Both operations must run in O(1) average time."
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            c = LRUCache(2)
            c.put(1, 1)
            c.put(2, 2)
            assert c.get(1) == 1
            c.put(3, 3)
            assert c.get(2) == -1
            c.put(4, 4)
            assert c.get(1) == -1
            assert c.get(3) == 3
            assert c.get(4) == 4
        """),
    ),
    Problem(
        id="code_05_word_break",
        category="coding",
        prompt=(
            "Write a Python function `word_break(s: str, word_dict: list[str]) -> bool` "
            "that returns True if `s` can be segmented into one or more words from `word_dict`. "
            "Words from the dictionary can be reused."
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            assert word_break("leetcode", ["leet","code"]) == True
            assert word_break("applepenapple", ["apple","pen"]) == True
            assert word_break("catsandog", ["cats","dog","sand","and","cat"]) == False
            assert word_break("aaaaaaa", ["a","aa","aaa"]) == True
            assert word_break("", ["a"]) == True
        """),
    ),
    Problem(
        id="code_06_serialize_tree",
        category="coding",
        prompt=(
            "Implement Python functions to serialize and deserialize a binary tree.\n\n"
            "Define a TreeNode class: `class TreeNode:\\n    def __init__(self, val=0, left=None, right=None):\\n        self.val = val\\n        self.left = left\\n        self.right = right`\n\n"
            "Write:\n"
            "- `serialize(root: TreeNode | None) -> str`\n"
            "- `deserialize(data: str) -> TreeNode | None`\n\n"
            "The output of serialize fed into deserialize must reconstruct the original tree."
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            def trees_equal(a, b):
                if a is None and b is None: return True
                if a is None or b is None: return False
                return a.val == b.val and trees_equal(a.left, b.left) and trees_equal(a.right, b.right)

            t1 = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
            assert trees_equal(deserialize(serialize(t1)), t1)
            assert deserialize(serialize(None)) is None
            t2 = TreeNode(1, None, TreeNode(2, None, TreeNode(3)))
            assert trees_equal(deserialize(serialize(t2)), t2)
        """),
    ),
    Problem(
        id="code_07_min_window",
        category="coding",
        prompt=(
            "Write a Python function `min_window(s: str, t: str) -> str` "
            "that returns the minimum window substring of `s` that contains "
            "all characters in `t` (including duplicates). "
            "Return empty string if no such window exists."
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            assert min_window("ADOBECODEBANC", "ABC") == "BANC"
            assert min_window("a", "a") == "a"
            assert min_window("a", "aa") == ""
            r = min_window("aaabbbccc", "abc")
            assert set("abc").issubset(set(r)) and len(r) <= 5
        """),
    ),
    Problem(
        id="code_08_nested_iterator",
        category="coding",
        prompt=(
            "Write a Python class `NestedIterator` that flattens a nested list of integers.\n\n"
            "The nested list is represented as a Python list where each element is either "
            "an integer or another nested list. Example: `[1, [2, [3]], 4]`\n\n"
            "Implement:\n"
            "- `__init__(self, nested_list)` — initialize with the nested list\n"
            "- `next(self) -> int` — return the next integer\n"
            "- `has_next(self) -> bool` — return True if there are more integers"
        ),
        grader="code",
        test_code=textwrap.dedent("""\
            it = NestedIterator([[1,1],2,[1,1]])
            result = []
            while it.has_next():
                result.append(it.next())
            assert result == [1,1,2,1,1]

            it2 = NestedIterator([1,[4,[6]]])
            result2 = []
            while it2.has_next():
                result2.append(it2.next())
            assert result2 == [1,4,6]

            it3 = NestedIterator([])
            assert not it3.has_next()

            it4 = NestedIterator([[],[1],[]])
            result4 = []
            while it4.has_next():
                result4.append(it4.next())
            assert result4 == [1]
        """),
    ),

    # ── Math (7) ──────────────────────────────────────────────────────
    Problem(
        id="math_01_mississippi",
        category="math",
        prompt=(
            "How many distinct ways can you arrange the letters of MISSISSIPPI? "
            "Give the final numerical answer only."
        ),
        grader="exact",
        expected="34650",
    ),
    Problem(
        id="math_02_power_mod",
        category="math",
        prompt=(
            "Find the remainder when 2^100 is divided by 7. "
            "Give the final numerical answer only."
        ),
        grader="exact",
        expected="2",
    ),
    Problem(
        id="math_03_triangle_area",
        category="math",
        prompt=(
            "Find the area of the triangle with vertices at (0,0), (4,0), (0,3). "
            "Give the final numerical answer only."
        ),
        grader="exact",
        expected="6",
    ),
    Problem(
        id="math_04_probability",
        category="math",
        prompt=(
            "5 cards are drawn from a standard 52-card deck without replacement. "
            "What is the probability of getting exactly 2 aces? "
            "Give the answer as a simplified fraction."
        ),
        grader="exact",
        expected="2162/54145",
        aliases=["0.03993", "0.0399", "2162/54145"],
    ),
    Problem(
        id="math_05_cubic",
        category="math",
        prompt=(
            "Solve the equation x^3 - 6x^2 + 11x - 6 = 0. "
            "List all solutions separated by commas."
        ),
        grader="contains",
        expected="1,2,3",
    ),
    Problem(
        id="math_06_series",
        category="math",
        prompt=(
            "Compute the sum 1/1! + 1/2! + 1/3! + ... + 1/10!. "
            "Give the answer as a decimal rounded to 6 decimal places."
        ),
        grader="exact",
        expected="1.718282",
        aliases=["1.71828", "1.718282"],
    ),
    Problem(
        id="math_07_last_two_digits",
        category="math",
        prompt=(
            "What are the last two digits of 7^2026? "
            "Give the final two-digit numerical answer only."
        ),
        grader="exact",
        expected="49",
    ),

    # ── General (5) ───────────────────────────────────────────────────
    Problem(
        id="gen_01_languages",
        category="general",
        prompt=(
            "List exactly 5 programming languages created before 1980. "
            "One per line, no numbering, no extra text."
        ),
        grader="format",
        expected="5_lines",
    ),
    Problem(
        id="gen_02_chemistry",
        category="general",
        prompt="What is the chemical formula for sulfuric acid? Give only the formula.",
        grader="exact",
        expected="H2SO4",
        aliases=["h2so4", "H₂SO₄"],
    ),
    Problem(
        id="gen_03_translate",
        category="general",
        prompt="Translate 'The cat sat on the mat' into French. Give only the translation.",
        grader="contains",
        expected="Le chat",
    ),
    Problem(
        id="gen_04_haiku",
        category="general",
        prompt=(
            "Write a haiku about recursion. "
            "It must be exactly 3 lines."
        ),
        grader="format",
        expected="3_lines",
    ),
    Problem(
        id="gen_05_tcp_udp",
        category="general",
        prompt=(
            "Explain the difference between TCP and UDP in exactly 2 sentences."
        ),
        grader="format",
        expected="2_sentences",
    ),
]


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def query_model(prompt: str, port: int, model: str) -> dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "reasoning_effort": "medium",
    }
    start = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]["message"]
    return {
        "content": choice.get("content", ""),
        "reasoning": choice.get("reasoning_content", ""),
        "tokens": data.get("usage", {}).get("completion_tokens", 0),
        "elapsed": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Code extraction + execution
# ---------------------------------------------------------------------------

def extract_code(content: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
    if blocks:
        impl_blocks = [b for b in blocks if re.search(r'^(?:def |class )', b, re.MULTILINE)]
        code = "\n\n".join(impl_blocks) if impl_blocks else blocks[0]
    else:
        code = content
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith((">>>", "...")):
            continue
        if stripped.startswith("print("):
            continue
        try:
            line.encode("ascii")
        except UnicodeEncodeError:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def run_code_test(code: str, test_code: str) -> tuple[bool, str]:
    full = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr[-500:] if result.stderr else result.stdout[-500:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        finally:
            Path(f.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def extract_number(text: str) -> str:
    text = text.strip().rstrip(".")
    last_lines = "\n".join(text.split("\n")[-3:])
    latex_fracs = re.findall(r'\\d?frac\{(-?\d+)\}\{(\d+)\}', text)
    if latex_fracs:
        n, d = latex_fracs[-1]
        return f"{n}/{d}"
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    answer_pat = re.findall(r'(?:answer|result|=)\s*[:is]*\s*\**\s*(-?[\d./]+)', last_lines, re.IGNORECASE)
    if answer_pat:
        return answer_pat[-1]
    fractions = re.findall(r'-?\d+/\d+', last_lines)
    if fractions:
        return fractions[-1]
    nums = re.findall(r'-?\d+(?:\.\d+)?', last_lines)
    if nums:
        return nums[-1]
    nums_all = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums_all:
        return nums_all[-1]
    return text.strip()


def grade_problem(problem: Problem, content: str) -> tuple[bool, str]:
    if problem.grader == "code":
        code = extract_code(content)
        if not code.strip():
            return False, "no code extracted"
        passed, err = run_code_test(code, problem.test_code)
        return passed, err

    if problem.grader == "exact":
        answer = extract_number(content)
        targets = [problem.expected] + problem.aliases
        for t in targets:
            if answer == t:
                return True, ""
            t_clean = t.replace(" ", "").lower()
            a_clean = answer.replace(" ", "").lower()
            if a_clean == t_clean:
                return True, ""
            try:
                if abs(float(a_clean) - float(t_clean)) < 0.001:
                    return True, ""
            except ValueError:
                pass
        return False, f"got {answer!r}, expected {problem.expected!r}"

    if problem.grader == "contains":
        targets = problem.expected.split(",")
        text_lower = content.lower()
        for t in targets:
            if t.strip().lower() not in text_lower:
                return False, f"missing {t.strip()!r}"
        return True, ""

    if problem.grader == "format":
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        if problem.expected == "5_lines":
            if len(lines) == 5:
                return True, ""
            return False, f"expected 5 lines, got {len(lines)}"
        if problem.expected == "3_lines":
            if len(lines) == 3:
                return True, ""
            return False, f"expected 3 lines, got {len(lines)}"
        if problem.expected == "2_sentences":
            sentences = re.split(r'[.!?]+', content.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) == 2:
                return True, ""
            return False, f"expected 2 sentences, got {len(sentences)}"
        return False, f"unknown format: {problem.expected}"

    return False, f"unknown grader: {problem.grader}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(port: int, model: str, label: str, output_dir: str) -> dict:
    results = []
    scores = {"coding": [0, 0], "math": [0, 0], "general": [0, 0]}

    for p in PROBLEMS:
        print(f"  [{p.category}] {p.id}...", end=" ", flush=True)
        try:
            resp = query_model(p.prompt, port, model)
            content = resp["content"]
            passed, detail = grade_problem(p, content)
        except Exception as e:  # noqa: BLE001
            content = ""
            resp = {"content": "", "reasoning": "", "tokens": 0, "elapsed": 0}
            passed = False
            detail = str(e)

        scores[p.category][1] += 1
        if passed:
            scores[p.category][0] += 1

        status = "PASS" if passed else "FAIL"
        print(f"{status} ({resp['elapsed']}s, {resp['tokens']} tok)"
              + (f" — {detail}" if detail and not passed else ""))

        results.append({
            "id": p.id,
            "category": p.category,
            "passed": passed,
            "detail": detail,
            "content": content[:2000],
            "reasoning_len": len(resp.get("reasoning", "")),
            "tokens": resp["tokens"],
            "elapsed": resp["elapsed"],
        })

    total_pass = sum(v[0] for v in scores.values())
    total_count = sum(v[1] for v in scores.values())

    summary = {
        "label": label,
        "model": model,
        "total": f"{total_pass}/{total_count}",
        "coding": f"{scores['coding'][0]}/{scores['coding'][1]}",
        "math": f"{scores['math'][0]}/{scores['math'][1]}",
        "general": f"{scores['general'][0]}/{scores['general'][1]}",
        "problems": results,
    }

    print(f"\n  === {label} ===")
    print(f"  Coding:  {scores['coding'][0]}/{scores['coding'][1]}")
    print(f"  Math:    {scores['math'][0]}/{scores['math'][1]}")
    print(f"  General: {scores['general'][0]}/{scores['general'][1]}")
    print(f"  Total:   {total_pass}/{total_count}")

    out_path = Path(output_dir) / f"eval_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Focused intelligence benchmark")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--label", required=True, help="aeon or official")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    print(f"=== Eval: {args.label} (model={args.model}, port={args.port}) ===\n")
    run_eval(args.port, args.model, args.label, args.output_dir)


if __name__ == "__main__":
    main()
