import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

SCRIPTS_DIR = pathlib.Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SPEC = importlib.util.spec_from_file_location(
    "bench_infernix", SCRIPTS_DIR / "bench_infernix.py")
BI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BI)

FSPEC = importlib.util.spec_from_file_location(
    "bench_infernix_frontier", SCRIPTS_DIR / "bench_infernix_frontier.py")
BF = importlib.util.module_from_spec(FSPEC)
FSPEC.loader.exec_module(BF)


def write_log(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return BI.LogParser(path).poll() or BI.LogParser(path)


def parse_done(lines):
    """LogParser.reqs for one `req#N done` line per entry."""
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        for line in lines:
            f.write(line + "\n")
        path = f.name
    parser = BI.LogParser(path)
    parser.poll()
    return parser


class AcceptanceParsingTest(unittest.TestCase):
    """The acceptance counters must follow whichever drafter the server ran,
    not a hard-coded name (a renamed or added backend must parse unchanged)."""

    def test_dflash2_line(self):
        p = parse_done([
            "2026-11-14 10:00:00.123  INFO  req#7 done | openai-chat | stop token"
            " | prompt 33,399 | output 578 | cache 8,040 (24.1%) | TTFT 3.5s"
            " | total 5.7s | prefill 10.8k tok/s | decode 266.2 tok/s"
            " | dflash2 accepted 299/456 (65.6%) | expert cache 100.0% hit",
        ])
        d = p.reqs[7]
        self.assertEqual(d["drafter"], "dflash2")
        self.assertEqual((d["acc_a"], d["acc_b"]), (299, 456))
        self.assertEqual(d["decode_tps"], 266.2)
        self.assertEqual(d["prefill_tps"], 10800.0)

    def test_other_drafter_names(self):
        for name in ("mtp", "dflash", "eagle", "mixed speculation"):
            p = parse_done([
                "2026-11-14 10:00:00.000  INFO  req#1 done | openai-chat |"
                " stop token | prompt 58 | output 64 | cache 0 (0.0%) |"
                " TTFT 30.6 ms | total 224 ms | decode 327.4 tok/s"
                " | {0} accepted 41/50 (82.0%)".format(name),
            ])
            d = p.reqs[1]
            self.assertEqual(d["drafter"], name)
            self.assertEqual((d["acc_a"], d["acc_b"]), (41, 50))

    def test_comma_separated_counts(self):
        p = parse_done([
            "2026-11-14 10:00:00.000  INFO  req#2 done | openai-chat |"
            " output limit | prompt 100 | output 8,000 | cache 0 (0.0%) |"
            " decode 300.0 tok/s | dflash2 accepted 1,234/5,678 (21.7%)",
        ])
        self.assertEqual((p.reqs[2]["acc_a"], p.reqs[2]["acc_b"]), (1234, 5678))

    def test_ngram_and_archive_segments_do_not_match(self):
        # "ngram a/b accepted, r rounds" and the draft archive report
        # "accepted, gen ..." - neither carries the parenthesised percentage,
        # so neither may be taken for the speculative acceptance counters.
        p = parse_done([
            "2026-11-14 10:00:00.000  INFO  req#3 done | openai-chat |"
            " stop token | prompt 1,000 | output 200 | cache 0 (0.0%) |"
            " decode 300.0 tok/s | ngram 12/20 accepted, 7 rounds |"
            " archive bound 30/40 accepted, gen 3, 5 sources",
        ])
        self.assertNotIn("acc_a", p.reqs[3])
        self.assertNotIn("drafter", p.reqs[3])
        # ... but a spec segment beside an ngram segment still parses.
        p = parse_done([
            "2026-11-14 10:00:00.000  INFO  req#4 done | openai-chat |"
            " stop token | prompt 1,000 | output 200 | cache 0 (0.0%) |"
            " decode 300.0 tok/s | mixed speculation accepted 90/120 (75.0%)"
            " | ngram 12/20 accepted, 7 rounds",
        ])
        self.assertEqual((p.reqs[4]["acc_a"], p.reqs[4]["acc_b"]), (90, 120))
        self.assertEqual(p.reqs[4]["drafter"], "mixed speculation")


class PercentileTest(unittest.TestCase):
    def test_linear_interpolation(self):
        vals = list(range(1, 11))
        self.assertAlmostEqual(BF.pct(vals, 0.5), 5.5)
        self.assertAlmostEqual(BF.pct(vals, 0.1), 1.9)
        self.assertAlmostEqual(BF.pct(vals, 0.9), 9.1)

    def test_edges(self):
        self.assertIsNone(BF.pct([], 0.5))
        self.assertEqual(BF.pct([7], 0.5), 7)
        self.assertEqual(BF.pct([3, 1, 2], 0.0), 1)
        self.assertEqual(BF.pct([3, 1, 2], 1.0), 3)


def point(c, med, total):
    return {"concurrency": c, "per_user_med_tok_s": med,
            "total_output_tok_s": total}


class FrontierTest(unittest.TestCase):
    def setUp(self):
        self.points = [point(1, 40.0, 40.0), point(2, 25.0, 70.0),
                       point(4, 12.0, 95.0), point(8, 5.0, 60.0)]

    def test_pareto_set_drops_dominated_points(self):
        front = BF.frontier(self.points)
        self.assertEqual([p["concurrency"] for p in front], [4, 2, 1])

    def test_frontier_keeps_equal_axes(self):
        # a tie on one axis with a strict gain on the other stays a frontier
        # point; an exact duplicate does not add a second one.
        pts = [point(2, 30.0, 70.0), point(4, 30.0, 90.0), point(6, 30.0, 70.0)]
        front = BF.frontier(pts)
        self.assertEqual([p["concurrency"] for p in front], [4])

    def test_points_without_metrics_are_skipped(self):
        pts = self.points + [point(3, None, None), point(5, 1.0, None)]
        front = BF.frontier(pts)
        self.assertEqual([p["concurrency"] for p in front], [4, 2, 1])

    def test_sla_pick_highest_total_above_the_sla(self):
        self.assertEqual(BF.sla_pick(self.points, 30.0)["concurrency"], 1)
        self.assertEqual(BF.sla_pick(self.points, 20.0)["concurrency"], 2)
        self.assertIsNone(BF.sla_pick(self.points, 50.0))


def done_event(seed, ts_ms, comp, decode_s, ttft_s, prompt=1000, cache=0,
               accepted=None, drafted=None):
    ev = {
        "event": "request_done",
        "timestamp_unix_ms": ts_ms,
        "request": {"sampling": {"seed": seed}},
        "result": {"prompt_tokens": prompt, "completion_tokens": comp,
                   "prefix_cache_hit_tokens": cache,
                   "computed_prefill_tokens": prompt - cache},
        "timings_seconds": {"ttft": ttft_s, "decode": decode_s, "total": 1.0},
        "speculative": {"accepted_tokens": accepted or 0,
                        "drafted_tokens": drafted or 0},
    }
    return ev


def start_event(seed, ts_ms):
    return {"event": "request_start", "timestamp_unix_ms": ts_ms,
            "request": {"sampling": {"seed": seed}}}


class LevelMetricsTest(unittest.TestCase):
    def test_metrics_from_request_log_events(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for ev in (start_event(1, 1_000),
                       done_event(1, 61_000, comp=301, decode_s=10.0,
                                  ttft_s=0.5, prompt=30_000, cache=0,
                                  accepted=200, drafted=250),
                       start_event(2, 2_000),
                       done_event(2, 22_000, comp=101, decode_s=4.0,
                                  ttft_s=0.25, prompt=31_000, cache=30_000,
                                  accepted=60, drafted=100),
                       # a different level's seed is not ours to count
                       start_event(3, 3_000),
                       done_event(3, 63_000, comp=1000, decode_s=1.0,
                                  ttft_s=9.0, accepted=1000, drafted=1000)):
                f.write(json.dumps(ev) + "\n")
            path = f.name
        done, starts, errors = BF.load_request_records(path, {1, 2})
        self.assertEqual(errors, 0)
        m = BF.level_metrics(done, starts, errors)
        self.assertEqual(m["requests_done"], 2)
        self.assertEqual(m["span_s"], 60.0)
        self.assertEqual(m["output_tokens"], 402)
        self.assertEqual(m["total_output_tok_s"], round(402 / 60.0, 1))
        # per-user decode tok/s: (301-1)/10 = 30.0 and (101-1)/4 = 25.0
        self.assertEqual(m["per_user_med_tok_s"], 27.5)
        self.assertEqual(m["per_user_p10_tok_s"], 25.5)
        self.assertEqual(m["ttft_p50_s"], 0.375)
        self.assertEqual(m["ttft_p90_s"], 0.475)
        self.assertEqual(m["accept_pct"], round(100.0 * 260 / 350, 1))
        self.assertEqual(m["cache_hit_pct"], round(100.0 * 30_000 / 61_000, 1))

    def test_no_completed_requests(self):
        self.assertIsNone(BF.level_metrics({}, {}, 0))


if __name__ == "__main__":
    unittest.main()
