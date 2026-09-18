import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

SCRIPTS_DIR = pathlib.Path(__file__).parents[1] / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "hicache_digest_check.py"
BENCH_SCRIPT_PATH = SCRIPTS_DIR / "11_bench_sglang.sh"
SPEC = importlib.util.spec_from_file_location("hicache_digest_check", SCRIPT_PATH)
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)

SHA_A = "a" * 64
SHA_B = "b" * 64


def page_line(direction, key, sha, pool="kv", component="kv"):
    return (
        "2026-09-18 00:00:00 WARNING hicache_storage: HiCache page digest "
        f"direction={direction} pool={pool} component={component} key={key} "
        f"host_page=7 bytes=8192 sha256={sha}\n"
    )


def transfer_line(direction, component, host_sha, device_sha, exact,
                  device_field="device_index", kv=True):
    prefix = "HiCache KV transfer digest" if kv else "HiCache transfer digest"
    return (
        "2026-09-18 00:00:00 WARNING pool_host: "
        f"{prefix} direction={direction} component={component} host_page=7 "
        f"{device_field}=11 bytes=4096 host_sha256={host_sha} "
        f"device_sha256={device_sha} exact={exact}\n"
    )


def omit_metrics(expected, reported, error=None):
    return {
        "ok": False,
        "expected_markers": expected,
        "response_markers": reported,
        "error": error if error is not None else "markers missing from response: " + ", ".join(
            m for m in expected if m not in reported),
    }


GOOD_REPLAY = {"ok": True, "matched": True,
               "expected_markers": ["11111111", "22222222", "33333333"],
               "response_markers": ["11111111", "22222222", "33333333"]}
EXACT_DIGESTS = {"ok": True}


class DigestCheckTest(unittest.TestCase):
    def run_check(self, cold_lines, restore_lines, *, add_required_transfers=True):
        if add_required_transfers:
            cold_lines = list(cold_lines) + [
                transfer_line("device_to_host", component, SHA_A, SHA_A, "True")
                for component in sorted(DIAG.REQUIRED_KV_TRANSFER_COMPONENTS)
            ]
        with tempfile.TemporaryDirectory() as tmp:
            cold = pathlib.Path(tmp) / "cold.log"
            restore = pathlib.Path(tmp) / "restore.log"
            cold.write_text("".join(cold_lines))
            restore.write_text("".join(restore_lines))
            return DIAG.digest_check(str(cold), str(restore))

    def test_paired_storage_digests_pass_even_with_unpaired_template_reads(self):
        out = self.run_check(
            [page_line("write", "k1", SHA_A)],
            [page_line("read", "k1", SHA_A), page_line("read", "legacy", SHA_B)],
        )
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["paired_keys"], 1)
        self.assertEqual(out["unpaired_read_keys"], 1)

    def test_restore_read_mismatching_cold_write_fails(self):
        out = self.run_check(
            [page_line("write", "k1", SHA_A)],
            [page_line("read", "k1", SHA_B)],
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["mismatch_count"], 1)
        self.assertIn("mismatched=1", out["error"])

    def test_conflicting_repeated_cold_writes_fail(self):
        out = self.run_check(
            [page_line("write", "k1", SHA_A), page_line("write", "k1", SHA_B)],
            [page_line("read", "k1", SHA_A)],
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["write_conflict_count"], 1)

    def test_exact_kv_transfer_digests_parse_including_kv_scale_components(self):
        out = self.run_check(
            [
                page_line("write", "k1", SHA_A),
                transfer_line("write", "kv_k", SHA_A, SHA_A, "True"),
                transfer_line("write", "kv_v", SHA_A, SHA_A, "True"),
                transfer_line("write", "kv_scale_k", SHA_A, SHA_A, "True"),
                transfer_line("write", "kv_scale_v", SHA_A, SHA_A, "True"),
            ],
            [
                page_line("read", "k1", SHA_A),
                transfer_line("read", "kv_scale_k", SHA_A, SHA_A, "True"),
                transfer_line("read", "kv_scale_v", SHA_A, SHA_A, "True"),
            ],
        )
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["transfer_records"], 10)
        self.assertEqual(out["transfer_exact_false"], 0)
        self.assertEqual(
            out["transfer_components"],
            {"kv_k": 2, "kv_v": 2, "kv_scale_k": 3, "kv_scale_v": 3},
        )

    def test_missing_dynamic_scale_transfer_digest_is_a_hard_failure(self):
        out = self.run_check(
            [
                page_line("write", "k1", SHA_A),
                transfer_line("device_to_host", "kv_k", SHA_A, SHA_A, "True"),
                transfer_line("device_to_host", "kv_v", SHA_A, SHA_A, "True"),
            ],
            [page_line("read", "k1", SHA_A)],
            add_required_transfers=False,
        )
        self.assertFalse(out["ok"])
        self.assertEqual(
            out["missing_transfer_components"], ["kv_scale_k", "kv_scale_v"]
        )

    def test_exact_false_kv_scale_transfer_digest_is_a_hard_failure(self):
        out = self.run_check(
            [page_line("write", "k1", SHA_A),
             transfer_line("write", "kv_scale_v", SHA_A, SHA_B, "False")],
            [page_line("read", "k1", SHA_A)],
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["transfer_exact_false"], 1)
        self.assertIn("transfer_exact_false=1", out["error"])
        self.assertEqual(out["transfer_failures"][0]["component"], "kv_scale_v")

    def test_mamba_pool_transfer_digest_line_parses(self):
        out = self.run_check(
            [page_line("write", "k1", SHA_A),
             transfer_line("read", "recurrent_state", SHA_A, SHA_B, "False",
                           device_field="device_slot", kv=False)],
            [page_line("read", "k1", SHA_A)],
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["transfer_exact_false"], 1)
        self.assertEqual(out["transfer_failures"][0]["component"], "recurrent_state")

    def test_exact_flag_contradicting_digests_fails(self):
        out = self.run_check(
            [page_line("write", "k1", SHA_A),
             transfer_line("write", "kv_k", SHA_A, SHA_B, "True")],
            [page_line("read", "k1", SHA_A)],
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["transfer_flag_conflicts"], 1)


class VarianceClassificationTest(unittest.TestCase):
    def test_pure_omission_replays_and_classifies_as_variance(self):
        metrics = omit_metrics(["11111111", "22222222", "33333333"],
                               ["11111111", "22222222"])
        self.assertTrue(DIAG.needs_replay(metrics))
        variance = DIAG.classify_generation_variance(metrics, GOOD_REPLAY, EXACT_DIGESTS)
        self.assertIsNotNone(variance)
        self.assertEqual(variance["classification"], "generation_variance")
        self.assertEqual(variance["missing_markers"], ["33333333"])

    def test_sequence_mismatch_error_with_strict_omission_classifies(self):
        expected = ["11111111", "22222222", "33333333"]
        reported = ["22222222", "33333333"]
        metrics = omit_metrics(
            expected,
            reported,
            error=f"response marker sequence mismatch: expected {expected}, got {reported}",
        )
        self.assertTrue(DIAG.needs_replay(metrics))
        self.assertIsNotNone(
            DIAG.classify_generation_variance(metrics, GOOD_REPLAY, EXACT_DIGESTS)
        )

    def test_interior_omission_still_replays_and_classifies(self):
        metrics = omit_metrics(["11111111", "22222222", "33333333"],
                               ["11111111", "33333333"])
        variance = DIAG.classify_generation_variance(metrics, GOOD_REPLAY, EXACT_DIGESTS)
        self.assertIsNotNone(variance)
        self.assertEqual(variance["missing_markers"], ["22222222"])

    def test_foreign_marker_is_not_a_replay_candidate(self):
        metrics = omit_metrics(["11111111", "22222222", "33333333"],
                               ["11111111", "deadbeef", "22222222"])
        self.assertFalse(DIAG.needs_replay(metrics))
        self.assertIsNone(
            DIAG.classify_generation_variance(metrics, GOOD_REPLAY, EXACT_DIGESTS))

    def test_reordered_markers_are_not_a_replay_candidate(self):
        metrics = {
            "ok": False,
            "expected_markers": ["11111111", "22222222", "33333333"],
            "response_markers": ["22222222", "11111111", "33333333"],
            "error": "markers appear out of prompt order",
        }
        self.assertFalse(DIAG.needs_replay(metrics))
        self.assertIsNone(
            DIAG.classify_generation_variance(metrics, GOOD_REPLAY, EXACT_DIGESTS))

    def test_digest_conflict_or_exact_false_blocks_variance(self):
        metrics = omit_metrics(["11111111", "22222222", "33333333"], ["11111111"])
        for digests in (None, {"ok": False}, {}):
            self.assertIsNone(
                DIAG.classify_generation_variance(metrics, GOOD_REPLAY, digests))

    def test_repeated_omission_on_replay_remains_failure(self):
        metrics = omit_metrics(["11111111", "22222222", "33333333"], ["11111111"])
        failed_replay = omit_metrics(["11111111", "22222222", "33333333"],
                                     ["11111111", "22222222"])
        self.assertIsNone(
            DIAG.classify_generation_variance(metrics, failed_replay, EXACT_DIGESTS))
        self.assertIsNone(
            DIAG.classify_generation_variance(metrics, None, EXACT_DIGESTS))

    def test_passing_run_never_classifies_as_variance(self):
        metrics = {"ok": True, "expected_markers": ["11111111"],
                   "response_markers": ["11111111"]}
        self.assertIsNone(
            DIAG.classify_generation_variance(metrics, GOOD_REPLAY, EXACT_DIGESTS))


class HarnessScriptTest(unittest.TestCase):
    def test_standalone_revisit_is_in_same_loop_as_cold_probe(self):
        script = BENCH_SCRIPT_PATH.read_text()
        section = script[
            script.index("local -A PF=") : script.index("local -a CC_PF=()")
        ]
        self.assertEqual(section.count("for target in ${CORRECT_TOKENS}; do"), 1)
        self.assertLess(
            section.index('correct_cold_${target}.json'),
            section.index('correct_revisit_${target}.json'),
        )


class CliTest(unittest.TestCase):
    def test_check_command_writes_json_and_reports_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            cold = pathlib.Path(tmp) / "cold.log"
            restore = pathlib.Path(tmp) / "restore.log"
            out = pathlib.Path(tmp) / "check.json"
            cold.write_text(page_line("write", "k1", SHA_A))
            restore.write_text(page_line("read", "k1", SHA_B))
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = DIAG.main(["check", str(cold), str(restore), str(out)])
            self.assertEqual(code, 1)
            data = json.loads(out.read_text())
            self.assertFalse(data["ok"])
            self.assertIn("digest check failed", stderr.getvalue())

    def test_needs_replay_command_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "candidate.json"
            candidate.write_text(json.dumps(
                omit_metrics(["11111111", "22222222"], ["11111111"])))
            other = pathlib.Path(tmp) / "other.json"
            other.write_text(json.dumps({"ok": False, "error": "connection refused"}))
            missing = pathlib.Path(tmp) / "absent.json"
            self.assertEqual(DIAG.main(["needs-replay", str(candidate)]), 0)
            self.assertEqual(DIAG.main(["needs-replay", str(other)]), 1)
            self.assertEqual(DIAG.main(["needs-replay", str(missing)]), 1)


if __name__ == "__main__":
    unittest.main()
