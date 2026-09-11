import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "bench_sglang_request.py"
SPEC = importlib.util.spec_from_file_location("bench_sglang_request", SCRIPT_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class MarkerSequenceTest(unittest.TestCase):
    def test_system_markers_serialize_before_prompt_markers(self):
        system = "Agent Unique marker AAAAAAAA. Revision Unique marker bbbbbbbb."
        prompt = "Body Unique marker CcCcCcCc."

        self.assertEqual(
            BENCH.expected_marker_sequence(system, prompt),
            ["aaaaaaaa", "bbbbbbbb", "cccccccc"],
        )

    def test_repeated_generated_markers_do_not_change_expected_prefix(self):
        expected = ["11111111", "22222222", "33333333"]
        response = "11111111 22222222 33333333 11111111 22222222"

        self.assertEqual(
            BENCH.response_marker_sequence(response, len(expected)),
            expected,
        )

    def test_missing_final_marker_remains_a_failure(self):
        expected = ["11111111", "22222222", "33333333"]
        response = "11111111 22222222"

        self.assertNotEqual(
            BENCH.response_marker_sequence(response, len(expected)),
            expected,
        )

    def test_foreign_or_reordered_marker_remains_a_failure(self):
        expected = ["11111111", "22222222", "33333333"]
        response = "11111111 deadbeef 33333333 22222222"

        self.assertNotEqual(
            BENCH.response_marker_sequence(response, len(expected)),
            expected,
        )

    def test_hex_substrings_are_not_markers(self):
        response = "x11111111 222222222 33333333z 44444444"

        self.assertEqual(BENCH.response_marker_sequence(response, 4), ["44444444"])

    def test_expected_markers_require_standalone_eight_hex(self):
        prompt = (
            "Unique marker 1111111. Unique marker 222222222. "
            "Unique marker deadbeefz. Unique marker A1B2C3D4."
        )

        self.assertEqual(BENCH.expected_marker_sequence(None, prompt), ["a1b2c3d4"])

    def test_system_file_and_marker_mode_build_messages(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(
                    [
                        b'data: {"choices":[{"delta":{"content":"aaaaaaaa cccccccc"}}]}\n',
                        b'data: {"usage":{"prompt_tokens":2,"completion_tokens":2},"choices":[]}\n',
                        b"data: [DONE]\n",
                    ]
                )

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return Response()

        with tempfile.TemporaryDirectory() as tmpdir:
            system_path = pathlib.Path(tmpdir) / "system.txt"
            system_path.write_text("Agent Unique marker AAAAAAAA.")
            argv = [
                str(SCRIPT_PATH),
                "--url",
                "http://127.0.0.1:8000/v1/chat/completions",
                "--model",
                "test-model",
                "--system-file",
                str(system_path),
                "--prompt",
                "Body Unique marker CCCCCCCC.",
                "--must-match-markers",
            ]
            stdout = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(BENCH, "urlopen", side_effect=fake_urlopen),
                redirect_stdout(stdout),
            ):
                BENCH.main()

        self.assertEqual(captured["timeout"], 1800)
        self.assertEqual(
            captured["body"]["messages"],
            [
                {"role": "system", "content": "Agent Unique marker AAAAAAAA."},
                {"role": "user", "content": "Body Unique marker CCCCCCCC."},
            ],
        )
        result = json.loads(stdout.getvalue())
        self.assertTrue(result["ok"])
        self.assertTrue(result["matched"])

    def test_system_inputs_are_mutually_exclusive(self):
        argv = [
            str(SCRIPT_PATH),
            "--url",
            "http://127.0.0.1:8000/v1/chat/completions",
            "--model",
            "test-model",
            "--prompt",
            "test",
            "--system",
            "inline",
            "--system-file",
            "system.txt",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            BENCH.main()

        self.assertEqual(raised.exception.code, 2)

    def test_marker_mode_requires_an_expected_marker(self):
        argv = [
            str(SCRIPT_PATH),
            "--url",
            "http://127.0.0.1:8000/v1/chat/completions",
            "--model",
            "test-model",
            "--prompt",
            "no marker",
            "--must-match-markers",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            BENCH.main()

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
