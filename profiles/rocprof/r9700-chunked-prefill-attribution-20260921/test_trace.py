import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("chunk_trace", Path(__file__).with_name("trace.py"))
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class TraceTest(unittest.TestCase):
    def test_command_selects_one_cold_ordinary_eight_chunk_prefill(self):
        benchmark, profiler = trace.commands()
        for flag, value in (("--concurrency", "1"), ("-p", "8192"),
                            ("--prefill-chunk", "1024"), ("--draft-tokens", "0"),
                            ("-r", "1"), ("--warmup", "0")):
            self.assertEqual(benchmark[benchmark.index(flag) + 1], value)
        self.assertEqual(profiler[profiler.index("--") + 1:], benchmark)
        for flag in ("--selected-regions", "--marker-trace", "--kernel-trace", "--memory-copy-trace"):
            self.assertIn(flag, profiler)
        self.assertEqual(benchmark.count("--profile-measured"), 1)

    def test_existing_output_rejects_before_checks_or_subprocess(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            trace, "OUTPUT", Path(directory)
        ), patch.object(trace.subprocess, "run", side_effect=AssertionError("must not launch")):
            with self.assertRaisesRegex(ValueError, "already exists"):
                trace.preflight()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_busy_device_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            trace, "OUTPUT", Path(directory) / "run"
        ), patch.object(trace, "idle", side_effect=ValueError("busy")), patch.object(
            trace.subprocess, "run", side_effect=AssertionError("must not launch")
        ):
            with self.assertRaisesRegex(ValueError, "busy"):
                trace.execute({})
            self.assertFalse(trace.OUTPUT.exists())

    def test_changed_frozen_binary_rejects_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bench"
            binary.write_bytes(b"changed")
            frozen = root / "inputs.json"
            frozen.write_text(json.dumps({"artifact_type": "ninfer_r9700_chunk_campaign_inputs",
                              "schema_version": 1, "files": {"bench": "old"}}))
            with patch.object(trace, "REPO", root), patch.object(trace, "BENCH", binary), patch.object(
                trace, "FROZEN", frozen
            ), patch.object(trace, "OUTPUT", root / "run"), patch.object(
                trace.subprocess, "run", side_effect=AssertionError("must not launch")
            ):
                with self.assertRaisesRegex(ValueError, "differs from frozen"):
                    trace.preflight()
                self.assertFalse(trace.OUTPUT.exists())

    def test_closure_does_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "closure.json").write_text("owned")
            with patch.object(trace, "OUTPUT", root):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    trace.close_trace()
            self.assertEqual((root / "closure.json").read_text(), "owned")


if __name__ == "__main__":
    unittest.main()
