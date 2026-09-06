#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402


def cell(tokens: int, incumbent: list[float] | None = None,
         challenger: list[float] | None = None) -> dict:
    incumbent = incumbent or [1.0 + index * 0.001 for index in range(14)]
    challenger = challenger or [0.80 + index * 0.0008 for index in range(14)]
    incumbent_median = statistics.median(incumbent)
    challenger_median = statistics.median(challenger)
    return {
        "schema": "ninfer.r9700.fp8_target_gate_up_small_t_cell.v1",
        "production_routing_changed": False,
        "tokens": tokens,
        "rows": 34816,
        "columns": 5120,
        "timing_scope": "complete_bf16_quantize_matmul_status_poison",
        "incumbent_samples_ms": incumbent,
        "challenger_samples_ms": challenger,
        "incumbent_median_ms": incumbent_median,
        "challenger_median_ms": challenger_median,
        "challenger_over_incumbent": challenger_median / incumbent_median,
        "projected_64_call_saving_ms": 64.0 * (incumbent_median - challenger_median),
        "decision": "unmade_until_gpu_measurement_is_validated",
    }


class AnalyzerTest(unittest.TestCase):
    def test_accepts_conservative_balanced_cell(self) -> None:
        result = analyze.analyze_cell(cell(5))
        self.assertTrue(result["decision"]["accepted"])
        self.assertEqual(len(result["derived"]["forward_candidate_over_incumbent"]), 7)
        self.assertEqual(len(result["derived"]["reverse_candidate_over_incumbent"]), 7)

    def test_rejects_slow_cell_without_rejecting_valid_report(self) -> None:
        result = analyze.analyze_cell(cell(6, challenger=[1.05] * 14))
        self.assertFalse(result["decision"]["accepted"])
        self.assertLess(result["derived"]["projected_64_call_saving_ms"], 0.0)

    def test_rejects_launch_order_instability(self) -> None:
        challenger = [0.80 if index % 2 == 0 else 0.99 for index in range(14)]
        report = cell(5, incumbent=[1.0] * 14, challenger=challenger)
        result = analyze.analyze_cell(report)
        self.assertFalse(result["decision"]["accepted"])
        self.assertGreater(result["derived"]["order_ratio_delta"], 0.03)

    def test_rejects_mutated_raw_sample(self) -> None:
        report = cell(5)
        report["challenger_samples_ms"][0] = 0.95
        with self.assertRaisesRegex(ValueError, "differs from raw samples"):
            analyze.analyze_cell(report)

    def test_rejects_nonfinite_raw_sample(self) -> None:
        report = cell(5)
        report["challenger_samples_ms"][3] = math.nan
        with self.assertRaisesRegex(ValueError, "not finite"):
            analyze.analyze_cell(report)

    def test_rejects_duplicate_json_key(self) -> None:
        raw = json.dumps(cell(5)) + "\n" + json.dumps(cell(6))
        raw = raw.replace('"tokens": 5', '"tokens": 5, "tokens": 5', 1)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            analyze.parse_raw(raw)

    def test_rejects_wrong_cell_order(self) -> None:
        raw = json.dumps(cell(6)) + "\n" + json.dumps(cell(5))
        with self.assertRaisesRegex(ValueError, "ordered T5/T6"):
            analyze.parse_raw(raw)

    def test_rejects_contract_mutation(self) -> None:
        report = copy.deepcopy(cell(5))
        report["production_routing_changed"] = True
        with self.assertRaisesRegex(ValueError, "identity/scope"):
            analyze.analyze_cell(report)

    def test_direct_invocation_loads_without_package_context(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(Path(analyze.__file__).resolve()), "--help"],
            cwd="/tmp", capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_full_main_success_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            binary = package / "qualifier"
            assembly = package / "qualifier.s"
            binary.write_bytes(b"binary-fixture")
            assembly.write_bytes(b"assembly-fixture")
            binary_sha = analyze.identity(binary)["sha256"]
            assembly_sha = analyze.identity(assembly)["sha256"]
            plan = json.loads((Path(analyze.__file__).parent / "plan.json").read_text())
            provenance = json.loads(
                (Path(analyze.__file__).parent / "build-provenance.json").read_text())
            provenance["outputs"]["binary"]["sha256"] = binary_sha
            provenance["outputs"]["assembly"]["sha256"] = assembly_sha
            (package / "plan.json").write_text(json.dumps(plan))
            (package / "build-provenance.json").write_text(json.dumps(provenance))
            (package / "regression.stdout").write_text(
                "FP8 target gate-up T5/T6 regression PASS\n")
            (package / "regression.stderr").write_bytes(b"")
            (package / "regression.exit").write_text("0\n")
            raw = json.dumps(cell(5), allow_nan=False) + "\n" + \
                json.dumps(cell(6), allow_nan=False) + "\n"
            (package / "benchmark.stdout").write_text(raw)
            (package / "benchmark.stderr").write_bytes(b"")
            (package / "benchmark.exit").write_text("0\n")
            (package / "static.stdout").write_text(analyze.STATIC)
            (package / "hardware.txt").write_text(analyze.HARDWARE)
            (package / "power-before.txt").write_text("auto\n")
            (package / "power-after.txt").write_text("auto\n")
            arguments = ["analyze.py", "--raw", str(package / "benchmark.stdout"),
                         "--cell-t5", str(package / "cell-t5.json"), "--cell-t6",
                         str(package / "cell-t6.json"), "--summary",
                         str(package / "summary.json")]
            with mock.patch.object(analyze, "PACKAGE", package), \
                 mock.patch.object(analyze, "BINARY", binary), \
                 mock.patch.object(analyze, "ASSEMBLY", assembly), \
                 mock.patch.object(analyze, "BINARY_SHA", binary_sha), \
                 mock.patch.object(analyze, "ASSEMBLY_SHA", assembly_sha), \
                 mock.patch.object(sys, "argv", arguments):
                self.assertEqual(analyze.main(), 0)
            summary = json.loads((package / "summary.json").read_text())
            self.assertEqual(summary["status"], "accepted")
            self.assertTrue(summary["decision"]["requires_both_cells"])


if __name__ == "__main__":
    unittest.main()
