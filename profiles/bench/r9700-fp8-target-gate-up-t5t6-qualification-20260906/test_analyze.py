#!/usr/bin/env python3

from __future__ import annotations

import copy
import contextlib
import io
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
    incumbent = incumbent or [1.00] * 14
    challenger = challenger or [0.80] * 14
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
        "incumbent_median_ms": round(incumbent_median, 2),
        "challenger_median_ms": round(challenger_median, 2),
        "challenger_over_incumbent": round(challenger_median / incumbent_median, 2),
        "projected_64_call_saving_ms": round(
            64.0 * (incumbent_median - challenger_median), 2),
        "decision": "unmade_until_gpu_measurement_is_validated",
    }


class AnalyzerTest(unittest.TestCase):
    def test_retained_cells_have_robust_reject_certificates(self) -> None:
        raw = (Path(analyze.__file__).parent / "benchmark.stdout").read_text()
        results = [analyze.analyze_cell(value) for value in analyze.parse_raw(raw)]
        self.assertEqual([value["tokens"] for value in results], [5, 6])
        for value in results:
            certificate = value["robust_reject_certificate"]
            self.assertGreater(certificate["challenger_median_lower_ms"],
                               certificate["incumbent_median_upper_ms"])
            self.assertFalse(value["decision"]["accepted"])

    def test_possible_pass_is_ambiguous_and_never_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous or could pass"):
            analyze.analyze_cell(cell(5))

    def test_rejects_slow_cell_without_rejecting_valid_report(self) -> None:
        result = analyze.analyze_cell(cell(6, challenger=[1.05] * 14))
        self.assertFalse(result["decision"]["accepted"])
        self.assertLess(result["derived"]["projected_64_call_saving_ms"], 0.0)
        self.assertFalse(
            result["robust_reject_certificate"]["positive_projected_saving_possible"])
        self.assertFalse(
            result["serialization"]["acceptance_from_rounded_evidence_permitted"])

    def test_order_failure_with_possible_saving_remains_ambiguous(self) -> None:
        challenger = [0.80 if index % 2 == 0 else 0.99 for index in range(14)]
        report = cell(5, incumbent=[1.0] * 14, challenger=challenger)
        with self.assertRaisesRegex(ValueError, "ambiguous or could pass"):
            analyze.analyze_cell(report)

    def test_rejects_mutated_raw_sample(self) -> None:
        report = cell(5)
        report["challenger_samples_ms"] = [0.95] * 14
        with self.assertRaisesRegex(ValueError, "incompatible with retained rounded samples"):
            analyze.analyze_cell(report)

    def test_rejects_finer_than_retained_serialization(self) -> None:
        report = cell(5)
        report["incumbent_samples_ms"][0] = 1.001
        with self.assertRaisesRegex(ValueError, "not serialized at the retained 0.01 precision"):
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

    def test_validation_failure_has_distinct_exit_status(self) -> None:
        with mock.patch.object(analyze, "main", side_effect=ValueError("invalid evidence")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(analyze.entrypoint(), 2)
        self.assertIn("invalid evidence", stderr.getvalue())

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
            provenance["outputs"]["binary"] = analyze.capture_identity(
                binary, binary.stat().st_size, binary_sha)
            provenance["outputs"]["assembly"] = analyze.capture_identity(
                assembly, assembly.stat().st_size, assembly_sha)
            (package / "plan.json").write_text(json.dumps(plan))
            (package / "build-provenance.json").write_text(json.dumps(provenance))
            (package / "regression.stdout").write_text(
                "FP8 target gate-up T5/T6 regression PASS\n")
            (package / "regression.stderr").write_bytes(b"")
            (package / "regression.exit").write_text("0\n")
            raw = json.dumps(cell(5, challenger=[1.10] * 14), allow_nan=False) + "\n" + \
                json.dumps(cell(6, challenger=[1.05] * 14), allow_nan=False) + "\n"
            (package / "benchmark.stdout").write_text(raw)
            (package / "benchmark.stderr").write_bytes(b"")
            (package / "benchmark.exit").write_text("0\n")
            (package / "result.sha256").write_text("retained capture receipt fixture\n")
            repair = {
                "artifact_type":
                    "ninfer_r9700_fp8_target_gate_up_t5t6_analysis_repair",
                "schema_version": 1,
                "status": "analysis_only_no_gpu_rerun",
                "gpu_rerun_permitted": False,
                "serialization_quantum_ms": 0.01,
                "capture_analyzer": {
                    "sha256":
                        "e2d8931c86fa7666887dfc5dda9688cc8beea1cddd959e936a9b6bc97dc402c0"
                },
                "repaired_analyzer": analyze.identity(Path(analyze.__file__)),
                "capture_result_closure": analyze.identity(package / "result.sha256"),
                "capture_benchmark_stdout": analyze.identity(package / "benchmark.stdout"),
            }
            (package / "analysis-repair.json").write_text(json.dumps(repair))
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
                 mock.patch.object(analyze, "BINARY_BYTES", binary.stat().st_size), \
                 mock.patch.object(analyze, "ASSEMBLY_BYTES", assembly.stat().st_size), \
                 mock.patch.object(sys, "argv", arguments):
                self.assertEqual(analyze.main(), 1)
            summary = json.loads((package / "summary.json").read_text())
            self.assertEqual(summary["status"], "rejected")
            self.assertTrue(summary["decision"]["requires_both_cells"])


if __name__ == "__main__":
    unittest.main()
