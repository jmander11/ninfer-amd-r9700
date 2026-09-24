#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.bench.prepare_fp8_gate_up_hardware_proof import prepare


class PrepareFp8GateUpHardwareProofTest(unittest.TestCase):
    def test_binds_inputs_and_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualifier, admission, profiler, capture = (root / name for name in
                ("qual", "admission.json", "rocprofv3", "capture.so"))
            qualifier.write_bytes(b"qualifier")
            admission.write_bytes(b"{}")
            profiler.write_bytes(b"profiler")
            capture.write_bytes(b"capture")
            report, trace, plan_path = (root / "reports/report.json", root / "trace",
                                        root / "plan.json")
            plan = prepare(qualifier, admission, report, trace, root / "captures", capture,
                           plan_path, profiler)
            self.assertEqual(plan["qualifier"]["sha256"],
                             hashlib.sha256(b"qualifier").hexdigest())
            self.assertEqual(plan["admission_report"]["sha256"],
                             hashlib.sha256(b"{}").hexdigest())
            self.assertEqual(plan["qualification"], "gate_up")
            self.assertEqual(plan["schema"],
                             "ninfer.r9700.fp8_gate_up_hardware_proof_plan.v1")
            self.assertEqual(plan["command"][-2:], ["--output", str(report.absolute())])
            with self.assertRaisesRegex(ValueError, "reuse existing plan"):
                prepare(qualifier, admission, report, trace, root / "captures", capture,
                        plan_path, profiler)

    def test_attention_profile_binds_distinct_shape_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("qual", "admission.json", "rocprofv3", "capture.so")]
            for path in paths:
                path.write_bytes(b"x")
            plan = prepare(paths[0], paths[1], root / "report.json", root / "trace",
                           root / "captures", paths[3], root / "plan.json", paths[2],
                           "attention_qk_gate_value")
            self.assertEqual(plan["qualification"], "attention_qk_gate_value")
            self.assertEqual(plan["shape"], {"tokens": 2048, "rows": 7168, "columns": 5120})
            self.assertEqual(plan["schema"],
                             "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof_plan.v1")
            self.assertIn("attention_qk_gate_value_hardware", plan["command"])


if __name__ == "__main__":
    unittest.main()
