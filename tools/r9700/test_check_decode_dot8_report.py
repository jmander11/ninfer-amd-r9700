#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.r9700.check_decode_dot8_report import INVENTORY, validate


def fixture() -> dict:
    shapes = []
    for (rows, columns), calls in INVENTORY.items():
        wmma = [1.0] * 7
        dot8 = [0.9] * 7
        shapes.append({"rows": rows, "columns": columns, "tokens": 1,
                       "exact_decode_calls": calls, "maximum_bf16_steps": 0,
                       "wmma_ms": 1.0, "dot8_ms": 0.9, "dot8_over_wmma": 0.9,
                       "robust_upper_ratio": 0.9,
                       "wmma_samples_ms": wmma, "dot8_samples_ms": dot8})
    saving = sum(INVENTORY.values()) * 0.1
    return {"schema_version": 2, "status": "passed",
            "correctness": {"passed": True},
            "static_and_resources": {"runtime_resources_passed": True},
            "performance_decision": {"every_cell_passed": True,
                                     "exact_call_weighted_saving_lower_ms_per_token": saving,
                                     "aggregate_passed": True, "accepted": True},
            "shapes": shapes}


class DecodeDot8ReportTest(unittest.TestCase):
    def write(self, root: Path, report: dict) -> Path:
        path = root / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_accepts_recomputed_seven_shape_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(validate(self.write(Path(directory), fixture())))

    def test_rejects_bad_call_weight_and_optimistic_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = fixture()
            report["shapes"][0]["exact_decode_calls"] += 1
            with self.assertRaisesRegex(ValueError, "incorrect exact-call weight"):
                validate(self.write(Path(directory), report))
        with tempfile.TemporaryDirectory() as directory:
            report = fixture()
            report["shapes"][0]["robust_upper_ratio"] = 0.8
            with self.assertRaisesRegex(ValueError, "inconsistent robust_upper_ratio"):
                validate(self.write(Path(directory), report))


if __name__ == "__main__":
    unittest.main()
