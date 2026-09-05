#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cu_mask_repeat1_validate", HERE / "validate.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metrics(iu4_half: float, iu4_quarter: float,
            q4_half: float | None = None, q4_quarter: float | None = None):
    q4_half = iu4_half if q4_half is None else q4_half
    q4_quarter = iu4_quarter if q4_quarter is None else q4_quarter
    return {
        "full64": {"iu4": 400.0, "stream": 600.0, "q4_ms": 100.0},
        "half32": {"iu4": 400.0 * iu4_half, "stream": 570.0,
                   "q4_ms": 100.0 / q4_half},
        "quarter16": {"iu4": 400.0 * iu4_quarter, "stream": 480.0,
                      "q4_ms": 100.0 / q4_quarter},
    }


class RepeatContractTest(unittest.TestCase):
    def test_live_plan_contract(self):
        plan = json.loads((HERE / "plan.json").read_text())
        MODULE.validate_plan(plan, verify_inputs=False)
        prior = json.loads((HERE.parent /
            "r9700-n16k16-q4-cu-mask-causal-prepare-20260905/plan.json").read_text())
        for field in ("hardware", "workload", "arms", "power", "inputs"):
            self.assertEqual(plan[field], prior[field])
        self.assertEqual(plan["peak_probe"]["controls"], prior["peak_probe"]["controls"])
        self.assertEqual(plan["classification"]["relative_curve_tolerance"], 0.10)
        self.assertEqual(plan["peak_probe"]["iu4_mask_response_relative_tolerance"], 0.15)

    def test_auto_boost_within_fifteen_percent_is_valid(self):
        result = MODULE.classify(metrics(0.575, 0.2875))
        self.assertEqual(result["classification"], "cu_local_service")
        self.assertEqual(result["nominal_iu4_guard_relative_tolerance"], 0.15)
        self.assertEqual(result["causal_curve_relative_tolerance"], 0.10)
        self.assertFalse(result["pooled_with_prior"])
        self.assertFalse(result["further_repeat_authorized"])

    def test_nominal_response_beyond_fifteen_percent_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "within 15%"):
            MODULE.classify(metrics(0.576, 0.25))
        with self.assertRaisesRegex(ValueError, "within 15%"):
            MODULE.classify(metrics(0.5, 0.288))

    def test_causal_curve_bound_remains_ten_percent(self):
        result = MODULE.classify(metrics(0.5, 0.25, 0.551, 0.276))
        self.assertEqual(result["classification"], "mixed_or_inconclusive")
        self.assertFalse(result["within_ten_percent"]["iu4"])

    def test_invalid_first_run_ratios_cross_only_the_new_nominal_guard(self):
        result = MODULE.classify(metrics(0.5511457706407582, 0.279208102655735,
            0.6029022719661692, 0.30739990483186225))
        self.assertEqual(result["classification"], "mixed_or_inconclusive")
        self.assertFalse(result["within_ten_percent"]["iu4"])

    def test_missing_arm_rejected(self):
        value = metrics(0.5, 0.25)
        del value["quarter16"]
        with self.assertRaisesRegex(ValueError, "all three arms"):
            MODULE.classify(value)

    def test_nonpositive_or_boolean_metric_rejected(self):
        for bad in (0, -1, True):
            value = metrics(0.5, 0.25)
            value["half32"]["q4_ms"] = bad
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                MODULE.classify(value)


if __name__ == "__main__":
    unittest.main()
