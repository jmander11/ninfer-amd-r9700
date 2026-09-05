#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_post_hybrid_prefill import project


ROOT = Path(__file__).resolve().parents[2]


class PostHybridPrefillProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = ROOT / (
            "profiles/rocprof/diagnostic-p2048-current-production-selected-region-20260904/"
            "measured-selected-region-attribution.json"
        )
        self.baseline = ROOT / "profiles/bench/r9700-dense-bk-crossover-p2048-c1-20260904.json"
        self.hybrid = ROOT / (
            "profiles/bench/r9700-fp8-four-role-pre-shared-workspace-p2048-c1-20260904.json"
        )
        self.roles = [
            ROOT / "profiles/bench/r9700-fp8-vs-a8q4-gate-up-linear-execution-20260904.json",
            ROOT
            / "profiles/bench/r9700-fp8-vs-a8q4-attention-qk-gate-value-linear-execution-20260904.json",
            ROOT
            / "profiles/bench/r9700-fp8-vs-a8q4-gdn-query-key-linear-execution-20260904.json",
        ]

    def test_retained_inputs_produce_the_bounded_inventory(self) -> None:
        report = project(self.trace, self.baseline, self.hybrid, self.roles, 2000.0)
        self.assertAlmostEqual(report["floor"]["remaining_wall_gap_ms"], 235.509372)
        self.assertEqual(report["selected_fp8_roles"]["semantic_calls"], 144)
        self.assertEqual(report["remaining_q4_roles"]["semantic_calls"], 176)
        self.assertAlmostEqual(
            report["selected_fp8_roles"]["projected_saving_ms"], 198.49288
        )
        self.assertEqual(report["dense_attention"]["semantic_layer_calls"], 16)
        self.assertEqual(report["gdn_recurrence"]["semantic_calls"], 48)
        self.assertEqual(report["xattention"]["current_trace_calls"], 0)
        self.assertGreater(
            report["minimum_unselected_two_family_bound"]["removable_upper_bound_ms"],
            report["floor"]["remaining_wall_gap_ms"],
        )

    def test_mismatched_trace_workload_is_rejected(self) -> None:
        trace = json.loads(self.trace.read_text(encoding="utf-8"))
        trace["workload"]["concurrency"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact C1/P2048"):
                project(path, self.baseline, self.hybrid, self.roles, 2000.0)


if __name__ == "__main__":
    unittest.main()
