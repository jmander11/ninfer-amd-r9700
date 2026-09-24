from __future__ import annotations

import unittest

from tools.bench.analyze_e4m3_hybrid import (
    expected_projection_objects,
    role_candidates,
    select_roles,
)


class E4M3HybridAnalysisTest(unittest.TestCase):
    def test_projection_schedule_is_complete_and_role_consistent(self) -> None:
        objects = expected_projection_objects()
        self.assertEqual(len(objects), 320)
        self.assertEqual(len({name for name, _ in objects}), 320)
        self.assertEqual(sum(name.endswith("mlp/gate_up") for name, _ in objects), 64)
        self.assertEqual(sum("/attention/query_key" in name for name, _ in objects), 16)
        self.assertEqual(sum("/gdn/query_key" in name for name, _ in objects), 48)

    def test_role_aggregation_and_knapsack_are_deterministic(self) -> None:
        calls = [
            {
                "name": "text/layers/0/mlp/down",
                "role": "text.mlp.down",
                "added_resident_bytes": 6,
                "q4_measured_service_ns": 12,
                "fp8_compute_floor_ns": 2.0,
                "projected_saving_ns": 10.0,
            },
            {
                "name": "text/layers/1/mlp/down",
                "role": "text.mlp.down",
                "added_resident_bytes": 6,
                "q4_measured_service_ns": 12,
                "fp8_compute_floor_ns": 2.0,
                "projected_saving_ns": 10.0,
            },
            {
                "name": "text/layers/0/mlp/gate_up",
                "role": "text.mlp.gate_up",
                "added_resident_bytes": 10,
                "q4_measured_service_ns": 16,
                "fp8_compute_floor_ns": 1.0,
                "projected_saving_ns": 15.0,
            },
        ]
        candidates = role_candidates(calls)
        selected = select_roles(candidates, 12)
        self.assertEqual(selected["selected_roles"], ["text.mlp.down"])
        self.assertEqual(selected["selected_object_count"], 2)
        self.assertEqual(selected, select_roles(list(reversed(candidates)), 12))


if __name__ == "__main__":
    unittest.main()
