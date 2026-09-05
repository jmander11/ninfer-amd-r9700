from __future__ import annotations

import unittest

from tools.bench.analyze_e4m3_capacity import (
    HEADROOM_BYTES,
    MAX_CONTEXT,
    assemble_report,
    canonical_json,
    fp8_linear_workspace,
    project_cell,
    project_workspace,
    q4_a8_workspace,
)
from tools.convert.qwen3_8_27b_r9700 import e4m3_inventory


def _source(concurrency: int, group: int) -> dict[str, object]:
    stride = 1_810_432 if group == 16 else 1_740_800
    sequence = 7_000_000_000 + concurrency * 10_000
    workspace = 603_619_587
    graph = concurrency * 1_000
    weights = 15_000_000_000
    device = 33_979_155_456
    return {
        "config": {
            "concurrency": concurrency,
            "spec": "mtp",
            "draft_tokens": 3,
            "use_device_graph": True,
            "kv_value_group": group,
        },
        "memory": {
            "max_context": MAX_CONTEXT,
            "kv_capacity_mode": "auto",
            "kv_capacity_page_groups": 4096,
            "kv_capacity_max_page_groups": concurrency * 4096,
            "weights": {"capacity_bytes": weights},
            "available_after_weights_bytes": device - weights,
            "sequence": {"capacity_bytes": sequence},
            "workspace": {"capacity_bytes": workspace},
            "request_transient": {"capacity_bytes": 0},
            "device_graph_allowance_bytes": graph,
            "minimum_runtime_reservation_bytes": sequence + workspace + graph,
            "kv_capacity_increment_bytes": 0 if concurrency == 1 else stride,
        },
    }


class E4M3CapacityAnalysisTest(unittest.TestCase):
    def test_workspace_substitution_matches_planner_scope_geometry(self) -> None:
        self.assertEqual(q4_a8_workspace(2048), 36_765_700)
        self.assertEqual(
            fp8_linear_workspace(2048),
            {
                "activation_bytes": 35_659_776,
                "hipblaslt_conservative_cap_bytes": 285_212_672,
                "linear_scratch_bytes": 320_872_448,
            },
        )
        projected = project_workspace(2048, 301_809_923)
        self.assertEqual(projected["planner_phase_bytes_excluding_linear_scratch"], 265_043_968)
        self.assertEqual(projected["projected_planner_workspace_bytes"], 585_916_671)

    def test_capacity_fails_closed_below_minimum_context_reservation(self) -> None:
        cell = project_cell(
            _source(2, 16),
            value_group=16,
            workspace_bytes=585_916_671,
            resident_weight_bytes=e4m3_inventory.DEVICE_ARENA_BYTES,
        )
        self.assertEqual(cell["status"], "startup_inadmissible_minimum_context")
        self.assertIsNone(cell["aggregate_capacity_tokens"])
        self.assertGreater(cell["minimum_startup_shortfall_bytes"], 0)

    def test_report_is_canonical_and_covers_all_eight_cells_per_scenario(self) -> None:
        summary = [
            {"prefill_chunk": 8192, "workspace_capacity_bytes": 1_207_238_915},
            {"prefill_chunk": 2048, "workspace_capacity_bytes": 301_809_923},
        ]
        sources = {
            group: {concurrency: _source(concurrency, group) for concurrency in range(1, 5)}
            for group in (16, 32)
        }
        report = assemble_report(
            workspace_summary=summary,
            planner_sources=sources,
            provenance={"hash": "fixed"},
        )
        self.assertEqual([item["workspace"]["prefill_chunk"] for item in report["scenarios"]], [2048, 8192])
        self.assertEqual(sum(len(group["cells"]) for scenario in report["scenarios"] for group in scenario["groups"]), 16)
        self.assertEqual(report["planner_contract"]["automatic_headroom_bytes"], HEADROOM_BYTES)
        self.assertEqual(canonical_json(report), canonical_json(report))


if __name__ == "__main__":
    unittest.main()
