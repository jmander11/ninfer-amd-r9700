from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.bench.decide_fp8_gate_up import (
    _load_bound,
    decide,
    gate_up_inventory,
    sha256_file,
    validate_qualifier,
)


def _qualifier(fp8: float = 4.0, q4: float = 8.0) -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[2]
    source_path = source_root / "tools/r9700/fp8_gate_up_qual.hip"
    executable = Path("/bin/true")
    return {
        "schema": "ninfer.r9700.fp8_gate_up_qualification.v1",
        "artifact_type": "ninfer_r9700_fp8_gate_up_comparison",
        "schema_version": 2,
        "provenance": {
            "source_root": str(source_root),
            "source_sha256": sha256_file(source_path),
            "executable_path": str(executable),
            "executable_sha256": sha256_file(executable),
        },
        "pass": True,
        "shape": {"tokens": 2048, "rows": 34816, "columns": 5120},
        "fp8_profile": "E4M3-outer-vec32f-hipBLASLt-top-supported",
        "q4_control": "A8Q4G64-m64n128-pingpong-production",
        "hardware": {
            "device": "AMD Radeon AI PRO R9700", "architecture": "gfx1201",
            "wave_size": 32, "vendor_id": "0x1002", "device_id": "0x7551",
        },
        "power_profile": {"required": "auto", "before": "auto", "after": "auto"},
        "direct_weight_binding": True,
        "represented_format_axis_oracle": {
            "probe_count": 9,
            "tokens": [0, 17, 2047],
            "rows": [0, 12345, 34815],
            "fp8_max_bf16_steps": 1,
            "q4_max_bf16_steps": 2,
        },
        "outer_vector_scales": True,
        "nonfinite_status_poisoning": True,
        "q4_nonfinite_status_poisoning": True,
        "no_clobber_rejection": True,
        "fp8_complete_ms_samples": [fp8] * 14,
        "q4_complete_ms_samples": [q4] * 14,
        "fp8_complete_median_ms": fp8,
        "q4_complete_median_ms": q4,
        "fp8_over_q4_speedup": q4 / fp8,
        "trial_order": "7 paired repetitions, each fp8,q4 then q4,fp8",
        "balanced_interleaved_pairs": 7,
        "current_q4_implementation_underperforms_fp8": q4 / fp8 > 1.0,
        "current_q4_implementation_poor_vs_fp8": q4 / fp8 > 1.0,
        "selected_workspace_bytes": 1024,
        "algorithm": {
            "selected_workspace_bytes": 1024,
            "algorithm_max_workspace_bytes": 2048,
            "heuristic_returned": 2,
        },
    }


def _inputs(budget: int = 6_000_000_000):
    inventory = gate_up_inventory()
    provenance = {
        name: {"path": name, "sha256": name[0] * 64}
        for name in ("qualifier", "hybrid", "capacity", "benchmark", "kernel_trace", "marker_trace")
    }
    hybrid = {
        "schema": "ninfer.r9700.e4m3-q4-hybrid-selection.v1",
        "model": {"baseline_prefill_seconds": 1.5, "baseline_total_seconds": 1.51},
        "provenance": {
            name: provenance[name]
            for name in ("capacity", "benchmark", "kernel_trace", "marker_trace")
        },
        "ranked_role_candidates": [{
            **inventory,
            "q4_measured_service_ns": 400_000_000,
        }],
        "scenarios": [
            {
                "prefill_chunk": chunk,
                "cells": [
                    {
                        "kv_value_group": group,
                        "concurrency": concurrency,
                        "maximum_fp8_over_q4_bytes": budget,
                    }
                    for group in (16, 32)
                    for concurrency in range(1, 5)
                ],
            }
            for chunk in (2048, 8192)
        ],
    }
    device = 34_000_000_000
    minimum_runtime = device - (1 << 30) - 15_159_815_680 - budget
    capacity = {
        "schema": "ninfer.r9700.e4m3-capacity-analysis.v1",
        "planner_contract": {
            "speculative_backend": "mtp", "draft_tokens": 3, "device_graph": True,
            "max_context": 262144, "automatic_headroom_bytes": 1 << 30,
            "device_capacity_bytes": device,
        },
        "scenarios": [
            {
                "workspace": {"prefill_chunk": chunk},
                "groups": [
                    {
                        "kv_value_group": group,
                        "cells": [
                            {
                                "concurrency": concurrency,
                                "projected_minimum_runtime_reservation_bytes": minimum_runtime,
                            }
                            for concurrency in range(1, 5)
                        ],
                    }
                    for group in (16, 32)
                ],
            }
            for chunk in (2048, 8192)
        ],
    }
    benchmark = {
        "schema_version": 20,
        "config": {
            "q4_prefill_cta_profile": "m64n128-pingpong-production",
            "concurrency": 1,
            "spec": "none",
        },
        "tests": [{
            "n_prompt": 2048, "n_gen": 0,
            "prefill_seconds_mean": 1.5, "total_seconds_mean": 1.51,
        }],
    }
    return hybrid, capacity, benchmark, provenance


class GateUpDecisionTest(unittest.TestCase):
    def test_exact_inventory_and_proceed_decision(self) -> None:
        inventory = gate_up_inventory()
        self.assertEqual(inventory["object_count"], 64)
        self.assertEqual(inventory["added_resident_bytes"], 5_356_650_496)
        hybrid, capacity, benchmark, provenance = _inputs()
        report = decide(
            qualifier=_qualifier(), hybrid=hybrid, capacity=capacity,
            benchmark=benchmark, trace_gate_up_service_ns=400_000_000,
            provenance=provenance
        )
        self.assertEqual(report["decision"]["verdict"], "proceed")
        self.assertEqual(len(report["capacity"]["cells"]), 16)
        self.assertTrue(report["capacity"]["all_cells_preserve_minimum_startup"])
        self.assertEqual(
            report["whole_p2048_projection"]["projected_gate_up_fp8_service_ns"],
            200_000_000,
        )

    def test_slower_measurement_rejects_and_shape_drift_fails_closed(self) -> None:
        hybrid, capacity, benchmark, provenance = _inputs()
        report = decide(
            qualifier=_qualifier(9.0, 8.0), hybrid=hybrid,
            capacity=capacity, benchmark=benchmark,
            trace_gate_up_service_ns=400_000_000, provenance=provenance
        )
        self.assertEqual(report["decision"]["verdict"], "reject")
        malformed = copy.deepcopy(_qualifier())
        malformed["shape"]["rows"] = 1
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_qualifier(malformed)

    def test_capacity_shortfall_rejects(self) -> None:
        hybrid, capacity, benchmark, provenance = _inputs(budget=5_000_000_000)
        report = decide(
            qualifier=_qualifier(), hybrid=hybrid, capacity=capacity,
            benchmark=benchmark, trace_gate_up_service_ns=400_000_000,
            provenance=provenance
        )
        self.assertEqual(report["decision"]["verdict"], "reject")
        self.assertFalse(report["capacity"]["all_cells_preserve_minimum_startup"])

    def test_input_hash_is_mandatory_and_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"schema": "fixture"}))
            self.assertEqual(_load_bound(path, sha256_file(path))["schema"], "fixture")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _load_bound(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
