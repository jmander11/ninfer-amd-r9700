from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tools.bench.decide_fp8_post_gate_up import (
    QUALIFICATIONS,
    SELECTED_ROLES,
    decide,
    sha256_file,
    validate_qualifier,
)


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _qualifier(qualification_id: str, fp8: float = 4.0, q4: float = 8.0) -> dict[str, object]:
    tokens, rows, columns = QUALIFICATIONS[qualification_id]["shape"]
    source = ROOT / "tools/r9700/fp8_gate_up_qual.hip"
    executable = Path("/bin/true")
    return {
        "schema": "ninfer.r9700.fp8_projection_qualification.v1",
        "artifact_type": "ninfer_r9700_fp8_projection_comparison",
        "schema_version": 1,
        "qualification_id": qualification_id,
        "provenance": {
            "source_root": str(ROOT), "source_sha256": sha256_file(source),
            "executable_path": str(executable), "executable_sha256": sha256_file(executable),
        },
        "hardware": {
            "device": "AMD Radeon AI PRO R9700", "architecture": "gfx1201", "wave_size": 32,
            "vendor_id": "0x1002", "device_id": "0x7551",
        },
        "power_profile": {"required": "auto", "before": "auto", "after": "auto"},
        "shape": {"tokens": tokens, "rows": rows, "columns": columns},
        "fp8_profile": "E4M3-outer-vec32f-hipBLASLt-top-supported",
        "q4_control": "A8Q4G64-m64n128-pingpong-production",
        "direct_weight_binding": True, "outer_vector_scales": True,
        "nonfinite_status_poisoning": True, "q4_nonfinite_status_poisoning": True,
        "no_clobber_rejection": True,
        "represented_format_axis_oracle": {
            "probe_count": 9, "tokens": [0, 17, 2047], "rows": [0, rows // 2, rows - 1],
            "fp8_max_bf16_steps": 1, "q4_max_bf16_steps": 2,
        },
        "fp8_complete_ms_samples": [fp8] * 14,
        "q4_complete_ms_samples": [q4] * 14,
        "fp8_complete_median_ms": fp8, "q4_complete_median_ms": q4,
        "fp8_over_q4_speedup": q4 / fp8,
        "current_q4_implementation_underperforms_fp8": q4 / fp8 > 1.0,
        "current_q4_implementation_poor_vs_fp8": q4 / fp8 > 1.0,
        "balanced_interleaved_pairs": 7,
        "trial_order": "7 paired repetitions, each fp8,q4 then q4,fp8",
        "selected_workspace_bytes": 0,
        "algorithm": {"selected_workspace_bytes": 0, "algorithm_max_workspace_bytes": 1,
                      "heuristic_returned": 1},
        "pass": True,
    }


def _inputs():
    gate_path = ROOT / "profiles/bench/r9700-fp8-gate-up-decision-20260904.json"
    hybrid_path = ROOT / "profiles/bench/r9700-e4m3-q4-hybrid-selection-20260904.json"
    capacity_path = ROOT / "profiles/bench/r9700-e4m3-capacity-analysis-20260904.json"
    provenance = {
        "gate_decision": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        "hybrid": {"path": str(hybrid_path), "sha256": sha256_file(hybrid_path)},
        "capacity": {"path": str(capacity_path), "sha256": sha256_file(capacity_path)},
    }
    for qualification_id in QUALIFICATIONS:
        provenance[qualification_id] = {"path": qualification_id, "sha256": "a" * 64}
    return _read(str(gate_path.relative_to(ROOT))), _read(str(hybrid_path.relative_to(ROOT))), _read(str(capacity_path.relative_to(ROOT))), provenance


class PostGateUpDecisionTest(unittest.TestCase):
    def test_capacity_frontier_selects_three_roles_and_two_shapes(self) -> None:
        gate, hybrid, capacity, provenance = _inputs()
        qualifiers = {name: _qualifier(name) for name in QUALIFICATIONS}
        report = decide(gate=gate, hybrid=hybrid, capacity=capacity,
                        qualifiers=qualifiers, provenance=provenance)
        self.assertEqual(tuple(report["selection_contract"]["selected_roles"]), SELECTED_ROLES)
        self.assertEqual(report["selection_contract"]["tightest_post_gate_up_slack_bytes"], 1_026_070_017)
        self.assertEqual(report["inventory"]["additional_added_resident_bytes"], 1_024_065_536)
        self.assertEqual(report["whole_p2048_projection"]["additional_measured_q4_service_ns"], 87_724_946)
        self.assertEqual(report["decision"]["verdict"], "proceed")
        self.assertEqual(min(cell["capacity_slack_bytes"] for cell in report["capacity"]["cells"]), 2_004_481)

    def test_slower_shape_rejects_and_axis_drift_fails_closed(self) -> None:
        gate, hybrid, capacity, provenance = _inputs()
        qualifiers = {name: _qualifier(name) for name in QUALIFICATIONS}
        qualifiers["gdn_query_key"] = _qualifier("gdn_query_key", 9.0, 8.0)
        report = decide(gate=gate, hybrid=hybrid, capacity=capacity,
                        qualifiers=qualifiers, provenance=provenance)
        self.assertEqual(report["decision"]["verdict"], "reject")
        malformed = copy.deepcopy(_qualifier("attention_qk_gate_value"))
        malformed["represented_format_axis_oracle"]["rows"][1] += 1
        with self.assertRaisesRegex(ValueError, "axis oracle"):
            validate_qualifier(malformed, "attention_qk_gate_value")

    def test_capacity_authority_drift_fails_closed(self) -> None:
        gate, hybrid, capacity, provenance = _inputs()
        capacity = copy.deepcopy(capacity)
        capacity["scenarios"][0]["groups"][0]["cells"][0]["projected_minimum_runtime_reservation_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "capacity arithmetic"):
            decide(gate=gate, hybrid=hybrid, capacity=capacity,
                   qualifiers={name: _qualifier(name) for name in QUALIFICATIONS},
                   provenance=provenance)


if __name__ == "__main__":
    unittest.main()
