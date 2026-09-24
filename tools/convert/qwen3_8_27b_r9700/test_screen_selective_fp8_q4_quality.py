from __future__ import annotations

import unittest

import torch

from .screen_selective_fp8_q4_quality import (
    SELECTED_ROLES,
    assemble_report,
    measure,
    selected_specs,
)


def _records():
    records = []
    full = []
    for spec in selected_specs():
        role = ".".join(("text", *spec.name.split("/")[3:]))
        value = 0.02 + len(records) * 1e-7
        item = {
            "name": spec.name, "role": role, "shape": list(spec.shape),
            "sampled_row_indices": [0, spec.shape[0] - 1], "sampled_rows": 2,
            "sampled_elements": 2 * spec.shape[1], "nonfinite_values": 0, "zero_rows": 0,
            "e4m3": {"relative_l2": value, "max_abs": value * 2,
                      "squared_error": value * value, "squared_reference": 1.0},
            "q4g64": {"relative_l2": value * 2, "max_abs": value * 4,
                       "squared_error": value * value * 4, "squared_reference": 1.0},
            "e4m3_over_q4g64": {"relative_l2": 0.5, "max_abs": 0.5},
        }
        records.append(item)
        full.append({"name": spec.name, "sampled_row_indices": item["sampled_row_indices"],
                     "relative_l2": value, "max_abs": value * 2})
    full.append({"name": "other", "sampled_row_indices": [0, 1],
                 "relative_l2": 0.1, "max_abs": 0.2})
    reference = {
        "identity": {"path": "all.json", "sha256": "a" * 64},
        "aggregate": {"relative_l2": 0.03, "max_abs": 0.2},
        "tensors": full,
    }
    return records, reference


class SelectiveQualityScreenTest(unittest.TestCase):
    def test_exact_selected_inventory(self) -> None:
        specs = selected_specs()
        self.assertEqual(len(specs), 144)
        roles = {".".join(("text", *spec.name.split("/")[3:])) for spec in specs}
        self.assertEqual(roles, set(SELECTED_ROLES))

    def test_measure_compares_e4m3_and_q4g64(self) -> None:
        source = torch.tensor([[0.0, 0.125, -0.25, 0.5] * 16], dtype=torch.bfloat16)
        result = measure("text/layers/0/mlp/gate_up", (1, 64), (0,), source)
        self.assertEqual(result["sampled_rows"], 1)
        self.assertLess(result["e4m3"]["relative_l2"], result["q4g64"]["relative_l2"])

    def test_report_is_role_aggregated_and_detects_no_global_worst(self) -> None:
        records, reference = _records()
        report = assemble_report(
            rows_per_tensor=2, tensors=list(reversed(records)), source={}, implementation={},
            all_matrix_reference=reference, worst_count=3,
        )
        self.assertEqual(set(report["per_role"]), set(SELECTED_ROLES))
        self.assertEqual(report["aggregate"]["object_count"], 144)
        self.assertAlmostEqual(report["aggregate"]["e4m3_over_q4g64"]["relative_l2"], 0.5)
        self.assertFalse(report["quality_concentration"]["contains_global_e4m3_relative_l2_worst"])


if __name__ == "__main__":
    unittest.main()
