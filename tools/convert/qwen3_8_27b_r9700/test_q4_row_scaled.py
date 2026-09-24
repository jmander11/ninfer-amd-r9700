from __future__ import annotations

import unittest

import torch

from .q4_row_scaled import compare_sample, quantize_dequantize
from .screen_q4_row_scaled_quality import assemble_report, select_row_indices
from .screen_q4_group_sizes_quality import _production_model, measure


class Q4RowScaledTest(unittest.TestCase):
    def test_row_scaled_uses_one_scale_and_q4g64_uses_each_group(self) -> None:
        source = torch.zeros((1, 128), dtype=torch.bfloat16)
        source[0, :64] = torch.linspace(-1, 1, 64).to(torch.bfloat16)
        source[0, 64:] = torch.linspace(-16, 16, 64).to(torch.bfloat16)
        grouped = quantize_dequantize(source, group_size=64)
        row = quantize_dequantize(source, group_size=None)
        self.assertEqual(grouped.shape, source.shape)
        self.assertEqual(row.shape, source.shape)
        result = compare_sample(source)
        self.assertLess(result["q4g64"]["relative_l2"],
                        result["q4_row_scaled"]["relative_l2"])
        self.assertGreater(result["ratios"]["relative_l2"], 1.0)

    def test_zero_rows_and_invalid_inputs(self) -> None:
        source = torch.zeros((2, 64), dtype=torch.bfloat16)
        self.assertTrue(torch.equal(
            quantize_dequantize(source, group_size=None),
            torch.zeros_like(source, dtype=torch.float32),
        ))
        with self.assertRaisesRegex(TypeError, "rank-two BF16"):
            quantize_dequantize(source.float(), group_size=None)
        padded = quantize_dequantize(source[:, :63], group_size=64)
        self.assertEqual(tuple(padded.shape), (2, 63))
        for group_size in (128, 256, 512):
            padded = quantize_dequantize(source[:, :63], group_size=group_size)
            self.assertEqual(tuple(padded.shape), (2, 63))

    def test_selection_is_deterministic_and_includes_endpoints(self) -> None:
        selected = select_row_indices("matrix", 4096, 8)
        self.assertEqual(selected, select_row_indices("matrix", 4096, 8))
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 4095)

    def test_report_aggregates_both_codecs_and_ratios(self) -> None:
        def record(name: str, grouped: float, row: float) -> dict[str, object]:
            return {
                "name": name, "shape": [1, 64], "sampled_row_indices": [0],
                "sampled_rows": 1, "sampled_elements": 64, "zero_rows": 0,
                "q4g64": {"squared_error": grouped * grouped,
                           "squared_reference": 1.0, "relative_l2": grouped,
                           "max_abs": grouped * 2},
                "q4_row_scaled": {"squared_error": row * row,
                                   "squared_reference": 1.0, "relative_l2": row,
                                   "max_abs": row * 2},
                "ratios": {"relative_l2": row / grouped, "max_abs": row / grouped},
            }
        report = assemble_report(
            rows_per_tensor=1,
            tensors=[record("b", 0.1, 0.3), record("a", 0.2, 0.4)],
            source={"index": {"sha256": "1" * 64}},
            implementation={"tool": "2" * 64},
            worst_count=2,
        )
        aggregate = report["aggregate"]
        self.assertAlmostEqual(aggregate["candidate_over_control"]["relative_l2"],
                               (0.3**2 + 0.4**2) ** 0.5 /
                               (0.1**2 + 0.2**2) ** 0.5)
        self.assertEqual([item["name"] for item in report["tensors"]], ["a", "b"])
        self.assertEqual(report["sampling"]["candidate_tensor_count"], 2)

    def test_group_size_screen_and_production_reduction_model(self) -> None:
        source = torch.linspace(-4, 4, 1152).reshape(1, 1152).to(torch.bfloat16)
        record = measure("vision", (1, 1152), (0,), source)
        self.assertEqual(set(record["codecs"]), {"q4g64", "q4g128", "q4g256", "q4g512"})
        baseline = _production_model(64)
        for group, denominator in ((128, 2), (256, 4), (512, 8)):
            candidate = _production_model(group)
            self.assertEqual(candidate["scale_request_bytes"] * denominator,
                             baseline["scale_request_bytes"])
            self.assertEqual(candidate["fp32_group_accumulations"] * denominator,
                             baseline["fp32_group_accumulations"])
            self.assertLess(candidate["conservative_split_low_plus_16_high_max_abs"],
                            candidate["int32_limit"])


if __name__ == "__main__":
    unittest.main()
