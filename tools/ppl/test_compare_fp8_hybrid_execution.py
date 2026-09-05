#!/usr/bin/env python3
"""Focused numerical tests for FP8/Q4 source-to-product localization."""

from __future__ import annotations

import unittest

from tools.ppl import compare_fp8_hybrid_execution as comparison


class HybridExecutionLocalizationTest(unittest.TestCase):
    def test_pair_is_directional_and_position_aligned(self) -> None:
        result = comparison._pair(
            [1.0, 11.0, 2.0], [2.0, 9.0, 2.5],
            [1, 2, 3], [1, 4, 3],
        )
        self.assertAlmostEqual(result["mean_nll_delta"], -1.0 / 6.0)
        self.assertEqual(result["new_severe_positions"], 0)
        self.assertEqual(result["repaired_severe_position_indices"], [1])
        self.assertEqual(result["argmax_mismatches"], 1)
        self.assertEqual(result["argmax_first_mismatch"], 1)

    def test_greedy_attribution_partitions_every_position(self) -> None:
        result = comparison._greedy_attribution(
            [1, 1, 1, 1, 1],
            [1, 2, 1, 2, 2],
            [1, 1, 3, 2, 3],
        )
        self.assertEqual(result, {
            "positions": 5,
            "all_three_equal": 1,
            "source_only_differs_from_bf16_product_repairs": 1,
            "product_only_differs_from_bf16_runtime_introduces": 1,
            "both_differ_from_bf16_same_prediction": 1,
            "both_differ_from_bf16_different_predictions": 1,
        })

    def test_severe_attribution_distinguishes_repair_and_introduction(self) -> None:
        result = comparison._severe_attribution(
            [11.0, 1.0, 1.0, 1.0],
            [12.0, 11.0, 1.0, 1.0],
            [13.0, 1.0, 12.0, 1.0],
        )
        self.assertEqual(result["new_vs_bf16_common"], [])
        self.assertEqual(result["new_vs_bf16_source_only_product_repairs"], [1])
        self.assertEqual(result["new_vs_bf16_product_only_runtime_introduces"], [2])

    def test_32k_pair_requires_all_16383_positions(self) -> None:
        count = 16383
        nll = [1.0] * count
        argmax = [7] * count
        result = comparison._pair(nll, nll, argmax, argmax)
        self.assertEqual(result["severe_positions_compared"], count)
        self.assertEqual(result["argmax_compared"], count)
        with self.assertRaisesRegex(ValueError, "exactly aligned"):
            comparison._pair(nll, nll[:-1], argmax, argmax)


if __name__ == "__main__":
    unittest.main()
