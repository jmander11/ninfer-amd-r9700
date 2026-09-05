#!/usr/bin/env python3

from __future__ import annotations

import unittest

from tools.bench.post_hybrid_profile import (
    FP8_CALLS,
    Q4_ACTIVATION_SYMBOLS,
    Q4_CALLS,
    bucket,
)


REGIONS = {
    "post_mixer": "ninfer.post-mixer.prefill.layer payload=0",
    "attention": "ninfer.attention.prefill.layer payload=0",
    "gdn": "ninfer.gdn.prefill.layer payload=0",
}


def _rows(stage: str, symbol: str, count: int) -> list[dict[str, object]]:
    return [
        {"symbol": symbol, "roctx_region": REGIONS[stage], "duration_ns": 1_000_000}
        for _ in range(count)
    ]


class PostHybridProfileBucketTest(unittest.TestCase):
    def fixture(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        dispatches: list[dict[str, object]] = []
        for stage, count in FP8_CALLS.items():
            dispatches += _rows(stage, "fp8_quantize_activation_kernel", count)
            dispatches += _rows(stage, "Cijk_F8BS", count)
            dispatches += _rows(stage, "poison_nonfinite_output", count)
        for stage, count in Q4_CALLS.items():
            dispatches += _rows(stage, Q4_ACTIVATION_SYMBOLS[stage], count)
            dispatches += _rows(stage, "a8q4g64_linear_prefill_cta_kernel", count)
        dispatches += _rows("attention", "dense_full_score_qk_kernel", 16)
        dispatches += _rows("attention", "dense_full_score_maximum_kernel", 16)
        dispatches += _rows("attention", "dense_full_score_pv_kernel", 16)
        dispatches += _rows("gdn", "ordinary_kernel<true>", 48)
        attribution = {
            "operator_attribution": [
                {"execution_category": "base_text_prefill", "stage": "attention",
                 "operator_family": "dense_attention", "calls": 48,
                 "summed_duration_ms": 48.0},
                {"execution_category": "base_text_prefill", "stage": "gdn",
                 "operator_family": "gdn_recurrence", "calls": 48,
                 "summed_duration_ms": 48.0},
            ],
            "kernel_execution_categories": [
                {"category": "base_text_prefill", "independent_summed_duration_ms": 890.0}
            ],
            "prefill_no_kernel_wall_ms": 7.0,
        }
        return dispatches, attribution

    def test_exact_inventory_is_bucketed(self) -> None:
        dispatches, attribution = self.fixture()
        result = bucket(dispatches, attribution)
        self.assertEqual(result["selected_fp8"]["semantic_calls"], 144)
        self.assertEqual(result["selected_fp8"]["summed_duration_ms"], 432.0)
        self.assertEqual(result["remaining_q4"]["semantic_calls"], 176)
        self.assertEqual(result["remaining_q4"]["summed_duration_ms"], 352.0)
        self.assertEqual(result["dense_full_score_attention"]["semantic_layer_calls"], 16)
        self.assertEqual(result["gdn_recurrence"]["semantic_calls"], 48)
        self.assertEqual(result["other_base_text_kernel_service"]["summed_duration_ms"], 10.0)
        self.assertEqual(result["kernel_inactive_text_prefill_wall"]["summed_duration_ms"], 7.0)

    def test_missing_selected_dispatch_fails_closed(self) -> None:
        dispatches, attribution = self.fixture()
        dispatches.pop(0)
        with self.assertRaisesRegex(ValueError, "FP8 activation count"):
            bucket(dispatches, attribution)

    def test_superseded_post_mixer_quantizer_fails_closed(self) -> None:
        dispatches, attribution = self.fixture()
        for row in dispatches:
            if row["symbol"] == Q4_ACTIVATION_SYMBOLS["post_mixer"]:
                row["symbol"] = "a8g64_quantize_activation_kernel"
        with self.assertRaisesRegex(ValueError, "post_mixer Q4 activation count 0 != 64"):
            bucket(dispatches, attribution)


if __name__ == "__main__":
    unittest.main()
