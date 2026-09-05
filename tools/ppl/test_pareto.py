#!/usr/bin/env python3
"""Tests for multi-objective R9700 candidate classification."""

from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pareto


WORKLOADS = ["prefill_8k_c1", "decode_8k_c1", "decode_8k_c4"]
CHUNK_SELECTION = {
    "path": "/chunk-selection.json", "sha256": "9" * 64,
    "selection_rule": "global_maximin_normalized_prefill_then_workspace_then_smaller_chunk_v2",
    "selected_prefill_chunk": 4096,
}


def migration_receipt(weights_id: str) -> dict:
    recipe_id = {
        "r9700-q4g64-n16k16-eval": "r9700-all-q4g64-n16k16-eval-v1",
        "r9700-q4-w8-mse-n16k16-eval":
            "r9700-source-q4-n16k16-promoted-w8-source-mse8-eval-v1",
        "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
            "r9700-q4g64-f8e4m3-four-role-n16k16-eval-v1",
    }[weights_id]
    value = {"path": "/receipt.json", "sha256": "f" * 64, "recipe_id": recipe_id,
             "object_plan_sha256": "e" * 64, "source_artifact_sha256": "d" * 64,
             "source_receipt_sha256": "c" * 64, "transcoder_sha256": "b" * 64}
    if weights_id == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        value.update({"selection_sha256": "b2ceeb63c581c0f26aab5a4d8c0958da34d836fcc5c47d377bce709eaf37e3e8", "source_index_sha256": "9" * 64,
                      "source_ranking_sha256": "8" * 64})
    else:
        value["receipt_producer_sha256"] = "7" * 64
    return value


def shortlist_head_gate(*, extra_rounds: int = 0) -> dict:
    cells = {}
    for prompt in (8192, 32768):
        for concurrency in range(1, 5):
            expected = 64 * concurrency
            repetitions = [{
                "repetition": repetition,
                "observed_rounds": expected + extra_rounds,
                "expected_minimum_rounds": expected,
                "minimum_met": extra_rounds == 0,
            } for repetition in range(3)]
            cells[f"whole-pp{prompt}+tg256_c{concurrency}"] = {
                "generated_tokens_per_lane": 256, "draft_window": 3,
                "minimum_rounds_per_lane": 64, "concurrency": concurrency,
                "expected_rounds_per_repetition": expected,
                "expected_rounds_all_repetitions": expected * 3,
                "observed_rounds_all_repetitions": (expected + extra_rounds) * 3,
                "all_repetitions_minimum": extra_rounds == 0,
                "repetitions": repetitions,
            }
    return {
        "status": (
            "q4_round_gate_pass_pending_trace"
            if extra_rounds == 0 else "conditional_head_precision_required"
        ),
        "head_format": "Q4G64_F16S", "activation_profile": "A8G64",
        "all_repetitions_minimum": extra_rounds == 0,
        "counter_semantics": (
            "per-repetition counters sum request-lane rounds; every whole g256 MTP3 "
            "cell requires concurrency * 64 rounds"
        ),
        "cells": cells,
    }


def candidate(
    name: str,
    *,
    mean_nll_delta: float,
    severe_rate: float,
    speed: tuple[float, float, float],
    capacity: int,
    eligible: bool = True,
    tier: str = "accuracy",
    score_seconds: float = 1.0,
    scored_positions: int = 10000,
    new_severe_positions: int | None = None,
    group: int = 16,
    weight_recipe: str = "recipe-a",
    xattention_profile: str = "dense",
) -> dict:
    if new_severe_positions is None:
        new_severe_positions = round(severe_rate * scored_positions)
    return {
        "name": name,
        "prefill_chunk": 4096,
        "weight_recipe": weight_recipe,
        "cache_profile": {
            "value_group": group,
            "plane_layouts": {
                "key": "token-fastest-head-major",
                "value": "feature-fastest-page-major",
                "value_scale": "feature-fastest-page-major",
            },
        },
        "execution_profile": {
            "q4_activation_bits": 8,
            "w8_activation_bits": 8,
            "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
            "xattention_profile": xattention_profile,
        },
        "shortlist_head_precision_gate": shortlist_head_gate(),
        "quality": {
            "eligible": eligible,
            "tier": tier,
            "mean_nll_delta": mean_nll_delta,
            "complete_finite_aligned": True,
            "scored_positions": scored_positions,
            "new_severe_positions": new_severe_positions,
        },
        "whole_inference_tokens_per_second": dict(zip(WORKLOADS, speed, strict=True)),
        "capacity": {
            "measurement_kind": "resolved_effective_maximum",
            "binding_constraint": "device_memory",
            "tokens": capacity,
        },
        "score_seconds": score_seconds,
    }


class ParetoTest(unittest.TestCase):
    def test_rejects_non_product_concurrency_cells(self) -> None:
        row = candidate(
            "candidate", mean_nll_delta=0.0, severe_rate=0.0,
            speed=(1.0, 1.0, 1.0), capacity=1,
        )
        with self.assertRaisesRegex(ValueError, "C=1..4"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_speed_workloads": ["prefill_8k_c1", "decode_8k_c5"],
                "candidates": [row],
            })

    def test_rejects_unrequested_non_product_candidate_cells(self) -> None:
        row = candidate(
            "candidate", mean_nll_delta=0.0, severe_rate=0.0,
            speed=(1.0, 1.0, 1.0), capacity=1,
        )
        row["whole_inference_tokens_per_second"]["whole_8k_c5"] = 1.0
        with self.assertRaisesRegex(ValueError, r"non-product C5\+"):
            self.classify([row])

        del row["whole_inference_tokens_per_second"]["whole_8k_c5"]
        row["capacity_by_cell"] = {"c5": row.pop("capacity")}
        with self.assertRaisesRegex(ValueError, r"non-product C5\+"):
            self.classify([row])

    def classify(self, candidates: list[dict]) -> dict:
        return pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input",
            "schema_version": 4,
            "required_speed_workloads": WORKLOADS,
            "candidates": candidates,
        })

    def test_input_schema_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema v4"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input",
                "schema_version": 2,
                "required_speed_workloads": WORKLOADS,
                "candidates": [],
            })

    def test_dominated_requires_no_worse_on_every_objective(self) -> None:
        better = candidate(
            "better", mean_nll_delta=0.01, severe_rate=0.0005,
            speed=(100.0, 80.0, 300.0), capacity=32768,
        )
        worse = candidate(
            "worse", mean_nll_delta=0.02, severe_rate=0.0007,
            speed=(90.0, 70.0, 250.0), capacity=16384,
        )
        result = self.classify([worse, better])
        rows = {row["name"]: row for row in result["candidates"]}
        self.assertEqual(result["frontier"], ["better"])
        self.assertEqual(rows["worse"]["dominated_by"], ["better"])

    def test_quality_speed_tradeoff_keeps_both_candidates(self) -> None:
        quality = candidate(
            "quality", mean_nll_delta=0.005, severe_rate=0.0,
            speed=(80.0, 70.0, 250.0), capacity=32768,
        )
        speed = candidate(
            "speed", mean_nll_delta=0.015, severe_rate=0.0005,
            speed=(120.0, 100.0, 350.0), capacity=32768,
        )
        self.assertEqual(self.classify([quality, speed])["frontier"], ["quality", "speed"])

    def test_score_seconds_cannot_select_or_dominate(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768, score_seconds=1.0,
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768, score_seconds=100.0,
        )
        result = self.classify([first, second])
        self.assertFalse(result["score_seconds_is_objective"])
        self.assertEqual(result["frontier"], ["first", "second"])

    def test_missing_whole_inference_or_failed_quality_is_not_comparable(self) -> None:
        missing = candidate(
            "missing", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768,
        )
        del missing["whole_inference_tokens_per_second"]["decode_8k_c4"]
        failed = candidate(
            "failed", mean_nll_delta=0.03, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768, eligible=False,
        )
        result = self.classify([missing, failed])
        self.assertEqual(result["frontier"], [])
        rows = {row["name"]: row for row in result["candidates"]}
        self.assertIn("quality_guardrails_not_met", rows["failed"]["reasons"])
        self.assertIn(
            "missing_positive_whole_inference_speed:decode_8k_c4",
            rows["missing"]["reasons"],
        )

    def test_declared_eligibility_is_recomputed(self) -> None:
        dishonest = candidate(
            "dishonest", mean_nll_delta=0.03, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768, eligible=True,
        )
        result = self.classify([dishonest])
        row = result["candidates"][0]
        self.assertFalse(row["comparable"])
        self.assertIn("quality_guardrails_not_met", row["reasons"])
        self.assertIn("declared_quality_eligibility_mismatch", row["reasons"])

    def test_measured_all_q4_a8_is_capacity_speed_eligible(self) -> None:
        all_q4 = candidate(
            "all-q4-a8", mean_nll_delta=0.03950884100572821,
            severe_rate=9 / 4095, speed=(100.0, 80.0, 300.0), capacity=65536,
            tier="capacity-speed", scored_positions=4095, new_severe_positions=9,
        )
        result = self.classify([all_q4])
        row = result["candidates"][0]
        self.assertTrue(row["comparable"])
        self.assertEqual(
            row["objectives"]["quality_cells"]["quality"]["new_severe_position_budget"], 11
        )

    def test_quality_and_capacity_cells_are_compared_independently(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768,
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768,
        )
        for row, q8, q32, c1, c4 in (
            (first, 0.005, 0.015, 40000, 30000),
            (second, 0.010, 0.010, 35000, 35000),
        ):
            base = row.pop("quality")
            row.pop("capacity")
            row["quality_cells"] = {
                "8k": {**base, "mean_nll_delta": q8},
                "32k": {**base, "mean_nll_delta": q32},
            }
            row["capacity_by_cell"] = {
                "c1": {"measurement_kind": "resolved_effective_maximum",
                       "binding_constraint": "model_context", "tokens": c1},
                "c4": {"measurement_kind": "resolved_effective_maximum",
                       "binding_constraint": "device_memory", "tokens": c4},
            }
        result = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_quality_cells": ["8k", "32k"],
            "required_capacity_cells": ["c1", "c4"],
            "required_speed_workloads": WORKLOADS,
            "candidates": [first, second],
        })
        self.assertEqual(result["frontier"], ["first", "second"])

    def test_workload_feasibility_cannot_populate_capacity_objective(self) -> None:
        row = candidate(
            "censored", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768,
        )
        row["capacity"]["measurement_kind"] = "required_workload_feasibility"
        result = self.classify([row])
        self.assertFalse(result["candidates"][0]["comparable"])
        self.assertIn(
            "capacity_is_not_resolved_effective_maximum",
            result["candidates"][0]["reasons"],
        )

    def test_same_recipe_selection_maximizes_worst_throughput_ratio(self) -> None:
        balanced = candidate(
            "balanced", mean_nll_delta=0.010, severe_rate=0.0005,
            speed=(100.0, 90.0, 90.0), capacity=32768, group=16,
        )
        spiky = candidate(
            "spiky", mean_nll_delta=0.005, severe_rate=0.0004,
            speed=(80.0, 100.0, 100.0), capacity=65536, group=32,
        )
        selection = self.classify([spiky, balanced])["same_recipe_static_profile_selection"]
        self.assertEqual(selection["selections"][0]["winner"], "balanced")
        self.assertEqual(
            selection["selections"][0]["decisive_stage"],
            "maximin_whole_inference_throughput",
        )
        self.assertEqual(
            selection["selections"][0]["normalized_objectives"]["balanced"]
            ["minimum_throughput_ratio"],
            0.9,
        )

    def test_same_recipe_selection_uses_capacity_tie_break(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=1, group=16,
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=1, group=32,
        )
        for row, c1, c4 in ((first, 100, 90), (second, 80, 100)):
            row.pop("capacity")
            row["capacity_by_cell"] = {
                "c1": {"measurement_kind": "resolved_effective_maximum",
                       "binding_constraint": "device_memory", "tokens": c1},
                "c4": {"measurement_kind": "resolved_effective_maximum",
                       "binding_constraint": "device_memory", "tokens": c4},
            }
        result = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_capacity_cells": ["c1", "c4"],
            "required_speed_workloads": WORKLOADS,
            "candidates": [first, second],
        })
        selected = result["same_recipe_static_profile_selection"]["selections"][0]
        self.assertEqual(selected["winner"], "first")
        self.assertEqual(selected["decisive_stage"], "maximin_resolved_capacity")

    def test_same_recipe_selection_uses_quality_budget_tie_break(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=32,
        )
        for row, delta, severe in ((first, 0.010, 2), (second, 0.002, 7)):
            quality = row.pop("quality")
            row["quality_cells"] = {
                "8k": {**quality, "mean_nll_delta": delta,
                       "new_severe_positions": severe},
                "32k": {**quality, "mean_nll_delta": delta,
                        "new_severe_positions": severe},
            }
        result = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_quality_cells": ["8k", "32k"],
            "required_speed_workloads": WORKLOADS,
            "candidates": [first, second],
        })
        selected = result["same_recipe_static_profile_selection"]["selections"][0]
        self.assertEqual(selected["winner"], "first")
        self.assertEqual(selected["decisive_stage"], "minimum_quality_budget_consumption")

    def test_same_recipe_selection_uses_canonical_profile_for_identical_vectors(self) -> None:
        g32 = candidate(
            "g32", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=32,
        )
        g16 = candidate(
            "g16", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
        )
        selected = self.classify([g32, g16])["same_recipe_static_profile_selection"][
            "selections"
        ][0]
        self.assertEqual(selected["winner"], "g16")
        self.assertEqual(selected["decisive_stage"], "canonical_static_profile_identity")
        self.assertEqual(selected["winner_cache_profile"], g16["cache_profile"])
        self.assertEqual(selected["winner_execution_profile"], g16["execution_profile"])

    def test_static_profile_tie_break_retains_attention_identity(self) -> None:
        sparse = candidate(
            "sparse", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768,
            xattention_profile="b128-s16-tau900",
        )
        dense = candidate(
            "dense", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768,
        )
        selected = self.classify([sparse, dense])["same_recipe_static_profile_selection"][
            "selections"
        ][0]
        self.assertEqual(selected["winner"], "sparse")
        self.assertEqual(
            selected["winner_execution_profile"]["xattention_profile"],
            "b128-s16-tau900",
        )

    def test_missing_execution_profile_is_rejected(self) -> None:
        row = candidate(
            "missing-profile", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768,
        )
        del row["execution_profile"]
        with self.assertRaisesRegex(ValueError, "execution_profile identity"):
            self.classify([row])

    def test_ineligible_profile_is_not_in_same_recipe_selection(self) -> None:
        valid = candidate(
            "valid", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
        )
        invalid = candidate(
            "invalid", mean_nll_delta=0.03, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=32, eligible=False,
        )
        selection = self.classify([valid, invalid])["same_recipe_static_profile_selection"]
        self.assertEqual(selection["selections"], [])
        self.assertEqual(selection["not_collapsed"], ["valid"])

    def test_different_recipes_are_not_collapsed(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
            weight_recipe="recipe-a",
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=32,
            weight_recipe="recipe-b",
        )
        selection = self.classify([first, second])["same_recipe_static_profile_selection"]
        self.assertEqual(selection["selections"], [])
        self.assertEqual(selection["not_collapsed"], ["first", "second"])

    def test_missing_zero_and_nonfinite_speed_cells_are_not_comparable(self) -> None:
        rows = []
        for name, value in (("missing", None), ("zero", 0.0), ("nan", math.nan),
                            ("infinite", math.inf)):
            row = candidate(
                name, mean_nll_delta=0.01, severe_rate=0.0,
                speed=(100.0, 80.0, 300.0), capacity=32768,
            )
            if value is None:
                del row["whole_inference_tokens_per_second"][WORKLOADS[1]]
            else:
                row["whole_inference_tokens_per_second"][WORKLOADS[1]] = value
            rows.append(row)
        result = self.classify(rows)
        self.assertEqual(result["frontier"], [])
        for row in result["candidates"]:
            self.assertIn(
                f"missing_positive_whole_inference_speed:{WORKLOADS[1]}", row["reasons"]
            )

    def test_required_workloads_reject_duplicates(self) -> None:
        row = candidate(
            "candidate", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 80.0, 300.0), capacity=32768,
        )
        with self.assertRaisesRegex(ValueError, "nonempty unique string list"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_speed_workloads": [WORKLOADS[0], WORKLOADS[0]],
                "candidates": [row],
            })

    def test_json_loader_rejects_duplicate_nested_workload_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key: decode"):
            pareto.load_payload(
                '{"required_speed_workloads":["decode"],"candidates":['
                '{"whole_inference_tokens_per_second":{"decode":1,"decode":2}}]}'
            )

    def test_provenance_artifact_identity_overrides_untrusted_recipe_label(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
            weight_recipe="same-label",
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=32,
            weight_recipe="same-label",
        )
        provenance = [
            {"candidate": "first", "artifact": {
                "weights_id": "weights", "sha256": "a" * 64,
            }},
            {"candidate": "second", "artifact": {
                "weights_id": "weights", "sha256": "b" * 64,
            }},
        ]
        result = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": WORKLOADS,
            "candidates": [first, second],
            "source_provenance": provenance,
        })
        selection = result["same_recipe_static_profile_selection"]
        self.assertEqual(selection["selections"], [])
        self.assertEqual(selection["not_collapsed"], ["first", "second"])
        self.assertEqual(
            result["candidates"][0]["weight_recipe"],
            {"kind": "artifact", "weights_id": "weights", "sha256": "a" * 64},
        )

    def test_matching_provenance_groups_candidates_despite_declared_labels(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
            weight_recipe="label-a",
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=32,
            weight_recipe="label-b",
        )
        artifact = {"weights_id": "weights", "sha256": "a" * 64}
        result = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": WORKLOADS,
            "candidates": [first, second],
            "source_provenance": [
                {"candidate": "first", "artifact": artifact},
                {"candidate": "second", "artifact": artifact},
            ],
        })
        selection = result["same_recipe_static_profile_selection"]
        self.assertEqual(selection["selections"][0]["frontier_candidates"],
                         ["first", "second"])
        self.assertEqual(selection["not_collapsed"], [])

    def test_invalid_or_ambiguous_provenance_is_not_used_for_grouping(self) -> None:
        first = candidate(
            "first", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
            weight_recipe="same-label",
        )
        second = candidate(
            "second", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=32,
            weight_recipe="same-label",
        )
        result = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": WORKLOADS,
            "candidates": [first, second],
            "source_provenance": [
                {"candidate": "first", "artifact": {
                    "weights_id": "weights", "sha256": "not-a-digest",
                }},
                {"candidate": "second", "artifact": {
                    "weights_id": "weights", "sha256": "a" * 64,
                }},
                {"candidate": "second", "artifact": {
                    "weights_id": "weights", "sha256": "a" * 64,
                }},
            ],
        })
        selection = result["same_recipe_static_profile_selection"]
        self.assertEqual(selection["selections"], [])
        self.assertEqual(selection["not_collapsed"], ["first", "second"])

    def test_output_candidate_order_is_canonical(self) -> None:
        first = candidate(
            "a", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768,
        )
        second = candidate(
            "b", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768,
        )
        forward = self.classify([first, second])
        reverse = self.classify([second, first])
        self.assertEqual(forward, reverse)

    def test_quality_fractions_use_each_declared_tier_budget(self) -> None:
        accuracy = candidate(
            "accuracy", mean_nll_delta=0.01, severe_rate=0.0,
            speed=(100.0, 100.0, 100.0), capacity=32768, group=16,
            tier="accuracy", scored_positions=10000, new_severe_positions=4,
        )
        capacity_speed = candidate(
            "capacity-speed", mean_nll_delta=math.log(1.05) / 2.0, severe_rate=0.0,
            speed=(101.0, 101.0, 101.0), capacity=32768, group=32,
            tier="capacity-speed", scored_positions=10000, new_severe_positions=25,
        )
        selection = self.classify([accuracy, capacity_speed])[
            "same_recipe_static_profile_selection"
        ]["selections"][0]
        normalized = selection["normalized_objectives"]
        self.assertEqual(
            normalized["accuracy"]["quality_budget_fraction_by_cell"]["quality"],
            {"mean_nll_delta": 0.5, "new_severe_positions": 0.4},
        )
        self.assertEqual(
            normalized["capacity-speed"]["quality_budget_fraction_by_cell"]["quality"],
            {"mean_nll_delta": 0.5, "new_severe_positions": 1.0},
        )

    def test_required_mode_rejects_single_recipe_authority(self) -> None:
        rows = []
        for group, profile in (
            (16, "dense"), (32, "dense"),
            (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
        ):
            winner = group == 16 and profile == "b128-s16-tau900"
            rows.append(candidate(
                f"g{group}-{profile}",
                mean_nll_delta=0.005 if winner else 0.010,
                severe_rate=0.0,
                speed=(120.0, 120.0, 360.0) if winner else (100.0, 100.0, 300.0),
                capacity=65536 if winner else 32768,
                group=group,
                xattention_profile=profile,
            ))
        artifact = {
            "weights_id": "r9700-q4g64-n16k16-eval",
            "sha256": "a" * 64,
            "conversion_receipt": migration_receipt("r9700-q4g64-n16k16-eval"),
        }
        with self.assertRaisesRegex(ValueError, "exactly all three product recipe"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input",
                "schema_version": 4,
                "required_speed_workloads": WORKLOADS,
                "require_single_static_profile_selection": True,
                "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
                "candidates": rows,
                "source_provenance": [
                    {"candidate": row["name"], "artifact": artifact} for row in rows
                ],
            })

    def test_required_mode_selects_one_terminal_winner_across_recipes(self) -> None:
        rows = []
        provenance = []
        for recipe, quality, winning_profile, winning_speed in (
            ("r9700-q4g64-n16k16-eval", 0.005, (16, "b128-s16-tau900"), 120.0),
            ("r9700-q4-w8-mse-n16k16-eval", 0.010, (32, "dense"), 130.0),
            ("r9700-q4g64-f8e4m3-four-role-n16k16-eval", 0.007, (16, "dense"), 125.0),
        ):
            artifact = {
                "weights_id": recipe,
                "sha256": chr(ord("a") + list(pareto.TERMINAL_RECIPE_PROFILES).index(recipe)) * 64,
                "conversion_receipt": migration_receipt(recipe),
            }
            for group, profile in (
                (16, "dense"), (32, "dense"),
                (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
            ):
                name = f"{recipe}-g{group}-{profile}"
                speed = winning_speed if (group, profile) == winning_profile else 90.0
                row = candidate(
                    name, mean_nll_delta=quality, severe_rate=0.0,
                    speed=(speed, speed, speed), capacity=32768,
                    group=group, xattention_profile=profile,
                )
                rows.append(row)
                provenance.append({"candidate": name, "artifact": artifact})
        result = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": WORKLOADS,
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
            "candidates": rows,
            "source_provenance": provenance,
        })
        terminal = result["terminal_production_selection"]
        self.assertEqual(
            terminal["eligible_profile_winners"],
            ["r9700-q4-w8-mse-n16k16-eval-g32-dense",
             "r9700-q4g64-f8e4m3-four-role-n16k16-eval-g16-dense",
             "r9700-q4g64-n16k16-eval-g16-b128-s16-tau900"],
        )
        self.assertEqual(terminal["winner"], "r9700-q4-w8-mse-n16k16-eval-g32-dense")
        self.assertEqual(
            terminal["decisive_stage"], "maximin_whole_inference_throughput"
        )
        self.assertEqual(
            terminal["winner_artifact"]["weights_id"], "r9700-q4-w8-mse-n16k16-eval"
        )
        self.assertEqual(result["selected_prefill_chunk"], 4096)
        self.assertEqual(result["prefill_chunk_selection"], CHUNK_SELECTION)
        self.assertTrue(all(row["prefill_chunk"] == 4096 for row in result["candidates"]))
        self.assertEqual(
            terminal["production_status"],
            "selected_route_pending_shortlist_head_trace_and_niah",
        )

        conditional_rows = [
            {**row, "shortlist_head_precision_gate": shortlist_head_gate(extra_rounds=1)}
            for row in rows
        ]
        conditional = pareto.classify({
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": WORKLOADS,
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
            "candidates": conditional_rows,
            "source_provenance": provenance,
        })
        self.assertEqual(
            conditional["terminal_production_selection"]["production_status"],
            "conditional_head_precision_required",
        )
        self.assertTrue(all(row["comparable"] for row in conditional["candidates"]))

        missing_head_gate = [{key: value for key, value in row.items()
                              if key != "shortlist_head_precision_gate"} for row in rows]
        with self.assertRaisesRegex(ValueError, "exact Q4 shortlist-head round gate"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_speed_workloads": WORKLOADS,
                "require_single_static_profile_selection": True,
                "selected_prefill_chunk": 4096,
                "prefill_chunk_selection": CHUNK_SELECTION,
                "candidates": missing_head_gate,
                "source_provenance": provenance,
            })

        without_chunk_authority = {
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": WORKLOADS,
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": 4096,
            "candidates": rows, "source_provenance": provenance,
        }
        with self.assertRaisesRegex(ValueError, "global prefill-chunk authority"):
            pareto.classify(without_chunk_authority)

        mismatched_chunk = [{**row, "prefill_chunk": 2048} if index == 0 else row
                            for index, row in enumerate(rows)]
        with self.assertRaisesRegex(ValueError, "does not bind selected prefill chunk"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_speed_workloads": WORKLOADS,
                "require_single_static_profile_selection": True,
                "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
                "candidates": mismatched_chunk,
                "source_provenance": provenance,
            })

        duplicate_rows = []
        duplicate_provenance = []
        for row in rows[:4]:
            duplicate = {**row, "name": f"duplicate-{row['name']}"}
            duplicate_rows.append(duplicate)
            duplicate_provenance.append({
                "candidate": duplicate["name"],
                "artifact": {"weights_id": "r9700-q4g64-n16k16-eval",
                             "sha256": "c" * 64,
                             "conversion_receipt": migration_receipt(
                                 "r9700-q4g64-n16k16-eval")},
            })
        with self.assertRaisesRegex(ValueError, "one artifact hash per weights_id"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_speed_workloads": WORKLOADS,
                "require_single_static_profile_selection": True,
                "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
                "candidates": [*rows, *duplicate_rows],
                "source_provenance": [*provenance, *duplicate_provenance],
            })

        fourth_rows = []
        fourth_provenance = []
        for row in rows[:4]:
            fourth = {**row, "name": f"fourth-{row['name']}"}
            fourth_rows.append(fourth)
            fourth_provenance.append({
                "candidate": fourth["name"],
                "artifact": {"weights_id": "unqualified-recipe", "sha256": "c" * 64},
            })
        with self.assertRaisesRegex(ValueError, "unsupported terminal weight recipe"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_speed_workloads": WORKLOADS,
                "require_single_static_profile_selection": True,
                "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
                "candidates": [*rows, *fourth_rows],
                "source_provenance": [*provenance, *fourth_provenance],
            })

    def test_required_four_profile_selection_rejects_declared_recipe(self) -> None:
        rows = [
            candidate(
                f"g{group}-{profile}", mean_nll_delta=0.01, severe_rate=0.0,
                speed=(100.0, 100.0, 300.0), capacity=32768, group=group,
                xattention_profile=profile,
            )
            for group, profile in (
                (16, "dense"), (32, "dense"),
                (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
            )
        ]
        with self.assertRaisesRegex(ValueError, "unsupported terminal weight recipe"):
            pareto.classify({
                "artifact_type": "ninfer_r9700_pareto_input",
                "schema_version": 4,
                "required_speed_workloads": WORKLOADS,
                "require_single_static_profile_selection": True,
                "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
                "candidates": rows,
            })


if __name__ == "__main__":
    unittest.main()
