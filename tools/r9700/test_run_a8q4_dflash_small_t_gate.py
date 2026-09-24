import copy
import math
import statistics
import unittest

from tools.r9700.run_a8q4_dflash_small_t_gate import (
    CELL_SCHEMA, CRITERION, ORACLE, PRIOR_ELIGIBLE, SHAPES, SCOPE, TOKENS,
    validate_cell, validate_prior_summary,
)

EXECUTABLE = {"path": "/qualified/binary", "sha256": "a" * 64}
SOURCES = {"kernel": "b" * 64, "contract": "c" * 64, "qualifier": "d" * 64}


def report(candidate: float = 0.8, incumbent: float = 1.0) -> dict:
    incumbent_forward = [incumbent + index * 0.001 for index in range(7)]
    incumbent_reverse = [incumbent + 0.002 + index * 0.001 for index in range(7)]
    candidate_forward = [candidate + index * 0.001 for index in range(7)]
    candidate_reverse = [candidate + 0.002 + index * 0.001 for index in range(7)]
    incumbent_balanced = [(a + b) * 0.5 for a, b in zip(
        incumbent_forward, incumbent_reverse)]
    candidate_balanced = [(a + b) * 0.5 for a, b in zip(
        candidate_forward, candidate_reverse)]
    forward_ratios = [a / b for a, b in zip(candidate_forward, incumbent_forward)]
    reverse_ratios = [a / b for a, b in zip(candidate_reverse, incumbent_reverse)]
    ratios = forward_ratios + reverse_ratios
    forward_median = statistics.median(forward_ratios)
    reverse_median = statistics.median(reverse_ratios)
    ratio_mean = statistics.mean(ratios)
    ratio_sd = statistics.stdev(ratios)
    ratio_upper = ratio_mean + 2.0 * ratio_sd / math.sqrt(14)
    order_ratio_delta = abs(forward_median - reverse_median)
    accepted = (
        statistics.median(candidate_balanced) < statistics.median(incumbent_balanced) and
        statistics.median(candidate_forward) < statistics.median(incumbent_forward) and
        statistics.median(candidate_reverse) < statistics.median(incumbent_reverse) and
        forward_median < 1.0 and reverse_median < 1.0 and ratio_upper < 1.0 and
        order_ratio_delta <= 0.02
    )
    decision = {
        "criterion": CRITERION,
        "incumbent_median_ms": statistics.median(incumbent_balanced),
        "candidate_median_ms": statistics.median(candidate_balanced),
        "candidate_over_incumbent": (
            statistics.median(candidate_balanced) / statistics.median(incumbent_balanced)
        ),
        "incumbent_forward_median_ms": statistics.median(incumbent_forward),
        "candidate_forward_median_ms": statistics.median(candidate_forward),
        "incumbent_reverse_median_ms": statistics.median(incumbent_reverse),
        "candidate_reverse_median_ms": statistics.median(candidate_reverse),
        "forward_ratio_median": forward_median,
        "reverse_ratio_median": reverse_median,
        "paired_ratio_mean": ratio_mean,
        "paired_ratio_standard_deviation": ratio_sd,
        "paired_ratio_upper": ratio_upper,
        "order_ratio_delta": order_ratio_delta,
        "maximum_order_ratio_delta": 0.02,
        "accepted": accepted,
    }
    return {
        "schema": CELL_SCHEMA,
        "status": "passed" if accepted else "rejected",
        "production_dispatch_changed": False,
        "scope": SCOPE,
        "hardware": {
            "device": "AMD Radeon AI PRO R9700", "architecture": "gfx1201",
            "pci_bus_id": "0000:03:00.0", "pci_vendor_device": "1002:7551",
            "integrated": False, "wavefront_width": 32,
            "power_profile_before_after": "auto",
        },
        "shape": {"rows": 34816, "columns": 5120, "tokens": 8},
        "numeric": {
            "oracle": ORACLE, "maximum_bf16_steps_allowed": 2,
            "candidate_maximum_bf16_steps": 1, "incumbent_maximum_bf16_steps": 1,
            "pairwise_bit_exact_diagnostic": False,
        },
        "timing": {
            "method": "unprofiled HIP events", "iterations_per_sample": 10,
            "forward_reverse_pairs": 7,
            "incumbent_forward_ms": incumbent_forward,
            "candidate_forward_ms": candidate_forward,
            "candidate_reverse_ms": candidate_reverse,
            "incumbent_reverse_ms": incumbent_reverse,
            "incumbent_balanced_ms": incumbent_balanced,
            "candidate_balanced_ms": candidate_balanced,
            "forward_candidate_over_incumbent": forward_ratios,
            "reverse_candidate_over_incumbent": reverse_ratios,
        },
        "decision": decision,
        "executable": EXECUTABLE,
        "sources": SOURCES,
    }


class GateTest(unittest.TestCase):
    def validate(self, value: dict) -> dict:
        return validate_cell(value, 34816, 5120, 8, EXECUTABLE, SOURCES)

    def test_inventory_is_exact_unresolved_flattened_union(self) -> None:
        self.assertEqual(SHAPES, ((34816, 5120),))
        self.assertEqual(TOKENS, (8, 10, 12, 15, 16, 18, 20, 24))
        self.assertEqual(len(SHAPES) * len(TOKENS), 8)

    def test_accepts_independently_recomputed_cell(self) -> None:
        self.assertTrue(self.validate(report())["eligible"])

    def test_retains_losing_cell_as_ineligible(self) -> None:
        value = report(candidate=1.2)
        self.assertEqual(value["status"], "rejected")
        self.assertFalse(self.validate(value)["eligible"])

    def test_rejects_missing_direct_oracle(self) -> None:
        value = report()
        value["numeric"]["candidate_maximum_bf16_steps"] = 3
        with self.assertRaisesRegex(ValueError, "direct oracle"):
            self.validate(value)

    def test_rejects_mutated_balanced_sample(self) -> None:
        value = report()
        value["timing"]["candidate_balanced_ms"][2] += 0.01
        with self.assertRaisesRegex(ValueError, "inconsistent candidate_balanced"):
            self.validate(value)

    def test_rejects_mutated_decision(self) -> None:
        value = report()
        value["decision"]["paired_ratio_upper"] += 0.01
        with self.assertRaisesRegex(ValueError, "decision.paired_ratio_upper"):
            self.validate(value)

    def test_rejects_nonfinite_raw_or_decision(self) -> None:
        for section, name in (("timing", "candidate_forward_ms"),
                              ("decision", "candidate_over_incumbent")):
            with self.subTest(section=section):
                value = report()
                if section == "timing":
                    value[section][name][0] = float("nan")
                else:
                    value[section][name] = float("inf")
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.validate(value)

    def test_rejects_wrong_identity_or_scope(self) -> None:
        for mutation in ("scope", "hardware", "executable", "sources"):
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(report())
                if mutation == "scope":
                    value["scope"] = "broader"
                elif mutation == "hardware":
                    value["hardware"]["device"] = "other"
                elif mutation == "executable":
                    value["executable"]["sha256"] = "0" * 64
                else:
                    value["sources"]["kernel"] = "0" * 64
                with self.assertRaises(ValueError):
                    self.validate(value)

    def test_prior_summary_requires_exact_retained_eligibility(self) -> None:
        eligible = [
            {"rows": rows, "columns": columns, "tokens": tokens}
            for rows, columns, tokens in PRIOR_ELIGIBLE
        ]
        value = {
            "schema": "ninfer.r9700.a8q4-dflash-small-t-gate.v2",
            "status": "passed", "complete_screen": True, "routing_authorized": False,
            "required_cell_count": 39, "eligible_cells": eligible,
            "forbidden_cells": [{}] * 36, "cells": [{"evidence_valid": True}] * 39,
        }
        validate_prior_summary(value)
        value["eligible_cells"] = eligible[:-1]
        with self.assertRaisesRegex(ValueError, "eligibility"):
            validate_prior_summary(value)


if __name__ == "__main__":
    unittest.main()
