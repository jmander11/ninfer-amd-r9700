"""Host-only tests for the BF16 attention PV determinism probe."""

from __future__ import annotations

import copy
import math
import unittest

from . import pv_determinism_probe as probe

try:
    import torch
except (ImportError, OSError):
    torch = None


class PvDeterminismProbeTest(unittest.TestCase):
    def test_chunks_are_absolute_ascending_and_cover_partial_tail(self) -> None:
        self.assertEqual(probe.source_chunks(8192), ((0, 8192),))
        self.assertEqual(probe.source_chunks(8193), ((0, 8192), (8192, 8193)))
        self.assertEqual(
            probe.source_chunks(32768),
            ((0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)),
        )

    def test_closed_form_denominator_matches_scalar_sum(self) -> None:
        for rows in (1, 250, 251, 252, 8193):
            for query, head in ((0, 0), (7, 2), (31, 5)):
                expected = sum(
                    ((source + query * 17 + head * 29) % 251) + 1
                    for source in range(rows)
                )
                self.assertEqual(probe.probability_denominator(rows, query, head), expected)

    def test_sampled_oracle_uses_represented_fp32_operands(self) -> None:
        rows = 37
        coordinate = (7, 2, 31)
        expected = math.fsum(
            probe.represented_probability(source, rows, 7, 2)
            * probe.represented_value(source, 31)
            for source in range(rows)
        )
        self.assertEqual(probe.sampled_fp64_oracle(rows, coordinate), expected)

    @unittest.skipIf(torch is None, "PyTorch is not installed in the host test interpreter")
    def test_cpu_tensor_routes_match_fp64_samples(self) -> None:
        probabilities, values = probe._make_inputs(torch, 257)
        one_shot = probe.one_shot_pv(torch, probabilities, values)
        fixed = probe.fixed_order_pv(torch, probabilities, values, source_chunk=64)
        self.assertEqual(tuple(fixed.shape), (32, 6, 256))
        self.assertTrue(probe._accuracy(torch, one_shot, 257)["pass"])
        self.assertTrue(probe._accuracy(torch, fixed, 257)["pass"])
        self.assertLessEqual(float((fixed - one_shot).abs().max().item()), 1.0e-6)

    def test_comparison_requires_exact_route_hashes(self) -> None:
        report = self._report()
        comparison = probe.compare_reports(report, copy.deepcopy(report))
        self.assertTrue(comparison["identity_equal"])
        self.assertTrue(comparison["all_routes_exact"])
        changed = copy.deepcopy(report)
        changed["cases"][-1]["routes"]["one-shot-einsum"]["sha256"] = "b" * 64
        comparison = probe.compare_reports(report, changed)
        self.assertFalse(comparison["all_routes_exact"])
        self.assertTrue(
            comparison["cases"][-1]["routes"]["fixed-ascending-chunks"]["exact"]
        )

    @staticmethod
    def _report() -> dict:
        cases = []
        for rows in probe.SOURCE_EXTENTS:
            route = {"sha256": "a" * 64, "finite": True,
                     "accuracy": {"pass": True}}
            cases.append({"source_rows": rows, "input_sha256": "c" * 64,
                          "chunks": [list(value) for value in probe.source_chunks(rows)],
                          "routes": {"one-shot-einsum": copy.deepcopy(route),
                                     "fixed-ascending-chunks": copy.deepcopy(route)}})
        return {"artifact_type": probe.SCHEMA, "schema_version": probe.SCHEMA_VERSION,
                "generator": probe.GENERATOR, "geometry": {}, "fixed_route": {},
                "provenance": {}, "all_accuracy_pass": True, "cases": cases}


if __name__ == "__main__":
    unittest.main()
