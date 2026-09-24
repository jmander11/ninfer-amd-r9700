#!/usr/bin/env python3

import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("production_t5_analyze", HERE / "analyze.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(candidate: float = 0.5) -> dict:
    incumbent = [1.0] * 7
    challenger = [candidate] * 7
    return {
        "schema": "ninfer.r9700.a8q4-dflash-mlp-down-small-t-cell.v1",
        "status": "passed" if candidate < 1.0 else "rejected",
        "production_dispatch_changed": False,
        "scope": "standalone Q4 MLP-down N5120/K17408 at exact DFlash T5 only; no production routing",
        "hardware": {"device": "AMD Radeon AI PRO R9700", "architecture": "gfx1201", "pci_bus_id": "0000:13:00.0", "pci_vendor_device": "1002:7551", "integrated": False, "wavefront_width": 32, "power_profile_before_after": "auto"},
        "shape": {"rows": 5120, "columns": 17408, "tokens": 5},
        "numeric": {"oracle": "independent FP64 represented A8G64 x Q4G64 formula", "maximum_bf16_steps_allowed": 2, "candidate_maximum_bf16_steps": 1, "incumbent_maximum_bf16_steps": 1, "pairwise_bit_exact_diagnostic": True},
        "timing": {"method": "unprofiled HIP events", "iterations_per_sample": 10, "forward_reverse_pairs": 7, "incumbent_forward_ms": incumbent, "candidate_forward_ms": challenger, "candidate_reverse_ms": challenger, "incumbent_reverse_ms": incumbent, "incumbent_balanced_ms": incumbent, "candidate_balanced_ms": challenger, "forward_candidate_over_incumbent": challenger, "reverse_candidate_over_incumbent": challenger},
        "decision": {"criterion": MODULE.CRITERION, "incumbent_median_ms": 1.0, "candidate_median_ms": candidate, "candidate_over_incumbent": candidate, "incumbent_forward_median_ms": 1.0, "candidate_forward_median_ms": candidate, "incumbent_reverse_median_ms": 1.0, "candidate_reverse_median_ms": candidate, "forward_ratio_median": candidate, "reverse_ratio_median": candidate, "paired_ratio_mean": candidate, "paired_ratio_standard_deviation": 0.0, "paired_ratio_upper": candidate, "order_ratio_delta": 0.0, "maximum_order_ratio_delta": 0.02, "accepted": candidate < 1.0},
        "executable": {"path": str(MODULE.BINARY), "sha256": MODULE.BINARY_SHA},
        "sources": MODULE.SOURCES,
    }


class AnalyzeTest(unittest.TestCase):
    def test_accept_and_valid_reject(self) -> None:
        self.assertTrue(MODULE.validate(report())["accepted"])
        self.assertFalse(MODULE.validate(report(1.1))["accepted"])

    def test_raw_mutation_rejected(self) -> None:
        value = report()
        value["timing"]["candidate_forward_ms"][0] = 0.7
        with self.assertRaisesRegex(ValueError, "differs"):
            MODULE.validate(value)

    def test_nonfinite_and_bool_rejected(self) -> None:
        for bad in (float("nan"), True):
            value = report()
            value["timing"]["candidate_forward_ms"][0] = bad
            with self.assertRaises(ValueError):
                MODULE.validate(value)

    def test_identity_and_claim_mutations_rejected(self) -> None:
        for path, replacement in (("sources", {}), ("scope", "production"), ("production_dispatch_changed", True)):
            value = report()
            value[path] = replacement
            with self.assertRaises(ValueError):
                MODULE.validate(value)


if __name__ == "__main__":
    unittest.main()
