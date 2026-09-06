from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mlp_down_analyze", HERE / "analyze.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def report(accepted: bool) -> dict:
    arm = [1.0] * 7
    candidate_value = 0.9 if accepted else 1.1
    candidate = [candidate_value] * 7
    ratio = [candidate_value] * 7
    return {
        "schema": "ninfer.r9700.a8q4-dflash-mlp-down-small-t-cell.v1",
        "status": "passed" if accepted else "rejected",
        "production_dispatch_changed": False,
        "scope": "standalone Q4 MLP-down N5120/K17408 at exact DFlash T5 only; no production routing",
        "shape": {"rows": 5120, "columns": 17408, "tokens": 5},
        "hardware": {"device": "AMD Radeon AI PRO R9700", "architecture": "gfx1201", "pci_bus_id": "0000:03:00.0", "pci_vendor_device": "1002:7551", "integrated": False, "wavefront_width": 32, "power_profile_before_after": "auto"},
        "numeric": {"oracle": "independent FP64 represented A8G64 x Q4G64 formula", "maximum_bf16_steps_allowed": 2, "candidate_maximum_bf16_steps": 2, "incumbent_maximum_bf16_steps": 2, "pairwise_bit_exact_diagnostic": True},
        "timing": {"method": "unprofiled HIP events", "iterations_per_sample": 10, "forward_reverse_pairs": 7, "incumbent_forward_ms": arm, "candidate_forward_ms": candidate, "candidate_reverse_ms": candidate, "incumbent_reverse_ms": arm, "incumbent_balanced_ms": arm, "candidate_balanced_ms": candidate, "forward_candidate_over_incumbent": ratio, "reverse_candidate_over_incumbent": ratio},
        "decision": {"criterion": "both launch-order medians faster, two-standard-error paired ratio upper below one, order ratio delta at most 0.02", "accepted": accepted, "incumbent_median_ms": 1.0, "candidate_median_ms": candidate_value, "candidate_over_incumbent": candidate_value, "incumbent_forward_median_ms": 1.0, "candidate_forward_median_ms": candidate_value, "incumbent_reverse_median_ms": 1.0, "candidate_reverse_median_ms": candidate_value, "forward_ratio_median": candidate_value, "reverse_ratio_median": candidate_value, "paired_ratio_mean": candidate_value, "paired_ratio_standard_deviation": 0.0, "paired_ratio_upper": candidate_value, "order_ratio_delta": 0.0, "maximum_order_ratio_delta": 0.02},
        "executable": {"path": str(MODULE.EXPECTED_BINARY), "sha256": MODULE.EXPECTED_BINARY_SHA},
        "sources": {"kernel": "692471102a3c31e0a71e923c6fd5044717bef3afa8ef60a7cbcc5a8d22bd0a52", "contract": "41bf31711813bfc5919961447dd540ed3f009d99a5c665826b555e995ab03a51", "qualifier": "fd2f93b3f0214c50d926e94f5e5e10928d00fc40e24f1b873214cc0ae0d6ec1a", "shared_harness": "69af9a07e18bf331a175dfae6c02fb12db71baebad5e63552bf5744116612486"},
    }


class AnalyzeTest(unittest.TestCase):
    def run_case(self, accepted: bool) -> int:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            MODULE.PACKAGE = package
            (package / "cell-t5.json").write_text(json.dumps(report(accepted)))
            prior = sys.argv
            try:
                sys.argv = ["analyze.py", "--report", str(package / "cell-t5.json"), "--summary", str(package / "summary.json")]
                status = MODULE.main()
            finally:
                sys.argv = prior
            summary = json.loads((package / "summary.json").read_text())
            self.assertFalse(summary["production_eligible"])
            self.assertIn("prior-static-rejection", summary["t6"])
            return status

    def test_accept_and_reject_are_published(self) -> None:
        self.assertEqual(self.run_case(True), 0)
        self.assertEqual(self.run_case(False), 1)

    def test_production_claim_is_rejected(self) -> None:
        value = report(True)
        value["production_dispatch_changed"] = True
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            MODULE.PACKAGE = package
            (package / "cell-t5.json").write_text(json.dumps(value))
            prior = sys.argv
            try:
                sys.argv = ["analyze.py", "--report", str(package / "cell-t5.json"), "--summary", str(package / "summary.json")]
                with self.assertRaisesRegex(ValueError, "production isolation"):
                    MODULE.main()
            finally:
                sys.argv = prior

    def test_raw_timing_mutation_is_rejected(self) -> None:
        value = report(True)
        value["timing"]["candidate_forward_ms"][0] = 0.8
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            MODULE.PACKAGE = package
            (package / "cell-t5.json").write_text(json.dumps(value))
            prior = sys.argv
            try:
                sys.argv = ["analyze.py", "--report", str(package / "cell-t5.json"), "--summary", str(package / "summary.json")]
                with self.assertRaisesRegex(ValueError, "raw timing arms"):
                    MODULE.main()
            finally:
                sys.argv = prior

    def test_nonfinite_raw_timing_is_rejected(self) -> None:
        value = report(True)
        value["timing"]["candidate_forward_ms"][0] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            MODULE.PACKAGE = package
            (package / "cell-t5.json").write_text(json.dumps(value))
            prior = sys.argv
            try:
                sys.argv = ["analyze.py", "--report", str(package / "cell-t5.json"), "--summary", str(package / "summary.json")]
                with self.assertRaises(ValueError):
                    MODULE.main()
            finally:
                sys.argv = prior

    def test_decision_criterion_mutation_is_rejected(self) -> None:
        value = report(True)
        value["decision"]["criterion"] = "trust me"
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            MODULE.PACKAGE = package
            (package / "cell-t5.json").write_text(json.dumps(value))
            prior = sys.argv
            try:
                sys.argv = ["analyze.py", "--report", str(package / "cell-t5.json"), "--summary", str(package / "summary.json")]
                with self.assertRaisesRegex(ValueError, "criterion"):
                    MODULE.main()
            finally:
                sys.argv = prior


if __name__ == "__main__":
    unittest.main()
