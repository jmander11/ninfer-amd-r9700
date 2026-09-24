from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ppl import compare_selective_a8q4_source as comparison
from tools.ppl.compare_q4_group_source import _against


class CompareSelectiveA8Q4SourceTest(unittest.TestCase):
    def test_stale_qkv_and_raw_inventory_scope_rejects(self) -> None:
        scope = comparison._scope()
        report = {"matrix_scope": scope}
        comparison._validate_exact_scope(report, Path("valid.json"))
        stale = dict(scope)
        stale["selected_raw_source_row_count"] -= 6144
        stale["selected_raw_source_tensor_count"] -= 48
        report["matrix_scope"] = stale
        with self.assertRaisesRegex(ValueError, "selected-row scope differs"):
            comparison._validate_exact_scope(report, Path("stale-qkv.json"))

    @staticmethod
    def _report(profile: str, nlls, argmax, bf16_nll, bf16_argmax) -> dict:
        return {
            "profile": profile,
            "source": {"config_sha256": "a", "index_sha256": "b",
                       "shards_sha256": {"s": "c"}, "corpus_ids_sha256": "d"},
            "execution": {"profile": "same"},
            "matrix_scope": {"selected_logical_matrix_count": 160},
            "sampled_source_gate": {"sha256": "e"},
            "against_bf16": _against(bf16_nll, nlls, bf16_argmax, argmax),
        }

    def test_candidate_absolute_gate_and_pair_are_both_retained(self) -> None:
        bf16_nll = [1.0] * 4095
        control_nll = [1.01] * 4095
        candidate_nll = [1.02] * 4095
        argmax = list(range(4095))
        control = self._report(
            "a8g64-q4g64-control", control_nll, argmax, bf16_nll, argmax
        )
        candidate = self._report(
            "a8g128-q4g128-mse", candidate_nll, argmax, bf16_nll, argmax
        )
        bf16 = {"source_config_sha256": "a", "source_index_sha256": "b",
                "source_shards_sha256": {"s": "c"}, "corpus_ids_sha256": "d"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            comparison, "_load_score",
            side_effect=[(control, control_nll, argmax), (candidate, candidate_nll, argmax)],
        ), patch.object(comparison, "_load_bf16", return_value=(bf16, bf16_nll, argmax)), \
                patch.object(comparison, "sha256_file", return_value="f" * 64):
            root = Path(temporary)
            report = comparison.compare(root / "control", root / "candidate", root / "bf16")
        self.assertTrue(report["pass"])
        self.assertEqual(report["acceptance"]["maximum_new_severe_positions"], 11)
        self.assertTrue(
            report["acceptance"]["control_and_candidate_must_pass_direct_bf16_gate"]
        )
        self.assertIn("candidate_minus_control", report)


if __name__ == "__main__":
    unittest.main()
