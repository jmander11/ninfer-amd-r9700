from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ppl import compare_selective_q4g128_source as comparison
from tools.ppl.selective_q4g128_source_diagnostic import _gate


class CompareSelectiveQ4G128SourceTest(unittest.TestCase):
    @staticmethod
    def _report(codec: str) -> dict:
        return {
            "codec_profile": codec,
            "source": {"config_sha256": "a", "index_sha256": "b",
                       "shards_sha256": {"s": "c"}, "corpus_ids_sha256": "d"},
            "workload": {"tokens": 8192, "skip": "half", "prefill_chunk": 4096,
                         "schedule": "prefill", "device": 0},
            "execution": {"profile": "same"},
            "matrix_scope": {"selected_logical_matrix_count": 160},
            "sampled_source_gate": {"sha256": "e"},
        }

    def test_comparison_gates_absolute_quality_and_reports_pair(self) -> None:
        bf16_nll = [1.0] * 4095
        bf16_arg = list(range(4095))
        control_nll = [1.01] * 4095
        candidate_nll = [1.02] * 4095
        control = self._report("q4g64-absmax")
        candidate = self._report("q4g128-mse")
        control["quality_gate"] = _gate(control_nll, bf16_arg, bf16_nll, bf16_arg)
        candidate["quality_gate"] = _gate(candidate_nll, bf16_arg, bf16_nll, bf16_arg)
        bf16 = {
            "source_config_sha256": "a", "source_index_sha256": "b",
            "source_shards_sha256": {"s": "c"}, "corpus_ids_sha256": "d",
        }
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            comparison, "_load_score",
            side_effect=[(control, control_nll, bf16_arg),
                         (candidate, candidate_nll, bf16_arg)],
        ), patch.object(comparison, "_load_bf16", return_value=(bf16, bf16_nll, bf16_arg)), \
                patch.object(comparison, "sha256_file", return_value="f" * 64):
            root = Path(temporary)
            result = comparison.compare(root / "g64.json", root / "g128.json",
                                        root / "bf16.json")
        self.assertTrue(result["pass"])
        self.assertIn("q4g128_minus_q4g64", result)
        self.assertTrue(result["acceptance"]["paired_candidate_minus_control_is_diagnostic"])


if __name__ == "__main__":
    unittest.main()
