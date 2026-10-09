#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_prefill_p129_tail_trace import bf16_float, compare


def trace(role: str) -> dict:
    return {
        "artifact_type": "ninfer_qwen3_text_prefill_p129_tail_trace",
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_evidence_eligible": False,
        "absolute_frontier": 129,
        "prefill_base": 0 if role == "fresh" else 128,
        "prefill_tokens": 129 if role == "fresh" else 1,
        "tail_hidden_kind": "final_rmsnorm_bf16",
        "tail_hidden_bf16_bits": [0x3F80] * 5120,
        "target_logits_kind": "full_lm_head_bf16",
        "token_domain": 248077,
        "top1": {"token": 7, "bf16_bits": 0x4000, "value": bf16_float(0x4000)},
        "top2": {"token": 9, "bf16_bits": 0x3F80, "value": bf16_float(0x3F80)},
        "top1_top2_margin": 1.0,
    }


class CompareTest(unittest.TestCase):
    def test_exact_boundary(self) -> None:
        result = compare(trace("fresh"), trace("append"))
        self.assertEqual(result["classification"], "p129_prefill_boundary_is_exact")
        self.assertTrue(result["tail_hidden"]["exact"])

    def test_reports_hidden_difference_before_logits(self) -> None:
        append = trace("append")
        append["tail_hidden_bf16_bits"][63] = 0x3F81
        result = compare(trace("fresh"), append)
        self.assertEqual(result["classification"], "final_normalized_tail_hidden_differs")
        self.assertEqual(result["tail_hidden"]["first_mismatch"]["index"], 63)
        self.assertEqual(result["tail_hidden"]["maximum_bf16_steps"], 1)

    def test_rejects_inconsistent_margin(self) -> None:
        append = trace("append")
        append["top1_top2_margin"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "margin"):
            compare(trace("fresh"), append)

    def test_rejects_role_swap(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "prefill_base"):
            compare(trace("append"), trace("fresh"))

    def test_does_not_accept_nonfinite_hidden(self) -> None:
        append = copy.deepcopy(trace("append"))
        append["tail_hidden_bf16_bits"][0] = 0x7FC0
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            compare(trace("fresh"), append)


if __name__ == "__main__":
    unittest.main()
