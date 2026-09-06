import unittest

from tools.r9700.run_a8q4_dflash_small_t_gate import SHAPES, TOKENS, validate_cell


def report(status: str = "passed") -> dict:
    timing = {name: [0.9] * 7 for name in (
        "incumbent_forward_ms", "candidate_forward_ms", "candidate_reverse_ms",
        "incumbent_reverse_ms", "incumbent_balanced_ms", "candidate_balanced_ms",
        "forward_candidate_over_incumbent", "reverse_candidate_over_incumbent")}
    return {
        "schema": "ninfer.r9700.a8q4-dflash-small-t-cell.v1",
        "status": status,
        "production_dispatch_changed": False,
        "shape": {"rows": 4096, "columns": 5120, "tokens": 4},
        "numeric": {"maximum_bf16_steps_allowed": 2,
                    "candidate_maximum_bf16_steps": 1,
                    "incumbent_maximum_bf16_steps": 1},
        "timing": timing,
        "decision": {"accepted": status == "passed"},
    }


class GateTest(unittest.TestCase):
    def test_inventory_is_exact_39_cells(self) -> None:
        self.assertEqual(len(SHAPES), 13)
        self.assertEqual(TOKENS, (4, 5, 6))
        self.assertEqual(len(SHAPES) * len(TOKENS), 39)

    def test_accepts_bound_cell(self) -> None:
        self.assertEqual(validate_cell(report(), 4096, 5120, 4)["status"], "passed")

    def test_rejects_missing_direct_oracle(self) -> None:
        value = report()
        value["numeric"]["candidate_maximum_bf16_steps"] = 3
        with self.assertRaisesRegex(ValueError, "direct oracle"):
            validate_cell(value, 4096, 5120, 4)


if __name__ == "__main__":
    unittest.main()
