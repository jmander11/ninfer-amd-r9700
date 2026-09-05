"""Focused CPU tests for the decision-owned four-role FP8/Q4 inventory."""

from __future__ import annotations

from collections import Counter
import unittest

from . import fp8_hybrid_decision, fp8_hybrid_inventory, q4_inventory


class Fp8HybridInventoryTest(unittest.TestCase):
    def test_exact_decision_set_replaces_one_q4_object_plane_per_name(self) -> None:
        fp8_hybrid_inventory.validate_inventory()
        decision = fp8_hybrid_decision.load_decision()
        self.assertEqual(decision.matrix_names, fp8_hybrid_decision.DECISION.matrix_names)
        self.assertEqual(len(decision.matrix_names), 144)
        self.assertEqual(
            decision.selection_sha256,
            "b2ceeb63c581c0f26aab5a4d8c0958da34d836fcc5c47d377bce709eaf37e3e8",
        )
        selected = set(decision.matrix_names)
        represented_names: list[str] = []
        for base, hybrid in zip(
            q4_inventory.TENSOR_SPECS,
            fp8_hybrid_inventory.TENSOR_SPECS,
            strict=True,
        ):
            self.assertEqual((hybrid.name, hybrid.shape), (base.name, base.shape))
            if base.name in selected:
                self.assertEqual(base.format, "Q4G64_F16S")
                self.assertEqual(hybrid.format, "F8E4M3_ROW_F32S")
                self.assertEqual(hybrid.layout, "row-scaled-k128-v1")
                represented_names.append(hybrid.name)
            else:
                self.assertEqual(hybrid, base)
        self.assertEqual(set(represented_names), selected)
        self.assertEqual(len(represented_names), len(set(represented_names)))

    def test_four_roles_counts_and_fixed_capacity(self) -> None:
        roles = Counter(name.split("/", 3)[3] for name in fp8_hybrid_decision.DECISION.matrix_names)
        self.assertEqual(
            roles,
            {
                "mlp/gate_up": 64,
                "attention/query_key": 16,
                "attention/gate_value": 16,
                "gdn/query_key": 48,
            },
        )
        self.assertEqual(
            fp8_hybrid_inventory.FORMAT_COUNTS,
            {
                "BF16": 582,
                "FP32": 96,
                "I32": 1,
                "Q4G64_F16S": 295,
                "F8E4M3_ROW_F32S": 144,
            },
        )
        self.assertEqual(fp8_hybrid_inventory.TENSOR_ENCODED_BYTES, 21_540_517_792)
        self.assertEqual(fp8_hybrid_inventory.DEVICE_ARENA_BYTES, 21_540_531_712)


if __name__ == "__main__":
    unittest.main()
