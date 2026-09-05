"""Dependency-light contract tests for the R9700 Q4 DFlash2 companion."""

from __future__ import annotations

from collections import Counter
import unittest

from tools.convert.qwen3.common.inventory import TensorSpec
from tools.convert.qwen3_8_27b_r9700 import dflash2_q4_inventory as inventory


class DFlash2Q4InventoryTest(unittest.TestCase):
    def test_exact_fused_and_source_inventory(self) -> None:
        inventory.validate_inventory()
        self.assertEqual(len(inventory.SOURCE_BINDINGS), 66)
        self.assertEqual(len(inventory.SOURCE_NAMES), 81)
        self.assertEqual(len(set(inventory.SOURCE_NAMES)), 81)
        self.assertEqual(
            Counter(spec.format for spec in inventory.TENSOR_SPECS),
            Counter({"BF16": 34, "Q4G64_F16S": 32}),
        )
        self.assertEqual(
            inventory.FORMAT_ENCODED_BYTES,
            {"BF16": 254_814_720, "Q4G64_F16S": 954_654_720},
        )
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 1_209_469_440)

        qkv = [
            binding
            for binding in inventory.SOURCE_BINDINGS
            if binding.artifact.name.endswith("attention/query_key_value")
        ]
        gate_up = [
            binding
            for binding in inventory.SOURCE_BINDINGS
            if binding.artifact.name.endswith("mlp/gate_up")
        ]
        self.assertEqual(len(qkv), 5)
        self.assertTrue(all(len(binding.sources) == 3 for binding in qkv))
        self.assertEqual(len(gate_up), 5)
        self.assertTrue(all(len(binding.sources) == 2 for binding in gate_up))

    def test_three_explicit_base_compositions(self) -> None:
        self.assertEqual(inventory.ALL_Q4_WEIGHTS_ID,
                         "r9700-q4g64-n16k16-dflash2-q4-eval")
        self.assertEqual(inventory.MIXED_WEIGHTS_ID,
                         "r9700-q4-w8-mse-n16k16-dflash2-q4-eval")
        self.assertEqual(
            inventory.HYBRID_WEIGHTS_ID,
            "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval",
        )
        self.assertEqual(len(inventory.ALL_Q4_OBJECT_SPECS), 1190)
        self.assertEqual(len(inventory.MIXED_OBJECT_SPECS), 1190)
        self.assertEqual(len(inventory.HYBRID_OBJECT_SPECS), 1190)
        self.assertEqual(inventory.ALL_Q4_TENSOR_BYTES, 16_369_271_200)
        self.assertEqual(inventory.MIXED_TENSOR_BYTES, 24_077_646_752)
        self.assertEqual(inventory.HYBRID_TENSOR_BYTES, 22_749_987_232)
        self.assertEqual(inventory.ALL_Q4_DEVICE_ARENA_BYTES, 16_369_285_120)
        self.assertEqual(inventory.MIXED_DEVICE_ARENA_BYTES, 24_077_660_672)
        self.assertEqual(inventory.HYBRID_DEVICE_ARENA_BYTES, 22_750_001_152)
        for combined in (
            inventory.ALL_Q4_OBJECT_SPECS,
            inventory.MIXED_OBJECT_SPECS,
            inventory.HYBRID_OBJECT_SPECS,
        ):
            self.assertEqual(
                tuple(spec.name for spec in combined[-66:]),
                tuple(spec.name for spec in inventory.TENSOR_SPECS),
            )
            self.assertTrue(all(isinstance(spec, TensorSpec) for spec in combined[-66:]))


if __name__ == "__main__":
    unittest.main()
