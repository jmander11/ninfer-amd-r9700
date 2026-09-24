"""Dependency-light exact plan tests for explicit R9700 evaluation profiles."""

from __future__ import annotations

from collections import Counter
import unittest

from tools.artifact.container import (
    ResourceSpec as ArtifactResourceSpec,
    TensorObject,
    TensorSpec as ArtifactTensorSpec,
    plan_objects,
)
from tools.convert.qwen3_8_27b_r9700 import (
    q4_inventory,
    q4_w8_inventory,
    q4_w8_mse_inventory,
    source_inventory,
    w8_mse_inventory,
    w8_bf16_attention_qk_inventory,
    w8_bf16_attention_vo_inventory,
    w8_bf16_embedding_inventory,
    w8_bf16_gdn_qk_inventory,
)


def _synthetic_plan(inventory):
    specs = tuple(
        ArtifactTensorSpec(spec.name, spec.shape, spec.format, spec.layout)
        if isinstance(spec, inventory.TensorSpec)
        else ArtifactResourceSpec(spec.name, spec.encoding, 1)
        for spec in inventory.OBJECT_SPECS
    )
    return plan_objects(specs)


class LowBitEvaluationPlanTest(unittest.TestCase):
    def check_common_plan(self, inventory, *, payload_span: int) -> None:
        inventory.validate_inventory()
        self.assertEqual(len(inventory.TENSOR_SPECS), 1118)
        self.assertEqual(len(inventory.OBJECT_SPECS), 1124)
        self.assertEqual(
            tuple(spec.name for spec in inventory.TENSOR_SPECS),
            tuple(spec.name for spec in source_inventory.TENSOR_SPECS),
        )
        self.assertEqual(
            tuple(spec.shape for spec in inventory.TENSOR_SPECS),
            tuple(spec.shape for spec in source_inventory.TENSOR_SPECS),
        )
        objects = _synthetic_plan(inventory)
        self.assertEqual(len(objects), 1124)
        self.assertEqual(len({obj.name for obj in objects}), 1124)
        self.assertEqual(
            Counter(obj.format for obj in objects if isinstance(obj, TensorObject)),
            Counter(inventory.FORMAT_COUNTS),
        )
        self.assertEqual(
            sum(obj.bytes for obj in objects if isinstance(obj, TensorObject)),
            inventory.TENSOR_ENCODED_BYTES,
        )
        self.assertEqual(objects[-1].offset + objects[-1].bytes, payload_span)

    def test_all_q4_identity_counts_and_exact_plan(self) -> None:
        self.assertEqual(q4_inventory.WEIGHTS_ID, "r9700-q4g64-n16k16-eval")
        self.assertEqual(
            q4_inventory.FORMAT_COUNTS,
            {"BF16": 582, "FP32": 96, "I32": 1, "Q4G64_F16S": 439},
        )
        self.assertEqual(q4_inventory.TENSOR_ENCODED_BYTES, 15_159_801_760)
        self.assertEqual(q4_inventory.DEVICE_ARENA_BYTES, 15_159_815_680)
        for source, candidate in zip(
            source_inventory.TENSOR_SPECS, q4_inventory.TENSOR_SPECS, strict=True
        ):
            expected = (
                source.format
                if source.format in q4_inventory.DIRECT_FORMATS
                else "Q4G64_F16S"
            )
            self.assertEqual(candidate.format, expected)
        self.check_common_plan(q4_inventory, payload_span=15_159_815_936)

    def test_w8_mse_identity_preserves_the_complete_w8_plan(self) -> None:
        inventory = w8_mse_inventory
        self.assertEqual(inventory.WEIGHTS_ID, "r9700-w8g32-mse-eval")
        self.assertEqual(
            inventory.FORMAT_COUNTS,
            {"BF16": 582, "FP32": 96, "I32": 1, "W8G32_F16S": 439},
        )
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 30_260_413_792)
        self.assertEqual(inventory.DEVICE_ARENA_BYTES, 30_260_425_984)
        self.assertEqual(inventory.TENSOR_SPECS, w8_mse_inventory.inventory.TENSOR_SPECS)
        self.check_common_plan(inventory, payload_span=30_260_426_240)

    def test_q4_w8_identity_counts_and_exact_plan(self) -> None:
        self.assertEqual(q4_w8_inventory.WEIGHTS_ID, "r9700-q4-w8-n16k16-eval")
        self.assertEqual(
            q4_w8_inventory.FORMAT_COUNTS,
            {
                "BF16": 582,
                "FP32": 96,
                "I32": 1,
                "Q4G64_F16S": 183,
                "W8G32_F16S": 256,
            },
        )
        self.assertEqual(q4_w8_inventory.TENSOR_ENCODED_BYTES, 22_868_177_312)
        self.assertEqual(q4_w8_inventory.DEVICE_ARENA_BYTES, 22_868_191_232)
        for source, candidate in zip(
            source_inventory.TENSOR_SPECS,
            q4_w8_inventory.TENSOR_SPECS,
            strict=True,
        ):
            expected = source.format
            if source.format not in q4_w8_inventory.DIRECT_FORMATS:
                expected = "Q4G64_F16S" if source.format == "Q4G64_F16S" else "W8G32_F16S"
            self.assertEqual(candidate.format, expected)
        self.check_common_plan(q4_w8_inventory, payload_span=22_868_191_488)

    def test_q4_w8_mse_identity_preserves_the_complete_q4_w8_plan(self) -> None:
        inventory = q4_w8_mse_inventory
        self.assertEqual(inventory.WEIGHTS_ID, "r9700-q4-w8-mse-n16k16-eval")
        self.assertEqual(
            inventory.FORMAT_COUNTS,
            {
                "BF16": 582,
                "FP32": 96,
                "I32": 1,
                "Q4G64_F16S": 183,
                "W8G32_F16S": 256,
            },
        )
        self.assertEqual(inventory.FORMAT_ENCODED_BYTES, q4_w8_inventory.FORMAT_ENCODED_BYTES)
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 22_868_177_312)
        self.assertEqual(inventory.DEVICE_ARENA_BYTES, 22_868_191_232)
        self.assertEqual(inventory.OBJECT_SPECS, q4_w8_inventory.OBJECT_SPECS)
        self.check_common_plan(inventory, payload_span=22_868_191_488)

    def test_w8_bf16_embedding_identity_counts_and_exact_plan(self) -> None:
        inventory = w8_bf16_embedding_inventory
        self.assertEqual(inventory.WEIGHTS_ID, "r9700-w8-bf16-embed-eval")
        self.assertEqual(
            inventory.FORMAT_COUNTS,
            {"BF16": 583, "FP32": 96, "I32": 1, "W8G32_F16S": 438},
        )
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 31_452_349_792)
        self.assertEqual(inventory.DEVICE_ARENA_BYTES, 31_452_361_984)
        for source, candidate in zip(
            source_inventory.TENSOR_SPECS,
            inventory.TENSOR_SPECS,
            strict=True,
        ):
            if source.name == "text/token_embedding":
                expected = "BF16"
            else:
                expected = (
                    source.format
                    if source.format in inventory.DIRECT_FORMATS
                    else "W8G32_F16S"
                )
            self.assertEqual(candidate.format, expected)
        self.check_common_plan(inventory, payload_span=31_452_362_240)

    def test_w8_bf16_attention_vo_identity_counts_and_exact_plan(self) -> None:
        inventory = w8_bf16_attention_vo_inventory
        self.assertEqual(inventory.WEIGHTS_ID, "r9700-w8-bf16-attn-vo-eval")
        self.assertEqual(
            inventory.FORMAT_COUNTS,
            {"BF16": 614, "FP32": 96, "I32": 1, "W8G32_F16S": 407},
        )
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 31_282_775_392)
        self.assertEqual(inventory.DEVICE_ARENA_BYTES, 31_282_787_584)
        promoted = 0
        for source, candidate in zip(
            source_inventory.TENSOR_SPECS,
            inventory.TENSOR_SPECS,
            strict=True,
        ):
            full_attention_vo = any(
                source.name == f"text/layers/{layer}/{role}"
                for layer in source_inventory.FULL_ATTENTION_LAYERS
                for role in ("attention/gate_value", "attention/output")
            )
            if full_attention_vo:
                expected = "BF16"
                promoted += 1
            else:
                expected = (
                    source.format
                    if source.format in inventory.DIRECT_FORMATS
                    else "W8G32_F16S"
                )
            self.assertEqual(candidate.format, expected)
        self.assertEqual(promoted, 32)
        self.check_common_plan(inventory, payload_span=31_282_787_840)

    def test_w8_bf16_attention_qk_identity_counts_and_exact_plan(self) -> None:
        inventory = w8_bf16_attention_qk_inventory
        self.assertEqual(inventory.WEIGHTS_ID, "r9700-w8-bf16-attn-qk-eval")
        self.assertEqual(
            inventory.FORMAT_COUNTS,
            {"BF16": 598, "FP32": 96, "I32": 1, "W8G32_F16S": 423},
        )
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 30_810_916_192)
        self.assertEqual(inventory.DEVICE_ARENA_BYTES, 30_810_928_384)
        promoted = 0
        for source, candidate in zip(
            source_inventory.TENSOR_SPECS,
            inventory.TENSOR_SPECS,
            strict=True,
        ):
            attention_qk = any(
                source.name == f"text/layers/{layer}/attention/query_key"
                for layer in source_inventory.FULL_ATTENTION_LAYERS
            )
            if attention_qk:
                expected = "BF16"
                promoted += 1
            else:
                expected = (
                    source.format
                    if source.format in inventory.DIRECT_FORMATS
                    else "W8G32_F16S"
                )
            self.assertEqual(candidate.format, expected)
        self.assertEqual(promoted, 16)
        self.check_common_plan(inventory, payload_span=30_810_928_640)

    def test_w8_bf16_gdn_qk_identity_counts_and_exact_plan(self) -> None:
        inventory = w8_bf16_gdn_qk_inventory
        self.assertEqual(inventory.WEIGHTS_ID, "r9700-w8-bf16-gdn-qk-eval")
        self.assertEqual(
            inventory.FORMAT_COUNTS,
            {"BF16": 630, "FP32": 96, "I32": 1, "W8G32_F16S": 391},
        )
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 31_204_132_192)
        self.assertEqual(inventory.DEVICE_ARENA_BYTES, 31_204_144_384)
        promoted = 0
        for source, candidate in zip(
            source_inventory.TENSOR_SPECS,
            inventory.TENSOR_SPECS,
            strict=True,
        ):
            gdn_qk = any(
                source.name == f"text/layers/{layer}/gdn/query_key"
                for layer in source_inventory.GDN_LAYERS
            )
            if gdn_qk:
                expected = "BF16"
                promoted += 1
            else:
                expected = (
                    source.format
                    if source.format in inventory.DIRECT_FORMATS
                    else "W8G32_F16S"
                )
            self.assertEqual(candidate.format, expected)
        self.assertEqual(promoted, 48)
        self.check_common_plan(inventory, payload_span=31_204_144_640)


if __name__ == "__main__":
    unittest.main()
