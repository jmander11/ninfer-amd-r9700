"""Exact synthetic tests for BF16-source DFlash2 matrix recipes."""

from __future__ import annotations

import hashlib
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from tools.convert.qwen3.common.inventory import BF16, Q4, W8, tensor_spec
from tools.convert.qwen3_8_27b_r9700 import dflash2_matrix_recipes as recipes
from tools.convert.qwen3_8_27b_r9700 import dflash2_q4_inventory as inventory


class DFlash2MatrixRecipesTest(unittest.TestCase):
    def test_exact_candidate_plans_preserve_all_nonmatrices(self) -> None:
        expected = {
            recipes.CANONICAL_Q4G64: (Q4, 954_654_720, 1_209_469_440),
            recipes.SOURCE_MSE_Q4G64: (Q4, 954_654_720, 1_209_469_440),
            recipes.SOURCE_MSE_W8G32: (W8, 1_909_309_440, 2_164_124_160),
        }
        canonical = inventory.source_bindings_for_recipe(recipes.CANONICAL_Q4G64)
        for key, (matrix_format, matrix_bytes, total_bytes) in expected.items():
            bindings = inventory.source_bindings_for_recipe(key)
            plan = inventory.matrix_recipe_summary(key)
            self.assertEqual(plan["matrix_format"], matrix_format)
            self.assertEqual(plan["format_counts"], {BF16: 34, matrix_format: 32})
            self.assertEqual(plan["format_encoded_bytes"][matrix_format], matrix_bytes)
            self.assertEqual(plan["tensor_encoded_bytes"], total_bytes)
            if key == recipes.CANONICAL_Q4G64:
                self.assertEqual(
                    plan["selection_status"],
                    "registered-evaluation-control-with-bound-artifact-identities",
                )
                self.assertEqual(plan["materialization"], "existing-canonical-converter-route")
            else:
                self.assertEqual(
                    plan["selection_status"], "candidate-only-no-artifact-identity"
                )
                self.assertEqual(plan["materialization"], "disabled-until-base-selection")
            self.assertEqual(
                tuple((row.artifact.name, row.artifact.shape, row.sources) for row in bindings),
                tuple((row.artifact.name, row.artifact.shape, row.sources) for row in canonical),
            )
            for baseline, candidate in zip(canonical, bindings, strict=True):
                if baseline.artifact.format == BF16:
                    self.assertEqual(candidate.artifact, baseline.artifact)

            codebooks = [
                row.artifact for row in bindings if row.artifact.name.endswith("codebook")
            ]
            self.assertEqual(len(codebooks), 2)
            self.assertTrue(all(spec.format == BF16 for spec in codebooks))

    def test_independent_scalar_payloads_fix_exact_layout_and_objective(self) -> None:
        values = [((index * 37) % 257 - 128) / 19 for index in range(16 * 64)]
        expected = {
            recipes.CANONICAL_Q4G64: (
                1088,
                "945020d37e9c8f0ac83d45fe983c67ae8834e780b93b23bc0f71c2aca3dbfc1c",
            ),
            recipes.SOURCE_MSE_Q4G64: (
                1088,
                "1076e2d9f4d4a833aaf5ade0ae9cdf90c80de2eab17ecc4a18f69a02aca17fe0",
            ),
            recipes.SOURCE_MSE_W8G32: (
                2176,
                "db488754464590d79c3b0ceeb5101b0e071ad24d1a0b2205cdc6df7863bf1f0f",
            ),
        }
        for key, (size, digest) in expected.items():
            payload = recipes.encode_matrix_reference(values, 16, 64, key)
            self.assertEqual(len(payload), size)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_production_dispatch_is_recipe_exact(self) -> None:
        class RepresentedBf16Matrix:
            dtype = "torch.bfloat16"
            shape = (16, 64)

        tensor = RepresentedBf16Matrix()
        q4 = tensor_spec("matrix", (16, 64), Q4)
        w8 = tensor_spec("matrix", (16, 64), W8)
        family_conversion = ModuleType("tools.convert.qwen3.common.conversion")
        canonical = Mock(return_value=b"canonical")
        family_conversion.encode_tensor_payload = canonical
        mse_quantize = ModuleType("tools.convert.qwen3_8_27b_r9700.mse_quantize")
        q4_mse = Mock(return_value=b"q4-mse")
        w8_mse = Mock(return_value=b"w8-mse")
        mse_quantize.quantize_and_encode_q4g64_mse = q4_mse
        mse_quantize.quantize_and_encode_mse = w8_mse
        with patch.dict(
            sys.modules,
            {
                family_conversion.__name__: family_conversion,
                mse_quantize.__name__: mse_quantize,
            },
        ):
            self.assertEqual(
                recipes.encode_matrix_payload(
                    tensor, q4, recipes.CANONICAL_Q4G64, "cpu"
                ),
                b"canonical",
            )
            canonical.assert_called_once_with(tensor, q4, "cpu")
            self.assertEqual(
                recipes.encode_matrix_payload(
                    tensor, q4, recipes.SOURCE_MSE_Q4G64, "cpu"
                ),
                b"q4-mse",
            )
            q4_mse.assert_called_once_with(tensor, device="cpu")
            self.assertEqual(
                recipes.encode_matrix_payload(
                    tensor, w8, recipes.SOURCE_MSE_W8G32, "cpu"
                ),
                b"w8-mse",
            )
            w8_mse.assert_called_once_with(tensor, device="cpu")

        with self.assertRaisesRegex(ValueError, "requires W8G32_F16S"):
            recipes.encode_matrix_payload(
                tensor, q4, recipes.SOURCE_MSE_W8G32, "cpu"
            )
        with self.assertRaisesRegex(ValueError, "requires a matrix"):
            recipes.encode_matrix_payload(
                tensor,
                tensor_spec("selector", (16,), BF16),
                recipes.CANONICAL_Q4G64,
                "cpu",
            )
        bad_source = RepresentedBf16Matrix()
        bad_source.dtype = "torch.float32"
        with self.assertRaisesRegex(TypeError, "represented BF16"):
            recipes.encode_matrix_payload(
                bad_source, q4, recipes.CANONICAL_Q4G64, "cpu"
            )


if __name__ == "__main__":
    unittest.main()
