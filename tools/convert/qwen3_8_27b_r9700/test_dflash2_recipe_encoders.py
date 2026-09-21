"""Exact CPU production-encoder checks against independent represented-BF16 oracles."""

from __future__ import annotations

import importlib.util
import unittest

from tools.convert.qwen3.common.inventory import tensor_spec
from tools.convert.qwen3_8_27b_r9700 import dflash2_matrix_recipes as recipes


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "CPU Torch and NumPy are required for real encoder qualification")
class DFlash2RecipeEncodersTest(unittest.TestCase):
    def test_cpu_encoders_match_represented_source_oracle(self) -> None:
        import torch

        initialized = torch.cuda.is_initialized()
        for rows, columns in ((16, 64), (16, 130)):
            source = torch.tensor(
                [((index * 37) % 257 - 128) / 19 for index in range(rows * columns)],
                dtype=torch.bfloat16,
                device="cpu",
            ).reshape(rows, columns)
            represented = source.float().flatten().tolist()
            for recipe in recipes.RECIPES:
                with self.subTest(recipe=recipe.key, rows=rows, columns=columns):
                    spec = tensor_spec("matrix", (rows, columns), recipe.matrix_format)
                    self.assertEqual(
                        recipes.encode_matrix_payload(source, spec, recipe, "cpu"),
                        recipes.encode_matrix_reference(represented, rows, columns, recipe),
                    )
        self.assertEqual(torch.cuda.is_initialized(), initialized)


if __name__ == "__main__":
    unittest.main()
