"""Cross-check the Torch converter encoder against the independent scalar oracle."""

from __future__ import annotations

import os
import random
import unittest

from tools.convert.qwen3_8_27b_r9700.codec import (
    decode_q4g64_reference,
    encode_q4g64_mse_reference,
    encode_q4g64_n16k16_reference,
    encode_q4g64_reference,
    encode_w8g32_mse_reference,
    encode_w8g32_reference,
)

try:
    import torch
    from tools.convert.common.quantize import quantize_and_encode, quantize_matrix
    from tools.convert.qwen3_8_27b_r9700.mse_quantize import (
        quantize_and_encode_mse,
        quantize_and_encode_q4g64_mse,
        quantize_q4g64_matrix_mse,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "requires the project Torch environment")
class VectorizedW8G32CodecTest(unittest.TestCase):
    def test_bf16_source_matches_independent_scalar_packing(self) -> None:
        values = [
            -127.0,
            -126.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            126.5,
            127.0,
        ] + [float((index % 9) - 4) for index in range(25)]
        matrix = torch.tensor(values, dtype=torch.bfloat16).reshape(1, 33)
        represented = matrix.to(dtype=torch.float32).reshape(-1).tolist()
        expected = encode_w8g32_reference(represented, 1, 33)
        actual = quantize_and_encode(matrix, "W8G32_F16S", device="cpu")
        self.assertEqual(actual, expected)

    def test_mse_refined_path_matches_independent_scalar_recipe(self) -> None:
        values = [10.0] + [1.0] * 31 + [float((index % 13) - 6) for index in range(33)]
        matrix = torch.tensor(values, dtype=torch.bfloat16).reshape(1, 65)
        represented = matrix.to(dtype=torch.float32).reshape(-1).tolist()
        expected = encode_w8g32_mse_reference(represented, 1, 65)
        actual = quantize_and_encode_mse(matrix, device="cpu")
        self.assertEqual(actual, expected)

    def test_mse_refined_path_matches_scalar_across_tails_and_ranges(self) -> None:
        rng = random.Random(93201)
        for case in range(32):
            rows = rng.randint(1, 4)
            columns = rng.randint(1, 257)
            values = [
                rng.uniform(-10.0, 10.0) * (2.0 ** rng.randint(-15, 10))
                for _ in range(rows * columns)
            ]
            matrix = torch.tensor(values, dtype=torch.bfloat16).reshape(rows, columns)
            represented = matrix.to(dtype=torch.float32).reshape(-1).tolist()
            expected = encode_w8g32_mse_reference(represented, rows, columns)
            actual = quantize_and_encode_mse(matrix, device="cpu")
            self.assertEqual(actual, expected, f"case {case}, shape {(rows, columns)}")


@unittest.skipIf(torch is None, "requires the project Torch environment")
class VectorizedQ4G64CodecTest(unittest.TestCase):
    def assert_logical_q4_equal(self, actual, expected_payload, rows, columns) -> None:
        expected_codes, expected_scales = decode_q4g64_reference(
            expected_payload, rows, columns)
        self.assertEqual(actual.codes.cpu().reshape(-1).tolist(), expected_codes)
        self.assertEqual(actual.scales.cpu().contiguous().view(torch.uint16).reshape(-1).tolist(),
                         expected_scales)

    def test_canonical_q4_path_matches_independent_scalar_packing(self) -> None:
        values = [-7.0, -6.5, -1.5, -0.5, 0.5, 1.5, 6.5, 7.0]
        values += [float((index % 11) - 5) for index in range(57)]
        matrix = torch.tensor(values, dtype=torch.bfloat16).reshape(1, 65)
        represented = matrix.to(dtype=torch.float32).reshape(-1).tolist()
        expected = encode_q4g64_reference(represented, 1, 65)
        actual = quantize_matrix(matrix, "Q4G64_F16S", device="cpu")
        self.assert_logical_q4_equal(actual, expected, 1, 65)

    def test_q4_mse_matches_scalar_across_tails_scales_and_extrema(self) -> None:
        rng = random.Random(46401)
        cases = [
            (1, 1, [2.0**-30]),
            (1, 64, [-8.0] + [-1.0] * 63),
            (1, 65, [-7.0, 7.0] + [0.0] * 63),
        ]
        for _ in range(32):
            rows = rng.randint(1, 4)
            columns = rng.randint(1, 257)
            values = [
                rng.uniform(-10.0, 10.0) * (2.0 ** rng.randint(-15, 10))
                for _ in range(rows * columns)
            ]
            cases.append((rows, columns, values))
        for case, (rows, columns, values) in enumerate(cases):
            matrix = torch.tensor(values, dtype=torch.bfloat16).reshape(rows, columns)
            represented = matrix.to(dtype=torch.float32).reshape(-1).tolist()
            expected = encode_q4g64_mse_reference(represented, rows, columns)
            actual = quantize_q4g64_matrix_mse(matrix, device="cpu")
            try:
                self.assert_logical_q4_equal(actual, expected, rows, columns)
            except AssertionError as exc:
                raise AssertionError(f"case {case}, shape {(rows, columns)}") from exc

    def test_q4_mse_rejects_nonfinite_and_is_deterministic(self) -> None:
        matrix = torch.zeros((2, 129), dtype=torch.bfloat16)
        first = quantize_q4g64_matrix_mse(matrix, device="cpu")
        second = quantize_q4g64_matrix_mse(matrix, device="cpu")
        self.assertTrue(torch.equal(first.codes, second.codes))
        self.assertTrue(torch.equal(first.scales, second.scales))
        matrix[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            quantize_q4g64_matrix_mse(matrix, device="cpu")


@unittest.skipUnless(
    os.environ.get("NINFER_RUN_R9700_CODEC_TESTS") == "1",
    "set NINFER_RUN_R9700_CODEC_TESTS=1 for the physical gfx1201 codec gate",
)
class R9700Q4G64CodecTest(unittest.TestCase):
    def test_q4_mse_exact_bytes_match_scalar_and_cpu_on_gfx1201(self) -> None:
        if torch is None or not torch.cuda.is_available():
            self.skipTest("requires ROCm Torch and a visible R9700")
        properties = torch.cuda.get_device_properties(0)
        self.assertEqual(properties.name, "AMD Radeon AI PRO R9700")
        self.assertEqual(properties.gcnArchName, "gfx1201")

        rng = random.Random(9700464)
        cases = [
            (16, 1, [2.0**-30] * 16),
            (16, 64, ([-8.0] + [-1.0] * 63) * 16),
            (16, 65, ([-7.0, 7.0] + [0.0] * 63) * 16),
            (16, 129, [0.0] * (16 * 129)),
        ]
        for _ in range(12):
            rows = 16
            columns = rng.randint(1, 257)
            values = [
                rng.uniform(-10.0, 10.0) * (2.0 ** rng.randint(-15, 10))
                for _ in range(rows * columns)
            ]
            cases.append((rows, columns, values))

        for case, (rows, columns, values) in enumerate(cases):
            matrix = torch.tensor(values, dtype=torch.bfloat16).reshape(rows, columns)
            represented = matrix.to(dtype=torch.float32).reshape(-1).tolist()
            expected = encode_q4g64_n16k16_reference(
                represented, rows, columns, refined=True)
            cpu = quantize_and_encode_q4g64_mse(matrix, device="cpu")
            rocm = quantize_and_encode_q4g64_mse(matrix, device="cuda")
            self.assertEqual(cpu, expected, f"CPU case {case}, shape {(rows, columns)}")
            self.assertEqual(
                rocm, expected, f"ROCm case {case}, shape {(rows, columns)}"
            )


if __name__ == "__main__":
    unittest.main()
