"""No-Torch tests for the independent R9700 grouped codec oracles."""

from __future__ import annotations

import struct
import random
import unittest

from tools.convert.qwen3_8_27b_r9700 import codec
from tools.convert.qwen3_8_27b_r9700 import inventory


class W8G32ReferenceCodecTest(unittest.TestCase):
    def test_ties_signed_codes_and_scale_plane_are_canonical(self) -> None:
        source = [-127.0, -126.5, -1.5, -0.5, 0.5, 1.5, 126.5, 127.0] + [0.0] * 24
        payload = codec.encode_w8g32_reference(source, 1, 32)
        self.assertEqual(len(payload), 264)
        self.assertEqual(
            payload[:8],
            bytes((0x81, 0x82, 0xFE, 0x00, 0x00, 0x02, 0x7E, 0x7F)),
        )
        self.assertEqual(payload[8:256], bytes(248))
        self.assertEqual(
            payload[256:],
            struct.pack("<HHHH", 0x3C00, 0x0000, 0x0000, 0x0000),
        )

    def test_k_padding_and_group_scales_are_explicit(self) -> None:
        payload = codec.encode_w8g32_reference([127.0] * 33, 1, 33)
        codes, scales = codec.decode_w8g32_reference(payload, 1, 33)
        self.assertEqual(len(payload), 264)
        self.assertEqual(codes[:33], [127] * 33)
        self.assertEqual(codes[33:], [0] * 95)
        self.assertEqual(scales, [0x3C00, 0x3C00, 0x0000, 0x0000])

    def test_underflow_uses_one_half_subnormal_and_nonfinite_rejects(self) -> None:
        payload = codec.encode_w8g32_reference([2.0**-30] + [0.0] * 31, 1, 32)
        _, scales = codec.decode_w8g32_reference(payload, 1, 32)
        self.assertEqual(scales, [codec.FP16_MIN_SUBNORMAL_BITS, 0x0000, 0x0000, 0x0000])
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            codec.encode_w8g32_reference([float("inf")] * 32, 1, 32)

    def test_mse_refinement_improves_exact_decoded_group_error(self) -> None:
        source = [10.0] + [1.0] * 31
        baseline = codec.encode_w8g32_reference(source, 1, 32)
        refined = codec.encode_w8g32_mse_reference(source, 1, 32)
        baseline_codes, baseline_scales = codec.decode_w8g32_reference(
            baseline, 1, 32
        )
        refined_codes, refined_scales = codec.decode_w8g32_reference(
            refined, 1, 32
        )

        def decoded_sse(codes: list[int], scale_bits: int) -> float:
            scale = struct.unpack("<e", struct.pack("<H", scale_bits))[0]
            return sum(
                (item - code * scale) ** 2
                for item, code in zip(source, codes, strict=True)
            )

        self.assertLess(
            decoded_sse(refined_codes[:32], refined_scales[0]),
            decoded_sse(baseline_codes[:32], baseline_scales[0]),
        )
        self.assertEqual(
            codec.encode_w8g32_mse_reference(source, 1, 32), refined
        )

    def test_mse_refinement_retains_padding_and_rejects_nonfinite(self) -> None:
        payload = codec.encode_w8g32_mse_reference([1.0] * 33, 1, 33)
        codes, scales = codec.decode_w8g32_reference(payload, 1, 33)
        self.assertEqual(codes[33:], [0] * 95)
        self.assertEqual(scales[2:], [0, 0])
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            codec.encode_w8g32_mse_reference([float("nan")] * 32, 1, 32)


class Q4G64ReferenceCodecTest(unittest.TestCase):
    def test_ties_signed_codes_and_nibble_order_are_canonical(self) -> None:
        row = [-7.0, -6.5, -1.5, -0.5, 0.5, 1.5, 6.5, 7.0] + [0.0] * 56
        source = row * 16
        payload = codec.encode_q4g64_reference(source, 16, 64)
        codes, scales = codec.decode_q4g64_reference(payload, 16, 64)
        self.assertEqual(len(payload), 1088)
        self.assertEqual(payload[:2], bytes((0xA9, 0x0E)))
        self.assertEqual(codes[:8], [-7, -6, -2, 0, 0, 2, 6, 7])
        self.assertEqual(codes[64:128], [0] * 64)
        self.assertEqual(scales[:2], [0x3C00, 0x0000])

    def test_tail_padding_scale_underflow_and_nonfinite_are_explicit(self) -> None:
        row = [2.0**-30] + [0.0] * 63 + [1.0]
        payload = codec.encode_q4g64_mse_reference(row * 16, 16, 65)
        codes, scales = codec.decode_q4g64_reference(payload, 16, 65)
        self.assertEqual(codes[65:128], [0] * 63)
        self.assertEqual(scales[0], codec.FP16_MIN_SUBNORMAL_BITS)
        self.assertNotEqual(scales[1], 0)
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            codec.encode_q4g64_mse_reference([float("nan")] * (16 * 64), 16, 64)
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            codec.encode_q4g64_reference([float("inf")] * (16 * 64), 16, 64)

    def test_mse_refinement_improves_and_reaches_signed_minimum(self) -> None:
        row = [-8.0] + [-1.0] * 63
        source = row * 16
        baseline = codec.encode_q4g64_reference(source, 16, 64)
        refined = codec.encode_q4g64_mse_reference(source, 16, 64)
        baseline_codes, baseline_scales = codec.decode_q4g64_reference(
            baseline, 16, 64
        )
        refined_codes, refined_scales = codec.decode_q4g64_reference(
            refined, 16, 64
        )

        def decoded_sse(codes: list[int], scale_bits: int) -> float:
            scale = struct.unpack("<e", struct.pack("<H", scale_bits))[0]
            return sum(
                (item - code * scale) ** 2
                for item, code in zip(row, codes, strict=True)
            )

        self.assertEqual(refined_codes[0], codec.Q4_QMIN)
        self.assertLess(
            decoded_sse(refined_codes[:64], refined_scales[0]),
            decoded_sse(baseline_codes[:64], baseline_scales[0]),
        )
        self.assertEqual(codec.encode_q4g64_mse_reference(source, 16, 64), refined)

    def test_baseline_wins_equal_error_tie_deterministically(self) -> None:
        source = [0.0] * (16 * 64)
        baseline = codec.encode_q4g64_reference(source, 16, 64)
        self.assertEqual(codec.encode_q4g64_mse_reference(source, 16, 64), baseline)

    def test_mse_never_regresses_baseline_group_sse(self) -> None:
        rng = random.Random(4464001)
        for case in range(32):
            rows = rng.choice((16, 32))
            columns = rng.randint(1, 193)
            source = [
                rng.uniform(-8.0, 8.0) * (2.0 ** rng.randint(-12, 8))
                for _ in range(rows * columns)
            ]
            baseline_codes, baseline_scales = codec.decode_q4g64_reference(
                codec.encode_q4g64_reference(source, rows, columns), rows, columns
            )
            refined_codes, refined_scales = codec.decode_q4g64_reference(
                codec.encode_q4g64_mse_reference(source, rows, columns), rows, columns
            )
            padded = (columns + 127) // 128 * 128
            groups = padded // codec.Q4_GROUP_SIZE
            for row in range(rows):
                physical = source[row * columns : (row + 1) * columns]
                physical += [0.0] * (padded - columns)
                for group in range(groups):
                    begin = row * padded + group * codec.Q4_GROUP_SIZE
                    values = physical[
                        group * codec.Q4_GROUP_SIZE : (group + 1) * codec.Q4_GROUP_SIZE
                    ]
                    base_scale = struct.unpack(
                        "<e", struct.pack("<H", baseline_scales[row * groups + group])
                    )[0]
                    refined_scale = struct.unpack(
                        "<e", struct.pack("<H", refined_scales[row * groups + group])
                    )[0]
                    baseline_sse = sum(
                        (value - code * base_scale) ** 2
                        for value, code in zip(
                            values,
                            baseline_codes[begin : begin + codec.Q4_GROUP_SIZE],
                            strict=True,
                        )
                    )
                    refined_sse = sum(
                        (value - code * refined_scale) ** 2
                        for value, code in zip(
                            values,
                            refined_codes[begin : begin + codec.Q4_GROUP_SIZE],
                            strict=True,
                        )
                    )
                    self.assertLessEqual(refined_sse, baseline_sse, f"case {case}")


class R9700CandidateInventoryTest(unittest.TestCase):
    def test_complete_qwen38_inventory_and_explicit_candidate_identity(self) -> None:
        inventory.validate_inventory()
        self.assertEqual(len(inventory.TENSOR_SPECS), 1118)
        self.assertEqual(len(inventory.OBJECT_SPECS), 1124)
        self.assertEqual(
            inventory.FORMAT_COUNTS,
            {"BF16": 582, "FP32": 96, "I32": 1, "W8G32_F16S": 439},
        )
        self.assertEqual(inventory.WEIGHTS_ID, "r9700-int-candidate")


if __name__ == "__main__":
    unittest.main()
