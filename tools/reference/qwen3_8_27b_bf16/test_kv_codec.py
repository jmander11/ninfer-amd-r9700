"""Exact CPU tests for the BF16-source cache-only diagnostic codecs."""

from __future__ import annotations

import math
import struct
import unittest

from . import kv_codec


def next_float32(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


class ExactE4M3FnTest(unittest.TestCase):
    def test_all_finite_words_round_trip(self) -> None:
        for word in range(256):
            if word & 0x7F == 0x7F:
                with self.assertRaisesRegex(ValueError, "NaN word"):
                    kv_codec.decode_e4m3fn(word)
                continue
            decoded = kv_codec.decode_e4m3fn(word)
            self.assertTrue(math.isfinite(decoded))
            self.assertEqual(kv_codec.encode_e4m3fn(decoded), word)

    def test_ties_underflow_saturation_and_signed_zero(self) -> None:
        self.assertEqual(kv_codec.encode_e4m3fn(1.0625), 0x38)
        self.assertEqual(kv_codec.encode_e4m3fn(1.1875), 0x3A)
        underflow_midpoint = math.ldexp(1.0, -10)
        self.assertEqual(kv_codec.encode_e4m3fn(underflow_midpoint), 0x00)
        self.assertEqual(kv_codec.encode_e4m3fn(next_float32(underflow_midpoint)), 0x01)
        self.assertEqual(kv_codec.encode_e4m3fn(1000.0), 0x7E)
        self.assertEqual(kv_codec.encode_e4m3fn(-1000.0), 0xFE)
        self.assertEqual(kv_codec.encode_e4m3fn(0.0), 0x00)
        self.assertEqual(kv_codec.encode_e4m3fn(-0.0), 0x80)
        self.assertLess(math.copysign(1.0, kv_codec.decode_e4m3fn(0x80)), 0.0)
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                kv_codec.encode_e4m3fn(value)


class ExactInt4Fp16Test(unittest.TestCase):
    def test_fp16_round_to_nearest_even_scale(self) -> None:
        midpoint = 1.0 + math.ldexp(1.0, -11)
        self.assertEqual(kv_codec.fp32_to_fp16_rne(midpoint), 0x3C00)
        self.assertEqual(kv_codec.fp32_to_fp16_rne(next_float32(midpoint)), 0x3C01)
        self.assertEqual(kv_codec.fp16_to_fp32(0x3C00), 1.0)
        self.assertEqual(kv_codec.fp16_to_fp32(0x0001), math.ldexp(1.0, -24))

    def test_code_ties_endpoints_and_low_lane_first_packing(self) -> None:
        values = (
            0.5,
            1.5,
            2.5,
            3.5,
            -0.5,
            -1.5,
            -2.5,
            -3.5,
            7.0,
            -7.0,
            0.0,
            1.0,
            -1.0,
            6.0,
            -6.0,
            0.0,
        )
        encoded = kv_codec.encode_value_group(values, 16)
        expected = (0, 2, 2, 4, 0, -2, -2, -4, 7, -7, 0, 1, -1, 6, -6, 0)
        self.assertEqual(encoded.scale_bits, 0x3C00)
        self.assertEqual(encoded.codes, expected)
        self.assertEqual(kv_codec.unpack_signed_int4(encoded.packed_codes, 16), expected)
        self.assertEqual(encoded.packed_codes[0], 0x20)
        self.assertEqual(encoded.packed_codes[4], 0x97)
        self.assertEqual(encoded.decoded(), tuple(float(value) for value in expected))
        with self.assertRaisesRegex(ValueError, "reserved"):
            kv_codec.unpack_signed_int4(bytes((0x08,)), 2)

    def test_zero_and_finite_scale_underflow_are_canonical(self) -> None:
        for values in ((-0.0,) * 16, (math.ldexp(1.0, -24),) * 16):
            encoded = kv_codec.encode_value_group(values, 16)
            self.assertEqual(encoded.scale_bits, 0)
            self.assertEqual(encoded.codes, (0,) * 16)
            self.assertEqual(encoded.packed_codes, bytes(8))

    def test_nonfinite_and_scale_overflow_are_rejected(self) -> None:
        invalid = [0.0] * 16
        invalid[3] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            kv_codec.encode_value_group(invalid, 16)
        fp32_max = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
        with self.assertRaisesRegex(OverflowError, "finite FP16"):
            kv_codec.encode_value_group((fp32_max,) * 16, 16)

    def test_g16_and_g32_have_distinct_group_scales(self) -> None:
        values = tuple(float(lane - 8) / 8.0 for lane in range(16)) + tuple(
            float(lane - 7) * 2.0 for lane in range(16)
        )
        first = kv_codec.encode_value_group(values[:16], 16)
        second = kv_codec.encode_value_group(values[16:], 16)
        whole = kv_codec.encode_value_group(values, 32)
        split_error = sum(
            (source - decoded) ** 2
            for source, decoded in zip(
                values, first.decoded() + second.decoded(), strict=True
            )
        )
        whole_error = sum(
            (source - decoded) ** 2
            for source, decoded in zip(values, whole.decoded(), strict=True)
        )
        self.assertLess(split_error, whole_error)


try:
    import torch

    from .kv_tensor_codec import (
        QuantizedFullAttentionCache,
        dequantize_keys,
        dequantize_values,
        quantize_keys,
        quantize_values,
        validate_codec_device,
    )

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is not installed")
class TensorCodecCpuCrossCheckTest(unittest.TestCase):
    def test_device_preflight_accepts_exact_cpu_codec(self) -> None:
        validate_codec_device(torch.device("cpu"), 16)
        validate_codec_device(torch.device("cpu"), 32)

    def test_vectorized_key_codes_match_independent_authority(self) -> None:
        source = [0.0, -0.0, 1.0, -1.0, 448.0, -448.0, 1.0625, 1.1875, 1000.0]
        tensor = torch.tensor(source, dtype=torch.float32)
        codes = quantize_keys(tensor)
        self.assertEqual(
            codes.view(torch.uint8).tolist(),
            [kv_codec.encode_e4m3fn(value) for value in source],
        )
        expected = [kv_codec.decode_e4m3fn(code) for code in codes.view(torch.uint8).tolist()]
        self.assertEqual(dequantize_keys(codes).tolist(), expected)

    def test_representative_bf16_planes_match_scalar_authority(self) -> None:
        torch.manual_seed(23)
        keys = (torch.randn(3, 4, 256, dtype=torch.float32) * 24.0).to(torch.bfloat16)
        key_words = quantize_keys(keys).view(torch.uint8).flatten().tolist()
        self.assertEqual(
            key_words,
            [kv_codec.encode_e4m3fn(value) for value in keys.float().flatten().tolist()],
        )

        values = (torch.randn(3, 4, 256, dtype=torch.float32) * 3.0).to(torch.bfloat16)
        represented = values.float().reshape(-1, 256).tolist()
        for group_size in (16, 32):
            expected_groups = [
                kv_codec.encode_value_group(row[begin : begin + group_size], group_size)
                for row in represented
                for begin in range(0, 256, group_size)
            ]
            packed, scales = quantize_values(values, group_size)
            self.assertEqual(
                bytes(packed.flatten().tolist()),
                b"".join(group.packed_codes for group in expected_groups),
            )
            self.assertEqual(
                [int(word) & 0xFFFF for word in scales.view(torch.int16).flatten().tolist()],
                [group.scale_bits for group in expected_groups],
            )

    def test_vectorized_value_planes_match_independent_authority(self) -> None:
        values = [
            0.5,
            1.5,
            2.5,
            3.5,
            -0.5,
            -1.5,
            -2.5,
            -3.5,
            7.0,
            -7.0,
            0.0,
            1.0,
            -1.0,
            6.0,
            -6.0,
            0.0,
        ] + [float(lane - 7) * 2.0 for lane in range(16)]
        tensor = torch.tensor(values, dtype=torch.bfloat16).reshape(1, 1, 32)
        represented = tensor.float().flatten().tolist()
        for group_size in (16, 32):
            expected_groups = [
                kv_codec.encode_value_group(
                    represented[begin : begin + group_size], group_size
                )
                for begin in range(0, 32, group_size)
            ]
            packed, scales = quantize_values(tensor, group_size)
            self.assertEqual(
                bytes(packed.flatten().tolist()),
                b"".join(group.packed_codes for group in expected_groups),
            )
            self.assertEqual(
                [int(word) & 0xFFFF for word in scales.view(torch.int16).flatten().tolist()],
                [group.scale_bits for group in expected_groups],
            )
            self.assertEqual(
                dequantize_values(packed, scales, group_size).flatten().tolist(),
                [value for group in expected_groups for value in group.decoded()],
            )

    def test_append_and_use_boundaries_preserve_planes(self) -> None:
        cache = QuantizedFullAttentionCache(3, 1, 32, 16, torch.device("cpu"))
        keys = torch.linspace(-2.0, 2.0, 96, dtype=torch.bfloat16).reshape(3, 1, 32)
        values = torch.linspace(-7.0, 7.0, 96, dtype=torch.bfloat16).reshape(3, 1, 32)
        cache.append(0, keys[:2], values[:2])
        cache.append(2, keys[2:], values[2:])
        represented_keys, represented_values = cache.read(3)
        self.assertEqual(represented_keys.dtype, torch.float32)
        self.assertEqual(represented_values.dtype, torch.float32)
        self.assertEqual(cache.key_codes.dtype, torch.float8_e4m3fn)
        self.assertEqual(cache.value_codes.dtype, torch.uint8)
        self.assertEqual(cache.value_scales.dtype, torch.float16)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            cache.append(1, keys[:1], values[:1])

    def test_tensor_zero_underflow_and_scale_overflow(self) -> None:
        tiny = torch.full((1, 1, 16), math.ldexp(1.0, -24), dtype=torch.bfloat16)
        packed, scales = quantize_values(tiny, 16)
        self.assertEqual(scales.view(torch.int16).item() & 0xFFFF, 0)
        self.assertEqual(bytes(packed.flatten().tolist()), bytes(8))
        maximum = torch.full(
            (1, 1, 16), torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(OverflowError, "finite FP16"):
            quantize_values(maximum, 16)


if __name__ == "__main__":
    unittest.main()
