from __future__ import annotations

import struct
import unittest

import torch

from tools.artifact.layouts import (
    RowPlanes,
    assemble_row_planes,
    decode_direct,
    decode_q4_n16k16_codes,
    decode_row_split_codes,
    dequantize_row_split,
    encode_direct,
    encode_q4_n16k16,
    encode_row_split,
    encoded_size,
    gather_row_planes,
    q4_n16k16_geometry,
    row_scaled_geometry,
    row_split_geometry,
    split_row_planes,
    transcode_q4_n16k16,
)


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


def _direct_cases():
    return (
        (
            "BF16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "FP32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "I32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
    )


ROW_SPLIT_PLANE_CASES = (
    ("Q5G64_F16S", 65, (-16, -15, -1, 0, 1, 15), b"\x10\x0f\xf1", b"\x07"),
    (
        "Q6G64_F16S",
        65,
        (-32, -31, -17, -16, -1, 0, 15, 31),
        b"\x10\x0f\x0f\xff",
        b"\xea\x43",
    ),
    ("W8G32_F16S", 33, (-127, -1, 0, 1, 127), b"\x81\xff\x00\x01\x7f", b""),
)


class LayoutsTest(unittest.TestCase):
    def test_direct_layout_preserves_exact_little_endian_words(self) -> None:
        for format_name, tensor, words, word_format, word_view in _direct_cases():
            with self.subTest(format_name=format_name):
                expected = struct.pack("<" + word_format * len(words), *words)
                payload = encode_direct(tensor, format_name)
                self.assertEqual(payload, expected)
                decoded = decode_direct(payload, format_name, tensor.shape)
                self.assertTrue(torch.equal(decoded.view(word_view), tensor.view(word_view)))

                if format_name == "BF16":
                    with self.assertRaises(TypeError):
                        encode_direct(tensor.float(), format_name)

    def test_row_split_geometry_and_encoded_size_are_derived_from_format_and_shape(self) -> None:
        geometry = row_split_geometry("Q5G64_F16S", (2, 130))
        self.assertEqual(
            (
                geometry.k_pad,
                geometry.groups_per_row,
                geometry.base_bytes,
                geometry.high_offset,
                geometry.high_bytes,
                geometry.scale_offset,
                geometry.scale_bytes,
                geometry.payload_bytes,
            ),
            (256, 4, 256, 256, 64, 512, 16, 528),
        )
        self.assertEqual(encoded_size("row-split-k128-v1", "Q5G64_F16S", (2, 130)), 528)

        w8 = row_split_geometry("W8G32_F16S", (1, 4304))
        self.assertEqual(
            (w8.k_pad, w8.groups_per_row, w8.base_row_bytes, w8.high_row_bytes),
            (4352, 136, 4352, 0),
        )
        q4 = row_split_geometry("Q4G64_F16S", (16, 4304))
        self.assertEqual(
            (q4.k_pad, q4.base_row_bytes, q4.high_bytes, q4.scale_row_bytes),
            (4352, 2176, 0, 136),
        )

    def test_row_scaled_geometry_is_distinct_and_k128_padded(self) -> None:
        geometry = row_scaled_geometry("F8E4M3_ROW_F32S", (3, 129))
        self.assertEqual(
            (
                geometry.k_pad,
                geometry.code_row_bytes,
                geometry.code_bytes,
                geometry.scale_row_bytes,
                geometry.scale_offset,
                geometry.scale_bytes,
                geometry.payload_bytes,
            ),
            (256, 256, 768, 4, 768, 12, 780),
        )
        self.assertEqual(encoded_size("row-scaled-k128-v1", "F8E4M3_ROW_F32S", (3, 129)), 780)

        with self.assertRaisesRegex(ValueError, "does not accept"):
            encoded_size("row-split-k128-v1", "F8E4M3_ROW_F32S", (3, 129))
        with self.assertRaisesRegex(ValueError, "does not accept"):
            encoded_size("row-scaled-k128-v1", "W8G32_F16S", (3, 129))

    def test_row_split_plane_bit_order_and_round_trip(self) -> None:
        for format_name, k, prefix, base_prefix, high_prefix in ROW_SPLIT_PLANE_CASES:
            with self.subTest(format_name=format_name):
                geometry = row_split_geometry(format_name, (1, k))
                group_size = geometry.k_pad // geometry.groups_per_row
                codes = torch.zeros((1, geometry.groups_per_row, group_size), dtype=torch.int8)
                codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
                scales = torch.zeros((1, geometry.groups_per_row), dtype=torch.float16)
                scales[0, 0] = 1.5
                scales[0, 1] = 0.25

                payload = encode_row_split(codes, scales, format_name, (1, k))
                self.assertEqual(len(payload), geometry.payload_bytes)
                planes = split_row_planes(payload, geometry)
                self.assertEqual(bytes(planes.base[: len(base_prefix)]), base_prefix)
                self.assertEqual(bytes(planes.high[: len(high_prefix)]), high_prefix)
                self.assertEqual(
                    payload[geometry.base_bytes : geometry.high_offset],
                    bytes(geometry.high_offset - geometry.base_bytes),
                )
                self.assertEqual(
                    payload[geometry.high_offset + geometry.high_bytes : geometry.scale_offset],
                    bytes(geometry.scale_offset - geometry.high_offset - geometry.high_bytes),
                )

                decoded_scales, decoded_codes = decode_row_split_codes(payload, format_name, (1, k))
                self.assertTrue(torch.equal(decoded_scales, scales))
                self.assertTrue(torch.equal(decoded_codes, codes))

    def test_q4_n16k16_exact_tile_order_and_round_trip(self) -> None:
        shape = (16, 65)
        geometry = q4_n16k16_geometry(shape)
        codes = torch.zeros((16, geometry.groups_per_row, 64), dtype=torch.int8)
        scales = torch.zeros((16, geometry.groups_per_row), dtype=torch.float16)
        for row in range(16):
            codes[row, 0, :4] = torch.tensor((row & 7, -8, -1, 7), dtype=torch.int8)
            scales[row, 0] = row + 1
        payload = encode_q4_n16k16(codes, scales, shape)
        self.assertEqual(len(payload), geometry.payload_bytes)
        # Pair 0, lane `row`, eight-byte fragment: the first packed byte is row-specific.
        for row in range(16):
            self.assertEqual(payload[row * 8], (row & 7) | 0x80)
        decoded_scales, decoded_codes = decode_q4_n16k16_codes(payload, shape)
        self.assertTrue(torch.equal(decoded_scales, scales))
        self.assertTrue(torch.equal(decoded_codes, codes))
        self.assertEqual(
            encoded_size("r9700-q4g64-n16-k16-v1", "Q4G64_F16S", shape), geometry.payload_bytes
        )

    def test_q4_row_split_is_an_exact_permutation_of_n16k16(self) -> None:
        shape = (32, 130)
        geometry = q4_n16k16_geometry(shape)
        generator = torch.Generator().manual_seed(3)
        codes = torch.randint(
            -8, 8, (32, geometry.groups_per_row, 64), dtype=torch.int8, generator=generator
        )
        codes[:, 2, 130 - 128 :] = 0  # K padding (features 130..255) codes are zero
        codes[:, 3] = 0
        scales = (torch.rand((32, geometry.groups_per_row), generator=generator) + 0.01).half()
        rows = encode_row_split(codes, scales, "Q4G64_F16S", shape)
        tiled = encode_q4_n16k16(codes, scales, shape)
        self.assertEqual(encoded_size("row-split-k128-v1", "Q4G64_F16S", shape), len(rows))
        self.assertEqual(len(rows), len(tiled))
        # Row r's codes are one contiguous run and its scales follow the aligned code plane.
        self.assertEqual(
            rows[5 * geometry.groups_per_row * 32 : 6 * geometry.groups_per_row * 32],
            bytes(
                (int(codes[5].flatten()[2 * i]) & 15)
                | ((int(codes[5].flatten()[2 * i + 1]) & 15) << 4)
                for i in range(geometry.groups_per_row * 32)
            ),
        )
        decoded_scales, decoded_codes = decode_row_split_codes(rows, "Q4G64_F16S", shape)
        self.assertTrue(torch.equal(decoded_scales, scales))
        self.assertTrue(torch.equal(decoded_codes, codes))
        self.assertEqual(b"".join(transcode_q4_n16k16(rows, shape)), tiled)
        self.assertEqual(b"".join(transcode_q4_n16k16(tiled, shape, inverse=True)), rows)

    def test_consecutive_views_arbitrary_gathers_and_standalone_assembly(self) -> None:
        format_name = "Q5G64_F16S"
        shape = (4, 130)
        geometry = row_split_geometry(format_name, shape)
        codes = (
            torch.arange(geometry.n * geometry.groups_per_row * 64, dtype=torch.int32)
            .remainder(32)
            .sub(16)
            .to(torch.int8)
            .reshape(geometry.n, geometry.groups_per_row, 64)
        )
        codes.reshape(geometry.n, geometry.k_pad)[:, geometry.k :] = 0
        scales = torch.tensor(
            [
                [0.25, 0.5, 1.0, 0.0],
                [0.5, 1.0, 1.5, 0.0],
                [1.0, 1.5, 2.0, 0.0],
                [1.5, 2.0, 2.5, 0.0],
            ],
            dtype=torch.float16,
        )
        payload = encode_row_split(codes, scales, format_name, shape)

        consecutive = split_row_planes(payload, geometry, 1, 2)
        self.assertIsInstance(consecutive.base, memoryview)
        self.assertIs(consecutive.base.obj, payload)
        standalone = assemble_row_planes(consecutive, format_name, shape[1])
        consecutive_scales, consecutive_codes = decode_row_split_codes(
            standalone, format_name, (2, shape[1])
        )
        self.assertTrue(torch.equal(consecutive_scales, scales[1:3]))
        self.assertTrue(torch.equal(consecutive_codes, codes[1:3]))

        gathered = gather_row_planes(payload, geometry, [3, 1])
        self.assertIsInstance(gathered, RowPlanes)
        gathered_payload = assemble_row_planes(gathered, format_name, shape[1])
        gathered_scales, gathered_codes = decode_row_split_codes(
            gathered_payload, format_name, (2, shape[1])
        )
        self.assertTrue(torch.equal(gathered_scales, scales[[3, 1]]))
        self.assertTrue(torch.equal(gathered_codes, codes[[3, 1]]))
        expected = (codes[[3, 1]].float() * scales[[3, 1]].float().unsqueeze(-1)).reshape(
            2, geometry.k_pad
        )[:, : shape[1]]
        actual = dequantize_row_split(gathered, format_name, (2, shape[1]), dtype=torch.float32)
        self.assertTrue(torch.equal(actual, expected))

        resident = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        tensor_gather = gather_row_planes(
            resident, geometry, torch.tensor([2, 0], dtype=torch.long)
        )
        for plane in (tensor_gather.base, tensor_gather.high, tensor_gather.scale):
            self.assertIsInstance(plane, torch.Tensor)
        tensor_payload = assemble_row_planes(tensor_gather, format_name, shape[1])
        tensor_scales, tensor_codes = decode_row_split_codes(
            tensor_payload, format_name, (2, shape[1])
        )
        self.assertTrue(torch.equal(tensor_scales, scales[[2, 0]]))
        self.assertTrue(torch.equal(tensor_codes, codes[[2, 0]]))


if __name__ == "__main__":
    unittest.main()
