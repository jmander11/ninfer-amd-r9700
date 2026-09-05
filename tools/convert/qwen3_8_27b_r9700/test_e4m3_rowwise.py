"""CPU-only checks for the isolated rowwise-E4M3 weight vertical slice."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import struct
import unittest

import torch

from . import codec
from .e4m3_rowwise import (
    decode_e4m3_rowwise,
    encode_e4m3_rowwise,
    encode_e4m3_rowwise_chunks,
    error_metrics,
)


class E4M3RowwiseScalarOracleTest(unittest.TestCase):
    def test_rne_saturation_signed_zero_padding_and_scale_bytes(self) -> None:
        source = [448.0, 1.0625, 1.1875, -0.0]
        payload = codec.encode_e4m3_rowwise_reference(source, 1, 4)
        decoded, codes, scales = codec.decode_e4m3_rowwise_reference(payload, 1, 4)
        self.assertEqual(len(payload), 260)
        self.assertEqual(codes[:4], [0x7E, 0x38, 0x3A, 0x80])
        self.assertEqual(codes[4:], [0] * 124)
        self.assertEqual(payload[128:256], bytes(128))
        self.assertEqual(payload[256:], struct.pack("<f", 1.0))
        self.assertEqual(scales, [1.0])
        self.assertEqual(decoded[:3], [448.0, 1.0, 1.25])
        self.assertEqual(math.copysign(1.0, decoded[3]), -1.0)

    def test_zero_row_and_malformed_payload_are_rejected_canonically(self) -> None:
        payload = codec.encode_e4m3_rowwise_reference([-0.0] * 3, 1, 3)
        decoded, codes, scales = codec.decode_e4m3_rowwise_reference(payload, 1, 3)
        self.assertEqual(decoded, [0.0, 0.0, 0.0])
        self.assertEqual(codes, [0] * 128)
        self.assertEqual(scales, [0.0])
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            codec.encode_e4m3_rowwise_reference([float("nan")], 1, 1)
        with self.assertRaisesRegex(ValueError, "NaN word"):
            codec.decode_e4m3_rowwise_reference(bytes([0x7F]) + bytes(259), 1, 1)
        with self.assertRaisesRegex(ValueError, "zero-scale"):
            codec.decode_e4m3_rowwise_reference(bytes([1]) + bytes(259), 1, 1)
        nonzero_k_padding = bytearray(
            codec.encode_e4m3_rowwise_reference([1.0], 1, 1)
        )
        nonzero_k_padding[1] = 1
        with self.assertRaisesRegex(ValueError, "K128 padding"):
            codec.decode_e4m3_rowwise_reference(bytes(nonzero_k_padding), 1, 1)
        nonzero_plane_padding = bytearray(
            codec.encode_e4m3_rowwise_reference([1.0], 1, 1)
        )
        nonzero_plane_padding[128] = 1
        with self.assertRaisesRegex(ValueError, "alignment padding"):
            codec.decode_e4m3_rowwise_reference(bytes(nonzero_plane_padding), 1, 1)


class E4M3RowwiseVectorizedCodecTest(unittest.TestCase):
    def test_vectorized_bytes_match_scalar_oracle_at_real_qwen_widths(self) -> None:
        rng = random.Random(8431201)
        for columns in (5120, 6144, 17408):
            values = [rng.gauss(0.0, 0.021) for _ in range(2 * columns)]
            matrix = torch.tensor(values, dtype=torch.bfloat16).reshape(2, columns)
            represented = matrix.float().reshape(-1).tolist()
            expected = codec.encode_e4m3_rowwise_reference(
                represented, 2, columns
            )
            actual = encode_e4m3_rowwise(matrix)
            self.assertEqual(actual, expected, f"K={columns}")
            self.assertEqual(
                b"".join(encode_e4m3_rowwise_chunks(matrix, rows_per_chunk=1)),
                expected,
                f"streamed K={columns}",
            )
            decoded = decode_e4m3_rowwise(actual, 2, columns)
            scalar_decoded, _, _ = codec.decode_e4m3_rowwise_reference(
                actual, 2, columns
            )
            self.assertEqual(decoded.reshape(-1).tolist(), scalar_decoded)
            metrics = error_metrics(matrix, decoded)
            self.assertLess(metrics["relative_l2"], 0.03)

    def test_dtype_shape_nonfinite_and_payload_validation(self) -> None:
        with self.assertRaisesRegex(TypeError, "represented BF16"):
            encode_e4m3_rowwise(torch.ones((1, 128), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "rank-2"):
            encode_e4m3_rowwise(torch.ones((128,), dtype=torch.bfloat16))
        nonfinite = torch.zeros((1, 128), dtype=torch.bfloat16)
        nonfinite[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            encode_e4m3_rowwise(nonfinite)
        payload = bytearray(encode_e4m3_rowwise(torch.ones((1, 128), dtype=torch.bfloat16)))
        payload[0] = 0xFF
        with self.assertRaisesRegex(ValueError, "NaN word"):
            decode_e4m3_rowwise(bytes(payload), 1, 128)


@unittest.skipUnless(
    os.environ.get("NINFER_QWEN38_BF16_MODEL"),
    "set NINFER_QWEN38_BF16_MODEL for the real-source CPU codec gate",
)
class E4M3RowwiseRealSourceTest(unittest.TestCase):
    def test_representative_text_matrix_rows_have_bounded_error(self) -> None:
        from safetensors import safe_open

        root = Path(os.environ["NINFER_QWEN38_BF16_MODEL"])
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        cases = {
            "model.language_model.layers.0.mlp.gate_proj.weight": (17408, 5120),
            "model.language_model.layers.3.self_attn.o_proj.weight": (5120, 6144),
            "model.language_model.layers.0.mlp.down_proj.weight": (5120, 17408),
            "lm_head.weight": (248320, 5120),
        }
        for name, expected_shape in cases.items():
            with safe_open(root / index[name], framework="pt", device="cpu") as source:
                view = source.get_slice(name)
                self.assertEqual(tuple(view.get_shape()), expected_shape)
                sample = view[:8, :]
            decoded = decode_e4m3_rowwise(
                encode_e4m3_rowwise(sample), *sample.shape
            )
            metrics = error_metrics(sample, decoded)
            self.assertLess(metrics["relative_l2"], 0.03, name)


if __name__ == "__main__":
    unittest.main()
