"""CPU checks for the FP8LUT4 codebook codec: exact E4M3 words, payload framing, reconstruction."""

from __future__ import annotations

import unittest

import torch

from tools.artifact.layouts import fp8lut4_geometry, fp8lut4_magnitude_table, decode_fp8lut4, encode_fp8lut4_planes
from tools.artifact.numeric import round_e4m3fn_magnitude

from .fp8lut4_codec import Calibration, encode_chunks, interleave_gate_up, quantize_rows


class Fp8Lut4CodecTest(unittest.TestCase):
    def test_e4m3_rounding_matches_torch_ties_to_even_with_saturation(self) -> None:
        for numerator in range(0, 520):
            for exponent in range(-30, 10):
                value = min(numerator * 2.0 ** exponent, 448.0)
                expected = torch.tensor(value).to(torch.float8_e4m3fn).view(torch.uint8).item()
                self.assertEqual(round_e4m3fn_magnitude(numerator, exponent), expected)

    def test_group_codebook_is_monotone_and_spans_the_exponent_range(self) -> None:
        table = torch.tensor(fp8lut4_magnitude_table(), dtype=torch.uint8)
        values = table.view(torch.float8_e4m3fn).to(torch.float64)
        self.assertTrue(bool((values[:, 1:] >= values[:, :-1]).all()))
        self.assertEqual(values[26 * 8].tolist(), [0.0, 0.8125, 1.75, 2.5, 3.5, 4.5, 6.0, 7.5])
        self.assertEqual(values[255, 7].item(), 448.0)           # (7.5 * 1.875 * 32) saturates
        self.assertEqual(values[0].tolist(), [0.0] * 8)          # E = -26 flushes to zero
        self.assertTrue(bool((values[:, 0] == 0).all()))        # every codebook has a zero level

    def test_payload_round_trip_matches_quantized_words(self) -> None:
        generator = torch.Generator().manual_seed(3)
        source = torch.randn(48, 200, generator=generator)  # 48 rows: three 16-row tiles
        source[:, 17] *= 40.0                                    # outlier column
        source[5] = 0.0                                          # zero row
        weight = source.to(torch.bfloat16)
        payload = b"".join(encode_chunks(weight, rows_per_chunk=16))
        geometry = fp8lut4_geometry((48, 200))
        self.assertEqual(len(payload), geometry.payload_bytes)
        decoded = decode_fp8lut4(payload, (48, 200))
        padded = torch.nn.functional.pad(weight.float(), (0, geometry.k_pad - 200))
        codes, groups, scales = quantize_rows(padded)
        table = torch.tensor(fp8lut4_magnitude_table(), dtype=torch.uint8)
        magnitude = table.view(torch.float8_e4m3fn).to(torch.float64)[
            groups.long().repeat_interleave(32, 1), (codes & 7).long()]
        expected = torch.where((codes & 8) != 0, -magnitude, magnitude) * scales.double()[:, None]
        self.assertTrue(torch.equal(decoded, expected[:, :200]))
        self.assertTrue(bool((decoded[5] == 0).all()))
        error = (decoded - weight.double()).norm() / weight.double().norm()
        self.assertLess(error.item(), 0.1)

    def test_tile_positions_match_the_layout_contract(self) -> None:
        # 32 rows x 256 columns: tile (b, s), slot 16 * half + r % 16, 16 bytes per slot.
        codes = torch.zeros(32, 256, dtype=torch.uint8)
        groups = torch.zeros(32, 8, dtype=torch.uint8)
        codes[21, 2 * 64 + 32 + 6] = 5            # r=21: b=1, s=2, half=1, byte 3 low nibble
        codes[21, 2 * 64 + 32 + 7] = 9            # same byte, high nibble
        groups[21, 2 * 2 + 1] = 77                # k/32 = 5 -> s=2, half=1
        payload = encode_fp8lut4_planes(codes, groups, torch.ones(32))
        geometry = fp8lut4_geometry((32, 256))
        tile = 1 * (256 // 64) + 2
        slot = 16 * 1 + 21 % 16
        self.assertEqual(payload[tile * 512 + slot * 16 + 3], 5 | (9 << 4))
        self.assertEqual(sum(payload[:geometry.code_bytes]), 5 | (9 << 4))
        self.assertEqual(payload[geometry.group_offset + tile * 32 + slot], 77)

    def test_gate_up_interleave_pairs_features_in_sixteen_row_tiles(self) -> None:
        features = 32
        weight = torch.arange(2 * features, dtype=torch.float32)[:, None].repeat(1, 4)
        stored = interleave_gate_up(weight)[:, 0].tolist()
        # Tile 1: gate features 8..15, then up features 8..15 (source rows 40..47).
        self.assertEqual(stored[16:32], list(range(8, 16)) + list(range(40, 48)))

    def test_calibrated_rounding_reduces_output_error_and_round_trips(self) -> None:
        generator = torch.Generator().manual_seed(5)
        columns = 256
        mixing = torch.randn(columns, columns, generator=generator) * 0.3 + torch.eye(columns)
        inputs = (torch.randn(4096, columns, generator=generator) @ mixing) * torch.exp(
            torch.randn(columns, generator=generator))
        inputs[:, 40] = 0.0                                      # never-active input
        weight = torch.randn(32, columns, generator=generator).to(torch.bfloat16)
        calibration = Calibration(inputs.t() @ inputs, 0.01)
        table = torch.tensor(fp8lut4_magnitude_table(), dtype=torch.uint8)

        def decoded(codes, groups, scales):
            magnitude = table.view(torch.float8_e4m3fn).to(torch.float64)[
                groups.long().repeat_interleave(32, 1), (codes & 7).long()]
            return torch.where((codes & 8) != 0, -magnitude, magnitude) * scales.double()[:, None]

        def output_error(words):
            return ((decoded(*words) - weight.double()) @ inputs.double().t()).norm().item()

        calibrated = quantize_rows(weight.float(), calibration)
        self.assertLess(output_error(calibrated), 0.8 * output_error(quantize_rows(weight.float())))
        self.assertTrue(bool((decoded(*calibrated)[:, 40] == 0).all()))
        payload = b"".join(encode_chunks(weight, rows_per_chunk=16, calibration=calibration))
        self.assertTrue(torch.equal(decode_fp8lut4(payload, (32, columns)), decoded(*calibrated)))

    def test_nonfinite_source_is_rejected(self) -> None:
        weight = torch.zeros(2, 64)
        weight[1, 3] = float("nan")
        with self.assertRaises(ValueError):
            quantize_rows(weight)


if __name__ == "__main__":
    unittest.main()
