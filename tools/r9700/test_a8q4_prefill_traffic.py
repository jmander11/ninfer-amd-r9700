#!/usr/bin/env python3

import unittest

from a8q4_prefill_traffic import (
    M32_N64,
    M64_N64,
    M64_N128,
    M64_N256,
    OLD_WAVE16,
    W8_M64_N64,
    W8_OLD_WAVE16,
    Shape,
    requested_traffic,
    requested_w8_traffic,
    parse_w8_shape,
    with_roofline,
    with_w8_roofline,
)


class A8Q4PrefillTrafficTest(unittest.TestCase):
    def test_one_group_old_wave_matches_source_load_ownership(self) -> None:
        traffic = requested_traffic(Shape(16, 16, 64), OLD_WAVE16)
        self.assertEqual(traffic.activation_code_bytes, 1024)
        self.assertEqual(traffic.weight_code_bytes, 512)
        self.assertEqual(traffic.activation_scale_bytes, 512)
        self.assertEqual(traffic.weight_scale_bytes, 64)
        self.assertEqual(traffic.status_bytes, 4)
        self.assertEqual(traffic.output_bytes, 512)
        self.assertEqual(traffic.total_requested_bytes, 2628)
        self.assertEqual(traffic.logical_ops, 32768)
        self.assertEqual(traffic.issued_iu4_ops, 65536)

    def test_m32_n64_stage_and_lds_are_exact(self) -> None:
        traffic = requested_traffic(Shape(32, 64, 64), M32_N64)
        self.assertEqual(traffic.activation_code_bytes, 2048)
        self.assertEqual(traffic.weight_code_bytes, 2048)
        self.assertEqual(traffic.activation_scale_bytes, 64)
        self.assertEqual(traffic.weight_scale_bytes, 128)
        self.assertEqual(traffic.static_lds_bytes, 4288)
        self.assertEqual(traffic.total_requested_bytes, 8416)
        self.assertEqual(traffic.waves, 8)

    def test_m64_n64_stage_and_lds_are_exact(self) -> None:
        traffic = requested_traffic(Shape(64, 64, 64), M64_N64)
        self.assertEqual(traffic.static_lds_bytes, 6400)
        self.assertEqual(traffic.total_requested_bytes, 14656)
        self.assertEqual(traffic.waves, 16)

    def test_tail_counts_only_represented_requests_but_issued_ops_round(self) -> None:
        shape = Shape(17, 65, 128)
        traffic = requested_traffic(shape, M32_N64)
        self.assertEqual(traffic.logical_ops, 2 * 17 * 65 * 128)
        self.assertEqual(traffic.issued_iu4_ops, 4 * 32 * 128 * 128)
        self.assertEqual(traffic.output_bytes, 17 * 65 * 2)
        self.assertEqual(traffic.blocks, 2)

    def test_real_shape_tiling_preserves_wave_count_and_reduces_requests(self) -> None:
        shape = Shape(4096, 34816, 5120)
        old = requested_traffic(shape, OLD_WAVE16)
        m32 = requested_traffic(shape, M32_N64)
        m64 = requested_traffic(shape, M64_N64)
        self.assertEqual(old.waves, m32.waves)
        self.assertEqual(old.waves, m64.waves)
        self.assertLess(m32.total_requested_bytes, old.total_requested_bytes)
        self.assertLess(m64.total_requested_bytes, m32.total_requested_bytes)
        self.assertGreater(m64.logical_ops_per_requested_byte,
                           m32.logical_ops_per_requested_byte)

    def test_m64_n256_halves_activation_rereads_at_exact_production_rows(self) -> None:
        weights = {(7168, 5120): 32, (4096, 5120): 48, (12288, 5120): 48,
                   (5120, 6144): 65, (34816, 5120): 65, (5120, 17408): 65,
                   (5120, 10240): 1, (1024, 5120): 0}
        production_total = challenger_total = 0
        for (rows, columns), count in weights.items():
            production = requested_traffic(Shape(2048, rows, columns), M64_N128)
            challenger = requested_traffic(Shape(2048, rows, columns), M64_N256)
            self.assertEqual(challenger.activation_code_bytes * 2,
                             production.activation_code_bytes)
            self.assertEqual(challenger.weight_code_bytes,
                             production.weight_code_bytes)
            self.assertEqual(challenger.output_bytes, production.output_bytes)
            self.assertEqual(challenger.static_lds_bytes, 12928)
            production_total += count * production.total_requested_bytes
            challenger_total += count * challenger.total_requested_bytes
        self.assertAlmostEqual(production_total / challenger_total, 1.3184984511)

    def test_roofline_accounts_for_two_iu4_planes(self) -> None:
        traffic = requested_traffic(Shape(16, 16, 64), OLD_WAVE16)
        result = with_roofline(traffic, bandwidth_gbps=100.0, iu4_tops=20.0)
        roofline = result["roofline"]
        self.assertAlmostEqual(roofline["compute_logical_tops"], 10.0)
        self.assertAlmostEqual(
            roofline["combined_service_roof_ms"],
            max(2628 / 100e9 * 1e3, 65536 / 20e12 * 1e3),
        )

    def test_invalid_group_extent_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            requested_traffic(Shape(1, 1, 65), OLD_WAVE16)

    def test_w8_old_wave_one_aligned_k128_tile(self) -> None:
        traffic = requested_w8_traffic(Shape(16, 16, 128), W8_OLD_WAVE16)
        self.assertEqual(traffic.activation_code_bytes, 2048)
        self.assertEqual(traffic.weight_code_bytes, 2048)
        self.assertEqual(traffic.activation_scale_bytes, 2048)
        self.assertEqual(traffic.weight_scale_bytes, 256)
        self.assertEqual(traffic.status_bytes, 4)
        self.assertEqual(traffic.output_bytes, 512)
        self.assertEqual(traffic.total_requested_bytes, 6916)
        self.assertEqual(traffic.logical_ops, 65536)
        self.assertEqual(traffic.issued_iu8_ops, 65536)

    def test_w8_m64_n64_staging_scales_status_and_lds(self) -> None:
        traffic = requested_w8_traffic(Shape(64, 64, 128), W8_M64_N64)
        self.assertEqual(traffic.activation_code_bytes, 8192)
        self.assertEqual(traffic.weight_code_bytes, 8192)
        self.assertEqual(traffic.activation_scale_bytes, 512)
        self.assertEqual(traffic.weight_scale_bytes, 512)
        self.assertEqual(traffic.status_bytes, 64)
        self.assertEqual(traffic.output_bytes, 8192)
        self.assertEqual(traffic.total_requested_bytes, 25664)
        self.assertEqual(traffic.static_lds_bytes, 4352)
        self.assertEqual(traffic.waves, 16)

    def test_w8_partial_g32_and_tile_tails_charge_padded_issue(self) -> None:
        shape = parse_w8_shape("129,65,193")
        traffic = requested_w8_traffic(shape, W8_M64_N64)
        self.assertEqual(traffic.padded_columns, 256)
        self.assertEqual(traffic.logical_ops, 2 * 129 * 65 * 193)
        self.assertEqual(traffic.issued_iu8_ops, 2 * 192 * 128 * 256)
        self.assertEqual(traffic.output_bytes, 129 * 65 * 2)
        self.assertEqual(traffic.blocks, 6)
        self.assertGreater(traffic.issued_iu8_ops, traffic.logical_ops)

    def test_w8_real_shapes_preserve_waves_and_reduce_requests(self) -> None:
        for rows, columns in ((12288, 5120), (5120, 17408),
                              (5120, 6144), (7168, 5120)):
            shape = Shape(4096, rows, columns)
            old = requested_w8_traffic(shape, W8_OLD_WAVE16)
            cta = requested_w8_traffic(shape, W8_M64_N64)
            self.assertEqual(old.waves, cta.waves)
            self.assertLess(cta.total_requested_bytes, old.total_requested_bytes)
            self.assertEqual(cta.static_lds_bytes, 4352)

    def test_w8_roofline_uses_single_exact_iu8_path(self) -> None:
        traffic = requested_w8_traffic(Shape(64, 64, 128), W8_M64_N64)
        result = with_w8_roofline(traffic, bandwidth_gbps=100.0, iu8_tops=20.0)
        roofline = result["roofline"]
        self.assertAlmostEqual(roofline["compute_logical_tops"], 20.0)
        self.assertAlmostEqual(
            roofline["combined_service_roof_ms"],
            max(25664 / 100e9 * 1e3, (2 * 64 * 64 * 128) / 20e12 * 1e3),
        )


if __name__ == "__main__":
    unittest.main()
