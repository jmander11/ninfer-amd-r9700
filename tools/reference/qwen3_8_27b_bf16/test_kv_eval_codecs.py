"""CPU checks for the evaluation-only round-trip cache codecs."""

from __future__ import annotations

import math
import random
import unittest

import torch

from . import kv_codec, kv_eval_codecs, protocol

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def scalar_e2m1_rne(value: float) -> float:
    """Independent nearest-even E2M1 rounding with saturation, by exhaustive comparison."""
    magnitude = min(abs(value), 6.0)
    best = min(
        range(len(E2M1)),
        # Ties prefer the even-mantissa code: odd E2M1 indices carry mantissa bit 1.
        key=lambda index: (abs(E2M1[index] - magnitude), index & 1),
    )
    return math.copysign(E2M1[best], value)


def scalar_nvfp4_group(values: list[float], divisor: float) -> list[float]:
    amax = max(abs(item) for item in values)
    scale = kv_codec.decode_e4m3fn(kv_codec.encode_e4m3fn(amax / divisor))
    if scale == 0.0:
        return [0.0] * len(values)
    return [scalar_e2m1_rne(item / scale) * scale for item in values]


class Nvfp4RoundTripTest(unittest.TestCase):
    def test_matches_scalar_product_semantics(self) -> None:
        rng = random.Random(7)
        rows = []
        for _ in range(64):
            rows.append([rng.gauss(0.0, rng.choice((0.01, 1.0, 30.0))) for _ in range(16)])
        # Exact E2M1 ties after a power-of-two scale (amax 6 -> scale 1).
        rows.append([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -5.0, 0, 0, 0, 0, 0])
        rows.append([0.0] * 16)
        source = torch.tensor(rows, dtype=torch.float32)
        for four_over_six in (False, True):
            actual = kv_eval_codecs.round_trip_nvfp4(source, four_over_six=four_over_six)
            for row, decoded in zip(rows, actual.tolist(), strict=True):
                six = scalar_nvfp4_group(row, 6.0)
                expected = six
                if four_over_six:
                    four = scalar_nvfp4_group(row, 4.0)
                    six_error = sum((a - b) ** 2 for a, b in zip(six, row, strict=True))
                    four_error = sum((a - b) ** 2 for a, b in zip(four, row, strict=True))
                    expected = four if four_error < six_error else six
                for got, want in zip(decoded, expected, strict=True):
                    self.assertAlmostEqual(got, want, places=6)

    def test_four_over_six_never_increases_error(self) -> None:
        source = torch.randn(4096, 256, generator=torch.Generator().manual_seed(3))
        plain = kv_eval_codecs.round_trip_nvfp4(source, four_over_six=False)
        chosen = kv_eval_codecs.round_trip_nvfp4(source, four_over_six=True)
        self.assertLess(
            float((chosen - source).square().sum()), float((plain - source).square().sum())
        )


class TurboQuantRoundTripTest(unittest.TestCase):
    def test_rotation_is_orthonormal(self) -> None:
        matrix = kv_eval_codecs.hadamard_matrix(torch.device("cpu"))
        self.assertTrue(torch.equal(matrix @ matrix.T, torch.eye(256)))
        signs = kv_eval_codecs.rotation_signs()
        self.assertEqual(set(signs), {-1, 1})
        self.assertLess(abs(sum(signs)), 48)

    def test_reconstruction_against_float64_definition(self) -> None:
        source = torch.randn(64, 256, generator=torch.Generator().manual_seed(5)) * 3.0
        for name in ("tq3", "tq3p5", "tq4"):
            codec = kv_eval_codecs.TurboQuantRoundTrip(name, torch.device("cpu"))
            actual = codec(source).double()
            matrix = kv_eval_codecs.hadamard_matrix(torch.device("cpu")).double()
            signs = torch.tensor(kv_eval_codecs.rotation_signs(), dtype=torch.float64)
            bits = kv_eval_codecs.TQ_BITS[name]
            for row, decoded in zip(source.double(), actual, strict=True):
                norm = float(row.norm())
                unit = (matrix @ (signs * row)) * 16.0 / norm
                codes = []
                for value, width in zip(unit.tolist(), bits, strict=True):
                    positive = kv_eval_codecs.LLOYD_MAX_LEVELS[width]
                    levels = [-level for level in positive] + list(positive)
                    codes.append(min(levels, key=lambda level: abs(level - value)))
                code = torch.tensor(codes, dtype=torch.float64)
                scale = kv_codec.fp16_to_fp32(kv_codec.fp32_to_fp16_rne(norm / float(code.norm())))
                expected = signs * (matrix.T @ (code * scale))
                self.assertLess(float((decoded - expected).abs().max()), 1e-5 * norm)
                self.assertAlmostEqual(float(decoded.norm()) / norm, 1.0, delta=1e-3)

    def test_error_ordering_on_gaussian_vectors(self) -> None:
        source = torch.randn(4096, 256, generator=torch.Generator().manual_seed(11))
        energy = float(source.square().sum())
        errors = {}
        for name in ("tq3", "tq3p5", "tq4"):
            codec = kv_eval_codecs.TurboQuantRoundTrip(name, torch.device("cpu"))
            errors[name] = float((codec(source) - source).square().sum()) / energy
        nvfp4 = kv_eval_codecs.round_trip_nvfp4(source, four_over_six=False)
        errors["nvfp4"] = float((nvfp4 - source).square().sum()) / energy
        # Expected relative MSE about 0.0094 / 0.022 / 0.034 and NVFP4 0.0090 (norm-corrected
        # reconstruction is slightly above the plain Lloyd-Max distortion).
        self.assertLess(errors["tq4"], 0.0115)
        self.assertLess(errors["tq3p5"], 0.026)
        self.assertLess(errors["tq3"], 0.040)
        self.assertLess(errors["tq4"], errors["tq3p5"])
        self.assertLess(errors["tq3p5"], errors["tq3"])
        self.assertLess(errors["nvfp4"], 0.0095)

    def test_zero_vector_decodes_to_zero(self) -> None:
        codec = kv_eval_codecs.TurboQuantRoundTrip("tq4", torch.device("cpu"))
        self.assertTrue(torch.equal(codec(torch.zeros(2, 256)), torch.zeros(2, 256)))


class SchemeRegistryTest(unittest.TestCase):
    def test_every_scheme_names_a_codec_pair(self) -> None:
        self.assertEqual(
            set(protocol.ROUND_TRIP_CACHE_SCHEMES.values()), set(kv_eval_codecs.KEY_VALUE_CODECS)
        )
        self.assertEqual(kv_eval_codecs.head_token_bytes("nvfp4"), 288)
        self.assertEqual(kv_eval_codecs.head_token_bytes("k-tq4-v-tq3"), 228)


if __name__ == "__main__":
    unittest.main()
