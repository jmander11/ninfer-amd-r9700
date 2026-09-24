import unittest

import torch

from tools.ppl import selective_a8q4_source_diagnostic as diagnostic


class SelectiveA8Q4SourceDiagnosticTest(unittest.TestCase):
    def test_weight_quantization_is_chunk_independent(self) -> None:
        torch.manual_seed(31)
        weight = (torch.randn(7, 256) * 0.1).to(torch.bfloat16)
        for profile in diagnostic.PROFILES:
            first = diagnostic.quantize_weight_rows(weight, profile, row_chunk=2)
            second = diagnostic.quantize_weight_rows(weight, profile, row_chunk=7)
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))

    def test_activation_codec_has_exact_represented_fp16_scale(self) -> None:
        source = torch.zeros((2, 128), dtype=torch.bfloat16)
        source[0, :4] = torch.tensor([-127.0, -126.5, 126.5, 127.0], dtype=torch.bfloat16)
        source[1, 0] = torch.tensor(2.0**-126, dtype=torch.bfloat16)
        codes, scales = diagnostic.quantize_activation(source, 128)
        self.assertEqual(scales.dtype, torch.float16)
        self.assertEqual(int(codes[0, 0, 0]), -127)
        self.assertEqual(int(codes[0, 0, 3]), 127)
        self.assertEqual(scales[0, 0].item(), 1.0)
        self.assertGreater(scales[1, 0].item(), 0.0)

    def test_grouped_formula_matches_independent_small_integer_oracle(self) -> None:
        activation = torch.tensor(
            [[1.0, -2.0] * 32, [-1.0, 3.0] * 32], dtype=torch.bfloat16
        )
        weight = torch.tensor(
            [[2.0, -1.0] * 32, [-2.0, 1.0] * 32, [1.0, 1.0] * 32],
            dtype=torch.bfloat16,
        )
        codes, scales = diagnostic.quantize_weight_rows(
            weight, "a8g64-q4g64-control", row_chunk=1
        )
        span = diagnostic.QuantizedRowSpan(0, 3, codes, scales)
        actual = diagnostic.represented_linear(activation, span, 64)
        acodes, ascales = diagnostic.quantize_activation(activation, 64)
        expected = torch.empty((2, 3), dtype=torch.bfloat16)
        for token in range(2):
            for row in range(3):
                total = torch.tensor(0.0, dtype=torch.float32)
                dot = sum(int(acodes[token, 0, k]) * int(codes[row, 0, k]) for k in range(64))
                total = torch.tensor(
                    float(dot) * float(ascales[token, 0]) * float(scales[row, 0]),
                    dtype=torch.float32,
                )
                expected[token, row] = total.to(torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_grouped_formula_accumulates_two_groups_before_one_bf16_cast(self) -> None:
        activation = torch.tensor(
            [[1.0, -2.0] * 64, [-1.0, 3.0] * 64], dtype=torch.bfloat16
        )
        weight = torch.tensor(
            [[2.0, -1.0] * 64, [-2.0, 1.0] * 64], dtype=torch.bfloat16
        )
        codes, scales = diagnostic.quantize_weight_rows(
            weight, "a8g64-q4g64-control", row_chunk=1
        )
        span = diagnostic.QuantizedRowSpan(0, 2, codes, scales)
        actual = diagnostic.represented_linear(activation, span, 64)
        acodes, ascales = diagnostic.quantize_activation(activation, 64)
        expected = torch.empty((2, 2), dtype=torch.bfloat16)
        for token in range(2):
            for row in range(2):
                total = torch.tensor(0.0, dtype=torch.float32)
                for group in range(2):
                    dot = sum(
                        int(acodes[token, group, k]) * int(codes[row, group, k])
                        for k in range(64)
                    )
                    factor = torch.tensor(
                        float(ascales[token, group]) * float(scales[row, group]),
                        dtype=torch.float32,
                    )
                    total = torch.addcmul(
                        total, torch.tensor(float(dot), dtype=torch.float32), factor
                    )
                expected[token, row] = total.to(torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_scoped_dispatch_preserves_bf16_prefix_and_restores_backend(self) -> None:
        class Backend:
            @staticmethod
            def _linear(x, weight):
                return (x @ weight.t()).to(torch.bfloat16)

        backend = Backend()
        checkpoint = diagnostic.ActivationInclusiveCheckpoint(None, "a8g64-q4g64-control")
        activation = torch.ones((1, 64), dtype=torch.bfloat16)
        weight = torch.ones((3, 64), dtype=torch.bfloat16)
        codes, scales = diagnostic.quantize_weight_rows(weight[1:], checkpoint.profile)
        checkpoint.bindings[id(weight)] = (
            weight, (diagnostic.QuantizedRowSpan(1, 3, codes, scales),)
        )
        original = backend._linear
        with diagnostic.ScopedLinearDispatcher(backend, checkpoint):
            result = backend._linear(activation, weight)
            self.assertEqual(result.shape, (1, 3))
            self.assertEqual(result[0, 0], original(activation, weight[:1])[0, 0])
        self.assertIs(backend._linear, original)

    def test_scoped_dispatch_restores_after_exception(self) -> None:
        class Backend:
            @staticmethod
            def _linear(x, weight):
                return x

        backend = Backend()
        checkpoint = diagnostic.ActivationInclusiveCheckpoint(None, "a8g128-q4g128-mse")
        original = backend._linear
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with diagnostic.ScopedLinearDispatcher(backend, checkpoint):
                raise RuntimeError("fixture")
        self.assertIs(backend._linear, original)


if __name__ == "__main__":
    unittest.main()
