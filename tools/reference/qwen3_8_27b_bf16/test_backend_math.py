"""Optional short-tensor cross-checks for the scorer's accelerator math routes."""

from __future__ import annotations

import unittest


try:
    import torch
    import fla  # noqa: F401

    _DEPENDENCIES = torch.cuda.is_available()
except (ImportError, OSError):
    _DEPENDENCIES = False


@unittest.skipUnless(_DEPENDENCIES, "ROCm PyTorch and FLA are not installed")
class FLaCrossCheckTest(unittest.TestCase):
    def test_attention_pv_uses_ordered_full_and_partial_absolute_chunks(self) -> None:
        from .backend import _attention_pv
        from .protocol import ATTENTION_PV_SOURCE_ROW_CHUNK

        device = torch.device("cuda", 0)
        for source_rows in (ATTENTION_PV_SOURCE_ROW_CHUNK,
                            ATTENTION_PV_SOURCE_ROW_CHUNK + 1):
            with self.subTest(source_rows=source_rows):
                probabilities = torch.arange(
                    2 * 2 * source_rows, device=device, dtype=torch.float32
                ).reshape(2, 2, source_rows)
                probabilities = probabilities.remainder(29).mul_(1.0 / 31.0)
                values = torch.arange(
                    source_rows * 3, device=device, dtype=torch.float32
                ).reshape(source_rows, 3)
                values = values.remainder(23).add_(-11.0).mul_(1.0 / 17.0)

                expected = torch.zeros((4, 3), device=device, dtype=torch.float32)
                for begin in range(0, source_rows, ATTENTION_PV_SOURCE_ROW_CHUNK):
                    end = min(source_rows, begin + ATTENTION_PV_SOURCE_ROW_CHUNK)
                    expected.add_(torch.mm(
                        probabilities[..., begin:end].reshape(4, end - begin),
                        values[begin:end],
                    ))
                actual = _attention_pv(probabilities, values)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual, expected.reshape(2, 2, 3)))

                fp64_oracle = torch.einsum(
                    "ths,sd->thd", probabilities.double(), values.double()
                )
                self.assertTrue(torch.allclose(
                    actual.double(), fp64_oracle, atol=2e-3, rtol=2e-5
                ))

    def test_attention_pv_rejects_mismatched_source_rows(self) -> None:
        from .backend import _attention_pv

        device = torch.device("cuda", 0)
        with self.assertRaisesRegex(ValueError, "source-row counts differ"):
            _attention_pv(
                torch.zeros((1, 1, 3), device=device, dtype=torch.float32),
                torch.zeros((2, 1), device=device, dtype=torch.float32),
            )

    def test_fused_recurrent_and_state_continuation_match_explicit_recurrence(self) -> None:
        from .backend import _gdn_recurrence, _gdn_recurrence_naive

        torch.manual_seed(17)
        device = torch.device("cuda", 0)
        rows = 5
        q = torch.randn(rows, 16, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        q = (q.float() / torch.linalg.vector_norm(q.float(), dim=-1, keepdim=True)).to(
            torch.bfloat16
        )
        k = (k.float() / torch.linalg.vector_norm(k.float(), dim=-1, keepdim=True)).to(
            torch.bfloat16
        )
        value = torch.randn(rows, 48, 128, device=device, dtype=torch.bfloat16)
        decay = -torch.rand(rows, 48, device=device, dtype=torch.float32)
        beta = torch.rand(rows, 48, device=device, dtype=torch.float32)
        state = torch.randn(
            48, 128, 128, device=device, dtype=torch.float32
        ) / 128.0

        expected, expected_state = _gdn_recurrence_naive(
            q, k, value, decay, beta, state.clone()
        )
        actual, actual_state = _gdn_recurrence(q, k, value, decay, beta, state.clone())
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2))
        self.assertTrue(
            torch.allclose(actual_state, expected_state, atol=2e-2, rtol=2e-2)
        )

        # Long scoring calls the fused recurrence once per prefill span and carries the returned
        # FP32 state into the next call. Exercise that seam independently of the single call.
        first, carried = _gdn_recurrence(
            q[:3], k[:3], value[:3], decay[:3], beta[:3], state.clone()
        )
        second, segmented_state = _gdn_recurrence(
            q[3:], k[3:], value[3:], decay[3:], beta[3:], carried
        )
        segmented = torch.cat((first, second), dim=0)
        self.assertTrue(
            torch.allclose(segmented.float(), expected.float(), atol=2e-2, rtol=2e-2)
        )
        self.assertTrue(
            torch.allclose(segmented_state, expected_state, atol=2e-2, rtol=2e-2)
        )

    def test_single_token_dispatch_is_the_explicit_recurrence(self) -> None:
        from .backend import _gdn_recurrence, _gdn_recurrence_naive

        torch.manual_seed(29)
        device = torch.device("cuda", 0)
        q = torch.randn(1, 16, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        value = torch.randn(1, 48, 128, device=device, dtype=torch.bfloat16)
        decay = -torch.rand(1, 48, device=device, dtype=torch.float32)
        beta = torch.rand(1, 48, device=device, dtype=torch.float32)
        state = torch.randn(48, 128, 128, device=device, dtype=torch.float32) / 128.0
        expected, expected_state = _gdn_recurrence_naive(
            q, k, value, decay, beta, state.clone()
        )
        actual, actual_state = _gdn_recurrence(
            q, k, value, decay, beta, state.clone()
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_state, expected_state))


if __name__ == "__main__":
    unittest.main()
