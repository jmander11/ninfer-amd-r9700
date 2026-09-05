from __future__ import annotations

import copy
import unittest

from .compare_stage_traces import _expected_checkpoints, compare, validate_trace
from .stage_trace import (
    LOGIT_ROWS, TRACE_ENVIRONMENT_KEYS, TRACE_ROWS, TYPE, VERSION,
    resolved_preferred_blas_library,
)
from .protocol import (
    ATTENTION_PV_EXECUTION,
    DETERMINISTIC_ENVIRONMENT,
    DETERMINISTIC_EXECUTION_PROFILE,
    GDN_RECURRENCE_EXECUTION,
    MATMUL_REDUCTION_EXECUTION,
    TRITON_CODEGEN_EXECUTION,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


class StageTraceContractTest(unittest.TestCase):
    def trace(self, run_id: str = "run-a") -> dict:
        prompt_tokens = 129
        skip_tokens = 32
        checkpoints = [
            {
                "name": name,
                "begin": begin,
                "end": end,
                "dtype": (
                    "torch.float32"
                    if any(token in name for token in (
                        "masked-qk-fp32", "softmax-fp32", "pv-fp32"
                    ))
                    else "torch.bfloat16"
                ),
                "shape": shape,
                "sha256": SHA_A,
                "finite": "masked-qk-fp32" not in name,
                "nan": False,
                "positive_infinity": False,
                "negative_infinity": "masked-qk-fp32" in name,
            }
            for name, begin, end, shape in _expected_checkpoints(prompt_tokens, skip_tokens)
        ]
        return {
            "artifact_type": TYPE,
            "schema_version": VERSION,
            "quality_evidence": False,
            "prompt_tokens": prompt_tokens,
            "skip_tokens": skip_tokens,
            "prefill_chunk": 4096,
            "trace_rows": TRACE_ROWS,
            "logit_rows": LOGIT_ROWS,
            "run": {"id": run_id, "process_id": 123},
            "source": {
                "config_sha256": SHA_A,
                "index_sha256": SHA_A,
                "shards_sha256": {"model.safetensors": SHA_A},
                "corpus_ids_sha256": SHA_A,
            },
            "provenance": {
                "execution_profile": DETERMINISTIC_EXECUTION_PROFILE,
                "attention_pv": dict(ATTENTION_PV_EXECUTION),
                "gdn_recurrence": dict(GDN_RECURRENCE_EXECUTION),
                "tunable_op": {"enabled": False},
                "triton_codegen": dict(TRITON_CODEGEN_EXECUTION),
                "python": "3.11",
                "python_executable": "/python",
                "python_executable_sha256": SHA_A,
                "platform": "test",
                "torch": "test",
                "torch_git": "test",
                "hip": "test",
                "device_index": 0,
                "device_name": "test",
                "device_arch": "gfx1201",
                "matmul_precision": "highest",
                "preferred_blas_library": "hipblas",
                "deterministic_algorithms": {"enabled": True, "warn_only": False},
                "matmul_reduction": dict(MATMUL_REDUCTION_EXECUTION),
                "environment": {
                    key: DETERMINISTIC_ENVIRONMENT.get(key)
                    for key in TRACE_ENVIRONMENT_KEYS
                },
                "scorer_python_tree_sha256": SHA_A,
                "fla_python_tree_sha256": SHA_A,
                "distributions": {"torch": {"version": "test"}},
            },
            "checkpoints": checkpoints,
            "scores": {"nll_sha256": SHA_A, "argmax_sha256": SHA_A},
        }


    def test_compare_exact_and_first_mismatch(self) -> None:
        first = self.trace()
        second = self.trace("run-b")
        self.assertTrue(compare(first, second)["exact"])
        second["checkpoints"][4]["sha256"] = SHA_B
        result = compare(first, second)
        self.assertFalse(result["exact"])
        self.assertEqual(result["first_mismatch"], second["checkpoints"][4]["name"])

    def test_layer_three_attention_detail_inventory_is_bounded_and_ordered(self) -> None:
        rows = _expected_checkpoints(4097, 2048)
        names = [row[0] for row in rows]
        detail = [row for row in rows if ".attention-detail." in row[0]]
        self.assertEqual(len(detail), 22)
        self.assertEqual(detail[0], (
            "layer-03.attention-detail.normalized-input.last",
            4032, 4096, [64, 5120],
        ))
        self.assertEqual(detail[-1], (
            "layer-03.attention-detail.post-residual.last",
            4032, 4096, [64, 5120],
        ))
        self.assertLess(
            names.index("layer-03.attention-detail.post-residual.last"),
            names.index("layer-03.attention.first"),
        )
        self.assertFalse(any("layer-07.attention-detail" in name for name in names))
        kernel = [row for row in detail if ".kv-" in row[0]]
        self.assertEqual(len(kernel), 12)
        self.assertEqual(kernel[0], (
            "layer-03.attention-detail.kv-00.masked-qk-fp32.last",
            4032, 4096, [64, 6, 4096],
        ))
        self.assertEqual(kernel[2], (
            "layer-03.attention-detail.kv-00.pv-fp32.last",
            4032, 4096, [64, 6, 256],
        ))

    def test_layer_three_attention_detail_collects_only_global_tail(self) -> None:
        class FakeTensor:
            def __init__(self, rows: int, tail: tuple[int, ...] = (5120,)) -> None:
                self.shape = (rows, *tail)

            def __getitem__(self, item: slice):
                start = 0 if item.start is None else item.start
                stop = self.shape[0] if item.stop is None else item.stop
                return FakeTensor(stop - start, self.shape[1:])

            def clone(self):
                return self

        from .stage_trace import StageTrace

        trace = StageTrace(
            None, None, prompt_tokens=130, skip_tokens=64, prefill_chunk=64,
            device_index=0, provenance={},
        )
        trace.capture_attention_detail(3, "normalized-input", FakeTensor(64), 0, 64)
        trace.capture_attention_detail(7, "normalized-input", FakeTensor(64), 64, 128)
        self.assertEqual(trace.items, [])
        trace.capture_attention_detail(3, "normalized-input", FakeTensor(64), 64, 128)
        trace.capture_attention_detail(3, "normalized-input", FakeTensor(1), 128, 129)
        name, begin, end, parts = trace.items[0]
        self.assertEqual(name, "layer-03.attention-detail.normalized-input.last")
        self.assertEqual((begin, end), (65, 129))
        self.assertEqual([part.shape[0] for part in parts], [63, 1])

    def test_attention_kernel_detail_ignores_short_earlier_prefix(self) -> None:
        class FakeTensor:
            def __init__(self, rows: int, tail: tuple[int, ...]) -> None:
                self.shape = (rows, *tail)

            def __getitem__(self, item: slice):
                start = 0 if item.start is None else item.start
                stop = self.shape[0] if item.stop is None else item.stop
                return FakeTensor(stop - start, self.shape[1:])

            def clone(self):
                return self

        from .stage_trace import StageTrace

        trace = StageTrace(
            None, None, prompt_tokens=130, skip_tokens=64, prefill_chunk=64,
            device_index=0, provenance={},
        )
        # The first span sees only its local 64-key prefix and cannot overlap the global tail.
        trace.capture_attention_kernel_detail(
            "masked-qk-fp32", 0, FakeTensor(32, (6, 64)), 0, 32
        )
        self.assertEqual(trace.items, [])
        trace.capture_attention_kernel_detail(
            "masked-qk-fp32", 0, FakeTensor(64, (6, 129)), 65, 129
        )
        name, begin, end, parts = trace.items[0]
        self.assertEqual(name, "layer-03.attention-detail.kv-00.masked-qk-fp32.last")
        self.assertEqual((begin, end), (65, 129))
        self.assertEqual([part.shape[0] for part in parts], [64])

    def test_rejects_provenance_drift(self) -> None:
        first = self.trace()
        second = self.trace("run-b")
        second["source"]["corpus_ids_sha256"] = SHA_B
        with self.assertRaisesRegex(ValueError, "source"):
            compare(first, second)

    def test_rejects_attention_pv_identity_drift(self) -> None:
        trace = self.trace()
        trace["provenance"]["attention_pv"]["source_row_chunk"] = 4096
        with self.assertRaisesRegex(ValueError, "provenance.attention_pv"):
            validate_trace(trace)

    def test_rejects_gdn_recurrence_identity_drift(self) -> None:
        trace = self.trace()
        trace["provenance"]["gdn_recurrence"]["multi_token_implementation"] = (
            "fla.ops.gated_delta_rule.chunk_gated_delta_rule"
        )
        with self.assertRaisesRegex(ValueError, "provenance.gdn_recurrence"):
            validate_trace(trace)

    def test_rejects_tunable_or_codegen_drift(self) -> None:
        trace = self.trace()
        trace["provenance"]["tunable_op"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "provenance.tunable_op"):
            validate_trace(trace)
        trace = self.trace()
        trace["provenance"]["triton_codegen"]["amd_use_async_copy"] = True
        with self.assertRaisesRegex(ValueError, "provenance.triton_codegen"):
            validate_trace(trace)
        trace = self.trace()
        trace["provenance"]["matmul_reduction"]["allow_tf32"] = True
        with self.assertRaisesRegex(ValueError, "provenance.matmul_reduction"):
            validate_trace(trace)

    def test_rejects_noncanonical_or_forbidden_environment(self) -> None:
        trace = self.trace()
        trace["provenance"]["environment"]["TRITON_DEFAULT_FP_FUSION"] = "0"
        with self.assertRaisesRegex(ValueError, "TRITON_DEFAULT_FP_FUSION"):
            validate_trace(trace)
        trace = self.trace()
        trace["provenance"]["environment"]["TRITON_OVERRIDE_ARCH"] = "gfx942"
        with self.assertRaisesRegex(ValueError, "TRITON_OVERRIDE_ARCH"):
            validate_trace(trace)
        trace = self.trace()
        trace["provenance"]["environment"]["PYTORCH_ALLOC_CONF"] = "backend:native"
        with self.assertRaisesRegex(ValueError, "PYTORCH_ALLOC_CONF"):
            validate_trace(trace)

    def test_rejects_malformed_but_identical_inventory(self) -> None:
        malformed = self.trace()
        malformed["checkpoints"] = []
        second = copy.deepcopy(malformed)
        second["run"]["id"] = "run-b"
        with self.assertRaisesRegex(ValueError, "checkpoint inventory"):
            compare(malformed, second)

    def test_rejects_same_run_twice(self) -> None:
        trace = self.trace()
        with self.assertRaisesRegex(ValueError, "distinct runs"):
            compare(trace, copy.deepcopy(trace))

    def test_rejects_wrong_stage_order_and_ranges(self) -> None:
        wrong_order = self.trace()
        wrong_order["checkpoints"][0], wrong_order["checkpoints"][1] = (
            wrong_order["checkpoints"][1], wrong_order["checkpoints"][0]
        )
        with self.assertRaisesRegex(ValueError, "embedding.first"):
            validate_trace(wrong_order)
        wrong_range = self.trace()
        wrong_range["checkpoints"][-1]["begin"] -= 1
        with self.assertRaisesRegex(ValueError, "logits.last"):
            validate_trace(wrong_range)

    def test_rejects_nonfinite_checkpoint_and_bad_score_hash(self) -> None:
        nonfinite = self.trace()
        nonfinite["checkpoints"][0]["finite"] = False
        with self.assertRaisesRegex(ValueError, "embedding.first"):
            validate_trace(nonfinite)
        bad_score = self.trace()
        bad_score["scores"]["nll_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "scores.nll"):
            validate_trace(bad_score)

    def test_masked_qk_allows_only_expected_negative_infinity(self) -> None:
        trace = self.trace()
        masked = next(
            row for row in trace["checkpoints"] if "masked-qk-fp32" in row["name"]
        )
        self.assertFalse(masked["finite"])
        self.assertTrue(masked["negative_infinity"])
        validate_trace(trace)
        masked["nan"] = True
        with self.assertRaisesRegex(ValueError, "masked-qk-fp32"):
            validate_trace(trace)

    def test_requires_complete_determinism_environment(self) -> None:
        trace = self.trace()
        trace["provenance"]["environment"]["FLA_USE_FAST_OPS"] = "0"
        del trace["provenance"]["environment"]["ROCBLAS_DEFAULT_ATOMICS_MODE"]
        with self.assertRaisesRegex(ValueError, "provenance.environment"):
            validate_trace(trace)

    def test_rejects_fla_fast_math_environment(self) -> None:
        trace = self.trace()
        trace["provenance"]["environment"]["FLA_USE_FAST_OPS"] = "1"
        with self.assertRaisesRegex(ValueError, "FLA_USE_FAST_OPS"):
            validate_trace(trace)

    def test_normalizes_resolved_pytorch_blas_backend_for_rocm(self) -> None:
        class Backend:
            def __init__(self, name: str) -> None:
                self.name = name

        class Cuda:
            @staticmethod
            def preferred_blas_library():
                return Backend("Cublaslt")

        class Torch:
            class backends:
                cuda = Cuda()

            class version:
                hip = "7.2"

        self.assertEqual(resolved_preferred_blas_library(Torch()), "hipblaslt")
        Torch.backends.cuda.preferred_blas_library = lambda: Backend("Cublas")
        self.assertEqual(resolved_preferred_blas_library(Torch()), "hipblas")
        Torch.backends.cuda.preferred_blas_library = lambda: Backend("Default")
        self.assertEqual(resolved_preferred_blas_library(Torch()), "default")
        Torch.backends.cuda.preferred_blas_library = lambda: Backend("Ck")
        self.assertEqual(resolved_preferred_blas_library(Torch()), "ck")
        Torch.backends.cuda.preferred_blas_library = lambda: Backend("Unknown")
        with self.assertRaisesRegex(RuntimeError, "unknown PyTorch preferred BLAS"):
            resolved_preferred_blas_library(Torch())

    def test_rejects_invalid_resolved_blas_backend(self) -> None:
        trace = self.trace()
        trace["provenance"]["preferred_blas_library"] = "auto"
        with self.assertRaisesRegex(ValueError, "preferred_blas_library"):
            validate_trace(trace)


if __name__ == "__main__":
    unittest.main()
