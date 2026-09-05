#!/usr/bin/env python3
"""Focused tests for aligned PPL quality gates and exact execution parity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
import validate_fp8_hybrid_execution_gate as hybrid_execution_gate


def make_cell(*, mean_nll: float = 2.0, terrible_tokens: int = 1) -> dict:
    return {
        "prompt_tokens": 4,
        "skip_tokens": 0,
        "tokens_scored": 3,
        "argmax_tokens": 3,
        "non_finite": 0,
        "mean_nll": mean_nll,
        "terrible_tokens": terrible_tokens,
    }


def bf16_execution_provenance() -> dict:
    return {
        "profile": run.BF16_DETERMINISTIC_EXECUTION_PROFILE,
        "python": "3.11 fixture",
        "python_executable": "/fixture/python3",
        "python_executable_sha256": "1" * 64,
        "platform": "Linux-fixture",
        "torch": "fixture",
        "torch_git": "fixture",
        "hip": "fixture",
        "device_index": 0,
        "device_name": "AMD Radeon AI PRO R9700",
        "device_arch": "gfx1201",
        "stage_trace_enabled": False,
        "attention_pv": dict(run.BF16_ATTENTION_PV_EXECUTION),
        "gdn_recurrence": dict(run.BF16_GDN_RECURRENCE_EXECUTION),
        "tunable_op": {"enabled": False},
        "triton_codegen": dict(run.BF16_TRITON_CODEGEN_EXECUTION),
        "matmul_precision": "highest",
        "preferred_blas_library": "hipblas",
        "deterministic_algorithms": {"enabled": True, "warn_only": False},
        "matmul_reduction": {
            "allow_tf32": False,
            "allow_fp16_reduced_precision_reduction": True,
            "allow_bf16_reduced_precision_reduction": True,
            "allow_fp16_accumulation": False,
        },
        "environment": {
            **{key: None for key in run.BF16_EXECUTION_ENVIRONMENT_KEYS},
            **run.BF16_DETERMINISTIC_ENVIRONMENT,
        },
        "scorer_python_tree_sha256": "2" * 64,
        "fla_python_tree_sha256": "3" * 64,
        "distributions": {
            name: {"version": "fixture", "record_sha256": None, "direct_url_sha256": None}
            for name in ("torch", "triton", "flash-linear-attention", "safetensors")
        },
    }


class Bf16ExecutionProvenanceTest(unittest.TestCase):
    def test_requires_exact_structured_attention_pv_identity(self) -> None:
        value = bf16_execution_provenance()
        encoded = run.validate_bf16_execution_provenance(value)
        self.assertEqual(json.loads(encoded), value)

        value["attention_pv"]["source_row_chunk"] = 4096
        with self.assertRaisesRegex(SystemExit, "attention_pv"):
            run.validate_bf16_execution_provenance(value)

    def test_rejects_superseded_v2_profile(self) -> None:
        value = bf16_execution_provenance()
        value["profile"] = "rocm-hipblas-no-atomics-strict-deterministic-pv-v2"
        with self.assertRaisesRegex(SystemExit, "profile"):
            run.validate_bf16_execution_provenance(value)

    def test_rejects_chunk_gdn_recurrence_identity(self) -> None:
        value = bf16_execution_provenance()
        value["gdn_recurrence"]["multi_token_implementation"] = (
            "fla.ops.gated_delta_rule.chunk_gated_delta_rule"
        )
        with self.assertRaisesRegex(SystemExit, "gdn_recurrence"):
            run.validate_bf16_execution_provenance(value)

    def test_rejects_fla_fast_math_execution(self) -> None:
        value = bf16_execution_provenance()
        value["environment"]["FLA_USE_FAST_OPS"] = "1"
        with self.assertRaisesRegex(SystemExit, "deterministic execution environment"):
            run.validate_bf16_execution_provenance(value)

    def test_rejects_tunable_or_triton_codegen_drift(self) -> None:
        value = bf16_execution_provenance()
        value["tunable_op"]["enabled"] = True
        with self.assertRaisesRegex(SystemExit, "tunable_op"):
            run.validate_bf16_execution_provenance(value)
        value = bf16_execution_provenance()
        value["triton_codegen"]["amd_global_prefetch"] = 1
        with self.assertRaisesRegex(SystemExit, "triton_codegen"):
            run.validate_bf16_execution_provenance(value)
        value = bf16_execution_provenance()
        value["matmul_reduction"]["allow_tf32"] = True
        with self.assertRaisesRegex(SystemExit, "noncanonical matmul reductions"):
            run.validate_bf16_execution_provenance(value)

    def test_rejects_noncanonical_or_forbidden_environment(self) -> None:
        value = bf16_execution_provenance()
        value["environment"]["TRITON_DEFAULT_FP_FUSION"] = "0"
        with self.assertRaisesRegex(SystemExit, "deterministic execution environment"):
            run.validate_bf16_execution_provenance(value)
        value = bf16_execution_provenance()
        value["environment"]["TRITON_OVERRIDE_ARCH"] = "gfx942"
        with self.assertRaisesRegex(SystemExit, "forbidden execution override"):
            run.validate_bf16_execution_provenance(value)


class ExactArgmaxStatsTest(unittest.TestCase):
    def test_exact_nonempty_sequence(self) -> None:
        stats = run.exact_argmax_stats([4, 5, 6], [4, 5, 6])
        self.assertTrue(stats["argmax_exact"])
        self.assertEqual(stats["argmax_mismatches"], 0)
        self.assertIsNone(stats["argmax_first_mismatch"])

    def test_shift_and_missing_tail_are_not_hidden(self) -> None:
        shifted = run.exact_argmax_stats([4, 7, 6], [4, 5, 6])
        self.assertFalse(shifted["argmax_exact"])
        self.assertEqual(shifted["argmax_mismatches"], 1)
        self.assertEqual(shifted["argmax_first_mismatch"], 1)

        truncated = run.exact_argmax_stats([4, 5], [4, 5, 6])
        self.assertFalse(truncated["argmax_exact"])
        self.assertEqual(truncated["argmax_mismatches"], 1)
        self.assertEqual(truncated["argmax_first_mismatch"], 2)

    def test_empty_sequences_are_not_exact(self) -> None:
        self.assertFalse(run.exact_argmax_stats([], [])["argmax_exact"])


class PairedNllTest(unittest.TestCase):
    def test_pairing_requires_equal_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "lost alignment"):
            run.paired_nll_stats([1.0, 2.0], [1.0])
        with self.assertRaisesRegex(ValueError, "lost alignment"):
            run.paired_delta_se([1.0], [1.0, 2.0])

    def test_delta_summary_retains_signed_worst_positions(self) -> None:
        stats = run.paired_delta_summary([1.0, 2.5, 1.0, 5.0], [1.0, 2.0, 3.0, 4.0])
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats["delta_abs_max"], 2.0)
        self.assertEqual(stats["worst_delta_tokens"][0]["position"], 2)
        self.assertEqual(stats["worst_delta_tokens"][0]["delta_nll"], -2.0)
        self.assertGreaterEqual(stats["delta_abs_p99"], stats["delta_abs_p95"])

    def test_delta_summary_rejects_nonfinite_sidecars(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run.paired_delta_summary([1.0, float("nan")], [1.0, 2.0])

    def test_severe_membership_uses_new_positions_not_aggregate_count(self) -> None:
        stats = run.severe_position_stats(
            [10.0, 2.0, 3.0],
            [1.0, 10.0, 3.0],
            threshold=10.0,
            maximum_new_rate=0.0,
            minimum_budget=0,
        )
        self.assertEqual(stats["new_severe_positions"], 1)
        self.assertEqual(stats["repaired_severe_positions"], 1)
        self.assertFalse(stats["new_severe_positions_pass"])

    def test_severe_budget_scales_with_a_short_sample_floor(self) -> None:
        stats = run.severe_position_stats(
            [10.0] + [1.0] * 4094,
            [1.0] * 4095,
            threshold=10.0,
            maximum_new_rate=0.001,
            minimum_budget=4,
        )
        self.assertEqual(stats["new_severe_position_budget"], 5)
        self.assertTrue(stats["new_severe_positions_pass"])

    def test_capacity_speed_tier_admits_nine_of_4095_severe_positions(self) -> None:
        tier = run.QUALITY_TIERS["capacity-speed"]
        stats = run.severe_position_stats(
            [10.0] * 9 + [1.0] * (4095 - 9),
            [1.0] * 4095,
            threshold=10.0,
            maximum_new_rate=tier["maximum_new_severe_rate"],
            minimum_budget=tier["minimum_new_severe_budget"],
        )
        self.assertEqual(stats["new_severe_position_budget"], 11)
        self.assertTrue(stats["new_severe_positions_pass"])
        self.assertAlmostEqual(tier["maximum_mean_nll_delta"], math.log(1.05))


class ApplyBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_nlls = [1.0, 2.0, 3.0]
        self.base_argmax = [10, 11, 12]
        self.baseline = make_cell()
        baseline_nll, failed = run.apply_baseline(
            self.baseline,
            self.base_nlls,
            None,
            {},
            run.BASELINE,
            None,
            self.base_argmax,
            None,
            self.baseline["terrible_tokens"],
        )
        self.assertFalse(failed)
        self.assertEqual(baseline_nll, 2.0)

    def apply_candidate(
        self,
        cell: dict,
        nlls: list[float],
        argmax: list[int],
        *,
        maximum_new_severe_rate: float = run.DEFAULT_MAX_NEW_SEVERE_RATE,
        minimum_new_severe_budget: int = run.DEFAULT_MIN_NEW_SEVERE_BUDGET,
    ) -> bool:
        _, failed = run.apply_baseline(
            cell,
            nlls,
            self.baseline["mean_nll"],
            {"r9700-g16": 0.1},
            "r9700-g16",
            self.base_nlls,
            argmax,
            self.base_argmax,
            self.baseline["terrible_tokens"],
            maximum_new_severe_rate,
            minimum_new_severe_budget,
        )
        return failed

    def test_exact_candidate_with_no_new_terrible_tokens_passes(self) -> None:
        candidate = make_cell(mean_nll=2.05, terrible_tokens=1)
        self.assertFalse(self.apply_candidate(candidate, [1.05, 2.05, 3.05], [10, 11, 12]))
        self.assertTrue(candidate["pass"])
        self.assertEqual(candidate["terrible_delta"], 0)

    def test_argmax_flip_is_diagnostic_with_good_mean_nll(self) -> None:
        candidate = make_cell(mean_nll=1.9, terrible_tokens=0)
        self.assertFalse(self.apply_candidate(candidate, [0.9, 1.9, 2.9], [10, 99, 12]))
        self.assertTrue(candidate["pass"])
        self.assertFalse(candidate["argmax_exact"])
        self.assertFalse(candidate["argmax_identity_is_gate"])
        self.assertAlmostEqual(candidate["argmax_flip_rate"], 1.0 / 3.0)

    def test_new_severe_position_fails_even_when_aggregate_count_is_unchanged(self) -> None:
        candidate = make_cell(mean_nll=2.0, terrible_tokens=1)
        failed = self.apply_candidate(
            candidate,
            [10.0, 2.0, 3.0],
            self.base_argmax,
            maximum_new_severe_rate=0.0,
            minimum_new_severe_budget=0,
        )
        self.assertTrue(failed)
        self.assertEqual(candidate["terrible_delta"], 0)
        self.assertEqual(candidate["new_severe_positions"], 1)
        self.assertFalse(candidate["new_severe_positions_pass"])

    def test_sidecar_truncation_cannot_pass(self) -> None:
        candidate = make_cell(mean_nll=2.0, terrible_tokens=1)
        with self.assertRaisesRegex(ValueError, "lost alignment"):
            self.apply_candidate(candidate, [1.0, 2.0], [10, 11])


class CompiledProfileIdentityTest(unittest.TestCase):
    def write_cell(
        self, path: Path, *, scheme: str, group: int, skip_tokens: int = 128,
        q4_activation_bits: int = 8, w8_activation_bits: int = 8,
        fp8_qk_wmma: bool = True, xattention: bool = False,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "scheme": scheme,
                    "weights": "model.ninfer",
                    "model_id": run.MODEL_ID,
                    "weights_id": "candidate",
                    "kv_format": "fp8-k-int4-v",
                    "kv_value_group": group,
                    "kv_plane_layouts": run.R9700_KV_PLANE_LAYOUTS,
                    "q4_activation_bits": q4_activation_bits,
                    "w8_activation_bits": w8_activation_bits,
                    "fp8_qk_wmma_enabled": fp8_qk_wmma,
                    "fp8_qk_wmma_profile": run.FP8_QK_WMMA_PROFILE,
                    "fp8_qk_wmma_t1_min_context": run.FP8_QK_WMMA_T1_MIN_CONTEXT,
                    "fp8_qk_wmma_t2_min_context": run.FP8_QK_WMMA_T2_MIN_CONTEXT,
                    "xattention_qualification": xattention,
                    **(
                        {
                            "xattention_profile": "b128-s16-tau900",
                            "xattention_find_block": 128,
                            "xattention_stride": 16,
                            "xattention_tau_permille": 900,
                        }
                        if xattention
                        else {}
                    ),
                    "schedule": "decode",
                    "spec": "none",
                    "draft_tokens": 0,
                    "device_graph": True,
                    "prefill_chunk": 4096,
                    "skip_tokens": skip_tokens,
                    "prompt_tokens": 257,
                    "tokens_scored": 128,
                    "argmax_tokens": 128,
                    "non_finite": 0,
                    "terrible_tokens": 0,
                    "terrible_nll": run.TERRIBLE_NLL,
                    "sum_nll": 128.0,
                    "mean_nll": 1.0,
                    "max_nll": 2.0,
                    "ppl": 2.718,
                    "score_seconds": 1.0,
                }
            ),
            encoding="utf-8",
        )
        path.with_suffix(".nllf32").write_bytes(struct.pack("<128f", *([1.0] * 128)))
        path.with_suffix(".argmaxi32").write_bytes(struct.pack("<128i", *([1] * 128)))

    def test_candidate_rejects_mislabeled_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_wrong_group(*_args, **_kwargs) -> None:
                self.write_cell(cell_path, scheme="r9700-g32", group=16)

            with mock.patch.object(run.subprocess, "run", side_effect=write_wrong_group):
                with self.assertRaisesRegex(SystemExit, "kv_value_group=16"):
                    run.run_cell(
                        Path("ninfer-ppl-g32"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g32", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate",
                    )

    def test_candidate_accepts_matching_compiled_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_matching_group(*_args, **_kwargs) -> None:
                self.write_cell(cell_path, scheme="r9700-g16", group=16)

            with mock.patch.object(run.subprocess, "run", side_effect=write_matching_group):
                cell = run.run_cell(
                    Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                    "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                    "candidate",
                )
            self.assertEqual(cell["kv_value_group"], 16)
            self.assertEqual(cell["kv_plane_layouts"], run.R9700_KV_PLANE_LAYOUTS)
            self.assertIn("command", cell)
            self.assertRegex(cell["nll_sha256"], r"^[0-9a-f]{64}$")

    def test_python_scorer_subprocess_uses_runner_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def publish(_command, **_kwargs) -> None:
                cell_path.write_text("{}", encoding="utf-8")
                cell_path.with_suffix(".nllf32").write_bytes(struct.pack("<f", 1.0))
                cell_path.with_suffix(".argmaxi32").write_bytes(struct.pack("<i", 1))

            with mock.patch.object(run, "validate_cell_report"), mock.patch.object(
                run.subprocess, "run", side_effect=publish
            ) as execute:
                cell = run.run_cell(
                    Path("reference.py"), Path("source"), Path("corpus.ids"),
                    run.BASELINE, "prefill", "half", 4, 4096, 0, cell_path, [],
                    run.BF16_WEIGHTS_ID,
                )
            command = execute.call_args.args[0]
            self.assertEqual(command[:2], [sys.executable, "reference.py"])
            self.assertEqual(cell["command"], command)

    def test_python_scorer_preflight_uses_runner_interpreter(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(run.subprocess, "run", return_value=completed) as execute:
            run.preflight_python_scorer(Path("reference.py"))
        command = execute.call_args.args[0]
        self.assertEqual(command[:2], [sys.executable, "-c"])
        self.assertIn("torch.version", command[2])

        failed = subprocess.CompletedProcess([], 1, "", "ModuleNotFoundError: torch")
        with mock.patch.object(run.subprocess, "run", return_value=failed), self.assertRaisesRegex(
            SystemExit, "requires a ROCm PyTorch environment"
        ):
            run.preflight_python_scorer(Path("reference.py"))

    def test_candidate_rejects_mislabeled_plane_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_wrong_layout(*_args, **_kwargs) -> None:
                self.write_cell(cell_path, scheme="r9700-g16", group=16)
                cell = json.loads(cell_path.read_text(encoding="utf-8"))
                cell["kv_plane_layouts"]["key"] = "feature-fastest-page-major"
                cell_path.write_text(json.dumps(cell), encoding="utf-8")

            with mock.patch.object(run.subprocess, "run", side_effect=write_wrong_layout):
                with self.assertRaisesRegex(SystemExit, "kv_plane_layouts"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate",
                    )

    def test_candidate_rejects_mislabeled_q4_activation_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_wrong_width(*_args, **_kwargs) -> None:
                self.write_cell(
                    cell_path, scheme="r9700-g16", group=16, q4_activation_bits=4
                )

            with mock.patch.object(run.subprocess, "run", side_effect=write_wrong_width):
                with self.assertRaisesRegex(SystemExit, "expected compiled A8"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate", 8,
                    )

    def test_candidate_rejects_mislabeled_w8_activation_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_wrong_width(*_args, **_kwargs) -> None:
                self.write_cell(
                    cell_path, scheme="r9700-g16", group=16, w8_activation_bits=16
                )

            with mock.patch.object(run.subprocess, "run", side_effect=write_wrong_width):
                with self.assertRaisesRegex(SystemExit, "w8_activation_bits=16"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate", 8, 8, True,
                    )

    def test_candidate_rejects_mislabeled_fp8_qk_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_wrong_profile(*_args, **_kwargs) -> None:
                self.write_cell(
                    cell_path, scheme="r9700-g16", group=16, fp8_qk_wmma=False
                )

            with mock.patch.object(run.subprocess, "run", side_effect=write_wrong_profile):
                with self.assertRaisesRegex(SystemExit, "fp8_qk_wmma_enabled=False"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate", 8, 8, True,
                    )

    def test_candidate_rejects_stale_fp8_qk_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_stale_classifier(*_args, **_kwargs) -> None:
                self.write_cell(cell_path, scheme="r9700-g16", group=16)
                payload = json.loads(cell_path.read_text(encoding="utf-8"))
                payload["fp8_qk_wmma_t2_min_context"] = 1
                cell_path.write_text(json.dumps(payload), encoding="utf-8")

            with mock.patch.object(run.subprocess, "run", side_effect=write_stale_classifier):
                with self.assertRaisesRegex(SystemExit, "t2_min_context=1"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate", 8, 8, True,
                    )

    def test_candidate_requires_compile_bound_xattention_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_xattention(*_args, **_kwargs) -> None:
                self.write_cell(
                    cell_path, scheme="r9700-g16", group=16, xattention=True
                )

            with mock.patch.object(run.subprocess, "run", side_effect=write_xattention):
                cell = run.run_cell(
                    Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                    "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                    "candidate", 8, 8, True, "b128-s16-tau900",
                )
            self.assertEqual(cell["xattention_profile"], "b128-s16-tau900")

            with mock.patch.object(run.subprocess, "run", side_effect=write_xattention):
                with self.assertRaisesRegex(SystemExit, "xattention_qualification=True"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate",
                    )

            def write_stale_dense(*_args, **_kwargs) -> None:
                self.write_cell(cell_path, scheme="r9700-g16", group=16)
                payload = json.loads(cell_path.read_text(encoding="utf-8"))
                payload["xattention_profile"] = "stale-profile"
                cell_path.write_text(json.dumps(payload), encoding="utf-8")

            with mock.patch.object(run.subprocess, "run", side_effect=write_stale_dense):
                with self.assertRaisesRegex(SystemExit, "dense report retains XAttention fields"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate",
                    )

    def test_candidate_rejects_skip_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell_path = Path(directory) / "cell.json"

            def write_wrong_skip(*_args, **_kwargs) -> None:
                self.write_cell(
                    cell_path, scheme="r9700-g16", group=16, skip_tokens=127
                )

            with mock.patch.object(run.subprocess, "run", side_effect=write_wrong_skip):
                with self.assertRaisesRegex(SystemExit, "skip_tokens=127"):
                    run.run_cell(
                        Path("ninfer-ppl-g16"), Path("model.ninfer"), Path("corpus.ids"),
                        "r9700-g16", "decode", "half", 257, 4096, 0, cell_path, [],
                        "candidate",
                    )


class WeightInputBoundaryTest(unittest.TestCase):
    def test_bf16_reference_requires_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            candidate = root / "candidate.ninfer"
            candidate.write_bytes(b"fixture")
            run.validate_weights_input(run.BASELINE, source)
            with self.assertRaisesRegex(SystemExit, "source directory"):
                run.validate_weights_input(run.BASELINE, candidate)

    def test_candidate_requires_artifact_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            candidate = root / "candidate.ninfer"
            candidate.write_bytes(b"fixture")
            run.validate_weights_input("r9700-g16", candidate)
            with self.assertRaisesRegex(SystemExit, "candidate artifact"):
                run.validate_weights_input("r9700-g16", source)


class CampaignInputProvenanceTest(unittest.TestCase):
    def write_artifact(self, path: Path, weights_id: str, *, tail: bytes = b"") -> None:
        directory = json.dumps(
            {
                "identity": {"model_id": run.MODEL_ID, "weights_id": weights_id},
                "objects": [],
            },
            separators=(",", ":"),
        ).encode()
        path.write_bytes(run.NINFER_PREFIX.pack(run.NINFER_MAGIC, len(directory)) + directory + tail)

    def write_corpus(self, root: Path, text: str) -> Path:
        path = root / "corpus.ids"
        payload = text.encode()
        path.write_bytes(payload)
        (root / "corpus.manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "ninfer_ppl_corpus",
                    "schema_version": 2,
                    "model_id": run.MODEL_ID,
                    "add_special_tokens": False,
                    "chat_template": False,
                    "tokens": len(payload.split()),
                    "ids_sha256": hashlib.sha256(payload).hexdigest(),
                    "source": "test",
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_candidate_pair_requires_identical_bytes_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "g16.ninfer"
            second = root / "g32.ninfer"
            self.write_artifact(first, "candidate")
            self.write_artifact(second, "candidate")
            info = run.require_same_candidate_artifact(first, second)
            self.assertEqual(info["weights_id"], "candidate")
            self.assertEqual(info["sha256"], hashlib.sha256(first.read_bytes()).hexdigest())
            second.write_bytes(second.read_bytes() + b"different")
            with self.assertRaisesRegex(SystemExit, "byte-identical"):
                run.require_same_candidate_artifact(first, second)

    def test_hybrid_artifact_requires_exact_adjacent_conversion_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "hybrid.ninfer"
            decision = run.fp8_hybrid_decision.DECISION
            self.write_artifact(artifact, decision.weights_id)
            artifact_sha256 = run.file_sha256(artifact)
            receipt_path = Path(str(artifact) + ".conversion.json")
            receipt = {
                "identity": {"model_id": run.MODEL_ID, "weights_id": decision.weights_id},
                "target_key": run.fp8_hybrid_inventory.TARGET_KEY,
                "recipe_id": decision.recipe_id,
                "source": {"index_sha256": "1" * 64, "ranking_sha256": "2" * 64},
                "artifact": {
                    "path": str(artifact.resolve()), "bytes": artifact.stat().st_size,
                    "sha256": artifact_sha256,
                },
                "candidate": {
                    "status": "registered-evaluation-only",
                    "weight_recipe_selected": False,
                    "selection_sha256": decision.selection_sha256,
                    "format_counts": run.fp8_hybrid_inventory.FORMAT_COUNTS,
                    "format_encoded_bytes": run.fp8_hybrid_inventory.FORMAT_ENCODED_BYTES,
                    "tensor_encoded_bytes": run.fp8_hybrid_inventory.TENSOR_ENCODED_BYTES,
                    "device_arena_bytes": run.fp8_hybrid_inventory.DEVICE_ARENA_BYTES,
                    "object_plan_sha256": "3" * 64,
                },
                "migration": {
                    "source_artifact": {"sha256": "4" * 64},
                    "source_conversion_receipt": {"sha256": "5" * 64},
                    "transcoder": {"sha256": "6" * 64},
                },
            }
            with (mock.patch.object(run, "_validate_n16_migration_ancestry"),
                  self.assertRaisesRegex(SystemExit, "receipt is missing")):
                run.inspect_candidate_artifact(artifact)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch.object(run, "_validate_n16_migration_ancestry"):
                inspected = run.inspect_candidate_artifact(artifact)
            self.assertEqual(
                inspected["conversion_receipt"]["selection_sha256"],
                decision.selection_sha256,
            )
            run.require_fp8_hybrid_candidate(inspected)
            receipt_bytes = receipt_path.read_bytes()
            receipt_target = root / "receipt-target.json"
            receipt_target.write_bytes(receipt_bytes)
            receipt_path.unlink()
            receipt_path.symlink_to(receipt_target)
            with (mock.patch.object(run, "_validate_n16_migration_ancestry"),
                  self.assertRaisesRegex(SystemExit, "regular file")):
                run.inspect_candidate_artifact(artifact)
            receipt_path.unlink()
            receipt_path.write_bytes(receipt_bytes)
            artifact_alias = root / "hybrid-alias.ninfer"
            artifact_alias.symlink_to(artifact)
            with self.assertRaisesRegex(SystemExit, "nonsymlink"):
                run.inspect_candidate_artifact(artifact_alias)
            with self.assertRaisesRegex(SystemExit, "authority-bound hybrid"):
                run.require_fp8_hybrid_candidate({"weights_id": "different"})
            receipt["candidate"]["selection_sha256"] = "4" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with (mock.patch.object(run, "_validate_n16_migration_ancestry"),
                  self.assertRaisesRegex(SystemExit, "selection authority differs")):
                run.inspect_candidate_artifact(artifact)

    def test_corpus_manifest_hash_decimal_domain_and_count_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = self.write_corpus(root, "1 2 3\n")
            info = run.validate_corpus(corpus, 3)
            self.assertEqual(info["tokens"], 3)
            corpus.write_text("1 -2 3\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "ASCII decimal"):
                run.validate_corpus(corpus, 3)


class ExecutionParityTest(unittest.TestCase):
    def write_sidecars(self, path: Path, nlls: list[float], argmax: list[int]) -> None:
        path.with_suffix(".nllf32").write_bytes(struct.pack("<" + "f" * len(nlls), *nlls))
        path.with_suffix(".argmaxi32").write_bytes(
            struct.pack("<" + "i" * len(argmax), *argmax)
        )

    def test_sidecar_parity_gates_both_nll_and_exact_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "graph.json"
            second = root / "eager.json"
            self.write_sidecars(first, [1.0, 2.0], [10, 11])
            self.write_sidecars(second, [1.0, 2.01], [10, 11])
            self.assertFalse(run.sidecar_parity(first, second, max_abs_nll=0.0)["pass"])
            relaxed = run.sidecar_parity(first, second, max_abs_nll=0.02)
            self.assertTrue(relaxed["pass"])
            self.assertEqual(relaxed["comparison_kind"], "same-route-execution")
            self.assertTrue(relaxed["complete_finite_aligned"])
            self.assertTrue(relaxed["argmax_identity_is_gate"])
            self.write_sidecars(second, [1.0, 2.0], [10, 99])
            self.assertFalse(run.sidecar_parity(first, second, max_abs_nll=0.02)["pass"])


    def test_schedule_comparison_reports_flips_without_gating_on_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefill = root / "prefill.json"
            decode = root / "decode.json"
            self.write_sidecars(prefill, [1.0, 2.0, 3.0], [10, 11, 12])
            self.write_sidecars(decode, [1.01, 1.98, 3.12], [10, 99, 98])
            comparison = run.schedule_sidecar_comparison(
                prefill, decode, max_abs_nll=0.13
            )
            self.assertTrue(comparison["pass"])
            self.assertTrue(comparison["complete_finite_aligned"])
            self.assertFalse(comparison["argmax_identity_is_gate"])
            self.assertFalse(comparison["argmax_exact"])
            self.assertEqual(comparison["argmax_mismatches"], 2)
            self.assertAlmostEqual(comparison["mean_delta_nll"], 0.11 / 3.0, places=6)
            self.assertAlmostEqual(comparison["max_abs_delta_nll"], 0.12, places=6)
            self.assertFalse(
                run.schedule_sidecar_comparison(prefill, decode, max_abs_nll=0.1)["pass"]
            )

    def test_schedule_comparison_requires_finite_aligned_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefill = root / "prefill.json"
            decode = root / "decode.json"
            self.write_sidecars(prefill, [1.0, 2.0], [10, 11])
            self.write_sidecars(decode, [1.0, float("nan")], [10, 11])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                run.schedule_sidecar_comparison(prefill, decode, max_abs_nll=0.2)
            self.write_sidecars(decode, [1.0, 2.0], [10])
            with self.assertRaisesRegex(ValueError, "not position-aligned"):
                run.schedule_sidecar_comparison(prefill, decode, max_abs_nll=0.2)

    def test_cell_rejects_nonfinite_nll_and_out_of_domain_argmax(self) -> None:
        cell = make_cell()
        self.assertFalse(run.cell_ok(cell, [1.0, float("nan"), 3.0], [10, 11, 12]))
        self.assertFalse(run.cell_ok(cell, [1.0, 2.0, 3.0], [10, 11, run.TOKEN_DOMAIN]))


class HybridExecutionGateTest(unittest.TestCase):
    @staticmethod
    def parity() -> dict:
        return {
            "pass": True,
            "max_abs_nll_gate": 0.0,
            "complete_finite_aligned": True,
            "argmax_exact": True,
        }

    def payload(self) -> dict:
        base = {
            "scheme": "r9700-g16", "prompt_tokens": run.DEFAULT_TOKENS,
            "schedule": "decode", "pass": True, "quality_eligible": True,
        }
        return {
            "artifact_type": run.CAMPAIGN_ARTIFACT_TYPE,
            "schema_version": run.CAMPAIGN_SCHEMA_VERSION,
            "required_candidate_identity": "fp8-hybrid-selection-authority",
            "lengths": [run.DEFAULT_TOKENS], "schedules": ["decode"],
            "spec": "mtp", "draft_tokens": 3,
            "execution_parity_max_abs_nll": 0.0, "pass": True,
            "cells": [
                {**base, "spec": "mtp", "draft_tokens": 3, "device_graph": True},
                {**base, "spec": "mtp", "draft_tokens": 3, "device_graph": False,
                 "device_graph_parity": self.parity()},
                {**base, "spec": "none", "draft_tokens": 0, "device_graph": True,
                 "spec_parity": self.parity()},
                {**base, "spec": "mtp", "draft_tokens": 4, "device_graph": True,
                 "draft_window_parity": self.parity()},
            ],
        }

    def test_requires_complete_exact_execution_matrix(self) -> None:
        found = hybrid_execution_gate.validate_execution_coverage(self.payload())
        self.assertEqual(set(found), {"mtp3_graph", "mtp3_eager", "ordinary_graph", "mtp4_graph"})
        broken = self.payload()
        broken["cells"][1]["device_graph_parity"]["argmax_exact"] = False
        with self.assertRaisesRegex(ValueError, "not exact and complete"):
            hybrid_execution_gate.validate_execution_coverage(broken)

    def test_compiled_width_csv_is_profile_bound_and_unique(self) -> None:
        raw = (
            "prefill,max_concurrency,mtp_width,dflash_width,prepared_width\n"
            "4096,4,4,0,1\n4096,4,4,0,2\n4096,4,4,0,4\n4096,4,4,0,4096\n"
        )
        self.assertEqual(
            hybrid_execution_gate.parse_width_inventory(raw, (4096, 4, 4, 0)),
            [1, 2, 4, 4096],
        )
        with self.assertRaisesRegex(ValueError, "not sorted and unique"):
            hybrid_execution_gate.parse_width_inventory(
                raw.replace("\n4096,4,4,0,4", "\n4096,4,4,0,2"), (4096, 4, 4, 0)
            )


class CampaignOrchestrationTest(unittest.TestCase):
    def write_artifact(self, path: Path) -> None:
        directory = json.dumps(
            {
                "identity": {"model_id": run.MODEL_ID, "weights_id": "candidate"},
                "objects": [],
            },
            separators=(",", ":"),
        ).encode()
        path.write_bytes(run.NINFER_PREFIX.pack(run.NINFER_MAGIC, len(directory)) + directory)

    def test_quality_tier_has_no_silent_default(self) -> None:
        argv = ["run.py", "--bf16-reference-weights", "missing"]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
            run.main()
        self.assertEqual(raised.exception.code, 2)

    def test_two_schedules_require_explicit_measured_nll_bound(self) -> None:
        argv = [
            "run.py", "--bf16-reference-weights", "missing", "--quality-tier", "accuracy"
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(
            SystemExit, "schedule-parity-max-abs-nll is required"
        ):
            run.main()

    def test_single_schedule_does_not_require_schedule_nll_bound(self) -> None:
        argv = [
            "run.py", "--bf16-reference-weights", "missing", "--quality-tier", "accuracy",
            "--schedule", "prefill",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(
            SystemExit, "independent BF16 scorer not found"
        ):
            run.main()

    def test_invalid_reuse_authority_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bf16"
            source.mkdir()
            scorer = root / "ppl.py"
            scorer.write_text("fixture", encoding="utf-8")
            ids = root / "corpus.ids"
            ids.write_text("1 2 3 4\n", encoding="utf-8")
            output = root / "must-not-exist"
            argv = [
                "run.py",
                "--bf16-reference-weights", str(source),
                "--bf16-reference-ppl-bin", str(scorer),
                "--ids", str(ids),
                "--profiles", "bf16-reference",
                "--schedule", "prefill", "--spec", "none", "--no-extras",
                "--quality-tier", "accuracy", "--allow-ungated",
                "--reuse-bf16-campaign", str(root / "invalid-campaign.json"),
                "--bf16-repeat-comparison", str(root / "invalid-repeat.json"),
                "--out", str(output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run, "validate_weights_input"
            ), mock.patch.object(
                run, "preflight_python_scorer"
            ), mock.patch.object(
                run, "ensure_corpus"
            ), mock.patch.object(
                run, "validate_corpus", return_value={}
            ), mock.patch.object(
                run, "validate_bf16_repeat_comparison",
                side_effect=SystemExit("invalid reuse authority"),
            ), self.assertRaisesRegex(SystemExit, "invalid reuse authority"):
                run.main()
            self.assertFalse(output.exists())

    def test_python_preflight_failure_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bf16"
            source.mkdir()
            scorer = root / "ppl.py"
            scorer.write_text("fixture", encoding="utf-8")
            output = root / "must-not-exist"
            argv = [
                "run.py", "--bf16-reference-weights", str(source),
                "--bf16-reference-ppl-bin", str(scorer),
                "--profiles", "bf16-reference", "--schedule", "prefill",
                "--spec", "none", "--no-extras", "--quality-tier", "accuracy",
                "--allow-ungated", "--out", str(output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run, "validate_weights_input"
            ), mock.patch.object(
                run, "preflight_python_scorer", side_effect=SystemExit("bad Python")
            ), self.assertRaisesRegex(SystemExit, "bad Python"):
                run.main()
            self.assertFalse(output.exists())

    def test_reused_bf16_campaign_is_hash_and_provenance_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bf16"
            source.mkdir()
            scorer = root / "ppl.py"
            scorer.write_text("fixture", encoding="utf-8")
            ids = root / "corpus.ids"
            ids.write_text("1 2 3 4\n", encoding="utf-8")
            candidate = root / "candidate.ninfer"
            candidate.write_bytes(b"artifact")
            cell_path = root / "dense" / "4.prefill.bf16-reference.json"
            cell_path.parent.mkdir()
            source_shards = {
                f"model-{part:05d}-of-{run.SOURCE_SHARD_COUNT:05d}.safetensors":
                    f"{part:064x}"
                for part in range(1, run.SOURCE_SHARD_COUNT + 1)
            }
            raw = {
                "scheme": run.BASELINE,
                "model_id": run.MODEL_ID,
                "weights_id": run.BF16_WEIGHTS_ID,
                "weight_format": "source-bf16",
                "formula_profile": "checkpoint-direct-qwen3.8-source-bf16",
                "kv_format": "bf16-reference",
                "kv_value_group": None,
                "cache_only_diagnostic": False,
                "cache_diagnostic_scope": None,
                "cache_key_codec": None,
                "cache_value_codec": None,
                "cache_append_boundary": None,
                "cache_use_boundary": None,
                "schedule": "prefill",
                "spec": "none",
                "draft_tokens": 0,
                "speculative_execution": False,
                "device_graph": False,
                "prefill_chunk": 4096,
                "prompt_tokens": 4,
                "skip_tokens": 2,
                "terrible_nll": run.TERRIBLE_NLL,
                "tokens_scored": 1,
                "argmax_tokens": 1,
                "non_finite": 0,
                "terrible_tokens": 0,
                "mean_nll": 1.0,
                "sum_nll": 1.0,
                "max_nll": 1.0,
                "ppl": math.e,
                "source_config_sha256": "a" * 64,
                "source_index_sha256": "b" * 64,
                "source_shards_sha256": source_shards,
                "corpus_ids_sha256": "d" * 64,
                "source_tensor_count": run.SOURCE_TENSOR_COUNT,
                "source_text_tensor_count": run.SOURCE_TEXT_TENSOR_COUNT,
                "source_shard_count": run.SOURCE_SHARD_COUNT,
                "execution_provenance": bf16_execution_provenance(),
            }
            cell_path.write_text(json.dumps(raw), encoding="utf-8")
            cell_path.with_suffix(".nllf32").write_bytes(struct.pack("<f", 1.0))
            cell_path.with_suffix(".argmaxi32").write_bytes(struct.pack("<i", 7))
            command = [
                str(scorer), "--weights", str(source), "--ids", str(ids),
                "--scheme", run.BASELINE, "--schedule", "prefill", "--skip", "half",
                "--tokens", "4", "--prefill-chunk", "4096", "--device", "0",
                "--out-json", str(cell_path),
            ]
            cell = {
                **raw,
                "command": command,
                "nll_sha256": run.file_sha256(cell_path.with_suffix(".nllf32")),
                "argmax_sha256": run.file_sha256(cell_path.with_suffix(".argmaxi32")),
                "complete_finite_aligned": True,
            }
            artifact = {"weights_id": "weights", "sha256": "c" * 64}
            corpus = {"path": str(ids.resolve()), "ids_sha256": "d" * 64}
            scorer_identity = {
                "path": str(scorer.resolve()), "bytes": scorer.stat().st_size,
                "sha256": run.file_sha256(scorer),
            }
            candidate_cells = []
            for profile in ("r9700-g16", "r9700-g32"):
                candidate_path = root / "dense" / f"4.prefill.{profile}.json"
                candidate_raw = {
                    "scheme": profile, "prompt_tokens": 4, "skip_tokens": 2,
                    "tokens_scored": 1, "argmax_tokens": 1, "non_finite": 0,
                    "mean_nll": 1.0,
                }
                candidate_path.write_text(json.dumps(candidate_raw), encoding="utf-8")
                candidate_path.with_suffix(".nllf32").write_bytes(struct.pack("<f", 1.0))
                candidate_path.with_suffix(".argmaxi32").write_bytes(struct.pack("<i", 7))
                candidate_cells.append({
                    **candidate_raw,
                    "command": ["candidate-scorer", "--out-json", str(candidate_path)],
                    "nll_sha256": run.file_sha256(candidate_path.with_suffix(".nllf32")),
                    "argmax_sha256": run.file_sha256(candidate_path.with_suffix(".argmaxi32")),
                    "complete_finite_aligned": True,
                })
            campaign = root / "dense" / "results.json"
            campaign_payload = {
                "artifact_type": run.CAMPAIGN_ARTIFACT_TYPE,
                "schema_version": run.CAMPAIGN_SCHEMA_VERSION,
                "xattention_profile": "dense",
                "model_id": run.MODEL_ID,
                "reference_weights_id": run.BF16_WEIGHTS_ID,
                "candidate_artifact": artifact,
                "corpus": corpus,
                "lengths": [4],
                "skip": "half",
                "prefill_chunk": 4096,
                "schedules": ["prefill"],
                "spec": "none",
                "draft_tokens": 0,
                "quality_tier": "capacity-speed",
                "gates": {
                    "r9700-g16": math.log(1.05), "r9700-g32": math.log(1.05),
                },
                "terrible_nll": run.TERRIBLE_NLL,
                "weights_inputs": {
                    run.BASELINE: str(source), "r9700-g16": str(candidate),
                    "r9700-g32": str(candidate),
                },
                "scorers": {
                    run.BASELINE: scorer_identity,
                    "r9700-g16": {"path": "/dense-g16", "bytes": 1, "sha256": "e" * 64},
                    "r9700-g32": {"path": "/dense-g32", "bytes": 1, "sha256": "f" * 64},
                },
                "reference_source": {
                    "config_sha256": "a" * 64,
                    "index_sha256": "b" * 64,
                    "shards_sha256": source_shards,
                    "tensor_count": run.SOURCE_TENSOR_COUNT,
                    "text_tensor_count": run.SOURCE_TEXT_TENSOR_COUNT,
                    "shard_count": run.SOURCE_SHARD_COUNT,
                },
                "reference_execution": bf16_execution_provenance(),
                "cells": [cell, *candidate_cells],
                "pass": True,
            }
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            options = dict(
                bf16_weights=source, bf16_scorer=scorer, scorer_identity=scorer_identity,
                ids=ids, corpus_provenance=corpus,
                lengths=[4], skip="half", prefill_chunk=4096, device=0,
            )
            with mock.patch.object(run, "validate_cell_report"):
                reused = run.load_reused_bf16_cells(campaign, **options)
            self.assertEqual(set(reused), {4})
            campaign_payload["cells"][0]["command"] = [
                campaign_payload["cells"][0]["execution_provenance"]["python_executable"],
                *campaign_payload["cells"][0]["command"],
            ]
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"):
                reused = run.load_reused_bf16_cells(campaign, **options)
            self.assertEqual(set(reused), {4})
            campaign_payload["cells"][0]["command"] = command
            campaign_payload["cells"][0]["command"][4] = os.path.relpath(ids, Path.cwd())
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"):
                reused = run.load_reused_bf16_cells(campaign, **options)
            self.assertEqual(set(reused), {4})
            campaign_payload["cells"][0]["command"][12] = "8"
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "command differs"
            ):
                run.load_reused_bf16_cells(campaign, **options)
            campaign_payload["cells"][0]["command"][12] = "4"
            campaign_payload["lengths"] = [4, 8]
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"):
                reused = run.load_reused_bf16_cells(campaign, **options)
            self.assertEqual(set(reused), {4})
            campaign_payload["lengths"] = [8]
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "every requested length"
            ):
                run.load_reused_bf16_cells(campaign, **options)
            campaign_payload["lengths"] = [4]
            campaign_payload["cells"][0]["execution_provenance"]["environment"][
                "ROCBLAS_DEFAULT_ATOMICS_MODE"
            ] = "1"
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "deterministic execution environment"
            ):
                run.load_reused_bf16_cells(campaign, **options)
            campaign_payload["cells"][0]["execution_provenance"] = (
                bf16_execution_provenance()
            )
            campaign_payload["xattention_profile"] = "b128-s16-tau900"
            campaign_payload["quality_tier"] = "accuracy"
            campaign_payload["gates"] = {
                "r9700-g16": math.log(1.02), "r9700-g32": math.log(1.02),
            }
            campaign_payload["pass"] = False
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"):
                reused = run.load_reused_bf16_cells(campaign, **options)
            self.assertEqual(set(reused), {4})
            self.assertEqual(reused[4][0]["reused_bf16_campaign"]["sha256"], run.file_sha256(campaign))
            campaign_payload["candidate_artifact"] = {
                "weights_id": "different-recipe", "sha256": "9" * 64
            }
            campaign_payload["weights_inputs"]["r9700-g16"] = "/different/g16.ninfer"
            campaign_payload["weights_inputs"]["r9700-g32"] = "/different/g32.ninfer"
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"):
                reused = run.load_reused_bf16_cells(campaign, **options)
            self.assertEqual(set(reused), {4})
            campaign_payload["prefill_chunk"] = 2048
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "prefill_chunk"
            ):
                run.load_reused_bf16_cells(campaign, **options)
            campaign_payload["prefill_chunk"] = 4096
            campaign_payload["weights_inputs"][run.BASELINE] = "/different/bf16-source"
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "source path differs"
            ):
                run.load_reused_bf16_cells(campaign, **options)
            campaign_payload["weights_inputs"][run.BASELINE] = str(source)
            campaign_payload["scorers"][run.BASELINE] = {
                **scorer_identity, "sha256": "8" * 64
            }
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "scorer identity differs"
            ):
                run.load_reused_bf16_cells(campaign, **options)
            campaign_payload["scorers"][run.BASELINE] = scorer_identity
            campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
            retained_raw = cell_path.read_text(encoding="utf-8")
            cell_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "raw scorer fields differ"
            ):
                run.load_reused_bf16_cells(campaign, **options)
            cell_path.write_text(retained_raw, encoding="utf-8")
            cell_path.with_suffix(".nllf32").write_bytes(struct.pack("<f", 2.0))
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "sidecar hashes do not match"
            ):
                run.load_reused_bf16_cells(campaign, **options)

    def test_reused_candidate_campaign_is_bound_and_drops_old_bf16_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = root / "corpus.ids"
            ids.write_text("1 2 3 4\n", encoding="utf-8")
            weights = root / "candidate.ninfer"
            weights.write_bytes(b"candidate")
            scorer = root / "candidate-ppl"
            scorer.write_text("scorer", encoding="utf-8")
            cell_path = root / "old" / "4.prefill.r9700-g16.json"
            cell_path.parent.mkdir()
            raw = {field: 0 for field in run.CANDIDATE_SCORER_REPORT_FIELDS}
            raw.update({
                "scheme": "r9700-g16", "prompt_tokens": 4, "skip_tokens": 2,
                "tokens_scored": 1, "argmax_tokens": 1, "non_finite": 0,
                "mean_nll": 1.25,
            })
            cell_path.write_text(json.dumps(raw), encoding="utf-8")
            cell_path.with_suffix(".nllf32").write_bytes(struct.pack("<f", 1.25))
            cell_path.with_suffix(".argmaxi32").write_bytes(struct.pack("<i", 7))
            command = [
                str(scorer), "--weights", str(weights), "--ids", str(ids),
                "--scheme", "r9700-g16", "--schedule", "prefill", "--skip", "half",
                "--tokens", "4", "--prefill-chunk", "4096", "--device", "0",
                "--out-json", str(cell_path),
            ]
            cell = {
                **raw,
                "command": command,
                "nll_sha256": run.file_sha256(cell_path.with_suffix(".nllf32")),
                "argmax_sha256": run.file_sha256(cell_path.with_suffix(".argmaxi32")),
                "delta_mean_nll": 9.0,
                "quality_eligible": False,
            }
            scorer_identity = {
                "path": str(scorer.resolve()), "bytes": scorer.stat().st_size,
                "sha256": run.file_sha256(scorer),
            }
            candidate_artifact = {"weights_id": "candidate", "sha256": "a" * 64}
            corpus = {"path": str(ids.resolve()), "ids_sha256": "b" * 64}
            campaign = root / "old" / "results.json"
            payload = {
                "artifact_type": run.CAMPAIGN_ARTIFACT_TYPE,
                "schema_version": run.CAMPAIGN_SCHEMA_VERSION,
                "model_id": run.MODEL_ID,
                "candidate_artifact": candidate_artifact,
                "corpus": corpus,
                "lengths": [4],
                "skip": "half",
                "prefill_chunk": 4096,
                "schedules": ["prefill"],
                "spec": "none",
                "draft_tokens": 0,
                "terrible_nll": run.TERRIBLE_NLL,
                "q4_activation_bits": 8,
                "w8_activation_bits": 8,
                "candidate_kv_plane_layouts": run.R9700_KV_PLANE_LAYOUTS,
                "fp8_qk_wmma_enabled": True,
                "fp8_qk_wmma_profile": run.FP8_QK_WMMA_PROFILE,
                "fp8_qk_wmma_t1_min_context": run.FP8_QK_WMMA_T1_MIN_CONTEXT,
                "fp8_qk_wmma_t2_min_context": run.FP8_QK_WMMA_T2_MIN_CONTEXT,
                "xattention_profile": "dense",
                "weights_inputs": {"r9700-g16": str(weights)},
                "scorers": {"r9700-g16": scorer_identity},
                "cells": [cell],
            }
            campaign.write_text(json.dumps(payload), encoding="utf-8")
            options = dict(
                selected_candidates=["r9700-g16"],
                candidate_artifact=candidate_artifact,
                profile_weights={"r9700-g16": weights},
                profile_bins={"r9700-g16": scorer},
                scorer_provenance={"r9700-g16": scorer_identity},
                ids=ids,
                corpus_provenance=corpus,
                lengths=[4], skip="half", prefill_chunk=4096, device=0,
                expected_q4_activation_bits=8, expected_w8_activation_bits=8,
                expected_fp8_qk_wmma=True, expected_xattention_profile="dense",
            )
            with mock.patch.object(run, "validate_cell_report"):
                reused = run.load_reused_candidate_cells(campaign, **options)
            imported, imported_path = reused[("r9700-g16", 4)]
            self.assertEqual(imported_path, cell_path)
            self.assertNotIn("delta_mean_nll", imported)
            self.assertNotIn("quality_eligible", imported)
            self.assertEqual(
                imported["reused_candidate_campaign"]["sha256"],
                run.file_sha256(campaign),
            )

            partial_options = {
                **options,
                "selected_candidates": ["r9700-g16", "r9700-g32"],
                "candidate_artifact": {
                    **candidate_artifact,
                    "g16_path": str(weights.resolve()),
                    "g32_path": str(weights.resolve()),
                },
                "profile_weights": {
                    "r9700-g16": weights, "r9700-g32": weights,
                },
                "profile_bins": {
                    "r9700-g16": scorer, "r9700-g32": scorer,
                },
                "scorer_provenance": {
                    "r9700-g16": scorer_identity, "r9700-g32": scorer_identity,
                },
                "lengths": [4, 5],
                "allow_partial": True,
            }
            with mock.patch.object(run, "validate_cell_report"):
                partial = run.load_reused_candidate_cells(campaign, **partial_options)
            self.assertEqual(set(partial), {("r9700-g16", 4)})

            payload["scorers"]["r9700-g16"] = {
                **scorer_identity, "sha256": "c" * 64
            }
            campaign.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "scorer identity differs"
            ):
                run.load_reused_candidate_cells(campaign, **options)
            payload["scorers"]["r9700-g16"] = scorer_identity
            campaign.write_text(json.dumps(payload), encoding="utf-8")
            cell_path.with_suffix(".nllf32").write_bytes(struct.pack("<f", 2.0))
            with mock.patch.object(run, "validate_cell_report"), self.assertRaisesRegex(
                SystemExit, "sidecar hashes do not match"
            ):
                run.load_reused_candidate_cells(campaign, **options)

    def test_both_lengths_receive_candidate_execution_parity_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bf16"
            source.mkdir()
            scorer = root / "scorer"
            scorer.write_text("fixture", encoding="utf-8")
            candidate = root / "candidate.ninfer"
            self.write_artifact(candidate)
            corpus = root / "corpus.ids"
            corpus_payload = (("1 " * (run.LONG_TOKENS - 1)) + "1\n").encode()
            corpus.write_bytes(corpus_payload)
            (root / "corpus.manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "ninfer_ppl_corpus",
                        "schema_version": 2,
                        "model_id": run.MODEL_ID,
                        "add_special_tokens": False,
                        "chat_template": False,
                        "tokens": run.LONG_TOKENS,
                        "ids_sha256": hashlib.sha256(corpus_payload).hexdigest(),
                        "source": "test",
                    }
                ),
                encoding="utf-8",
            )
            output = root / "out"
            calls: list[tuple[str, int, tuple[str, ...]]] = []

            def fake_run_cell(
                _ppl_bin: Path,
                _weights: Path,
                _ids: Path,
                profile_name: str,
                schedule: str,
                skip: str,
                tokens: int,
                _prefill_chunk: int,
                _device: int,
                cell_path: Path,
                extra: list[str],
                _expected_weights_id: str,
                _expected_q4_activation_bits: int,
                _expected_w8_activation_bits: int,
                _expected_fp8_qk_wmma: bool,
                _expected_xattention_profile: str,
            ) -> dict:
                calls.append((profile_name, tokens, tuple(extra)))
                requested_skip = tokens // 2 if skip == "half" else int(skip)
                score_skip = max(requested_skip, 1) if schedule == "decode" else requested_skip
                scored = tokens - score_skip - 1
                cell_path.with_suffix(".nllf32").write_bytes(
                    struct.pack("<" + "f" * scored, *([1.0] * scored))
                )
                cell_path.with_suffix(".argmaxi32").write_bytes(
                    struct.pack("<" + "i" * scored, *([7] * scored))
                )
                cell = {
                    "scheme": profile_name,
                    "prompt_tokens": tokens,
                    "skip_tokens": score_skip,
                    "tokens_scored": scored,
                    "argmax_tokens": scored,
                    "non_finite": 0,
                    "terrible_tokens": 0,
                    "mean_nll": 1.0,
                    "max_nll": 1.0,
                    "ppl": math.e,
                    "schedule": schedule,
                    "spec": run._extra_value(extra, "--spec") or "none",
                    "device_graph": "--no-device-graph" not in extra,
                }
                if profile_name == run.BASELINE:
                    cell.update(
                        {
                            "source_config_sha256": "a" * 64,
                            "source_index_sha256": "b" * 64,
                            "source_shards_sha256": {
                                f"model-{part:05d}-of-{run.SOURCE_SHARD_COUNT:05d}.safetensors":
                                    f"{part:064x}"
                                for part in range(1, run.SOURCE_SHARD_COUNT + 1)
                            },
                            "source_tensor_count": run.SOURCE_TENSOR_COUNT,
                            "source_text_tensor_count": run.SOURCE_TEXT_TENSOR_COUNT,
                            "source_shard_count": run.SOURCE_SHARD_COUNT,
                            "execution_provenance": bf16_execution_provenance(),
                        }
                    )
                cell_path.write_text(json.dumps(cell), encoding="utf-8")
                return cell

            argv = [
                "run.py",
                "--bf16-reference-weights", str(source),
                "--bf16-reference-ppl-bin", str(scorer),
                "--g16-ppl-bin", str(scorer),
                "--g32-ppl-bin", str(scorer),
                "--g16-weights", str(candidate),
                "--g32-weights", str(candidate),
                "--ids", str(corpus),
                "--gate", "r9700-g16=0.02",
                "--gate", "r9700-g32=0.02",
                "--quality-tier", "accuracy",
                "--schedule-parity-max-abs-nll", "0.0",
                "--no-position-extras",
                "--out", str(output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run, "preflight_python_scorer"
            ), mock.patch.object(
                run, "run_cell", side_effect=fake_run_cell
            ):
                self.assertEqual(run.main(), 0)

            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertTrue(result["pass"])
            self.assertEqual(result["schema_version"], run.CAMPAIGN_SCHEMA_VERSION)
            self.assertEqual(len(result["reference_source"]["shards_sha256"]), 18)
            self.assertEqual(result["quality_tier"], "accuracy")
            self.assertEqual(result["w8_activation_bits"], 8)
            self.assertEqual(result["xattention_profile"], "dense")
            self.assertEqual(
                result["candidate_kv_plane_layouts"], run.R9700_KV_PLANE_LAYOUTS
            )
            self.assertEqual(result["quality_gate_contract"]["tier"], "accuracy")
            self.assertEqual(
                result["quality_gate_contract"]["prefill_decode_schedule"],
                {
                    "complete_finite_aligned_sidecars": True,
                    "maximum_absolute_nll_delta": 0.0,
                    "argmax_identity": "diagnostic_only",
                },
            )
            self.assertEqual(
                result["quality_gate_contract"]["same_route_execution_variants"][
                    "argmax_identity"
                ],
                "required",
            )
            self.assertEqual(result["lengths"], [run.DEFAULT_TOKENS, run.LONG_TOKENS])
            self.assertFalse(result["position_extras_enabled"])
            self.assertFalse(any(tokens == run.SHORT_TOKENS for _profile, tokens, _extra in calls))
            self.assertEqual(len(result["parity"]), 6)
            self.assertTrue(
                all(
                    row["comparison_kind"] == "prefill-decode-schedule"
                    and row["complete_finite_aligned"]
                    and not row["argmax_identity_is_gate"]
                    for row in result["parity"]
                )
            )
            for profile in ("r9700-g16", "r9700-g32"):
                profile_cells = [cell for cell in result["cells"] if cell["scheme"] == profile]
                self.assertTrue(
                    all(
                        len(cell["worst_delta_tokens"]) == 10
                        and cell["worst_delta_tokens"][0]
                        == {"position": 0, "delta_nll": 0.0, "abs_delta_nll": 0.0}
                        for cell in profile_cells
                    )
                )
                for tokens in (run.DEFAULT_TOKENS, run.LONG_TOKENS):
                    length_cells = [
                        cell for cell in profile_cells if cell["prompt_tokens"] == tokens
                    ]
                    self.assertTrue(
                        any("device_graph_parity" in cell for cell in length_cells)
                    )
                    self.assertTrue(any("spec_parity" in cell for cell in length_cells))
                    self.assertTrue(
                        all(
                            parity["argmax_identity_is_gate"]
                            for cell in length_cells
                            for key, parity in cell.items()
                            if key in ("device_graph_parity", "spec_parity", "draft_window_parity")
                        )
                    )
                    self.assertTrue(all(cell["quality_tier"] == "accuracy" for cell in length_cells))
            self.assertFalse(
                any(
                    profile == run.BASELINE and "--no-device-graph" in extra
                    for profile, _tokens, extra in calls
                )
            )


if __name__ == "__main__":
    unittest.main()
