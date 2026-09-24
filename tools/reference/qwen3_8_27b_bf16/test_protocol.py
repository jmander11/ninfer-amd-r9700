"""Dependency-light tests for the BF16 source scorer protocol and source mapping."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from . import protocol
from . import scorer


def options(
    root: Path,
    output: str = "-",
    scheme: str = protocol.BF16_SCHEME,
    trace: Path | None = None,
) -> protocol.ScorerOptions:
    return protocol.ScorerOptions(
        weights=root,
        ids=root / "corpus.ids",
        scheme=scheme,
        schedule="prefill",
        skip="half",
        tokens=0,
        prefill_chunk=4096,
        device=0,
        out_json=output,
        spec="none",
        draft_tokens=0,
        device_graph=False,
        trace_json=trace,
    )


def provenance() -> protocol.ScoreProvenance:
    return protocol.ScoreProvenance(
        "a" * 64,
        "b" * 64,
        (("model-00001-of-00018.safetensors", "d" * 64),),
        "c" * 64,
        1199,
        851,
        18,
        {
            "profile": protocol.DETERMINISTIC_EXECUTION_PROFILE,
            "preferred_blas_library": "hipblas",
            "deterministic_algorithms": {"enabled": True, "warn_only": False},
            "attention_pv": dict(protocol.ATTENTION_PV_EXECUTION),
            "gdn_recurrence": dict(protocol.GDN_RECURRENCE_EXECUTION),
            "environment": dict(protocol.DETERMINISTIC_ENVIRONMENT),
        },
    )


class _NamedBackend:
    name = "Cublas"


class _FakeTorch:
    class version:
        hip = "7.0"

    class backends:
        class cuda:
            class matmul:
                allow_tf32 = False
                allow_fp16_reduced_precision_reduction = True
                allow_bf16_reduced_precision_reduction = True
                allow_fp16_accumulation = False

            @staticmethod
            def preferred_blas_library():
                return _NamedBackend()

    class cuda:
        class tunable:
            @staticmethod
            def is_enabled():
                return False

    def __init__(self) -> None:
        self.enabled = False
        self.warn_only = True

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self.enabled = enabled
        self.warn_only = warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only

    @staticmethod
    def get_float32_matmul_precision() -> str:
        return "highest"


class DeterministicExecutionContractTest(unittest.TestCase):
    def test_v3_binds_exact_gdn_dispatch_and_formats(self) -> None:
        self.assertTrue(protocol.DETERMINISTIC_EXECUTION_PROFILE.endswith("pv-gdn-v3"))
        self.assertEqual(
            protocol.GDN_RECURRENCE_EXECUTION,
            {
                "dispatch": "T=1-project-explicit;T>1-fla-fused-recurrent",
                "single_token_implementation": "project-explicit-fp32-recurrence-v1",
                "multi_token_implementation": (
                    "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule"
                ),
                "qkv_input_dtype": "bfloat16",
                "gate_state_input_dtype": "float32",
                "output_dtype": "bfloat16",
                "final_state_dtype": "float32",
                "output_final_state": True,
                "scale": 128 ** -0.5,
            },
        )

    def test_absent_environment_is_fixed_and_exact_values_are_accepted(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                protocol.establish_deterministic_environment(),
                protocol.DETERMINISTIC_ENVIRONMENT,
            )
            self.assertEqual(
                {key: os.environ[key] for key in protocol.DETERMINISTIC_ENVIRONMENT},
                protocol.DETERMINISTIC_ENVIRONMENT,
            )
            protocol.establish_deterministic_environment()

    def test_v3_binds_exact_process_environment(self) -> None:
        self.assertEqual(
            protocol.DETERMINISTIC_ENVIRONMENT,
            {
                "TORCH_BLAS_PREFER_HIPBLASLT": "0",
                "ROCBLAS_DEFAULT_ATOMICS_MODE": "0",
                "FLA_USE_FAST_OPS": "0",
                "PYTORCH_TUNABLEOP_ENABLED": "0",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                "TRITON_DEFAULT_FP_FUSION": "1",
                "TRITON_INTERPRET": "0",
                "TRITON_KERNEL_OVERRIDE": "0",
                "ROCBLAS_USE_HIPBLASLT": "0",
                "ROCBLAS_USE_HIPBLASLT_BATCHED": "0",
                "CUDA_LAUNCH_BLOCKING": "0",
                "HIP_LAUNCH_BLOCKING": "0",
                "AMD_SERIALIZE_KERNEL": "0",
                "AMD_SERIALIZE_COPY": "0",
                "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            },
        )
        self.assertEqual(
            protocol.FORBIDDEN_EXECUTION_ENVIRONMENT,
            (
                "TRITON_OVERRIDE_ARCH",
                "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH",
                "ROCBLAS_TENSILE_LIBPATH",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_HIP_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
            ),
        )

    def test_conflicting_caller_environment_fails_without_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TORCH_BLAS_PREFER_HIPBLASLT": "1", "ROCBLAS_DEFAULT_ATOMICS_MODE": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "TORCH_BLAS_PREFER_HIPBLASLT"):
                protocol.establish_deterministic_environment()
            self.assertEqual(os.environ["TORCH_BLAS_PREFER_HIPBLASLT"], "1")

    def test_forbidden_environment_override_fails_even_when_empty(self) -> None:
        with mock.patch.dict(os.environ, {"TRITON_OVERRIDE_ARCH": ""}, clear=True):
            with self.assertRaisesRegex(ValueError, "TRITON_OVERRIDE_ARCH"):
                protocol.establish_deterministic_environment()
        with mock.patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": ""}, clear=True):
            with self.assertRaisesRegex(ValueError, "PYTORCH_ALLOC_CONF"):
                protocol.establish_deterministic_environment()

    def test_resolved_triton_codegen_requires_exact_gfx1201_defaults(self) -> None:
        class Values:
            default_fp_fusion = True
            interpret = False
            override = False
            override_arch = None
            use_buffer_ops = False
            use_buffer_atomics = True
            global_prefetch = 0
            local_prefetch = 0
            use_async_copy = False
            use_block_pingpong = None
            use_in_thread_transpose = None
            scalarize_packed_fops = False

        class Driver:
            class active:
                @staticmethod
                def get_current_target():
                    return type("Target", (), {"backend": "hip", "arch": "gfx1201"})()

        triton = type("Triton", (), {
            "knobs": type("Knobs", (), {
                "language": Values(), "runtime": Values(), "compilation": Values(),
                "amd": Values(),
            })(),
            "runtime": type("Runtime", (), {"driver": Driver()})(),
        })()
        self.assertEqual(
            protocol.resolved_triton_codegen(triton),
            protocol.TRITON_CODEGEN_EXECUTION,
        )
        triton.knobs.amd.global_prefetch = 1
        with self.assertRaisesRegex(RuntimeError, "Triton codegen"):
            protocol.resolved_triton_codegen(triton)

    def test_strict_torch_mode_is_enabled_and_hipblas_verified(self) -> None:
        torch = _FakeTorch()
        protocol.enable_strict_torch_determinism(torch)
        self.assertTrue(torch.enabled)
        self.assertFalse(torch.warn_only)
        self.assertEqual(protocol.resolved_preferred_blas_library(torch), "hipblas")

    def test_scorer_rejects_conflict_before_preflight_or_result_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "retained.json"
            result.write_text("retained\n", encoding="utf-8")
            argv = [
                "--weights", str(root / "missing-source"),
                "--ids", str(root / "missing.ids"),
                "--out-json", str(result),
            ]
            with mock.patch.dict(
                os.environ,
                {"TORCH_BLAS_PREFER_HIPBLASLT": "1", "ROCBLAS_DEFAULT_ATOMICS_MODE": "0"},
                clear=True,
            ), contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(scorer.main(argv), 1)
            self.assertIn("TORCH_BLAS_PREFER_HIPBLASLT", stderr.getvalue())
            self.assertEqual(result.read_text(encoding="utf-8"), "retained\n")

    def test_fla_fast_math_conflict_fails_before_dependency_import(self) -> None:
        with mock.patch.dict(os.environ, {"FLA_USE_FAST_OPS": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "FLA_USE_FAST_OPS"):
                protocol.establish_deterministic_environment()
            self.assertEqual(os.environ["FLA_USE_FAST_OPS"], "1")


def exact_config() -> dict:
    layer_types = [
        "full_attention" if layer in protocol.FULL_ATTENTION_LAYERS else "linear_attention"
        for layer in range(64)
    ]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "language_model_only": False,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "output_gate_type": "swish",
            "hidden_act": "silu",
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": False,
            "layer_types": layer_types,
            "rope_parameters": {
                "rope_theta": 10000000,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            },
        },
    }


class ArgumentContractTest(unittest.TestCase):
    def base(self) -> list[str]:
        return ["--weights", "/checkpoint", "--ids", "/corpus.ids"]

    def test_product_labels_do_not_enter_formula_signature(self) -> None:
        ordinary = protocol.parse_options(self.base())
        mtp = protocol.parse_options(self.base() + ["--spec", "mtp", "--draft-tokens", "3"])
        eager = protocol.parse_options(self.base() + ["--no-device-graph"])
        decode = protocol.parse_options(self.base() + ["--schedule", "decode"])
        self.assertEqual(protocol.formula_signature(ordinary), protocol.formula_signature(mtp))
        self.assertEqual(protocol.formula_signature(ordinary), protocol.formula_signature(eager))
        self.assertEqual(protocol.formula_signature(ordinary), protocol.formula_signature(decode))
        prefill_zero = protocol.parse_options(self.base() + ["--skip", "0"])
        decode_zero = protocol.parse_options(
            self.base() + ["--skip", "0", "--schedule", "decode"]
        )
        self.assertEqual(
            protocol.formula_signature(prefill_zero), protocol.formula_signature(decode_zero)
        )
        self.assertEqual(protocol.resolve_score_begin(8, "prefill", "0"), 0)
        self.assertEqual(protocol.resolve_score_begin(8, "decode", "0"), 1)

    def test_cache_profile_is_an_explicit_formula_input(self) -> None:
        baseline = protocol.parse_options(self.base())
        g16 = protocol.parse_options(
            self.base() + ["--scheme", protocol.G16_CACHE_SCHEME]
        )
        g32 = protocol.parse_options(
            self.base() + ["--scheme", protocol.G32_CACHE_SCHEME]
        )
        self.assertIsNone(protocol.cache_value_group(baseline.scheme))
        self.assertEqual(protocol.cache_value_group(g16.scheme), 16)
        self.assertEqual(protocol.cache_value_group(g32.scheme), 32)
        self.assertNotEqual(protocol.formula_signature(baseline), protocol.formula_signature(g16))
        self.assertNotEqual(protocol.formula_signature(g16), protocol.formula_signature(g32))

    def test_product_flags_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --draft-tokens"):
            protocol.parse_options(self.base() + ["--spec", "mtp"])
        with self.assertRaisesRegex(ValueError, "requires --spec mtp"):
            protocol.parse_options(self.base() + ["--draft-tokens", "3"])
        with self.assertRaisesRegex(ValueError, "no Device Graph"):
            protocol.parse_options(self.base() + ["--device-graph"])
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                protocol.parse_options(self.base() + ["--spec", "dflash"])
        with self.assertRaisesRegex(ValueError, "must be bf16-reference"):
            protocol.parse_options(self.base() + ["--scheme", "r9700-g16"])

    def test_trace_cannot_collide_with_result_triplet(self) -> None:
        for trace in ("result.json", "result.nllf32", "result.argmaxi32"):
            with self.subTest(trace=trace), self.assertRaisesRegex(ValueError, "separate"):
                protocol.parse_options(
                    self.base()
                    + ["--out-json", "result.json", "--trace-json", trace]
                )

    def test_executable_help_needs_no_model_dependencies(self) -> None:
        executable = Path(__file__).with_name("ppl.py")
        completed = subprocess.run(
            [sys.executable, str(executable), "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("complete BF16 source directory", completed.stdout)


class ScoreIndexTest(unittest.TestCase):
    def test_skip_and_decode_match_product_score_index(self) -> None:
        self.assertEqual(protocol.resolve_skip(8, "half"), 4)
        self.assertEqual(protocol.resolve_skip(3, "half"), 1)
        self.assertEqual(protocol.resolve_decode_prefix(8, 0), 1)
        self.assertEqual(protocol.resolve_decode_prefix(8, 4), 4)
        with self.assertRaisesRegex(ValueError, "leaves no"):
            protocol.resolve_skip(8, "7")

    def test_layer_major_spans_do_not_change_with_schedule_label(self) -> None:
        expected = [(0, 4), (4, 8), (8, 10)]
        self.assertEqual(list(protocol.layer_spans("prefill", 10, 0, 4)), expected)
        self.assertEqual(list(protocol.layer_spans("decode", 10, 5, 4)), expected)

    def test_input_vocab_and_scored_domain_are_distinct(self) -> None:
        protocol.validate_scored_targets([248319, 1, 2], 0)
        with self.assertRaisesRegex(ValueError, "public token domain"):
            protocol.validate_scored_targets([1, 248077], 0)


class ResultLifecycleTest(unittest.TestCase):
    def test_clear_result_removes_stale_trace_even_with_stdout_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.json"
            trace.write_text("stale")
            protocol.clear_result(options(root, trace=trace))
            self.assertFalse(trace.exists())

    def test_clear_result_removes_result_sidecars_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            trace = root / "trace.json"
            paths = (output, output.with_suffix(".nllf32"),
                     output.with_suffix(".argmaxi32"), trace)
            for path in paths:
                path.write_text("stale")
            protocol.clear_result(options(root, str(output), trace=trace))
            self.assertTrue(all(not path.exists() for path in paths))


class SourceMappingTest(unittest.TestCase):
    def test_inventory_is_complete_and_output_head_is_untied(self) -> None:
        requirements = protocol.expected_text_tensors()
        names = {item.name for item in requirements}
        self.assertEqual(len(requirements), 851)
        self.assertIn("model.language_model.embed_tokens.weight", names)
        self.assertIn("lm_head.weight", names)
        self.assertNotEqual(
            "model.language_model.embed_tokens.weight", "lm_head.weight"
        )
        conv = next(
            item
            for item in requirements
            if item.name == "model.language_model.layers.0.linear_attn.conv1d.weight"
        )
        self.assertEqual(conv.shape, (10240, 1, 4))

    def test_query_gate_rows_are_head_interleaved(self) -> None:
        self.assertEqual(protocol.q_projection_row(0, 0, gate=False), 0)
        self.assertEqual(protocol.q_projection_row(0, 0, gate=True), 256)
        self.assertEqual(protocol.q_projection_row(1, 0, gate=False), 512)
        self.assertEqual(protocol.q_projection_row(23, 255, gate=True), 12287)

    def test_gdn_qkv_slices_are_exact(self) -> None:
        self.assertEqual(
            protocol.GDN_QKV_RANGES, ((0, 2048), (2048, 4096), (4096, 10240))
        )


class CheckpointPreflightTest(unittest.TestCase):
    def build_fake_source(self, root: Path, *, omit_shard: int | None = None) -> None:
        (root / "config.json").write_text(json.dumps(exact_config()), encoding="utf-8")
        shard_names = [
            f"model-{part:05d}-of-00018.safetensors" for part in range(1, 19)
        ]
        weight_map = {
            item.name: shard_names[index % len(shard_names)]
            for index, item in enumerate(protocol.expected_text_tensors())
        }
        for index in range(protocol.SOURCE_TENSOR_COUNT - len(weight_map)):
            weight_map[f"model.visual.synthetic_test_tensor.{index}"] = shard_names[
                (len(weight_map) + index) % len(shard_names)
            ]
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": protocol.SOURCE_TOTAL_BYTES},
                    "weight_map": weight_map,
                }
            ),
            encoding="utf-8",
        )
        for part, shard in enumerate(shard_names, 1):
            if part != omit_shard:
                (root / shard).write_bytes(b"metadata-sentinel")

    def test_complete_index_and_shard_set_passes_dependency_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_fake_source(root)
            weight_map = protocol.validate_checkpoint_files(root)
            self.assertEqual(len(weight_map), protocol.SOURCE_TENSOR_COUNT)
            shard_hashes = dict(protocol.source_shard_sha256(root, weight_map))
            self.assertEqual(len(shard_hashes), protocol.SHARD_COUNT)
            self.assertEqual(
                set(shard_hashes),
                {
                    f"model-{part:05d}-of-{protocol.SHARD_COUNT:05d}.safetensors"
                    for part in range(1, protocol.SHARD_COUNT + 1)
                },
            )

    def test_missing_shard_fails_before_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_fake_source(root, omit_shard=7)
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                protocol.validate_checkpoint_files(root)


class SidecarContractTest(unittest.TestCase):
    def test_aligned_sidecars_and_json_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "cell.json"
            vectors = protocol.ScoreVectors(5, 1, (1.25, 2.5, 3.75), (10, 11, 12), 4.0)
            payload = protocol.write_result(options(root, str(result)), vectors, provenance())
            self.assertEqual(payload["tokens_scored"], 3)
            self.assertEqual(payload["argmax_tokens"], 3)
            self.assertEqual(
                struct.unpack("<3f", result.with_suffix(".nllf32").read_bytes()),
                vectors.nlls,
            )
            self.assertEqual(
                struct.unpack("<3i", result.with_suffix(".argmaxi32").read_bytes()),
                vectors.argmax,
            )
            disk = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(disk["kv_format"], "bf16-reference")
            self.assertIsNone(disk["kv_value_group"])
            self.assertEqual(disk["weight_format"], "source-bf16")
            self.assertEqual(
                disk["source_shards_sha256"],
                {"model-00001-of-00018.safetensors": "d" * 64},
            )
            self.assertEqual(
                disk["formula_profile"], "checkpoint-direct-qwen3.8-source-bf16"
            )
            self.assertEqual(disk["execution_provenance"], provenance().execution)
            self.assertFalse(disk["cache_only_diagnostic"])
            self.assertIsNone(disk["cache_diagnostic_scope"])
            self.assertIsNone(disk["cache_key_codec"])
            self.assertIsNone(disk["cache_value_codec"])
            self.assertFalse(disk["speculative_execution"])
            self.assertFalse(disk["device_graph"])

    def test_cache_diagnostic_sidecar_proves_weight_codec_and_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = protocol.ScoreVectors(5, 1, (1.25, 2.5, 3.75), (10, 11, 12), 4.0)
            for scheme, group in (
                (protocol.G16_CACHE_SCHEME, 16),
                (protocol.G32_CACHE_SCHEME, 32),
            ):
                result = root / f"{scheme}.json"
                payload = protocol.write_result(
                    options(root, str(result), scheme), vectors, provenance()
                )
                self.assertEqual(payload["scheme"], scheme)
                self.assertEqual(payload["weights_id"], "bf16-source")
                self.assertEqual(payload["weight_format"], "source-bf16")
                self.assertEqual(
                    payload["formula_profile"], "checkpoint-direct-qwen3.8-source-bf16"
                )
                self.assertEqual(payload["kv_format"], "fp8-k-int4-v")
                self.assertEqual(payload["kv_value_group"], group)
                self.assertTrue(payload["cache_only_diagnostic"])
                self.assertEqual(payload["cache_diagnostic_scope"], "full-attention-kv-only")
                self.assertEqual(payload["cache_key_codec"], "ocp-e4m3fn-rne-satfinite")
                self.assertIn("signed-int4", payload["cache_value_codec"])
                self.assertIn("post-RoPE K", payload["cache_append_boundary"])
                self.assertIn("FP32 attention", payload["cache_use_boundary"])

    def test_misaligned_or_nonfinite_vectors_cannot_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "cell.json"
            with self.assertRaisesRegex(ValueError, "alignment"):
                protocol.write_result(
                    options(root, str(result)),
                    protocol.ScoreVectors(4, 0, (1.0,), (1, 2), 0.1),
                    provenance(),
                )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                protocol.write_result(
                    options(root, str(result)),
                    protocol.ScoreVectors(3, 0, (float("nan"),), (1,), 0.1),
                    provenance(),
                )
            self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()
