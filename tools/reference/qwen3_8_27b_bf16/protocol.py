"""Dependency-free CLI, checkpoint, and result contracts for the BF16 scorer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
from typing import Iterable, Sequence


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "bf16-source"
BF16_SCHEME = "bf16-reference"
G16_CACHE_SCHEME = "bf16-source-kv-g16"
G32_CACHE_SCHEME = "bf16-source-kv-g32"
CACHE_DIAGNOSTIC_GROUPS = {
    G16_CACHE_SCHEME: 16,
    G32_CACHE_SCHEME: 32,
}
TOKEN_DOMAIN = 248077
VOCAB_SIZE = 248320
HIDDEN_SIZE = 5120
LAYERS = 64
FULL_ATTENTION_LAYERS = tuple(range(3, LAYERS, 4))
TERRIBLE_NLL = 10.0
SHARD_COUNT = 18
SOURCE_TENSOR_COUNT = 1199
SOURCE_TOTAL_BYTES = 55_562_855_904
GDN_QKV_RANGES = ((0, 2048), (2048, 4096), (4096, 10240))
DETERMINISTIC_EXECUTION_PROFILE = (
    "rocm-hipblas-no-atomics-strict-deterministic-pv-gdn-v3"
)
ATTENTION_PV_SOURCE_ROW_CHUNK = 8192
ATTENTION_PV_EXECUTION = {
    "implementation": "torch-mm-source-row-chunked-v1",
    "source_row_chunk": ATTENTION_PV_SOURCE_ROW_CHUNK,
    "source_row_order": "ascending-absolute",
    "final_partial_chunk": True,
    "operand_dtype": "float32",
    "accumulation_dtype": "float32",
    "accumulation_order": "ordered-in-place-add",
}
GDN_RECURRENCE_EXECUTION = {
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
}
DETERMINISTIC_ENVIRONMENT = {
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
}
FORBIDDEN_EXECUTION_ENVIRONMENT = (
    "TRITON_OVERRIDE_ARCH",
    "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH",
    "ROCBLAS_TENSILE_LIBPATH",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
)
EXECUTION_ENVIRONMENT_KEYS = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
    *DETERMINISTIC_ENVIRONMENT,
    *FORBIDDEN_EXECUTION_ENVIRONMENT,
    "FLA_CACHE_MODE", "FLA_CACHE_RESULTS", "FLA_CONFIG_DIR",
    "FLA_DISABLE_BACKEND_DISPATCH", "FLA_DISABLE_TENSOR_CACHE",
    "TRITON_CACHE_DIR",
)
TRITON_CODEGEN_EXECUTION = {
    "target_backend": "hip",
    "target_arch": "gfx1201",
    "default_fp_fusion": True,
    "interpret": False,
    "kernel_override": False,
    "override_arch": None,
    "amd_use_buffer_ops": False,
    "amd_use_buffer_atomics": True,
    "amd_global_prefetch": 0,
    "amd_local_prefetch": 0,
    "amd_use_async_copy": False,
    "amd_use_block_pingpong": False,
    "amd_use_in_thread_transpose": False,
    "amd_scalarize_packed_fops": False,
}
MATMUL_REDUCTION_EXECUTION = {
    "allow_tf32": False,
    "allow_fp16_reduced_precision_reduction": True,
    "allow_bf16_reduced_precision_reduction": True,
    "allow_fp16_accumulation": False,
}


@dataclass(frozen=True, slots=True)
class TensorRequirement:
    name: str
    shape: tuple[int, ...]
    dtype: str = "BF16"


@dataclass(frozen=True, slots=True)
class ScorerOptions:
    weights: Path
    ids: Path
    scheme: str
    schedule: str
    skip: str
    tokens: int
    prefill_chunk: int
    device: int
    out_json: str
    spec: str
    draft_tokens: int
    device_graph: bool
    trace_json: Path | None = None


@dataclass(frozen=True, slots=True)
class ScoreVectors:
    prompt_tokens: int
    skip_tokens: int
    nlls: tuple[float, ...]
    argmax: tuple[int, ...]
    score_seconds: float


@dataclass(frozen=True, slots=True)
class ScoreProvenance:
    config_sha256: str
    index_sha256: str
    shard_sha256: tuple[tuple[str, str], ...]
    corpus_ids_sha256: str
    source_tensor_count: int
    source_text_tensor_count: int
    source_shard_count: int
    execution: dict


def establish_deterministic_environment() -> dict[str, str]:
    """Fix the process-wide BLAS profile before importing any accelerator package."""
    forbidden = [key for key in FORBIDDEN_EXECUTION_ENVIRONMENT if key in os.environ]
    if forbidden:
        raise ValueError(
            "deterministic BF16 execution forbids environment override(s): "
            + ", ".join(forbidden)
        )
    for key, required in DETERMINISTIC_ENVIRONMENT.items():
        current = os.environ.get(key)
        if current is not None and current != required:
            raise ValueError(
                f"{key}={current!r} conflicts with required deterministic value {required!r}"
            )
    for key, required in DETERMINISTIC_ENVIRONMENT.items():
        os.environ[key] = required
    return dict(DETERMINISTIC_ENVIRONMENT)


def resolved_preferred_blas_library(torch) -> str:
    """Return PyTorch's CUDA-named BLAS enum using its actual ROCm meaning."""
    value = torch.backends.cuda.preferred_blas_library()
    name = getattr(value, "name", None)
    mapping = {
        "Default": "default",
        "Cublas": "hipblas" if torch.version.hip else "cublas",
        "Cublaslt": "hipblaslt" if torch.version.hip else "cublaslt",
        "Ck": "ck",
    }
    try:
        return mapping[name]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"unknown PyTorch preferred BLAS backend {value!r}") from exc


def enable_strict_torch_determinism(torch) -> None:
    """Enable and verify the one supported deterministic PyTorch execution route."""
    torch.use_deterministic_algorithms(True, warn_only=False)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch strict deterministic algorithms did not enable")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("PyTorch deterministic algorithms remained warn-only")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("deterministic BF16 scorer requires highest FP32 matmul precision")
    if torch.cuda.tunable.is_enabled():
        raise RuntimeError("deterministic BF16 scorer requires TunableOp disabled")
    if resolved_matmul_reduction(torch) != MATMUL_REDUCTION_EXECUTION:
        raise RuntimeError("deterministic BF16 scorer has noncanonical matmul reductions")
    preferred = resolved_preferred_blas_library(torch)
    if preferred != "hipblas":
        raise RuntimeError(
            f"deterministic BF16 scorer resolved BLAS backend {preferred!r}; expected 'hipblas'"
        )


def resolved_matmul_reduction(torch) -> dict:
    matmul = torch.backends.cuda.matmul
    return {key: getattr(matmul, key, None) for key in MATMUL_REDUCTION_EXECUTION}


def resolved_triton_codegen(triton) -> dict:
    """Return and validate the effective gfx1201 Triton lowering identity."""
    knobs = triton.knobs
    target = triton.runtime.driver.active.get_current_target()
    arch = str(target.arch).split(":", 1)[0]
    raw_block_pingpong = knobs.amd.use_block_pingpong
    raw_in_thread_transpose = knobs.amd.use_in_thread_transpose
    result = {
        "target_backend": target.backend,
        "target_arch": arch,
        "default_fp_fusion": knobs.language.default_fp_fusion,
        "interpret": knobs.runtime.interpret,
        "kernel_override": knobs.compilation.override,
        "override_arch": knobs.runtime.override_arch,
        "amd_use_buffer_ops": knobs.amd.use_buffer_ops,
        "amd_use_buffer_atomics": knobs.amd.use_buffer_atomics,
        "amd_global_prefetch": knobs.amd.global_prefetch,
        "amd_local_prefetch": knobs.amd.local_prefetch,
        "amd_use_async_copy": knobs.amd.use_async_copy,
        "amd_use_block_pingpong": (
            arch in {"gfx942", "gfx950"}
            if raw_block_pingpong is None else raw_block_pingpong
        ),
        "amd_use_in_thread_transpose": (
            arch == "gfx942"
            if raw_in_thread_transpose is None else raw_in_thread_transpose
        ),
        "amd_scalarize_packed_fops": knobs.amd.scalarize_packed_fops,
    }
    if result != TRITON_CODEGEN_EXECUTION:
        raise RuntimeError(
            f"deterministic BF16 Triton codegen {result!r}; "
            f"expected {TRITON_CODEGEN_EXECUTION!r}"
        )
    return result


def execution_provenance(torch, device_index: int, *, stage_trace_enabled: bool) -> dict:
    """Describe the full stable implementation/runtime identity of a scorer result."""
    from .gdn_determinism_probe import distribution_record, file_sha256, python_tree_sha256

    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("refusing to publish without strict deterministic algorithms")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("refusing to publish with warn-only deterministic algorithms")
    preferred = resolved_preferred_blas_library(torch)
    if preferred != "hipblas":
        raise RuntimeError("refusing to publish a non-hipBLAS BF16 reference result")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("refusing to publish non-highest FP32 matmul precision")
    tunable_enabled = torch.cuda.tunable.is_enabled()
    if tunable_enabled:
        raise RuntimeError("refusing to publish with TunableOp enabled")
    if any(os.environ.get(key) != value for key, value in DETERMINISTIC_ENVIRONMENT.items()):
        raise RuntimeError("deterministic BF16 environment changed after PyTorch import")
    if any(key in os.environ for key in FORBIDDEN_EXECUTION_ENVIRONMENT):
        raise RuntimeError("forbidden BF16 environment override appeared after import")

    root = Path(__file__).resolve().parent
    executable = Path(sys.executable).resolve()
    properties = torch.cuda.get_device_properties(device_index)
    device_arch = getattr(properties, "gcnArchName", None)
    if not isinstance(device_arch, str) or device_arch.split(":", 1)[0] != "gfx1201":
        raise RuntimeError(f"deterministic BF16 scorer requires gfx1201, got {device_arch!r}")
    fla = importlib.import_module("fla")
    triton = importlib.import_module("triton")
    fla_root = Path(fla.__file__).resolve().parent
    reductions = resolved_matmul_reduction(torch)
    if reductions != MATMUL_REDUCTION_EXECUTION:
        raise RuntimeError("refusing to publish noncanonical matmul reductions")
    return {
        "profile": DETERMINISTIC_EXECUTION_PROFILE,
        "python": sys.version,
        "python_executable": str(executable),
        "python_executable_sha256": file_sha256(executable),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "hip": torch.version.hip,
        "device_index": device_index,
        "device_name": properties.name,
        "device_arch": device_arch,
        "stage_trace_enabled": stage_trace_enabled,
        "attention_pv": dict(ATTENTION_PV_EXECUTION),
        "gdn_recurrence": dict(GDN_RECURRENCE_EXECUTION),
        "tunable_op": {"enabled": tunable_enabled},
        "triton_codegen": resolved_triton_codegen(triton),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "preferred_blas_library": preferred,
        "deterministic_algorithms": {"enabled": True, "warn_only": False},
        "matmul_reduction": reductions,
        "environment": {key: os.environ.get(key) for key in EXECUTION_ENVIRONMENT_KEYS},
        "scorer_python_tree_sha256": python_tree_sha256(root),
        "fla_python_tree_sha256": python_tree_sha256(fla_root),
        "distributions": {
            key: distribution_record(key)
            for key in ("torch", "triton", "flash-linear-attention", "safetensors")
        },
    }


def expected_text_tensors() -> tuple[TensorRequirement, ...]:
    """The complete source tensor domain consumed by the text target formula."""
    requirements = [
        TensorRequirement("model.language_model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN_SIZE))
    ]
    for layer in range(LAYERS):
        prefix = f"model.language_model.layers.{layer}."
        requirements.append(
            TensorRequirement(prefix + "input_layernorm.weight", (HIDDEN_SIZE,))
        )
        if layer in FULL_ATTENTION_LAYERS:
            requirements.extend(
                (
                    TensorRequirement(prefix + "self_attn.q_proj.weight", (12288, HIDDEN_SIZE)),
                    TensorRequirement(prefix + "self_attn.k_proj.weight", (1024, HIDDEN_SIZE)),
                    TensorRequirement(prefix + "self_attn.v_proj.weight", (1024, HIDDEN_SIZE)),
                    TensorRequirement(prefix + "self_attn.q_norm.weight", (256,)),
                    TensorRequirement(prefix + "self_attn.k_norm.weight", (256,)),
                    TensorRequirement(prefix + "self_attn.o_proj.weight", (HIDDEN_SIZE, 6144)),
                )
            )
        else:
            requirements.extend(
                (
                    TensorRequirement(prefix + "linear_attn.A_log", (48,)),
                    TensorRequirement(prefix + "linear_attn.dt_bias", (48,)),
                    TensorRequirement(prefix + "linear_attn.conv1d.weight", (10240, 1, 4)),
                    TensorRequirement(prefix + "linear_attn.in_proj_a.weight", (48, HIDDEN_SIZE)),
                    TensorRequirement(prefix + "linear_attn.in_proj_b.weight", (48, HIDDEN_SIZE)),
                    TensorRequirement(
                        prefix + "linear_attn.in_proj_qkv.weight", (10240, HIDDEN_SIZE)
                    ),
                    TensorRequirement(prefix + "linear_attn.in_proj_z.weight", (6144, HIDDEN_SIZE)),
                    TensorRequirement(prefix + "linear_attn.norm.weight", (128,)),
                    TensorRequirement(prefix + "linear_attn.out_proj.weight", (HIDDEN_SIZE, 6144)),
                )
            )
        requirements.extend(
            (
                TensorRequirement(prefix + "post_attention_layernorm.weight", (HIDDEN_SIZE,)),
                TensorRequirement(prefix + "mlp.gate_proj.weight", (17408, HIDDEN_SIZE)),
                TensorRequirement(prefix + "mlp.up_proj.weight", (17408, HIDDEN_SIZE)),
                TensorRequirement(prefix + "mlp.down_proj.weight", (HIDDEN_SIZE, 17408)),
            )
        )
    requirements.extend(
        (
            TensorRequirement("model.language_model.norm.weight", (HIDDEN_SIZE,)),
            TensorRequirement("lm_head.weight", (VOCAB_SIZE, HIDDEN_SIZE)),
        )
    )
    result = tuple(requirements)
    if len(result) != 851 or len({item.name for item in result}) != len(result):
        raise RuntimeError("internal Qwen3.8 BF16 text inventory is incomplete")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score Qwen3.8-27B directly from its complete BF16 safetensors checkpoint."
    )
    parser.add_argument("--weights", type=Path, required=True, help="complete BF16 source directory")
    parser.add_argument("--ids", type=Path, required=True, help="whitespace-separated token ids")
    parser.add_argument(
        "--scheme",
        default=BF16_SCHEME,
        help=(
            f"report/formula profile: {BF16_SCHEME}, {G16_CACHE_SCHEME}, or "
            f"{G32_CACHE_SCHEME}"
        ),
    )
    parser.add_argument("--schedule", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--skip", default="half", help="half or a nonnegative token count")
    parser.add_argument("--tokens", type=int, default=0, help="score the first N ids; zero means all")
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-json", default="-", help="JSON path or - for stdout")
    parser.add_argument("--trace-json", type=Path, help="separate diagnostic stage-hash report")
    parser.add_argument(
        "--spec",
        choices=("none", "mtp"),
        default="none",
        help="comparison label; target logits remain the independent BF16 formula",
    )
    parser.add_argument("--draft-tokens", type=int, default=0)
    graph = parser.add_mutually_exclusive_group()
    graph.add_argument("--device-graph", dest="device_graph", action="store_true")
    graph.add_argument("--no-device-graph", dest="device_graph", action="store_false")
    parser.set_defaults(device_graph=False)
    return parser


def parse_options(argv: Sequence[str] | None = None) -> ScorerOptions:
    args = build_parser().parse_args(argv)
    if args.tokens < 0:
        raise ValueError("--tokens must be nonnegative")
    if args.prefill_chunk <= 0:
        raise ValueError("--prefill-chunk must be positive")
    if args.device < 0:
        raise ValueError("--device must be nonnegative")
    if args.scheme != BF16_SCHEME and args.scheme not in CACHE_DIAGNOSTIC_GROUPS:
        raise ValueError(
            f"--scheme must be {BF16_SCHEME}, {G16_CACHE_SCHEME}, or {G32_CACHE_SCHEME}"
        )
    parse_skip_text(args.skip)
    if args.spec == "mtp":
        if not 1 <= args.draft_tokens <= 5:
            raise ValueError("--spec mtp requires --draft-tokens in [1,5]")
    elif args.draft_tokens != 0:
        raise ValueError("--draft-tokens requires --spec mtp")
    if args.device_graph:
        raise ValueError(
            "the independent BF16 scorer has no Device Graph; use --no-device-graph"
        )
    if args.trace_json is not None and args.out_json != "-":
        output = Path(args.out_json).resolve()
        result_paths = {
            output,
            output.with_suffix(".nllf32"),
            output.with_suffix(".argmaxi32"),
        }
        if args.trace_json.resolve() in result_paths:
            raise ValueError("--trace-json must be separate from the result and sidecar paths")
    return ScorerOptions(
        weights=args.weights,
        ids=args.ids,
        scheme=args.scheme,
        schedule=args.schedule,
        skip=args.skip,
        tokens=args.tokens,
        prefill_chunk=args.prefill_chunk,
        device=args.device,
        out_json=args.out_json,
        spec=args.spec,
        draft_tokens=args.draft_tokens,
        device_graph=False,
        trace_json=args.trace_json,
    )


def cache_value_group(scheme: str) -> int | None:
    if scheme == BF16_SCHEME:
        return None
    try:
        return CACHE_DIAGNOSTIC_GROUPS[scheme]
    except KeyError as exc:
        raise ValueError(f"unknown BF16-source cache profile {scheme!r}") from exc


def formula_signature(options: ScorerOptions) -> tuple[Path, Path, int, int, int, int | None]:
    """Execution inputs affecting logits; score-index and product labels are absent."""
    return (
        options.weights,
        options.ids,
        options.tokens,
        options.prefill_chunk,
        options.device,
        cache_value_group(options.scheme),
    )


def q_projection_row(head: int, feature: int, *, gate: bool) -> int:
    if not 0 <= head < 24 or not 0 <= feature < 256:
        raise ValueError("query projection coordinate is outside Qwen3.8")
    return head * 512 + (256 if gate else 0) + feature


def parse_skip_text(text: str) -> int | None:
    if text == "half":
        return None
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise ValueError("--skip must be half or a nonnegative integer") from exc
    if value < 0:
        raise ValueError("--skip must be half or a nonnegative integer")
    return value


def resolve_skip(prompt_tokens: int, text: str) -> int:
    if prompt_tokens < 2:
        raise ValueError("score requires at least two prompt tokens")
    maximum = prompt_tokens - 2
    explicit = parse_skip_text(text)
    skip = min(prompt_tokens // 2, maximum) if explicit is None else explicit
    if skip > maximum:
        raise ValueError("score skip leaves no teacher-forced target")
    return skip


def resolve_decode_prefix(prompt_tokens: int, skip: int) -> int:
    if prompt_tokens < 3:
        raise ValueError("decode score requires at least three prompt tokens")
    prefix = max(skip, 1)
    if prefix + 1 >= prompt_tokens:
        raise ValueError("decode score skip leaves no T=1 target")
    return prefix


def resolve_score_begin(prompt_tokens: int, schedule: str, skip_text: str) -> int:
    skip = resolve_skip(prompt_tokens, skip_text)
    if schedule == "prefill":
        return skip
    if schedule == "decode":
        return resolve_decode_prefix(prompt_tokens, skip)
    raise ValueError("unknown BF16 score schedule")


def layer_spans(
    schedule: str, length: int, prefix: int, prefill_chunk: int
) -> Iterable[tuple[int, int]]:
    """Efficient oracle spans; the schedule label never changes the BF16 formula."""
    if schedule not in ("prefill", "decode"):
        raise ValueError("unknown BF16 score schedule")
    if length < 0 or prefill_chunk <= 0:
        raise ValueError("invalid BF16 layer span geometry")
    if schedule == "decode" and not 0 <= prefix <= length:
        raise ValueError("decode prefix is outside the represented hidden sequence")
    for begin in range(0, length, prefill_chunk):
        yield begin, min(length, begin + prefill_chunk)


def read_ids(path: Path, limit: int) -> list[int]:
    try:
        values = [int(item, 10) for item in path.read_text(encoding="utf-8").split()]
    except OSError as exc:
        raise ValueError(f"failed to read token ids: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"token id file is not an integer sequence: {path}") from exc
    if limit:
        values = values[:limit]
    if len(values) < 2:
        raise ValueError("score requires at least two token ids")
    invalid = next((token for token in values if not 0 <= token < VOCAB_SIZE), None)
    if invalid is not None:
        raise ValueError(f"token id {invalid} is outside the Qwen3.8 checkpoint vocabulary")
    return values


def validate_scored_targets(ids: Sequence[int], score_begin: int) -> None:
    invalid = next((token for token in ids[score_begin + 1 :] if token >= TOKEN_DOMAIN), None)
    if invalid is not None:
        raise ValueError(
            f"scored target token {invalid} is outside the Qwen3.8 public token domain"
        )


def _require_members(mapping: dict, expected: dict, label: str) -> None:
    for name, value in expected.items():
        if mapping.get(name) != value:
            raise ValueError(f"{label}.{name} is {mapping.get(name)!r}; expected {value!r}")


def validate_checkpoint_files(source: Path) -> dict[str, str]:
    """Validate exact identity and complete shard presence without model dependencies."""
    if not source.is_dir():
        raise ValueError(f"BF16 source directory not found: {source}")
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"BF16 source is missing {exc.filename}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed BF16 source JSON: {exc}") from exc

    _require_members(
        config,
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "language_model_only": False,
            "tie_word_embeddings": False,
        },
        "config",
    )
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("config.text_config is missing")
    _require_members(
        text,
        {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "output_gate_type": "swish",
            "hidden_act": "silu",
            "num_hidden_layers": LAYERS,
            "full_attention_interval": 4,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 17408,
            "vocab_size": VOCAB_SIZE,
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
        },
        "config.text_config",
    )
    expected_types = [
        "full_attention" if layer in FULL_ATTENTION_LAYERS else "linear_attention"
        for layer in range(LAYERS)
    ]
    if text.get("layer_types") != expected_types:
        raise ValueError("config.text_config.layer_types does not match Qwen3.8")
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict):
        raise ValueError("config.text_config.rope_parameters is missing")
    _require_members(
        rope,
        {
            "rope_theta": 10000000,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
        "config.text_config.rope_parameters",
    )

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()
    ):
        raise ValueError("model.safetensors.index.json has no string weight_map")
    if len(weight_map) != SOURCE_TENSOR_COUNT:
        raise ValueError(
            f"BF16 source index has {len(weight_map)} tensors; expected {SOURCE_TENSOR_COUNT}"
        )
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("total_size") != SOURCE_TOTAL_BYTES:
        actual = metadata.get("total_size") if isinstance(metadata, dict) else None
        raise ValueError(
            f"BF16 source index total_size is {actual!r}; expected {SOURCE_TOTAL_BYTES}"
        )
    missing_names = [item.name for item in expected_text_tensors() if item.name not in weight_map]
    if missing_names:
        raise ValueError(f"BF16 source index is missing text tensor {missing_names[0]}")
    expected_shards = {
        f"model-{part:05d}-of-{SHARD_COUNT:05d}.safetensors"
        for part in range(1, SHARD_COUNT + 1)
    }
    indexed_shards = set(weight_map.values())
    if indexed_shards != expected_shards:
        missing = sorted(expected_shards - indexed_shards)
        extra = sorted(indexed_shards - expected_shards)
        raise ValueError(f"BF16 source shard index differs; missing={missing}, extra={extra}")
    for shard in sorted(indexed_shards):
        path = source / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"BF16 source shard is missing or empty: {path}")
    return dict(weight_map)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_shard_sha256(source: Path, weight_map: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Hash the exact indexed checkpoint payload files in stable shard-name order."""
    return tuple(
        (name, file_sha256(source / name))
        for name in sorted(set(weight_map.values()))
    )


def ids_sha256(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in ids:
        digest.update(struct.pack("<i", token))
    return digest.hexdigest()


def result_payload(
    options: ScorerOptions, vectors: ScoreVectors, provenance: ScoreProvenance
) -> dict:
    if len(vectors.nlls) != len(vectors.argmax):
        raise ValueError("BF16 scorer vectors lost index alignment")
    non_finite = sum(not math.isfinite(value) for value in vectors.nlls)
    if non_finite:
        raise ValueError("BF16 scorer produced a non-finite token NLL")
    finite = [value for value in vectors.nlls if math.isfinite(value)]
    total = math.fsum(vectors.nlls) if non_finite == 0 else float("nan")
    mean = total / len(vectors.nlls) if vectors.nlls and non_finite == 0 else float("nan")
    try:
        perplexity = math.exp(mean)
    except OverflowError:
        perplexity = float("inf")
    value_group = cache_value_group(options.scheme)
    cache_diagnostic = value_group is not None
    return {
        "scheme": options.scheme,
        "weights": str(options.weights),
        "model_id": MODEL_ID,
        "weights_id": WEIGHTS_ID,
        "weight_format": "source-bf16",
        "formula_profile": "checkpoint-direct-qwen3.8-source-bf16",
        "source_config_sha256": provenance.config_sha256,
        "source_index_sha256": provenance.index_sha256,
        "source_shards_sha256": dict(provenance.shard_sha256),
        "corpus_ids_sha256": provenance.corpus_ids_sha256,
        "source_tensor_count": provenance.source_tensor_count,
        "source_text_tensor_count": provenance.source_text_tensor_count,
        "source_shard_count": provenance.source_shard_count,
        "execution_provenance": provenance.execution,
        "kv_format": "fp8-k-int4-v" if cache_diagnostic else "bf16-reference",
        "kv_value_group": value_group,
        "cache_only_diagnostic": cache_diagnostic,
        "cache_diagnostic_scope": "full-attention-kv-only" if cache_diagnostic else None,
        "cache_key_codec": "ocp-e4m3fn-rne-satfinite" if cache_diagnostic else None,
        "cache_value_codec": (
            "signed-int4-rne-symmetric-7-fp16-group-scale"
            if cache_diagnostic
            else None
        ),
        "cache_append_boundary": (
            "represented BF16 post-RoPE K and represented BF16 projected V"
            if cache_diagnostic
            else None
        ),
        "cache_use_boundary": (
            "decode stored K/V to FP32 attention operands"
            if cache_diagnostic
            else None
        ),
        "schedule": options.schedule,
        "spec": options.spec,
        "draft_tokens": options.draft_tokens,
        "speculative_execution": False,
        "device_graph": False,
        "prefill_chunk": options.prefill_chunk,
        "skip_tokens": vectors.skip_tokens,
        "prompt_tokens": vectors.prompt_tokens,
        "tokens_scored": len(vectors.nlls),
        "argmax_tokens": len(vectors.argmax),
        "non_finite": non_finite,
        "terrible_tokens": sum(value >= TERRIBLE_NLL for value in finite),
        "terrible_nll": TERRIBLE_NLL,
        "sum_nll": total,
        "mean_nll": mean,
        "max_nll": max(vectors.nlls, default=0.0),
        "ppl": perplexity,
        "score_seconds": vectors.score_seconds,
    }


def _sidecar_path(json_path: Path, suffix: str) -> Path:
    return json_path.with_suffix(suffix)


def clear_result(options: ScorerOptions) -> None:
    targets = []
    if options.out_json != "-":
        path = Path(options.out_json)
        targets.extend(
            (path, _sidecar_path(path, ".nllf32"), _sidecar_path(path, ".argmaxi32"))
        )
    if options.trace_json is not None:
        targets.append(options.trace_json)
    for target in targets:
        target.unlink(missing_ok=True)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_result(
    options: ScorerOptions, vectors: ScoreVectors, provenance: ScoreProvenance
) -> dict:
    payload = result_payload(options, vectors, provenance)
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if options.out_json == "-":
        print(encoded.decode("utf-8"), end="")
        return payload
    path = Path(options.out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    nll_payload = struct.pack("<" + "f" * len(vectors.nlls), *vectors.nlls)
    argmax_payload = struct.pack("<" + "i" * len(vectors.argmax), *vectors.argmax)
    # Publish JSON last: the runner never observes a fresh cell descriptor with stale sidecars.
    _atomic_write(_sidecar_path(path, ".nllf32"), nll_payload)
    _atomic_write(_sidecar_path(path, ".argmaxi32"), argmax_payload)
    _atomic_write(path, encoded)
    return payload


__all__ = [
    "ATTENTION_PV_EXECUTION",
    "ATTENTION_PV_SOURCE_ROW_CHUNK",
    "BF16_SCHEME",
    "CACHE_DIAGNOSTIC_GROUPS",
    "DETERMINISTIC_ENVIRONMENT",
    "DETERMINISTIC_EXECUTION_PROFILE",
    "EXECUTION_ENVIRONMENT_KEYS",
    "FORBIDDEN_EXECUTION_ENVIRONMENT",
    "FULL_ATTENTION_LAYERS",
    "GDN_QKV_RANGES",
    "GDN_RECURRENCE_EXECUTION",
    "G16_CACHE_SCHEME",
    "G32_CACHE_SCHEME",
    "MATMUL_REDUCTION_EXECUTION",
    "MODEL_ID",
    "ScoreProvenance",
    "ScoreVectors",
    "ScorerOptions",
    "TOKEN_DOMAIN",
    "TRITON_CODEGEN_EXECUTION",
    "TensorRequirement",
    "WEIGHTS_ID",
    "build_parser",
    "cache_value_group",
    "clear_result",
    "enable_strict_torch_determinism",
    "establish_deterministic_environment",
    "execution_provenance",
    "expected_text_tensors",
    "file_sha256",
    "formula_signature",
    "ids_sha256",
    "layer_spans",
    "parse_options",
    "q_projection_row",
    "read_ids",
    "resolve_decode_prefix",
    "resolve_score_begin",
    "resolve_skip",
    "resolved_preferred_blas_library",
    "resolved_matmul_reduction",
    "resolved_triton_codegen",
    "result_payload",
    "source_shard_sha256",
    "validate_checkpoint_files",
    "validate_scored_targets",
    "write_result",
]
