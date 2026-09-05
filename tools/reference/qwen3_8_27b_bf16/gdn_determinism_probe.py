#!/usr/bin/env python3
"""Standalone determinism probe for the BF16 scorer's GDN recurrence boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Sequence


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


SCHEMA = "ninfer_qwen3_8_27b_bf16_gdn_determinism_probe"
SCHEMA_VERSION = 1
ROUTES = ("chunk", "fused-recurrent", "project-naive")
GENERATOR = "integer-pattern-l2norm-bf16-v1"
ENVIRONMENT_KEYS = (
    "FLA_CACHE_MODE",
    "FLA_CACHE_RESULTS",
    "FLA_CONFIG_DIR",
    "FLA_DISABLE_BACKEND_DISPATCH",
    "FLA_DISABLE_TENSOR_CACHE",
    "FLA_FLASH_QLA",
    "FLA_GPU_NAME",
    "HIP_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "ROCR_VISIBLE_DEVICES",
    "TRITON_CACHE_DIR",
)


@dataclass(frozen=True, slots=True)
class Options:
    rows: int
    repeats: int
    device: int
    routes: tuple[str, ...]
    deterministic_algorithms: bool
    out_json: str


def _parse_routes(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--routes must be a nonempty unique comma-separated list")
    invalid = [value for value in values if value not in ROUTES]
    if invalid:
        raise argparse.ArgumentTypeError(
            "unknown recurrence route(s): " + ", ".join(invalid)
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay fixed CPU-generated GDN inputs through FLA chunk, FLA fused-recurrent, "
            "and the project FP32 recurrence."
        )
    )
    parser.add_argument(
        "--rows", type=int, default=4096,
        help="4096 reproduces the scorer's inter-chunk grid; use 64 with project-naive",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--routes", type=_parse_routes, default=("chunk", "fused-recurrent"),
        help="comma-separated routes; project-naive is bounded to at most 256 rows",
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help="enable torch deterministic-algorithm enforcement before input generation/execution",
    )
    parser.add_argument("--out-json", required=True, help="JSON path or - for stdout")
    return parser


def parse_options(argv: Sequence[str] | None = None) -> Options:
    args = build_parser().parse_args(argv)
    if not 1 <= args.rows <= 4096:
        raise ValueError("--rows must be in [1,4096]")
    if not 2 <= args.repeats <= 20:
        raise ValueError("--repeats must be in [2,20]")
    if args.device < 0:
        raise ValueError("--device must be nonnegative")
    if "project-naive" in args.routes and args.rows > 256:
        raise ValueError(
            "project-naive is bounded to 256 rows; omit it for a larger chunk-only probe"
        )
    return Options(
        args.rows, args.repeats, args.device, args.routes,
        args.deterministic_algorithms, args.out_json,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def python_tree_sha256(root: Path) -> str:
    """Bind an imported Python implementation without hashing mutable bytecode caches."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def distribution_record(name: str) -> dict:
    """Bind installed distribution metadata whose RECORD binds its installed files."""
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "record_sha256": None, "direct_url_sha256": None}
    files = tuple(distribution.files or ())
    record = next((item for item in files if str(item).endswith(".dist-info/RECORD")), None)
    direct_url = next(
        (item for item in files if str(item).endswith(".dist-info/direct_url.json")), None
    )
    return {
        "version": distribution.version,
        "record_sha256": (
            file_sha256(Path(distribution.locate_file(record))) if record is not None else None
        ),
        "direct_url_sha256": (
            file_sha256(Path(distribution.locate_file(direct_url)))
            if direct_url is not None else None
        ),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(out_json: str, payload: dict) -> None:
    if payload.get("artifact_type") != SCHEMA or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("refusing to write a malformed GDN determinism report")
    if out_json == "-":
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        _atomic_json(Path(out_json), payload)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tensor_bytes(torch, value) -> bytes:
    return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def _tensor_sha256(torch, value) -> str:
    return hashlib.sha256(_tensor_bytes(torch, value)).hexdigest()


def _result_sha256(torch, output, state) -> str:
    digest = hashlib.sha256()
    digest.update(b"output-bf16\0")
    digest.update(_tensor_bytes(torch, output))
    digest.update(b"final-state-fp32\0")
    digest.update(_tensor_bytes(torch, state))
    return digest.hexdigest()


def _input_sha256(torch, tensors: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for name, value in tensors.items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(_tensor_bytes(torch, value))
    return digest.hexdigest()


def _integer_pattern(torch, shape: tuple[int, ...], multiplier: int, offset: int, divisor: float):
    count = 1
    for extent in shape:
        count *= extent
    indices = torch.arange(count, dtype=torch.int64)
    values = ((indices * multiplier + offset) % 257 - 128).to(torch.float32) / divisor
    return values.reshape(shape)


def _make_inputs(torch, rows: int) -> dict[str, object]:
    q = _integer_pattern(torch, (rows, 16, 128), 37, 11, 64.0)
    k = _integer_pattern(torch, (rows, 16, 128), 53, 29, 64.0)
    q = (q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1.0e-6)).to(
        torch.bfloat16
    )
    k = (k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1.0e-6)).to(
        torch.bfloat16
    )
    value = _integer_pattern(torch, (rows, 48, 128), 71, 7, 48.0).to(torch.bfloat16)
    decay = -(
        ((_integer_pattern(torch, (rows, 48), 19, 3, 1.0) + 128.0) % 257.0)
        + 1.0
    ) / 4096.0
    beta = (
        (_integer_pattern(torch, (rows, 48), 23, 17, 1.0) + 128.0) % 257.0
    ) / 256.0
    state = _integer_pattern(torch, (48, 128, 128), 31, 5, 8192.0)
    return {"q": q, "k": k, "value": value, "decay": decay, "beta": beta, "state": state}


def _execute_route(
    torch, backend, chunk_recurrent, fused_recurrent, route: str,
    tensors: dict[str, object],
):
    q = tensors["q"].to(backend_device := tensors["device"])
    k = tensors["k"].to(backend_device)
    value = tensors["value"].to(backend_device)
    decay = tensors["decay"].to(backend_device)
    beta = tensors["beta"].to(backend_device)
    state = tensors["state"].to(backend_device)
    if route == "project-naive":
        return backend._gdn_recurrence_naive(q, k, value, decay, beta, state.clone())
    arguments = {
        "q": q.unsqueeze(0).clone(),
        "k": k.unsqueeze(0).clone(),
        "v": value.unsqueeze(0).clone(),
        "g": decay.unsqueeze(0).clone(),
        "beta": beta.unsqueeze(0).clone(),
        "scale": backend.GDN_SCALE,
        "initial_state": state.unsqueeze(0).clone(),
        "output_final_state": True,
    }
    if route == "chunk":
        output, final = chunk_recurrent(**arguments)
    elif route == "fused-recurrent":
        output, final = fused_recurrent(**arguments)
    else:  # protected by parser; keep the execution boundary explicit
        raise ValueError(f"unknown recurrence route {route}")
    return output.squeeze(0).to(torch.bfloat16), final.squeeze(0).float()


def _delta(torch, actual, reference) -> dict:
    difference = actual.detach().float().cpu() - reference.detach().float().cpu()
    absolute = difference.abs()
    finite = torch.isfinite(absolute)
    finite_values = absolute[finite]
    return {
        "different_elements": int(torch.count_nonzero(difference).item()),
        "nonfinite_elements": int(torch.count_nonzero(~finite).item()),
        "max_abs": float(finite_values.max().item()) if finite_values.numel() else 0.0,
        "mean_abs": float(finite_values.mean().item()) if finite_values.numel() else 0.0,
    }


def run_probe(options: Options) -> dict:
    try:
        import torch
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
        from tools.reference.qwen3_8_27b_bf16 import backend
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("probe requires the scorer's ROCm PyTorch and FLA environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("probe requires a ROCm accelerator")
    torch.use_deterministic_algorithms(options.deterministic_algorithms)
    device = torch.device("cuda", options.device)
    properties = torch.cuda.get_device_properties(device)
    fla_chunk_module = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    fla_fused_module = importlib.import_module("fla.ops.gated_delta_rule.fused_recurrent")
    cpu_inputs = _make_inputs(torch, options.rows)
    input_identity = {
        "generator": GENERATOR,
        "sha256": _input_sha256(torch, cpu_inputs),
        "tensors": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": _tensor_sha256(torch, value)}
            for name, value in cpu_inputs.items()
        },
    }
    execution_inputs = dict(cpu_inputs)
    execution_inputs["device"] = device
    route_results: dict[str, list[dict]] = {}
    retained: dict[str, list[tuple[object, object]]] = {}
    with torch.inference_mode():
        for route in options.routes:
            route_results[route] = []
            retained[route] = []
            for repeat in range(options.repeats):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                output, state = _execute_route(
                    torch, backend, chunk_gated_delta_rule,
                    fused_recurrent_gated_delta_rule, route, execution_inputs
                )
                torch.cuda.synchronize(device)
                retained[route].append((output.detach().cpu(), state.detach().cpu()))
                route_results[route].append({
                    "repeat": repeat,
                    "seconds": time.perf_counter() - started,
                    "output_sha256": _tensor_sha256(torch, output),
                    "final_state_sha256": _tensor_sha256(torch, state),
                    "combined_sha256": _result_sha256(torch, output, state),
                    "output_finite": bool(torch.isfinite(output).all().item()),
                    "final_state_finite": bool(torch.isfinite(state).all().item()),
                })
    comparisons = []
    for route, values in retained.items():
        reference_output, reference_state = values[0]
        for repeat, (output, state) in enumerate(values[1:], 1):
            comparisons.append({
                "kind": "within-route",
                "route": route,
                "reference_repeat": 0,
                "repeat": repeat,
                "output": _delta(torch, output, reference_output),
                "final_state": _delta(torch, state, reference_state),
            })
    authority = "project-naive" if "project-naive" in retained else options.routes[0]
    authority_output, authority_state = retained[authority][0]
    for route in options.routes:
        if route == authority:
            continue
        output, state = retained[route][0]
        comparisons.append({
            "kind": "cross-route",
            "route": route,
            "reference_route": authority,
            "output": _delta(torch, output, authority_output),
            "final_state": _delta(torch, state, authority_state),
        })
    repeat_exact = {
        route: len({run["combined_sha256"] for run in runs}) == 1
        for route, runs in route_results.items()
    }
    module_root = Path(__file__).resolve().parent
    python_executable = Path(sys.executable).resolve()
    matmul = torch.backends.cuda.matmul
    return {
        "artifact_type": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "rows": options.rows,
        "repeats": options.repeats,
        "route_order": list(options.routes),
        "purpose": (
            "scorer-4096-inter-chunk-replay"
            if options.rows == 4096 and {"chunk", "fused-recurrent"}.issubset(options.routes)
            else "short-formula-check"
            if options.rows <= 256 and "project-naive" in options.routes
            else "custom-recurrence-replay"
        ),
        "input": input_identity,
        "routes": route_results,
        "repeat_hashes_exact": repeat_exact,
        "all_repeat_hashes_exact": all(repeat_exact.values()),
        "comparisons": comparisons,
        "provenance": {
            "python": sys.version,
            "python_executable": str(python_executable),
            "python_executable_sha256": file_sha256(python_executable),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("torch", "triton", "flash-linear-attention", "safetensors")
            },
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_hip_version": torch.version.hip,
            "deterministic_algorithms_requested": options.deterministic_algorithms,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_reduction": {
                name: getattr(matmul, name, None)
                for name in (
                    "allow_tf32",
                    "allow_fp16_reduced_precision_reduction",
                    "allow_bf16_reduced_precision_reduction",
                    "allow_fp16_accumulation",
                )
            },
            "device_index": options.device,
            "device_name": properties.name,
            "device_arch": getattr(properties, "gcnArchName", None),
            "environment": {name: os.environ.get(name) for name in ENVIRONMENT_KEYS},
            "fla_runtime": {
                "cache_mode": os.environ.get("FLA_CACHE_MODE", "disabled"),
                "cache_results": os.environ.get("FLA_CACHE_RESULTS", "1") == "1",
                "backend_dispatch_disabled":
                    os.environ.get("FLA_DISABLE_BACKEND_DISPATCH", "0") == "1",
                "tensor_cache_disabled":
                    os.environ.get("FLA_DISABLE_TENSOR_CACHE", "0") == "1",
            },
            "sources": {
                name: file_sha256(module_root / name)
                for name in ("gdn_determinism_probe.py", "backend.py")
            },
            "fla_sources": {
                "chunk": file_sha256(Path(fla_chunk_module.__file__)),
                "fused_recurrent": file_sha256(Path(fla_fused_module.__file__)),
                "python_tree": python_tree_sha256(Path(fla_chunk_module.__file__).parents[2]),
            },
            "distributions": {
                name: distribution_record(name)
                for name in ("torch", "triton", "flash-linear-attention", "safetensors")
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        options = parse_options(argv)
        report = run_probe(options)
        write_report(options.out_json, report)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ninfer-bf16-gdn-determinism-probe: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
