#!/usr/bin/env python3
"""Fresh-process determinism probe for the BF16 scorer's FP32 attention PV."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time
from typing import Sequence
import uuid


SCHEMA = "ninfer_qwen3_8_27b_bf16_pv_determinism_probe"
COMPARISON_SCHEMA = "ninfer_qwen3_8_27b_bf16_pv_determinism_comparison"
SCHEMA_VERSION = 1
SOURCE_EXTENTS = (8192, 8193, 16384, 32768)
SOURCE_CHUNK = 8192
QUERY_ROWS = 32
QUERY_HEADS = 6
HEAD_DIM = 256
SAMPLES = ((0, 0, 0), (0, 5, 255), (7, 2, 31), (15, 4, 127),
           (23, 1, 191), (31, 3, 64))
ATOL = 5.0e-4
RTOL = 5.0e-4
GENERATOR = "closed-form-fp32-probability-value-v1"
ENVIRONMENT_KEYS = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
    "ROCBLAS_DEFAULT_ATOMICS_MODE", "TORCH_BLAS_PREFER_HIPBLASLT",
)
DETERMINISTIC_ENVIRONMENT = {
    "TORCH_BLAS_PREFER_HIPBLASLT": "0",
    "ROCBLAS_DEFAULT_ATOMICS_MODE": "0",
}


def source_chunks(rows: int, chunk: int = SOURCE_CHUNK) -> tuple[tuple[int, int], ...]:
    if rows <= 0 or chunk <= 0:
        raise ValueError("PV rows and source chunk must be positive")
    return tuple((begin, min(begin + chunk, rows)) for begin in range(0, rows, chunk))


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def probability_denominator(source_rows: int, query_row: int, query_head: int) -> int:
    """Exact integer denominator for the represented periodic probability numerators."""
    offset = (query_row * 17 + query_head * 29) % 251
    cycles, tail = divmod(source_rows, 251)
    total = cycles * (251 * 252 // 2)
    total += sum(((offset + source) % 251) + 1 for source in range(tail))
    return total


def represented_probability(source: int, source_rows: int, query_row: int,
                            query_head: int) -> float:
    numerator = ((source + query_row * 17 + query_head * 29) % 251) + 1
    return _f32(_f32(float(numerator)) / _f32(float(
        probability_denominator(source_rows, query_row, query_head))))


def represented_value(source: int, feature: int) -> float:
    integer = ((source * 13 + feature * 37 + 11) % 257) - 128
    return _f32(_f32(float(integer)) / _f32(128.0))


def sampled_fp64_oracle(source_rows: int, sample: tuple[int, int, int]) -> float:
    query_row, query_head, feature = sample
    denominator = probability_denominator(source_rows, query_row, query_head)
    terms = (
        _f32(_f32(float(((source + query_row * 17 + query_head * 29) % 251) + 1)) /
             _f32(float(denominator))) * represented_value(source, feature)
        for source in range(source_rows)
    )
    return math.fsum(terms)


def fixed_order_pv(torch, probabilities, values, source_chunk: int = SOURCE_CHUNK):
    """FP32 PV with absolute, ascending source chunks and ordered FP32 accumulation."""
    if probabilities.dtype != torch.float32 or values.dtype != torch.float32:
        raise ValueError("PV probe operands must be FP32")
    if probabilities.ndim != 3 or values.ndim != 2:
        raise ValueError("PV probe operands must be [T,H,S] and [S,D]")
    if probabilities.shape[2] != values.shape[0]:
        raise ValueError("PV probe source extents differ")
    result = None
    output_shape = (*probabilities.shape[:2], values.shape[1])
    for begin, end in source_chunks(probabilities.shape[2], source_chunk):
        left = probabilities[:, :, begin:end].contiguous().view(-1, end - begin)
        partial = torch.mm(left, values[begin:end]).view(output_shape)
        if result is None:
            result = partial
        else:
            result.add_(partial)
    assert result is not None
    return result


def one_shot_pv(torch, probabilities, values):
    return torch.einsum("ths,sd->thd", probabilities, values)


def _tensor_bytes(torch, value) -> bytes:
    return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_inputs(torch, source_rows: int):
    source = torch.arange(source_rows, dtype=torch.int64)
    query = torch.arange(QUERY_ROWS, dtype=torch.int64)[:, None, None]
    head = torch.arange(QUERY_HEADS, dtype=torch.int64)[None, :, None]
    numerator = ((source[None, None, :] + query * 17 + head * 29) % 251 + 1).to(torch.float32)
    denominators = torch.tensor(
        [[probability_denominator(source_rows, row, h) for h in range(QUERY_HEADS)]
         for row in range(QUERY_ROWS)], dtype=torch.float32,
    )
    probabilities = numerator / denominators[:, :, None]
    feature = torch.arange(HEAD_DIM, dtype=torch.int64)[None, :]
    values = (((source[:, None] * 13 + feature * 37 + 11) % 257) - 128).to(
        torch.float32) / 128.0
    return probabilities, values


def _accuracy(torch, output, source_rows: int) -> dict:
    host = output.detach().cpu()
    records = []
    passed = True
    for sample in SAMPLES:
        expected = sampled_fp64_oracle(source_rows, sample)
        actual = float(host[sample].item())
        absolute = abs(actual - expected)
        tolerance = ATOL + RTOL * abs(expected)
        ok = math.isfinite(actual) and absolute <= tolerance
        passed = passed and ok
        records.append({
            "coordinate": list(sample), "actual_fp32": actual,
            "oracle_fp64": expected, "absolute_error": absolute,
            "tolerance": tolerance, "pass": ok,
        })
    return {"pass": passed, "atol": ATOL, "rtol": RTOL, "samples": records}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def establish_environment() -> None:
    for key, required in DETERMINISTIC_ENVIRONMENT.items():
        current = os.environ.get(key)
        if current is not None and current != required:
            raise ValueError(f"{key}={current!r} conflicts with required value {required!r}")
    os.environ.update(DETERMINISTIC_ENVIRONMENT)


def preferred_blas_library(torch) -> str:
    value = torch.backends.cuda.preferred_blas_library()
    mapping = {
        "Default": "default",
        "Cublas": "hipblas" if torch.version.hip else "cublas",
        "Cublaslt": "hipblaslt" if torch.version.hip else "cublaslt",
        "Ck": "ck",
    }
    try:
        return mapping[value.name]
    except (AttributeError, KeyError) as error:
        raise RuntimeError(f"unknown PyTorch BLAS backend {value!r}") from error


def run_probe(device_index: int) -> dict:
    establish_environment()
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("PV probe requires the scorer's ROCm PyTorch environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("PV probe requires a ROCm accelerator")
    if device_index < 0:
        raise ValueError("--device must be nonnegative")
    torch.use_deterministic_algorithms(True, warn_only=False)
    preferred_blas = preferred_blas_library(torch)
    if preferred_blas != "hipblas":
        raise RuntimeError(f"PV probe requires hipBLAS, resolved {preferred_blas!r}")
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    cases = []
    with torch.inference_mode():
        for source_rows in SOURCE_EXTENTS:
            host_probabilities, host_values = _make_inputs(torch, source_rows)
            input_digest = hashlib.sha256()
            input_digest.update(_tensor_bytes(torch, host_probabilities))
            input_digest.update(_tensor_bytes(torch, host_values))
            probabilities = host_probabilities.to(device)
            values = host_values.to(device)
            routes = {}
            retained = {}
            for name, implementation in (
                ("one-shot-einsum", one_shot_pv),
                ("fixed-ascending-chunks", fixed_order_pv),
            ):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = implementation(torch, probabilities, values)
                torch.cuda.synchronize(device)
                retained[name] = output.detach().cpu()
                routes[name] = {
                    "seconds": time.perf_counter() - started,
                    "sha256": _sha256_bytes(_tensor_bytes(torch, output)),
                    "finite": bool(torch.isfinite(output).all().item()),
                    "accuracy": _accuracy(torch, output, source_rows),
                }
            difference = retained["fixed-ascending-chunks"] - retained["one-shot-einsum"]
            cases.append({
                "source_rows": source_rows,
                "input_sha256": input_digest.hexdigest(),
                "chunks": [list(value) for value in source_chunks(source_rows)],
                "routes": routes,
                "cross_route": {
                    "different_elements": int(torch.count_nonzero(difference).item()),
                    "max_abs": float(difference.abs().max().item()),
                },
            })
            del probabilities, values, host_probabilities, host_values, retained
    executable = Path(sys.executable).resolve()
    source = Path(__file__).resolve()
    return {
        "artifact_type": SCHEMA, "schema_version": SCHEMA_VERSION,
        "quality_evidence": False,
        "run": {"id": str(uuid.uuid4()), "process_id": os.getpid()},
        "generator": GENERATOR,
        "geometry": {"query_rows": QUERY_ROWS, "query_heads": QUERY_HEADS,
                     "head_dim": HEAD_DIM, "source_extents": list(SOURCE_EXTENTS)},
        "fixed_route": {"source_chunk": SOURCE_CHUNK, "order": "absolute-ascending",
                        "partial": "torch.mm-fp32", "accumulation": "ordered-fp32-add"},
        "all_accuracy_pass": all(
            route["accuracy"]["pass"] and route["finite"]
            for case in cases for route in case["routes"].values()),
        "cases": cases,
        "provenance": {
            "python": sys.version, "python_executable": str(executable),
            "python_executable_sha256": _file_sha256(executable),
            "platform": platform.platform(), "torch": torch.__version__,
            "torch_git": torch.version.git_version, "hip": torch.version.hip,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "preferred_blas_library": preferred_blas,
            "device_index": device_index, "device_name": properties.name,
            "device_arch": getattr(properties, "gcnArchName", None),
            "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
            "source_sha256": _file_sha256(source),
            "distributions": {name: _distribution_version(name) for name in ("torch", "triton")},
        },
    }


def validate_report(payload: dict) -> None:
    if payload.get("artifact_type") != SCHEMA or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a PV determinism probe schema-v1 report")
    if payload.get("generator") != GENERATOR:
        raise ValueError("PV determinism probe generator differs")
    cases = payload.get("cases")
    if not isinstance(cases, list) or [case.get("source_rows") for case in cases] != list(SOURCE_EXTENTS):
        raise ValueError("PV determinism probe source inventory differs")
    if not payload.get("all_accuracy_pass"):
        raise ValueError("PV determinism probe failed its FP64 sampled oracle")
    for case in cases:
        if set(case.get("routes", {})) != {"one-shot-einsum", "fixed-ascending-chunks"}:
            raise ValueError("PV determinism probe route inventory differs")
        if case.get("chunks") != [list(value) for value in source_chunks(case["source_rows"])]:
            raise ValueError("PV determinism probe chunk partition differs")


def compare_reports(first: dict, second: dict) -> dict:
    validate_report(first)
    validate_report(second)
    identity_keys = ("generator", "geometry", "fixed_route", "provenance")
    identity_equal = all(first.get(key) == second.get(key) for key in identity_keys)
    cases = []
    all_exact = identity_equal
    for left, right in zip(first["cases"], second["cases"], strict=True):
        routes = {}
        for route in ("one-shot-einsum", "fixed-ascending-chunks"):
            exact = left["routes"][route]["sha256"] == right["routes"][route]["sha256"]
            routes[route] = {"exact": exact, "first_sha256": left["routes"][route]["sha256"],
                             "second_sha256": right["routes"][route]["sha256"]}
            all_exact = all_exact and exact
        input_exact = left["input_sha256"] == right["input_sha256"]
        all_exact = all_exact and input_exact
        cases.append({"source_rows": left["source_rows"], "input_exact": input_exact,
                      "routes": routes})
    return {"artifact_type": COMPARISON_SCHEMA, "schema_version": SCHEMA_VERSION,
            "quality_evidence": False,
            "identity_equal": identity_equal, "all_routes_exact": all_exact, "cases": cases}


def _write_json(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"), type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.compare:
            payload = compare_reports(*(json.loads(path.read_text()) for path in args.compare))
        else:
            payload = run_probe(args.device)
            validate_report(payload)
        _write_json(args.out_json, payload)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ninfer-bf16-pv-determinism-probe: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
