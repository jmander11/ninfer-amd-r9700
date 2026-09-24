#!/usr/bin/env python3
"""Sampled FP64 validation of the production-span fused GDN recurrence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Sequence
import uuid


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.reference.qwen3_8_27b_bf16.gdn_determinism_probe import (  # noqa: E402
    GENERATOR,
    _input_sha256,
    _make_inputs,
    _tensor_bytes,
    _tensor_sha256,
    file_sha256,
)
from tools.reference.qwen3_8_27b_bf16.protocol import (  # noqa: E402
    GDN_RECURRENCE_EXECUTION,
    enable_strict_torch_determinism,
    establish_deterministic_environment,
    execution_provenance,
)


SCHEMA = "ninfer_qwen3_8_27b_bf16_gdn_full_span_probe"
SCHEMA_VERSION = 1
ROW_EXTENTS = (4095, 4096)
KEY_HEADS = 16
VALUE_HEADS = 48
KEY_DIM = 128
VALUE_DIM = 128
SAMPLE_COLUMNS = ((0, 0), (5, 127), (17, 31), (32, 64), (47, 126))
OUTPUT_ATOL = 3.0e-2
OUTPUT_RTOL = 3.0e-2
STATE_ATOL = 3.0e-2
STATE_RTOL = 3.0e-2


def sampled_rows(rows: int) -> tuple[int, ...]:
    if rows < 4:
        raise ValueError("full-span GDN probe requires at least four rows")
    return (0, 1, rows // 2, rows - 2, rows - 1)


def sampled_column_fp64(
    q_rows: Sequence[Sequence[float]],
    k_rows: Sequence[Sequence[float]],
    values: Sequence[float],
    decays: Sequence[float],
    betas: Sequence[float],
    initial_column: Sequence[float],
    output_rows: Sequence[int],
    *,
    scale: float,
) -> tuple[dict[int, float], tuple[float, ...]]:
    """Evaluate one complete value-head/state column with FP64 scalar arithmetic.

    Each prediction and output still reduces all represented K coordinates. Sampling only
    independent value columns avoids materializing the full HxKxV FP64 state trajectory.
    """
    rows = len(q_rows)
    if not (
        rows == len(k_rows) == len(values) == len(decays) == len(betas)
        and rows > 0
    ):
        raise ValueError("sampled GDN operands have inconsistent row extents")
    key_dim = len(initial_column)
    if key_dim == 0 or any(len(row) != key_dim for row in q_rows) or any(
        len(row) != key_dim for row in k_rows
    ):
        raise ValueError("sampled GDN operands have inconsistent key dimensions")
    requested = tuple(output_rows)
    if len(set(requested)) != len(requested) or any(row < 0 or row >= rows for row in requested):
        raise ValueError("sampled GDN output rows must be unique and in range")

    wanted = set(requested)
    current = [float(value) for value in initial_column]
    outputs: dict[int, float] = {}
    for row in range(rows):
        key = k_rows[row]
        query = q_rows[row]
        decay = math.exp(float(decays[row]))
        current = [value * decay for value in current]
        prediction = math.fsum(current[index] * float(key[index]) for index in range(key_dim))
        delta = float(betas[row]) * (float(values[row]) - prediction)
        current = [
            current[index] + float(key[index]) * delta
            for index in range(key_dim)
        ]
        if row in wanted:
            outputs[row] = math.fsum(
                current[index] * float(query[index]) for index in range(key_dim)
            ) * scale
    return {row: outputs[row] for row in requested}, tuple(current)


def _tolerance(actual: float, expected: float, atol: float, rtol: float) -> dict:
    absolute = abs(actual - expected)
    limit = atol + rtol * abs(expected)
    return {
        "actual": actual,
        "oracle_fp64": expected,
        "absolute_error": absolute,
        "tolerance": limit,
        "pass": math.isfinite(actual) and math.isfinite(expected) and absolute <= limit,
    }


def _fp64_sha256(values: Sequence[float]) -> str:
    return hashlib.sha256(struct.pack("<" + "d" * len(values), *values)).hexdigest()


def _validate_case(torch, cpu_inputs: dict[str, object], output, final_state) -> dict:
    rows = cpu_inputs["q"].shape[0]
    output_rows = sampled_rows(rows)
    output_columns = {
        coordinate: output[:, coordinate[0], coordinate[1]].detach().float().cpu()
        for coordinate in SAMPLE_COLUMNS
    }
    state_columns = {
        coordinate: final_state[coordinate[0], :, coordinate[1]].detach().float().cpu()
        for coordinate in SAMPLE_COLUMNS
    }
    records = []
    passed = True
    scale = 128 ** -0.5
    for value_head, feature in SAMPLE_COLUMNS:
        key_head = value_head // (VALUE_HEADS // KEY_HEADS)
        expected_outputs, expected_column = sampled_column_fp64(
            cpu_inputs["q"][:, key_head, :].float().tolist(),
            cpu_inputs["k"][:, key_head, :].float().tolist(),
            cpu_inputs["value"][:, value_head, feature].float().tolist(),
            cpu_inputs["decay"][:, value_head].float().tolist(),
            cpu_inputs["beta"][:, value_head].float().tolist(),
            cpu_inputs["state"][value_head, :, feature].float().tolist(),
            output_rows,
            scale=scale,
        )
        output_checks = []
        for row in output_rows:
            check = _tolerance(
                float(output_columns[(value_head, feature)][row].item()),
                expected_outputs[row], OUTPUT_ATOL, OUTPUT_RTOL,
            )
            output_checks.append({"row": row, **check})
            passed = passed and check["pass"]
        actual_tensor = state_columns[(value_head, feature)]
        actual_column = actual_tensor.tolist()
        state_checks = [
            _tolerance(actual, expected, STATE_ATOL, STATE_RTOL)
            for actual, expected in zip(actual_column, expected_column, strict=True)
        ]
        state_pass = all(check["pass"] for check in state_checks)
        passed = passed and state_pass
        worst_index = max(
            range(KEY_DIM), key=lambda index: state_checks[index]["absolute_error"]
        )
        records.append({
            "value_head": value_head,
            "key_head": key_head,
            "value_feature": feature,
            "output_rows": output_checks,
            "final_state_column": {
                "elements_checked": KEY_DIM,
                "pass": state_pass,
                "actual_fp32_sha256": _tensor_sha256(torch, actual_tensor),
                "oracle_fp64_sha256": _fp64_sha256(expected_column),
                "max_absolute_error": state_checks[worst_index]["absolute_error"],
                "worst": {"key_feature": worst_index, **state_checks[worst_index]},
            },
        })
    return {
        "pass": passed,
        "output_criterion": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL},
        "final_state_criterion": {"atol": STATE_ATOL, "rtol": STATE_RTOL},
        "columns": records,
    }


def run_probe(device_index: int) -> dict:
    if device_index < 0:
        raise ValueError("--device must be nonnegative")
    establish_deterministic_environment()
    try:
        import torch
        from tools.reference.qwen3_8_27b_bf16 import backend
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("full-span GDN probe requires the scorer's ROCm environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("full-span GDN probe requires a ROCm accelerator")
    enable_strict_torch_determinism(torch)
    device = torch.device("cuda", device_index)
    cases = []
    with torch.inference_mode():
        for rows in ROW_EXTENTS:
            cpu_inputs = _make_inputs(torch, rows)
            inputs = {name: value.to(device) for name, value in cpu_inputs.items()}
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output, final_state = backend._gdn_recurrence(
                inputs["q"], inputs["k"], inputs["value"], inputs["decay"],
                inputs["beta"], inputs["state"].clone(),
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            validation = _validate_case(torch, cpu_inputs, output, final_state)
            cases.append({
                "rows": rows,
                "input": {
                    "combined_sha256": _input_sha256(torch, cpu_inputs),
                    "tensors": {
                        name: {
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                            "sha256": _tensor_sha256(torch, value),
                        }
                        for name, value in cpu_inputs.items()
                    },
                },
                "fused_recurrent": {
                    "seconds": elapsed,
                    "output_dtype": str(output.dtype),
                    "output_sha256": hashlib.sha256(_tensor_bytes(torch, output)).hexdigest(),
                    "output_finite": bool(torch.isfinite(output).all().item()),
                    "final_state_dtype": str(final_state.dtype),
                    "final_state_sha256": hashlib.sha256(
                        _tensor_bytes(torch, final_state)
                    ).hexdigest(),
                    "final_state_finite": bool(torch.isfinite(final_state).all().item()),
                },
                "sampled_fp64_oracle": validation,
            })
            del inputs, cpu_inputs, output, final_state

    source = Path(__file__).resolve()
    backend_source = source.with_name("backend.py")
    return {
        "artifact_type": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "quality_evidence": False,
        "run": {"id": str(uuid.uuid4()), "process_id": os.getpid()},
        "generator": GENERATOR,
        "geometry": {
            "row_extents": list(ROW_EXTENTS),
            "key_heads": KEY_HEADS,
            "value_heads": VALUE_HEADS,
            "key_dim": KEY_DIM,
            "value_dim": VALUE_DIM,
            "sample_columns": [list(value) for value in SAMPLE_COLUMNS],
            "sample_rows": "0,1,T//2,T-2,T-1",
        },
        "route": dict(GDN_RECURRENCE_EXECUTION),
        "oracle": {
            "implementation": "serial-sampled-column-fp64-v1",
            "complete_rows": True,
            "prediction_key_terms": KEY_DIM,
            "sampled_independent_value_columns": len(SAMPLE_COLUMNS),
            "final_state_elements_checked": len(SAMPLE_COLUMNS) * KEY_DIM,
        },
        "all_pass": all(
            case["fused_recurrent"]["output_finite"]
            and case["fused_recurrent"]["final_state_finite"]
            and case["sampled_fp64_oracle"]["pass"]
            for case in cases
        ),
        "cases": cases,
        "provenance": {
            "execution": execution_provenance(
                torch, device_index, stage_trace_enabled=False
            ),
            "sources": {
                source.name: file_sha256(source),
                backend_source.name: file_sha256(backend_source),
            },
        },
    }


def write_report(path: Path, payload: dict) -> None:
    if payload.get("artifact_type") != SCHEMA or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("refusing to write a malformed full-span GDN probe report")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate production-span fused GDN against sampled FP64 columns."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        options = build_parser().parse_args(argv)
        report = run_probe(options.device)
        write_report(options.out_json, report)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ninfer-bf16-gdn-full-span-probe: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
