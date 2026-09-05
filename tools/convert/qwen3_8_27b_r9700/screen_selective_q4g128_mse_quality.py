"""Qualification-only CPU screen for selective source-MSE Q4G128.

This diagnostic never creates an artifact and deliberately does not register a
Q4G128 numeric format.  It samples the exact 160 Text matrices that account for
the production P2048 post-mixer-down and GDN value-Z/output Q4 calls, comparing
the current canonical Q4G64 representation with canonical and source-MSE
Q4G128 representations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch

from tools.artifact.numeric import QuantFormat

from . import fp8_hybrid_inventory, mse_quantize, preflight_identity, source_inventory
from . import source_recipe
from .q4_row_scaled import quantize_dequantize
from .screen_q4_row_scaled_quality import (
    SampledShardReader,
    _sample_expression,
    canonical_json,
    select_row_indices,
    sha256_file,
    write_no_clobber,
)


SCHEMA = "ninfer.qwen3_8_27b.selective-q4g128-mse-source-screen.v1"
STATUS = "qualification_only_no_artifact_or_runtime_format"
ROWS_PER_TENSOR = 8
MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2 = 1.03
Q4G128_DIAGNOSTIC = QuantFormat("Q4G128_F16S_DIAGNOSTIC_ONLY", 4, 128, -8, 7)

ROLE_SHAPES_AND_COUNTS = {
    "gdn_output": ((5120, 6144), 48),
    "gdn_value_z": ((12288, 5120), 48),
    "mlp_down": ((5120, 17408), 64),
}


def _role(name: str) -> str | None:
    if name.endswith("/gdn/output"):
        return "gdn_output"
    if name.endswith("/gdn/value_z"):
        return "gdn_value_z"
    if name.endswith("/mlp/down"):
        return "mlp_down"
    return None


def selective_specs() -> tuple[fp8_hybrid_inventory.TensorSpec, ...]:
    """Return and validate the exact production-P2048 selective scope."""

    actual = tuple(
        spec for spec in fp8_hybrid_inventory.TENSOR_SPECS
        if spec.name.startswith("text/layers/")
        and spec.format == fp8_hybrid_inventory.Q4
        and _role(spec.name) is not None
    )
    expected_names = {
        *(f"text/layers/{layer}/mlp/down" for layer in range(64)),
        *(
            f"text/layers/{layer}/gdn/{suffix}"
            for layer in range(64)
            if layer not in source_inventory.FULL_ATTENTION_LAYERS
            for suffix in ("value_z", "output")
        ),
    }
    actual_names = {spec.name for spec in actual}
    if len(actual) != 160 or actual_names != expected_names:
        raise ValueError("selective Q4G128 scope differs from the exact 160 Text matrices")
    counts = Counter((_role(spec.name), spec.shape) for spec in actual)
    expected_counts = Counter(
        {(role, shape): count for role, (shape, count) in ROLE_SHAPES_AND_COUNTS.items()}
    )
    if counts != expected_counts:
        raise ValueError(f"selective Q4G128 shapes/counts differ: {counts!r}")
    return actual


def _metrics(source: torch.Tensor, decoded: torch.Tensor) -> dict[str, float]:
    error = source.to(torch.float64) - decoded.to(torch.float64)
    squared_error = float(torch.sum(error * error).item())
    squared_reference = float(torch.sum(source.to(torch.float64) ** 2).item())
    return {
        "relative_l2": math.sqrt(squared_error / squared_reference)
        if squared_reference else 0.0,
        "max_abs": float(torch.max(torch.abs(error)).item()) if error.numel() else 0.0,
        "squared_error": squared_error,
        "squared_reference": squared_reference,
    }


def _mse_q4g128_decode(source: torch.Tensor) -> torch.Tensor:
    if source.dtype != torch.bfloat16 or source.ndim != 2:
        raise TypeError("selective Q4G128 source must be a rank-two BF16 tensor")
    rows, columns = map(int, source.shape)
    if columns <= 0 or columns % Q4G128_DIAGNOSTIC.group_size:
        raise ValueError("selective Q4G128 sampled width must be a positive multiple of 128")
    grouped = (
        source.to(dtype=torch.float32).numpy()
        .reshape(rows, columns // Q4G128_DIAGNOSTIC.group_size,
                 Q4G128_DIAGNOSTIC.group_size)
    )
    scales = mse_quantize._optimize_group_scales(grouped, Q4G128_DIAGNOSTIC)
    codes = mse_quantize._codes(
        grouped, scales, Q4G128_DIAGNOSTIC.qmin, Q4G128_DIAGNOSTIC.qmax
    )
    represented = codes.astype(np.float32) * scales.astype(np.float32)[..., None]
    return torch.from_numpy(represented.reshape(rows, columns))


def measure(spec: fp8_hybrid_inventory.TensorSpec, indices: tuple[int, ...],
            source: torch.Tensor) -> dict[str, object]:
    role = _role(spec.name)
    if role is None:
        raise ValueError(f"{spec.name}: outside selective Q4G128 scope")
    if source.dtype != torch.bfloat16 or tuple(source.shape) != (len(indices), spec.shape[1]):
        raise ValueError(f"{spec.name}: sampled BF16 source has wrong shape or dtype")
    if not bool(torch.isfinite(source).all()):
        raise ValueError(f"{spec.name}: sampled source contains nonfinite values")
    decoded = {
        "q4g64_absmax": quantize_dequantize(source, group_size=64),
        "q4g128_absmax": quantize_dequantize(source, group_size=128),
        "q4g128_mse": _mse_q4g128_decode(source),
    }
    metrics = {name: _metrics(source, value) for name, value in decoded.items()}
    control = metrics["q4g64_absmax"]["relative_l2"]
    return {
        "name": spec.name,
        "role": role,
        "shape": list(spec.shape),
        "sampled_row_indices": list(indices),
        "sampled_rows": len(indices),
        "sampled_elements": source.numel(),
        "zero_rows": int((source == 0).all(dim=1).sum().item()),
        "metrics": metrics,
        "q4g128_mse_over_q4g64_relative_l2": (
            metrics["q4g128_mse"]["relative_l2"] / control if control else None
        ),
    }


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "sampled_rows": sum(int(row["sampled_rows"]) for row in rows),
        "sampled_elements": sum(int(row["sampled_elements"]) for row in rows),
        "zero_rows": sum(int(row["zero_rows"]) for row in rows),
    }
    for codec in ("q4g64_absmax", "q4g128_absmax", "q4g128_mse"):
        squared_error = sum(float(row["metrics"][codec]["squared_error"]) for row in rows)
        squared_reference = sum(
            float(row["metrics"][codec]["squared_reference"]) for row in rows
        )
        result[codec] = {
            "relative_l2": math.sqrt(squared_error / squared_reference)
            if squared_reference else 0.0,
            "max_abs": max(float(row["metrics"][codec]["max_abs"]) for row in rows),
            "squared_error": squared_error,
            "squared_reference": squared_reference,
        }
    control = result["q4g64_absmax"]["relative_l2"]
    result["q4g128_absmax_over_q4g64_relative_l2"] = (
        result["q4g128_absmax"]["relative_l2"] / control if control else None
    )
    result["q4g128_mse_over_q4g64_relative_l2"] = (
        result["q4g128_mse"]["relative_l2"] / control if control else None
    )
    result["q4g128_mse_over_q4g128_absmax_relative_l2"] = (
        result["q4g128_mse"]["relative_l2"]
        / result["q4g128_absmax"]["relative_l2"]
        if result["q4g128_absmax"]["relative_l2"] else None
    )
    return result


def assemble_report(*, tensors: Sequence[dict[str, object]], source: Mapping[str, object],
                    implementation: Mapping[str, str], command: Sequence[str],
                    rows_per_tensor: int = ROWS_PER_TENSOR) -> dict[str, object]:
    ordered = sorted(tensors, key=lambda row: str(row["name"]))
    aggregate = _aggregate(ordered)
    roles = {
        role: _aggregate([row for row in ordered if row["role"] == role])
        for role in sorted(ROLE_SHAPES_AND_COUNTS)
    }
    ratios = {
        role: float(value["q4g128_mse_over_q4g64_relative_l2"])
        for role, value in roles.items()
    }
    ratio = float(aggregate["q4g128_mse_over_q4g64_relative_l2"])
    passed = ratio <= MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2 and all(
        value <= MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2 for value in ratios.values()
    )
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "scope": {
            "matrix_count": 160,
            "roles": {
                role: {"shape": list(shape), "count": count}
                for role, (shape, count) in sorted(ROLE_SHAPES_AND_COUNTS.items())
            },
            "excludes_mtp": True,
        },
        "sampling": {
            "algorithm": "endpoints-plus-sha256-counter-v1",
            "rows_per_tensor": rows_per_tensor,
        },
        "formats": {
            "control": "canonical signed Q4G64 with FP16 scales",
            "diagnostic_control": "canonical signed Q4G128 with FP16 scales",
            "candidate": "eight-step source-MSE signed Q4G128 with FP16 scales",
            "candidate_registered": False,
        },
        "aggregate": aggregate,
        "roles": roles,
        "decision": {
            "maximum_candidate_over_control_relative_l2":
                MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2,
            "aggregate_candidate_over_control_relative_l2": ratio,
            "role_candidate_over_control_relative_l2": ratios,
            "sampled_source_gate_pass": passed,
            "next_if_pass": "source-only 8K PPL; no artifact or runtime implementation",
        },
        "source": dict(source),
        "implementation_sha256": dict(sorted(implementation.items())),
        "command": list(command),
        "tensors": ordered,
    }


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)


def validate_report(report: Mapping[str, object]) -> None:
    """Recompute the exact scope, aggregates, ratios, and decision from a report."""

    if report.get("schema") != SCHEMA or report.get("status") != STATUS:
        raise ValueError("selective Q4G128 report schema/status differs")
    tensors = report.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("selective Q4G128 report tensors are missing")
    specs = selective_specs()
    expected = {spec.name: spec for spec in specs}
    if len(tensors) != 160 or {row.get("name") for row in tensors} != set(expected):
        raise ValueError("selective Q4G128 report does not contain the exact 160 matrices")
    rows_per_tensor = report.get("sampling", {}).get("rows_per_tensor")
    if type(rows_per_tensor) is not int or rows_per_tensor <= 0:
        raise ValueError("selective Q4G128 rows_per_tensor is invalid")
    for row in tensors:
        spec = expected[row["name"]]
        if row.get("shape") != list(spec.shape) or row.get("role") != _role(spec.name):
            raise ValueError(f"{spec.name}: report shape/role differs")
        indices = select_row_indices(spec.name, spec.shape[0], rows_per_tensor)
        if row.get("sampled_row_indices") != list(indices):
            raise ValueError(f"{spec.name}: sampled row selection differs")
        if row.get("sampled_rows") != len(indices):
            raise ValueError(f"{spec.name}: sampled row count differs")
        if row.get("sampled_elements") != len(indices) * spec.shape[1]:
            raise ValueError(f"{spec.name}: sampled element count differs")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != {
            "q4g64_absmax", "q4g128_absmax", "q4g128_mse"
        }:
            raise ValueError(f"{spec.name}: metric set differs")
        references = set()
        for codec, values in metrics.items():
            numeric = [float(values[key]) for key in
                       ("relative_l2", "max_abs", "squared_error", "squared_reference")]
            if not all(math.isfinite(value) and value >= 0 for value in numeric):
                raise ValueError(f"{spec.name}: {codec} metrics are invalid")
            references.add(numeric[3])
            expected_l2 = math.sqrt(numeric[2] / numeric[3]) if numeric[3] else 0.0
            if not _close(numeric[0], expected_l2):
                raise ValueError(f"{spec.name}: {codec} relative-L2 differs")
        if len(references) != 1:
            raise ValueError(f"{spec.name}: squared-reference differs across codecs")
    recomputed = _aggregate(tensors)
    aggregate = report.get("aggregate")
    for key, expected_value in recomputed.items():
        actual = aggregate.get(key)
        if isinstance(expected_value, dict):
            if set(actual) != set(expected_value) or any(
                not _close(float(actual[name]), float(value))
                for name, value in expected_value.items()
            ):
                raise ValueError(f"selective Q4G128 aggregate {key} differs")
        elif isinstance(expected_value, float):
            if not _close(float(actual), expected_value):
                raise ValueError(f"selective Q4G128 aggregate {key} differs")
        elif actual != expected_value:
            raise ValueError(f"selective Q4G128 aggregate {key} differs")
    recomputed_roles = {
        role: _aggregate([row for row in tensors if row["role"] == role])
        for role in sorted(ROLE_SHAPES_AND_COUNTS)
    }
    if report.get("roles") != recomputed_roles:
        raise ValueError("selective Q4G128 role aggregates differ")
    ratio = float(recomputed["q4g128_mse_over_q4g64_relative_l2"])
    role_ratios = {
        role: float(values["q4g128_mse_over_q4g64_relative_l2"])
        for role, values in recomputed_roles.items()
    }
    decision = report.get("decision", {})
    expected_pass = ratio <= MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2 and all(
        value <= MAX_CANDIDATE_OVER_CONTROL_RELATIVE_L2 for value in role_ratios.values()
    )
    if (decision.get("aggregate_candidate_over_control_relative_l2") != ratio
            or decision.get("role_candidate_over_control_relative_l2") != role_ratios
            or decision.get("sampled_source_gate_pass") is not expected_pass):
        raise ValueError("selective Q4G128 decision differs")


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        Path(__file__).resolve(),
        root / "tools/convert/qwen3_8_27b_r9700/mse_quantize.py",
        root / "tools/convert/qwen3_8_27b_r9700/q4_row_scaled.py",
        root / "tools/convert/qwen3_8_27b_r9700/fp8_hybrid_inventory.py",
        root / "tools/convert/qwen3_8_27b_r9700/source_recipe.py",
        root / "tools/convert/qwen3_8_27b_r9700/screen_q4_row_scaled_quality.py",
    )
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def _validate_provenance(report: Mapping[str, object], model_dir: Path) -> None:
    current_source = preflight_identity.source_checkpoint(model_dir)
    if report.get("source") != current_source:
        raise ValueError("selective Q4G128 report source provenance differs")
    if report.get("implementation_sha256") != dict(sorted(_implementation_hashes().items())):
        raise ValueError("selective Q4G128 report implementation provenance differs")


def run_screen(model_dir: Path, output: Path, *, rows_per_tensor: int,
               command: Sequence[str]) -> dict[str, object]:
    if rows_per_tensor <= 0:
        raise ValueError("rows-per-tensor must be positive")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite quality report: {output}")
    source = preflight_identity.source_checkpoint(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = dict(index["weight_map"])
    records = []
    specs = selective_specs()
    with SampledShardReader(model_dir, weight_map) as reader:
        for ordinal, spec in enumerate(specs, 1):
            indices = select_row_indices(spec.name, spec.shape[0], rows_per_tensor)
            sampled = _sample_expression(
                source_recipe.RECIPES_BY_NAME[spec.name].expression, indices, reader, []
            )
            records.append(measure(spec, indices, sampled))
            if ordinal % 40 == 0:
                print(f"sampled {ordinal}/{len(specs)} matrices", flush=True)
    report = assemble_report(
        tensors=records,
        source=source,
        implementation=_implementation_hashes(),
        command=command,
        rows_per_tensor=rows_per_tensor,
    )
    validate_report(report)
    write_no_clobber(output, canonical_json(report))
    reopened = json.loads(output.read_text(encoding="utf-8"))
    validate_report(reopened)
    _validate_provenance(reopened, model_dir)
    return reopened


def validate_file(path: Path, model_dir: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_report(report)
    _validate_provenance(report, model_dir)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--validate", type=Path)
    parser.add_argument("--rows-per-tensor", type=int, default=ROWS_PER_TENSOR)
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", __spec__.name if __spec__ else __name__,
               *(list(argv) if argv is not None else sys.argv[1:])]
    if args.validate is not None:
        report = validate_file(args.validate, args.model_dir)
    else:
        report = run_screen(args.model_dir, args.output,
                            rows_per_tensor=args.rows_per_tensor, command=command)
    print(json.dumps(report["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
