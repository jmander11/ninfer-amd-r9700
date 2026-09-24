#!/usr/bin/env python3
"""Validate retained split-512 numerical/timing admission evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


SCHEMA = "ninfer_r9700_split512_attention_qualification"
POWER_PATH = "/sys/class/drm/card2/device/power_dpm_force_performance_level"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = {
    "qualifier": "tools/r9700/split512_attention_qual.hip",
    "static_validator": "tools/r9700/check_split512_static.py",
    "independent_oracle": "tests/ops/fp8_int4_kv_oracle.h",
    "contract_header": "src/ops/r9700/kv/fp8_int4_kv_attention.h",
    "kernel": "src/ops/r9700/kv/fp8_int4_kv_attention.hip",
}
SOURCE_SUFFIXES = set(SOURCE_PATHS.values())
SHAPES = {
    (group, context, rows, active)
    for group in (16, 32)
    for context in (8192, 32768)
    for rows, active in ((1, 1), (4, 1), (4, 2), (4, 3), (4, 4))
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_power_profile(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _positive_samples(value: Any, iterations: int, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == iterations, f"{label} length differs")
    _require(
        all(_number(sample) and sample > 0 for sample in value),
        f"{label} must contain finite positive timings",
    )
    return [float(sample) for sample in value]


def split_workspace_bytes(rows: int, context: int) -> int:
    chunks = (context + 511) // 512
    return 4 * (rows * 24 * context + rows * 24 * chunks * (256 + 2))


def validate_report(
    report: Any,
    expected_executable: Path,
    expected_source_root: Path = REPO_ROOT,
    power_reader: Callable[[Path], str] = _read_power_profile,
) -> None:
    _require(isinstance(report, dict), "report must be an object")
    _require(report.get("schema") == SCHEMA, "wrong split-512 report schema")
    _require(type(report.get("schema_version")) is int and report["schema_version"] == 2,
             "wrong split-512 report schema version")
    _require(report.get("architecture") == "gfx1201", "wrong architecture")
    _require(
        report.get("disposition") == "unpromoted-qualification-candidate",
        "wrong candidate disposition",
    )
    _require(
        report.get("physical_qualification")
        == {"status": "passed", "scope": "full-numerical-rejection-device-graph"},
        "physical qualification is incomplete",
    )
    power = report.get("power_profile")
    _require(
        isinstance(power, dict)
        and power.get("value") == "auto"
        and power.get("required") == "auto"
        and power.get("path") == POWER_PATH,
        "timing power profile is not the exact auto contract",
    )
    try:
        live_power = power_reader(Path(POWER_PATH)).strip()
    except OSError as error:
        raise ValueError(f"cannot read live power profile: {error}") from error
    _require(live_power == "auto", "live power profile is not auto")
    device = report.get("device")
    _require(
        isinstance(device, dict)
        and device.get("name") == "AMD Radeon AI PRO R9700"
        and device.get("architecture") == "gfx1201"
        and type(device.get("wave_size")) is int
        and device["wave_size"] == 32
        and type(device.get("runtime_version")) is int
        and device["runtime_version"] > 0
        and type(device.get("driver_version")) is int
        and device["driver_version"] > 0,
        "device provenance is incomplete",
    )
    executable = report.get("executable")
    _require(
        isinstance(executable, dict)
        and isinstance(executable.get("path"), str)
        and Path(executable["path"]).is_absolute(),
        "executable identity is incomplete",
    )
    _digest(executable.get("sha256"), "executable.sha256")
    _require(isinstance(expected_executable, Path), "expected executable is required")
    _require(expected_executable.is_file(), "bound executable is unavailable")
    executable_path = expected_executable.resolve(strict=True)
    _require(Path(executable["path"]) == executable_path,
             "reported executable path differs from expected executable")
    _require(_file_sha256(executable_path) == executable["sha256"],
             "bound executable bytes changed")
    reported_root = report.get("source_root")
    _require(isinstance(reported_root, str) and Path(reported_root).is_absolute(),
             "source root is incomplete")
    _require(expected_source_root.is_dir(), "expected source root is unavailable")
    canonical_expected_root = expected_source_root.resolve(strict=True)
    source_root = Path(reported_root)
    _require(source_root.is_dir() and source_root.resolve(strict=True) == source_root
             and source_root == canonical_expected_root,
             "source root is not the expected exact canonical directory")
    sources = report.get("sources")
    _require(isinstance(sources, list) and len(sources) == len(SOURCE_PATHS), "source inventory differs")
    observed_sources: set[str] = set()
    for index, source in enumerate(sources):
        _require(
            isinstance(source, dict)
            and source.get("role") in SOURCE_PATHS
            and isinstance(source.get("path"), str)
            and Path(source["path"]).is_absolute(),
            f"sources[{index}] identity is incomplete",
        )
        role = source["role"]
        relative = Path(SOURCE_PATHS[role])
        source_path = Path(source["path"])
        _require(role not in observed_sources and source_path == source_root / relative,
                 f"sources[{index}] is not the exact required source")
        _digest(source.get("sha256"), f"sources[{index}].sha256")
        _require(source_path.is_file() and source_path.resolve(strict=True) == source_path,
                 f"sources[{index}] source is unavailable or noncanonical")
        _require(_file_sha256(source_path) == source["sha256"],
                 f"sources[{index}] source bytes changed")
        observed_sources.add(role)
    _require(observed_sources == set(SOURCE_PATHS), "source inventory is not exact")
    workspace = report.get("workspace_contract")
    maximum_context = 262144
    overflow_context = (1 << 64) - 1
    expected_workspace = {
        "status": "passed",
        "maximum_context": maximum_context,
        "t1_maximum_bytes": split_workspace_bytes(1, maximum_context),
        "t4_maximum_bytes": split_workspace_bytes(4, maximum_context),
        "maximum_plus_one_context": maximum_context + 1,
        "t1_maximum_plus_one_bytes": 0,
        "t4_maximum_plus_one_bytes": 0,
        "overflow_context": overflow_context,
        "t1_overflow_bytes": 0,
        "t4_overflow_bytes": 0,
    }
    _require(isinstance(workspace, dict)
             and all(type(workspace.get(key)) is int
                     for key in expected_workspace if key != "status")
             and workspace == expected_workspace,
             "native workspace boundary evidence differs")
    _require(report.get("sample_order") == "alternating-interleaved", "wrong sample order")
    _require(type(report.get("warmup_iterations_per_route")) is int
             and report["warmup_iterations_per_route"] == 3, "wrong warmup count")
    _require(
        report.get("incumbent_scope") == "complete-production-semantic-attention-leaf"
        and report.get("split_scope")
        == "complete-qk-plus-partial-softmax-pv-plus-merge-leaf",
        "timed leaf scope differs",
    )
    measurements = report.get("measurements")
    _require(isinstance(measurements, list) and len(measurements) == len(SHAPES), "measurement inventory differs")
    observed_shapes: set[tuple[int, int, int, int]] = set()
    observed_iterations: int | None = None
    for index, row in enumerate(measurements):
        _require(isinstance(row, dict), f"measurements[{index}] must be an object")
        shape_values = (
            row.get("value_group"), row.get("context"), row.get("query_rows"),
            row.get("active_rows"),
        )
        _require(all(type(value) is int for value in shape_values),
                 f"measurements[{index}] shape types differ")
        shape = shape_values
        _require(shape in SHAPES and shape not in observed_shapes, f"measurements[{index}] shape differs")
        observed_shapes.add(shape)
        iterations = row.get("iterations")
        _require(type(iterations) is int and iterations == 20, "iteration count is not exactly 20")
        if observed_iterations is None:
            observed_iterations = iterations
        _require(iterations == observed_iterations, "iteration count differs across rows")
        _require(
            type(row.get("split_workspace_bytes")) is int
            and row.get("split_workspace_bytes") == split_workspace_bytes(shape[2], shape[1]),
            f"measurements[{index}] workspace differs",
        )
        expected_route = (
            "fp8-q-wmma-score-plus-fp32-softmax-int4-pv"
            if shape[2] == 1
            else "bf16-q-score-streaming-online-fp32-softmax-int4-pv"
        )
        _require(row.get("incumbent_route") == expected_route, f"measurements[{index}] route differs")
        incumbent = _positive_samples(row.get("incumbent_samples_ms"), iterations, "incumbent samples")
        split = _positive_samples(row.get("split_samples_ms"), iterations, "split samples")
        incumbent_mean = sum(incumbent) / iterations
        split_mean = sum(split) / iterations
        for key, expected in (
            ("incumbent_mean_ms", incumbent_mean),
            ("split_mean_ms", split_mean),
            ("incumbent_over_split", incumbent_mean / split_mean),
        ):
            actual = row.get(key)
            _require(
                _number(actual)
                and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=1e-9),
                f"measurements[{index}].{key} differs from raw samples",
            )
    _require(observed_shapes == SHAPES, "measurement Cartesian set is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--executable", type=Path, required=True,
                        help="exact qualification executable recorded in the report")
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT,
                        help="expected source tree (defaults to the current repository)")
    args = parser.parse_args()
    validate_report(json.loads(args.report.read_text(encoding="utf-8")),
                    args.executable, args.source_root)
    print("split-512 timing report passed")


if __name__ == "__main__":
    main()
