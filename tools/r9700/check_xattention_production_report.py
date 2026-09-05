#!/usr/bin/env python3
"""Strictly validate retained XAttention production-scale admission evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


POWER_PATH = "/sys/class/drm/card2/device/power_dpm_force_performance_level"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = {
    "qualifier": "tools/r9700/xattention_prefill_qual.hip",
    "contract_header": "src/ops/r9700/kv/fp8_int4_kv_xattention.h",
    "kernel": "src/ops/r9700/kv/fp8_int4_kv_xattention.hip",
    "profile": "src/ops/r9700/kv/r9700_xattention_profile.h",
    "dense_control_kernel": "src/ops/r9700/kv/fp8_int4_kv_attention.hip",
    "independent_oracle": "tests/ops/xattention_oracle.h",
    "oracle_codec": "tests/ops/fp8_int4_kv_oracle.h",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any, label: str) -> None:
    require(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_power_profile(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def median(samples: list[float]) -> float:
    ordered = sorted(samples)
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0)


def samples(value: Any, count: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == count, f"{label} length differs")
    require(all(number(item) and item > 0
                for item in value), f"{label} must contain finite positive timings")
    return [float(item) for item in value]


def workspace_bytes(rows: int, context: int) -> int:
    query_blocks = (rows + 127) // 128
    rank_capacity = 1 << (((context + 127) // 128) - 1).bit_length()
    keep_stride = (context + 63) // 64
    slots = 24 * query_blocks
    rank_elements = slots * rank_capacity
    sizes = (
        4 * context * 256 * 2,
        rank_elements * 4,
        rank_elements * 8 * 4,
        slots * keep_stride * 2,
        rank_elements,
        slots * 4,
    )
    total = 0
    for size in sizes:
        total = (total + 255) & ~255
        total += size
    return (total + 255) & ~255


def close(actual: Any, expected: float, label: str) -> None:
    require(number(actual)
            and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=1e-9),
            f"{label} differs from raw samples")


def validate_report(
    report: Any, expected_group: int,
    expected_executable: Path,
    expected_source_root: Path = REPO_ROOT,
    power_reader: Callable[[Path], str] = read_power_profile,
) -> None:
    require(isinstance(report, dict), "report must be an object")
    require(report.get("schema") == "ninfer_r9700_xattention_qualification"
            and report.get("schema_version") == 6, "wrong report schema")
    require(report.get("architecture") == "gfx1201", "wrong architecture")
    require(report.get("disposition") == "qualification-candidate", "wrong disposition")
    require(report.get("physical_qualification") == {
        "status": "passed", "scope": "full-numerical-rejection-before-timing"
    }, "physical qualification is incomplete")
    require(report.get("power_profile") == {
        "path": POWER_PATH, "value": "auto", "required": "auto"
    }, "power profile is not the exact auto contract")
    try:
        live_power = power_reader(Path(POWER_PATH)).strip()
    except OSError as error:
        raise ValueError(f"cannot read live power profile: {error}") from error
    require(live_power == "auto", "live power profile is not auto")
    group = report.get("value_group")
    require(type(expected_group) is int and expected_group in (16, 32)
            and type(group) is int and group == expected_group,
            "value group differs")
    require(report.get("stride") == 16 and report.get("tau_permille") == 900,
            "XAttention profile differs")
    require(report.get("timing_fixture") == "structured-concentrated-e4m3fn-v1"
            and report.get("hot_page_period") == 16
            and report.get("sample_order") == "rotating-interleaved"
            and report.get("stage_samples_are_independent") is True,
            "timing protocol differs")
    require(report.get("rank_stage") ==
            "fp8-k-pack-plus-two-pass-bf16-wmma-plus-b128-select"
            and report.get("consumer_stage") ==
            "b16-query-b16-key-bf16-wmma-online-fp32-softmax-direct-int4-v",
            "stage identity differs")
    oracle = report.get("physical_oracle")
    require(isinstance(oracle, dict) and oracle.get("status") == "passed"
            and oracle.get("absolute_tolerance") == 0.0003
            and oracle.get("relative_tolerance") == 0.0003
            and oracle.get("criterion") == "per-element-absolute-or-relative"
            and number(oracle.get("max_abs_error")) and oracle["max_abs_error"] >= 0,
            "physical oracle differs")
    require(number(oracle.get("max_normalized_error"))
            and 0 <= oracle["max_normalized_error"] <= 1,
            "physical oracle normalized error failed")
    require(math.isclose(oracle["max_normalized_error"],
                         oracle["max_abs_error"] / oracle["absolute_tolerance"],
                         rel_tol=2e-6, abs_tol=1e-12),
            "physical oracle metrics are inconsistent")
    device = report.get("device")
    require(isinstance(device, dict) and isinstance(device.get("name"), str) and device["name"]
            and isinstance(device.get("architecture"), str)
            and device["architecture"].startswith("gfx1201")
            and device.get("wave_size") == 32
            and type(device.get("runtime_version")) is int and device["runtime_version"] > 0
            and type(device.get("driver_version")) is int and device["driver_version"] > 0,
            "device identity is incomplete")
    executable = report.get("executable")
    require(isinstance(executable, dict)
            and Path(executable.get("path", "")).is_absolute(),
            "executable identity is incomplete")
    digest(executable.get("sha256"), "executable.sha256")
    require(isinstance(expected_executable, Path), "expected executable is required")
    require(expected_executable.is_file(), "bound executable is unavailable")
    executable_path = expected_executable.resolve(strict=True)
    require(Path(executable["path"]) == executable_path,
            "reported executable path differs from expected executable")
    require(file_sha256(executable_path) == executable["sha256"],
            "bound executable bytes changed")
    source_rows = report.get("sources")
    require(isinstance(source_rows, list) and len(source_rows) == len(SOURCE_PATHS),
            "source inventory differs")
    reported_root = report.get("source_root")
    require(isinstance(reported_root, str) and Path(reported_root).is_absolute(),
            "source root is incomplete")
    source_root = Path(reported_root)
    require(expected_source_root.is_dir(), "expected source root is unavailable")
    canonical_expected_root = expected_source_root.resolve(strict=True)
    require(source_root.is_dir() and source_root.resolve(strict=True) == source_root
            and source_root == canonical_expected_root,
            "source root is not the expected exact canonical directory")
    observed: set[str] = set()
    for index, source in enumerate(source_rows):
        require(isinstance(source, dict) and source.get("role") in SOURCE_PATHS,
                f"sources[{index}] role differs")
        role = source["role"]
        source_path = Path(source.get("path", ""))
        suffix = Path(SOURCE_PATHS[role])
        require(role not in observed and source_path.is_absolute()
                and source_path.parts[-len(suffix.parts):] == suffix.parts,
                f"sources[{index}] path differs")
        digest(source.get("sha256"), f"sources[{index}].sha256")
        require(source_path == source_root / suffix,
                "sources do not bind the reported exact source root")
        require(source_path.is_file(), f"{role} source is unavailable")
        require(source_path.resolve(strict=True) == source_path,
                f"{role} source path is not canonical")
        require(file_sha256(source_path) == source["sha256"],
                f"{role} source bytes changed")
        observed.add(role)
    require(observed == set(SOURCE_PATHS), "source inventory is incomplete")

    rows = report.get("measurements")
    require(isinstance(rows, list) and len(rows) == 2, "measurement inventory differs")
    observed_contexts: set[int] = set()
    iteration_count: int | None = None
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and row.get("context") in (8192, 32768)
                and row["context"] not in observed_contexts and row.get("query_rows") == 128,
                f"measurements[{index}] geometry differs")
        observed_contexts.add(row["context"])
        iterations = row.get("iterations")
        require(type(iterations) is int and 1 <= iterations <= 20,
                f"measurements[{index}] iterations differ")
        require(type(row.get("workspace_bytes")) is int,
                f"measurements[{index}] workspace type differs")
        iteration_count = iterations if iteration_count is None else iteration_count
        require(iterations == iteration_count, "iteration count differs across rows")
        require(row.get("workspace_bytes") == workspace_bytes(128, row["context"]),
                f"measurements[{index}] workspace differs")
        require(number(row.get("mean_keep_fraction"))
                and 0 < row["mean_keep_fraction"] <= 1,
                f"measurements[{index}] keep fraction differs")
        sparse = samples(row.get("sparse_samples_ms"), iterations, "sparse samples")
        rank = samples(row.get("rank_samples_ms"), iterations, "rank samples")
        consumer = samples(row.get("consumer_samples_ms"), iterations, "consumer samples")
        dense = samples(row.get("dense_samples_ms"), iterations, "dense samples")
        close(row.get("sparse_median_ms"), median(sparse), "sparse median")
        close(row.get("rank_median_ms"), median(rank), "rank median")
        close(row.get("consumer_median_ms"), median(consumer), "consumer median")
        close(row.get("dense_median_ms"), median(dense), "dense median")
        close(row.get("dense_over_sparse"), median(dense) / median(sparse), "speed ratio")
    require(observed_contexts == {8192, 32768}, "measurement contexts are incomplete")

    production = report.get("production_scale")
    require(isinstance(production, dict)
            and production.get("fixture") ==
            "rotating-address-concentrated-e4m3fn-nonzero-int4-v1"
            and production.get("context") == 8192
            and production.get("query_rows") == 4096
            and type(production.get("rotating_replicas")) is int
            and production.get("rotating_replicas") == 4
            and type(production.get("workspace_bytes_per_replica")) is int
            and production.get("workspace_bytes_per_replica") == workspace_bytes(4096, 8192)
            and production.get("value_fixture") ==
            "signed-int4-feature-1-minus2-token-head-varying-fp16-scale"
            and production.get("output_oracle") ==
            "all-elements-fp64-softmax-pv-from-device-keep-list"
            and production.get("oracle_relative_l2_tolerance") == 0.0003
            and production.get("oracle_gross_max_abs_tolerance") == 0.002,
            "production-scale contract differs")
    iterations = production.get("iterations")
    require(type(iterations) is int and iterations == iteration_count,
            "production-scale iterations differ")
    require(number(production.get("oracle_relative_l2"))
            and 0 <= production["oracle_relative_l2"] <= 0.0003
            and number(production.get("oracle_max_abs_error"))
            and 0 <= production["oracle_max_abs_error"] <= 0.002,
            "production-scale oracle failed")
    require(number(production.get("mean_keep_fraction"))
            and 0 < production["mean_keep_fraction"] <= 1,
            "production-scale keep fraction differs")
    sparse = samples(production.get("sparse_samples_ms"), iterations, "production sparse samples")
    rank = samples(production.get("rank_samples_ms"), iterations, "production rank samples")
    consumer = samples(production.get("consumer_samples_ms"), iterations,
                       "production consumer samples")
    close(production.get("sparse_median_ms"), median(sparse), "production sparse median")
    close(production.get("rank_median_ms"), median(rank), "production rank median")
    close(production.get("consumer_median_ms"), median(consumer),
          "production consumer median")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--value-group", type=int, choices=(16, 32), required=True)
    parser.add_argument("--executable", type=Path, required=True,
                        help="expected current or explicitly retained qualification executable")
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT,
                        help="expected source tree (defaults to the current repository)")
    args = parser.parse_args()
    validate_report(json.loads(args.report.read_text(encoding="utf-8")), args.value_group,
                    args.executable, args.source_root)
    print(f"G{args.value_group} XAttention production-scale report passed")


if __name__ == "__main__":
    main()
