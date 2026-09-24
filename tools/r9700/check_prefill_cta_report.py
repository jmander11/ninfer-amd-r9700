#!/usr/bin/env python3
"""Strictly validate retained A8Q4/A8W8 cooperative-CTA admission evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


POWER_PATH = "/sys/class/drm/card2/device/power_dpm_force_performance_level"
REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENS = {1024, 2048, 4096, 8192}
DEVICE_NAME = "AMD Radeon AI PRO R9700"
CONFIG = {
    "q4": {
        "schema": "ninfer_r9700_a8q4g64_prefill_cta_qualification",
        "shapes": {(7168, 5120), (4096, 5120), (12288, 5120), (5120, 6144),
                   (34816, 5120), (5120, 17408), (5120, 10240), (1024, 5120)},
        "trials": 7,
        "qualifier": "tools/r9700/a8q4_shape_sweep_qual.hip",
        "candidate": {"tile_tokens": 64, "tile_rows": 64, "waves": 16,
                      "k_stage": 64, "activation_reuse": 4, "weight_reuse": 4,
                      "fragment_major_lds_swizzle": True, "production_dispatch": False},
        "resources": {"registers": 62, "static_shared_bytes": 6400, "local_bytes": 0},
        "scope": {"text": [[7168, 5120], [4096, 5120], [12288, 5120],
                            [5120, 6144], [34816, 5120], [5120, 17408]],
                  "mtp_bulk": [[5120, 10240], [1024, 5120]],
                  "tokens": [1024, 2048, 4096, 8192]},
    },
    "w8": {
        "schema": "ninfer_r9700_a8w8g32_prefill_cta_qualification",
        "shapes": {(7168, 5120), (12288, 5120), (5120, 17408), (5120, 6144)},
        "trials": 5,
        "qualifier": "tools/r9700/w8a8_wmma_linear_qual.hip",
        "candidate": {"tile_tokens": 64, "tile_rows": 64, "waves": 16,
                      "group_k": 32, "activation_reuse": 4, "weight_reuse": 4,
                      "fragment_major_lds_swizzle": True, "production_dispatch": False},
        "resources": {"registers": 50, "static_shared_bytes": 4352, "local_bytes": 0},
        "scope": {"mixed_text": [[7168, 5120], [12288, 5120], [5120, 6144],
                                  [5120, 17408]], "tokens": [1024, 2048, 4096, 8192]},
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any, label: str) -> None:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"{label} must be a lowercase SHA-256")


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


def exact_typed(value: Any, expected: Any) -> bool:
    """Compare retained protocol values without Python's bool/int equivalence."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            exact_typed(value[key], wanted) for key, wanted in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            exact_typed(item, wanted) for item, wanted in zip(value, expected, strict=True)
        )
    return value == expected


def validate_report(
    report: Any, recipe: str, expected_executable: Path,
    expected_source_root: Path = REPO_ROOT,
    power_reader: Callable[[Path], str] = read_power_profile,
) -> None:
    config = CONFIG[recipe]
    require(isinstance(report, dict), "report must be an object")
    require(report.get("schema") == config["schema"] and
            type(report.get("schema_version")) is int and report["schema_version"] == 3,
            "wrong report schema")
    require(report.get("disposition") == "unpromoted-qualification-candidate",
            "candidate must remain unpromoted")
    require(exact_typed(report.get("candidate"), config["candidate"]),
            "candidate geometry differs")
    require(exact_typed(report.get("resources"), config["resources"]),
            "kernel resources differ")
    require(exact_typed(report.get("qualified_tuple_scope"), config["scope"]),
            "qualified tuple scope differs")
    require(report.get("incumbent_scope") == "post-quantization-a8-linear-leaf" and
            report.get("challenger_scope") == "post-quantization-a8-linear-leaf",
            "timed leaf scopes differ")
    require(exact_typed(report.get("physical_qualification"), {
        "status": "passed", "scope": "full-numerical-rejection-before-timing"},
        ), "qualification phase is incomplete")
    require(exact_typed(report.get("power_profile"), {
        "path": POWER_PATH, "value": "auto", "required": "auto"},
        ), "power profile is not the exact auto contract")
    try:
        live_power = power_reader(Path(POWER_PATH)).strip()
    except OSError as error:
        raise ValueError(f"cannot read live power profile: {error}") from error
    require(live_power == "auto", "live power profile is not auto")
    hardware = report.get("hardware")
    require(isinstance(hardware, dict) and hardware.get("architecture") == "gfx1201" and
            type(hardware.get("wave_size")) is int and hardware["wave_size"] == 32 and
            hardware.get("device") == DEVICE_NAME and
            type(hardware.get("vram_bytes")) is int and hardware["vram_bytes"] > 0 and
            type(hardware.get("hip_runtime_version")) is int and
            hardware["hip_runtime_version"] > 0 and
            type(hardware.get("hip_driver_version")) is int and
            hardware["hip_driver_version"] > 0, "hardware identity is incomplete")
    toolchain = report.get("toolchain")
    require(isinstance(toolchain, dict) and isinstance(toolchain.get("compiler"), str) and
            bool(toolchain["compiler"]) and toolchain.get("offload_architecture") == "gfx1201",
            "toolchain identity is incomplete")
    executable = report.get("executable")
    require(isinstance(executable, dict) and Path(executable.get("path", "")).is_absolute(),
            "executable path must be absolute")
    digest(executable.get("sha256"), "executable.sha256")
    require(isinstance(expected_executable, Path), "expected executable is required")
    require(expected_executable.is_file(), "bound executable is unavailable")
    executable_path = expected_executable.resolve(strict=True)
    require(Path(executable["path"]) == executable_path,
            "reported executable path differs from expected executable")
    require(file_sha256(executable_path) == executable["sha256"],
            "bound executable bytes changed")
    sources = report.get("sources")
    require(isinstance(sources, list) and len(sources) == 4, "source inventory differs")
    expected = {
        "qualifier": config["qualifier"],
        "contract_header": "src/ops/r9700/linear/r9700_linear.h",
        "kernel": "src/ops/r9700/linear/r9700_linear.hip",
        "independent_oracle": config["qualifier"],
    }
    reported_root = report.get("source_root")
    require(isinstance(reported_root, str) and Path(reported_root).is_absolute(),
            "source root is incomplete")
    source_root = Path(reported_root)
    require(expected_source_root.is_dir(), "expected source root is unavailable")
    canonical_expected_root = expected_source_root.resolve(strict=True)
    require(source_root.is_dir() and source_root.resolve(strict=True) == source_root and
            source_root == canonical_expected_root,
            "source root is not the expected exact canonical directory")
    observed: set[str] = set()
    for source in sources:
        require(isinstance(source, dict) and source.get("role") in expected,
                "source role differs")
        role = source["role"]
        source_path = Path(source.get("path", ""))
        require(role not in observed and source_path.is_absolute() and
                source_path == source_root / expected[role], f"{role} source path differs")
        digest(source.get("sha256"), f"{role}.sha256")
        require(source_path.is_file(), f"{role} source is unavailable")
        require(source_path.resolve(strict=True) == source_path,
                f"{role} source path is not canonical")
        require(file_sha256(source_path) == source["sha256"],
                f"{role} source bytes changed")
        observed.add(role)
    require(observed == set(expected), "source roles are incomplete")
    timing = report.get("timing")
    require(exact_typed(timing, {
                "sample_order": "alternating-interleaved",
                "warmup_iterations_per_route": 1,
                "trials": config["trials"],
                "method": "unprofiled HIP events, median",
                "token_extents": [1024, 2048, 4096, 8192],
            }),
            "timing protocol differs")
    edge = report.get("edge_checks")
    require(isinstance(edge, dict) and edge.get("nonzero_status_all_output_nan") is True and
            edge.get("partial_group_padded_tail_poison_ignored") is True and
            edge.get("packed_code_alignment_bytes") == 4 and
            edge.get("host_owned_stored_weight_bytes") is True and
            edge.get("oracle_decodes_uploaded_bytes") is True and
            type(edge.get("tail_maximum_bf16_steps")) is int and
            0 <= edge["tail_maximum_bf16_steps"] <= 2, "tail numerical evidence differs")
    require(exact_typed(report.get("oracle"), {
                "kind": "independent FP64 complete quantized formula",
                "small_case": "all outputs",
                "real_shapes": "8 rows x 8 tokens",
                "maximum_bf16_steps_allowed": 2,
            }),
            "oracle descriptor differs")
    expected_cases = {(n, k, t) for n, k in config["shapes"] for t in TOKENS}
    cases = report.get("cases")
    require(isinstance(cases, list) and len(cases) == len(expected_cases), "case inventory differs")
    observed_cases: set[tuple[int, int, int]] = set()
    for index, row in enumerate(cases):
        require(isinstance(row, dict), f"cases[{index}] must be an object")
        if recipe == "q4":
            shape = (row.get("rows"), row.get("columns"), row.get("tokens"))
            old_key, new_key = "one_wave_ms", "cooperative_cta_ms"
        else:
            pair = row.get("shape")
            require(isinstance(pair, list) and len(pair) == 2, f"cases[{index}].shape differs")
            shape = (pair[0], pair[1], row.get("tokens"))
            old_key, new_key = "one_wave_median_ms", "cooperative_cta_median_ms"
        require(shape in expected_cases and shape not in observed_cases, f"cases[{index}] tuple differs")
        require(row.get("exhaustive_oracle") is False, "retained real case must use sampled oracle")
        if recipe == "w8":
            expected_roles = {(7168, 5120): "text_attention_gate_value",
                              (12288, 5120): "text_gdn_value_z",
                              (5120, 17408): "text_mlp_down",
                              (5120, 6144): "text_attention_gdn_output"}
            require(row.get("role") == expected_roles[shape[:2]], "W8 shape role differs")
        require(type(row.get("maximum_bf16_steps")) is int and
                0 <= row["maximum_bf16_steps"] <= 2, "real-shape numerical evidence differs")
        observed_cases.add(shape)
        old_samples = row.get("one_wave_samples_ms")
        new_samples = row.get("cooperative_cta_samples_ms")
        require(isinstance(old_samples, list) and isinstance(new_samples, list) and
                len(old_samples) == len(new_samples) == config["trials"] and
                all(number(x) and x > 0
                    for x in old_samples + new_samples), "raw timing samples differ")
        old_median = sorted(old_samples)[len(old_samples) // 2]
        new_median = sorted(new_samples)[len(new_samples) // 2]
        require(number(row.get(old_key)) and number(row.get(new_key)) and
                math.isclose(row[old_key], old_median, rel_tol=2e-6, abs_tol=1e-9) and
                math.isclose(row[new_key], new_median, rel_tol=2e-6, abs_tol=1e-9),
                "reported median differs from raw samples")
        require(new_median < old_median and number(row.get("cooperative_over_one_wave")) and
                math.isclose(row["cooperative_over_one_wave"], new_median / old_median,
            rel_tol=2e-6, abs_tol=1e-9), "candidate timing admission differs")
    require(observed_cases == expected_cases, "case Cartesian set is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=sorted(CONFIG))
    parser.add_argument("report", type=Path)
    parser.add_argument("--executable", type=Path, required=True,
                        help="exact qualification executable recorded by the report")
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT,
                        help="authoritative source tree (defaults to the current repository)")
    args = parser.parse_args()
    validate_report(json.loads(args.report.read_text(encoding="utf-8")), args.recipe,
                    args.executable, args.source_root)
    print(f"{args.recipe} prefill CTA report passed")


if __name__ == "__main__":
    main()
