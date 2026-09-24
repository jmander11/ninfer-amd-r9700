#!/usr/bin/env python3
"""Validate one immutable incumbent-vs-LDS-scope CTA physical A/B report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
POWER_PATH = "/sys/class/drm/card2/device/power_dpm_force_performance_level"
TOKENS = (1024, 2048, 4096, 8192)
MAXIMUM_PER_TUPLE_RATIO = 1.01
CONFIG = {
    "q4": {
        "artifact_type": "ninfer_r9700_prefill_cta_lds_scope_ab",
        "shapes": ((7168, 5120), (4096, 5120), (12288, 5120),
                   (5120, 6144), (34816, 5120), (5120, 17408),
                   (5120, 10240), (1024, 5120)),
        "weights": {(7168, 5120): 32, (4096, 5120): 48,
                    (12288, 5120): 48, (5120, 6144): 65,
                    (34816, 5120): 65, (5120, 17408): 65,
                    (5120, 10240): 1, (1024, 5120): 0},
        "qualifier": "tools/r9700/a8q4_shape_sweep_qual.hip",
        "profile": "src/ops/r9700/linear/r9700_q4_activation_profile.h",
        "incumbent_lds": 6400,
        "challenger_lds": 6400,
    },
    "w8": {
        "artifact_type": "ninfer_r9700_prefill_cta_lds_scope_ab",
        "shapes": ((7168, 5120), (12288, 5120),
                   (5120, 17408), (5120, 6144)),
        "weights": {(7168, 5120): 16, (12288, 5120): 48,
                    (5120, 17408): 65, (5120, 6144): 65},
        "qualifier": "tools/r9700/w8a8_wmma_linear_qual.hip",
        "profile": "src/ops/r9700/linear/r9700_w8_activation_profile.h",
        "incumbent_lds": 4352,
        "challenger_lds": 4352,
    },
    "q4-n128": {
        "artifact_type": "ninfer_r9700_prefill_cta_n128_ab",
        "shapes": ((7168, 5120), (4096, 5120), (12288, 5120),
                   (5120, 6144), (34816, 5120), (5120, 17408),
                   (5120, 10240), (1024, 5120)),
        "weights": {(7168, 5120): 32, (4096, 5120): 48,
                    (12288, 5120): 48, (5120, 6144): 65,
                    (34816, 5120): 65, (5120, 17408): 65,
                    (5120, 10240): 1, (1024, 5120): 0},
        "qualifier": "tools/r9700/a8q4_shape_sweep_qual.hip",
        "profile": "src/ops/r9700/linear/r9700_q4_activation_profile.h",
        "incumbent_lds": 6400,
        "challenger_lds": 8576,
    },
    "q4-m128n128": {
        "artifact_type": "ninfer_r9700_prefill_cta_m128n128_ab",
        "shapes": ((7168, 5120), (4096, 5120), (12288, 5120),
                   (5120, 6144), (34816, 5120), (5120, 17408),
                   (5120, 10240), (1024, 5120)),
        "weights": {(7168, 5120): 32, (4096, 5120): 48,
                    (12288, 5120): 48, (5120, 6144): 65,
                    (34816, 5120): 65, (5120, 17408): 65,
                    (5120, 10240): 1, (1024, 5120): 0},
        "qualifier": "tools/r9700/a8q4_shape_sweep_qual.hip",
        "profile": "src/ops/r9700/linear/r9700_q4_activation_profile.h",
        "incumbent_lds": 8576,
        "challenger_lds": 12800,
    },
    "q4-pingpong": {
        "artifact_type": "ninfer_r9700_prefill_cta_pingpong_ab",
        "shapes": ((7168, 5120), (4096, 5120), (12288, 5120),
                   (5120, 6144), (34816, 5120), (5120, 17408),
                   (5120, 10240), (1024, 5120)),
        "weights": {(7168, 5120): 32, (4096, 5120): 48,
                    (12288, 5120): 48, (5120, 6144): 65,
                    (34816, 5120): 65, (5120, 17408): 65,
                    (5120, 10240): 1, (1024, 5120): 0},
        "qualifier": "tools/r9700/a8q4_shape_sweep_qual.hip",
        "profile": "src/ops/r9700/linear/r9700_q4_activation_profile.h",
        "incumbent_lds": 8576,
        "challenger_lds": 17152,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def exact_typed(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            exact_typed(value[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            exact_typed(item, wanted) for item, wanted in zip(value, expected, strict=True))
    return value == expected


def digest(value: Any, label: str) -> None:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value),
            f"{label} must be a lowercase SHA-256")


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_power(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _samples(row: dict[str, Any], route: str) -> tuple[list[float], float]:
    forward = row.get(f"{route}_forward_ms")
    reverse = row.get(f"{route}_reverse_ms")
    balanced = row.get(f"{route}_balanced_ms")
    require(all(isinstance(values, list) and len(values) == 7
                for values in (forward, reverse, balanced)),
            f"{route} raw sample inventory differs")
    require(all(number(value) and value > 0
                for values in (forward, reverse, balanced) for value in values),
            f"{route} samples must be finite and positive")
    for index in range(7):
        require(math.isclose(balanced[index], (forward[index] + reverse[index]) / 2.0,
                             rel_tol=2e-6, abs_tol=1e-9),
                f"{route} balanced sample is not derived from forward/reverse")
    median = statistics.median(balanced)
    require(number(row.get(f"{route}_median_ms")) and
            math.isclose(row[f"{route}_median_ms"], median,
                         rel_tol=2e-6, abs_tol=1e-9),
            f"{route} median is not derived from balanced samples")
    return balanced, median


def validate_report(report: Any, recipe: str, executable: Path,
                    source_root: Path = REPO_ROOT,
                    power_reader: Callable[[Path], str] = read_power) -> None:
    config = CONFIG[recipe]
    require(isinstance(report, dict), "report must be an object")
    require(report.get("artifact_type") == config["artifact_type"] and
            type(report.get("schema_version")) is int and report["schema_version"] == 1,
            "wrong report schema")
    status = report.get("status")
    terminal_recipe = recipe in ("q4-m128n128", "q4-pingpong")
    require(status in ({"passed", "rejected"} if terminal_recipe else {"passed"}) and
            report.get("recipe") == recipe and
            report.get("disposition") ==
                ("qualification_only_unpromoted" if status == "passed"
                 else "qualification_rejected_production_unchanged"),
            "report result or recipe differs")
    if terminal_recipe:
        require(exact_typed(report.get("production_state"), {
            "dispatch_changed": False, "selected_route": "M64xN128-pingpong"}),
            "production state differs")
    require(exact_typed(report.get("timing"), {
        "method": "unprofiled HIP events",
        "warmup_iterations_per_route": 1,
        "balanced_forward_reverse_pairs": 7,
        "balanced_sample": "mean of same-route forward and reverse measurements",
        "token_extents": list(TOKENS),
    }), "timing protocol differs")
    expected_numerical = {
        "completed_before_timing": True,
        "incumbent_challenger_bit_exact": True,
        "active_output_coverage": "full rewrite and bit parity after non-incumbent poison",
        "independent_oracle": "FP64 represented quantized formula at sampled coordinates",
        "maximum_bf16_steps_allowed": 2,
        "tail_status_and_alignment_gate": "passed",
    }
    require(exact_typed(report.get("numerical_qualification"), expected_numerical),
            "numerical qualification differs")
    require(exact_typed(report.get("power_profile"), {
        "path": POWER_PATH, "required": "auto", "before": "auto", "after": "auto"},
        ), "power evidence differs")
    require(power_reader(Path(POWER_PATH)).strip() == "auto", "live power profile is not auto")
    hardware = report.get("hardware")
    require(isinstance(hardware, dict) and hardware.get("device") == "AMD Radeon AI PRO R9700" and
            hardware.get("architecture") == "gfx1201" and
            type(hardware.get("wave_size")) is int and hardware["wave_size"] == 32 and
            type(hardware.get("vram_bytes")) is int and hardware["vram_bytes"] > 0 and
            type(hardware.get("hip_runtime_version")) is int and
            hardware["hip_runtime_version"] > 0 and
            type(hardware.get("hip_driver_version")) is int and
            hardware["hip_driver_version"] > 0,
            "hardware identity differs")
    resources = report.get("resources")
    require(isinstance(resources, dict) and
            type(resources.get("incumbent_registers")) is int and
            0 < resources["incumbent_registers"] <= 96 and
            type(resources.get("challenger_registers")) is int and
            0 < resources["challenger_registers"] <= 96 and
            type(resources.get("incumbent_static_shared_bytes")) is int and
            resources["incumbent_static_shared_bytes"] == config["incumbent_lds"] and
            type(resources.get("challenger_static_shared_bytes")) is int and
            resources["challenger_static_shared_bytes"] == config["challenger_lds"] and
            type(resources.get("incumbent_local_bytes")) is int and
            resources["incumbent_local_bytes"] == 0 and
            type(resources.get("challenger_local_bytes")) is int and
            resources["challenger_local_bytes"] == 0 and
            (recipe not in ("q4-m128n128", "q4-pingpong") or
             (type(resources.get("challenger_max_threads_per_block")) is int and
              resources["challenger_max_threads_per_block"] ==
                  (1024 if recipe == "q4-m128n128" else 512))),
            "resource identity differs")
    toolchain = report.get("toolchain")
    require(isinstance(toolchain, dict) and isinstance(toolchain.get("compiler"), str) and
            bool(toolchain["compiler"]) and toolchain.get("offload_architecture") == "gfx1201",
            "toolchain identity differs")
    expected_executable = executable.resolve(strict=True)
    identity = report.get("executable")
    require(isinstance(identity, dict) and Path(identity.get("path", "")) == expected_executable,
            "executable path differs")
    digest(identity.get("sha256"), "executable.sha256")
    require(file_sha256(expected_executable) == identity["sha256"], "executable bytes changed")
    expected_root = source_root.resolve(strict=True)
    require(Path(report.get("source_root", "")) == expected_root, "source root differs")
    expected_sources = {
        "qualifier": config["qualifier"],
        "contract_header": "src/ops/r9700/linear/r9700_linear.h",
        "kernel": "src/ops/r9700/linear/r9700_linear.hip",
        "dispatch_profile": config["profile"],
    }
    sources = report.get("sources")
    require(isinstance(sources, list) and len(sources) == len(expected_sources),
            "source inventory differs")
    seen: set[str] = set()
    for source in sources:
        require(isinstance(source, dict) and source.get("role") in expected_sources and
                source["role"] not in seen, "source role differs")
        role = source["role"]
        path = expected_root / expected_sources[role]
        require(Path(source.get("path", "")) == path and path.is_file() and
                path.resolve(strict=True) == path, f"{role} path differs")
        digest(source.get("sha256"), f"{role}.sha256")
        require(file_sha256(path) == source["sha256"], f"{role} bytes changed")
        seen.add(role)
    require(seen == set(expected_sources), "source roles differ")

    expected_cases = {(rows, columns, tokens) for rows, columns in config["shapes"]
                      for tokens in TOKENS}
    cases = report.get("cases")
    require(isinstance(cases, list) and len(cases) == len(expected_cases),
            "case Cartesian inventory differs")
    observed: set[tuple[int, int, int]] = set()
    medians: dict[tuple[int, int, int], tuple[float, float]] = {}
    failing: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        require(isinstance(row, dict), f"cases[{index}] must be an object")
        require(type(row.get("rows")) is int and type(row.get("columns")) is int and
                type(row.get("tokens")) is int,
                f"cases[{index}] dimensions must be exact integers")
        key = (row.get("rows"), row.get("columns"), row.get("tokens"))
        require(key in expected_cases and key not in observed, f"cases[{index}] tuple differs")
        require(row.get("bit_exact") is True and
                type(row.get("maximum_bf16_steps")) is int and
                0 <= row["maximum_bf16_steps"] <= 2,
                f"cases[{index}] numerical result differs")
        _, incumbent = _samples(row, "incumbent")
        _, challenger = _samples(row, "challenger")
        ratio = challenger / incumbent
        require(number(row.get("challenger_over_incumbent")) and
                math.isclose(row["challenger_over_incumbent"], ratio,
                             rel_tol=2e-6, abs_tol=1e-9),
                f"cases[{index}] ratio is not derived")
        maximum_ratio = (1.0 if recipe == "q4-pingpong"
                         else MAXIMUM_PER_TUPLE_RATIO)
        if ratio > maximum_ratio:
            failing.append({"rows": key[0], "columns": key[1], "tokens": key[2],
                            "challenger_over_incumbent": row["challenger_over_incumbent"],
                            "reason": ("challenger_over_incumbent_exceeds_1.0"
                                       if recipe == "q4-pingpong"
                                       else "challenger_over_incumbent_exceeds_1.01")})
        observed.add(key)
        medians[key] = (incumbent, challenger)
    require(observed == expected_cases, "case Cartesian inventory is incomplete")
    if terminal_recipe:
        require(exact_typed(report.get("failing_tuples"), failing),
                "failing tuple inventory is not derived")
    else:
        require(not failing, "material challenger tuple regression")

    weights = [{"rows": rows, "columns": columns, "count": count}
               for (rows, columns), count in config["weights"].items()]
    incumbent_weighted = sum(config["weights"][(rows, columns)] * medians[(rows, columns, 2048)][0]
                             for rows, columns in config["shapes"])
    challenger_weighted = sum(config["weights"][(rows, columns)] * medians[(rows, columns, 2048)][1]
                              for rows, columns in config["shapes"])
    decision = report.get("decision")
    required_speedup = 1.5 if recipe == "q4-pingpong" else 1.0
    terminal_accepted = (not failing and
                         challenger_weighted * required_speedup <= incumbent_weighted and
                         incumbent_weighted - challenger_weighted >=
                         (150 if recipe == "q4-m128n128" else 0))
    require(isinstance(decision, dict) and type(decision.get("token_extent")) is int and
            decision["token_extent"] == 2048 and
            exact_typed(decision.get("weights"), weights) and
            decision.get("maximum_per_tuple_ratio") ==
                (1.0 if recipe == "q4-pingpong"
                 else MAXIMUM_PER_TUPLE_RATIO) and
            decision.get("no_material_per_tuple_regression") is (not failing) and
            decision.get("weighted_challenger_faster") is
                (challenger_weighted < incumbent_weighted) and
            (recipe != "q4-pingpong" or
             decision.get("required_speedup_met") is
                (challenger_weighted * required_speedup <= incumbent_weighted)) and
            (not terminal_recipe or decision.get("accepted") is terminal_accepted),
            "decision contract differs")
    expected_saving = 150 if recipe == "q4-m128n128" else 0
    if recipe == "q4-m128n128":
        require(type(decision.get("minimum_weighted_saving_ms")) is int and
                decision["minimum_weighted_saving_ms"] == expected_saving and
                number(decision.get("weighted_saving_ms")),
                "weighted saving contract differs")
    if recipe == "q4-pingpong":
        require(decision.get("minimum_weighted_speedup") == required_speedup,
                "weighted speedup contract differs")
    require(number(decision.get("weighted_incumbent_ms")) and
            number(decision.get("weighted_challenger_ms")) and
            number(decision.get("challenger_over_incumbent")) and
            math.isclose(decision["weighted_incumbent_ms"], incumbent_weighted,
                         rel_tol=2e-6, abs_tol=1e-9) and
            math.isclose(decision["weighted_challenger_ms"], challenger_weighted,
                         rel_tol=2e-6, abs_tol=1e-9) and
            math.isclose(decision["challenger_over_incumbent"],
                         challenger_weighted / incumbent_weighted,
                         rel_tol=2e-6, abs_tol=1e-9) and
            (not terminal_recipe or
             math.isclose(decision["weighted_saving_ms"],
                          incumbent_weighted - challenger_weighted,
                          rel_tol=2e-6, abs_tol=1e-9)) and
            ((status == "passed" and terminal_accepted) or
             (status == "rejected" and not terminal_accepted and
              (bool(failing) or challenger_weighted * required_speedup > incumbent_weighted or
               incumbent_weighted - challenger_weighted < expected_saving))),
            "weighted T2048 decision or terminal status is not derived")
    # Final TOCTOU boundary after all report reads and derivations.
    require(file_sha256(expected_executable) == identity["sha256"], "executable changed during validation")
    for source in sources:
        require(file_sha256(Path(source["path"])) == source["sha256"],
                f"{source['role']} changed during validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=sorted(CONFIG))
    parser.add_argument("report", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    before = file_sha256(args.report)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, args.recipe, args.executable, args.source_root)
    require(file_sha256(args.report) == before, "report changed during validation")
    print(f"{args.recipe} CTA physical A/B report validated: status="
          f"{report['status']}")


if __name__ == "__main__":
    main()
