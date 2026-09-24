#!/usr/bin/env python3
"""Fail-closed validation for the one reopened retained FP8 M128N256 qualifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "ninfer.r9700.fp8_gate_up_m128n256_retained_ab.v1"
OBJECT_SHA256 = "a853adcd8282ff5410e69069bdafe1f808b1869e76b224b984520569001d318c"
ASSEMBLY_SHA256 = "f84b82472e7d1b944927020abf2550ba1cd2051c2f0e71f3e3d32cf7ddfbcd85"
MATRIX_LIMIT_MS = 3.18103603125
COMPLETE_LIMIT_MS = 3.2809701875
FIXED_OVERHEAD_MS = 0.09993415625
SAVING_LIMIT_MS = 51.438603
OPERATIONS = 730_144_440_320.0
RUNTIME_LIBRARY_PATH = "/opt/rocm/core-10.0/lib"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def finite_positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def median7(value: object, name: str) -> float:
    if not isinstance(value, list) or len(value) != 7 or not all(
        finite_positive(item) for item in value
    ):
        raise ValueError(f"{name} must have exactly seven positive finite samples")
    ordered = sorted(float(item) for item in value)
    return ordered[3]


def check_static(assembly: Path, retained_object: Path) -> dict[str, Any]:
    if sha256(assembly) != ASSEMBLY_SHA256 or sha256(retained_object) != OBJECT_SHA256:
        raise ValueError("retained assembly/object identity mismatch")
    text = assembly.read_text(encoding="utf-8")
    required_once = (
        ".amdgcn_target \"amdgcn-amd-amdhsa--gfx1201\"",
        ".amdhsa_group_segment_fixed_size 24576",
        ".amdhsa_private_segment_fixed_size 0",
        ".amdhsa_wavefront_size32 1",
        ".amdhsa_workgroup_processor_mode 1",
        ".amdhsa_next_free_vgpr 97",
        ".max_flat_workgroup_size: 1024",
        "; Occupancy: 12",
        "; VGPRBlocks: 12",
        "; ScratchSize: 0",
    )
    for token in required_once:
        if text.count(token) != 1:
            raise ValueError(f"retained assembly does not contain exactly one {token!r}")
    symbol = "fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernel"
    match = re.search(
        rf"{symbol}[^:]*:.*?\.Lfunc_end0:", text, flags=re.DOTALL
    )
    if match is None:
        raise ValueError("cannot isolate retained candidate symbol")
    body = match.group(0)
    if body.count("v_wmma_f32_16x16x16_fp8_fp8") != 8:
        raise ValueError("retained candidate must have exactly eight static FP8 WMMA sites")
    if any(opcode in body for opcode in ("v_wmma_i32", "v_wmma_f32_16x16x16_bf16")):
        raise ValueError("retained candidate contains a forbidden matrix opcode")
    return {
        "logical_vgpr": 97,
        "descriptor_allocation_rounded_vgpr": 104,
        "waves_per_simd": 12,
        "waves_per_workgroup": 32,
        "resident_workgroups_per_wgp": 1,
        "static_lds_bytes": 24576,
        "scratch_bytes": 0,
        "fp8_wmma_static_sites": 8,
    }


def validate_ldd(text: str) -> dict[str, str]:
    if "not found" in text:
        raise ValueError("qualification executable has an unresolved shared library")
    required = {
        "hipblaslt": f"libhipblaslt.so.1 => {RUNTIME_LIBRARY_PATH}/libhipblaslt.so.1 ",
        "hip_runtime": f"libamdhip64.so.7 => {RUNTIME_LIBRARY_PATH}/libamdhip64.so.7 ",
    }
    for name, line_prefix in required.items():
        if sum(line_prefix in line for line in text.splitlines()) != 1:
            raise ValueError(f"ldd does not bind exactly one pinned {name} library")
    return {"LD_LIBRARY_PATH": RUNTIME_LIBRARY_PATH,
            "hipblaslt": f"{RUNTIME_LIBRARY_PATH}/libhipblaslt.so.1",
            "hip_runtime": f"{RUNTIME_LIBRARY_PATH}/libamdhip64.so.7"}


def validate_resource_report(report: dict[str, Any], exit_code: int, stderr: str) -> dict[str, int]:
    expected = {"runtime_reported_vgpr": 101, "static_lds_bytes": 24576,
                "private_bytes": 0, "maximum_threads": 1024}
    if (report.get("schema") != "ninfer.r9700.fp8_gate_up.resources.v1" or
            any(report.get(name) != value for name, value in expected.items()) or
            type(report.get("active_blocks_per_wgp")) is not int or
            report["active_blocks_per_wgp"] < 1):
        raise ValueError("runtime resource diagnostic differs from exact runtime-reported identity")
    if exit_code != 0 or stderr:
        raise ValueError("runtime resource diagnostic did not exit cleanly")
    return {**expected, "active_blocks_per_wgp": report["active_blocks_per_wgp"]}


def validate_report(report: dict[str, Any], exit_code: int, stderr: str) -> dict[str, Any]:
    exact_true = (
        "alias_rejection", "no_clobber_rejection", "status_complete_poison",
        "output_canaries_intact", "eager_capture_replay_same_addresses",
    )
    resources = report.get("resources")
    oracle = report.get("represented_fp64_oracle")
    if (
        report.get("schema") != SCHEMA
        or report.get("shape_N_K_T") != [34816, 5120, 2048]
        or report.get("incumbent_solution") != 123104
        or report.get("timing_scope") != (
            "matrix_only_same_represented_fp8_inputs; complete is projected with frozen "
            "retained quantize+poison overhead"
        )
        or not isinstance(resources, dict)
        or {name: resources.get(name) for name in (
            "runtime_reported_vgpr", "static_lds_bytes", "private_bytes", "maximum_threads"
        )} != {"runtime_reported_vgpr": 101, "static_lds_bytes": 24576,
               "private_bytes": 0, "maximum_threads": 1024}
        or type(resources.get("active_blocks_per_wgp")) is not int
        or resources["active_blocks_per_wgp"] < 1
        or not isinstance(oracle, dict)
        or oracle.get("probe_count") != 2192
        or any(type(oracle.get(name)) is not int or not 0 <= oracle[name] <= 1 for name in (
            "eager_maximum_bf16_steps", "graph_maximum_bf16_steps",
            "incumbent_maximum_bf16_steps"))
        or any(report.get(name) is not True for name in exact_true)
    ):
        raise ValueError("report does not describe the exact retained qualification")
    forward_incumbent = median7(
        report.get("forward_incumbent_matrix_ms_samples"), "forward incumbent")
    forward_challenger = median7(
        report.get("forward_challenger_matrix_ms_samples"), "forward challenger")
    reverse_challenger = median7(
        report.get("reverse_challenger_matrix_ms_samples"), "reverse challenger")
    reverse_incumbent = median7(
        report.get("reverse_incumbent_matrix_ms_samples"), "reverse incumbent")
    incumbent = min(forward_incumbent, reverse_incumbent)
    challenger = max(forward_challenger, reverse_challenger)
    projected_complete = challenger + FIXED_OVERHEAD_MS
    saving = 64.0 * min(forward_incumbent - forward_challenger,
                        reverse_incumbent - reverse_challenger)
    incumbent_order_delta = abs(forward_incumbent - reverse_incumbent) / incumbent
    challenger_order_delta = abs(forward_challenger - reverse_challenger) / min(
        forward_challenger, reverse_challenger)
    order_stable = incumbent_order_delta <= 0.05 and challenger_order_delta <= 0.05
    passed = (challenger <= MATRIX_LIMIT_MS and
              projected_complete <= COMPLETE_LIMIT_MS and
              saving >= SAVING_LIMIT_MS and order_stable)
    expected = {
        "forward_incumbent_matrix_median_ms": forward_incumbent,
        "forward_challenger_matrix_median_ms": forward_challenger,
        "reverse_challenger_matrix_median_ms": reverse_challenger,
        "reverse_incumbent_matrix_median_ms": reverse_incumbent,
        "incumbent_matrix_median_ms": incumbent,
        "challenger_matrix_median_ms": challenger,
        "incumbent_order_relative_delta": incumbent_order_delta,
        "challenger_order_relative_delta": challenger_order_delta,
        "maximum_order_relative_delta": 0.05,
        "fixed_complete_overhead_ms": FIXED_OVERHEAD_MS,
        "projected_challenger_complete_ms": projected_complete,
        "projected_64_call_saving_ms": saving,
        "maximum_challenger_ms": MATRIX_LIMIT_MS,
        "maximum_complete_ms": COMPLETE_LIMIT_MS,
        "minimum_64_call_saving_ms": SAVING_LIMIT_MS,
    }
    for name, value in expected.items():
        reported = report.get(name)
        if (type(reported) not in (int, float) or not math.isfinite(reported) or
                (reported < 0 and name != "projected_64_call_saving_ms") or not math.isclose(
                    float(reported), value, rel_tol=2e-8, abs_tol=1e-8)):
            raise ValueError(f"{name} differs from samples/frozen gate")
    if report.get("order_stable") is not order_stable:
        raise ValueError("reported launch-order verdict differs from samples")
    if report.get("pass") is not passed:
        raise ValueError("reported verdict differs from frozen gate")
    failure = "challenger failed the exact direct performance gate"
    if passed:
        if exit_code != 0 or failure in stderr:
            raise ValueError("passing report must have a successful process")
    elif exit_code != 1 or failure not in stderr:
        raise ValueError("rejected report must have the exact expected process failure")
    return {
        "verdict": "pass" if passed else "reject",
        "incumbent_matrix_median_ms": incumbent,
        "challenger_matrix_median_ms": challenger,
        "projected_challenger_complete_ms": projected_complete,
        "projected_64_call_saving_ms": saving,
        "challenger_useful_tflops": OPERATIONS / (challenger * 1.0e9),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resource-report", type=Path)
    parser.add_argument("--resource-stderr", type=Path)
    parser.add_argument("--resource-exit-code", type=int)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--ldd", type=Path)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    static = check_static(args.assembly.resolve(strict=True), args.object.resolve(strict=True))
    resource = None
    if args.resource_report is not None:
        if args.resource_stderr is None or args.resource_exit_code is None:
            raise ValueError("resource validation requires stderr and exit-code")
        resource_report = json.loads(
            args.resource_report.read_text(encoding="utf-8"),
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        resource = validate_resource_report(
            resource_report, args.resource_exit_code,
            args.resource_stderr.read_text(encoding="utf-8"))
    if args.report is None:
        print(json.dumps({"static": static, "runtime_resources": resource}, indent=2,
                         sort_keys=True, allow_nan=False))
        return 0
    if (resource is None or args.stderr is None or args.ldd is None or
            args.exit_code is None or args.output is None):
        raise ValueError("report validation requires resource report, stderr, ldd, exit-code, and output")
    report = json.loads(args.report.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    timing = validate_report(report, args.exit_code, args.stderr.read_text(encoding="utf-8"))
    runtime = validate_ldd(args.ldd.read_text(encoding="utf-8"))
    evidence = {
        "schema": "ninfer.r9700.fp8_gate_up_m128n256_retained_decision.v1",
        "retained_object_sha256": OBJECT_SHA256,
        "retained_assembly_sha256": ASSEMBLY_SHA256,
        "static": static,
        "runtime": runtime,
        "runtime_resources": resource,
        "timing": timing,
        "limitations": [
            "Direct exact-shape qualification only; no production routing changed.",
            "Complete time is matrix timing plus frozen fresh-trace quantize/poison overhead.",
            "A direct pass permits one matched whole-P2048 confirmation; it is not production admission.",
        ],
    }
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
