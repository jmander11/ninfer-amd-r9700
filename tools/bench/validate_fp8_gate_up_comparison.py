#!/usr/bin/env python3
"""Validate the matched fixed-shape FP8 versus production-A8Q4 qualifier report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "ninfer.r9700.fp8_gate_up_qualification.v1"
SHAPE = {"tokens": 2048, "rows": 34816, "columns": 5120}
OPERATIONS = 2.0 * SHAPE["tokens"] * SHAPE["rows"] * SHAPE["columns"]


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _samples(report: dict[str, Any], name: str) -> list[float]:
    values = report.get(name)
    if not (
        isinstance(values, list)
        and len(values) == 14
        and all(_positive(value) for value in values)
    ):
        raise ValueError(f"{name} must contain fourteen positive finite trials")
    return [float(value) for value in values]


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate(report: dict[str, Any], *, verify_files: bool = False) -> dict[str, Any]:
    required_true = (
        "direct_weight_binding",
        "outer_vector_scales",
        "nonfinite_status_poisoning",
        "q4_nonfinite_status_poisoning",
        "no_clobber_rejection",
        "pass",
    )
    hardware = report.get("hardware") if isinstance(report, dict) else None
    power = report.get("power_profile") if isinstance(report, dict) else None
    toolchain = report.get("toolchain") if isinstance(report, dict) else None
    algorithm = report.get("algorithm") if isinstance(report, dict) else None
    provenance = report.get("provenance") if isinstance(report, dict) else None
    oracle = report.get("represented_format_axis_oracle") if isinstance(report, dict) else None
    pci = hardware.get("pci_bus_id") if isinstance(hardware, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("artifact_type") != "ninfer_r9700_fp8_gate_up_comparison"
        or report.get("schema_version") != 2
        or not isinstance(provenance, dict)
        or not isinstance(provenance.get("source_root"), str)
        or not Path(provenance["source_root"]).is_absolute()
        or not isinstance(provenance.get("source_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance["source_sha256"]) is None
        or not isinstance(provenance.get("executable_path"), str)
        or not Path(provenance["executable_path"]).is_absolute()
        or not isinstance(provenance.get("executable_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance["executable_sha256"]) is None
        or not isinstance(hardware, dict)
        or hardware.get("ordinal") != 0
        or hardware.get("device") != "AMD Radeon AI PRO R9700"
        or not isinstance(pci, str)
        or re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci) is None
        or hardware.get("vendor_id") != "0x1002"
        or hardware.get("device_id") != "0x7551"
        or hardware.get("architecture") != "gfx1201"
        or hardware.get("wave_size") != 32
        or type(hardware.get("compute_units")) is not int
        or hardware["compute_units"] <= 0
        or type(hardware.get("l2_bytes")) is not int
        or hardware["l2_bytes"] <= 0
        or type(hardware.get("vram_bytes")) is not int
        or hardware["vram_bytes"] <= 0
        or not isinstance(power, dict)
        or power.get("path") != f"/sys/bus/pci/devices/{pci}/power_dpm_force_performance_level"
        or power.get("required") != "auto"
        or power.get("before") != "auto"
        or power.get("after") != "auto"
        or not isinstance(toolchain, dict)
        or any(type(toolchain.get(name)) is not int or toolchain[name] <= 0 for name in (
            "hip_runtime_version", "hip_driver_version", "hipblaslt_version"
        ))
        or report.get("shape") != SHAPE
        or report.get("fp8_profile") != "E4M3-outer-vec32f-hipBLASLt-top-supported"
        or report.get("q4_control") != "A8Q4G64-m64n128-pingpong-production"
        or report.get("balanced_interleaved_pairs") != 7
        or report.get("trial_order") != "7 paired repetitions, each fp8,q4 then q4,fp8"
        or not isinstance(oracle, dict)
        or oracle.get("probe_count") != 9
        or oracle.get("tokens") != [0, 17, 2047]
        or oracle.get("rows") != [0, 12345, 34815]
        or type(oracle.get("fp8_max_bf16_steps")) is not int
        or not 0 <= oracle["fp8_max_bf16_steps"] <= 1
        or type(oracle.get("q4_max_bf16_steps")) is not int
        or not 0 <= oracle["q4_max_bf16_steps"] <= 2
        or any(report.get(name) is not True for name in required_true)
        or type(report.get("selected_workspace_bytes")) is not int
        or not 0 <= report["selected_workspace_bytes"] <= (512 << 20)
        or not _nonnegative(report.get("selected_waves_count"))
        or not isinstance(algorithm, dict)
        or algorithm.get("heuristic_requested") != 8
        or type(algorithm.get("heuristic_returned")) is not int
        or not 1 <= algorithm["heuristic_returned"] <= 8
        or type(algorithm.get("selected_rank")) is not int
        or not 0 <= algorithm["selected_rank"] < algorithm["heuristic_returned"]
        or not isinstance(algorithm.get("opaque_data_hex"), str)
        or re.fullmatch(r"[0-9a-f]{32}", algorithm["opaque_data_hex"]) is None
        or algorithm.get("selected_workspace_bytes") != report["selected_workspace_bytes"]
        or type(algorithm.get("algorithm_max_workspace_bytes")) is not int
        or algorithm["algorithm_max_workspace_bytes"] < 0
        or algorithm.get("waves_count") != report["selected_waves_count"]
        or algorithm.get("compute") != "HIPBLAS_COMPUTE_32F"
        or algorithm.get("scale_type") != "HIP_R_32F"
        or algorithm.get("a") != "HIP_R_8F_E4M3-col[K,N]-ldK-opT-outer_vec32f"
        or algorithm.get("b") != "HIP_R_8F_E4M3-col[K,T]-ldK-outer_vec32f"
        or algorithm.get("d") != "HIP_R_16BF-col[N,T]-ldN"
    ):
        raise ValueError("report does not describe the admitted matched gate_up qualification")
    if verify_files:
        source_root = Path(provenance["source_root"]).resolve(strict=True)
        source = (source_root / "tools/r9700/fp8_gate_up_qual.hip").resolve(strict=True)
        executable = Path(provenance["executable_path"]).resolve(strict=True)
        if source_root != Path(provenance["source_root"]) or source.parent.parent.parent != source_root:
            raise ValueError("qualifier source_root is not canonical")
        if _sha256(source) != provenance["source_sha256"]:
            raise ValueError("qualifier source hash differs from the retained report")
        if _sha256(executable) != provenance["executable_sha256"]:
            raise ValueError("qualifier executable hash differs from the retained report")
        if Path(power["path"]).read_text(encoding="utf-8").strip() != "auto":
            raise ValueError("device-0 DPM profile is not auto during validation")
    fp8_samples = _samples(report, "fp8_complete_ms_samples")
    q4_samples = _samples(report, "q4_complete_ms_samples")
    fp8_ordered = sorted(fp8_samples)
    q4_ordered = sorted(q4_samples)
    fp8_median = 0.5 * (fp8_ordered[6] + fp8_ordered[7])
    q4_median = 0.5 * (q4_ordered[6] + q4_ordered[7])
    ratio = q4_median / fp8_median
    expected = {
        "fp8_complete_median_ms": fp8_median,
        "q4_complete_median_ms": q4_median,
        "fp8_complete_tflops": OPERATIONS / (fp8_median * 1.0e9),
        "q4_complete_tflops_equivalent": OPERATIONS / (q4_median * 1.0e9),
        "fp8_over_q4_speedup": ratio,
    }
    for name, value in expected.items():
        observed = report.get(name)
        if not (
            _positive(observed)
            and math.isclose(float(observed), value, rel_tol=2e-6, abs_tol=1e-9)
        ):
            raise ValueError(f"{name} differs from retained timing samples")
    underperforms = ratio > 1.0
    if report.get("current_q4_implementation_underperforms_fp8") is not underperforms:
        raise ValueError("Q4 implementation verdict differs from the matched timing ratio")
    if report.get("current_q4_implementation_poor_vs_fp8") is not underperforms:
        raise ValueError("Q4 poor-versus-FP8 verdict differs from the matched timing ratio")
    return {
        "fp8_complete_median_ms": fp8_median,
        "q4_complete_median_ms": q4_median,
        "fp8_over_q4_speedup": ratio,
        "current_q4_implementation_underperforms_fp8": underperforms,
        "current_q4_implementation_poor_vs_fp8": underperforms,
        "device": hardware,
        "power_profile": power,
        "toolchain": toolchain,
        "algorithm": algorithm,
        "verdict": (
            "current A8Q4G64 is slower than the matched FP8 complete path"
            if underperforms
            else "current A8Q4G64 is not slower than the matched FP8 complete path"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    before = _sha256(args.report.resolve(strict=True))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = validate(report, verify_files=True)
    if _sha256(args.report) != before:
        raise ValueError("retained report changed during validation")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
