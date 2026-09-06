#!/usr/bin/env python3
"""Fail-closed analyzer for the standalone FP8 target gate-up T5/T6 qualifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-fp8-target-gate-up-t5t6-qualification-20260906"
BINARY = ROOT / "tools/r9700/build/fp8_gate_up_small_t_qual"
ASSEMBLY = ROOT / "tools/r9700/build/fp8_gate_up_small_t.s"
BINARY_SHA = "4611e36dc2f1a5324a94a0da7f00c402b052f986dbf8ac4c504831d4327f4ece"
ASSEMBLY_SHA = "a463e37e04512213013302769fa946c2fd1a57565533782f60797f107f46071d"
BINARY_BYTES = 113272
ASSEMBLY_BYTES = 63959
SOURCE_COMMIT = "337940345d20cd0211fccd10bc4f7e9d0181bc2d"
CRITERION = (
    "both cells require overall candidate/incumbent median ratio at most 0.90, "
    "each launch-order median ratio at most 0.92, paired-ratio mean plus two "
    "standard errors below 0.95, launch-order median-ratio delta at most 0.03, "
    "and positive projected 64-call saving"
)
STATIC = (
    "FP8 target gate-up T5/T6 static PASS: logical_vgpr=54 logical_sgpr=19 "
    "lds=0 private=0 wmma_sites=4 wavefront=32 max_workgroup=128 "
    "vgpr_spills=0 sgpr_spills=0\n"
)
HARDWARE = (
    "device=0\nname=AMD Radeon AI PRO R9700\narchitecture=gfx1201\n"
    "wavefront=32\npci=0000:13:00.0\nvendor=0x1002\ndevice_id=0x7551\n"
)
LIMITATIONS = [
    "standalone represented-input operator qualification only",
    "does not establish whole-DFlash performance, exact-token parity, or production eligibility",
    "a passing result only justifies a separately reviewed production integration",
]
SERIALIZATION_QUANTUM_MS = 0.01
SERIALIZATION_HALF_STEP = SERIALIZATION_QUANTUM_MS / 2.0


def pairs(items: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"nonfinite JSON constant: {value}")), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"identity target is not a file: {resolved}")
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def capture_identity(path: Path, size: int, digest: str) -> dict:
    return {"path": str(path), "bytes": size, "sha256": digest}


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} is not finite numeric")
    return float(value)


def samples(cell: dict, name: str) -> list[float]:
    values = cell.get(name)
    if not isinstance(values, list) or len(values) != 14:
        raise ValueError(f"{name} must contain exactly 14 samples")
    result = [finite(value, f"{name}[{index}]") for index, value in enumerate(values)]
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{name} contains a nonpositive sample")
    for index, value in enumerate(result):
        require_cent_serialized(value, f"{name}[{index}]")
    return result


def require_cent_serialized(value: float, label: str) -> None:
    if not math.isclose(value / SERIALIZATION_QUANTUM_MS,
                        round(value / SERIALIZATION_QUANTUM_MS),
                        rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} is not serialized at the retained 0.01 precision")


def require_interval_overlap(actual: object, low: float, high: float, label: str) -> float:
    value = finite(actual, label)
    require_cent_serialized(value, label)
    if value + SERIALIZATION_HALF_STEP < low or value - SERIALIZATION_HALF_STEP > high:
        raise ValueError(f"reported {label} is incompatible with retained rounded samples")
    return value


def parse_raw(text: str) -> list[dict]:
    lines = [line for line in text.splitlines() if line]
    if len(lines) != 2:
        raise ValueError("benchmark stdout must contain exactly two JSON cells")
    result = []
    for line in lines:
        value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {value}")), object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise ValueError("benchmark cell is not an object")
        result.append(value)
    if [cell.get("tokens") for cell in result] != [5, 6]:
        raise ValueError("benchmark cells are not exact ordered T5/T6")
    return result


def analyze_cell(cell: dict) -> dict:
    expected_keys = {
        "schema", "production_routing_changed", "tokens", "rows", "columns",
        "timing_scope", "incumbent_samples_ms", "challenger_samples_ms",
        "incumbent_median_ms", "challenger_median_ms", "challenger_over_incumbent",
        "projected_64_call_saving_ms", "decision",
    }
    if set(cell) != expected_keys:
        raise ValueError("benchmark cell fields differ from the committed harness contract")
    tokens = cell.get("tokens")
    if isinstance(tokens, bool) or tokens not in (5, 6):
        raise ValueError("invalid token cell")
    if cell.get("schema") != "ninfer.r9700.fp8_target_gate_up_small_t_cell.v1" or \
       cell.get("production_routing_changed") is not False or \
       cell.get("rows") != 34816 or cell.get("columns") != 5120 or \
       cell.get("timing_scope") != "complete_bf16_quantize_matmul_status_poison" or \
       cell.get("decision") != "unmade_until_gpu_measurement_is_validated":
        raise ValueError("benchmark cell identity/scope differs")
    incumbent = samples(cell, "incumbent_samples_ms")
    challenger = samples(cell, "challenger_samples_ms")
    incumbent_median = statistics.median(incumbent)
    challenger_median = statistics.median(challenger)
    incumbent_low = incumbent_median - SERIALIZATION_HALF_STEP
    incumbent_high = incumbent_median + SERIALIZATION_HALF_STEP
    challenger_low = challenger_median - SERIALIZATION_HALF_STEP
    challenger_high = challenger_median + SERIALIZATION_HALF_STEP
    require_interval_overlap(cell.get("incumbent_median_ms"), incumbent_low, incumbent_high,
                             "incumbent_median_ms")
    require_interval_overlap(cell.get("challenger_median_ms"), challenger_low, challenger_high,
                             "challenger_median_ms")
    ratio_low = challenger_low / incumbent_high
    ratio_high = challenger_high / incumbent_low
    require_interval_overlap(cell.get("challenger_over_incumbent"), ratio_low, ratio_high,
                             "challenger_over_incumbent")
    saving = 64.0 * (incumbent_median - challenger_median)
    saving_low = 64.0 * (incumbent_low - challenger_high)
    saving_high = 64.0 * (incumbent_high - challenger_low)
    require_interval_overlap(cell.get("projected_64_call_saving_ms"), saving_low, saving_high,
                             "projected_64_call_saving_ms")

    # Recovery from the retained 0.01-ms serialization is deliberately one-way. Publish a
    # decision only when every value represented by the rounding intervals makes positive saving
    # impossible. Ambiguous evidence, including any possible pass, requires fresh higher-precision
    # timing and must never be accepted here.
    if challenger_low <= incumbent_high:
        raise ValueError(
            "retained 0.01-ms timing is ambiguous or could pass; analysis cannot accept it")

    forward = [challenger[index] / incumbent[index] for index in range(0, 14, 2)]
    reverse = [challenger[index] / incumbent[index] for index in range(1, 14, 2)]
    ratios = forward + reverse
    ratio_mean = statistics.mean(ratios)
    ratio_sd = statistics.stdev(ratios)
    ratio_upper = ratio_mean + 2.0 * ratio_sd / math.sqrt(len(ratios))
    forward_median = statistics.median(forward)
    reverse_median = statistics.median(reverse)
    order_delta = abs(forward_median - reverse_median)
    overall_ratio = challenger_median / incumbent_median
    accepted = False
    return {
        "schema": "ninfer.r9700.fp8_target_gate_up_t5t6_qualified_cell.v1",
        "status": "accepted" if accepted else "rejected",
        "production_eligible": False,
        "tokens": tokens,
        "rows": 34816,
        "columns": 5120,
        "timing_scope": "complete_bf16_quantize_matmul_status_poison",
        "raw_samples": {"incumbent_ms": incumbent, "challenger_ms": challenger},
        "derived": {
            "incumbent_median_ms": incumbent_median,
            "challenger_median_ms": challenger_median,
            "challenger_over_incumbent": overall_ratio,
            "forward_candidate_over_incumbent": forward,
            "reverse_candidate_over_incumbent": reverse,
            "forward_ratio_median": forward_median,
            "reverse_ratio_median": reverse_median,
            "paired_ratio_mean": ratio_mean,
            "paired_ratio_sample_standard_deviation": ratio_sd,
            "paired_ratio_two_standard_error_upper": ratio_upper,
            "order_ratio_delta": order_delta,
            "projected_64_call_saving_ms": saving,
        },
        "decision": {"accepted": accepted, "criterion": CRITERION},
        "serialization": {
            "quantum_ms": SERIALIZATION_QUANTUM_MS,
            "interpretation": "each retained timing represents value plus or minus 0.005 ms",
            "acceptance_from_rounded_evidence_permitted": False,
        },
        "robust_reject_certificate": {
            "incumbent_median_upper_ms": incumbent_high,
            "challenger_median_lower_ms": challenger_low,
            "positive_projected_saving_possible": False,
            "reason": "challenger median lower bound exceeds incumbent median upper bound",
        },
    }


def validate_fixed_inputs() -> None:
    plan = load_json(PACKAGE / "plan.json")
    if plan.get("artifact_type") != "ninfer_r9700_fp8_target_gate_up_t5t6_qualification_plan" or \
       plan.get("schema_version") != 1 or plan.get("status") != "prepared_no_gpu_execution" or \
       plan.get("source_commit") != SOURCE_COMMIT or \
       plan.get("production_routing_authorized") is not False or \
       plan.get("whole_dflash_authority") is not False or \
       plan.get("timing", {}).get("criterion") != CRITERION or \
       plan.get("limitations") != LIMITATIONS:
        raise ValueError("plan authority differs")
    if plan.get("scope") != (
        "standalone complete BF16 activation quantize plus FP8 target gate-up matmul "
        "at exact N34816/K5120 for T5 and T6") or \
       plan.get("hardware") != {
           "device": 0, "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201",
           "wavefront_width": 32, "pci_bus": "0000:13:00.0",
           "pci_vendor_device": "1002:7551", "power_profile": "auto"} or \
       plan.get("cells") != [{"tokens": 5, "rows": 34816, "columns": 5120},
                              {"tokens": 6, "rows": 34816, "columns": 5120}] or \
       plan.get("numeric") != {
           "oracle": "independent BF16-source E4M3 quantization plus complete represented-E4M3 FP64 dot formula",
           "coverage": "all activation codes/scales, every output written, one full-K row per N256 CTA and explicit rows 0/63/64/255/256/34815 for every token",
           "maximum_bf16_steps": 1, "regression_before_timing": True} or \
       plan.get("timing") != {
           "method": "unprofiled HIP events", "samples_per_cell": 14,
           "launch_order": "even incumbent-then-challenger; odd challenger-then-incumbent",
           "criterion": CRITERION} or \
       plan.get("static") != {
           "logical_vgpr": 54, "logical_sgpr": 19, "emitted_total_sgpr": 21,
           "wmma_fp8_sites": 4, "global_load_b64_sites": 5, "occupancy": 16,
           "wavefront_width": 32, "maximum_workgroup": 128, "lds_bytes": 0,
           "private_bytes": 0, "scratch_bytes": 0, "sgpr_spills": 0,
           "vgpr_spills": 0} or \
       plan.get("outputs") != {
           "regression_stdout": "regression.stdout", "regression_stderr": "regression.stderr",
           "regression_exit": "regression.exit", "benchmark_stdout": "benchmark.stdout",
           "benchmark_stderr": "benchmark.stderr", "benchmark_exit": "benchmark.exit",
           "static": "static.stdout", "hardware": "hardware.txt",
           "power_before": "power-before.txt", "power_after": "power-after.txt",
           "cell_t5": "cell-t5.json", "cell_t6": "cell-t6.json",
           "summary": "summary.json", "result_closure": "result.sha256"}:
        raise ValueError("plan workload/static/output contract differs")
    provenance = load_json(PACKAGE / "build-provenance.json")
    if provenance.get("source", {}).get("commit") != SOURCE_COMMIT or \
       provenance.get("outputs", {}).get("binary") != \
       capture_identity(BINARY, BINARY_BYTES, BINARY_SHA) or \
       provenance.get("outputs", {}).get("assembly") != \
       capture_identity(ASSEMBLY, ASSEMBLY_BYTES, ASSEMBLY_SHA):
        raise ValueError("build provenance differs")
    if provenance.get("compiler_driver", {}).get("sha256") != \
       "7b95d430bb8c4d4237f9b4935dbd6440e2067fde0979cc69bffb26ebd021464c" or \
       provenance.get("compiler", {}).get("sha256") != \
       "241bf4da7ec39bc00b68ed74f6be751516d9892ed990e8c7fa372bad18500247" or \
       provenance.get("analyzer_python", {}).get("sha256") != \
       "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118" or \
       provenance.get("runtime", {}).get("libhipblaslt", {}).get("sha256") != \
       "1e8545626323f9b22c65d36492e1686b831cb85c8f7c8be3fc21c0d6cbb0d055" or \
       provenance.get("runtime", {}).get("libamdhip64", {}).get("sha256") != \
       "817aeadfd9f62b68831ad89993163c7f1f470f30595e7e0da5fdc942193142a8":
        raise ValueError("toolchain/runtime provenance differs")
    if (PACKAGE / "regression.stdout").read_text() != \
       "FP8 target gate-up T5/T6 regression PASS\n" or \
       (PACKAGE / "regression.stderr").read_bytes() != b"" or \
       (PACKAGE / "regression.exit").read_text() != "0\n":
        raise ValueError("regression evidence differs")
    if (PACKAGE / "benchmark.stderr").read_bytes() != b"" or \
       (PACKAGE / "benchmark.exit").read_text() != "0\n":
        raise ValueError("benchmark process evidence differs")
    if (PACKAGE / "static.stdout").read_text() != STATIC:
        raise ValueError("static evidence differs")
    if (PACKAGE / "hardware.txt").read_text() != HARDWARE:
        raise ValueError("hardware evidence differs")
    if (PACKAGE / "power-before.txt").read_text() != "auto\n" or \
       (PACKAGE / "power-after.txt").read_text() != "auto\n":
        raise ValueError("power profile was not auto at both endpoints")
    repair = load_json(PACKAGE / "analysis-repair.json")
    if repair.get("artifact_type") != \
       "ninfer_r9700_fp8_target_gate_up_t5t6_analysis_repair" or \
       repair.get("schema_version") != 1 or \
       repair.get("status") != "analysis_only_no_gpu_rerun" or \
       repair.get("gpu_rerun_permitted") is not False or \
       repair.get("serialization_quantum_ms") != SERIALIZATION_QUANTUM_MS or \
       repair.get("capture_analyzer", {}).get("sha256") != \
       "e2d8931c86fa7666887dfc5dda9688cc8beea1cddd959e936a9b6bc97dc402c0" or \
       repair.get("repaired_analyzer") != identity(Path(__file__)) or \
       repair.get("capture_result_closure") != identity(PACKAGE / "result.sha256") or \
       repair.get("capture_benchmark_stdout") != identity(PACKAGE / "benchmark.stdout"):
        raise ValueError("analysis-repair provenance differs")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cell-t5", type=Path, required=True)
    parser.add_argument("--cell-t6", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    expected = (PACKAGE / "benchmark.stdout", PACKAGE / "cell-t5.json",
                PACKAGE / "cell-t6.json", PACKAGE / "summary.json")
    if (args.raw, args.cell_t5, args.cell_t6, args.summary) != expected or \
       any(path.exists() for path in expected[1:]):
        raise ValueError("analyzer paths are not exact fresh package outputs")
    validate_fixed_inputs()
    cells = [analyze_cell(cell) for cell in parse_raw(args.raw.read_text())]
    accepted = all(cell["decision"]["accepted"] for cell in cells)
    raw_names = [
        "regression.stdout", "regression.stderr", "regression.exit",
        "benchmark.stdout", "benchmark.stderr", "benchmark.exit", "static.stdout",
        "hardware.txt", "power-before.txt", "power-after.txt",
    ]
    summary = {
        "artifact_type": "ninfer_r9700_fp8_target_gate_up_t5t6_qualification",
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "production_eligible": False,
        "production_routing_authorized": False,
        "whole_dflash_authority": False,
        "source_commit": SOURCE_COMMIT,
        "binary": capture_identity(BINARY, BINARY_BYTES, BINARY_SHA),
        "assembly": capture_identity(ASSEMBLY, ASSEMBLY_BYTES, ASSEMBLY_SHA),
        "build_provenance": identity(PACKAGE / "build-provenance.json"),
        "analysis_repair": identity(PACKAGE / "analysis-repair.json"),
        "capture_result_closure": identity(PACKAGE / "result.sha256"),
        "hardware": {"device": 0, "name": "AMD Radeon AI PRO R9700",
                     "architecture": "gfx1201", "wavefront_width": 32,
                     "pci_bus": "0000:13:00.0", "pci_vendor_device": "1002:7551",
                     "power_profile_before_after": "auto"},
        "numeric_regression": "passed_before_timing",
        "decision": {"accepted": accepted, "criterion": CRITERION,
                     "requires_both_cells": True},
        "cells": [{"path": str(path), "tokens": cell["tokens"],
                   "status": cell["status"]}
                  for path, cell in zip(expected[1:3], cells, strict=True)],
        "raw_evidence": {name: identity(PACKAGE / name) for name in raw_names},
        "limitations": LIMITATIONS,
    }
    write_json(args.cell_t5, cells[0])
    write_json(args.cell_t6, cells[1])
    write_json(args.summary, summary)
    return 0 if accepted else 1


def entrypoint() -> int:
    try:
        return main()
    except Exception as error:
        print(f"fp8 target gate-up analysis failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
