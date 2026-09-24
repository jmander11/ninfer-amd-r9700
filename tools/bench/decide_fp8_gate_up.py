#!/usr/bin/env python3
"""Own the post-measurement proceed/reject decision for the FP8 MLP gate/up role."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

from tools.artifact.layouts import encoded_size
from tools.bench.analyze_e4m3_hybrid import enrich_calls, measured_q4_calls, role_candidates
from tools.convert.qwen3_8_27b_r9700 import e4m3_inventory, q4_inventory


SCHEMA = "ninfer.r9700.fp8-gate-up-decision.v1"
QUALIFIER_SCHEMA = "ninfer.r9700.fp8_gate_up_qualification.v1"
HYBRID_SCHEMA = "ninfer.r9700.e4m3-q4-hybrid-selection.v1"
CAPACITY_SCHEMA = "ninfer.r9700.e4m3-capacity-analysis.v1"
ROLE = "text.mlp.gate_up"
SHAPE = (2048, 34816, 5120)


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_bound(path: Path, expected_sha256: str) -> dict[str, object]:
    if not _valid_sha256(expected_sha256):
        raise ValueError(f"{path}: expected SHA-256 is malformed")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {actual} != expected {expected_sha256}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: report root is not an object")
    return value


def gate_up_inventory() -> dict[str, object]:
    q4 = {spec.name: spec for spec in q4_inventory.TENSOR_SPECS}
    fp8 = {spec.name: spec for spec in e4m3_inventory.TENSOR_SPECS}
    names = [f"text/layers/{layer}/mlp/gate_up" for layer in range(64)]
    q4_bytes = sum(encoded_size(q4[name].layout, q4[name].format, q4[name].shape) for name in names)
    fp8_bytes = sum(
        encoded_size(fp8[name].layout, fp8[name].format, fp8[name].shape) for name in names
    )
    if any(q4[name].shape != (34816, 5120) or fp8[name].shape != (34816, 5120) for name in names):
        raise ValueError("gate/up inventory shape differs from the fixed role")
    return {
        "role": ROLE,
        "object_count": len(names),
        "objects": names,
        "q4_resident_bytes": q4_bytes,
        "fp8_resident_bytes": fp8_bytes,
        "added_resident_bytes": fp8_bytes - q4_bytes,
    }


def validate_qualifier(report: Mapping[str, object]) -> dict[str, float | int]:
    if (
        report.get("schema") != QUALIFIER_SCHEMA
        or report.get("artifact_type") != "ninfer_r9700_fp8_gate_up_comparison"
        or report.get("schema_version") != 2
        or report.get("pass") is not True
        or report.get("shape") != {"tokens": SHAPE[0], "rows": SHAPE[1], "columns": SHAPE[2]}
        or report.get("fp8_profile") != "E4M3-outer-vec32f-hipBLASLt-top-supported"
        or report.get("q4_control") != "A8Q4G64-m64n128-pingpong-production"
    ):
        raise ValueError("qualifier schema, identity, or fixed shape differs")
    qualifier_provenance = report.get("provenance")
    if not isinstance(qualifier_provenance, Mapping):
        raise ValueError("qualifier provenance is absent")
    source_root = qualifier_provenance.get("source_root")
    executable_path = qualifier_provenance.get("executable_path")
    if not isinstance(source_root, str) or not isinstance(executable_path, str):
        raise ValueError("qualifier provenance paths are malformed")
    source_path = Path(source_root) / "tools/r9700/fp8_gate_up_qual.hip"
    executable = Path(executable_path)
    if (
        not source_path.is_file()
        or not executable.is_file()
        or qualifier_provenance.get("source_sha256") != sha256_file(source_path)
        or qualifier_provenance.get("executable_sha256") != sha256_file(executable)
    ):
        raise ValueError("qualifier source or executable live hash differs")
    hardware = report.get("hardware")
    power = report.get("power_profile")
    if not isinstance(hardware, Mapping) or (
        hardware.get("device") != "AMD Radeon AI PRO R9700"
        or hardware.get("architecture") != "gfx1201"
        or hardware.get("wave_size") != 32
        or hardware.get("vendor_id") != "0x1002"
        or hardware.get("device_id") != "0x7551"
    ):
        raise ValueError("qualifier hardware identity differs")
    if not isinstance(power, Mapping) or any(
        power.get(field) != "auto" for field in ("required", "before", "after")
    ):
        raise ValueError("qualifier did not retain the required auto power profile")
    required_true = (
        "direct_weight_binding",
        "outer_vector_scales",
        "nonfinite_status_poisoning",
        "q4_nonfinite_status_poisoning",
        "no_clobber_rejection",
    )
    if any(report.get(field) is not True for field in required_true):
        raise ValueError("qualifier correctness or rejection gate did not pass")
    oracle = report.get("represented_format_axis_oracle")
    if (
        not isinstance(oracle, Mapping)
        or oracle.get("probe_count") != 9
        or oracle.get("tokens") != [0, 17, 2047]
        or oracle.get("rows") != [0, 12345, 34815]
        or type(oracle.get("fp8_max_bf16_steps")) is not int
        or not 0 <= oracle["fp8_max_bf16_steps"] <= 1
        or type(oracle.get("q4_max_bf16_steps")) is not int
        or not 0 <= oracle["q4_max_bf16_steps"] <= 2
    ):
        raise ValueError("qualifier represented-format axis oracle differs")

    fp8_samples = report.get("fp8_complete_ms_samples")
    q4_samples = report.get("q4_complete_ms_samples")
    if (
        not isinstance(fp8_samples, list)
        or not isinstance(q4_samples, list)
        or len(fp8_samples) != 14
        or len(q4_samples) != 14
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (*fp8_samples, *q4_samples)
        )
        or report.get("balanced_interleaved_pairs") != 7
        or report.get("trial_order") != "7 paired repetitions, each fp8,q4 then q4,fp8"
    ):
        raise ValueError("qualifier matched sample set differs")
    fp8 = float(report["fp8_complete_median_ms"])
    q4 = float(report["q4_complete_median_ms"])
    speedup = float(report["fp8_over_q4_speedup"])
    if (
        not all(math.isfinite(value) and value > 0 for value in (fp8, q4, speedup))
        or not math.isclose(fp8, 0.5 * sum(sorted(map(float, fp8_samples))[6:8]), rel_tol=1e-6)
        or not math.isclose(q4, 0.5 * sum(sorted(map(float, q4_samples))[6:8]), rel_tol=1e-6)
        or not math.isclose(speedup, q4 / fp8, rel_tol=1e-6)
        or report.get("current_q4_implementation_underperforms_fp8") is not (speedup > 1.0)
        or report.get("current_q4_implementation_poor_vs_fp8") is not (speedup > 1.0)
    ):
        raise ValueError("qualifier medians or speedup are inconsistent")
    algorithm = report.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError("qualifier algorithm identity is absent")
    selected = int(report["selected_workspace_bytes"])
    if (
        selected != int(algorithm.get("selected_workspace_bytes", -1))
        or selected < 0
        or selected > int(algorithm.get("algorithm_max_workspace_bytes", -1))
        or int(algorithm.get("heuristic_returned", 0)) <= 0
    ):
        raise ValueError("qualifier selected workspace or heuristic is inconsistent")
    return {
        "fp8_complete_median_ms": fp8,
        "q4_complete_median_ms": q4,
        "fp8_over_q4_time_ratio": fp8 / q4,
        "fp8_over_q4_speedup": speedup,
        "selected_workspace_bytes": selected,
    }


def decide(
    *, qualifier: Mapping[str, object], hybrid: Mapping[str, object],
    capacity: Mapping[str, object], benchmark: Mapping[str, object],
    trace_gate_up_service_ns: int, provenance: Mapping[str, object]
) -> dict[str, object]:
    if hybrid.get("schema") != HYBRID_SCHEMA or capacity.get("schema") != CAPACITY_SCHEMA:
        raise ValueError("hybrid or capacity schema differs")
    capacity_identity = hybrid.get("provenance", {}).get("capacity", {})
    if capacity_identity.get("sha256") != provenance["capacity"]["sha256"]:
        raise ValueError("hybrid analysis does not bind the supplied capacity report")
    for name in ("benchmark", "kernel_trace", "marker_trace"):
        identity = hybrid.get("provenance", {}).get(name, {})
        if identity.get("sha256") != provenance[name]["sha256"]:
            raise ValueError(f"hybrid analysis does not bind supplied {name}")

    benchmark_config = benchmark.get("config")
    benchmark_tests = benchmark.get("tests")
    if (
        benchmark.get("schema_version") != 20
        or not isinstance(benchmark_config, Mapping)
        or benchmark_config.get("q4_prefill_cta_profile") != "m64n128-pingpong-production"
        or benchmark_config.get("concurrency") != 1
        or benchmark_config.get("spec") != "none"
        or not isinstance(benchmark_tests, list)
        or len(benchmark_tests) != 1
        or benchmark_tests[0].get("n_prompt") != 2048
        or benchmark_tests[0].get("n_gen") != 0
    ):
        raise ValueError("benchmark is not the fresh production-pingpong P2048 authority")

    timing = validate_qualifier(qualifier)
    inventory = gate_up_inventory()
    candidates = [item for item in hybrid.get("ranked_role_candidates", []) if item.get("role") == ROLE]
    if len(candidates) != 1:
        raise ValueError("hybrid analysis lacks the unique gate/up role")
    candidate = candidates[0]
    if (
        candidate.get("object_count") != inventory["object_count"]
        or candidate.get("objects") != inventory["objects"]
        or candidate.get("added_resident_bytes") != inventory["added_resident_bytes"]
    ):
        raise ValueError("hybrid gate/up inventory differs from exact current inventories")
    if int(candidate.get("q4_measured_service_ns", -1)) != trace_gate_up_service_ns:
        raise ValueError("hybrid gate/up service differs from the supplied trace")

    contract = capacity.get("planner_contract")
    if not isinstance(contract, Mapping) or (
        contract.get("speculative_backend") != "mtp"
        or contract.get("draft_tokens") != 3
        or contract.get("device_graph") is not True
        or contract.get("max_context") != 262144
        or contract.get("automatic_headroom_bytes") != 1 << 30
    ):
        raise ValueError("capacity report differs from the MTP3 Device-Graph envelope")
    device_bytes = int(contract["device_capacity_bytes"])
    capacity_budgets = {}
    for scenario in capacity.get("scenarios", []):
        prefill_chunk = int(scenario["workspace"]["prefill_chunk"])
        for group in scenario["groups"]:
            for cell in group["cells"]:
                key = (prefill_chunk, int(group["kv_value_group"]), int(cell["concurrency"]))
                capacity_budgets[key] = (
                    device_bytes
                    - int(contract["automatic_headroom_bytes"])
                    - q4_inventory.DEVICE_ARENA_BYTES
                    - int(cell["projected_minimum_runtime_reservation_bytes"])
                )
    capacity_cells = []
    added = int(inventory["added_resident_bytes"])
    for scenario in hybrid.get("scenarios", []):
        prefill_chunk = int(scenario["prefill_chunk"])
        if prefill_chunk not in (2048, 8192):
            raise ValueError("hybrid analysis has an unknown prefill scenario")
        for cell in scenario["cells"]:
            key = (prefill_chunk, int(cell["kv_value_group"]), int(cell["concurrency"]))
            if key not in capacity_budgets:
                raise ValueError("capacity authority lacks a hybrid cell")
            budget = capacity_budgets[key]
            if int(cell["maximum_fp8_over_q4_bytes"]) != budget:
                raise ValueError("hybrid byte budget differs from direct capacity arithmetic")
            slack = budget - added
            capacity_cells.append(
                {
                    "prefill_chunk": prefill_chunk,
                    "kv_value_group": int(cell["kv_value_group"]),
                    "concurrency": int(cell["concurrency"]),
                    "maximum_fp8_over_q4_bytes": budget,
                    "gate_up_added_resident_bytes": added,
                    "capacity_slack_bytes": slack,
                    "capacity_preserved": slack >= 0,
                }
            )
    expected_cells = {(p, g, c) for p in (2048, 8192) for g in (16, 32) for c in range(1, 5)}
    actual_cells = {
        (cell["prefill_chunk"], cell["kv_value_group"], cell["concurrency"])
        for cell in capacity_cells
    }
    if actual_cells != expected_cells or len(capacity_cells) != 16:
        raise ValueError("hybrid analysis lacks the exact 16 capacity cells")

    measured_q4_ns = trace_gate_up_service_ns
    ratio = float(timing["fp8_over_q4_time_ratio"])
    projected_fp8_ns = measured_q4_ns * ratio
    saving_ns = measured_q4_ns - projected_fp8_ns
    baseline_prefill = float(benchmark_tests[0]["prefill_seconds_mean"])
    baseline_total = float(benchmark_tests[0]["total_seconds_mean"])
    if (
        baseline_prefill != float(hybrid["model"]["baseline_prefill_seconds"])
        or baseline_total != float(hybrid["model"]["baseline_total_seconds"])
    ):
        raise ValueError("hybrid baseline differs from the supplied benchmark")
    projected_prefill = baseline_prefill - saving_ns / 1e9
    projected_total = baseline_total - saving_ns / 1e9
    reasons = []
    if ratio >= 1.0:
        reasons.append("matched_complete_fp8_path_is_not_faster")
    if any(not cell["capacity_preserved"] for cell in capacity_cells):
        reasons.append("gate_up_role_exceeds_at_least_one_capacity_envelope")
    if projected_total >= baseline_total:
        reasons.append("projected_whole_p2048_does_not_improve")

    return {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "exact_all-layer_text_mlp_gate_up_role_only",
        "inventory": inventory,
        "matched_measurement": timing,
        "capacity": {
            "all_cells_preserve_minimum_startup": all(
                cell["capacity_preserved"] for cell in capacity_cells
            ),
            "cells": sorted(
                capacity_cells,
                key=lambda cell: (
                    cell["prefill_chunk"], cell["kv_value_group"], cell["concurrency"]
                ),
            ),
        },
        "whole_p2048_projection": {
            "measured_gate_up_q4_service_ns": measured_q4_ns,
            "projected_gate_up_fp8_service_ns": projected_fp8_ns,
            "projected_service_saving_ns": saving_ns,
            "baseline_prefill_seconds": baseline_prefill,
            "projected_prefill_seconds": projected_prefill,
            "projected_prefill_tok_s": SHAPE[0] / projected_prefill,
            "baseline_total_seconds": baseline_total,
            "projected_total_seconds": projected_total,
            "projected_whole_p2048_tok_s": SHAPE[0] / projected_total,
        },
        "decision": {
            "verdict": "proceed" if not reasons else "reject",
            "gate": (
                "matched complete FP8 median faster than Q4, exact gate/up bytes fit every "
                "P2048/P8192 G16/G32 C1..4 minimum-startup envelope, and projected whole improves"
            ),
            "reasons": reasons,
        },
        "provenance": dict(provenance),
    }


def canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qualifier", "hybrid", "capacity", "benchmark", "kernel-trace", "marker-trace"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "qualifier": args.qualifier,
        "hybrid": args.hybrid,
        "capacity": args.capacity,
        "benchmark": args.benchmark,
        "kernel_trace": args.kernel_trace,
        "marker_trace": args.marker_trace,
    }
    expected = {
        "qualifier": args.qualifier_sha256,
        "hybrid": args.hybrid_sha256,
        "capacity": args.capacity_sha256,
        "benchmark": args.benchmark_sha256,
        "kernel_trace": args.kernel_trace_sha256,
        "marker_trace": args.marker_trace_sha256,
    }
    loaded = {
        name: _load_bound(path, expected[name])
        for name, path in paths.items()
        if name in ("qualifier", "hybrid", "capacity")
    }
    for name in ("benchmark", "kernel_trace", "marker_trace"):
        if sha256_file(paths[name]) != expected[name]:
            raise ValueError(f"{name} SHA-256 differs")
    benchmark = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    trace_candidates = role_candidates(
        enrich_calls(measured_q4_calls(paths["kernel_trace"], paths["marker_trace"]))
    )
    trace_gate_up = [item for item in trace_candidates if item["role"] == ROLE]
    if len(trace_gate_up) != 1:
        raise ValueError("supplied trace lacks the unique gate/up role")
    provenance = {
        name: {"path": str(path), "sha256": expected[name]} for name, path in paths.items()
    }
    report = decide(
        qualifier=loaded["qualifier"], hybrid=loaded["hybrid"],
        capacity=loaded["capacity"], benchmark=benchmark,
        trace_gate_up_service_ns=int(trace_gate_up[0]["q4_measured_service_ns"]),
        provenance=provenance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_json(report))
    print(json.dumps(report["decision"], sort_keys=True))
    return 0 if report["decision"]["verdict"] == "proceed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
