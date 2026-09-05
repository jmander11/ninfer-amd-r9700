#!/usr/bin/env python3
"""Select capacity-feasible rowwise-E4M3 projection families from a measured Q4 trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from tools.artifact.layouts import encoded_size
from tools.convert.qwen3_8_27b_r9700 import e4m3_inventory, q4_inventory


SCHEMA = "ninfer.r9700.e4m3-q4-hybrid-selection.v1"
FP8_ARCHITECTURAL_CEILING_TFLOPS = 400.835
TOKENS = 2048


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _role(name: str) -> str:
    parts = name.split("/")
    if len(parts) == 5 and parts[:2] == ["text", "layers"]:
        return f"text.{parts[3]}.{parts[4]}"
    raise ValueError(f"projection object has no admitted shared role: {name}")


def expected_projection_objects() -> list[tuple[str, tuple[int, int]]]:
    result: list[tuple[str, tuple[int, int]]] = []
    full_attention = set(range(3, 64, 4))
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        if layer in full_attention:
            result.extend(
                (
                    (prefix + "attention/query_key", (7168, 5120)),
                    (prefix + "attention/gate_value", (7168, 5120)),
                    (prefix + "attention/output", (5120, 6144)),
                )
            )
        else:
            result.extend(
                (
                    (prefix + "gdn/query_key", (4096, 5120)),
                    (prefix + "gdn/value_z", (12288, 5120)),
                    (prefix + "gdn/output", (5120, 6144)),
                )
            )
        result.extend(
            (
                (prefix + "mlp/gate_up", (34816, 5120)),
                (prefix + "mlp/down", (5120, 17408)),
            )
        )
    return result


def measured_q4_calls(kernel_trace: Path, marker_trace: Path) -> list[dict[str, object]]:
    with marker_trace.open(newline="", encoding="utf-8") as source:
        markers = list(csv.DictReader(source))
    measured = [row for row in markers if row["Function"] == "ninfer_bench_measured"]
    if len(measured) != 1:
        raise ValueError("trace must contain exactly one measured marker")
    begin = int(measured[0]["Start_Timestamp"])
    end = int(measured[0]["End_Timestamp"])

    with kernel_trace.open(newline="", encoding="utf-8") as source:
        # Dispatch order preserves the program's quantize/linear pairing. Independent queues can
        # overlap timestamps, so timestamp sorting can invert an adjacent pair.
        kernels = sorted(csv.DictReader(source), key=lambda row: int(row["Dispatch_Id"]))
    relevant = [
        row
        for row in kernels
        if begin <= int(row["Start_Timestamp"])
        and int(row["End_Timestamp"]) <= end
        and (
            "a8g64_quantize_activation_kernel" in row["Kernel_Name"]
            or "a8q4g64_linear_prefill_cta_kernel" in row["Kernel_Name"]
        )
    ]
    pairs = []
    pending = None
    for row in relevant:
        if "quantize" in row["Kernel_Name"]:
            pending = row
        elif pending is not None:
            pairs.append((pending, row))
            pending = None
    expected = expected_projection_objects()
    if len(pairs) != len(expected):
        raise ValueError(f"measured trace has {len(pairs)} Q4 calls, expected {len(expected)}")

    calls = []
    for ordinal, ((name, shape), quantize, linear) in enumerate(
        ((item[0], item[1][0], item[1][1]) for item in zip(expected, pairs))
    ):
        if "quantize" not in quantize["Kernel_Name"] or "linear_prefill" not in linear["Kernel_Name"]:
            raise ValueError(f"Q4 call {ordinal} is not a quantize/linear pair")
        rows, columns = shape
        observed_columns = int(quantize["Grid_Size_X"]) * 2
        observed_rows = int(linear["Grid_Size_X"]) // 4
        if (observed_rows, observed_columns) != shape:
            raise ValueError(
                f"Q4 call {ordinal} shape {(observed_rows, observed_columns)} != {shape}"
            )
        service_ns = int(linear["End_Timestamp"]) - int(linear["Start_Timestamp"])
        calls.append(
            {
                "name": name,
                "role": _role(name),
                "shape": [rows, columns],
                "q4_measured_service_ns": service_ns,
            }
        )
    return calls


def enrich_calls(calls: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    q4_specs = {spec.name: spec for spec in q4_inventory.TENSOR_SPECS}
    fp8_specs = {spec.name: spec for spec in e4m3_inventory.TENSOR_SPECS}
    result = []
    for call in calls:
        name = str(call["name"])
        q4 = q4_specs[name]
        fp8 = fp8_specs[name]
        q4_bytes = encoded_size(q4.layout, q4.format, q4.shape)
        fp8_bytes = encoded_size(fp8.layout, fp8.format, fp8.shape)
        added = fp8_bytes - q4_bytes
        rows, columns = q4.shape
        operations = 2 * TOKENS * rows * columns
        fp8_floor_ns = operations / (FP8_ARCHITECTURAL_CEILING_TFLOPS * 1e12) * 1e9
        measured = int(call["q4_measured_service_ns"])
        saving = max(0.0, measured - fp8_floor_ns)
        result.append(
            {
                **call,
                "q4_payload_bytes": q4_bytes,
                "fp8_payload_bytes": fp8_bytes,
                "added_resident_bytes": added,
                "fp8_compute_floor_ns": fp8_floor_ns,
                "projected_saving_ns": saving,
                "projected_saving_ns_per_added_byte": saving / added,
            }
        )
    return result


def role_candidates(calls: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for call in calls:
        grouped.setdefault(str(call["role"]), []).append(call)
    result = []
    for role, objects in grouped.items():
        added = sum(int(item["added_resident_bytes"]) for item in objects)
        saving = sum(float(item["projected_saving_ns"]) for item in objects)
        result.append(
            {
                "role": role,
                "object_count": len(objects),
                "objects": [str(item["name"]) for item in objects],
                "added_resident_bytes": added,
                "q4_measured_service_ns": sum(
                    int(item["q4_measured_service_ns"]) for item in objects
                ),
                "fp8_compute_floor_ns": sum(float(item["fp8_compute_floor_ns"]) for item in objects),
                "projected_saving_ns": saving,
                "projected_saving_ns_per_added_byte": saving / added,
            }
        )
    return sorted(result, key=lambda item: str(item["role"]))


def select_roles(candidates: Sequence[Mapping[str, object]], budget: int) -> dict[str, object]:
    best: tuple[float, int, tuple[str, ...], tuple[Mapping[str, object], ...]] | None = None
    for count in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, count):
            cost = sum(int(item["added_resident_bytes"]) for item in subset)
            if cost > budget:
                continue
            saving = sum(float(item["projected_saving_ns"]) for item in subset)
            roles = tuple(sorted(str(item["role"]) for item in subset))
            key = (saving, -cost, tuple(reversed(roles)), subset)
            if best is None or key[:3] > best[:3]:
                best = key
    assert best is not None
    saving, negative_cost, _, subset = best
    return {
        "budget_bytes": budget,
        "selected_added_resident_bytes": -negative_cost,
        "unused_budget_bytes": budget + negative_cost,
        "selected_roles": sorted(str(item["role"]) for item in subset),
        "selected_object_count": sum(int(item["object_count"]) for item in subset),
        "projected_saving_ns": saving,
    }


def assemble_report(
    *, capacity: Mapping[str, object], calls: Sequence[Mapping[str, object]],
    baseline_prefill_seconds: float, baseline_total_seconds: float,
    provenance: Mapping[str, object]
) -> dict[str, object]:
    enriched = enrich_calls(calls)
    candidates = role_candidates(enriched)
    ranked_objects = sorted(
        enriched,
        key=lambda item: (-float(item["projected_saving_ns_per_added_byte"]), str(item["name"])),
    )
    ranked_roles = sorted(
        candidates,
        key=lambda item: (-float(item["projected_saving_ns_per_added_byte"]), str(item["role"])),
    )

    scenarios = []
    contract = capacity["planner_contract"]
    device = int(contract["device_capacity_bytes"])
    headroom = int(contract["automatic_headroom_bytes"])
    for scenario in capacity["scenarios"]:
        rows = []
        for group in scenario["groups"]:
            for cell in group["cells"]:
                budget = max(
                    0,
                    device
                    - headroom
                    - q4_inventory.DEVICE_ARENA_BYTES
                    - int(cell["projected_minimum_runtime_reservation_bytes"]),
                )
                selected = select_roles(candidates, budget)
                projected_seconds = max(
                    0.0, baseline_prefill_seconds - float(selected["projected_saving_ns"]) / 1e9
                )
                projected_total = max(
                    0.0, baseline_total_seconds - float(selected["projected_saving_ns"]) / 1e9
                )
                rows.append(
                    {
                        "kv_value_group": group["kv_value_group"],
                        "concurrency": cell["concurrency"],
                        "maximum_fp8_over_q4_bytes": budget,
                        **selected,
                        "optimistic_prefill_seconds_lower_bound": projected_seconds,
                        "optimistic_prefill_tok_s_upper_bound": (
                            TOKENS / projected_seconds if projected_seconds else None
                        ),
                        "optimistic_whole_p2048_seconds_lower_bound": projected_total,
                        "optimistic_whole_p2048_tok_s_upper_bound": (
                            TOKENS / projected_total if projected_total else None
                        ),
                    }
                )
        scenarios.append(
            {
                "prefill_chunk": scenario["workspace"]["prefill_chunk"],
                "projected_workspace_bytes": scenario["workspace"]["projected_planner_workspace_bytes"],
                "cells": rows,
            }
        )

    return {
        "schema": SCHEMA,
        "model": {
            "tokens": TOKENS,
            "baseline_prefill_seconds": baseline_prefill_seconds,
            "baseline_total_seconds": baseline_total_seconds,
            "q4_projection_calls": len(enriched),
            "q4_projection_service_ns": sum(int(item["q4_measured_service_ns"]) for item in enriched),
            "fp8_architectural_ceiling_tflops": FP8_ARCHITECTURAL_CEILING_TFLOPS,
            "projection_note": (
                "FP8 time uses the architectural compute floor; activation quantization, "
                "hipBLASLt launch and memory costs are omitted. Reported speed is an optimistic upper bound."
            ),
        },
        "selection_contract": {
            "unit": "all real Text-layer objects sharing one semantic projection role",
            "unmeasured_objects": "excluded from benefit and selection",
            "capacity_floor": "262144-token MTP3 Device-Graph minimum startup",
        },
        "ranked_role_candidates": ranked_roles,
        "ranked_projection_objects": ranked_objects,
        "scenarios": scenarios,
        "provenance": dict(provenance),
    }


def canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--kernel-trace", type=Path, required=True)
    parser.add_argument("--marker-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capacity = json.loads(args.capacity.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    calls = measured_q4_calls(args.kernel_trace, args.marker_trace)
    tests = benchmark["tests"]
    if len(tests) != 1 or int(tests[0]["n_prompt"]) != TOKENS:
        raise ValueError("benchmark is not the single P2048 authority")
    report = assemble_report(
        capacity=capacity,
        calls=calls,
        baseline_prefill_seconds=float(tests[0]["prefill_seconds_mean"]),
        baseline_total_seconds=float(tests[0]["total_seconds_mean"]),
        provenance={
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("capacity", args.capacity),
                ("benchmark", args.benchmark),
                ("kernel_trace", args.kernel_trace),
                ("marker_trace", args.marker_trace),
            )
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_json(report))
    print(json.dumps({"scenarios": report["scenarios"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
