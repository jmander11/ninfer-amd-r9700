#!/usr/bin/env python3
"""Select the next capacity-feasible FP8 projection roles after gate/up admission."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from tools.artifact.layouts import encoded_size
from tools.bench.analyze_e4m3_hybrid import expected_projection_objects
from tools.convert.qwen3_8_27b_r9700 import e4m3_inventory, q4_inventory


SCHEMA = "ninfer.r9700.fp8-post-gate-up-decision.v1"
QUALIFIER_SCHEMA = "ninfer.r9700.fp8_projection_qualification.v1"
GATE_SCHEMA = "ninfer.r9700.fp8-gate-up-decision.v1"
HYBRID_SCHEMA = "ninfer.r9700.e4m3-q4-hybrid-selection.v1"
CAPACITY_SCHEMA = "ninfer.r9700.e4m3-capacity-analysis.v1"
TOKENS = 2048
GATE_ROLE = "text.mlp.gate_up"
SELECTED_ROLES = (
    "text.attention.gate_value",
    "text.attention.query_key",
    "text.gdn.query_key",
)
QUALIFICATIONS = {
    "attention_qk_gate_value": {
        "shape": (TOKENS, 7168, 5120),
        "roles": ("text.attention.gate_value", "text.attention.query_key"),
    },
    "gdn_query_key": {
        "shape": (TOKENS, 4096, 5120),
        "roles": ("text.gdn.query_key",),
    },
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load_bound(path: Path, expected_sha256: str) -> dict[str, object]:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError(f"{path}: malformed expected SHA-256")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{path}: SHA-256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: report root is not an object")
    return value


def _role(name: str) -> str:
    parts = name.split("/")
    if len(parts) != 5 or parts[:2] != ["text", "layers"]:
        raise ValueError(f"invalid projection name {name}")
    return f"text.{parts[3]}.{parts[4]}"


def exact_role_inventory() -> dict[str, dict[str, object]]:
    q4 = {spec.name: spec for spec in q4_inventory.TENSOR_SPECS}
    fp8 = {spec.name: spec for spec in e4m3_inventory.TENSOR_SPECS}
    grouped: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    for name, shape in expected_projection_objects():
        grouped.setdefault(_role(name), []).append((name, shape))
    result: dict[str, dict[str, object]] = {}
    for role, objects in grouped.items():
        names = [name for name, _ in objects]
        shapes = {shape for _, shape in objects}
        if len(shapes) != 1:
            raise ValueError(f"role {role} does not have one fixed shape")
        if any(q4[name].shape != shape or fp8[name].shape != shape for name, shape in objects):
            raise ValueError(f"role {role} differs from the exact converter inventories")
        q4_bytes = sum(encoded_size(q4[name].layout, q4[name].format, q4[name].shape) for name in names)
        fp8_bytes = sum(
            encoded_size(fp8[name].layout, fp8[name].format, fp8[name].shape) for name in names
        )
        result[role] = {
            "role": role,
            "shape": list(next(iter(shapes))),
            "object_count": len(names),
            "objects": names,
            "q4_resident_bytes": q4_bytes,
            "fp8_resident_bytes": fp8_bytes,
            "added_resident_bytes": fp8_bytes - q4_bytes,
        }
    return result


def _median14(values: Sequence[object]) -> float:
    if (
        not isinstance(values, list)
        or len(values) != 14
        or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values)
    ):
        raise ValueError("qualifier must contain exactly 14 finite positive samples per route")
    ordered = sorted(float(x) for x in values)
    return 0.5 * (ordered[6] + ordered[7])


def validate_qualifier(
    report: Mapping[str, object], qualification_id: str
) -> dict[str, float | int]:
    contract = QUALIFICATIONS[qualification_id]
    tokens, rows, columns = contract["shape"]
    if (
        report.get("schema") != QUALIFIER_SCHEMA
        or report.get("artifact_type") != "ninfer_r9700_fp8_projection_comparison"
        or report.get("schema_version") != 1
        or report.get("qualification_id") != qualification_id
        or report.get("pass") is not True
        or report.get("shape") != {"tokens": tokens, "rows": rows, "columns": columns}
        or report.get("fp8_profile") != "E4M3-outer-vec32f-hipBLASLt-top-supported"
        or report.get("q4_control") != "A8Q4G64-m64n128-pingpong-production"
    ):
        raise ValueError(f"{qualification_id}: schema, identity, or fixed shape differs")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{qualification_id}: provenance absent")
    source_root = provenance.get("source_root")
    executable_path = provenance.get("executable_path")
    if not isinstance(source_root, str) or not isinstance(executable_path, str):
        raise ValueError(f"{qualification_id}: provenance paths malformed")
    source = Path(source_root) / "tools/r9700/fp8_gate_up_qual.hip"
    executable = Path(executable_path)
    if (
        not source.is_file()
        or not executable.is_file()
        or provenance.get("source_sha256") != sha256_file(source)
        or provenance.get("executable_sha256") != sha256_file(executable)
    ):
        raise ValueError(f"{qualification_id}: live source or executable hash differs")
    hardware = report.get("hardware")
    power = report.get("power_profile")
    if not isinstance(hardware, Mapping) or (
        hardware.get("device") != "AMD Radeon AI PRO R9700"
        or hardware.get("architecture") != "gfx1201"
        or hardware.get("wave_size") != 32
        or hardware.get("vendor_id") != "0x1002"
        or hardware.get("device_id") != "0x7551"
    ):
        raise ValueError(f"{qualification_id}: hardware identity differs")
    if not isinstance(power, Mapping) or any(
        power.get(field) != "auto" for field in ("required", "before", "after")
    ):
        raise ValueError(f"{qualification_id}: auto power identity differs")
    if any(
        report.get(field) is not True
        for field in (
            "direct_weight_binding", "outer_vector_scales", "nonfinite_status_poisoning",
            "q4_nonfinite_status_poisoning", "no_clobber_rejection",
        )
    ):
        raise ValueError(f"{qualification_id}: correctness/rejection gate failed")
    oracle = report.get("represented_format_axis_oracle")
    expected_rows = [0, rows // 2, rows - 1]
    if (
        not isinstance(oracle, Mapping)
        or oracle.get("probe_count") != 9
        or oracle.get("tokens") != [0, 17, 2047]
        or oracle.get("rows") != expected_rows
        or type(oracle.get("fp8_max_bf16_steps")) is not int
        or not 0 <= oracle["fp8_max_bf16_steps"] <= 1
        or type(oracle.get("q4_max_bf16_steps")) is not int
        or not 0 <= oracle["q4_max_bf16_steps"] <= 2
    ):
        raise ValueError(f"{qualification_id}: represented-format axis oracle differs")
    fp8 = _median14(report.get("fp8_complete_ms_samples"))
    q4 = _median14(report.get("q4_complete_ms_samples"))
    speedup = q4 / fp8
    reported_fp8 = report.get("fp8_complete_median_ms")
    reported_q4 = report.get("q4_complete_median_ms")
    reported_speedup = report.get("fp8_over_q4_speedup")
    if (
        type(reported_fp8) not in (int, float)
        or type(reported_q4) not in (int, float)
        or type(reported_speedup) not in (int, float)
        or report.get("balanced_interleaved_pairs") != 7
        or report.get("trial_order") != "7 paired repetitions, each fp8,q4 then q4,fp8"
        or not math.isclose(float(reported_fp8), fp8, rel_tol=1e-6)
        or not math.isclose(float(reported_q4), q4, rel_tol=1e-6)
        or not math.isclose(float(reported_speedup), speedup, rel_tol=1e-6)
        or report.get("current_q4_implementation_underperforms_fp8") is not (speedup > 1.0)
        or report.get("current_q4_implementation_poor_vs_fp8") is not (speedup > 1.0)
    ):
        raise ValueError(f"{qualification_id}: matched timing set differs")
    algorithm = report.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError(f"{qualification_id}: algorithm identity absent")
    selected_workspace = report.get("selected_workspace_bytes")
    if (
        type(selected_workspace) is not int
        or selected_workspace < 0
        or selected_workspace != algorithm.get("selected_workspace_bytes")
        or selected_workspace > int(algorithm.get("algorithm_max_workspace_bytes", -1))
        or int(algorithm.get("heuristic_returned", 0)) <= 0
    ):
        raise ValueError(f"{qualification_id}: selected algorithm workspace differs")
    return {
        "fp8_complete_median_ms": fp8,
        "q4_complete_median_ms": q4,
        "fp8_over_q4_time_ratio": fp8 / q4,
        "fp8_over_q4_speedup": speedup,
        "selected_workspace_bytes": selected_workspace,
    }


def _select_roles(candidates: Sequence[Mapping[str, object]], budget: int) -> tuple[str, ...]:
    best: tuple[int, int, int, tuple[str, ...]] | None = None
    for count in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, count):
            added = sum(int(item["added_resident_bytes"]) for item in subset)
            if added > budget:
                continue
            service = sum(int(item["q4_measured_service_ns"]) for item in subset)
            roles = tuple(sorted(str(item["role"]) for item in subset))
            shape_count = len({tuple(item["shape"]) for item in subset})
            key = (service, -shape_count, -added, roles)
            if best is None or key > best:
                best = key
    if best is None:
        raise ValueError("no capacity-feasible role selection")
    return best[3]


def decide(
    *, gate: Mapping[str, object], hybrid: Mapping[str, object], capacity: Mapping[str, object],
    qualifiers: Mapping[str, Mapping[str, object]], provenance: Mapping[str, object]
) -> dict[str, object]:
    if (
        gate.get("schema") != GATE_SCHEMA
        or gate.get("status") != "complete"
        or gate.get("decision", {}).get("verdict") != "proceed"
        or hybrid.get("schema") != HYBRID_SCHEMA
        or capacity.get("schema") != CAPACITY_SCHEMA
    ):
        raise ValueError("gate-up, hybrid, or capacity authority differs")
    for name in ("hybrid", "capacity"):
        if gate.get("provenance", {}).get(name, {}).get("sha256") != provenance[name]["sha256"]:
            raise ValueError(f"gate-up decision does not bind supplied {name} authority")

    inventory = exact_role_inventory()
    hybrid_candidates = hybrid.get("ranked_role_candidates")
    if not isinstance(hybrid_candidates, list):
        raise ValueError("hybrid role inventory absent")
    by_role = {str(item.get("role")): item for item in hybrid_candidates}
    if set(by_role) != set(inventory):
        raise ValueError("hybrid roles differ from exact current inventory")
    candidates = []
    for role, exact in inventory.items():
        observed = by_role[role]
        if any(observed.get(field) != exact[field] for field in ("object_count", "objects", "added_resident_bytes")):
            raise ValueError(f"hybrid {role} inventory differs")
        candidates.append({**exact, "q4_measured_service_ns": int(observed["q4_measured_service_ns"])})

    gate_inventory = gate.get("inventory")
    if not isinstance(gate_inventory, Mapping) or any(
        gate_inventory.get(field) != inventory[GATE_ROLE][field]
        for field in ("object_count", "objects", "added_resident_bytes")
    ):
        raise ValueError("gate-up report inventory differs")
    gate_cells = gate.get("capacity", {}).get("cells")
    if not isinstance(gate_cells, list) or len(gate_cells) != 16:
        raise ValueError("gate-up decision lacks 16 capacity cells")
    expected_cells = {(p, g, c) for p in (2048, 8192) for g in (16, 32) for c in range(1, 5)}
    actual_cells = {
        (int(cell["prefill_chunk"]), int(cell["kv_value_group"]), int(cell["concurrency"]))
        for cell in gate_cells
    }
    if actual_cells != expected_cells:
        raise ValueError("gate-up capacity cell identities differ")
    contract = capacity.get("planner_contract")
    if not isinstance(contract, Mapping) or (
        contract.get("speculative_backend") != "mtp"
        or contract.get("draft_tokens") != 3
        or contract.get("device_graph") is not True
        or contract.get("max_context") != 262144
        or contract.get("automatic_headroom_bytes") != 1 << 30
    ):
        raise ValueError("capacity report differs from the active MTP3 Device-Graph envelope")
    direct_budgets = {}
    for scenario in capacity.get("scenarios", []):
        chunk = int(scenario["workspace"]["prefill_chunk"])
        for group in scenario["groups"]:
            for cell in group["cells"]:
                key = (chunk, int(group["kv_value_group"]), int(cell["concurrency"]))
                direct_budgets[key] = (
                    int(contract["device_capacity_bytes"])
                    - int(contract["automatic_headroom_bytes"])
                    - q4_inventory.DEVICE_ARENA_BYTES
                    - int(cell["projected_minimum_runtime_reservation_bytes"])
                )
    if set(direct_budgets) != expected_cells:
        raise ValueError("capacity authority lacks the exact 16 cells")
    for cell in gate_cells:
        key = (int(cell["prefill_chunk"]), int(cell["kv_value_group"]), int(cell["concurrency"]))
        maximum = direct_budgets[key]
        if (
            int(cell["maximum_fp8_over_q4_bytes"]) != maximum
            or int(cell["capacity_slack_bytes"])
            != maximum - int(inventory[GATE_ROLE]["added_resident_bytes"])
        ):
            raise ValueError("gate-up capacity arithmetic differs from direct planner authority")
    minimum_slack = min(int(cell["capacity_slack_bytes"]) for cell in gate_cells)
    non_gate = [item for item in candidates if item["role"] != GATE_ROLE]
    selected = _select_roles(non_gate, minimum_slack)
    if selected != SELECTED_ROLES:
        raise ValueError(f"capacity-optimal fixed role set changed: {selected}")

    measurements = {
        qualification_id: validate_qualifier(qualifiers[qualification_id], qualification_id)
        for qualification_id in QUALIFICATIONS
    }
    selected_items = [inventory[role] | {"q4_measured_service_ns": by_role[role]["q4_measured_service_ns"]} for role in selected]
    added = sum(int(item["added_resident_bytes"]) for item in selected_items)
    total_added = int(inventory[GATE_ROLE]["added_resident_bytes"]) + added
    roles = []
    additional_saving = 0.0
    reasons = []
    for item in selected_items:
        role = str(item["role"])
        qualification_id = next(
            key for key, value in QUALIFICATIONS.items() if role in value["roles"]
        )
        timing = measurements[qualification_id]
        q4_service = int(item["q4_measured_service_ns"])
        projected_fp8 = q4_service * float(timing["fp8_over_q4_time_ratio"])
        saving = q4_service - projected_fp8
        additional_saving += saving
        if timing["fp8_over_q4_time_ratio"] >= 1.0:
            reasons.append(f"{qualification_id}_matched_fp8_path_is_not_faster")
        roles.append({**item, "qualification_id": qualification_id, "matched_measurement": timing,
                      "projected_fp8_service_ns": projected_fp8, "projected_saving_ns": saving})
    capacity_cells = []
    for cell in gate_cells:
        maximum = int(cell["maximum_fp8_over_q4_bytes"])
        slack = maximum - total_added
        capacity_cells.append({
            "prefill_chunk": int(cell["prefill_chunk"]),
            "kv_value_group": int(cell["kv_value_group"]),
            "concurrency": int(cell["concurrency"]),
            "maximum_fp8_over_q4_bytes": maximum,
            "total_selected_added_resident_bytes": total_added,
            "capacity_slack_bytes": slack,
            "capacity_preserved": slack >= 0,
        })
    if any(not cell["capacity_preserved"] for cell in capacity_cells):
        reasons.append("selected_roles_exceed_at_least_one_capacity_envelope")
    gate_projection = gate.get("whole_p2048_projection")
    if not isinstance(gate_projection, Mapping):
        raise ValueError("gate-up projection absent")
    projected_prefill = float(gate_projection["projected_prefill_seconds"]) - additional_saving / 1e9
    projected_total = float(gate_projection["projected_total_seconds"]) - additional_saving / 1e9
    if additional_saving <= 0:
        reasons.append("additional_roles_do_not_improve_projected_whole")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "gate_up_plus_capacity_optimal_fixed_projection_roles",
        "selection_contract": {
            "objective": "maximum measured Q4 service under the tightest post-gate-up 16-cell slack",
            "tie_break": "fewest qualifier shapes, then least added bytes, then lexical roles",
            "tightest_post_gate_up_slack_bytes": minimum_slack,
            "selected_roles": list(selected),
            "required_qualification_ids": sorted(QUALIFICATIONS),
        },
        "inventory": {
            "gate_up_added_resident_bytes": inventory[GATE_ROLE]["added_resident_bytes"],
            "additional_added_resident_bytes": added,
            "total_selected_added_resident_bytes": total_added,
            "additional_object_count": sum(int(item["object_count"]) for item in selected_items),
            "roles": roles,
        },
        "qualifications": measurements,
        "capacity": {
            "all_cells_preserve_minimum_startup": all(cell["capacity_preserved"] for cell in capacity_cells),
            "cells": sorted(capacity_cells, key=lambda x: (x["prefill_chunk"], x["kv_value_group"], x["concurrency"])),
        },
        "whole_p2048_projection": {
            "gate_up_projected_prefill_seconds": gate_projection["projected_prefill_seconds"],
            "gate_up_projected_total_seconds": gate_projection["projected_total_seconds"],
            "additional_measured_q4_service_ns": sum(int(item["q4_measured_service_ns"]) for item in selected_items),
            "additional_projected_fp8_service_ns": sum(float(item["projected_fp8_service_ns"]) for item in roles),
            "additional_projected_saving_ns": additional_saving,
            "projected_prefill_seconds": projected_prefill,
            "projected_prefill_tok_s": TOKENS / projected_prefill,
            "projected_total_seconds": projected_total,
            "projected_whole_p2048_tok_s": TOKENS / projected_total,
        },
        "decision": {
            "verdict": "proceed" if not reasons else "reject",
            "gate": "both fixed-shape FP8 paths beat Q4, all 16 cells retain minimum startup, and projected whole improves",
            "reasons": reasons,
        },
        "provenance": dict(provenance),
    }


def canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gate-decision", "hybrid", "capacity", *QUALIFICATIONS):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in ("gate-decision", "hybrid", "capacity")}
    paths.update({name: getattr(args, name) for name in QUALIFICATIONS})
    expected = {name: getattr(args, f"{name}_sha256") for name in paths}
    loaded = {name: _load_bound(path, expected[name]) for name, path in paths.items()}
    provenance = {name: {"path": str(path), "sha256": expected[name]} for name, path in paths.items()}
    report = decide(
        gate=loaded["gate_decision"], hybrid=loaded["hybrid"], capacity=loaded["capacity"],
        qualifiers={name: loaded[name] for name in QUALIFICATIONS}, provenance=provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_json(report))
    print(json.dumps(report["decision"], sort_keys=True))
    return 0 if report["decision"]["verdict"] == "proceed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
