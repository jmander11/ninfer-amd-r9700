#!/usr/bin/env python3
"""Deterministic R9700 planner-capacity projection for the E4M3 evaluation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from tools.convert.qwen3_8_27b_r9700 import e4m3_inventory


SCHEMA = "ninfer.r9700.e4m3-capacity-analysis.v1"
PAGE_TOKENS = 64
MAX_CONTEXT = 262_144
MINIMUM_PAGE_GROUPS = MAX_CONTEXT // PAGE_TOKENS
HEADROOM_BYTES = 1 << 30
K_MAX = 17_408
N_MAX = 34_816
ARENA_ALIGNMENT = 256


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _align_up(value: int, alignment: int = ARENA_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def q4_a8_workspace(tokens: int) -> int:
    """Mirror the admitted A8G64 two-plane activation workspace geometry."""

    padded_k = _align_up(K_MAX, 128)
    plane = tokens * padded_k // 2
    scales = tokens * (padded_k // 64) * 2
    high_offset = _align_up(plane)
    scale_offset = _align_up(high_offset + plane)
    status_offset = _align_up(scale_offset + scales)
    return status_offset + 4


def fp8_linear_workspace(tokens: int) -> dict[str, int]:
    activation = tokens * _align_up(K_MAX, 128) + tokens * 4
    hipblaslt_cap = tokens * N_MAX * 4
    return {
        "activation_bytes": activation,
        "hipblaslt_conservative_cap_bytes": hipblaslt_cap,
        "linear_scratch_bytes": activation + hipblaslt_cap,
    }


def _workspace_by_chunk(summary: Sequence[Mapping[str, object]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for row in summary:
        chunk = int(row["prefill_chunk"])
        workspace = int(row["workspace_capacity_bytes"])
        previous = result.setdefault(chunk, workspace)
        if previous != workspace:
            raise ValueError(f"prefill chunk {chunk} has inconsistent planner workspaces")
    return result


def project_workspace(chunk: int, incumbent_workspace: int) -> dict[str, int]:
    incumbent_linear = q4_a8_workspace(chunk)
    alignment_allowance = ARENA_ALIGNMENT - 1
    phase_without_linear = incumbent_workspace - incumbent_linear - alignment_allowance
    if phase_without_linear < 0:
        raise ValueError("incumbent planner workspace is smaller than its linear scope")
    fp8 = fp8_linear_workspace(chunk)
    projected = phase_without_linear + fp8["linear_scratch_bytes"] + alignment_allowance
    return {
        "prefill_chunk": chunk,
        "incumbent_planner_workspace_bytes": incumbent_workspace,
        "incumbent_a8q4_linear_scratch_bytes": incumbent_linear,
        "planner_phase_bytes_excluding_linear_scratch": phase_without_linear,
        **fp8,
        "arena_alignment_allowance_bytes": alignment_allowance,
        "projected_planner_workspace_bytes": projected,
    }


def project_cell(
    source: Mapping[str, object], *, value_group: int, workspace_bytes: int,
    resident_weight_bytes: int
) -> dict[str, object]:
    memory = source["memory"]
    if not isinstance(memory, Mapping):
        raise ValueError("planner source has no memory object")
    config = source.get("config", source)
    if not isinstance(config, Mapping):
        raise ValueError("planner source has no config object")
    concurrency = int(config["concurrency"])
    if (
        config.get("spec") != "mtp"
        or int(config.get("draft_tokens", 0)) != 3
        or config.get("use_device_graph") is not True
        or int(config.get("kv_value_group", 0)) != value_group
    ):
        raise ValueError("planner source differs from the MTP3 Device-Graph cache profile")
    if int(memory["max_context"]) != MAX_CONTEXT:
        raise ValueError("planner source does not use the 262144-token envelope")
    if memory["kv_capacity_mode"] != "auto":
        raise ValueError("planner source is not automatic capacity")
    minimum_pages = MINIMUM_PAGE_GROUPS
    maximum_pages = concurrency * minimum_pages
    if int(memory["kv_capacity_max_page_groups"]) != maximum_pages:
        raise ValueError("planner maximum page-group count differs")

    incumbent_workspace = int(memory["workspace"]["capacity_bytes"])
    graph = int(memory["device_graph_allowance_bytes"])
    request_transient = int(memory["request_transient"]["capacity_bytes"])
    minimum_reservation = int(memory["minimum_runtime_reservation_bytes"])
    sequence_minimum = minimum_reservation - incumbent_workspace - graph - request_transient
    stride = int(memory["kv_capacity_increment_bytes"])
    resolved_pages = int(memory["kv_capacity_page_groups"])
    expected_sequence = sequence_minimum + (resolved_pages - minimum_pages) * stride
    if expected_sequence != int(memory["sequence"]["capacity_bytes"]):
        raise ValueError("planner source affine sequence decomposition differs")

    device_bytes = int(memory["weights"]["capacity_bytes"]) + int(
        memory["available_after_weights_bytes"]
    )
    available = device_bytes - resident_weight_bytes
    capacity_budget = available - HEADROOM_BYTES
    projected_minimum = sequence_minimum + workspace_bytes + graph + request_transient
    if projected_minimum > capacity_budget:
        pages = None
        tokens = None
        per_lane = None
        remainder = None
        status = "startup_inadmissible_minimum_context"
    else:
        if minimum_pages == maximum_pages:
            pages = minimum_pages
        else:
            pages = min(
                maximum_pages,
                minimum_pages + (capacity_budget - projected_minimum) // stride,
            )
        tokens = pages * PAGE_TOKENS
        per_lane = (pages // concurrency) * PAGE_TOKENS
        remainder = (pages % concurrency) * PAGE_TOKENS
        status = "model_context_limited" if pages == maximum_pages else "device_memory_limited"

    return {
        "concurrency": concurrency,
        "status": status,
        "minimum_main_page_groups": minimum_pages,
        "maximum_main_page_groups": maximum_pages,
        "bytes_per_additional_main_page_group": stride,
        "sequence_minimum_bytes": sequence_minimum,
        "device_graph_allowance_bytes": graph,
        "request_transient_bytes": request_transient,
        "projected_minimum_runtime_reservation_bytes": projected_minimum,
        "available_after_weights_bytes": available,
        "automatic_headroom_bytes": HEADROOM_BYTES,
        "runtime_capacity_budget_bytes": capacity_budget,
        "minimum_startup_shortfall_bytes": max(0, projected_minimum - capacity_budget),
        "resolved_main_page_groups": pages,
        "aggregate_capacity_tokens": tokens,
        "equal_share_per_lane_tokens": per_lane,
        "equal_share_remainder_tokens": remainder,
    }


def assemble_report(
    *,
    workspace_summary: Sequence[Mapping[str, object]],
    planner_sources: Mapping[int, Mapping[int, Mapping[str, object]]],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    workspace_by_chunk = _workspace_by_chunk(workspace_summary)
    scenarios = []
    for chunk in (2048, 8192):
        if chunk not in workspace_by_chunk:
            raise ValueError(f"planner summary lacks P{chunk} workspace")
        workspace = project_workspace(chunk, workspace_by_chunk[chunk])
        groups = []
        for value_group in (16, 32):
            cells = [
                project_cell(
                    planner_sources[value_group][concurrency],
                    value_group=value_group,
                    workspace_bytes=workspace["projected_planner_workspace_bytes"],
                    resident_weight_bytes=e4m3_inventory.DEVICE_ARENA_BYTES,
                )
                for concurrency in range(1, 5)
            ]
            groups.append({"kv_value_group": value_group, "cells": cells})
        scenarios.append({"workspace": workspace, "groups": groups})

    device_capacities = {
        int(source["memory"]["weights"]["capacity_bytes"])
        + int(source["memory"]["available_after_weights_bytes"])
        for by_concurrency in planner_sources.values()
        for source in by_concurrency.values()
    }
    if len(device_capacities) != 1:
        raise ValueError("planner sources disagree on physical device capacity")
    device_capacity = device_capacities.pop()

    return {
        "schema": SCHEMA,
        "candidate": {
            "model_id": e4m3_inventory.MODEL_ID,
            "weights_id": e4m3_inventory.WEIGHTS_ID,
            "tensor_payload_bytes": e4m3_inventory.TENSOR_ENCODED_BYTES,
            "resident_device_arena_bytes": e4m3_inventory.DEVICE_ARENA_BYTES,
            "object_alignment_bytes": ARENA_ALIGNMENT,
            "arena_alignment_overhead_bytes": (
                e4m3_inventory.DEVICE_ARENA_BYTES - e4m3_inventory.TENSOR_ENCODED_BYTES
            ),
        },
        "planner_contract": {
            "speculative_backend": "mtp",
            "draft_tokens": 3,
            "device_graph": True,
            "max_context": MAX_CONTEXT,
            "page_tokens": PAGE_TOKENS,
            "device_capacity_bytes": device_capacity,
            "available_after_resident_weights_bytes": (
                device_capacity - e4m3_inventory.DEVICE_ARENA_BYTES
            ),
            "automatic_headroom_bytes": HEADROOM_BYTES,
            "hipblaslt_workspace_kind": "conservative_cap_pending_algorithm_selection",
        },
        "scenarios": scenarios,
        "provenance": dict(provenance),
    }


def canonical_json(report: Mapping[str, object]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    *, workspace_summary_path: Path, capacity_root: Path, output: Path
) -> dict[str, object]:
    workspace_summary = _read_json(workspace_summary_path)
    sources: dict[int, dict[int, Mapping[str, object]]] = {}
    source_records = []
    for group in (16, 32):
        sources[group] = {}
        for concurrency in range(1, 5):
            path = (
                capacity_root
                / f"pareto-capacity-dense-all-q4-g{group}-20260903"
                / "json/pareto_effective_capacity"
                / f"c{concurrency}/effective_capacity_mtp3.json"
            )
            sources[group][concurrency] = _read_json(path)
            source_records.append(
                {"path": str(path), "sha256": _sha256(path), "kv_value_group": group,
                 "concurrency": concurrency}
            )
    report = assemble_report(
        workspace_summary=workspace_summary,
        planner_sources=sources,
        provenance={
            "workspace_summary": {
                "path": str(workspace_summary_path),
                "sha256": _sha256(workspace_summary_path),
            },
            "capacity_reports": source_records,
            "inventory_source": {
                "path": "tools/convert/qwen3_8_27b_r9700/e4m3_inventory.py",
                "sha256": _sha256(Path(e4m3_inventory.__file__)),
            },
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_json(report))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-summary", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, default=Path("profiles/bench"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        workspace_summary_path=args.workspace_summary,
        capacity_root=args.capacity_root,
        output=args.output,
    )
    print(json.dumps({
        f"p{scenario['workspace']['prefill_chunk']}": [
            {"g": group["kv_value_group"], "statuses": [cell["status"] for cell in group["cells"]]}
            for group in scenario["groups"]
        ]
        for scenario in report["scenarios"]
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
