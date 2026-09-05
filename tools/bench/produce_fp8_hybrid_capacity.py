#!/usr/bin/env python3
"""Produce or validate the CPU-only current hybrid P8192/G16 capacity report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Mapping

SCHEMA = "ninfer.r9700.fp8-hybrid-current-capacity.v1"
HEADROOM = 1 << 30
PAGE_TOKENS = 64


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def parse_planner(text: str) -> list[dict[str, int]]:
    rows = [{key: int(value) for key, value in row.items()}
            for row in csv.DictReader(io.StringIO(text))]
    if [row["concurrency"] for row in rows] != [1, 2, 3, 4]:
        raise ValueError("planner capacity authority lacks exact C1..4 rows")
    if any(row["minimum_groups"] != 4096 or row["kv_increment_bytes"] != 1_810_432
           for row in rows):
        raise ValueError("planner page geometry differs from P262144/G16")
    return rows


def assemble(rows: list[dict[str, int]], weights: int, device: int) -> dict[str, object]:
    budget = device - weights - HEADROOM
    cells = []
    for row in rows:
        minimum = row["minimum_groups"]
        maximum = row["maximum_groups"]
        reservation = row["minimum_reservation_bytes"]
        increment = row["kv_increment_bytes"]
        if reservation > budget:
            groups = None
            resolved = None
            sequence = None
            slack = budget - reservation
        else:
            groups = min(maximum, minimum + (budget - reservation) // increment)
            resolved = reservation + (groups - minimum) * increment
            sequence = row["minimum_sequence_bytes"] + (groups - minimum) * increment
            slack = budget - resolved
        cells.append({**row, "resolved_groups": groups,
                      "aggregate_capacity_tokens": None if groups is None else groups * PAGE_TOKENS,
                      "resolved_sequence_bytes": sequence,
                      "resolved_reservation_bytes": resolved, "remaining_slack_bytes": slack,
                      "capacity_preserved": slack >= 0})
    return {"schema": SCHEMA,
            "scope": {"prefill_chunk": 8192, "kv_value_group": 16,
                      "speculative_backend": "mtp", "draft_tokens": 3,
                      "device_graph": True, "automatic_headroom_bytes": HEADROOM},
            "weights_capacity_bytes": weights, "device_capacity_bytes": device,
            "runtime_budget_after_weights_and_headroom_bytes": budget,
            "cells": cells,
            "prior_c4_slack": {"retained_claim_bytes": 2_004_481,
                               "current_derived_bytes": cells[-1]["remaining_slack_bytes"],
                               "confirmed_exact": cells[-1]["remaining_slack_bytes"] == 2_004_481}}


def current_report(planner: Path, weight_report: Path) -> dict[str, object]:
    completed = subprocess.run([str(planner), "--host-hybrid-capacity-csv"], check=True,
                               capture_output=True, text=True)
    source = json.loads(weight_report.read_text())
    memory = source["memory"]
    weights = int(memory["weights"]["capacity_bytes"])
    device = weights + int(memory["available_after_weights_bytes"])
    report = assemble(parse_planner(completed.stdout), weights, device)
    report["provenance"] = {
        "planner_executable": {"path": str(planner), "sha256": sha256(planner)},
        "weights_report": {"path": str(weight_report), "sha256": sha256(weight_report)},
    }
    return report


def canonical(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--weights-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    expected = current_report(args.planner, args.weights_report)
    if args.validate:
        if json.loads(args.output.read_text()) != expected:
            raise ValueError("hybrid capacity report differs from current planner/provenance")
    else:
        args.output.write_text(canonical(expected))
    print(json.dumps(expected["prior_c4_slack"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
