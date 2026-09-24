#!/usr/bin/env python3
"""Fail-closed analysis of the exact K5120 rows5/6 RMSNorm screen."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

PACKAGE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError("report must be an object")
    return value


def finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("timing value is not finite numeric")
    return float(value)


def arm(cell: dict, name: str) -> list[float]:
    value = cell.get(name)
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"{name} must contain seven samples")
    result = [finite(x) for x in value]
    if any(x <= 0.0 for x in result):
        raise ValueError(f"{name} contains a nonpositive sample")
    return result


def derive(cell: dict) -> dict:
    if set(cell) != {"features", "rows", "unit_offset", "epsilon", "iterations_per_sample",
                    "forward_reverse_pairs", "incumbent_forward_ms", "challenger_forward_ms",
                    "challenger_reverse_ms", "incumbent_reverse_ms"}:
        raise ValueError("cell fields differ")
    if cell["features"] != 5120 or cell["rows"] not in (5, 6) or cell["unit_offset"] is not True:
        raise ValueError("cell scope differs")
    if not math.isclose(finite(cell["epsilon"]), 1.0e-6, rel_tol=1e-7, abs_tol=0.0):
        raise ValueError("timing epsilon differs")
    if cell["iterations_per_sample"] != 256 or cell["forward_reverse_pairs"] != 7:
        raise ValueError("timing geometry differs")
    i_f = arm(cell, "incumbent_forward_ms")
    c_f = arm(cell, "challenger_forward_ms")
    c_r = arm(cell, "challenger_reverse_ms")
    i_r = arm(cell, "incumbent_reverse_ms")
    f_ratio = [c / i for c, i in zip(c_f, i_f)]
    r_ratio = [c / i for c, i in zip(c_r, i_r)]
    ratios = f_ratio + r_ratio
    mean = statistics.mean(ratios)
    deviation = statistics.stdev(ratios)
    upper = mean + 2.0 * deviation / math.sqrt(len(ratios))
    f_median, r_median = statistics.median(f_ratio), statistics.median(r_ratio)
    accepted = (statistics.median(c_f) < statistics.median(i_f) and
                statistics.median(c_r) < statistics.median(i_r) and
                f_median < 1.0 and r_median < 1.0 and upper < 1.0 and
                abs(f_median - r_median) <= 0.02)
    return {
        "rows": cell["rows"],
        "incumbent_median_ms": statistics.median([(a + b) / 2 for a, b in zip(i_f, i_r)]),
        "challenger_median_ms": statistics.median([(a + b) / 2 for a, b in zip(c_f, c_r)]),
        "forward_ratio_median": f_median,
        "reverse_ratio_median": r_median,
        "paired_ratio_mean": mean,
        "paired_ratio_standard_deviation": deviation,
        "paired_ratio_upper": upper,
        "order_ratio_delta": abs(f_median - r_median),
        "accepted": accepted,
    }


def exclusive_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def analyze(report: dict) -> dict:
    if set(report) != {"schema", "production_dispatch_changed", "numeric", "cells"}:
        raise ValueError("report fields differ")
    if report["schema"] != "ninfer.r9700.rmsnorm-k5120-rows56-cell.v1" or \
            report["production_dispatch_changed"] is not False:
        raise ValueError("schema or production isolation differs")
    numeric = report["numeric"]
    if not isinstance(numeric, dict) or numeric.get("oracle") != \
            "independent FP64 represented-BF16 RMSNorm formula" or \
            numeric.get("maximum_bf16_steps_allowed") != 2:
        raise ValueError("oracle contract differs")
    for key in ("incumbent_maximum_bf16_steps", "challenger_maximum_bf16_steps"):
        value = numeric.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2:
            raise ValueError("oracle result exceeds criterion")
    cells = report["cells"]
    if not isinstance(cells, list) or len(cells) != 2:
        raise ValueError("expected exactly two cells")
    decisions = [derive(cell) for cell in cells]
    if [entry["rows"] for entry in decisions] != [5, 6]:
        raise ValueError("exact row order differs")
    eligible = [entry["rows"] for entry in decisions if entry["accepted"]]
    return {
        "schema": "ninfer.r9700.rmsnorm-k5120-rows56-summary.v1",
        "status": "screen-complete",
        "production_dispatch_changed": False,
        "eligible_rows": eligible,
        "forbidden_rows": [row for row in (5, 6) if row not in eligible],
        "decision_criterion": "both launch-order medians faster; paired-ratio 2SE upper below one; order-ratio delta at most 0.02",
        "cells": decisions,
        "limitations": ["standalone exact-shape screen only", "no production routing or whole-DFlash claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.report != PACKAGE / "cells.json" or args.summary != PACKAGE / "summary.json":
        raise ValueError("paths differ from package contract")
    exclusive_json(args.summary, analyze(load(args.report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
