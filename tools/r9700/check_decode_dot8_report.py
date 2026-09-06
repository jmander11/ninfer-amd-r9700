#!/usr/bin/env python3
"""Validate and independently recompute an A8Q4 T=1 dot8 qualification report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

FACTOR = 4.4478
RATIO_LIMIT = 1.01
SAVING_FLOOR_MS = 5.0
INVENTORY = {
    (4096, 5120): 48,
    (5120, 6144): 64,
    (5120, 17408): 64,
    (7168, 5120): 32,
    (12288, 5120): 48,
    (34816, 5120): 64,
    (248320, 5120): 1,
}


def _close(lhs: float, rhs: float) -> bool:
    return math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def validate(path: Path) -> bool:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 2:
        raise ValueError("decode dot8 report must use schema_version 2")
    rows = report.get("shapes")
    if not isinstance(rows, list) or len(rows) != len(INVENTORY):
        raise ValueError("decode dot8 report must contain the exact seven-shape inventory")
    seen: set[tuple[int, int]] = set()
    every_cell = True
    weighted_saving_lower = 0.0
    for row in rows:
        key = (row.get("rows"), row.get("columns"))
        if key not in INVENTORY or key in seen or row.get("tokens") != 1:
            raise ValueError("decode dot8 report has an unknown, duplicate, or non-T1 cell")
        seen.add(key)
        if row.get("exact_decode_calls") != INVENTORY[key]:
            raise ValueError("decode dot8 report has an incorrect exact-call weight")
        incumbent = row.get("wmma_samples_ms")
        challenger = row.get("dot8_samples_ms")
        if (not isinstance(incumbent, list) or not isinstance(challenger, list) or
                len(incumbent) != 7 or len(challenger) != 7 or
                not all(math.isfinite(x) and x > 0.0 for x in incumbent + challenger)):
            raise ValueError("each decode dot8 cell requires seven positive finite samples")
        incumbent_median = statistics.median(incumbent)
        challenger_median = statistics.median(challenger)
        incumbent_uncertainty = FACTOR * statistics.median(
            abs(x - incumbent_median) for x in incumbent)
        challenger_uncertainty = FACTOR * statistics.median(
            abs(x - challenger_median) for x in challenger)
        incumbent_lower = incumbent_median - incumbent_uncertainty
        challenger_upper = challenger_median + challenger_uncertainty
        upper_ratio = (challenger_upper / incumbent_lower if incumbent_lower > 0.0
                       else float.fromhex("0x1.fffffffffffffp+1023"))
        for field, expected in (("wmma_ms", incumbent_median),
                                ("dot8_ms", challenger_median),
                                ("robust_upper_ratio", upper_ratio)):
            if not _close(float(row.get(field, math.nan)), expected):
                raise ValueError(f"decode dot8 report has inconsistent {field}")
        every_cell = every_cell and upper_ratio <= RATIO_LIMIT
        weighted_saving_lower += INVENTORY[key] * (incumbent_lower - challenger_upper)
    decision = report.get("performance_decision", {})
    aggregate = weighted_saving_lower >= SAVING_FLOOR_MS
    accepted = (report.get("correctness", {}).get("passed") is True and
                report.get("static_and_resources", {}).get("runtime_resources_passed") is True and
                every_cell and aggregate)
    if (decision.get("every_cell_passed") is not every_cell or
            decision.get("aggregate_passed") is not aggregate or
            decision.get("accepted") is not accepted or
            not _close(float(decision.get("exact_call_weighted_saving_lower_ms_per_token",
                                          math.nan)), weighted_saving_lower) or
            report.get("status") != ("passed" if accepted else "rejected")):
        raise ValueError("decode dot8 report has an inconsistent terminal decision")
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    accepted = validate(args.report)
    print(f"decode_dot8_report=valid status={'passed' if accepted else 'rejected'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
