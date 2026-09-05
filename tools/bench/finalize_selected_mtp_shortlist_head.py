#!/usr/bin/env python3
"""Finalize the fixed terminal-winner MTP trace into executed-head evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.bench.extract_embedded_code_object import extract
from tools.bench.produce_mtp_shortlist_head_evidence import (
    CODE_SYMBOL,
    DISPLAY_SYMBOL,
    HEAD_GRID_X,
    _load,
    _publish,
    produce,
)
from tools.bench.validate_profile_trace import _parse_database


def exact_trace_signature(database: Path, command: list[str]) -> tuple[str, list[str]]:
    dispatches, _ = _parse_database(database, " ".join(command))
    rows = [
        row for row in dispatches
        if row["grid"] == {"x": HEAD_GRID_X, "y": 1, "z": 1}
        and row["symbol"] == DISPLAY_SYMBOL
    ]
    if not rows:
        raise ValueError("trace contains no executed shortlist-head dispatch")
    symbols = {row["symbol"] for row in rows}
    stages = {row["roctx_region"] for row in rows}
    if len(symbols) != 1 or not stages or any(not isinstance(stage, str) or not stage for stage in stages):
        raise ValueError("trace does not provide one exact symbol and nonempty stage set")
    return next(iter(symbols)), sorted(stages)


def finalize(package: Path) -> dict:
    package = package.resolve(strict=True)
    plan_path = package / "plan.json"
    plan = _load(plan_path, "selected MTP shortlist-head plan")
    terminal = plan.get("terminal_selection")
    if not isinstance(terminal, dict) or not isinstance(terminal.get("path"), str):
        raise ValueError("plan lacks terminal selection binding")
    database = Path(str(plan.get("expected_trace_database", ""))).resolve(strict=True)
    benchmark_report = Path(plan["benchmark_command"][plan["benchmark_command"].index("--output-file") + 1])
    power = plan["required_power_profile"]
    executable = Path(plan["benchmark_executable"]["path"])
    artifact = Path(plan["artifact"]["path"])
    code_object = package / "selected-mtp-shortlist-head.hsaco"
    evidence = package / "shortlist-head-evidence.json"
    if os.path.lexists(code_object) or os.path.lexists(evidence):
        raise ValueError("refusing to overwrite shortlist-head finalization output")
    symbol, stages = exact_trace_signature(database, plan["benchmark_command"])
    try:
        extract(executable, code_object, code_symbol=CODE_SYMBOL)
        result = produce(
            plan_path=plan_path, benchmark_report=benchmark_report, trace_database=database,
            power_before=Path(power["before_evidence"]), power_after=Path(power["after_evidence"]),
            terminal_selection=Path(terminal["path"]), artifact=artifact, executable=executable,
            code_object=code_object, dispatch_symbol=symbol, stages=stages,
        )
        _publish(evidence, result)
    except BaseException:
        # Extraction is an intermediate of the evidence transaction.  A failed
        # validation must not strand it and make the immutable package impossible
        # to retry with corrected physical inputs.
        if not os.path.lexists(evidence):
            try:
                code_object.unlink()
            except FileNotFoundError:
                pass
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = finalize(args.package)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {result['operation_family']} evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
