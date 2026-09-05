#!/usr/bin/env python3
"""Prepare a fresh rocprofv3 trace for a fixed FP8-vs-Q4 qualifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Sequence


QUALIFICATIONS = {
    "gate_up": {
        "plan_schema": "ninfer.r9700.fp8_gate_up_hardware_proof_plan.v1",
        "shape": {"tokens": 2048, "rows": 34816, "columns": 5120},
        "output_stem": "gate_up_hardware",
    },
    "attention_qk_gate_value": {
        "plan_schema": "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof_plan.v1",
        "shape": {"tokens": 2048, "rows": 7168, "columns": 5120},
        "output_stem": "attention_qk_gate_value_hardware",
    },
}


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as source:
        return {"path": str(resolved), "bytes": resolved.stat().st_size,
                "sha256": hashlib.file_digest(source, "sha256").hexdigest()}


def prepare(qualifier: Path, admission_report: Path, report: Path, trace_dir: Path,
            capture_dir: Path, capture_library: Path, plan_path: Path,
            rocprofv3: Path, qualification: str = "gate_up") -> dict[str, object]:
    try:
        profile = QUALIFICATIONS[qualification]
    except KeyError as error:
        raise ValueError(f"unsupported qualification: {qualification}") from error
    qualifier_id = identity(qualifier)
    profiler_id = identity(rocprofv3)
    admission_id = identity(admission_report)
    capture_id = identity(capture_library)
    report = report.absolute()
    trace_dir = trace_dir.absolute()
    capture_dir = capture_dir.absolute()
    plan_path = plan_path.absolute()
    if report == trace_dir or trace_dir in report.parents or report in trace_dir.parents:
        raise ValueError("qualifier report and profiler output tree must be disjoint")
    if (capture_dir == trace_dir or trace_dir in capture_dir.parents or
            capture_dir in trace_dir.parents):
        raise ValueError("code-object capture and profiler output trees must be disjoint")
    for path, label in ((report, "report"), (trace_dir, "trace directory"),
                        (capture_dir, "capture directory"),
                        (plan_path, "plan")):
        if os.path.lexists(path):
            raise ValueError(f"refusing to reuse existing {label}: {path}")
    command = ["/usr/bin/env", f"LD_PRELOAD={capture_id['path']}",
               f"NINFER_CODE_OBJECT_CAPTURE_DIR={capture_dir}",
               profiler_id["path"], "-d", str(trace_dir), "-o", profile["output_stem"],
               "-f", "rocpd", "--kernel-trace", "--", qualifier_id["path"],
               "--output", str(report)]
    plan = {
        "schema": profile["plan_schema"],
        "qualification": qualification,
        "shape": profile["shape"],
        "qualifier": qualifier_id,
        "admission_report": admission_id,
        "rocprofv3": profiler_id,
        "capture_library": capture_id,
        "fresh_capture_directory": str(capture_dir),
        "fresh_report": str(report),
        "fresh_trace_directory": str(trace_dir),
        "command": command,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    trace_dir.parent.mkdir(parents=True, exist_ok=True)
    capture_dir.parent.mkdir(parents=True, exist_ok=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, sort_keys=True)
        output.write("\n")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualifier", required=True, type=Path)
    parser.add_argument("--admission-report", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--capture-library", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--rocprofv3", type=Path, default=Path("/opt/rocm/bin/rocprofv3"))
    parser.add_argument("--qualification", choices=tuple(QUALIFICATIONS), default="gate_up")
    args = parser.parse_args(argv)
    plan = prepare(args.qualifier, args.admission_report, args.report, args.trace_dir,
                   args.capture_dir, args.capture_library, args.plan, args.rocprofv3,
                   args.qualification)
    print(shlex.join(plan["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
