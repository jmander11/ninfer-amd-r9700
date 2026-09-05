#!/usr/bin/env python3
"""Validate the terminal winner's prepared C1 P2048 trace without inferring roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tools.bench.analyze_whole_profile import analyze
from tools.bench.prepare_whole_profile import ROCPROFV3
from tools.bench.reconcile_qwen3_8_27b_dispatches import _publish, _snapshot
from tools.bench.run_ninfer_bench_matrix import REPORT_SCHEMA_VERSION
from tools.bench.validate_profile_trace import _parse_database
from tools.bench.verify_selected_hardware_use import selected_route
from tools.bench.selected_loaded_code_objects import selected_loaded_fp8


HYBRID_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
def _same(value: object, expected: dict[str, Any], label: str) -> None:
    if (not isinstance(value, dict)
            or value.get("sha256") != expected["sha256"]
            or value.get("file_size_bytes") != expected["file_size_bytes"]
            or _snapshot(Path(str(value.get("path", ""))), label) != expected):
        raise ValueError(f"{label} differs from the selected trace plan")


def validate(plan_path: Path, report_path: Path, database_path: Path,
             power_before_path: Path, power_after_path: Path,
             selection_path: Path, artifact_path: Path, executable_path: Path,
             corpus_path: Path) -> dict[str, Any]:
    snapshots = {name: _snapshot(path, name) for name, path in (
        ("plan", plan_path), ("benchmark_report", report_path), ("database", database_path),
        ("power_before", power_before_path), ("power_after", power_after_path),
        ("terminal_selection", selection_path), ("artifact", artifact_path),
        ("benchmark_executable", executable_path), ("corpus", corpus_path),
    )}
    route = selected_route(selection_path)
    plan = json.loads(Path(snapshots["plan"]["path"]).read_text(encoding="utf-8"))
    workload = plan.get("workload") if isinstance(plan, dict) else None
    command = plan.get("benchmark_command") if isinstance(plan, dict) else None
    if (plan.get("artifact_type") != "ninfer_whole_profile_plan"
            or plan.get("schema_version") != 2 or plan.get("profile_kind") != "trace"
            or plan.get("status") != "command_only_not_executed_selected_hardware_trace"
            or plan.get("measured_region") != "ninfer_bench_measured"
            or plan.get("counters") != [] or plan.get("kernel_include_regex") is not None
            or not isinstance(workload, dict)
            or (workload.get("concurrency"), workload.get("prompt_tokens"),
                workload.get("generated_tokens"), workload.get("spec"),
                workload.get("draft_tokens")) != (1, 2048, 0, "none", 0)
            or not isinstance(command, list) or not all(isinstance(item, str) for item in command)):
        raise ValueError("plan is not the terminal C1 P2048 hardware trace contract")
    if (workload.get("kv_value_group"), workload.get("xattention_profile"),
            workload.get("prefill_chunk")) != (
            route["kv_value_group"], route["xattention_profile"], route["prefill_chunk"]):
        raise ValueError("plan workload differs from terminal selected route")
    for name, key in (("artifact", "artifact"), ("benchmark_executable", "benchmark_executable")):
        _same(plan.get(key), snapshots[name], name)
    if snapshots["artifact"]["sha256"] != route["artifact"]["sha256"] \
            or snapshots["benchmark_executable"]["sha256"] != route["executable"]["sha256"]:
        raise ValueError("explicit artifact/executable differs from terminal winner")
    terminal = plan.get("terminal_selection")
    if not isinstance(terminal, dict) or any(terminal.get(key) != value for key, value in (
        ("path", snapshots["terminal_selection"]["path"]),
        ("sha256", snapshots["terminal_selection"]["sha256"]),
        ("winner", route["winner"]), ("selected_prefill_chunk", route["prefill_chunk"]),
    )):
        raise ValueError("plan terminal selection binding differs")
    source = plan.get("source_matrix")
    if not isinstance(source, dict) or source.get("preset") != "pareto-whole":
        raise ValueError("plan lacks its selected pareto-whole authority")
    source_snapshot = _snapshot(Path(str(source.get("path", ""))), "source matrix")
    if source.get("sha256") != source_snapshot["sha256"]:
        raise ValueError("selected pareto-whole authority changed")
    expected_command = [
        snapshots["benchmark_executable"]["path"], "--weights", snapshots["artifact"]["path"],
        "--corpus", snapshots["corpus"]["path"], "-p", "2048", "-r", "1",
        "--warmup", "1", "--prefill-chunk", str(route["prefill_chunk"]),
        "--concurrency", "1", "--spec", "mtp", "--draft-tokens", "0",
        "--kv-capacity", "workload", "--output", "json", "--output-file",
        snapshots["benchmark_report"]["path"], "--profile-measured",
    ]
    if command != expected_command:
        raise ValueError("benchmark command differs from the exact selected trace command")
    expected_profiler = [
        str(ROCPROFV3), "--selected-regions", "-f", "rocpd", "-d",
        str(Path(snapshots["plan"]["path"]).parent / "rocprof-trace"),
        "--marker-trace", "--kernel-trace", "--memory-copy-trace", "--stats",
        "--summary", "--", *command,
    ]
    hybrid = route["weights_id"] == HYBRID_ID
    capture = plan.get("code_object_capture")
    capture_dir = None
    if hybrid:
        if not isinstance(capture, dict):
            raise ValueError("hybrid trace plan lacks load-time code-object capture")
        capture_library = capture.get("library")
        if not isinstance(capture_library, dict):
            raise ValueError("hybrid trace plan lacks capture-library identity")
        capture_library_now = _snapshot(
            Path(str(capture_library.get("path", ""))), "capture library")
        if capture_library != capture_library_now:
            raise ValueError("hybrid capture-library identity changed")
        capture_dir = Path(str(capture.get("directory", ""))).resolve(strict=True)
        if capture_dir != Path(snapshots["plan"]["path"]).parent / "code-objects":
            raise ValueError("hybrid capture directory differs from selected trace package")
        expected_profiler = [
            "/usr/bin/env", f"LD_PRELOAD={capture_library_now['path']}",
            f"NINFER_CODE_OBJECT_CAPTURE_DIR={capture_dir}", *expected_profiler,
        ]
    elif capture is not None:
        raise ValueError("non-hybrid trace plan must not request FP8 code-object capture")
    if plan.get("profiler_command") != expected_profiler:
        raise ValueError("profiler command differs from selected-region trace contract")
    if Path(snapshots["power_before"]["path"]).read_text().strip() != "auto" \
            or Path(snapshots["power_after"]["path"]).read_text().strip() != "auto":
        raise ValueError("selected trace lacks auto power endpoints")
    report = json.loads(Path(snapshots["benchmark_report"]["path"]).read_text(encoding="utf-8"))
    config, tests, environment = report.get("config"), report.get("tests"), report.get("environment")
    if (report.get("artifact_type") != "ninfer_bench_report"
            or report.get("schema_version") != REPORT_SCHEMA_VERSION
            or report.get("command") != " ".join(command)
            or Path(str(report.get("artifact", {}).get("path", ""))).resolve()
            != Path(snapshots["artifact"]["path"])
            or report.get("artifact", {}).get("file_size_bytes")
            != snapshots["artifact"]["file_size_bytes"]
            or report.get("load", {}).get("weights_id") != route["weights_id"]
            or not isinstance(environment, dict)
            or environment.get("gpu_name") != "AMD Radeon AI PRO R9700"
            or environment.get("architecture_name") != "gfx1201"
            or not isinstance(config, dict) or config.get("concurrency") != 1
            or config.get("prefill_chunk") != route["prefill_chunk"]
            or config.get("kv_value_group") != route["kv_value_group"]
            or config.get("spec") != "none" or config.get("draft_tokens") != 0
            or config.get("dflash_verify_width_requested") != 0
            or config.get("dflash_verify_width") != 0 or config.get("repetitions") != 1
            or config.get("warmup") != 1 or not isinstance(tests, list) or len(tests) != 1
            or (tests[0].get("kind"), tests[0].get("n_prompt"), tests[0].get("n_gen"))
            != ("pp", 2048, 0)):
        raise ValueError("benchmark report differs from selected C1 P2048 trace")
    sparse = route["xattention_profile"] == "b128-s16-tau900"
    expected_xattention = ({"xattention_qualification": False} if not sparse else {
        "xattention_qualification": True, "xattention_profile": "b128-s16-tau900",
        "xattention_find_block": 128, "xattention_stride": 16,
        "xattention_tau_permille": 900})
    if any(config.get(key) != value for key, value in expected_xattention.items()):
        raise ValueError("benchmark report attention profile differs from terminal route")
    if not sparse and any(key in config for key in (
            "xattention_profile", "xattention_find_block", "xattention_stride",
            "xattention_tau_permille")):
        raise ValueError("dense benchmark report carries sparse attention controls")
    rows, aggregates = _parse_database(database_path, " ".join(command))
    legacy = analyze(database_path, report_path)
    if sum(row["calls"] for row in legacy["kernel_execution_categories"]) \
            != aggregates["dispatch_count"]:
        raise ValueError("trace analyzer and lossless dispatch counts differ")
    loaded_fp8 = (selected_loaded_fp8(database_path, capture_dir, rows) if hybrid else [])
    return {
        "artifact_type": "ninfer_r9700_selected_profile_trace", "schema_version": 2,
        "status": "valid_attribution_only", "profile_timing_admissible": False,
        "plan": snapshots["plan"],
        "inputs": {name: value for name, value in snapshots.items() if name != "plan"},
        "authorities": {"source_matrix": source_snapshot}, "workload": workload,
        "selected_route": {"kind": "pp", "prompt_tokens": 2048, "concurrency": 1,
                           "prefill_chunk": route["prefill_chunk"],
                           "kv_value_group": route["kv_value_group"],
                           "xattention_profile": route["xattention_profile"],
                           "artifact": {**snapshots["artifact"], "weights_id": route["weights_id"]},
                           "benchmark_executable": snapshots["benchmark_executable"]},
        "power_profile": {"required": "auto", "before": "auto", "after": "auto",
                          "endpoint_only": True},
        "dispatches": rows, "aggregates": aggregates, "analyzer": legacy,
        "loaded_fp8_code_objects": loaded_fp8,
        "code_object_capture": capture,
        "limitations": ["dispatch timings are attribution evidence, not selection timing"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("plan", "benchmark-report", "database", "power-before", "power-after",
                 "selection", "artifact", "executable", "corpus", "out"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate(args.plan, args.benchmark_report, args.database, args.power_before,
                          args.power_after, args.selection, args.artifact, args.executable,
                          args.corpus)
        _publish(args.out, result)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
