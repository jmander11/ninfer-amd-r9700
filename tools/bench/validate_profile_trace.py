#!/usr/bin/env python3
"""Validate one provenance-bound selected-P2048 auto trace without inferring roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.bench.analyze_whole_profile import analyze
from tools.bench.prepare_whole_profile import ROCPROFV3
from tools.bench.run_ninfer_bench_matrix import REPORT_SCHEMA_VERSION


ARTIFACT_TYPE = "ninfer_r9700_selected_profile_trace"
SCHEMA_VERSION = 1
EXPECTED_GPU = "AMD Radeon AI PRO R9700"
EXPECTED_ARCH = "gfx1201"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved)}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity_matches(identity: object, actual: dict[str, Any], label: str) -> None:
    if not isinstance(identity, dict) or (
        Path(str(identity.get("path", ""))).resolve() != Path(actual["path"])
        or identity.get("sha256") != actual["sha256"]
        or identity.get("file_size_bytes") != actual["file_size_bytes"]
    ):
        raise ValueError(f"{label} identity differs from the profile plan")


def _authority(identity: object, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
        raise ValueError(f"profile plan lacks {label} authority")
    actual = _snapshot(Path(identity["path"]), label)
    if identity.get("sha256") != actual["sha256"]:
        raise ValueError(f"{label} authority bytes differ from the profile plan")
    if "bytes" in identity and identity["bytes"] != actual["file_size_bytes"]:
        raise ValueError(f"{label} authority size differs from the profile plan")
    return actual


def _one_table(tables: Iterable[str], prefix: str) -> str:
    matches = [name for name in tables if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"trace database must contain exactly one {prefix} table")
    return matches[0]


def _option(command: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"profile benchmark command lacks one {name}")
    return command[positions[0] + 1]


def _replace_option(command: list[str], name: str, value: str) -> None:
    _option(command, name)
    command[command.index(name) + 1] = value


def _wall_union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    begin, end = intervals[0]
    for next_begin, next_end in intervals[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def _nullable_integer(row: sqlite3.Row, columns: set[str], name: str,
                      minimum: int | None = None) -> int | None:
    if name not in columns or row[name] is None:
        return None
    value = row[name]
    if (isinstance(value, bool) or not isinstance(value, int)
            or (minimum is not None and value < minimum)):
        raise ValueError(f"trace dispatch {name} is not an integer")
    return value


def _parse_database(path: Path, expected_command: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        objects = {row[0] for row in connection.execute(
            "select name from sqlite_master where type in ('table','view')")}
        if not {"kernels", "regions", "memory_copies"} <= objects:
            raise ValueError("trace database lacks marker, kernel, or memory-copy views")
        process = _one_table(objects, "rocpd_info_process_")
        agent = _one_table(objects, "rocpd_info_agent_")
        commands = [row[0] for row in connection.execute(
            f'select command from "{process}" order by id')]
        if commands != [expected_command]:
            raise ValueError("trace database workload command differs from the profile plan")
        gpu_rows = list(connection.execute(
            f'select type, name, product_name, absolute_index from "{agent}" '
            'where type = ? and name = ? and product_name = ?',
            ("GPU", EXPECTED_ARCH, EXPECTED_GPU)))
        if len(gpu_rows) != 1:
            raise ValueError("trace database does not contain one R9700/gfx1201 agent")
        expected_agent = int(gpu_rows[0][3])

        columns = {row[1] for row in connection.execute('pragma table_info("kernels")')}
        required = {
            "dispatch_id", "name", "region", "start", "end", "duration", "stream_id",
            "agent_abs_index",
            "grid_x", "grid_y", "grid_z", "workgroup_x", "workgroup_y", "workgroup_z",
        }
        if not required <= columns:
            raise ValueError("trace kernels view lacks required lossless dispatch columns")
        optional = (
            "sgpr_count", "vgpr_count", "accum_vgpr_count", "lds_size", "scratch_size",
            "static_lds_size", "static_scratch_size",
        )
        selection = list(sorted(required)) + [name for name in optional if name in columns]
        query = ", ".join(f'"{name}"' for name in selection)
        raw_rows = list(connection.execute(
            f'select {query} from kernels order by start, dispatch_id'))
        sql_count, sql_service = connection.execute(
            "select count(*), coalesce(sum(duration), 0) from kernels").fetchone()
    finally:
        connection.close()
    if not raw_rows:
        raise ValueError("trace database contains no selected-region kernel dispatches")

    dispatches: list[dict[str, Any]] = []
    identities: set[int] = set()
    intervals: list[tuple[int, int]] = []
    for row in raw_rows:
        dispatch_id = _nullable_integer(row, columns, "dispatch_id", 0)
        start = _nullable_integer(row, columns, "start", 0)
        end = _nullable_integer(row, columns, "end", 1)
        duration = _nullable_integer(row, columns, "duration", 1)
        if dispatch_id is None or dispatch_id in identities:
            raise ValueError("trace dispatch IDs must be present and unique")
        if start is None or end is None or duration is None or end <= start or end - start != duration:
            raise ValueError(f"trace dispatch {dispatch_id} has inconsistent timing")
        symbol = row["name"]
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"trace dispatch {dispatch_id} lacks a symbol")
        if _nullable_integer(row, columns, "agent_abs_index", 0) != expected_agent:
            raise ValueError(f"trace dispatch {dispatch_id} did not execute on the R9700")
        identities.add(dispatch_id)
        intervals.append((start, end))
        dispatches.append({
            "dispatch_id": f"trace:{dispatch_id}",
            "rocprof_dispatch_id": dispatch_id,
            "symbol": symbol,
            "roctx_region": (
                row["region"] if isinstance(row["region"], str) and row["region"] else None),
            "start_ns": start, "end_ns": end, "duration_ns": duration,
            "stream_id": _nullable_integer(row, columns, "stream_id", 0),
            "grid": {axis: _nullable_integer(row, columns, f"grid_{axis}", 1)
                     for axis in ("x", "y", "z")},
            "workgroup": {axis: _nullable_integer(row, columns, f"workgroup_{axis}", 1)
                          for axis in ("x", "y", "z")},
            "resources": {
                "sgpr_count": _nullable_integer(row, columns, "sgpr_count", 0),
                "vgpr_count": _nullable_integer(row, columns, "vgpr_count", 0),
                "accum_vgpr_count": _nullable_integer(
                    row, columns, "accum_vgpr_count", 0),
                "lds_bytes": _nullable_integer(row, columns, "lds_size", 0),
                "scratch_bytes": _nullable_integer(row, columns, "scratch_size", 0),
                "static_lds_bytes": _nullable_integer(row, columns, "static_lds_size", 0),
                "static_scratch_bytes": _nullable_integer(
                    row, columns, "static_scratch_size", 0),
            },
        })
    service = sum(row["duration_ns"] for row in dispatches)
    if len(dispatches) != sql_count or service != sql_service:
        raise ValueError("lossless dispatch extraction does not conserve database count/service")
    return dispatches, {
        "dispatch_count": len(dispatches),
        "independent_device_service_time_ns": service,
        "device_wall_union_ns": _wall_union_ns(intervals),
    }


def validate(
    plan_path: Path, report_path: Path, database_path: Path, power_before_path: Path,
    power_after_path: Path, terminal_selection_path: Path, artifact_path: Path,
    executable_path: Path, corpus_path: Path,
) -> dict[str, Any]:
    paths = {
        "plan": plan_path, "benchmark_report": report_path, "database": database_path,
        "power_before": power_before_path, "power_after": power_after_path,
        "terminal_selection": terminal_selection_path, "artifact": artifact_path,
        "benchmark_executable": executable_path, "corpus": corpus_path,
    }
    snapshots = {name: _snapshot(path, name) for name, path in paths.items()}
    plan = _load_object(Path(snapshots["plan"]["path"]), "profile plan")
    if (
        plan.get("artifact_type") != "ninfer_whole_profile_plan"
        or type(plan.get("schema_version")) is not int or plan["schema_version"] != 2
        or plan.get("profile_kind") != "trace" or plan.get("counters") != []
        or plan.get("kernel_include_regex") is not None
        or plan.get("measured_region") != "ninfer_bench_measured"
        or plan.get("required_power_profile", {}).get("value") != "auto"
    ):
        raise ValueError("profile plan is not the exact schema-v2 auto trace contract")
    required_power = plan["required_power_profile"]
    if (
        Path(str(required_power.get("before_evidence", ""))).resolve()
        != Path(snapshots["power_before"]["path"])
        or Path(str(required_power.get("after_evidence", ""))).resolve()
        != Path(snapshots["power_after"]["path"])
    ):
        raise ValueError("power endpoint evidence paths differ from the profile plan")
    workload = plan.get("workload")
    if not isinstance(workload, dict) or any(workload.get(key) != value for key, value in (
        ("concurrency", 1), ("prompt_tokens", 2048), ("generated_tokens", 0),
        ("spec", "none"), ("draft_tokens", 0), ("dflash_verify_width", 0),
        ("xattention_profile", "dense"),
    )):
        raise ValueError("profile plan is not the selected dense C1 P2048 prefill point")
    _identity_matches(plan.get("artifact"), snapshots["artifact"], "artifact")
    _identity_matches(plan.get("benchmark_executable"), snapshots["benchmark_executable"],
                      "benchmark executable")
    _identity_matches(plan.get("corpus"), snapshots["corpus"], "corpus")

    terminal = plan.get("terminal_selection")
    if not isinstance(terminal, dict) or (
        Path(str(terminal.get("path", ""))).resolve()
        != Path(snapshots["terminal_selection"]["path"])
        or terminal.get("sha256") != snapshots["terminal_selection"]["sha256"]
        or terminal.get("bytes") != snapshots["terminal_selection"]["file_size_bytes"]
    ):
        raise ValueError("terminal selection authority differs from the profile plan")
    winner_artifact = terminal.get("winner_artifact")
    winner_cache = terminal.get("winner_cache_profile")
    if (
        not isinstance(winner_artifact, dict)
        or winner_artifact.get("weights_id") != plan["artifact"].get("weights_id")
        or winner_artifact.get("sha256") != snapshots["artifact"]["sha256"]
        or winner_artifact.get("conversion_receipt")
        != plan["artifact"].get("conversion_receipt")
        or not isinstance(winner_cache, dict)
        or winner_cache.get("value_group") != workload.get("kv_value_group")
        or terminal.get("selected_prefill_chunk") != workload.get("prefill_chunk")
    ):
        raise ValueError("terminal selection tuple differs from the planned P2048 route")
    terminal_value = _load_object(Path(snapshots["terminal_selection"]["path"]),
                                  "terminal selection authority")
    terminal_winner = terminal_value.get("terminal_production_selection")
    if (
        terminal_value.get("schema_version") != 7
        or terminal_value.get("selected_prefill_chunk") != workload.get("prefill_chunk")
        or not isinstance(terminal_winner, dict)
        or any(terminal_winner.get(key) != terminal.get(key) for key in (
            "winner", "winner_artifact", "winner_cache_profile", "winner_execution_profile"))
    ):
        raise ValueError("terminal selection file differs from its planned selected tuple")

    source_matrix = plan.get("source_matrix")
    low_context = plan.get("low_context_evaluation")
    if (
        not isinstance(source_matrix, dict)
        or source_matrix.get("preset") != "low-context-prefill"
        or not isinstance(low_context, dict)
        or type(low_context.get("passes_p2048_gate")) is not bool
        or type(low_context.get("minimum_p2048_tok_s")) not in (int, float)
        or type(low_context.get("observed_p2048_tok_s")) not in (int, float)
        or not math.isfinite(low_context["minimum_p2048_tok_s"])
        or not math.isfinite(low_context["observed_p2048_tok_s"])
    ):
        raise ValueError("profile plan lacks a valid low-context evaluation authority")
    authorities = {
        "source_matrix": _authority(source_matrix, "source matrix"),
        "source_report": _authority(source_matrix.get("report"), "source P2048 report"),
        "low_context_evaluation": _authority(low_context, "low-context evaluation"),
    }
    evaluation_value = _load_object(
        Path(authorities["low_context_evaluation"]["path"]), "low-context evaluation")
    if (
        evaluation_value.get("artifact_type")
        != "ninfer_r9700_low_context_prefill_evaluation"
        or evaluation_value.get("schema_version") != 1
        or evaluation_value.get("manifest") != {
            "path": source_matrix.get("path"), "sha256": source_matrix.get("sha256")}
        or evaluation_value.get("artifact") != plan.get("artifact")
        or evaluation_value.get("bench") != plan.get("benchmark_executable")
        or evaluation_value.get("terminal_selection") != terminal
        or evaluation_value.get("expected_kv_value_group") != workload.get("kv_value_group")
        or evaluation_value.get("selected_prefill_chunk") != workload.get("prefill_chunk")
        or any(evaluation_value.get(key) != low_context.get(key) for key in (
            "minimum_p2048_tok_s", "observed_p2048_tok_s", "passes_p2048_gate"))
    ):
        raise ValueError("low-context evaluation contents differ from the profile plan")

    command = plan.get("benchmark_command")
    profiler_command = plan.get("profiler_command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("profile plan benchmark command is invalid")
    if not isinstance(profiler_command, list) or not all(
        isinstance(value, str) for value in profiler_command):
        raise ValueError("profile plan profiler command is invalid")
    source_manifest = _load_object(Path(authorities["source_matrix"]["path"]),
                                   "source low-context manifest")
    source_records = [
        record for record in source_manifest.get("commands", [])
        if isinstance(record, dict) and record.get("suite") == "low_context_prefill"
        and record.get("case") == "prefill_p2048_dense_none"
        and record.get("concurrency") == 1
    ]
    if len(source_records) != 1:
        raise ValueError("source manifest lacks one exact dense C1 P2048 command")
    source_record = source_records[0]
    if (
        Path(str(source_record.get("report", ""))).resolve()
        != Path(authorities["source_report"]["path"])
        or not isinstance(source_record.get("command"), list)
        or not all(isinstance(value, str) for value in source_record["command"])
    ):
        raise ValueError("source manifest P2048 record differs from its bound report")
    expected_benchmark = list(source_record["command"])
    if "--profile-measured" in expected_benchmark:
        raise ValueError("source manifest P2048 command is already profiler-instrumented")
    _replace_option(expected_benchmark, "-r", "1")
    _replace_option(expected_benchmark, "--output-file", snapshots["benchmark_report"]["path"])
    expected_benchmark.append("--profile-measured")
    if command != expected_benchmark:
        raise ValueError("profile benchmark command is not the exact source P2048 derivation")
    expected_profiler = [
        str(ROCPROFV3), "--selected-regions", "-f", "rocpd", "-d",
        str(Path(snapshots["plan"]["path"]).parent / "rocprof-trace"),
        "--marker-trace", "--kernel-trace", "--memory-copy-trace", "--stats", "--summary",
        "--", *command,
    ]
    if profiler_command != expected_profiler:
        raise ValueError("profiler command differs from the exact marker/kernel trace contract")
    if command[0] != snapshots["benchmark_executable"]["path"]:
        raise ValueError("profile command executable differs")
    if Path(_option(command, "--weights")).resolve() != Path(snapshots["artifact"]["path"]):
        raise ValueError("profile command artifact differs")
    if Path(_option(command, "--corpus")).resolve() != Path(snapshots["corpus"]["path"]):
        raise ValueError("profile command corpus differs")
    if Path(_option(command, "--output-file")).resolve() != Path(snapshots["benchmark_report"]["path"]):
        raise ValueError("profile command benchmark report differs")
    if _option(command, "-p") != "2048" or _option(command, "-r") != "1":
        raise ValueError("profile command does not retain one P2048 repetition")
    if command.count("--profile-measured") != 1:
        raise ValueError("profile command lacks one measured-region boundary")

    report = _load_object(Path(snapshots["benchmark_report"]["path"]), "benchmark report")
    config, environment = report.get("config", {}), report.get("environment", {})
    report_artifact, report_load, tests = report.get("artifact", {}), report.get("load", {}), report.get("tests")
    if (
        report.get("artifact_type") != "ninfer_bench_report"
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("command") != " ".join(command)
        or environment.get("gpu_name") != EXPECTED_GPU
        or environment.get("architecture_name") != EXPECTED_ARCH
        or Path(str(report_artifact.get("path", ""))).resolve() != Path(snapshots["artifact"]["path"])
        or report_artifact.get("file_size_bytes") != snapshots["artifact"]["file_size_bytes"]
        or report_load.get("weights_id") != plan["artifact"].get("weights_id")
        or config.get("concurrency") != 1 or config.get("kv_value_group") != workload.get("kv_value_group")
        or config.get("prefill_chunk") != workload.get("prefill_chunk")
        or config.get("kv_plane_layouts") != workload.get("kv_plane_layouts")
        or config.get("spec") != "none" or config.get("draft_tokens") != 0
        or config.get("dflash_verify_width_requested") != 0 or config.get("dflash_verify_width") != 0
        or config.get("xattention_qualification") is not False
        or any(key in config for key in ("xattention_profile", "xattention_find_block",
                                         "xattention_stride", "xattention_tau_permille"))
        or config.get("repetitions") != 1 or config.get("warmup") != 1
        or not isinstance(tests, list) or len(tests) != 1
        or tests[0].get("kind") != "pp" or tests[0].get("n_prompt") != 2048
        or tests[0].get("n_gen") != 0
    ):
        raise ValueError("benchmark report is not the planned selected P2048 trace")
    before = Path(snapshots["power_before"]["path"]).read_text(encoding="utf-8").strip()
    after = Path(snapshots["power_after"]["path"]).read_text(encoding="utf-8").strip()
    if before != "auto" or after != "auto":
        raise ValueError("trace capture lacks auto before/after evidence")

    legacy_analysis = analyze(Path(snapshots["database"]["path"]),
                              Path(snapshots["benchmark_report"]["path"]))
    dispatches, aggregates = _parse_database(
        Path(snapshots["database"]["path"]), " ".join(command))
    analyzer_count = sum(row["calls"] for row in legacy_analysis["kernel_execution_categories"])
    analyzer_service = round(sum(
        row["independent_summed_duration_ms"]
        for row in legacy_analysis["kernel_execution_categories"]) * 1e6)
    if analyzer_count != aggregates["dispatch_count"] or analyzer_service != aggregates[
            "independent_device_service_time_ns"]:
        raise ValueError("lossless dispatch aggregates differ from analyzer count/service")

    for name, snapshot in snapshots.items():
        if _snapshot(Path(snapshot["path"]), name) != snapshot:
            raise ValueError(f"{name} changed while validating trace evidence")
    for name, snapshot in authorities.items():
        if _snapshot(Path(snapshot["path"]), name) != snapshot:
            raise ValueError(f"{name} authority changed while validating trace evidence")
    return {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "status": "valid_attribution_only", "profile_timing_admissible": False,
        "plan": snapshots["plan"],
        "inputs": {name: value for name, value in snapshots.items() if name != "plan"},
        "authorities": authorities, "workload": workload,
        "selected_route": {
            "kind": "pp",
            "prompt_tokens": 2048,
            "concurrency": 1,
            "prefill_chunk": workload["prefill_chunk"],
            "kv_value_group": workload["kv_value_group"],
            "xattention_profile": workload["xattention_profile"],
            "artifact": plan["artifact"],
            "benchmark_executable": plan["benchmark_executable"],
            "unprofiled_p2048": {
                "report": authorities["source_report"],
                "evaluation": authorities["low_context_evaluation"],
                "minimum_tok_s": low_context["minimum_p2048_tok_s"],
                "observed_tok_s": low_context["observed_p2048_tok_s"],
                "passes_gate": low_context["passes_p2048_gate"],
            },
        },
        "power_profile": {"required": "auto", "before": before, "after": after,
                          "endpoint_only": True},
        "dispatches": dispatches, "aggregates": aggregates,
        "analyzer": legacy_analysis,
        "limitations": [
            "Dispatch timings are profiler attribution evidence, not selection timing.",
            "ROCTX regions are retained verbatim; no model role is inferred.",
            "Power files prove endpoint states only, not uninterrupted state during collection.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("plan", "benchmark-report", "database", "power-before", "power-after",
                 "terminal-selection", "artifact", "executable", "corpus", "out"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    try:
        result = validate(args.plan, args.benchmark_report, args.database, args.power_before,
                          args.power_after, args.terminal_selection, args.artifact,
                          args.executable, args.corpus)
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(f"wrote validated auto trace evidence to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
