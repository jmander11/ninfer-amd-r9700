#!/usr/bin/env python3
"""Validate one provenance-bound selected-prefill rocprof PMC capture offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.bench.prepare_whole_profile import DISPATCH_COUNTERS, ROCPROFV3
from tools.bench.run_ninfer_bench_matrix import REPORT_SCHEMA_VERSION


ARTIFACT_TYPE = "ninfer_r9700_selected_profile_pmc"
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
    return {
        "path": str(resolved),
        "file_size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _one_table(tables: Iterable[str], prefix: str) -> str:
    matches = [name for name in tables if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"PMC database must contain exactly one {prefix} table")
    return matches[0]


def _decimal(value: object, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _same_decimal(left: Decimal, right: Decimal) -> bool:
    tolerance = max(Decimal("1e-9"), abs(left) * Decimal("1e-12"))
    return abs(left - right) <= tolerance


def _identity_matches(identity: object, actual: dict[str, Any], label: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"plan lacks {label} identity")
    if (
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


def _option(command: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"profile benchmark command lacks one {name}")
    return command[positions[0] + 1]


def _replace_option(command: list[str], name: str, value: str) -> None:
    _option(command, name)
    command[command.index(name) + 1] = value


def _parse_csv(path: Path) -> tuple[
    dict[tuple[int, str], Decimal], dict[int, tuple[str, int, int]], dict[str, dict[str, Any]]
]:
    required = {
        "Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value",
        "Start_Timestamp", "End_Timestamp",
    }
    aggregates: dict[tuple[int, str], Decimal] = defaultdict(Decimal)
    dispatches: dict[int, tuple[str, int, int]] = {}
    samples: dict[str, list[Any]] = {
        name: [0, 0, Decimal(0)] for name in DISPATCH_COUNTERS
    }
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("counter CSV lacks required native rocprof columns")
        for number, row in enumerate(reader, start=2):
            try:
                dispatch_id = int(row["Dispatch_Id"])
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"counter CSV row {number} has invalid dispatch metadata") from error
            name = row["Kernel_Name"]
            counter = row["Counter_Name"]
            if not name or end < start:
                raise ValueError(f"counter CSV row {number} has invalid kernel metadata")
            prior = dispatches.setdefault(dispatch_id, (name, start, end))
            if prior != (name, start, end):
                raise ValueError(f"counter CSV dispatch {dispatch_id} metadata is inconsistent")
            value = _decimal(row["Counter_Value"], f"counter CSV row {number} value")
            aggregates[(dispatch_id, counter)] += value
            if counter not in samples:
                samples[counter] = [0, 0, Decimal(0)]
            samples[counter][0] += 1
            samples[counter][1] += int(value != 0)
            samples[counter][2] += value
    if not dispatches:
        raise ValueError("counter CSV contains no dispatches")
    inventory = {counter for _dispatch, counter in aggregates}
    if inventory != set(DISPATCH_COUNTERS):
        raise ValueError("counter CSV does not contain the exact planned counter inventory")
    expected = set(DISPATCH_COUNTERS)
    for dispatch_id in dispatches:
        observed = {counter for dispatch, counter in aggregates if dispatch == dispatch_id}
        if observed != expected:
            raise ValueError(f"counter CSV dispatch {dispatch_id} has an incomplete counter set")
    return aggregates, dispatches, {
        name: {
            "samples": values[0],
            "nonzero_samples": values[1],
            "sum": str(values[2]),
            "state": "observed_nonzero" if values[1] else "observed_zero",
        }
        for name, values in samples.items() if name in expected
    }


def _parse_database(path: Path) -> tuple[
    dict[tuple[int, str], Decimal], dict[int, tuple[str, int, int]], dict[int, str | None], str
]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = [row[0] for row in connection.execute(
            "select name from sqlite_master where type='table'")]
        process = _one_table(tables, "rocpd_info_process_")
        agent = _one_table(tables, "rocpd_info_agent_")
        pmc_info = _one_table(tables, "rocpd_info_pmc_")
        pmc_event = _one_table(tables, "rocpd_pmc_event_")
        dispatch = _one_table(tables, "rocpd_kernel_dispatch_")
        symbol = _one_table(tables, "rocpd_info_kernel_symbol_")
        strings = _one_table(tables, "rocpd_string_")
        commands = [row[0] for row in connection.execute(
            f'select command from "{process}" order by id')]
        if len(commands) != 1 or not isinstance(commands[0], str):
            raise ValueError("PMC database does not contain one workload command")
        gpu_rows = list(connection.execute(
            f'select type, name, product_name from "{agent}" where type = ?', ("GPU",)))
        if len(gpu_rows) != 1 or tuple(gpu_rows[0]) != ("GPU", EXPECTED_ARCH, EXPECTED_GPU):
            raise ValueError("PMC database is not one R9700/gfx1201 capture")
        info_columns = {row[1] for row in connection.execute(f'pragma table_info("{pmc_info}")')}
        counter_column = "symbol" if "symbol" in info_columns else "name"
        query = f'''select k.dispatch_id, s.display_name kernel_name, k.start, k.end,
                           rs.string region, i.{counter_column} counter_name, e.value
                    from "{pmc_event}" e
                    join "{pmc_info}" i on i.id = e.pmc_id
                    join "{dispatch}" k on k.event_id = e.event_id
                    join "{symbol}" s on s.id = k.kernel_id
                    left join "{strings}" rs on rs.id = k.region_name_id
                    order by k.dispatch_id, i.{counter_column}, e.id'''
        aggregates: dict[tuple[int, str], Decimal] = defaultdict(Decimal)
        dispatches: dict[int, tuple[str, int, int]] = {}
        regions: dict[int, str | None] = {}
        rows = list(connection.execute(query))
        if not rows:
            raise ValueError("PMC database contains no joined counter dispatches")
        for row in rows:
            dispatch_id = int(row["dispatch_id"])
            metadata = (str(row["kernel_name"]), int(row["start"]), int(row["end"]))
            if metadata[2] < metadata[1]:
                raise ValueError(f"PMC database dispatch {dispatch_id} has a negative interval")
            if dispatches.setdefault(dispatch_id, metadata) != metadata:
                raise ValueError(f"PMC database dispatch {dispatch_id} metadata is inconsistent")
            region = None if row["region"] is None else str(row["region"])
            if dispatch_id in regions and regions[dispatch_id] != region:
                raise ValueError(f"PMC database dispatch {dispatch_id} has multiple ROCTX stages")
            regions[dispatch_id] = region
            counter = str(row["counter_name"])
            aggregates[(dispatch_id, counter)] += _decimal(
                row["value"], f"PMC database dispatch {dispatch_id} counter {counter}"
            )
        return aggregates, dispatches, regions, commands[0]
    finally:
        connection.close()


def _ratio(numerator: Decimal, denominator: Decimal) -> dict[str, Any]:
    if denominator == 0:
        return {"state": "unknown_zero_denominator", "value": None,
                "numerator": str(numerator), "denominator": str(denominator)}
    return {"state": "measured", "value": float(numerator / denominator),
            "numerator": str(numerator), "denominator": str(denominator)}


def validate(
    plan_path: Path, report_path: Path, csv_path: Path, database_path: Path,
    power_before_path: Path, power_after_path: Path, terminal_selection_path: Path,
    artifact_path: Path, executable_path: Path, corpus_path: Path,
) -> dict[str, Any]:
    paths = {
        "plan": plan_path, "benchmark_report": report_path, "counter_csv": csv_path,
        "database": database_path, "power_before": power_before_path,
        "power_after": power_after_path, "terminal_selection": terminal_selection_path,
        "artifact": artifact_path, "benchmark_executable": executable_path, "corpus": corpus_path,
    }
    snapshots = {name: _snapshot(path, name) for name, path in paths.items()}
    plan = _load_object(Path(snapshots["plan"]["path"]), "profile plan")
    if (
        plan.get("artifact_type") != "ninfer_whole_profile_plan"
        or plan.get("schema_version") != 2
        or plan.get("profile_kind") != "dispatch-pmc"
        or plan.get("counters") != list(DISPATCH_COUNTERS)
        or plan.get("required_power_profile", {}).get("value") != "profile_standard"
    ):
        raise ValueError("profile plan is not the exact schema-v2 dispatch-PMC contract")
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
    _identity_matches(
        plan.get("benchmark_executable"), snapshots["benchmark_executable"],
        "benchmark executable",
    )
    _identity_matches(plan.get("corpus"), snapshots["corpus"], "corpus")
    terminal = plan.get("terminal_selection")
    if not isinstance(terminal, dict) or (
        Path(str(terminal.get("path", ""))).resolve() != Path(snapshots["terminal_selection"]["path"])
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
    terminal_value = _load_object(
        Path(snapshots["terminal_selection"]["path"]), "terminal selection authority"
    )
    terminal_winner = terminal_value.get("terminal_production_selection")
    if (
        terminal_value.get("schema_version") != 7
        or terminal_value.get("selected_prefill_chunk") != workload.get("prefill_chunk")
        or not isinstance(terminal_winner, dict)
        or terminal_winner.get("winner") != terminal.get("winner")
        or terminal_winner.get("winner_artifact") != terminal.get("winner_artifact")
        or terminal_winner.get("winner_cache_profile") != terminal.get("winner_cache_profile")
        or terminal_winner.get("winner_execution_profile")
        != terminal.get("winner_execution_profile")
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
        "source_report": _authority(
            source_matrix.get("report"), "source P2048 report"),
        "low_context_evaluation": _authority(
            low_context, "low-context evaluation"),
    }
    evaluation_value = _load_object(
        Path(authorities["low_context_evaluation"]["path"]), "low-context evaluation"
    )
    if (
        evaluation_value.get("artifact_type")
        != "ninfer_r9700_low_context_prefill_evaluation"
        or evaluation_value.get("schema_version") != 1
        or evaluation_value.get("manifest") != {
            "path": source_matrix.get("path"), "sha256": source_matrix.get("sha256")
        }
        or evaluation_value.get("artifact") != plan.get("artifact")
        or evaluation_value.get("bench") != plan.get("benchmark_executable")
        or evaluation_value.get("terminal_selection") != terminal
        or evaluation_value.get("expected_kv_value_group") != workload.get("kv_value_group")
        or evaluation_value.get("selected_prefill_chunk") != workload.get("prefill_chunk")
        or any(
            evaluation_value.get(key) != low_context.get(key)
            for key in (
                "minimum_p2048_tok_s", "observed_p2048_tok_s", "passes_p2048_gate",
            )
        )
    ):
        raise ValueError("low-context evaluation contents differ from the profile plan")
    command = plan.get("benchmark_command")
    profiler_command = plan.get("profiler_command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("profile plan benchmark command is invalid")
    if not isinstance(profiler_command, list) or not all(
        isinstance(value, str) for value in profiler_command
    ):
        raise ValueError("profile plan profiler command is invalid")
    source_manifest_value = _load_object(
        Path(authorities["source_matrix"]["path"]), "source low-context manifest"
    )
    source_records = [
        record for record in source_manifest_value.get("commands", [])
        if isinstance(record, dict)
        and record.get("suite") == "low_context_prefill"
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
    expected_benchmark_command = list(source_record["command"])
    if "--profile-measured" in expected_benchmark_command:
        raise ValueError("source manifest P2048 command is already profiler-instrumented")
    _replace_option(expected_benchmark_command, "-r", "1")
    _replace_option(
        expected_benchmark_command, "--output-file", snapshots["benchmark_report"]["path"]
    )
    expected_benchmark_command.append("--profile-measured")
    if command != expected_benchmark_command:
        raise ValueError("profile benchmark command is not the exact source P2048 derivation")
    regex = plan.get("kernel_include_regex")
    if not isinstance(regex, str) or not regex.strip() or regex != regex.strip():
        raise ValueError("profile plan lacks one canonical kernel include regex")
    expected_profiler_command = [
        str(ROCPROFV3), "--selected-regions", "-f", "rocpd", "-d",
        str(Path(snapshots["plan"]["path"]).parent / "rocprof-dispatch-pmc"),
        "--marker-trace", "--kernel-trace", "--kernel-include-regex", regex,
        "--pmc", *DISPATCH_COUNTERS, "--", *command,
    ]
    if profiler_command != expected_profiler_command:
        raise ValueError("profiler command differs from the exact same-capture PMC contract")
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
    config = report.get("config", {})
    environment = report.get("environment", {})
    report_artifact = report.get("artifact", {})
    report_load = report.get("load", {})
    tests = report.get("tests")
    if (
        report.get("artifact_type") != "ninfer_bench_report"
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("command") != " ".join(command)
        or environment.get("gpu_name") != EXPECTED_GPU
        or environment.get("architecture_name") != EXPECTED_ARCH
        or Path(str(report_artifact.get("path", ""))).resolve()
        != Path(snapshots["artifact"]["path"])
        or report_artifact.get("file_size_bytes") != snapshots["artifact"]["file_size_bytes"]
        or report_load.get("weights_id") != plan["artifact"].get("weights_id")
        or config.get("concurrency") != 1
        or config.get("kv_value_group") != workload.get("kv_value_group")
        or config.get("prefill_chunk") != workload.get("prefill_chunk")
        or config.get("kv_plane_layouts") != workload.get("kv_plane_layouts")
        or config.get("spec") != "none"
        or config.get("draft_tokens") != 0
        or config.get("dflash_verify_width_requested") != 0
        or config.get("dflash_verify_width") != 0
        or config.get("xattention_qualification") is not False
        or any(key in config for key in (
            "xattention_profile", "xattention_find_block", "xattention_stride",
            "xattention_tau_permille",
        ))
        or config.get("repetitions") != 1
        or config.get("warmup") != 1
        or not isinstance(tests, list) or len(tests) != 1
        or tests[0].get("kind") != "pp" or tests[0].get("n_prompt") != 2048
        or tests[0].get("n_gen") != 0
    ):
        raise ValueError("benchmark report is not the planned selected P2048 capture")

    before = Path(snapshots["power_before"]["path"]).read_text(encoding="utf-8").strip()
    after = Path(snapshots["power_after"]["path"]).read_text(encoding="utf-8").strip()
    if before != "profile_standard" or after != "auto":
        raise ValueError("PMC capture lacks profile_standard capture and auto restoration evidence")

    csv_aggregates, csv_dispatches, counter_summary = _parse_csv(
        Path(snapshots["counter_csv"]["path"])
    )
    db_aggregates, db_dispatches, regions, db_command = _parse_database(
        Path(snapshots["database"]["path"])
    )
    if csv_dispatches != db_dispatches or set(csv_aggregates) != set(db_aggregates):
        raise ValueError("counter CSV and PMC database dispatch inventories differ")
    for key, value in csv_aggregates.items():
        if not _same_decimal(value, db_aggregates[key]):
            raise ValueError(f"counter CSV and PMC database aggregate differ for {key}")
    if db_command != " ".join(command):
        raise ValueError("PMC database workload command differs from the profile plan")

    has_regions = [bool(value) for value in regions.values()]
    marker_requested = "--marker-trace" in profiler_command
    if any(has_regions) and not all(has_regions):
        raise ValueError("PMC database has a partial ROCTX stage join")
    if marker_requested and not all(has_regions):
        raise ValueError("marker-enabled PMC capture lacks exact same-capture ROCTX stages")
    stage_join = {
        "state": "exact_same_capture" if all(has_regions) else "unavailable_not_captured",
        "basis": (
            "PMC event_id joined to the same kernel-dispatch event and its ROCTX region_name_id"
            if all(has_regions) else
            "no ROCTX region association is present; no separately captured stage was inferred"
        ),
        "dispatches": [
            {"dispatch_id": dispatch, "stage": regions[dispatch],
             "symbol": db_dispatches[dispatch][0]}
            for dispatch in sorted(regions)
        ] if all(has_regions) else [],
    }
    if all(has_regions):
        stage_totals: dict[str, dict[str, Decimal]] = defaultdict(
            lambda: defaultdict(Decimal)
        )
        stage_dispatches: dict[str, set[int]] = defaultdict(set)
        for (dispatch_id, counter), value in csv_aggregates.items():
            stage = regions[dispatch_id]
            assert stage is not None
            stage_totals[stage][counter] += value
            stage_dispatches[stage].add(dispatch_id)
        if any(
            values.get("TCP_REQ_MISS", Decimal(0))
            > values.get("TCP_REQ", Decimal(0))
            for values in stage_totals.values()
        ):
            raise ValueError("a ROCTX stage TCP miss count exceeds its request count")
        stage_join["stages"] = [
            {
                "stage": stage,
                "dispatch_count": len(stage_dispatches[stage]),
                "counter_sums": {
                    counter: str(values.get(counter, Decimal(0)))
                    for counter in DISPATCH_COUNTERS
                },
                "gl2_hit_ratio": _ratio(
                    values.get("GL2C_HIT", Decimal(0)),
                    values.get("GL2C_HIT", Decimal(0))
                    + values.get("GL2C_MISS", Decimal(0)),
                ),
                "tcp_hit_ratio": _ratio(
                    values.get("TCP_REQ", Decimal(0))
                    - values.get("TCP_REQ_MISS", Decimal(0)),
                    values.get("TCP_REQ", Decimal(0)),
                ),
            }
            for stage, values in sorted(stage_totals.items())
        ]
    else:
        stage_join["stages"] = []

    totals = {name: Decimal(counter_summary[name]["sum"]) for name in DISPATCH_COUNTERS}
    gl2_denominator = totals["GL2C_HIT"] + totals["GL2C_MISS"]
    tcp_requests = totals["TCP_REQ"]
    if totals["TCP_REQ_MISS"] > tcp_requests:
        raise ValueError("TCP miss count exceeds request count")
    gl2 = _ratio(totals["GL2C_HIT"], gl2_denominator)
    tcp = _ratio(tcp_requests - totals["TCP_REQ_MISS"], tcp_requests)
    if gl2["state"] != "measured" or tcp["state"] != "measured" or totals["SQ_WAVES"] <= 0:
        raise ValueError("PMC capture lacks positive GL2/TCP/SQ activity controls")

    for name, snapshot in snapshots.items():
        if _snapshot(Path(snapshot["path"]), name) != snapshot:
            raise ValueError(f"{name} changed while validating PMC evidence")
    for name, snapshot in authorities.items():
        if _snapshot(Path(snapshot["path"]), name) != snapshot:
            raise ValueError(f"{name} authority changed while validating PMC evidence")
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "valid_attribution_only",
        "profile_timing_admissible": False,
        "plan": snapshots["plan"],
        "inputs": {name: value for name, value in snapshots.items() if name != "plan"},
        "authorities": authorities,
        "workload": workload,
        "power_profile": {"required": "profile_standard", "before": before, "after": after},
        "counters": counter_summary,
        "metrics": {"gl2_hit_ratio": gl2, "tcp_hit_ratio": tcp,
                    "sq_waves": str(totals["SQ_WAVES"])},
        "roctx_stage_join": stage_join,
        "limitations": [
            "GL2/TCP ratios are relative event ratios, not absolute cache or HBM bandwidth.",
            "All-zero collected counters remain observed_zero; missing requested counters fail validation.",
            "Power files prove endpoint states only, not uninterrupted state during collection.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "plan", "benchmark-report", "counter-csv", "database", "power-before",
        "power-after", "terminal-selection", "artifact", "executable", "corpus", "out",
    ):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    try:
        result = validate(
            args.plan, args.benchmark_report, args.counter_csv, args.database,
            args.power_before, args.power_after, args.terminal_selection,
            args.artifact, args.executable, args.corpus,
        )
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(f"wrote validated PMC evidence to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
