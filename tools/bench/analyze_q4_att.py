#!/usr/bin/env python3
"""Validate and summarize one exact production-Q4 P2048 ATT dispatch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PLAN_SCHEMA = "ninfer.r9700.production_q4_p2048_att_plan.v2"
REPORT_SCHEMA = "ninfer.r9700.production_q4_p2048_att_attribution.v1"
KERNEL = "a8q4g64_linear_prefill_cta_kernel"
ROLE_REGION = "ninfer.post-mixer.prefill.post_mixer payload=0"
MATCH_ITERATION = 3
EXPECTED_Q4_DISPATCHES = 176
EXPECTED_GRID = (20480, 32, 1)
EXPECTED_WORKGROUP = (512, 1, 1)
ATT_COLUMNS = {
    "CodeObj", "Vaddr", "Instruction", "Hitcount", "Latency", "Stall", "Idle", "Source",
}
TOOLS = {
    "rocprofv3": Path("/opt/rocm/core-10.0/bin/rocprofv3"),
    "rocprofiler_sdk": Path("/opt/rocm/lib/librocprofiler-sdk.so.1.3.5"),
    "trace_decoder": Path("/opt/rocm/lib/librocprof-trace-decoder.so.0.2.0"),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved)}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(expected: object, actual: dict[str, Any], label: str) -> None:
    if not isinstance(expected, dict) or any((
        Path(str(expected.get("path", ""))).resolve() != Path(actual["path"]),
        expected.get("file_size_bytes") != actual["file_size_bytes"],
        expected.get("sha256") != actual["sha256"],
    )):
        raise ValueError(f"{label} identity differs from the ATT plan")


def _validate_closure(path: Path, expected: dict[str, dict[str, Any]]) -> None:
    entries: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if match is None:
            raise ValueError("prepared checksum closure has malformed content")
        resolved = Path(match.group(2)).resolve()
        if resolved in entries:
            raise ValueError("prepared checksum closure repeats a path")
        entries[resolved] = match.group(1)
    wanted = {Path(value["path"]): value["sha256"] for value in expected.values()}
    if entries != wanted:
        raise ValueError("prepared checksum closure differs from current package identities")


def _one_table(names: Iterable[str], prefix: str) -> str:
    matches = [name for name in names if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"database must contain exactly one {prefix} table")
    return matches[0]


def _q4_dispatches(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = [row[0] for row in connection.execute(
            "select name from sqlite_master where type='table'")]
        dispatch = _one_table(tables, "rocpd_kernel_dispatch_")
        symbols = _one_table(tables, "rocpd_info_kernel_symbol_")
        strings = _one_table(tables, "rocpd_string_")
        agents = _one_table(tables, "rocpd_info_agent_")
        processes = _one_table(tables, "rocpd_info_process_")
        gpu = list(connection.execute(
            f'''select id, name, product_name, absolute_index, type_index, extdata from "{agents}"
                where type='GPU' and name='gfx1201' '''))
        if len(gpu) != 1 or gpu[0]["product_name"] != "AMD Radeon AI PRO R9700":
            raise ValueError("database lacks exactly one R9700/gfx1201 GPU")
        process = list(connection.execute(f'select command, environment from "{processes}"'))
        if len(process) != 1 or not process[0]["command"]:
            raise ValueError("database lacks one process command")
        query = f'''select k.dispatch_id, k.start, k.end,
                           k.grid_size_x, k.grid_size_y, k.grid_size_z,
                           k.workgroup_size_x, k.workgroup_size_y, k.workgroup_size_z,
                           k.private_segment_size, k.group_segment_size,
                           s.display_name symbol, s.arch_vgpr_count vgpr,
                           r.string region
                    from "{dispatch}" k join "{symbols}" s on s.id=k.kernel_id
                    left join "{strings}" r on r.id=k.region_name_id
                    where s.display_name like ? order by k.start, k.dispatch_id'''
        rows = list(connection.execute(query, (f"%{KERNEL}%",)))
    finally:
        connection.close()
    result = []
    for row in rows:
        result.append({
            "dispatch_id": int(row["dispatch_id"]), "start": int(row["start"]),
            "end": int(row["end"]), "symbol": str(row["symbol"]),
            "region": row["region"],
            "grid": tuple(int(row[f"grid_size_{axis}"]) for axis in "xyz"),
            "workgroup": tuple(int(row[f"workgroup_size_{axis}"]) for axis in "xyz"),
            "private_bytes": int(row["private_segment_size"] or 0),
            "lds_bytes": int(row["group_segment_size"] or 0),
            "vgpr": int(row["vgpr"] or 0),
        })
    gpu_extdata = json.loads(gpu[0]["extdata"])
    return result, {
        "command": str(process[0]["command"]),
        "registered_library": json.loads(process[0]["environment"]).get(
            "ROCPROFILER_REGISTER_LIBRARY"),
        "agent_id": int(gpu[0]["id"]), "absolute_index": int(gpu[0]["absolute_index"]),
        "type_index": int(gpu[0]["type_index"]),
        "agent_handle": int(gpu_extdata.get("id", {}).get("handle", -1)),
        "gpu_index": int(gpu_extdata.get("gpu_index", -1)),
    }


def _selected_dispatch(rows: list[dict[str, Any]], *, require_full_inventory: bool) -> dict[str, Any]:
    if require_full_inventory and len(rows) != EXPECTED_Q4_DISPATCHES:
        raise ValueError("source PMC database lacks the exact 176 Q4 dispatches")
    if len(rows) < MATCH_ITERATION:
        raise ValueError("database lacks the selected Q4 match iteration")
    selected = rows[MATCH_ITERATION - 1]
    if (selected["region"] != ROLE_REGION or selected["grid"] != EXPECTED_GRID
            or selected["workgroup"] != EXPECTED_WORKGROUP
            or selected["private_bytes"] != 0 or KERNEL not in selected["symbol"]):
        raise ValueError("selected Q4 match iteration is not exact post-mixer/down P2048")
    return selected


def _resolve_att_outputs(raw_directory: Path, database_path: Path) -> tuple[Path, Path]:
    if not raw_directory.resolve(strict=True).is_dir():
        raise ValueError("ATT raw output is not a directory")
    rows, process = _q4_dispatches(database_path)
    _selected_dispatch(rows, require_full_inventory=True)
    if (process["agent_id"] != 1 or process["absolute_index"] != 1
            or process["type_index"] != 0 or process["gpu_index"] != 0
            or process["agent_handle"] < 0):
        raise ValueError("ATT database is not bound to the selected R9700 agent")
    trace_pattern = re.compile(
        r"production-q4-p2048-att_(\d+)_shader_engine_0_(\d+)\.att")
    trace_names = [child.name for child in raw_directory.iterdir()
                   if child.is_file() and child.suffix == ".att"]
    trace_matches = [(name, trace_pattern.fullmatch(name)) for name in trace_names]
    if (len(trace_matches) != 1 or trace_matches[0][1] is None
            or trace_matches[0][1].group(2) != "1"):
        raise ValueError("ATT raw directory lacks one exact R9700/SE0 selected trace")
    # rocprofv3's decoded filenames use the opaque ATT-agent handle and the
    # per-capture trace ordinal from its raw trace filename, not ROCPD IDs.
    trace_match = trace_matches[0][1]
    if int(trace_match.group(1)) != process["agent_handle"]:
        raise ValueError("ATT raw trace agent handle is not the R9700 database agent")
    stem = f"ui_output_agent_{trace_match.group(1)}_dispatch_{trace_match.group(2)}"
    expected_ui = raw_directory / stem
    expected_stats = raw_directory / f"stats_{stem}.csv"
    if not expected_ui.is_dir() or not expected_stats.is_file():
        raise ValueError("ATT decoder did not produce the exact selected-dispatch outputs")
    observed = {
        child.name for child in raw_directory.iterdir()
        if child.name.startswith("ui_output_agent_")
        or child.name.startswith("stats_ui_output_agent_")
    }
    if observed != {expected_ui.name, expected_stats.name}:
        raise ValueError("ATT raw directory contains extra or mismatched dispatch outputs")
    return expected_stats, expected_ui


def _number(value: str, label: str) -> int:
    try:
        number = int(value, 0)
    except ValueError as error:
        raise ValueError(f"ATT {label} is not an integer") from error
    if number < 0:
        raise ValueError(f"ATT {label} is negative")
    return number


def _region(instruction: str) -> str:
    opcode = instruction.lstrip("; ").split(maxsplit=1)[0] if instruction.strip("; ") else ""
    if opcode.startswith("v_wmma_i32_16x16x32_iu4"):
        return "iu4_wmma"
    if opcode.startswith(("s_wait_loadcnt", "global_load_")):
        return "global_load_and_wait"
    if opcode.startswith(("ds_", "s_wait_dscnt", "s_barrier_")):
        return "lds_and_barrier"
    if opcode.startswith(("v_lshl_add_u32", "v_cvt_f32_i32", "v_fma_mix_f32",
                          "v_fmac_f32", "v_dual_fmac_f32")):
        return "recombine_scale_accumulate"
    return "other"


def _read_att(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if set(reader.fieldnames or []) != ATT_COLUMNS:
            raise ValueError("ATT stats CSV has an unexpected column contract")
        for line, row in enumerate(reader, 2):
            instruction = row["Instruction"].strip()
            hit = _number(row["Hitcount"], f"line {line} Hitcount")
            if instruction.startswith(";") or hit == 0:
                continue
            item = {
                "code_object_id": _number(row["CodeObj"], f"line {line} CodeObj"),
                "vaddr": _number(row["Vaddr"], f"line {line} Vaddr"),
                "instruction": instruction, "hit": hit,
                "latency": _number(row["Latency"], f"line {line} Latency"),
                "stall": _number(row["Stall"], f"line {line} Stall"),
                "idle": _number(row["Idle"], f"line {line} Idle"),
                "source": row["Source"], "region": _region(instruction),
            }
            key = (item["code_object_id"], item["vaddr"])
            if key in seen:
                raise ValueError("ATT stats CSV repeats a code-object/PC row")
            seen.add(key)
            rows.append(item)
    if not rows:
        raise ValueError("ATT stats CSV has no executed instruction PCs")
    code_objects = {row["code_object_id"] for row in rows}
    if len(code_objects) != 1:
        raise ValueError("ATT rows span more than one code object")
    wmma = [row for row in rows if row["region"] == "iu4_wmma"]
    signs = [row["instruction"] for row in wmma]
    if (len(wmma) != 8 or sum("neg_lo:[0,1,0]" in line for line in signs) != 4
            or sum("neg_lo:[1,1,0]" in line for line in signs) != 4):
        raise ValueError("ATT capture lacks the exact eight signed IU4 WMMA PCs")
    regions: dict[str, dict[str, int | float]] = {}
    for name in ("global_load_and_wait", "lds_and_barrier", "iu4_wmma",
                 "recombine_scale_accumulate", "other"):
        selected = [row for row in rows if row["region"] == name]
        hit = sum(row["hit"] for row in selected)
        regions[name] = {
            "pc_count": len(selected), "hit": hit,
            "latency": sum(row["latency"] for row in selected),
            "stall": sum(row["stall"] for row in selected),
            "idle": sum(row["idle"] for row in selected),
            "stall_per_hit": 0.0 if hit == 0 else sum(row["stall"] for row in selected) / hit,
            "latency_per_hit": 0.0 if hit == 0 else sum(row["latency"] for row in selected) / hit,
        }
    for required in ("global_load_and_wait", "lds_and_barrier", "iu4_wmma",
                     "recombine_scale_accumulate"):
        if regions[required]["pc_count"] == 0:
            raise ValueError(f"ATT capture lacks executed {required} PCs")
    hotspots = sorted(rows, key=lambda row: (row["stall"], row["latency"]), reverse=True)[:32]
    return rows, {"code_object_id": next(iter(code_objects)), "regions": regions,
                  "hotspots_by_stall": hotspots}


def _read_static(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    labels = list(re.finditer(
        rf"(?m)^([^\n:]*{KERNEL}[^\n:]*):\s*(?:;[^\n]*)?$", text))
    if len(labels) != 1:
        raise ValueError("emitted assembly lacks one exact production Q4 function label")
    end = text.find("\n.Lfunc_end", labels[0].end())
    if end < 0:
        raise ValueError("emitted production Q4 function is unterminated")
    body = text[labels[0].start():end]
    metadata_end = text.find("; Occupancy:", end)
    if metadata_end < 0:
        raise ValueError("emitted production Q4 metadata is incomplete")
    metadata = text[end:metadata_end + 80]
    def one(pattern: str, label: str) -> int:
        values = re.findall(pattern, metadata, flags=re.MULTILINE)
        if len(values) != 1:
            raise ValueError(f"emitted production Q4 {label} is ambiguous")
        return int(values[0])
    wmma = re.findall(r"(?m)^\s*v_wmma_i32_16x16x32_iu4[^\n]*$", body)
    if (len(wmma) != 8 or sum("neg_lo:[0,1,0]" in line for line in wmma) != 4
            or sum("neg_lo:[1,1,0]" in line for line in wmma) != 4):
        raise ValueError("emitted production Q4 static IU4 topology differs")
    result = {
        "symbol": labels[0].group(1), "iu4_wmma_pc_count": 8,
        "vgpr": one(r"^; NumVgprs:\s*(\d+)", "VGPR count"),
        "scratch_bytes": one(r"^; ScratchSize:\s*(\d+)", "scratch size"),
        "lds_bytes": one(r"^; LDSByteSize:\s*(\d+)", "LDS size"),
        "occupancy": one(r"^; Occupancy:\s*(\d+)", "occupancy"),
    }
    if result != {**result, "vgpr": 88, "scratch_bytes": 0,
                  "lds_bytes": 17152, "occupancy": 16}:
        raise ValueError("emitted production Q4 resource contract differs")
    return result


def _validate_capture_log(path: Path) -> None:
    log = path.read_text(encoding="utf-8", errors="replace").lower()
    if any(term in log for term in (
            "data loss", "data-loss", "data lost", "dropped trace", "stitch incomplete",
            "wave incomplete", "sqtt data buffer full", "thread trace buffer full",
            "invalid buffer size:", "sqtt buffer size too low", "sqtt buffer size too high",
            "failed to allocate tracing buffers",
            "unable to start thread trace worker thread")):
        raise ValueError("ATT decoder reported incomplete or lost trace data")


def analyze(plan_path: Path, source_database_path: Path, source_attribution_path: Path,
            att_database_path: Path, raw_directory_path: Path, capture_log_path: Path,
            benchmark_report_path: Path, executable_path: Path, artifact_path: Path,
            corpus_path: Path, assembly_path: Path, power_before_path: Path,
            power_after_path: Path, closure_path: Path) -> dict[str, Any]:
    att_stats_path, att_ui_path = _resolve_att_outputs(raw_directory_path, att_database_path)
    paths = {
        "plan": plan_path, "source_pmc_database": source_database_path,
        "source_pmc_attribution": source_attribution_path, "att_database": att_database_path,
        "att_stats": att_stats_path, "capture_log": capture_log_path,
        "benchmark_report": benchmark_report_path, "benchmark_executable": executable_path,
        "artifact": artifact_path, "corpus": corpus_path, "emitted_assembly": assembly_path,
        "power_before": power_before_path, "power_after": power_after_path,
        "prepared_closure": closure_path,
    }
    snapshots = {name: _snapshot(path, name) for name, path in paths.items()}
    snapshots["att_ui_directory"] = {
        "path": str(att_ui_path.resolve(strict=True)), "entry_count": len(list(att_ui_path.iterdir()))}
    tool_snapshots = {name: _snapshot(path, name) for name, path in TOOLS.items()}
    plan = _load_object(plan_path, "ATT plan")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "command_only_not_executed":
        raise ValueError("not the exact executable ATT plan")
    for name in ("source_pmc_database", "source_pmc_attribution", "benchmark_executable",
                 "artifact", "corpus", "emitted_assembly", "runner", "postprocessor",
                 "analyzer", "tester", "parser_checker", "invalid_overflow_attempt",
                 "invalid_command_attempt", "invalid_wrong_agent_attempt",
                 "memory_headroom_report"):
        if name not in snapshots:
            identity = plan.get("inputs", {}).get(name)
            if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
                raise ValueError(f"ATT plan lacks {name} identity")
            snapshots[name] = _snapshot(Path(identity["path"]), name)
        _identity(plan.get("inputs", {}).get(name), snapshots[name], name)
    _validate_closure(closure_path, {
        name: snapshots[name] for name in (
            "plan", "runner", "postprocessor", "analyzer", "tester", "parser_checker")})
    headroom = _load_object(Path(snapshots["memory_headroom_report"]["path"]),
                            "memory headroom report")
    if (headroom.get("memory", {}).get("available_after_startup_bytes") != 8262778880
            or headroom.get("config", {}).get("max_context") != 2048
            or headroom.get("config", {}).get("concurrency") != 1):
        raise ValueError("memory headroom report does not support the ATT buffer bound")
    expected_tools = plan.get("toolchain", {})
    if any(expected_tools.get(f"{name}_sha256") != value["sha256"]
           for name, value in tool_snapshots.items()):
        raise ValueError("installed ATT toolchain differs from the plan")
    selection = plan.get("selection")
    if not isinstance(selection, dict) or any((
        selection.get("kernel_match_iteration") != MATCH_ITERATION,
        selection.get("role") != "post_mixer/down",
        selection.get("region") != ROLE_REGION,
        selection.get("grid") != list(EXPECTED_GRID),
        selection.get("workgroup") != list(EXPECTED_WORKGROUP),
    )):
        raise ValueError("ATT plan selection is not exact post-mixer/down dispatch three")
    if power_before_path.read_text(encoding="utf-8").strip() != "auto" or (
            power_after_path.read_text(encoding="utf-8").strip() != "auto"):
        raise ValueError("ATT capture did not preserve the auto power profile")
    _validate_capture_log(capture_log_path)
    source_rows, source_process = _q4_dispatches(source_database_path)
    source_selected = _selected_dispatch(source_rows, require_full_inventory=True)
    if source_selected["dispatch_id"] != selection.get("source_pmc_dispatch_id"):
        raise ValueError("source PMC dispatch identity differs from the ATT plan")
    source_attribution = _load_object(source_attribution_path, "source PMC attribution")
    if (source_attribution.get("schema")
            != "ninfer.r9700.production_q4_p2048_pmc_attribution.v2"
            or source_attribution.get("status") != "valid_attribution_only"
            or source_attribution.get("profiler", {}).get("version") != "1.3.5"):
        raise ValueError("source PMC attribution is not the valid gfx1201 authority")
    att_rows, att_process = _q4_dispatches(att_database_path)
    att_selected = _selected_dispatch(att_rows, require_full_inventory=True)
    expected_command = plan.get("benchmark_command")
    if att_process["command"] != expected_command:
        raise ValueError("ATT database process command differs from the plan")
    if not isinstance(att_process["registered_library"], str) or not (
            att_process["registered_library"].endswith("librocprofiler-sdk.so.1.3.5")):
        raise ValueError("ATT database did not load rocprofiler SDK 1.3.5")
    report = _load_object(benchmark_report_path, "benchmark report")
    config = report.get("config", {})
    if (report.get("artifact_type") != "ninfer_bench_report"
            or report.get("command") != expected_command
            or report.get("environment", {}).get("gpu_name") != "AMD Radeon AI PRO R9700"
            or report.get("environment", {}).get("architecture_name") != "gfx1201"
            or any(config.get(key) != value for key, value in (
                ("max_context", 2048), ("prefill_chunk", 4096),
                ("kv_value_group", 16), ("concurrency", 1), ("spec", "none"),
                ("draft_tokens", 0), ("xattention_qualification", False),
                ("q4_prefill_cta_profile", "m64n128-pingpong-production"),
            ))):
        raise ValueError("benchmark report is not the exact selected C1/P2048 workload")
    _, att = _read_att(att_stats_path)
    static = _read_static(assembly_path)
    return {
        "schema": REPORT_SCHEMA, "status": "valid_attribution_only",
        "performance_authority": False,
        "selection": {**selection, "att_dispatch_id": att_selected["dispatch_id"],
                      "symbol": att_selected["symbol"],
                      "resources": {key: att_selected[key]
                                    for key in ("vgpr", "lds_bytes", "private_bytes")}},
        "att": att,
        "static_production_kernel": static,
        "interpretation_limit": (
            "One CU/SIMD sample localizes decoded-PC latency/stall/idle only; it is neither "
            "a device-wide fraction nor a bandwidth or timing-selection measurement."),
        "provenance": {**snapshots, "toolchain": tool_snapshots},
        "source_process": source_process,
        "att_process": att_process,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("plan", "source_database", "source_attribution", "att_database", "raw_directory",
                 "capture_log", "benchmark_report", "executable", "artifact", "corpus",
                 "assembly", "power_before", "power_after", "closure", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.plan, args.source_database, args.source_attribution,
                     args.att_database, args.raw_directory, args.capture_log,
                     args.benchmark_report, args.executable, args.artifact, args.corpus,
                     args.assembly, args.power_before, args.power_after, args.closure)
    if args.output.exists():
        raise ValueError("refusing to overwrite ATT attribution output")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Q4 single-dispatch ATT attribution: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
