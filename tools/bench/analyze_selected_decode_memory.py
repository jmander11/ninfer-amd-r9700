#!/usr/bin/env python3
"""Validate selected ordinary-decode cache/wait captures and report defensible proxies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tools.bench.prepare_selected_decode_memory_profile import (
    COUNTERS, REPO, validate_profile_build_receipt,
)
from tools.bench.run_ninfer_bench_matrix import REPORT_SCHEMA_VERSION

ROUNDS = 256
FIRST_FRONTIER = 8192


def expected_round_regions() -> dict[str, int]:
    return {
        f"ninfer.decode.decode.ordinary_round payload={frontier}": 1
        for frontier in range(FIRST_FRONTIER, FIRST_FRONTIER + ROUNDS)
    }


def expected_dispatch_regions() -> set[str]:
    return set(expected_round_regions())


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_prepared_closure(root: Path, plan_path: Path) -> None:
    closure_path = root / "prepared.sha256"
    entries: dict[Path, str] = {}
    for line_number, line in enumerate(closure_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"prepared closure line {line_number} is malformed")
        try:
            int(parts[0], 16)
        except ValueError as error:
            raise ValueError(f"prepared closure line {line_number} has an invalid digest") from error
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"prepared closure line {line_number} has an invalid path")
        path = (REPO / relative).resolve()
        if path in entries:
            raise ValueError("prepared closure has a duplicate path")
        entries[path] = parts[0]
    required = {
        plan_path.resolve(), Path(__file__).resolve(),
        (REPO / "tools/bench/prepare_selected_decode_memory_profile.py").resolve(),
    }
    if not required.issubset(entries):
        raise ValueError("prepared closure lacks required profile authorities")
    for path, digest in entries.items():
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"prepared closure bytes changed: {path}")


def number(value: object, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} is invalid")
    return result


def ratio(a: Decimal, b: Decimal, label: str) -> float:
    if b <= 0:
        raise ValueError(f"{label} has a nonpositive denominator")
    value = float(Decimal(100) * a / b)
    if not math.isfinite(value) or value > 100:
        raise ValueError(f"{label} is invalid")
    return value


def tables(connection: sqlite3.Connection, prefix: str) -> str:
    rows = [row[0] for row in connection.execute(
        "select name from sqlite_master where type='table' and name like ?", (prefix + "%",))]
    if len(rows) != 1:
        raise ValueError(f"database lacks one {prefix} table")
    return rows[0]


def read_marker_counts(connection: sqlite3.Connection, marker_regions: str,
                       events: str, strings: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    query = f'''select e.extdata from "{marker_regions}" m
                join "{events}" e on e.id=m.event_id
                join "{strings}" n on n.id=m.name_id
                join "{strings}" c on c.id=e.category_id
                where n.string='roctxThreadRangeA'
                  and c.string='MARKER_CORE_RANGE_API' '''
    for row in connection.execute(query):
        try:
            extdata = json.loads(row["extdata"])
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("database has malformed ROCTX range metadata") from error
        message = extdata.get("message")
        if not isinstance(message, str):
            raise ValueError("database ROCTX range lacks a message")
        counts[message] += 1
    return dict(counts)


def read_database(path: Path, command: list[str]) -> tuple[dict[int, dict], dict, dict[str, int]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        agents = tables(connection, "rocpd_info_agent_")
        processes = tables(connection, "rocpd_info_process_")
        dispatches = tables(connection, "rocpd_kernel_dispatch_")
        symbols = tables(connection, "rocpd_info_kernel_symbol_")
        marker_regions = tables(connection, "rocpd_region_")
        events = tables(connection, "rocpd_event_")
        strings = tables(connection, "rocpd_string_")
        gpu = list(connection.execute(
            f'''select name, product_name, extdata from "{agents}"
                where type='GPU' and name='gfx1201' '''))
        if len(gpu) != 1 or gpu[0]["product_name"] != "AMD Radeon AI PRO R9700":
            raise ValueError("database lacks one R9700/gfx1201 agent")
        agent = json.loads(gpu[0]["extdata"])
        if (
            agent.get("cu_count") != 64 or agent.get("max_waves_per_cu") != 32
            or agent.get("wave_front_size") != 32 or agent.get("simd_count") != 128
        ):
            raise ValueError("database R9700 execution resources differ")
        process = list(connection.execute(f'''select command, environment from "{processes}"'''))
        if len(process) != 1 or process[0]["command"] != " ".join(command):
            raise ValueError("database process command differs from plan")
        environment = json.loads(process[0]["environment"])
        if not str(environment.get("ROCPROFILER_REGISTER_LIBRARY", "")).endswith(
            "librocprofiler-sdk.so.1.3.5"
        ):
            raise ValueError("database does not bind rocprofiler SDK 1.3.5")
        regions = {}
        query = f'''select k.dispatch_id, r.string region, s.display_name symbol,
                           k.grid_size_x, k.grid_size_y, k.grid_size_z,
                           k.workgroup_size_x, k.workgroup_size_y, k.workgroup_size_z,
                           k.group_segment_size, k.private_segment_size,
                           s.arch_vgpr_count, s.accum_vgpr_count
                    from "{dispatches}" k join "{symbols}" s on s.id=k.kernel_id
                    left join "{strings}" r on r.id=k.region_name_id'''
        for row in connection.execute(query):
            dispatch = int(row["dispatch_id"])
            metadata = {
                "region": str(row["region"]) if row["region"] is not None else None,
                "symbol": str(row["symbol"]),
                "grid_size": int(row["grid_size_x"] * row["grid_size_y"] * row["grid_size_z"]),
                "workgroup_size": int(row["workgroup_size_x"] * row["workgroup_size_y"]
                                      * row["workgroup_size_z"]),
                "static_lds_bytes": int(row["group_segment_size"]),
                "scratch_bytes": int(row["private_segment_size"]),
                "vgpr": int(row["arch_vgpr_count"] + row["accum_vgpr_count"]),
            }
            if dispatch in regions and regions[dispatch] != metadata:
                raise ValueError("database has inconsistent dispatch metadata")
            regions[dispatch] = metadata
        marker_counts = read_marker_counts(connection, marker_regions, events, strings)
        return regions, {
            "name": gpu[0]["name"], "product_name": gpu[0]["product_name"],
            "cu_count": agent["cu_count"], "simd_count": agent["simd_count"],
            "max_waves_per_cu": agent["max_waves_per_cu"],
            "wave_front_size": agent["wave_front_size"],
        }, marker_counts
    finally:
        connection.close()


def summarize(csv_path: Path, database: Path, command: list[str]) -> dict:
    regions, agent, marker_counts = read_database(database, command)
    expected_markers = expected_round_regions()
    observed_markers = {name: count for name, count in marker_counts.items()
                        if ".decode.ordinary_round " in name}
    if observed_markers != expected_markers:
        raise ValueError("marker trace lacks exact ordinary frontiers 8192..8447")
    required = {"Dispatch_Id", "Kernel_Name", "Grid_Size", "Workgroup_Size",
                "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Counter_Name",
                "Counter_Value"}
    values: dict[tuple[int, str], Decimal] = {}
    csv_metadata: dict[int, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("counter CSV lacks native fields")
        for line, row in enumerate(reader, 2):
            dispatch = int(row["Dispatch_Id"])
            metadata = {
                "symbol": row["Kernel_Name"], "grid_size": int(row["Grid_Size"]),
                "workgroup_size": int(row["Workgroup_Size"]),
                "allocated_lds_bytes": int(row["LDS_Block_Size"]),
                "scratch_bytes": int(row["Scratch_Size"]), "vgpr": int(row["VGPR_Count"]),
            }
            if csv_metadata.setdefault(dispatch, metadata) != metadata:
                raise ValueError("counter CSV has inconsistent dispatch metadata")
            counter = row["Counter_Name"]
            if counter not in COUNTERS or (dispatch, counter) in values:
                raise ValueError("counter CSV has unexpected or duplicate rows")
            values[dispatch, counter] = number(row["Counter_Value"], f"line {line}")
    # The outer range is emitted around each production Device Graph replay and its transaction
    # resolution. Nested layer ranges execute only while the graph is captured and therefore do
    # not identify replayed kernels.
    counter_dispatches = {dispatch for dispatch, _counter in values}
    if counter_dispatches != set(regions):
        raise ValueError("counter CSV and traced measured-region dispatch inventories differ")
    for dispatch in counter_dispatches:
        if {name for item, name in values if item == dispatch} != set(COUNTERS):
            raise ValueError("measured dispatch lacks exact counter inventory")
        db = regions[dispatch]
        csv_value = csv_metadata[dispatch]
        if (
            csv_value["symbol"] != db["symbol"]
            or csv_value["grid_size"] != db["grid_size"]
            or csv_value["workgroup_size"] != db["workgroup_size"]
            or csv_value["scratch_bytes"] != db["scratch_bytes"]
            or csv_value["vgpr"] != db["vgpr"]
            or csv_value["allocated_lds_bytes"]
            != ((db["static_lds_bytes"] + 255) // 256) * 256
        ):
            raise ValueError("counter CSV and database dispatch metadata differ")
    dispatch_regions = expected_dispatch_regions()
    decode = sorted(
        dispatch for dispatch, metadata in regions.items()
        if metadata["region"] in dispatch_regions
    )
    if not decode:
        raise ValueError("capture has no exact ROCTX ordinary-decode round dispatches")
    totals = {name: sum((values[d, name] for d in decode), Decimal(0)) for name in COUNTERS}
    for name in ("SQ_WAVES", "SQ_WAVE_CYCLES", "GRBM_GUI_ACTIVE", "GL2C_EA_RDREQ", "TCP_REQ"):
        if totals[name] <= 0:
            raise ValueError(f"decode capture has nonpositive required metric {name}")
    if totals["GL2C_HIT"] + totals["GL2C_MISS"] <= 0:
        raise ValueError("decode capture has zero GL2 activity")
    if totals["TCP_REQ_MISS"] > totals["TCP_REQ"]:
        raise ValueError("decode TCP misses exceed requests")
    inventory: dict[tuple, int] = defaultdict(int)
    for dispatch in decode:
        item = csv_metadata[dispatch]
        database_item = regions[dispatch]
        inventory[(item["symbol"], item["grid_size"], item["workgroup_size"],
                   database_item["static_lds_bytes"], item["allocated_lds_bytes"],
                   item["scratch_bytes"], item["vgpr"])] += 1
    return {
        "ordinary_decode_model_layer_dispatch_count": len(decode),
        "measured_dispatch_count": len(counter_dispatches),
        "ignored_non_ordinary_round_dispatch_count": len(counter_dispatches) - len(decode),
        "ordinary_round_marker_inventory": observed_markers,
        "dispatch_resource_inventory": [{
            "symbol": key[0], "grid_size": key[1], "workgroup_size": key[2],
            "static_lds_bytes": key[3], "allocated_lds_bytes": key[4],
            "scratch_bytes": key[5], "vgpr": key[6],
            "dispatch_count": count,
        } for key, count in sorted(inventory.items())],
        "counter_sums": {name: str(totals[name]) for name in COUNTERS},
        "proxy_metrics": {
            "gl2_hit_percent": ratio(totals["GL2C_HIT"], totals["GL2C_HIT"] + totals["GL2C_MISS"], "GL2 hit"),
            "l0_vector_cache_hit_percent": ratio(totals["TCP_REQ"] - totals["TCP_REQ_MISS"], totals["TCP_REQ"], "L0 hit"),
            "wave_dependency_wait_percent": ratio(totals["SQ_WAIT_ANY"], totals["SQ_WAVE_CYCLES"], "dependency wait"),
            "wave_issue_wait_percent": ratio(totals["SQ_WAIT_INST_ANY"], totals["SQ_WAVE_CYCLES"], "issue wait"),
            "occupancy_percent": ratio(
                totals["SQ_WAVE_CYCLES"],
                totals["GRBM_GUI_ACTIVE"] * Decimal(agent["cu_count"])
                * Decimal(agent["max_waves_per_cu"]), "occupancy",
            ),
            "relative_gl2_read_requests": str(totals["GL2C_EA_RDREQ"]),
            "gl2_write_stall_instance_sum_per_grbm_active_cycle": float(totals["GL2C_MC_WRREQ_STALL"] / totals["GRBM_GUI_ACTIVE"]),
        },
        "agent": agent,
    }


def validate_benchmark_report(path: Path, record: dict, plan: dict, concurrency: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    config = value.get("config", {})
    tests = value.get("tests", [])
    environment = value.get("environment", {})
    artifact = value.get("artifact", {})
    load = value.get("load", {})
    selected_xattention = plan["selected_route"]["execution_profile"]["xattention_profile"]
    xattention_matches = (
        config.get("xattention_qualification") is False
        and not any(key in config for key in (
            "xattention_profile", "xattention_find_block", "xattention_stride",
            "xattention_tau_permille",
        ))
    ) if selected_xattention == "dense" else (
        config.get("xattention_qualification") is True
        and config.get("xattention_profile") == selected_xattention
        and config.get("xattention_find_block") == 128
        and config.get("xattention_stride") == 16
        and config.get("xattention_tau_permille") == 900
    )
    if (
        value.get("artifact_type") != "ninfer_bench_report"
        or value.get("schema_version") != REPORT_SCHEMA_VERSION
        or value.get("command") != " ".join(record["benchmark_command"])
        or environment.get("gpu_name") != "AMD Radeon AI PRO R9700"
        or environment.get("architecture_name") != "gfx1201"
        or config.get("concurrency") != concurrency
        or config.get("spec") != "none"
        or config.get("draft_tokens") != 0
        or config.get("speculative_execution") is not False
        or config.get("proposal_head") != "full"
        or config.get("retain_token_ids") is not True
        or config.get("use_device_graph") is not True
        or config.get("decode_path") != "device_graph"
        or config.get("prefill_chunk") != plan["selected_route"]["prefill_chunk"]
        or config.get("kv_value_group")
        != plan["selected_route"]["cache_profile"]["value_group"]
        or config.get("corpus_path") != plan["corpus"]["path"]
        or config.get("corpus_tokens") != plan["corpus"]["tokens"]
        or any(config.get(key) != expected for key, expected in plan["compiled_route"].items())
        or not xattention_matches
        or config.get("repetitions") != 1
        or config.get("warmup") != 1
        or len(tests) != 1
        or not isinstance(tests[0], dict)
        or tests[0].get("kind") != "whole"
        or tests[0].get("n_prompt") != 8192
        or tests[0].get("n_gen") != 256
        or tests[0].get("requested_output_tokens") != 257
        or type(tests[0].get("decode_output_tok_s_mean")) not in (int, float)
        or not math.isfinite(tests[0]["decode_output_tok_s_mean"])
        or tests[0]["decode_output_tok_s_mean"] <= 0
        or tests[0].get("speculative", {}).get("enabled") is not False
        or not isinstance(tests[0].get("reps"), list)
        or len(tests[0]["reps"]) != 1
        or tests[0]["reps"][0].get("generated_output_tokens") != 257
        or artifact.get("path") != plan["artifact"]["path"]
        or artifact.get("file_size_bytes") != plan["artifact"]["file_size_bytes"]
        or load.get("weights_id") != plan["artifact"]["weights_id"]
        or load.get("target") != "qwen3_8_27b_r9700"
    ):
        raise ValueError(f"C{concurrency} benchmark report differs from selected ordinary route")
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "decode_output_tok_s": tests[0].get("decode_output_tok_s_mean"),
    }


def analyze(plan_path: Path, root: Path) -> dict:
    plan_path = plan_path.resolve(strict=True)
    root = root.resolve(strict=True)
    if plan_path != root / "plan.json":
        raise ValueError("decode profile plan must be the selected capture root plan")
    validate_prepared_closure(root, plan_path)
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    if (
        plan.get("artifact_type") != "ninfer_r9700_selected_decode_memory_profile_plan"
        or plan.get("schema_version") != 1 or plan.get("counters") != list(COUNTERS)
        or plan.get("workload", {}).get("concurrency") != [1, 2, 3, 4]
        or plan.get("workload", {}).get("spec") != "none"
        or set(plan.get("runs", {})) != {"1", "2", "3", "4"}
        or plan.get("corpus", {}).get("tokens") != 65536
        or plan.get("profile_timing_admissible") is not False
    ):
        raise ValueError("decode profile plan contract differs")
    receipt_identity = plan.get("profile_build_receipt", {})
    receipt_path = Path(str(receipt_identity.get("path", "")))
    if (
        not receipt_path.is_absolute() or not receipt_path.is_file()
        or receipt_path.stat().st_size != receipt_identity.get("file_size_bytes")
        or sha(receipt_path) != receipt_identity.get("sha256")
    ):
        raise ValueError("decode profile instrumentation receipt bytes changed")
    receipt, profile_executable, cache_path = validate_profile_build_receipt(
        receipt_path,
        selection_path=Path(plan["terminal_selection"]["path"]),
        selection_sha256=plan["terminal_selection"]["sha256"],
        manifest_path=Path(plan["source_matrix"]["path"]),
        manifest_sha256=plan["source_matrix"]["sha256"],
        artifact=plan["artifact"],
        terminal_executable=plan["terminal_timing_executable"],
        marker_source=plan["ordinary_round_marker_source"],
        selected_route=plan["selected_route"], compiled_route=plan["compiled_route"],
        hybrid_workspace_authority=plan.get("hybrid_shared_workspace_authority"),
    )
    if (
        receipt.get("instrumentation_executable")
        != plan.get("profile_benchmark_executable")
        or receipt.get("cmake_cache") != plan.get("profile_cmake_cache")
        or receipt.get("compile_database") != plan.get("profile_compile_database")
        or Path(plan["profile_benchmark_executable"]["path"]).resolve()
        != profile_executable
        or Path(plan["profile_cmake_cache"]["path"]).resolve() != cache_path
    ):
        raise ValueError("decode profile plan differs from instrumentation receipt")
    for key in ("profiler", "counter_catalog"):
        tool = plan.get(key, {})
        path = Path(str(tool.get("path", "")))
        if (
            not path.is_absolute() or not path.is_file()
            or path.stat().st_size != tool.get("file_size_bytes")
            or sha(path) != tool.get("sha256")
        ):
            raise ValueError(f"decode profile {key} bytes changed")
    corpus = plan["corpus"]
    corpus_path = Path(corpus["path"])
    if (not corpus_path.is_file() or corpus_path.stat().st_size != corpus["file_size_bytes"]
            or sha(corpus_path) != corpus["sha256"]):
        raise ValueError("decode profile corpus bytes changed")
    summaries = {}
    for concurrency in range(1, 5):
        record = plan["runs"][str(concurrency)]
        if not record.get("benchmark_command") or Path(
            record["benchmark_command"][0]
        ).resolve() != profile_executable:
            raise ValueError("decode capture command does not use instrumentation executable")
        prefix = f"selected-decode-c{concurrency}"
        raw = root / f"raw-c{concurrency}"
        before = root / f"power-c{concurrency}-before.txt"
        after = root / f"power-c{concurrency}-after.txt"
        if before.read_text().strip() != "profile_standard" or after.read_text().strip() != "auto":
            raise ValueError("decode capture lacks profile_standard/auto endpoint evidence")
        report_path = Path(record["output"]).resolve(strict=True)
        benchmark = validate_benchmark_report(report_path, record, plan, concurrency)
        summary = summarize(raw / f"{prefix}_counter_collection.csv",
                            raw / f"{prefix}_results.db", record["benchmark_command"])
        summary["benchmark_report"] = benchmark
        summary["proxy_metrics"]["relative_gl2_read_requests_per_output_token"] = float(
            Decimal(summary["counter_sums"]["GL2C_EA_RDREQ"])
            / Decimal(256 * concurrency)
        )
        summary["counter_csv"] = {"path": str(raw / f"{prefix}_counter_collection.csv"),
                                  "sha256": sha(raw / f"{prefix}_counter_collection.csv")}
        summary["database"] = {"path": str(raw / f"{prefix}_results.db"),
                               "sha256": sha(raw / f"{prefix}_results.db")}
        summaries[f"c{concurrency}"] = summary
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != hashlib.sha256(plan_raw).hexdigest():
        raise ValueError("decode profile plan changed while analyzing")
    return {
        "artifact_type": "ninfer_r9700_selected_decode_memory_proxy_evidence",
        "schema_version": 1, "status": "valid_proxy_attribution_only",
        "profile_timing_admissible": False,
        "plan": {"path": str(plan_path.resolve()), "sha256": sha(plan_path)},
        "terminal_timing_executable": plan["terminal_timing_executable"],
        "profile_benchmark_executable": plan["profile_benchmark_executable"],
        "profile_build_receipt": plan["profile_build_receipt"],
        "selected_route": plan["selected_route"], "concurrency": summaries,
        "physical_memory_bandwidth_bytes_per_second": None,
        "physical_peak_fraction": None,
        "stall_freedom": None,
        "conclusion_contract": "These model-layer counters can identify cache or broad wait pressure and C-scaling, but cannot prove near-peak physical GDDR6 bandwidth or absence of unexplained stalls.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    try:
        result = analyze(args.plan, args.root)
    except (OSError, sqlite3.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.out.name}.", dir=args.out.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        os.link(temporary, args.out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
