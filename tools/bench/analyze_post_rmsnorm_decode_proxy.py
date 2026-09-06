#!/usr/bin/env python3
"""Validate and reduce the two-pass post-RMSNorm ordinary-decode PMC diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCRIPT_REPO = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO))

from tools.bench.prepare_post_rmsnorm_decode_proxy import PASSES, REPO, identity, sha


ROUNDS = 256
ONE_ROUND_DISPATCHES = 1806


def number(value: object, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} is invalid")
    return result


def percent(numerator: Decimal, denominator: Decimal, label: str) -> float:
    if numerator < 0 or denominator <= 0 or numerator > denominator:
        raise ValueError(f"{label} has invalid operands")
    return float(Decimal(100) * numerator / denominator)


def quotient(numerator: Decimal, denominator: Decimal, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{label} has a nonpositive denominator")
    result = float(numerator / denominator)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite")
    return result


def parse_stream_probe(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    gpu = re.search(r"^GPU:\s+(.+)$", text, re.MULTILINE)
    arch = re.search(r"^Architecture:\s+(gfx1201) \(wave32\)$", text, re.MULTILINE)
    working = re.search(r"^Working set:\s+([0-9.]+) GiB per buffer \(([0-9.]+)x L2\)$",
                        text, re.MULTILINE)
    policy = re.search(r"^Trial policy:\s+(\d+) trials,", text, re.MULTILINE)
    checksum = re.search(r"^Read checksum exact:\s+[0-9a-f]{8}(?::[0-9a-f]{8}){3}$",
                         text, re.MULTILINE)
    if (gpu is None or "R9700" not in gpu.group(1) or arch is None or working is None
            or policy is None or int(policy.group(1)) != 5 or checksum is None
            or float(working.group(1)) != 4.0 or float(working.group(2)) < 4.0):
        raise ValueError("stream probe lacks exact R9700/4-GiB/5-trial/checksum contract")
    methods = {}
    row = re.compile(r"^(hipMemsetAsync \(write\)|kernel uint4 write|kernel uint4 read|"
                     r"kernel uint4 copy|kernel uint4x4 copy|hipMemcpyAsync D2D)\s+"
                     r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)%\s+([0-9.]+)$",
                     re.MULTILINE)
    for match in row.finditer(text):
        methods[match.group(1)] = {"payload_gbps": float(match.group(2)),
                                   "best_bus_gbps": float(match.group(3)),
                                   "median_bus_gbps": float(match.group(4)),
                                   "reported_peak_percent": float(match.group(5)),
                                   "best_ms": float(match.group(6))}
    if len(methods) != 6 or any(not all(math.isfinite(value) and value > 0
                                        for value in method.values()) for method in methods.values()):
        raise ValueError("stream probe lacks six finite positive method rows")
    return {"stdout": identity(path), "working_set_gib_per_buffer": float(working.group(1)),
            "working_set_over_l2": float(working.group(2)), "trials_per_method": 5,
            "methods": methods, "best_observed_bus_gbps": max(
                method["best_bus_gbps"] for method in methods.values())}


def validate_closure(root: Path) -> None:
    expected = {Path(__file__).resolve(),
                (REPO / "tools/bench/prepare_post_rmsnorm_decode_proxy.py").resolve(),
                (root / "plan.json").resolve(), (root / "commands.sh").resolve()}
    observed = {}
    for line in (root / "prepared.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (REPO / relative).resolve()
        if len(digest) != 64 or path in observed:
            raise ValueError("prepared closure is malformed")
        observed[path] = digest
    if set(observed) != expected or any(not path.is_file() or sha(path) != digest
                                        for path, digest in observed.items()):
        raise ValueError("prepared closure bytes changed")


def one_table(connection: sqlite3.Connection, prefix: str) -> str:
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
    """Read the exact rocprofiler tables needed by this frozen diagnostic."""

    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        agents = one_table(connection, "rocpd_info_agent_")
        processes = one_table(connection, "rocpd_info_process_")
        dispatches = one_table(connection, "rocpd_kernel_dispatch_")
        symbols = one_table(connection, "rocpd_info_kernel_symbol_")
        marker_regions = one_table(connection, "rocpd_region_")
        events = one_table(connection, "rocpd_event_")
        strings = one_table(connection, "rocpd_string_")
        gpu = list(connection.execute(
            f'''select name, product_name, extdata from "{agents}"
                where type='GPU' and name='gfx1201' '''))
        if len(gpu) != 1 or gpu[0]["product_name"] != "AMD Radeon AI PRO R9700":
            raise ValueError("database lacks one R9700/gfx1201 agent")
        agent = json.loads(gpu[0]["extdata"])
        if (agent.get("cu_count") != 64 or agent.get("max_waves_per_cu") != 32
                or agent.get("wave_front_size") != 32 or agent.get("simd_count") != 128):
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


def scaled_trace_inventory(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("selected-region trace inventory is missing")
    result = []
    for row in rows:
        if (not isinstance(row, dict)
                or not isinstance(row.get("symbol"), str) or not row["symbol"]
                or any(type(row.get(name)) is not int or row[name] < 0 for name in (
                    "grid_size", "workgroup_size", "static_lds_bytes", "scratch_bytes", "vgpr"))
                or type(row.get("dispatch_count")) is not int or row["dispatch_count"] <= 0):
            raise ValueError("selected-region trace inventory is malformed")
        result.append({
            "symbol": row["symbol"], "grid_size": row["grid_size"],
            "workgroup_size": row["workgroup_size"],
            "static_lds_bytes": row["static_lds_bytes"],
            "allocated_lds_bytes": ((row["static_lds_bytes"] + 255) // 256) * 256,
            "scratch_bytes": row["scratch_bytes"], "vgpr": row["vgpr"],
            "dispatch_count": row["dispatch_count"] * ROUNDS,
        })
    if sum(row["dispatch_count"] for row in rows) != ONE_ROUND_DISPATCHES:
        raise ValueError("selected-region trace is not the exact 1806-dispatch round")
    return sorted(result, key=lambda row: (
        row["symbol"], row["grid_size"], row["workgroup_size"], row["static_lds_bytes"],
        row["allocated_lds_bytes"], row["scratch_bytes"], row["vgpr"],
    ))


def validate_benchmark(path: Path, command: list[str], plan: dict) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token:
                       (_ for _ in ()).throw(ValueError(f"invalid JSON number {token}")))
    config, test = value.get("config", {}), value.get("tests", [{}])[0]
    rep = test.get("reps", [{}])[0]
    lanes = rep.get("generated_token_ids_by_lane")
    if (value.get("artifact_type") != "ninfer_bench_report" or value.get("schema_version") != 20
            or value.get("command") != " ".join(command)
            or value.get("environment", {}).get("architecture_name") != "gfx1201"
            or config.get("concurrency") != 1 or config.get("spec") != "none"
            or config.get("draft_tokens") != 0 or config.get("prefill_chunk") != 4096
            or config.get("kv_cache_format") != "fp8-k-int4-v" or config.get("kv_value_group") != 16
            or config.get("xattention_qualification") is not False
            or config.get("use_device_graph") is not True or config.get("decode_path") != "device_graph"
            or test.get("kind") != "whole" or test.get("n_prompt") != 8192
            or test.get("n_gen") != 256 or test.get("requested_output_tokens") != 257
            or rep.get("generated_output_tokens") != 257 or rep.get("decode_output_tokens") != 256
            or rep.get("decode_engine_tokens") != 256 or lanes != [plan["terminal_timing"]["generated_token_ids"]]):
        raise ValueError("profile benchmark differs from exact terminal ordinary workload")
    return identity(path)


def reduce_pass(csv_path: Path, database: Path, command: list[str], counters: tuple[str, ...]) -> dict:
    regions, agent, marker_counts = read_database(database, command)
    expected_markers = {f"ninfer.decode.decode.ordinary_round payload={frontier}": 1
                        for frontier in range(8192, 8448)}
    observed = {name: count for name, count in marker_counts.items()
                if ".decode.ordinary_round " in name}
    if observed != expected_markers:
        raise ValueError("capture lacks exact ordinary frontiers 8192..8447")
    required = {"Dispatch_Id", "Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Block_Size",
                "Scratch_Size", "VGPR_Count", "Counter_Name", "Counter_Value"}
    values: dict[tuple[int, str], Decimal] = {}
    metadata: dict[int, tuple] = {}
    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("counter CSV lacks native fields")
        for line, row in enumerate(reader, 2):
            dispatch = int(row["Dispatch_Id"])
            item = (row["Kernel_Name"], int(row["Grid_Size"]), int(row["Workgroup_Size"]),
                    int(row["LDS_Block_Size"]), int(row["Scratch_Size"]), int(row["VGPR_Count"]))
            if metadata.setdefault(dispatch, item) != item:
                raise ValueError("counter CSV has inconsistent dispatch metadata")
            counter = row["Counter_Name"]
            if counter not in counters or (dispatch, counter) in values:
                raise ValueError(f"line {line} has unexpected or duplicate counter")
            values[dispatch, counter] = number(row["Counter_Value"], f"line {line}")
    selected = sorted(dispatch for dispatch, item in regions.items()
                      if item["region"] in expected_markers)
    if not selected or set(metadata) != set(regions):
        raise ValueError("CSV/database measured dispatch inventories differ")
    for dispatch in metadata:
        if {counter for item, counter in values if item == dispatch} != set(counters):
            raise ValueError("dispatch lacks exact pass counter inventory")
        db, row = regions[dispatch], metadata[dispatch]
        if (row[0] != db["symbol"] or row[1] != db["grid_size"]
                or row[2] != db["workgroup_size"] or row[4] != db["scratch_bytes"]
                or row[5] != db["vgpr"]
                or row[3] != ((db["static_lds_bytes"] + 255) // 256) * 256):
            raise ValueError("CSV/database per-dispatch resources differ")
    totals = {counter: sum((values[d, counter] for d in selected), Decimal(0))
              for counter in counters}
    inventory = Counter((metadata[d][0], metadata[d][1], metadata[d][2],
                         regions[d]["static_lds_bytes"], metadata[d][3], metadata[d][4],
                         metadata[d][5]) for d in selected)
    per_symbol: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {counter: Decimal(0) for counter in counters})
    for dispatch in selected:
        for counter in counters:
            per_symbol[metadata[dispatch][0]][counter] += values[dispatch, counter]
    return {
        "ordinary_dispatch_count": len(selected), "agent": agent,
        "dispatch_inventory": [{"symbol": key[0], "grid_size": key[1],
                                "workgroup_size": key[2], "static_lds_bytes": key[3],
                                "allocated_lds_bytes": key[4], "scratch_bytes": key[5],
                                "vgpr": key[6], "dispatch_count": count}
                               for key, count in sorted(inventory.items())],
        "counter_sums": {name: str(value) for name, value in totals.items()},
        "per_symbol_counter_sums": {symbol: {name: str(value) for name, value in sums.items()}
                                    for symbol, sums in sorted(per_symbol.items())},
    }


def analyze(plan_path: Path, root: Path) -> dict:
    root, plan_path = root.resolve(strict=True), plan_path.resolve(strict=True)
    if plan_path != root / "plan.json":
        raise ValueError("plan must be package-root plan.json")
    validate_closure(root)
    raw = plan_path.read_bytes(); plan = json.loads(raw)
    if (plan.get("artifact_type") != "ninfer_r9700_post_rmsnorm_decode_proxy_plan"
            or plan.get("schema_version") != 1 or plan.get("profile_timing_admissible") is not False
            or set(plan.get("passes", {})) != set(PASSES)):
        raise ValueError("profile plan contract differs")
    for field in ("report", "executable"):
        pinned = plan["terminal_timing"][field]
        if identity(Path(pinned["path"])) != pinned:
            raise ValueError(f"terminal {field} changed")
    for field in ("artifact", "corpus"):
        pinned = plan[field]
        if identity(Path(pinned["path"])) != pinned:
            raise ValueError(f"{field} changed")
    for field in ("profiler", "counter_preflight"):
        pinned = plan[field]
        if identity(Path(pinned["path"])) != pinned:
            raise ValueError(f"{field} changed")
    probe_contract = plan.get("stream_probe", {})
    probe_executable = probe_contract.get("executable", {})
    if (identity(Path(str(probe_executable.get("path", "")))) != probe_executable
            or probe_contract.get("arguments") != ["--size-gib", "4", "--trials", "5"]
            or probe_contract.get("power_profile") != "auto"
            or probe_contract.get("working_set_gib_per_buffer") != 4
            or probe_contract.get("trials_per_method") != 5):
        raise ValueError("stream probe contract or executable changed")
    expected_inventory = scaled_trace_inventory(
        plan.get("selected_region_trace", {}).get("inventory")
    )
    expected_dispatches = ONE_ROUND_DISPATCHES * ROUNDS
    summaries = {}
    for label, counters in PASSES.items():
        record = plan["passes"][label]
        sq_count = sum(name.startswith(("SQ_", "SQC_")) for name in counters)
        if tuple(record.get("counters", ())) != counters or sq_count > 8:
            raise ValueError("pass counter contract differs")
        before, after = root / f"power-{label}-before.txt", root / f"power-{label}-after.txt"
        if before.read_text().strip() != "profile_standard" or after.read_text().strip() != "auto":
            raise ValueError("capture lacks profile_standard/auto endpoint evidence")
        command = record["benchmark_command"]
        benchmark = validate_benchmark(Path(record["benchmark_report"]), command, plan)
        raw_dir = Path(record["raw_directory"])
        csvs = list(raw_dir.glob("*_counter_collection.csv")); dbs = list(raw_dir.glob("*_results.db"))
        if len(csvs) != 1 or len(dbs) != 1:
            raise ValueError("pass lacks exactly one counter CSV and database")
        summary = reduce_pass(csvs[0], dbs[0], command, counters)
        if (summary["ordinary_dispatch_count"] != expected_dispatches
                or summary["dispatch_inventory"] != expected_inventory):
            raise ValueError(
                "PMC pass is not the exact 256x selected one-round dispatch inventory"
            )
        summary["benchmark_report"] = benchmark
        summary["counter_csv"] = identity(csvs[0]); summary["database"] = identity(dbs[0])
        summaries[label] = summary
    if summaries["cache-wait"]["dispatch_inventory"] != summaries["issue-lds"]["dispatch_inventory"]:
        raise ValueError("two PMC passes have different dispatch inventory multisets")
    a = {name: Decimal(value) for name, value in summaries["cache-wait"]["counter_sums"].items()}
    b = {name: Decimal(value) for name, value in summaries["issue-lds"]["counter_sums"].items()}
    agent = summaries["cache-wait"]["agent"]
    if summaries["issue-lds"]["agent"] != agent:
        raise ValueError("two PMC passes report different GPU execution geometry")
    occupancy_denominator = (a["GRBM_GUI_ACTIVE"] * Decimal(agent["cu_count"])
                             * Decimal(agent["max_waves_per_cu"]))
    simd_quarters = Decimal(agent["simd_count"] // 4)
    if (a["SQ_WAVE_CYCLES"] <= 0 or a["TCP_REQ"] <= 0
            or a["GL2C_HIT"] + a["GL2C_MISS"] <= 0 or a["GRBM_GUI_ACTIVE"] <= 0):
        raise ValueError("cache/wait pass has nonpositive proxy denominators")
    if a["TCP_REQ_MISS"] > a["TCP_REQ"]:
        raise ValueError("cache/wait pass has more TCP misses than requests")
    issue = b["SQ_INST_CYCLES_VALU"] + b["SQ_INST_CYCLES_VMEM"]
    if (issue <= 0 or b["SQC_LDS_IDX_ACTIVE"] <= 0 or b["SQ_WAVES"] <= 0
            or b["GRBM_GUI_ACTIVE"] <= 0):
        raise ValueError("issue/LDS pass has nonpositive proxy denominators")
    for label in PASSES:
        summaries[label]["instance_reduction"] = (
            "rocprofiler emits one native hardware-instance-reduced value per dispatch/counter; "
            "the analyzer sums those per-dispatch values, never takes one global maximum")
    stream_before = (root / "power-stream-before.txt").read_text().strip()
    stream_after = (root / "power-stream-after.txt").read_text().strip()
    if stream_before != "auto" or stream_after != "auto":
        raise ValueError("stream probe lacks auto/auto endpoint evidence")
    probe = parse_stream_probe(root / "stream-probe.stdout")
    probe["stderr"] = identity(root / "stream-probe.stderr")
    probe["power_before"] = "auto"; probe["power_after"] = "auto"
    result = {
        "artifact_type": "ninfer_r9700_post_rmsnorm_decode_proxy_analysis", "schema_version": 1,
        "status": "valid_diagnostic_proxy_only", "profile_timing_admissible": False,
        "plan": identity(plan_path), "terminal_timing": plan["terminal_timing"],
        "selected_region_trace": plan["selected_region_trace"], "passes": summaries,
        "proxy_metrics": {
            "gl2_hit_percent": percent(a["GL2C_HIT"], a["GL2C_HIT"] + a["GL2C_MISS"], "GL2 hit"),
            "tcp_hit_percent": percent(a["TCP_REQ"] - a["TCP_REQ_MISS"], a["TCP_REQ"], "TCP hit"),
            "sq_wait_any_over_wave_cycles_percent": percent(a["SQ_WAIT_ANY"], a["SQ_WAVE_CYCLES"], "SQ wait"),
            "sq_issue_wait_over_wave_cycles_percent": percent(a["SQ_WAIT_INST_ANY"], a["SQ_WAVE_CYCLES"], "SQ issue wait"),
            "occupancy_proxy_percent": percent(a["SQ_WAVE_CYCLES"], occupancy_denominator, "occupancy"),
            "relative_gl2_read_requests_per_output_token": float(a["GL2C_EA_RDREQ"] / Decimal(256)),
            "valu_busy_proxy_percent": percent(b["SQ_INST_CYCLES_VALU"], b["GRBM_GUI_ACTIVE"] * simd_quarters, "VALU busy"),
            "valu_issue_cycle_share_percent": percent(b["SQ_INST_CYCLES_VALU"], issue, "VALU issue share"),
            "vmem_issue_cycle_share_percent": percent(b["SQ_INST_CYCLES_VMEM"], issue, "VMEM issue share"),
            "lds_bank_conflict_over_index_active": float(b["SQC_LDS_BANK_CONFLICT"] / b["SQC_LDS_IDX_ACTIVE"]),
            "lds_utilization_proxy_percent": percent(b["SQC_LDS_IDX_ACTIVE"], b["GRBM_GUI_ACTIVE"] * simd_quarters, "LDS utilization"),
            "ta_busy_instance_sum_per_grbm_active_cycle_cache_wait": quotient(a["TA_TA_BUSY"], a["GRBM_GUI_ACTIVE"], "TA pressure pass A"),
            "ta_busy_instance_sum_per_grbm_active_cycle_issue_lds": quotient(b["TA_TA_BUSY"], b["GRBM_GUI_ACTIVE"], "TA pressure pass B"),
            "gl2_write_stall_instance_sum_per_grbm_active_cycle": quotient(a["GL2C_MC_WRREQ_STALL"], a["GRBM_GUI_ACTIVE"], "GL2 write stall"),
            "lds_instructions_per_output_token": float(b["SQ_INSTS_LDS"] / Decimal(256)),
            "tex_load_instructions_per_output_token": float(b["SQ_INSTS_TEX_LOAD"] / Decimal(256)),
            "tex_store_instructions_per_output_token": float(b["SQ_INSTS_TEX_STORE"] / Decimal(256)),
        },
        "same_session_auto_stream_probe": probe,
        "represented_minimum_bytes_per_round": None,
        "represented_minimum_byte_rate_gbps": None,
        "represented_minimum_rate_fraction_of_stream_probe": None,
        "represented_rate_limitation": "The heterogeneous whole-round dispatch inventory does not carry an exact represented-byte contract for every kernel; no minimum byte rate or stream-ceiling fraction is inferred.",
        "roof_decision": "proxy_review_required_no_physical_roof_conclusion",
        "physical_memory_bandwidth_bytes_per_second": None, "physical_peak_fraction": None,
        "stall_freedom": None,
        "conclusion_contract": "Per-dispatch native counters are reduced as ratio-of-sums within a co-collected pass. They can expose relative cache, request, wait, issue, or LDS pressure, but cannot prove physical GDDR6 bandwidth or absence of material stalls.",
    }
    if hashlib.sha256(plan_path.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise ValueError("plan changed during analysis")
    return result


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
            json.dump(result, output, indent=2, sort_keys=True, allow_nan=False); output.write("\n")
        os.link(temporary, args.out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
