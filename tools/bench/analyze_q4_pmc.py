#!/usr/bin/env python3
"""Validate and summarize the exact production-Q4 P2048 dispatch-PMC capture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence


PLAN_SCHEMA = "ninfer.r9700.production_q4_p2048_pmc_two_pass_plan.v1"
REPORT_SCHEMA = "ninfer.r9700.production_q4_p2048_pmc_attribution.v2"
KERNEL = "a8q4g64_linear_prefill_cta_kernel"
PASS_A_SUM_COUNTERS = {
    "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_WAVE_CYCLES", "SQ_WAVES",
    "GL2C_HIT", "GL2C_MISS", "TCP_REQ", "TCP_REQ_MISS", "GL2C_EA_RDREQ",
}
PASS_A_INSTANCE_SUM_COUNTERS = {"GL2C_MC_WRREQ_STALL", "GRBM_GUI_ACTIVE"}
PASS_B_SUM_COUNTERS = {
    "SQ_INST_CYCLES_VALU", "SQ_INST_CYCLES_VMEM", "SQ_INSTS_LDS",
    "SQ_INSTS_TEX_LOAD", "SQ_INSTS_TEX_STORE", "SQC_LDS_BANK_CONFLICT",
    "SQC_LDS_IDX_ACTIVE", "SQ_WAVES",
}
PASS_B_INSTANCE_SUM_COUNTERS = {"TA_TA_BUSY", "GRBM_GUI_ACTIVE"}
SUM_COUNTERS = PASS_A_SUM_COUNTERS | PASS_B_SUM_COUNTERS
INSTANCE_SUM_COUNTERS = PASS_A_INSTANCE_SUM_COUNTERS | PASS_B_INSTANCE_SUM_COUNTERS
PROFILER = {
    "name": "rocprofv3", "version": "1.3.5",
    "git_revision": "6b0e43f341195e203754e08f850e437ff2fc09f9",
    "rocm_version": "10.0.0",
}
PLAN_DIRECTORY = Path(
    "/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/rocprof/"
    "r9700-production-q4-p2048-pmc-two-pass-plan-20260905")
BENCHMARK_COMMANDS = {
    label: (
        "/ssdpool2nvme/local_llm/ninfer-amd-r9700/build-r9700-dense-selection-g16/"
        "bench/ninfer_bench --weights /ssdpool2nvme/local_llm/ninfer-amd-r9700/out/"
        "qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer --corpus "
        "/ssdpool2nvme/local_llm/ninfer-amd-r9700/bench/fixtures/bench_corpus.ids "
        "--device 0 --concurrency 1 -p 2048 --prefill-chunk 4096 --draft-tokens 0 "
        f"--output json --output-file {PLAN_DIRECTORY}/benchmark-{report}.json "
        "-r 1 --warmup 1 --profile-measured"
    )
    for label, report in (("cache_wait", "cache-wait"), ("issue_lds", "issue-lds"))
}
WORKLOAD = {
    "prompt_tokens": 2048, "generated_tokens": 0, "concurrency": 1,
    "prefill_chunk": 4096, "kv_value_group": 16, "xattention_profile": "dense",
    "spec": "none", "repetitions": 1, "warmup": 1,
}
PLAN_SHAPES = [
    {"role": "post_mixer/down", "m_tokens": 2048, "n_rows": 5120,
     "k_columns": 17408, "calls": 64, "grid_size_workitems": 655360,
     "expected_waves": 20480,
     "roctx_prefix": "ninfer.post-mixer.prefill.",
     "native_iu4_wmma_per_call": 44564480,
     "represented_unique_bytes_per_call": 105086980},
    {"role": "attention_and_gdn/output", "m_tokens": 2048, "n_rows": 5120,
     "k_columns": 6144, "calls": 64, "grid_size_workitems": 655360,
     "expected_waves": 20480,
     "roctx_prefixes": ["ninfer.attention.prefill.", "ninfer.gdn.prefill."],
     "native_iu4_wmma_per_call": 15728640,
     "represented_unique_bytes_per_call": 50659332},
    {"role": "gdn/value_z", "m_tokens": 2048, "n_rows": 12288,
     "k_columns": 5120, "calls": 48, "grid_size_workitems": 1572864,
     "expected_waves": 49152,
     "roctx_prefix": "ninfer.gdn.prefill.",
     "native_iu4_wmma_per_call": 31457280,
     "represented_unique_bytes_per_call": 94568452},
]
SHAPE_CONTRACTS = {
    "post_mixer_down_n5120_k17408": {"m_tokens": 2048, "n_rows": 5120,
        "k_columns": 17408, "grid_size_workitems": 655360, "expected_waves": 20480,
        "native_iu4_wmma_per_call": 44564480,
        "represented_unique_bytes_per_call": 105086980},
    "attention_output_n5120_k6144": {"m_tokens": 2048, "n_rows": 5120,
        "k_columns": 6144, "grid_size_workitems": 655360, "expected_waves": 20480,
        "native_iu4_wmma_per_call": 15728640,
        "represented_unique_bytes_per_call": 50659332},
    "gdn_output_n5120_k6144": {"m_tokens": 2048, "n_rows": 5120,
        "k_columns": 6144, "grid_size_workitems": 655360, "expected_waves": 20480,
        "native_iu4_wmma_per_call": 15728640,
        "represented_unique_bytes_per_call": 50659332},
    "gdn_value_z_n12288_k5120": {"m_tokens": 2048, "n_rows": 12288,
        "k_columns": 5120, "grid_size_workitems": 1572864, "expected_waves": 49152,
        "native_iu4_wmma_per_call": 31457280,
        "represented_unique_bytes_per_call": 94568452},
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _decimal(value: object, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _one_table(names: Iterable[str], prefix: str) -> str:
    matches = [name for name in names if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"database must contain exactly one {prefix} table")
    return matches[0]


def _read_agent(path: Path) -> tuple[int, int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source)
                if row.get("Agent_Type") == "GPU" and row.get("Name") == "gfx1201"]
    if len(rows) != 1 or rows[0].get("Product_Name") != "AMD Radeon AI PRO R9700":
        raise ValueError("agent CSV lacks exactly one R9700/gfx1201 GPU")
    cu_count = int(rows[0]["Cu_Count"])
    simd_count = int(rows[0]["Simd_Count"])
    if (cu_count, simd_count) != (64, 128):
        raise ValueError("unexpected R9700 CU/SIMD geometry")
    return cu_count, simd_count, {
        "device": rows[0]["Product_Name"], "architecture": rows[0]["Name"],
        "cu_count": cu_count, "simd_count": simd_count,
        "wavefront_size": int(rows[0]["Wave_Front_Size"]),
    }


def _read_csv(path: Path, counters: set[str]) -> tuple[
    dict[int, dict[str, Any]], dict[tuple[int, str], list[Decimal]]
]:
    required = {"Dispatch_Id", "Kernel_Name", "Grid_Size", "Workgroup_Size",
                "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Counter_Name",
                "Counter_Value", "Start_Timestamp", "End_Timestamp"}
    dispatches: dict[int, dict[str, Any]] = {}
    values: dict[tuple[int, str], list[Decimal]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("counter CSV lacks required native rocprof columns")
        for line, row in enumerate(reader, 2):
            try:
                dispatch = int(row["Dispatch_Id"])
                metadata = {
                    "symbol": row["Kernel_Name"], "grid_size": int(row["Grid_Size"]),
                    "workgroup_size": int(row["Workgroup_Size"]),
                    "lds_bytes": int(row["LDS_Block_Size"]),
                    "scratch_bytes": int(row["Scratch_Size"]),
                    "vgpr": int(row["VGPR_Count"]),
                    "start": int(row["Start_Timestamp"]), "end": int(row["End_Timestamp"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid dispatch metadata at CSV line {line}") from error
            if KERNEL not in metadata["symbol"] or metadata["workgroup_size"] != 512:
                raise ValueError(f"CSV dispatch {dispatch} is not the selected production kernel")
            if metadata["end"] < metadata["start"] or metadata["scratch_bytes"] != 0:
                raise ValueError(f"CSV dispatch {dispatch} has invalid interval or scratch")
            if dispatches.setdefault(dispatch, metadata) != metadata:
                raise ValueError(f"CSV dispatch {dispatch} metadata is inconsistent")
            counter = row["Counter_Name"]
            if counter not in counters:
                raise ValueError(f"unexpected counter in CSV: {counter}")
            key = (dispatch, counter)
            # Native rocprof CSV has already reduced the hardware-instance dimension into one
            # Counter_Value row. A second row for the same dispatch/counter is therefore not an
            # instance to sum: it is a concatenated/replayed pass or malformed capture whose
            # dispatch identity is ambiguous.
            if key in values:
                raise ValueError(
                    f"duplicate dispatch/counter row (possible profiler replay): "
                    f"{dispatch}, {counter}")
            values[key].append(
                _decimal(row["Counter_Value"], f"CSV line {line} counter value"))
    if not dispatches:
        raise ValueError("counter CSV has no dispatches")
    for dispatch in dispatches:
        observed = {counter for item, counter in values if item == dispatch}
        if observed != counters:
            raise ValueError(f"dispatch {dispatch} lacks the exact counter inventory")
    return dispatches, values


def _read_regions(path: Path) -> tuple[
    dict[int, tuple[str, str | None]], dict[str, Any], dict[str, Any]
]:
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
            f'''select name, product_name, extdata from "{agents}"
                where type='GPU' and name='gfx1201' '''))
        if len(gpu) != 1 or gpu[0]["product_name"] != "AMD Radeon AI PRO R9700":
            raise ValueError("database lacks exactly one R9700/gfx1201 agent")
        extdata = json.loads(gpu[0]["extdata"])
        result: dict[int, tuple[str, str | None]] = {}
        query = f'''select k.dispatch_id, s.display_name symbol, r.string region
                    from "{dispatch}" k join "{symbols}" s on s.id=k.kernel_id
                    left join "{strings}" r on r.id=k.region_name_id'''
        for row in connection.execute(query):
            dispatch_id = int(row["dispatch_id"])
            value = (str(row["symbol"]), None if row["region"] is None else str(row["region"]))
            if dispatch_id in result and result[dispatch_id] != value:
                raise ValueError(f"database dispatch {dispatch_id} metadata is inconsistent")
            result[dispatch_id] = value
        process_rows = list(connection.execute(
            f'''select command, environment from "{processes}"'''))
        if len(process_rows) != 1 or not process_rows[0]["command"]:
            raise ValueError("database lacks exactly one process command")
        environment = json.loads(process_rows[0]["environment"])
        registered_library = environment.get("ROCPROFILER_REGISTER_LIBRARY")
        if not isinstance(registered_library, str) or not registered_library.endswith(
                f"librocprofiler-sdk.so.{PROFILER['version']}"):
            raise ValueError("database process does not bind the expected profiler release")
        return result, {
            "device": gpu[0]["product_name"], "architecture": gpu[0]["name"],
            "cu_count": int(extdata["cu_count"]), "simd_count": int(extdata["simd_count"]),
            "wavefront_size": int(extdata["wave_front_size"]),
        }, {"command": str(process_rows[0]["command"]),
            "registered_library": registered_library}
    finally:
        connection.close()


def _classify(grid_size: int, region: str | None) -> str:
    if region is None:
        raise ValueError("selected dispatch lacks an exact ROCTX stage")
    if grid_size == 1572864 and region.startswith("ninfer.gdn.prefill."):
        return "gdn_value_z_n12288_k5120"
    if grid_size == 655360 and region.startswith("ninfer.post-mixer.prefill."):
        return "post_mixer_down_n5120_k17408"
    if grid_size == 655360 and region.startswith("ninfer.attention.prefill."):
        return "attention_output_n5120_k6144"
    if grid_size == 655360 and region.startswith("ninfer.gdn.prefill."):
        return "gdn_output_n5120_k6144"
    raise ValueError(f"unrecognized selected dispatch geometry/stage: {grid_size}, {region}")


def _ratio(numerator: Decimal, denominator: Decimal, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{label} has a nonpositive denominator")
    result = float(Decimal(100) * numerator / denominator)
    if not math.isfinite(result) or result > 100:
        raise ValueError(f"{label} is nonfinite or exceeds 100 percent")
    return result


def _quotient(numerator: Decimal, denominator: Decimal, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{label} has a nonpositive denominator")
    result = float(numerator / denominator)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite")
    return result


def _reduce(ids: list[int], values: dict[tuple[int, str], list[Decimal]],
            sum_counters: set[str], instance_sum_counters: set[str]) -> tuple[dict, dict]:
    summed = {name: sum((sum(values[(dispatch, name)]) for dispatch in ids), Decimal(0))
              for name in sum_counters}
    instance_summed = {
        name: sum((values[(dispatch, name)][0] for dispatch in ids), Decimal(0))
        for name in instance_sum_counters
    }
    return summed, instance_summed


def _summarize(pass_a_ids: list[int], pass_a_values: dict[tuple[int, str], list[Decimal]],
               pass_b_ids: list[int], pass_b_values: dict[tuple[int, str], list[Decimal]],
               cu_count: int, simd_count: int) -> dict[str, Any]:
    a_sum, a_max = _reduce(
        pass_a_ids, pass_a_values, PASS_A_SUM_COUNTERS, PASS_A_INSTANCE_SUM_COUNTERS)
    b_sum, b_max = _reduce(
        pass_b_ids, pass_b_values, PASS_B_SUM_COUNTERS, PASS_B_INSTANCE_SUM_COUNTERS)
    for name in ("SQ_WAVES", "SQ_WAVE_CYCLES", "GL2C_EA_RDREQ", "TCP_REQ"):
        if a_sum[name] <= 0:
            raise ValueError(f"pass A shape has nonpositive activity control: {name}")
    if b_sum["SQ_WAVES"] <= 0 or b_max["GRBM_GUI_ACTIVE"] <= 0:
        raise ValueError("pass B shape has nonpositive activity control")
    if a_sum["GL2C_HIT"] + a_sum["GL2C_MISS"] <= 0 or a_max["GRBM_GUI_ACTIVE"] <= 0:
        raise ValueError("shape has nonpositive GL2 or GRBM activity control")
    if a_sum["TCP_REQ_MISS"] > a_sum["TCP_REQ"]:
        raise ValueError("TCP misses exceed requests")
    a_active = a_max["GRBM_GUI_ACTIVE"]
    b_active = b_max["GRBM_GUI_ACTIVE"]
    metrics = {
        "wave_dependency_wait_percent": _ratio(a_sum["SQ_WAIT_ANY"], a_sum["SQ_WAVE_CYCLES"], "dependency wait"),
        "wave_issue_wait_percent": _ratio(a_sum["SQ_WAIT_INST_ANY"], a_sum["SQ_WAVE_CYCLES"], "issue wait"),
        "occupancy_percent": _ratio(
            a_sum["SQ_WAVE_CYCLES"], a_active * Decimal(cu_count) * Decimal(32),
            "occupancy"),
        "gl2_hit_percent": _ratio(a_sum["GL2C_HIT"], a_sum["GL2C_HIT"] + a_sum["GL2C_MISS"], "GL2 hit"),
        "l0_vector_cache_hit_percent": _ratio(
            a_sum["TCP_REQ"] - a_sum["TCP_REQ_MISS"], a_sum["TCP_REQ"],
            "L0 vector cache hit"),
    }
    lds_active = b_sum["SQC_LDS_IDX_ACTIVE"]
    metrics["lds_bank_conflict_percent"] = (
        None if lds_active == 0 else _ratio(b_sum["SQC_LDS_BANK_CONFLICT"], lds_active, "LDS conflict"))
    metrics["lds_utilization_percent"] = _ratio(
        lds_active, b_active * Decimal(32), "LDS utilization")
    normalized_activity = {
        "wave_life_cycles_per_wave": _quotient(
            a_sum["SQ_WAVE_CYCLES"], a_sum["SQ_WAVES"], "wave life"),
        "valu_cycles_per_wave": _quotient(
            b_sum["SQ_INST_CYCLES_VALU"], b_sum["SQ_WAVES"], "VALU cycles/wave"),
        "vmem_cycles_per_wave": _quotient(
            b_sum["SQ_INST_CYCLES_VMEM"], b_sum["SQ_WAVES"], "VMEM cycles/wave"),
        "lds_instructions_per_wave": _quotient(
            b_sum["SQ_INSTS_LDS"], b_sum["SQ_WAVES"], "LDS instructions/wave"),
        "tex_load_instructions_per_wave": _quotient(
            b_sum["SQ_INSTS_TEX_LOAD"], b_sum["SQ_WAVES"], "TEX loads/wave"),
        "tex_store_instructions_per_wave": _quotient(
            b_sum["SQ_INSTS_TEX_STORE"], b_sum["SQ_WAVES"], "TEX stores/wave"),
        "lds_index_active_per_wave": _quotient(
            lds_active, b_sum["SQ_WAVES"], "LDS index activity/wave"),
        "ta_busy_instance_sum_per_grbm_active_cycle": _quotient(
            b_max["TA_TA_BUSY"], b_active, "TA instance-sum activity/GRBM active"),
        "gl2_write_stall_instance_sum_per_grbm_active_cycle": _quotient(
            a_max["GL2C_MC_WRREQ_STALL"], a_active,
            "GL2 write-stall instance-sum activity/GRBM active"),
    }
    return {
        "dispatch_count_by_pass": {"cache_wait": len(pass_a_ids), "issue_lds": len(pass_b_ids)},
        "metrics_percent": metrics,
        "normalized_activity": normalized_activity,
        "activity_by_pass": {
            "cache_wait": {name.lower(): str(a_sum[name]) for name in sorted(a_sum)},
            "issue_lds": {name.lower(): str(b_sum[name]) for name in sorted(b_sum)},
        },
        "instance_sum_activity_by_pass": {
            "cache_wait": {name.lower(): str(a_max[name]) for name in sorted(a_max)},
            "issue_lds": {name.lower(): str(b_max[name]) for name in sorted(b_max)},
        },
        "physical_read_bandwidth": None,
        "wmma_utilization": None,
    }


def analyze(plan_path: Path, counter_csv_a: Path, database_a: Path, agent_csv_a: Path,
            power_before_a: Path, power_after_a: Path, counter_csv_b: Path,
            database_b: Path, agent_csv_b: Path, power_before_b: Path,
            power_after_b: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "command_only_not_executed":
        raise ValueError("not the exact production-Q4 P2048 PMC plan")
    if (plan.get("workload") != WORKLOAD or plan.get("expected_shapes") != PLAN_SHAPES
            or plan.get("expected_total_dispatches") != 176
            or plan.get("kernel_include_regex") != KERNEL
            or plan.get("profiler") != PROFILER
            or plan.get("benchmark_commands") != BENCHMARK_COMMANDS):
        raise ValueError("plan workload, kernel, or exact shape inventory differs")
    passes = plan.get("passes")
    expected_passes = {
        "cache_wait": PASS_A_SUM_COUNTERS | PASS_A_INSTANCE_SUM_COUNTERS,
        "issue_lds": PASS_B_SUM_COUNTERS | PASS_B_INSTANCE_SUM_COUNTERS,
    }
    if not isinstance(passes, dict) or {
        name: set(value.get("counters", [])) if isinstance(value, dict) else set()
        for name, value in passes.items()
    } != expected_passes:
        raise ValueError("plan has an unexpected two-pass counter inventory")
    for before, after in ((power_before_a, power_after_a), (power_before_b, power_after_b)):
        if before.read_text(encoding="utf-8").strip() != "profile_standard":
            raise ValueError("PMC capture did not begin under profile_standard")
        if after.read_text(encoding="utf-8").strip() != "auto":
            raise ValueError("power profile was not restored to auto")
    cu_count, simd_count, agent_a = _read_agent(agent_csv_a)
    cu_count_b, simd_count_b, agent_b = _read_agent(agent_csv_b)
    if (cu_count_b, simd_count_b, agent_b) != (cu_count, simd_count, agent_a):
        raise ValueError("PMC passes have different R9700 agent identities")
    agent = agent_a

    process_by_pass: dict[str, dict[str, Any]] = {}

    def read_pass(label: str, counter_csv: Path, database: Path, counters: set[str]):
        dispatches, values = _read_csv(counter_csv, counters)
        regions, database_agent, process = _read_regions(database)
        if database_agent != agent or not set(dispatches).issubset(regions):
            raise ValueError("CSV/database dispatch or agent identities differ")
        if process["command"] != BENCHMARK_COMMANDS[label]:
            raise ValueError(f"{label} database process command differs from the exact workload")
        process_by_pass[label] = process
        classified: dict[str, list[int]] = defaultdict(list)
        for dispatch, metadata in dispatches.items():
            symbol, region = regions[dispatch]
            if KERNEL not in symbol:
                raise ValueError(f"CSV/database symbol mismatch for dispatch {dispatch}")
            shape = _classify(metadata["grid_size"], region)
            expected_waves = Decimal(SHAPE_CONTRACTS[shape]["expected_waves"])
            if values[(dispatch, "SQ_WAVES")] != [expected_waves]:
                raise ValueError(
                    f"dispatch {dispatch} SQ_WAVES differs from exact grid/wave geometry")
            classified[shape].append(dispatch)
        return dispatches, values, classified

    dispatches_a, values_a, classified_a = read_pass(
        "cache_wait", counter_csv_a, database_a, expected_passes["cache_wait"])
    dispatches_b, values_b, classified_b = read_pass(
        "issue_lds", counter_csv_b, database_b, expected_passes["issue_lds"])
    expected = {
        "post_mixer_down_n5120_k17408": 64,
        "attention_output_n5120_k6144": 16,
        "gdn_output_n5120_k6144": 48,
        "gdn_value_z_n12288_k5120": 48,
    }
    for label, classified in (("cache_wait", classified_a), ("issue_lds", classified_b)):
        if Counter({name: len(ids) for name, ids in classified.items()}) != Counter(expected):
            raise ValueError(f"{label} selected shape inventory differs: {dict((k,len(v)) for k,v in classified.items())}")
    summaries = {
        name: _summarize(classified_a[name], values_a, classified_b[name], values_b,
                         cu_count, simd_count)
        for name in sorted(classified_a)
    }
    for name in classified_a:
        resources_by_pass = {}
        for label, dispatches, ids in (
            ("cache_wait", dispatches_a, classified_a[name]),
            ("issue_lds", dispatches_b, classified_b[name]),
        ):
            resources = {
                "workgroup_size": sorted({dispatches[item]["workgroup_size"] for item in ids}),
                "vgpr": sorted({dispatches[item]["vgpr"] for item in ids}),
                "lds_bytes": sorted({dispatches[item]["lds_bytes"] for item in ids}),
                "scratch_bytes": sorted({dispatches[item]["scratch_bytes"] for item in ids}),
                "waves_per_workgroup": 16,
            }
            if (resources["workgroup_size"] != [512] or resources["vgpr"] != [88]
                    or resources["lds_bytes"] != [17408] or resources["scratch_bytes"] != [0]):
                raise ValueError(f"{label}/{name} lacks selected production resources")
            resources_by_pass[label] = resources
        summaries[name]["resources_by_pass"] = resources_by_pass
        summaries[name]["shape_contract"] = {
            **SHAPE_CONTRACTS[name], "calls": len(classified_a[name]),
            "native_iu4_wmma_total":
                SHAPE_CONTRACTS[name]["native_iu4_wmma_per_call"] * len(classified_a[name]),
            "represented_unique_bytes_total":
                SHAPE_CONTRACTS[name]["represented_unique_bytes_per_call"] * len(classified_a[name]),
        }
    metric_names = tuple(next(iter(summaries.values()))["metrics_percent"])
    extrema = {}
    for metric in metric_names:
        available = [(name, row["metrics_percent"][metric]) for name, row in summaries.items()
                     if row["metrics_percent"][metric] is not None]
        extrema[metric] = None if not available else {
            "highest_shape": max(available, key=lambda item: item[1])[0],
            "highest_value": max(value for _name, value in available),
            "lowest_shape": min(available, key=lambda item: item[1])[0],
            "lowest_value": min(value for _name, value in available),
        }
    normalized_extrema = {}
    for metric in next(iter(summaries.values()))["normalized_activity"]:
        available = [(name, row["normalized_activity"][metric])
                     for name, row in summaries.items()]
        normalized_extrema[metric] = {
            "highest_shape": max(available, key=lambda item: item[1])[0],
            "highest_value": max(value for _name, value in available),
            "lowest_shape": min(available, key=lambda item: item[1])[0],
            "lowest_value": min(value for _name, value in available),
        }
    return {
        "schema": REPORT_SCHEMA,
        "status": "valid_attribution_only",
        "profile_timing_admissible": False,
        "power_profile": {"capture": "profile_standard", "restored": "auto"},
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)},
        "profiler": PROFILER,
        "passes": {
            "cache_wait": {
                "counter_csv": {"path": str(counter_csv_a.resolve()), "sha256": _sha256(counter_csv_a)},
                "database": {"path": str(database_a.resolve()), "sha256": _sha256(database_a)},
                "agent_csv": {"path": str(agent_csv_a.resolve()), "sha256": _sha256(agent_csv_a)},
                "process": process_by_pass["cache_wait"],
            },
            "issue_lds": {
                "counter_csv": {"path": str(counter_csv_b.resolve()), "sha256": _sha256(counter_csv_b)},
                "database": {"path": str(database_b.resolve()), "sha256": _sha256(database_b)},
                "agent_csv": {"path": str(agent_csv_b.resolve()), "sha256": _sha256(agent_csv_b)},
                "process": process_by_pass["issue_lds"],
            },
        },
        "agent": agent,
        "dispatch_count_by_pass": {"cache_wait": len(dispatches_a), "issue_lds": len(dispatches_b)},
        "native_iu4_wmma_total": 5368709120,
        "represented_unique_bytes_total": 14507049664,
        "shapes": summaries,
        "cross_shape_extrema": extrema,
        "cross_shape_normalized_activity_extrema": normalized_extrema,
        "decision_boundary": {
            "memory_cache_pressure": "Compare same-pass TA instance-sum/GRBM relative activity and per-wave VMEM/TEX activity, GL2/TCP hit ratios, relative GL2 read requests, and broad waits across exact shapes. TA and GL2 write-stall instance maxima are unrecoverable from rocprof's already instance-summed CSV row, so neither is reported as utilization.",
            "scheduler_dependency_or_reconstruction_pressure": "Compare dependency/issue waits, per-wave VALU cycle demand, wave life, VGPR metadata, LDS activity/conflicts, and static instruction inventories. Reconstruction cannot be isolated without a matched challenger; VALU cycle demand is not utilization because overlapping V_OP32 classes can double-count.",
            "physical_read_bandwidth": "unavailable: no valid gfx1201 request-size counters; GL2C_EA_RDREQ is relative request activity only",
            "wmma_utilization": "unavailable: gfx1201 has no separate WMMA-cycle counter; use exact static IU4 counts with auto timing only",
            "stall_freedom": "not provable: available wait counters overlap and do not decompose every cause"
        },
        "cross_pass_contract": "No dispatch-ID join and no ratio whose numerator and denominator come from different passes. Only independently validated exact-shape summaries are compared.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    for suffix in ("a", "b"):
        parser.add_argument(f"--counter-csv-{suffix}", required=True, type=Path)
        parser.add_argument(f"--database-{suffix}", required=True, type=Path)
        parser.add_argument(f"--agent-csv-{suffix}", required=True, type=Path)
        parser.add_argument(f"--power-before-{suffix}", required=True, type=Path)
        parser.add_argument(f"--power-after-{suffix}", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            args.plan,
            args.counter_csv_a, args.database_a, args.agent_csv_a,
            args.power_before_a, args.power_after_a,
            args.counter_csv_b, args.database_b, args.agent_csv_b,
            args.power_before_b, args.power_after_b,
        )
        with args.out.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    except (OSError, sqlite3.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
