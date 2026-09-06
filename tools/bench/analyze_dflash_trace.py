#!/usr/bin/env python3
"""Validate and attribute one exact C1 eager DFlash selected-region trace."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Sequence


MEASURED = "ninfer_bench_measured"
FULL_LAYERS = frozenset(range(3, 64, 4))
GDN_LAYERS = frozenset(range(64)) - FULL_LAYERS
TARGET_FAMILIES = {
    "ninfer.attention.verify.layer.full": FULL_LAYERS,
    "ninfer.attention.verify.attention": FULL_LAYERS,
    "ninfer.gdn.verify.layer.gdn": GDN_LAYERS,
    "ninfer.gdn.verify.gdn": GDN_LAYERS,
    "ninfer.post-mixer.verify.post_mixer": frozenset(range(64)),
}
PREFILL_PREFIXES = (
    "ninfer.prefill.prefill.chunk",
    "ninfer.attention.prefill.",
    "ninfer.gdn.prefill.",
    "ninfer.post-mixer.prefill.",
)
DFLASH_SERVICE_FAMILIES = {
    "ninfer.dflash.decode.dflash_round",
    "ninfer.dflash.decode.dflash.submit",
    "ninfer.dflash.decode.dflash.wait",
}
EXPECTED_STATUS = "command-only-not-executed"
EXPECTED_COMMON_WORKLOAD = {
    "concurrency": 1,
    "prompt_tokens": 128,
    "generated_tokens": 64,
    "prefill_chunk": 4096,
    "max_context": 256,
    "spec": "dflash",
    "proposal_head": "optimized",
    "device_graph": False,
    "repetitions": 1,
    "warmup": 1,
}
EXPECTED_POWER_PROFILE = {
    "path": "/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level",
    "required_before_after": "auto",
}
EXPECTED_CLAIMS = {
    "admissible": (
        "evaluation-only eager whole-DFlash dispatch attribution separating exact 64-layer "
        "target verification from combined proposal/head/round service for K4/W5 and K5/W6"
    ),
    "not_admissible": (
        "production recipe selection, production routing, Device-Graph throughput, profiled "
        "throughput authority, acceptance/quality admission, physical bandwidth, cache hit "
        "rate, or practical-ceiling closure"
    ),
}
EXPECTED_LIMITATIONS = [
    "The only exact local artifact/build pair is the retained legacy row-split canonical-Q4 "
    "evaluator; current N16/K16 DFlash companions are absent.",
    "The all-Q4 base is an evaluation control and has not won terminal production selection.",
    "Eager execution is required so per-layer ROCTX ranges remain attributable; its profiled "
    "timing is not a Device-Graph performance result.",
    "Empty or outer-round-associated dispatches are conservatively combined as "
    "proposal/head/round service or unattributed; symbols refine operators but cannot invent "
    "a missing semantic marker.",
    "The trace contains no performance counters and makes no memory-bandwidth, cache-hit, or "
    "stall claim.",
]


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _message(extdata: str) -> str:
    value = json.loads(extdata)
    message = value.get("message") if isinstance(value, dict) else None
    if not isinstance(message, str):
        raise ValueError("range marker lacks a string message")
    return message


def _split_payload(message: str) -> tuple[str, int]:
    parts = message.rsplit(" payload=", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"marker lacks exact integer payload: {message}")
    return parts[0], int(parts[1])


def _exact_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _matches_exact_fields(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        candidate = actual.get(key)
        if isinstance(value, bool):
            if candidate is not value:
                return False
        elif isinstance(value, int):
            if not _exact_int(candidate) or candidate != value:
                return False
        elif candidate != value:
            return False
    return True


def _validate_verify_ranges(
    ranges: list[tuple[int, int, str]], measured: tuple[int, int]
) -> tuple[int, dict[str, list[tuple[int, int]]]]:
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for begin, end, message in ranges:
        if begin < measured[0] or end > measured[1] or end <= begin:
            raise ValueError("marker interval lies outside measured range or is non-positive")
        if message == MEASURED or message.startswith(PREFILL_PREFIXES):
            continue
        family, payload = _split_payload(message)
        if family in DFLASH_SERVICE_FAMILIES:
            continue
        if family not in TARGET_FAMILIES or payload not in TARGET_FAMILIES[family]:
            raise ValueError(f"unexpected target-verification marker: {message}")
        grouped[family].append((begin, end, payload))
    counts = {family: len(grouped[family]) for family in TARGET_FAMILIES}
    if not counts or any(count == 0 for count in counts.values()):
        raise ValueError("trace has no complete target verification")
    rounds = counts["ninfer.post-mixer.verify.post_mixer"] // 64
    expected_counts = {
        family: rounds * len(payloads) for family, payloads in TARGET_FAMILIES.items()
    }
    if rounds <= 0 or counts != expected_counts:
        raise ValueError("target-verification marker counts are not complete 64-layer rounds")
    for family, payloads in TARGET_FAMILIES.items():
        actual = grouped[family]
        expected_sequence = [payload for _ in range(rounds) for payload in sorted(payloads)]
        if [row[2] for row in actual] != expected_sequence:
            raise ValueError(f"{family}: layer payload order differs")
    by_family: dict[str, list[tuple[int, int]]] = {
        family: [(begin, end) for begin, end, _ in rows]
        for family, rows in grouped.items()
    }
    by_message: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for family, rows in grouped.items():
        for begin, end, payload in rows:
            by_message[(family, payload)].append((begin, end))
    previous_outer_end: int | None = None
    for round_index in range(rounds):
        for layer in range(64):
            outer_family = ("ninfer.attention.verify.layer.full" if layer in FULL_LAYERS
                            else "ninfer.gdn.verify.layer.gdn")
            leaf_family = ("ninfer.attention.verify.attention" if layer in FULL_LAYERS
                           else "ninfer.gdn.verify.gdn")
            outer = by_message[(outer_family, layer)][round_index]
            inner = by_message[(leaf_family, layer)][round_index]
            post = by_message[("ninfer.post-mixer.verify.post_mixer", layer)][round_index]
            if not (outer[0] <= inner[0] < inner[1] <= post[0] < post[1] <= outer[1]):
                raise ValueError("target leaf/post-mixer is not ordered inside matching layer")
            if previous_outer_end is not None and outer[0] < previous_outer_end:
                raise ValueError("target layer outer ranges overlap or differ in order")
            previous_outer_end = outer[1]
    return rounds, by_family


def _symbol_family(name: str) -> str:
    rules = (
        ("a8q4g64_linear", "q4_linear"),
        ("dflash2_path_select", "dflash_selector"),
        ("dflash2_tree_select", "dflash_selector"),
        ("grouped_dynamic_conv", "dflash_convolution"),
        ("bidirectional", "dflash_attention"),
        ("swa", "dflash_swa"),
        ("rmsnorm", "normalization"),
        ("sampling", "sampling"),
        ("embedding", "embedding"),
    )
    return next((family for needle, family in rules if needle in name), "other")


def _validate_report(report: dict[str, Any], cell: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if report.get("artifact_type") != "ninfer_bench_report" or report.get("schema_version") != 20:
        raise ValueError("benchmark report is not the exact retained-build schema v20")
    config = report.get("config")
    expected_config = {
        "concurrency": 1, "spec": "dflash", "draft_tokens": cell["draft_tokens"],
        "dflash_verify_width_requested": cell["verify_width"],
        "dflash_verify_width": cell["verify_width"], "proposal_head": "optimized",
        "use_device_graph": False, "speculative_execution": True,
        "decode_path": "dflash_eager", "prefill_chunk": 4096,
        "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 32,
        "repetitions": 1, "warmup": 1, "max_context": 256,
    }
    if not _matches_exact_fields(config, expected_config):
        raise ValueError("benchmark report configuration differs")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1 or not isinstance(tests[0], dict):
        raise ValueError("benchmark report must contain exactly one test")
    test = tests[0]
    if (test.get("kind") != "whole" or not _exact_int(test.get("n_prompt")) or
            test["n_prompt"] != 128 or not _exact_int(test.get("n_gen")) or
            test["n_gen"] != 64 or not _exact_int(test.get("requested_output_tokens")) or
            test["requested_output_tokens"] != 65):
        raise ValueError("benchmark test is not exact P128/G64")
    speculative = test.get("speculative")
    if (not isinstance(speculative, dict) or speculative.get("enabled") is not True or
            speculative.get("draft_window") != cell["draft_tokens"] or
            not _exact_int(speculative.get("rounds"), minimum=1)):
        raise ValueError("benchmark report lacks DFlash rounds")
    for key in ("drafted_tokens", "accepted_tokens", "fallback_steps"):
        if not _exact_int(speculative.get(key)):
            raise ValueError(f"benchmark report lacks exact speculative {key}")
    for key in ("decode_seconds_mean", "decode_output_tok_s_mean", "decode_engine_tok_s_mean"):
        if not _positive_finite(test.get(key)):
            raise ValueError(f"benchmark report lacks positive finite {key}")
    reps = test.get("reps")
    if not isinstance(reps, list) or len(reps) != 1 or not isinstance(reps[0], dict):
        raise ValueError("benchmark report must contain exactly one measured repetition")
    rep = reps[0]
    if (rep.get("generated_output_tokens") != 65 or rep.get("decode_output_tokens") != 64 or
            not _exact_int(rep.get("decode_engine_tokens"), minimum=1)):
        raise ValueError("measured repetition did not complete exact G64 output")
    rep_speculative = rep.get("speculative")
    if rep_speculative != speculative:
        raise ValueError("one-repetition speculative accounting differs from aggregate")
    expected_engine_tokens = (
        speculative["rounds"] + speculative["accepted_tokens"] + speculative["fallback_steps"]
    )
    if rep["decode_engine_tokens"] != expected_engine_tokens:
        raise ValueError("decode engine tokens differ from speculative accounting")
    timings = rep.get("timings")
    if not isinstance(timings, dict) or any(
        not _positive_finite(timings.get(key))
        for key in ("prepare_seconds", "prefill_seconds", "decode_seconds", "total_seconds")
    ):
        raise ValueError("measured repetition lacks positive finite timings")
    vision_seconds = timings.get("vision_seconds")
    if (not isinstance(vision_seconds, (int, float)) or isinstance(vision_seconds, bool) or
            not math.isfinite(vision_seconds) or vision_seconds < 0):
        raise ValueError("measured repetition has invalid vision timing")
    return test, speculative


def analyze(plan_path: Path, cell_name: str) -> dict[str, Any]:
    plan = _json(plan_path)
    if plan.get("artifact_type") != "ninfer_r9700_dflash_owner_trace_plan" or plan.get("schema_version") != 1:
        raise ValueError("plan identity differs")
    if (plan.get("status") != EXPECTED_STATUS or
            plan.get("common_workload") != EXPECTED_COMMON_WORKLOAD or
            plan.get("power_profile") != EXPECTED_POWER_PROFILE or
            plan.get("claims") != EXPECTED_CLAIMS or
            plan.get("limitations") != EXPECTED_LIMITATIONS):
        raise ValueError("plan workload, power, claims, or limitations differ")
    root = Path(plan["output_root"])
    if plan_path.resolve() != (root / "plan.json").resolve():
        raise ValueError("plan path is outside its exact output root")
    cells = plan.get("cells")
    if not isinstance(cells, dict) or set(cells) != {"k4w5", "k5w6"} or cell_name not in cells:
        raise ValueError("plan must contain exactly K4/W5 and K5/W6")
    cell = cells[cell_name]
    if (cell_name, cell.get("draft_tokens"), cell.get("verify_width")) not in (
        ("k4w5", 4, 5), ("k5w6", 5, 6)
    ):
        raise ValueError("cell K/W identity differs")
    for label in ("benchmark_executable", "artifact", "corpus", "profiler", "inventory", "analyzer"):
        identity = plan[label]
        path = Path(identity["path"])
        if not path.is_file() or path.stat().st_size != identity["file_size_bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"{label} bytes differ from plan")
    for label in ("cmake_cache", "compile_commands"):
        identity = plan["build_configuration"][label]
        path = Path(identity["path"])
        if not path.is_file() or path.stat().st_size != identity["file_size_bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"build {label} bytes differ from plan")
    receipt = plan["artifact"]["conversion_report"]
    receipt_path = Path(receipt["path"])
    if not receipt_path.is_file() or receipt_path.stat().st_size != receipt["file_size_bytes"] or _sha256(receipt_path) != receipt["sha256"]:
        raise ValueError("artifact conversion report bytes differ from plan")
    cell_root = root / cell_name
    report_path = cell_root / "benchmark-report.json"
    database = cell_root / "raw" / f"{cell_name}_results.db"
    before, after = cell_root / "power-before.txt", cell_root / "power-after.txt"
    if before.read_text(encoding="utf-8").strip() != "auto" or after.read_text(encoding="utf-8").strip() != "auto":
        raise ValueError("power endpoints are not both auto")
    report = _json(report_path)
    if report.get("command") != cell["benchmark_command_string"]:
        raise ValueError("benchmark report command differs")
    report_environment = report.get("environment")
    if (not isinstance(report_environment, dict) or
            set(report_environment) != {"gpu_name", "architecture_name", "hip_runtime_version",
                                        "hip_driver_version", "device_id"} or
            report_environment.get("gpu_name") != "AMD Radeon AI PRO R9700" or
            report_environment.get("architecture_name") != "gfx1201" or
            report_environment.get("device_id") != 0 or
            any(not isinstance(report_environment.get(key), str) or not report_environment[key]
                for key in ("hip_runtime_version", "hip_driver_version"))):
        raise ValueError("benchmark report device identity differs")
    artifact_report = report.get("artifact")
    load_report = report.get("load")
    if not isinstance(artifact_report, dict) or artifact_report.get("path") != plan["artifact"]["path"] or artifact_report.get("file_size_bytes") != plan["artifact"]["file_size_bytes"] or not isinstance(load_report, dict) or load_report.get("target") != "qwen3_8_27b_r9700" or load_report.get("weights_id") != plan["artifact"]["weights_id"]:
        raise ValueError("benchmark report artifact identity differs")
    test, speculative = _validate_report(report, cell)

    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type in ('table','view')")}
        required = {"rocpd_info_process", "rocpd_info_agent", "regions", "kernels", "memory_copies"}
        if not required <= tables:
            raise ValueError("rocprof database lacks required views")
        processes = list(connection.execute('select nid,pid,command,environment,extdata from rocpd_info_process'))
        if len(processes) != 1 or processes[0]["command"] != cell["benchmark_command_string"]:
            raise ValueError("trace process argv differs")
        process_nid, process_pid = processes[0]["nid"], processes[0]["pid"]
        environment = json.loads(processes[0]["environment"])
        process_extdata = json.loads(processes[0]["extdata"])
        expected_raw = str(cell_root / "raw")
        if not isinstance(process_extdata, dict) or any(
            process_extdata.get(key) != value for key, value in {
                "output_path": expected_raw,
                "output_file": cell_name,
                "raw_output_path": expected_raw,
                "raw_output_file": cell_name,
                "rocpd_output": True,
                "kernel_rename": False,
            }.items()
        ):
            raise ValueError("trace process output identity differs")
        for key, value in {
            "ROCPROF_OUTPUT_FORMAT": "rocpd", "ROCPROF_MARKER_API_TRACE": "1",
            "ROCPROF_KERNEL_TRACE": "1", "ROCPROF_MEMORY_COPY_TRACE": "1",
            "ROCPROF_SELECTED_REGIONS": "1",
        }.items():
            if environment.get(key) != value:
                raise ValueError("trace profiler environment differs")
        agents = list(connection.execute("select * from rocpd_info_agent where type='GPU' order by absolute_index"))
        if any((row["nid"], row["pid"]) != (process_nid, process_pid) for row in agents):
            raise ValueError("GPU inventory ownership differs")
        r9700 = [row for row in agents if row["name"] == "gfx1201" and row["product_name"] == "AMD Radeon AI PRO R9700"]
        if len(r9700) != 1 or r9700[0]["logical_index"] != 1 or r9700[0]["type_index"] != 0:
            raise ValueError("trace lacks exact R9700 agent")
        topology = json.loads(r9700[0]["extdata"])
        if (topology.get("cu_count"), topology.get("simd_count"), topology.get("wave_front_size")) != (64, 128, 32):
            raise ValueError("R9700 topology differs")
        kernel_rows = list(connection.execute('select nid,pid,agent_abs_index,agent_log_index,agent_type_index,agent_type,dispatch_id,start,"end",duration,name,coalesce(region,"") region from kernels order by start'))
        if not kernel_rows or len({row["dispatch_id"] for row in kernel_rows}) != len(kernel_rows):
            raise ValueError("kernel dispatch inventory is empty or duplicated")
        expected_owner = (process_nid, process_pid, r9700[0]["absolute_index"], 1, 0, "GPU")
        if any(tuple(row[k] for k in ("nid","pid","agent_abs_index","agent_log_index","agent_type_index","agent_type")) != expected_owner for row in kernel_rows):
            raise ValueError("selected kernel did not execute on exact R9700 agent")
        range_rows = list(connection.execute('select nid,pid,category,name,start,"end",duration,extdata from regions order by start'))
        controls: list[tuple[int, int, str, dict[str, Any]]] = []
        ranges = []
        for row in range_rows:
            if (row["nid"], row["pid"]) != (process_nid, process_pid):
                raise ValueError("marker ownership differs")
            begin, end = int(row["start"]), int(row["end"])
            if begin < 0 or end <= begin or int(row["duration"]) != end - begin:
                raise ValueError("marker interval is invalid")
            if row["category"] == "MARKER_CONTROL_API":
                control_extdata = json.loads(row["extdata"])
                if control_extdata != {}:
                    raise ValueError("profiler control marker extdata differs")
                controls.append((begin, end, row["name"], control_extdata)); continue
            if row["category"] != "MARKER_CORE_RANGE_API" or row["name"] != "roctxThreadRangeA":
                raise ValueError("unexpected marker record")
            ranges.append((begin, end, _message(row["extdata"])))
        if [row[2] for row in controls] != ["roctxProfilerResume", "roctxProfilerPause"]:
            raise ValueError("profiler control markers differ")
        measured = [(b,e) for b,e,m in ranges if m == MEASURED]
        if len(measured) != 1:
            raise ValueError("trace lacks one measured range")
        if not (controls[0][1] <= measured[0][0] < measured[0][1] <= controls[1][0]):
            raise ValueError("resume, measured, and pause boundaries differ")
        verify_rounds, _ = _validate_verify_ranges(ranges, measured[0])
        fallback_steps = speculative.get("fallback_steps")
        if not isinstance(fallback_steps, int) or fallback_steps < 0 or (
            verify_rounds != speculative["rounds"] + fallback_steps
        ):
            raise ValueError("64-layer target-verification count differs from DFlash rounds")
        known_messages = {message for _,_,message in ranges}
        ranges_by_message: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for begin, end, message in ranges:
            ranges_by_message[message].append((begin, end))
        prefill = [
            (begin, end) for begin, end, message in ranges
            if message == "ninfer.prefill.prefill.chunk payload=128"
        ]
        if len(prefill) != 1:
            raise ValueError("trace lacks exact P128 prefill chunk marker")
        expected_prefill = {
            "ninfer.prefill.prefill.chunk": [128],
            "ninfer.attention.prefill.layer.full": sorted(FULL_LAYERS),
            "ninfer.attention.prefill.attention": sorted(FULL_LAYERS),
            "ninfer.gdn.prefill.layer.gdn": sorted(GDN_LAYERS),
            "ninfer.gdn.prefill.gdn": sorted(GDN_LAYERS),
            "ninfer.post-mixer.prefill.post_mixer": list(range(64)),
        }
        actual_prefill: dict[str, list[int]] = defaultdict(list)
        for _, _, message in ranges:
            if message.startswith(PREFILL_PREFIXES):
                family, payload = _split_payload(message)
                actual_prefill[family].append(payload)
        if dict(actual_prefill) != expected_prefill:
            raise ValueError("prompt-prefill marker inventory differs from one exact 64-layer P128 pass")
        previous_outer_end = None
        for layer in range(64):
            outer_family = ("ninfer.attention.prefill.layer.full" if layer in FULL_LAYERS
                            else "ninfer.gdn.prefill.layer.gdn")
            leaf_family = ("ninfer.attention.prefill.attention" if layer in FULL_LAYERS
                           else "ninfer.gdn.prefill.gdn")
            outer = ranges_by_message[f"{outer_family} payload={layer}"]
            leaf = ranges_by_message[f"{leaf_family} payload={layer}"]
            post = ranges_by_message[f"ninfer.post-mixer.prefill.post_mixer payload={layer}"]
            if len(outer) != 1 or len(leaf) != 1 or len(post) != 1:
                raise ValueError("prompt-prefill marker occurrence count differs")
            outer_interval, leaf_interval, post_interval = outer[0], leaf[0], post[0]
            if not (prefill[0][0] <= outer_interval[0] <= leaf_interval[0] < leaf_interval[1]
                    <= post_interval[0] < post_interval[1] <= outer_interval[1]
                    <= prefill[0][1]):
                raise ValueError("prefill leaf/post-mixer is not ordered inside chunk and layer")
            if previous_outer_end is not None and outer_interval[0] < previous_outer_end:
                raise ValueError("prefill layer outer ranges overlap or differ in order")
            previous_outer_end = outer_interval[1]
        stage_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        symbols: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for row in kernel_rows:
            begin, end, duration = int(row["start"]), int(row["end"]), int(row["duration"])
            if duration <= 0 or duration != end - begin or begin < measured[0][0] or end > measured[0][1]:
                raise ValueError("kernel interval differs or escapes measured range")
            region = row["region"]
            if region and region not in known_messages:
                raise ValueError(f"kernel has unknown region association: {region}")
            if region and region != MEASURED and not any(
                marker_begin <= begin for marker_begin, _ in ranges_by_message[region]
            ):
                raise ValueError("kernel is associated with a future marker occurrence")
            if region.startswith(tuple(TARGET_FAMILIES)):
                stage = "target_64_layer_verification"
            elif region.startswith(PREFILL_PREFIXES):
                stage = "prompt_prefill"
            elif not region and prefill[0][0] <= begin and end <= prefill[0][1]:
                stage = "prompt_prefill"
            elif region in ("", MEASURED) or any(
                region.startswith(family + " payload=")
                for family in DFLASH_SERVICE_FAMILIES
            ):
                stage = "proposal_head_and_round_service_or_unattributed"
            else:
                raise ValueError(f"kernel region is not classified: {region}")
            stage_stats[stage][0] += 1; stage_stats[stage][1] += duration
            family = _symbol_family(row["name"])
            symbols[(stage, family)][0] += 1; symbols[(stage, family)][1] += duration
        copy_rows = list(connection.execute('select nid,pid,start,"end",duration,size,name,coalesce(region_name,"") region from memory_copies order by start'))
        for row in copy_rows:
            if ((row["nid"], row["pid"]) != (process_nid, process_pid) or
                    int(row["duration"]) <= 0 or
                    int(row["duration"]) != int(row["end"])-int(row["start"]) or
                    int(row["start"]) < measured[0][0] or
                    int(row["end"]) > measured[0][1] or int(row["size"]) < 0):
                raise ValueError("memory-copy ownership/interval differs")
            if row["region"] and row["region"] not in known_messages:
                raise ValueError("memory copy has unknown region association")
            if row["region"] and row["region"] != MEASURED and not any(
                marker_begin <= int(row["start"])
                for marker_begin, _ in ranges_by_message[row["region"]]
            ):
                raise ValueError("memory copy is associated with a future marker occurrence")
    finally:
        connection.close()
    return {
        "artifact_type": "ninfer_r9700_dflash_owner_trace_evidence",
        "schema_version": 1,
        "status": "valid-evaluation-attribution-only",
        "cell": cell_name,
        "workload": {"concurrency": 1, "prompt_tokens": 128, "generated_tokens": 64,
                     "draft_tokens": cell["draft_tokens"], "verify_width": cell["verify_width"],
                     "device_graph": False, "proposal_head": "optimized"},
        "benchmark": {"decode_output_tok_s_profiled": test["decode_output_tok_s_mean"],
                      "decode_engine_tok_s_profiled": test["decode_engine_tok_s_mean"],
                      "profiled_timing_admissible_for_selection": False},
        "target_verification_rounds": verify_rounds,
        "stages": [{"stage": stage, "calls": values[0], "summed_duration_ms": values[1]/1e6}
                   for stage, values in sorted(stage_stats.items())],
        "operator_families": [
            {"stage": stage, "family": family, "calls": values[0], "summed_duration_ms": values[1]/1e6}
            for (stage, family), values in sorted(symbols.items(), key=lambda item: -item[1][1])
        ],
        "limitations": EXPECTED_LIMITATIONS,
        "claims": EXPECTED_CLAIMS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cell", required=True, choices=("k4w5", "k5w6"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite evidence: {args.out}")
    try:
        result = analyze(args.plan, args.cell)
    except (OSError, sqlite3.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
