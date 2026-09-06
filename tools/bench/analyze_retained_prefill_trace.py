#!/usr/bin/env python3
"""Validate and attribute the one retained-production C1/P2048 prefill trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

# Support both `python3 -m tools.bench...` and the immutable package's direct absolute
# script invocation. This executes before importing any repository package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.analyze_whole_profile import analyze
from tools.bench.run_ninfer_bench_matrix import R9700_KV_PLANE_LAYOUTS
from tools.bench.validate_profile_trace import _parse_database


PLAN_TYPE = "ninfer_r9700_retained_prefill_trace_plan"
EVIDENCE_TYPE = "ninfer_r9700_retained_prefill_trace"
EXPECTED_GPU = "AMD Radeon AI PRO R9700"
EXPECTED_ARCH = "gfx1201"
EXPECTED_MARKERS = {
    "ninfer_bench_measured": 1,
    "ninfer.prefill.prefill.chunk": 1,
    "ninfer.attention.prefill.layer.full": 16,
    "ninfer.attention.prefill.attention": 16,
    "ninfer.gdn.prefill.layer.gdn": 48,
    "ninfer.gdn.prefill.gdn": 48,
    "ninfer.post-mixer.prefill.post_mixer": 64,
}
EMPTY_REGION_INFRASTRUCTURE = {
    "__amd_rocclr_fillBufferUnAligned",
    "__amd_rocclr_copyBuffer",
    "void ninfer::ops::r9700::eager::(anonymous namespace)::fill_kernel<int>"
    "(int*, unsigned long, int)",
}
MAX_EMPTY_REGION_INFRASTRUCTURE_DISPATCHES = 112
EXPECTED_WORKLOAD = {
    "concurrency": 1, "prompt_tokens": 2048, "generated_tokens": 0,
    "prefill_chunk": 4096, "draft_tokens": 0, "spec": "none",
    "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 16,
    "xattention_profile": "dense",
}
EXPECTED_CLAIMS = {
    "admissible": (
        "selected-region dispatch inventory and stage/operator service attribution for the exact "
        "retained-production P2048 route"),
    "not_admissible": (
        "profiled throughput, physical memory bandwidth, peak utilization, causal stall "
        "attribution, or proof of practical-ceiling closure"),
}
EXPECTED_LIMITATIONS = [
    "The retained 1904.339303 tok/s unprofiled authority owns throughput; rocprof durations are attribution-only.",
    "ROCTX association owns stage attribution; kernel symbols classify operators but do not infer a missing stage marker.",
    "At most 112 known short runtime fill/copy dispatches may have an empty ROCTX region; they are surfaced and make stage attribution incomplete.",
    "The trace has no hardware counters and cannot establish physical HBM bandwidth, cache-hit rate, peak utilization, stall cause, or stall freedom.",
    "Power files prove only auto state at the capture endpoints.",
]


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} is not a safe regular file")
    return {
        "path": str(resolved),
        "file_size_bytes": resolved.stat().st_size,
        "sha256": _sha(resolved),
    }


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (
        (_ for _ in ()).throw(ValueError(f"{label} contains non-finite JSON {value}"))))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _match(identity: object, actual: dict[str, Any], label: str) -> None:
    if not isinstance(identity, dict) or identity != actual:
        raise ValueError(f"{label} differs from the immutable plan")


def _one_option(command: list[str], name: str) -> str:
    indexes = [index for index, value in enumerate(command) if value == name]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"benchmark command lacks exactly one {name}")
    return command[indexes[0] + 1]


def _one_table(objects: set[str], prefix: str) -> str:
    names = [name for name in objects if name.startswith(prefix)]
    if len(names) != 1:
        raise ValueError(f"trace database lacks exactly one {prefix} table")
    return names[0]


def _validate_database_contract(database: Path, command: list[str], root: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        objects = {row[0] for row in connection.execute(
            "select name from sqlite_master where type in ('table','view')")}
        process_table = _one_table(objects, "rocpd_info_process_")
        agent_table = _one_table(objects, "rocpd_info_agent_")
        processes = list(connection.execute(
            f'select nid, pid, command, environment, extdata from "{process_table}" order by id'))
        if len(processes) != 1 or processes[0]["command"] != " ".join(command):
            raise ValueError("trace database process argv differs from the planned benchmark")
        environment = json.loads(processes[0]["environment"])
        extdata = json.loads(processes[0]["extdata"])
        expected_environment = {
            "ROCPROF_OUTPUT_FILE_NAME": "retained-production-p2048",
            "ROCPROF_OUTPUT_PATH": str(root / "raw"),
            "ROCPROF_OUTPUT_FORMAT": "rocpd",
            "ROCPROF_MARKER_API_TRACE": "1",
            "ROCPROF_KERNEL_TRACE": "1",
            "ROCPROF_MEMORY_COPY_TRACE": "1",
            "ROCPROF_SELECTED_REGIONS": "1",
            "ROCPROFILER_REGISTER_LIBRARY":
                "/opt/rocm/core-10.0/lib/librocprofiler-sdk.so.1.3.5",
        }
        if not isinstance(environment, dict) or any(
                environment.get(key) != value for key, value in expected_environment.items()):
            raise ValueError("trace database profiler environment differs from the plan")
        expected_extdata = {
            "output_path": str(root / "raw"),
            "output_file": "retained-production-p2048",
            "raw_output_path": str(root / "raw"),
            "raw_output_file": "retained-production-p2048",
            "rocpd_output": True,
            "kernel_rename": False,
        }
        if not isinstance(extdata, dict) or any(
                extdata.get(key) != value for key, value in expected_extdata.items()):
            raise ValueError("trace database profiler output metadata differs from the plan")

        gpu_rows = list(connection.execute(
            f'select nid, pid, absolute_index, logical_index, type_index, name, product_name, '
            f'extdata from "{agent_table}" '
            "where type='GPU' order by absolute_index"))
        r9700 = [row for row in gpu_rows if row["name"] == EXPECTED_ARCH
                 and row["product_name"] == EXPECTED_GPU]
        if len(r9700) != 1:
            raise ValueError("trace database lacks exactly one R9700/gfx1201 agent")
        if r9700[0]["logical_index"] != 1 or r9700[0]["type_index"] != 0:
            raise ValueError("R9700 agent indices are not logical=1/type=0")
        agent_data = json.loads(r9700[0]["extdata"])
        if (not isinstance(agent_data, dict) or agent_data.get("cu_count") != 64
                or agent_data.get("simd_count") != 128
                or agent_data.get("wave_front_size") != 32):
            raise ValueError("R9700 agent topology is not 64 CU/128 SIMD/wave32")
        process_nid, process_pid = processes[0]["nid"], processes[0]["pid"]
        if any(row["nid"] != process_nid or row["pid"] != process_pid for row in gpu_rows):
            raise ValueError("GPU agent inventory is not owned by the traced process/node")
        r9700_index = r9700[0]["absolute_index"]
        ownership = [tuple(row) for row in connection.execute(
            "select nid, pid, agent_abs_index, agent_log_index, agent_type_index, agent_type, "
            "count(*) calls from kernels group by 1,2,3,4,5,6")]
        expected_ownership = [(process_nid, process_pid, r9700_index, 1, 0, "GPU",
                               sum(row[-1] for row in ownership))]
        if ownership != expected_ownership:
            raise ValueError("one or more selected kernels did not execute on the R9700")

        regions = []
        controls = []
        for row in connection.execute(
                'select nid, pid, category, name, start, "end", duration, extdata '
                'from regions order by start'):
            if row["nid"] != process_nid or row["pid"] != process_pid:
                raise ValueError("trace marker is not owned by the traced process/node")
            begin, end, duration = int(row["start"]), int(row["end"]), int(row["duration"])
            if begin < 0 or end <= begin or duration != end - begin:
                raise ValueError("trace contains an invalid marker/control interval")
            if row["category"] == "MARKER_CONTROL_API":
                controls.append((begin, end, row["name"], row["extdata"]))
                continue
            if row["category"] != "MARKER_CORE_RANGE_API" or row["name"] != "roctxThreadRangeA":
                raise ValueError("trace contains an unexpected marker record kind")
            data = json.loads(row["extdata"])
            message = data.get("message") if isinstance(data, dict) else None
            if not isinstance(message, str):
                raise ValueError("trace region lacks a string message")
            regions.append((begin, end, message))
        if (len(controls) != 2 or [row[2] for row in controls]
                != ["roctxProfilerResume", "roctxProfilerPause"]
                or any(json.loads(row[3]) != {} for row in controls)):
            raise ValueError("trace lacks exact profiler resume/pause control records")
        marker_counts: dict[str, int] = {}
        marker_payloads: dict[str, set[int]] = {}
        measured = []
        text = []
        for begin, end, message in regions:
            if end <= begin:
                raise ValueError("trace contains a non-positive marker interval")
            family = message.rsplit(" payload=", 1)[0]
            if family:
                marker_counts[family] = marker_counts.get(family, 0) + 1
                if family != "ninfer_bench_measured":
                    parts = message.rsplit(" payload=", 1)
                    if len(parts) != 2 or not parts[1].isdigit():
                        raise ValueError("prefill marker lacks an exact integer payload")
                    marker_payloads.setdefault(family, set()).add(int(parts[1]))
            if message == "ninfer_bench_measured":
                measured.append((begin, end))
            if family == "ninfer.prefill.prefill.chunk":
                if message != "ninfer.prefill.prefill.chunk payload=2048":
                    raise ValueError("Text-prefill marker does not carry exact P2048 payload")
                text.append((begin, end))
        if marker_counts != EXPECTED_MARKERS or len(measured) != 1 or len(text) != 1:
            raise ValueError("trace marker inventory differs from exact Qwen3.8-27B prefill")
        if not (controls[0][1] <= measured[0][0] < measured[0][1] <= controls[1][0]):
            raise ValueError("measured marker is not bounded by profiler resume/pause")
        all_layers = set(range(64))
        full_layers = set(range(3, 64, 4))
        gdn_layers = all_layers - full_layers
        expected_payloads = {
            "ninfer.prefill.prefill.chunk": {2048},
            "ninfer.attention.prefill.layer.full": full_layers,
            "ninfer.attention.prefill.attention": full_layers,
            "ninfer.gdn.prefill.layer.gdn": gdn_layers,
            "ninfer.gdn.prefill.gdn": gdn_layers,
            "ninfer.post-mixer.prefill.post_mixer": all_layers,
        }
        if marker_payloads != expected_payloads:
            raise ValueError("trace marker payload sets differ from exact 64-layer topology")
        if not (measured[0][0] <= text[0][0] <= text[0][1] <= measured[0][1]):
            raise ValueError("Text-prefill marker is not nested in the measured range")
        for begin, end, message in regions:
            family = message.rsplit(" payload=", 1)[0]
            if family and family != "ninfer_bench_measured" and not (
                    text[0][0] <= begin <= end <= text[0][1]):
                raise ValueError("prefill stage marker is not nested in the Text chunk")
        by_message = {message: (begin, end) for begin, end, message in regions if message}
        layer_ranges = []
        for layer in range(64):
            kind = ("attention.prefill.layer.full" if layer in full_layers
                    else "gdn.prefill.layer.gdn")
            leaf = ("attention.prefill.attention" if layer in full_layers
                    else "gdn.prefill.gdn")
            outer = by_message[f"ninfer.{kind} payload={layer}"]
            inner = by_message[f"ninfer.{leaf} payload={layer}"]
            post = by_message[f"ninfer.post-mixer.prefill.post_mixer payload={layer}"]
            if not (outer[0] <= inner[0] < inner[1] <= post[0] < post[1] <= outer[1]):
                raise ValueError("layer leaf/post-mixer marker order or nesting is invalid")
            layer_ranges.append(outer)
        if any(left[1] > right[0] for left, right in zip(layer_ranges, layer_ranges[1:])):
            raise ValueError("layer marker ranges overlap or are not sequential")

        kernel_rows = list(connection.execute(
            'select nid, pid, dispatch_id, name, coalesce(region, "") region, start, "end", '
            'duration from kernels'))
        dispatch_ids = [row["dispatch_id"] for row in kernel_rows]
        if (not dispatch_ids or any(type(value) is not int or value < 0 for value in dispatch_ids)
                or len(set(dispatch_ids)) != len(dispatch_ids)):
            raise ValueError("selected dispatch IDs are absent, invalid, or duplicated")
        for row in kernel_rows:
            if (row["nid"] != process_nid or row["pid"] != process_pid
                    or type(row["duration"]) is not int or row["duration"] <= 0
                    or row["end"] - row["start"] != row["duration"]):
                raise ValueError("selected kernel process/timing identity is invalid")
        infrastructure = [row for row in kernel_rows if not row["region"]
                          and text[0][0] <= row["start"] < row["end"] <= text[0][1]]
        crossing_unmarked = [row for row in kernel_rows if not row["region"]
                             and row["start"] < text[0][1] and row["end"] > text[0][0]
                             and row not in infrastructure]
        if crossing_unmarked:
            raise ValueError("unmarked selected kernel crosses a Text-prefill boundary")
        measured_other_unmarked = [row for row in kernel_rows if not row["region"]
                                   and row not in infrastructure]
        bad_infrastructure = sorted({row["name"] for row in infrastructure}
                                    - EMPTY_REGION_INFRASTRUCTURE)
        infra_durations = [int(row["end"]) - int(row["start"]) for row in infrastructure]
        text_wall = text[0][1] - text[0][0]
        if (bad_infrastructure
                or len(infrastructure) > MAX_EMPTY_REGION_INFRASTRUCTURE_DISPATCHES
                or any(duration > 20_000 for duration in infra_durations)
                or sum(infra_durations) > 1_000_000
                or sum(infra_durations) / text_wall > 0.001):
            raise ValueError("unmarked selected kernels exceed the bounded infrastructure allowance")
        for row in kernel_rows:
            if not (measured[0][0] <= row["start"] < row["end"] <= measured[0][1]):
                raise ValueError("selected kernel is not fully inside the measured range")
            region = row["region"]
            if region and region not in by_message:
                raise ValueError("selected kernel has an unexpected ROCTX association")
            if region and row["start"] < by_message[region][0]:
                raise ValueError("selected kernel starts before its associated ROCTX marker")
        copies = list(connection.execute(
            'select nid, pid, start, "end", duration, coalesce(region_name, "") region '
            "from memory_copies"))
        if copies:
            raise ValueError("exact retained P2048 selected region unexpectedly contains copies")
        return {
            "gpu_agents": [
                {"absolute_index": row["absolute_index"], "architecture": row["name"],
                 "product_name": row["product_name"],
                 "logical_index": row["logical_index"], "type_index": row["type_index"]}
                for row in gpu_rows
            ],
            "selected_kernel_owner": {
                "absolute_index": r9700_index, "architecture": EXPECTED_ARCH,
                "product_name": EXPECTED_GPU, "cu_count": 64, "simd_count": 128,
                "wave_front_size": 32,
            },
            "marker_inventory": marker_counts,
            "empty_region_infrastructure_dispatches": len(infrastructure),
            "empty_region_infrastructure_summed_duration_ns": sum(infra_durations),
            "empty_region_infrastructure_max_duration_ns": max(infra_durations, default=0),
            "empty_region_infrastructure_text_wall_fraction": (
                sum(infra_durations) / text_wall),
            "empty_region_infrastructure_symbols": sorted(
                {row["name"] for row in infrastructure}),
            "measured_other_unmarked_dispatches": len(measured_other_unmarked),
            "stage_attribution_complete": len(infrastructure) == 0,
        }
    finally:
        connection.close()


def _validate_unprofiled(authority: dict[str, Any], plan: dict[str, Any],
                          snapshot: dict[str, Any]) -> None:
    _match(plan.get("unprofiled_authority", {}).get("file"), snapshot,
           "unprofiled timing authority")
    decision = authority.get("decision")
    inputs = authority.get("input_identities")
    before, after = authority.get("power_before"), authority.get("power_after")
    expected = plan["unprofiled_authority"]
    if (
        authority.get("schema") != "ninfer_r9700_scalar_base_whole_ab"
        or authority.get("schema_version") != 1
        or authority.get("status") != "measured"
        or not isinstance(decision, dict)
        or decision.get("promotion_pass") is not True
        or not isinstance(inputs, dict)
        or not isinstance(before, dict)
        or not isinstance(after, dict)
        or before.get("sysfs_value") != "auto"
        or after.get("sysfs_value") != "auto"
        or before.get("device", {}).get("architecture") != EXPECTED_ARCH
        or after.get("device", {}).get("name") != EXPECTED_GPU
    ):
        raise ValueError("unprofiled authority is not the retained measured promotion result")
    milliseconds = decision.get("floor_diagnostic_candidate_prefill_median_ms")
    if (isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float))
            or not math.isfinite(milliseconds) or milliseconds <= 0):
        raise ValueError("unprofiled authority lacks a finite positive P2048 median")
    tokens_per_second = 2048.0 / (float(milliseconds) / 1000.0)
    if (milliseconds != expected.get("prefill_median_ms")
            or tokens_per_second != expected.get("prefill_tok_s")):
        raise ValueError("unprofiled P2048 summary differs from its authority")
    artifact = inputs.get("artifact")
    corpus = inputs.get("corpus")
    candidate_binary = inputs.get("build_receipts", {}).get("candidate", {}).get("binary")
    if (
        not isinstance(artifact, dict)
        or artifact.get("path") != str(Path(plan["artifact"]["path"]).relative_to(
            Path(plan["repository_root"])))
        or artifact.get("file_size_bytes") != plan["artifact"]["file_size_bytes"]
        or artifact.get("sha256") != plan["artifact"]["sha256"]
        or artifact.get("weights_id") != plan["artifact"]["weights_id"]
        or not isinstance(corpus, dict)
        or corpus.get("path") != str(Path(plan["corpus"]["path"]).relative_to(
            Path(plan["repository_root"])))
        or corpus.get("file_size_bytes") != plan["corpus"]["file_size_bytes"]
        or corpus.get("sha256") != plan["corpus"]["sha256"]
        or not isinstance(candidate_binary, dict)
        or candidate_binary.get("file_size_bytes")
            != plan["unprofiled_authority"]["prior_executable"]["file_size_bytes"]
        or candidate_binary.get("sha256")
            != plan["unprofiled_authority"]["prior_executable"]["sha256"]
        or Path(str(candidate_binary.get("path", "")))
            != Path(plan["unprofiled_authority"]["prior_executable"]["path"]).relative_to(
                Path(plan["repository_root"]))
    ):
        raise ValueError("unprofiled authority input identity differs from the planned route")


def _validate_repair(repair_path: Path, plan_snapshot: dict[str, Any],
                     plan_analyzer: dict[str, Any], current_analyzer: dict[str, Any],
                     outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    repair_snapshot = _snapshot(repair_path, "post-capture repair")
    repair = _load(Path(repair_snapshot["path"]), "post-capture repair")
    capture_root = Path(plan_snapshot["path"]).parent
    capture_commands = _snapshot(capture_root / "commands.sh", "capture commands")
    capture_closure = _snapshot(capture_root / "prepared.sha256", "capture closure")
    if (
        repair.get("artifact_type") != "ninfer_r9700_retained_prefill_trace_analysis_repair"
        or repair.get("schema_version") != 1
        or repair.get("status") != "analysis_only_no_recapture"
        or repair.get("cause")
            != "absolute analyzer invocation lacked repository root on sys.path"
        or repair.get("capture_plan") != plan_snapshot
        or repair.get("capture_analyzer") != plan_analyzer
        or repair.get("repaired_analyzer") != current_analyzer
        or repair.get("capture_commands") != capture_commands
        or repair.get("capture_closure") != capture_closure
        or repair.get("capture_outputs") != outputs
        or repair.get("recapture_permitted") is not False
    ):
        raise ValueError("post-capture repair does not bind the exact failed analysis boundary")
    return repair_snapshot, repair


def validate(plan_path: Path, root: Path,
             repair_path: Path | None = None) -> dict[str, Any]:
    plan_snapshot = _snapshot(plan_path, "plan")
    plan = _load(Path(plan_snapshot["path"]), "plan")
    if (
        plan.get("artifact_type") != PLAN_TYPE
        or plan.get("schema_version") != 1
        or plan.get("status") != "command_only_not_executed"
        or plan.get("profile_timing_admissible") is not False
        or Path(str(plan.get("repository_root", ""))).resolve() != Path.cwd().resolve()
        or plan.get("workload") != EXPECTED_WORKLOAD
        or plan.get("claims") != EXPECTED_CLAIMS
        or plan.get("limitations") != EXPECTED_LIMITATIONS
        or plan.get("power_profile") != {
            "path": "/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level",
            "required_before_after": "auto",
        }
    ):
        raise ValueError("input is not the exact retained-prefill trace plan")
    root = root.resolve(strict=True)
    if root != Path(plan_snapshot["path"]).parent or root != Path(plan["output_root"]):
        raise ValueError("output root differs from the immutable plan")

    identities = {}
    for label, key in (
        ("benchmark executable", "benchmark_executable"),
        ("artifact", "artifact"), ("corpus", "corpus"),
        ("profiler", "profiler"), ("legacy analyzer", "legacy_analyzer"),
    ):
        identity = plan.get(key)
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError(f"plan lacks {label} identity")
        actual = _snapshot(Path(identity["path"]), label)
        comparable = {name: identity.get(name) for name in actual}
        _match(comparable, actual, label)
        identities[key] = actual
    current_analyzer = _snapshot(Path(__file__), "retained-prefill analyzer")
    plan_analyzer = plan.get("analyzer")
    if not isinstance(plan_analyzer, dict):
        raise ValueError("plan lacks capture-time analyzer identity")
    authority_identity = plan.get("unprofiled_authority", {}).get("file")
    if not isinstance(authority_identity, dict) or not isinstance(
            authority_identity.get("path"), str):
        raise ValueError("plan lacks unprofiled timing authority")
    authority_snapshot = _snapshot(Path(authority_identity["path"]), "unprofiled authority")
    prior_identity = plan["unprofiled_authority"].get("prior_executable")
    if not isinstance(prior_identity, dict) or not isinstance(prior_identity.get("path"), str):
        raise ValueError("plan lacks prior unprofiled executable identity")
    prior_snapshot = _snapshot(Path(prior_identity["path"]), "prior unprofiled executable")
    _match(prior_identity, prior_snapshot, "prior unprofiled executable")
    authority = _load(Path(authority_snapshot["path"]), "unprofiled authority")
    _validate_unprofiled(authority, plan, authority_snapshot)

    outputs = plan.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("plan lacks outputs")
    expected_outputs = {
        "benchmark_report": str(root / "benchmark-report.json"),
        "database": str(root / "raw/retained-production-p2048_results.db"),
        "evidence": str(root / "evidence.json"),
        "power_before": str(root / "power-profile-before.txt"),
        "power_after": str(root / "power-profile-after.txt"),
    }
    if outputs != expected_outputs:
        raise ValueError("capture output names differ from the immutable plan")
    captured_outputs = {
        "benchmark_report": _snapshot(Path(outputs["benchmark_report"]), "benchmark report"),
        "database": _snapshot(Path(outputs["database"]), "trace database"),
        "power_before": _snapshot(Path(outputs["power_before"]), "power before"),
        "power_after": _snapshot(Path(outputs["power_after"]), "power after"),
    }
    repair_snapshot = None
    if plan_analyzer != current_analyzer:
        if repair_path is None:
            raise ValueError("capture-time analyzer changed without a post-capture repair authority")
        repair_snapshot, _ = _validate_repair(
            repair_path, plan_snapshot, plan_analyzer, current_analyzer, captured_outputs)
    elif repair_path is not None:
        raise ValueError("post-capture repair supplied although the capture analyzer is unchanged")
    identities["analyzer"] = current_analyzer
    report_path = Path(str(outputs.get("benchmark_report", ""))).resolve(strict=True)
    database_path = Path(str(outputs.get("database", ""))).resolve(strict=True)
    before_path = Path(str(outputs.get("power_before", ""))).resolve(strict=True)
    after_path = Path(str(outputs.get("power_after", ""))).resolve(strict=True)
    for path in (report_path, database_path, before_path, after_path):
        if path.parent != root and root not in path.parents:
            raise ValueError("capture output escaped the immutable output root")
    result_databases = sorted((root / "raw").glob("*_results.db"))
    if result_databases != [database_path]:
        raise ValueError("raw capture does not contain exactly the planned results database")

    command = plan.get("benchmark_command")
    profiler = plan.get("profiler_command")
    if (not isinstance(command, list) or not all(isinstance(value, str) for value in command)
            or not isinstance(profiler, list)
            or not all(isinstance(value, str) for value in profiler)):
        raise ValueError("plan command vectors are invalid")
    expected_profiler = [
        identities["profiler"]["path"], "--selected-regions", "-f", "rocpd", "-d",
        str(root / "raw"), "-o", "retained-production-p2048", "--marker-trace",
        "--kernel-trace", "--memory-copy-trace", "--", *command,
    ]
    if profiler != expected_profiler:
        raise ValueError("profiler command differs from the bounded trace contract")
    expected_command = [
        identities["benchmark_executable"]["path"], "--weights", identities["artifact"]["path"],
        "--corpus", identities["corpus"]["path"], "--device", "0", "--concurrency", "1",
        "-p", "2048", "--prefill-chunk", "4096", "--draft-tokens", "0", "--output",
        "json", "--output-file", str(report_path), "-r", "1", "--warmup", "1",
        "--profile-measured",
    ]
    if command != expected_command:
        raise ValueError("benchmark command differs from exact C1/P2048/G0/chunk4096 route")

    report_snapshot = _snapshot(report_path, "benchmark report")
    database_snapshot = _snapshot(database_path, "trace database")
    before_snapshot = _snapshot(before_path, "power before")
    after_snapshot = _snapshot(after_path, "power after")
    report = _load(report_path, "benchmark report")
    config = report.get("config")
    tests = report.get("tests")
    if (
        report.get("artifact_type") != "ninfer_bench_report"
        or report.get("schema_version") != 20
        or report.get("command") != " ".join(command)
        or report.get("environment", {}).get("gpu_name") != EXPECTED_GPU
        or report.get("environment", {}).get("architecture_name") != EXPECTED_ARCH
        or report.get("environment", {}).get("device_id") != 0
        or Path(str(report.get("artifact", {}).get("path", ""))).resolve()
            != Path(identities["artifact"]["path"])
        or report.get("artifact", {}).get("file_size_bytes")
            != identities["artifact"]["file_size_bytes"]
        or report.get("load", {}).get("weights_id") != plan["artifact"]["weights_id"]
        or not isinstance(config, dict)
        or config.get("max_context") != 2048
        or config.get("concurrency") != 1 or config.get("prefill_chunk") != 4096
        or config.get("kv_cache_format") != "fp8-k-int4-v"
        or config.get("kv_value_group") != 16
        or config.get("kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or config.get("spec") != "none" or config.get("draft_tokens") != 0
        or config.get("speculative_execution") is not False
        or config.get("dflash_verify_width_requested") != 0
        or config.get("dflash_verify_width") != 0
        or config.get("proposal_head") != "full"
        or config.get("use_device_graph") is not True
        or config.get("decode_path") != "device_graph"
        or config.get("decode_graph_prime") != {"primed": False, "output_tokens": 0}
        or config.get("q4_activation_bits") != 8
        or config.get("q4_prefill_cta_profile")
            != "m64n128-pingpong-n16-k16-scalar-base-production"
        or config.get("w8_activation_bits") != 8
        or config.get("fp8_qk_wmma_enabled") is not True
        or config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1"
        or config.get("fp8_qk_wmma_t1_min_context") != 64
        or config.get("fp8_qk_wmma_t2_min_context") != 320
        or config.get("xattention_qualification") is not False
        or config.get("repetitions") != 1 or config.get("warmup") != 1
        or not isinstance(tests, list) or len(tests) != 1
        or tests[0].get("kind") != "pp" or tests[0].get("n_prompt") != 2048
        or tests[0].get("n_gen") != 0
    ):
        raise ValueError("benchmark report is not the exact planned retained-prefill capture")
    test = tests[0]
    prefill_seconds = test.get("prefill_seconds_mean")
    prefill_tok_s = test.get("prefill_tok_s_mean")
    reps = test.get("reps")
    expected_speculative = {
        "enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
        "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
        "acceptance_length": None, "accepted_per_position": [],
    }
    null_metrics = ("decode_output_tok_s_mean", "decode_output_tok_s_stddev",
                    "decode_engine_tok_s_mean", "decode_engine_tok_s_stddev",
                    "whole_output_tok_s_mean", "whole_output_tok_s_stddev",
                    "decode_seconds_mean", "decode_seconds_stddev")
    if (test.get("requested_output_tokens") != 1
            or any(test.get(name) is not None for name in null_metrics)
            or isinstance(prefill_seconds, bool)
            or not isinstance(prefill_seconds, (int, float))
            or not math.isfinite(prefill_seconds) or prefill_seconds <= 0
            or isinstance(prefill_tok_s, bool)
            or not isinstance(prefill_tok_s, (int, float))
            or not math.isfinite(prefill_tok_s) or prefill_tok_s <= 0
            or not math.isclose(prefill_tok_s, 2048.0 / prefill_seconds,
                                rel_tol=1e-9, abs_tol=1e-9)
            or not isinstance(reps, list) or len(reps) != 1
            or test.get("speculative") != expected_speculative
            or reps[0].get("generated_output_tokens") != 1
            or reps[0].get("decode_output_tokens") is not None
            or reps[0].get("decode_engine_tokens") is not None
            or reps[0].get("speculative") != expected_speculative):
        raise ValueError("benchmark report timing/rep semantics are not exact P2048/G0")
    before = before_path.read_text(encoding="utf-8").strip()
    after = after_path.read_text(encoding="utf-8").strip()
    if before != "auto" or after != "auto":
        raise ValueError("capture lacks exact auto power endpoints")

    database_contract = _validate_database_contract(database_path, command, root)
    dispatches, aggregates = _parse_database(database_path, " ".join(command))
    attribution = analyze(database_path, report_path)
    if (
        sum(row["calls"] for row in attribution["kernel_execution_categories"])
            != aggregates["dispatch_count"]
        or round(sum(row["independent_summed_duration_ms"]
                     for row in attribution["kernel_execution_categories"]) * 1e6)
            != aggregates["independent_device_service_time_ns"]
    ):
        raise ValueError("attribution does not conserve database dispatch count/service")

    final_snapshots = {
        "plan": plan_snapshot, "benchmark_report": report_snapshot,
        "database": database_snapshot, "power_before": before_snapshot,
        "power_after": after_snapshot, "unprofiled_authority": authority_snapshot,
        "prior_unprofiled_executable": prior_snapshot,
        **identities,
    }
    if repair_snapshot is not None:
        final_snapshots["post_capture_repair"] = repair_snapshot
    commands_identity = plan.get("commands")
    if not isinstance(commands_identity, dict) or not isinstance(commands_identity.get("path"), str):
        raise ValueError("plan lacks commands identity")
    commands_snapshot = _snapshot(Path(commands_identity["path"]), "commands")
    _match(commands_identity, commands_snapshot, "commands")
    final_snapshots["commands"] = commands_snapshot
    for label, snapshot in final_snapshots.items():
        if _snapshot(Path(snapshot["path"]), label) != snapshot:
            raise ValueError(f"{label} changed while validating")
    return {
        "artifact_type": EVIDENCE_TYPE,
        "schema_version": 1,
        "status": "valid_attribution_only",
        "profile_timing_admissible": False,
        "inputs": final_snapshots,
        "workload": plan["workload"],
        "unprofiled_timing_authority": plan["unprofiled_authority"],
        "power_profile": {"required": "auto", "before": before, "after": after,
                          "endpoint_only": True},
        "aggregates": aggregates,
        "database_contract": database_contract,
        "dispatches": dispatches,
        "attribution": attribution,
        "claims": plan["claims"],
        "limitations": plan["limitations"],
    }


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repair", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.resolve() != args.root.resolve() / "evidence.json":
            raise ValueError("evidence output differs from the immutable plan")
        result = validate(args.plan, args.root, args.repair)
        _publish(args.out, result)
    except (OSError, sqlite3.Error, json.JSONDecodeError, KeyError, TypeError,
            ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
