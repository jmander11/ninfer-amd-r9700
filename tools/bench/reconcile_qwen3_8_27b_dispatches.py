#!/usr/bin/env python3
"""Reconcile one validated Qwen3.8-27B trace with an explicit static dispatch schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

TRACE_TYPE = "ninfer_r9700_selected_profile_trace"
SCHEDULE_TYPE = "ninfer_qwen3_8_27b_static_dispatch_schedule"
OUTPUT_TYPE = "ninfer_qwen3_8_27b_dispatch_reconciliation"
TIMING_TYPE = "ninfer_qwen3_8_27b_dispatch_timing"
INVENTORY_TYPE = "ninfer_qwen3_8_27b_dispatch_inventory"
SCHEMA_VERSION = 1
CLASSIFICATIONS = ("modeled", "unmodeled", "unsupported")


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved)}


def _verify_snapshot(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a file snapshot")
    path = Path(str(value.get("path", "")))
    actual = _snapshot(path, label)
    if value.get("path") != actual["path"] or value.get("sha256") != actual["sha256"]:
        raise ValueError(f"{label} path or SHA-256 changed")
    size = value.get("file_size_bytes", value.get("bytes"))
    if type(size) is not int or size != actual["file_size_bytes"]:
        raise ValueError(f"{label} file size changed")
    return actual


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _dimensions(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
        raise ValueError(f"{label} must contain exactly x/y/z")
    return {axis: _integer(value[axis], f"{label}.{axis}", 1) for axis in ("x", "y", "z")}


def _nullable_integer(value: object, label: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum)


def _wall_union_ns(rows: list[dict[str, Any]]) -> int:
    intervals = sorted((row["start_ns"], row["end_ns"]) for row in rows)
    begin, end = intervals[0]
    total = 0
    for next_begin, next_end in intervals[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_trace(trace_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace = _load(trace_path, "validated trace")
    schema = trace.get("schema_version")
    if (trace.get("artifact_type") != TRACE_TYPE or schema not in (1, 2)
            or trace.get("status") != "valid_attribution_only"
            or trace.get("profile_timing_admissible") is not False):
        raise ValueError("trace must be validated selected-profile trace schema v1")
    _verify_snapshot(trace.get("plan"), "trace plan")
    workload = trace.get("workload")
    if not isinstance(workload, dict) or (
        workload.get("concurrency"), workload.get("prompt_tokens"),
        workload.get("generated_tokens"), workload.get("spec"),
        workload.get("draft_tokens")
    ) != (1, 2048, 0, "none", 0):
        raise ValueError("trace is not the C1 P2048 spec-none workload")
    if workload.get("xattention_profile") not in ("dense", "b128-s16-tau900"):
        raise ValueError("trace has an invalid selected attention profile")
    if workload.get("kv_value_group") not in (16, 32):
        raise ValueError("trace has an invalid selected KV value group")
    _integer(workload.get("prefill_chunk"), "workload.prefill_chunk", 1)
    power = trace.get("power_profile")
    if not isinstance(power, dict) or any(power.get(key) != "auto"
                                          for key in ("required", "before", "after")):
        raise ValueError("trace endpoints must all be auto")
    inputs = trace.get("inputs")
    authorities = trace.get("authorities")
    expected_inputs = {"benchmark_report", "database", "power_before", "power_after",
                       "terminal_selection", "artifact", "benchmark_executable", "corpus"}
    expected_authorities = ({"source_matrix", "source_report", "low_context_evaluation"}
                            if schema == 1 else {"source_matrix"})
    if not isinstance(inputs, dict) or not expected_inputs.issubset(inputs):
        raise ValueError("trace lacks its complete input snapshots")
    if not isinstance(authorities, dict) or not expected_authorities.issubset(authorities):
        raise ValueError("trace lacks its complete authority snapshots")
    for name, value in inputs.items():
        _verify_snapshot(value, f"trace input {name}")
    for name, value in authorities.items():
        _verify_snapshot(value, f"trace authority {name}")
    route = trace.get("selected_route")
    if not isinstance(route, dict) or route.get("kind") != "pp":
        raise ValueError("trace lacks its validated selected_route")
    for name in ("artifact", "benchmark_executable"):
        selected = route.get(name)
        if not isinstance(selected, dict):
            raise ValueError(f"selected_route.{name} is absent")
        exact = _verify_snapshot(selected, f"selected route {name}")
        if exact != _verify_snapshot(inputs[name], f"trace input {name}"):
            raise ValueError(f"selected_route.{name} differs from trace inputs")
    _text(route["artifact"].get("weights_id"), "selected artifact weights_id")
    for name in ("prompt_tokens", "concurrency", "prefill_chunk", "kv_value_group",
                 "xattention_profile"):
        if route.get(name) != workload.get(name):
            raise ValueError(f"selected_route differs from workload at {name}")

    rows = trace.get("dispatches")
    if not isinstance(rows, list) or not rows:
        raise ValueError("trace dispatches must be a nonempty array")
    ids: set[str] = set()
    rocprof_ids: set[int] = set()
    collisions: set[tuple[Any, ...]] = set()
    signatures: set[tuple[Any, ...]] = set()
    validated = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"trace dispatch {index} is not an object")
        dispatch_id = _text(row.get("dispatch_id"), f"trace dispatch {index}.dispatch_id")
        rocprof_id = _integer(row.get("rocprof_dispatch_id"),
                              f"trace dispatch {dispatch_id}.rocprof_dispatch_id")
        if dispatch_id in ids or rocprof_id in rocprof_ids:
            raise ValueError("trace dispatch identifiers are not unique")
        ids.add(dispatch_id); rocprof_ids.add(rocprof_id)
        symbol = _text(row.get("symbol"), f"trace dispatch {dispatch_id}.symbol")
        marker = row.get("roctx_region")
        if marker is not None and (not isinstance(marker, str) or not marker):
            raise ValueError(f"trace dispatch {dispatch_id}.roctx_region must be null or nonempty")
        start = _integer(row.get("start_ns"), f"trace dispatch {dispatch_id}.start_ns")
        end = _integer(row.get("end_ns"), f"trace dispatch {dispatch_id}.end_ns", 1)
        duration = _integer(row.get("duration_ns"), f"trace dispatch {dispatch_id}.duration_ns", 1)
        if end <= start or end - start != duration:
            raise ValueError(f"trace dispatch {dispatch_id} has inconsistent duration")
        grid = _dimensions(row.get("grid"), f"trace dispatch {dispatch_id}.grid")
        workgroup = _dimensions(row.get("workgroup"), f"trace dispatch {dispatch_id}.workgroup")
        stream_id = _nullable_integer(
            row.get("stream_id"), f"trace dispatch {dispatch_id}.stream_id")
        resources = row.get("resources")
        resource_names = {
            "sgpr_count", "vgpr_count", "accum_vgpr_count", "lds_bytes", "scratch_bytes",
            "static_lds_bytes", "static_scratch_bytes",
        }
        if not isinstance(resources, dict) or set(resources) != resource_names:
            raise ValueError(
                f"trace dispatch {dispatch_id}.resources must contain the exact trace fields")
        resources = {
            name: _nullable_integer(value, f"trace dispatch {dispatch_id}.resources.{name}")
            for name, value in resources.items()
        }
        signature = (start, marker, symbol, tuple(grid.values()), tuple(workgroup.values()))
        if signature in signatures:
            collisions.add(signature)
        signatures.add(signature)
        validated.append(dict(row, dispatch_id=dispatch_id, rocprof_dispatch_id=rocprof_id,
                              symbol=symbol, roctx_region=marker, start_ns=start, end_ns=end,
                              duration_ns=duration, stream_id=stream_id, resources=resources,
                              grid=grid, workgroup=workgroup))
    if collisions:
        raise ValueError("trace has simultaneous indistinguishable dispatch collisions")
    if validated != sorted(validated, key=lambda row: (row["start_ns"], row["dispatch_id"])):
        raise ValueError("trace dispatch order is not start_ns then dispatch_id")
    aggregates = trace.get("aggregates")
    if not isinstance(aggregates, dict) or aggregates.get("dispatch_count") != len(validated):
        raise ValueError("trace dispatch count aggregate differs")
    if aggregates.get("independent_device_service_time_ns") != sum(
            row["duration_ns"] for row in validated):
        raise ValueError("trace service-time aggregate differs")
    if aggregates.get("device_wall_union_ns") != _wall_union_ns(validated):
        raise ValueError("trace wall-union aggregate differs")
    return trace, validated


def _validate_schedule(schedule_path: Path, workload: dict[str, Any],
                       selected_route: dict[str, Any], count: int
                       ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schedule = _load(schedule_path, "static schedule")
    if schedule.get("artifact_type") != SCHEDULE_TYPE or schedule.get("schema_version") != 1:
        raise ValueError("static schedule must be Qwen3.8-27B schema v1")
    if schedule.get("workload") != workload:
        raise ValueError("static schedule workload differs from validated trace")
    if schedule.get("selected_route") != selected_route:
        raise ValueError("static schedule selected route differs from validated trace")
    if schedule.get("dispatch_order") != "trace_start_ns_then_dispatch_id":
        raise ValueError("static schedule has an unsupported dispatch order")
    sources = schedule.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("static schedule must bind at least one source")
    for index, source in enumerate(sources):
        _verify_snapshot(source, f"static schedule source {index}")
    rows = schedule.get("dispatches")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"static schedule dispatch count {len(rows) if isinstance(rows, list) else 0} differs from trace {count}")
    calls: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    validated = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("schedule_index") != index:
            raise ValueError("static schedule indices must be exact and contiguous")
        marker = row.get("marker")
        if marker is not None and (not isinstance(marker, str) or not marker):
            raise ValueError(f"schedule {index}.marker must be null or nonempty")
        symbol = _text(row.get("symbol"), f"schedule {index}.symbol")
        grid = _dimensions(row.get("grid"), f"schedule {index}.grid")
        workgroup = _dimensions(row.get("workgroup"), f"schedule {index}.workgroup")
        layer = row.get("layer")
        if not isinstance(layer, dict) or set(layer) != {"kind", "index"}:
            raise ValueError(f"schedule {index}.layer must contain exactly kind/index")
        _text(layer.get("kind"), f"schedule {index}.layer.kind")
        if layer["index"] is not None:
            _integer(layer["index"], f"schedule {index}.layer.index")
        call = row.get("call")
        if not isinstance(call, dict) or set(call) != {"id", "ordinal", "dispatch_index", "dispatch_count"}:
            raise ValueError(f"schedule {index}.call has an invalid shape")
        call_id = _text(call.get("id"), f"schedule {index}.call.id")
        _integer(call.get("ordinal"), f"schedule {index}.call.ordinal")
        _integer(call.get("dispatch_index"), f"schedule {index}.call.dispatch_index")
        _integer(call.get("dispatch_count"), f"schedule {index}.call.dispatch_count", 1)
        classification = row.get("classification")
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"schedule {index} has invalid classification")
        if classification == "modeled":
            if marker is None:
                raise ValueError(f"modeled schedule {index} requires an exact ROCTX marker")
            for name in ("role", "operation", "format"):
                _text(row.get(name), f"schedule {index}.{name}")
            if not isinstance(row.get("parameters"), dict):
                raise ValueError(f"schedule {index}.parameters must be an object")
            if "reason" in row:
                raise ValueError("modeled schedule row cannot carry an uncovered reason")
        else:
            for name in ("role", "operation"):
                _text(row.get(name), f"schedule {index}.{name}")
            _text(row.get("reason"), f"schedule {index}.reason")
        exact = dict(row, marker=marker, symbol=symbol, grid=grid, workgroup=workgroup)
        validated.append(exact)
        calls[call_id].append((index, call, exact))
    ordinals = []
    for call_id, members in calls.items():
        indices = [item[0] for item in members]
        call = members[0][1]
        expected = list(range(indices[0], indices[0] + len(indices)))
        if indices != expected or len(members) != call["dispatch_count"]:
            raise ValueError(f"call {call_id} dispatches are not one exact contiguous multiplicity")
        if [item[1]["dispatch_index"] for item in members] != list(range(len(members))):
            raise ValueError(f"call {call_id} dispatch order is not contiguous")
        if any(item[1] != {**call, "dispatch_index": offset}
               for offset, item in enumerate(members)):
            raise ValueError(f"call {call_id} metadata differs within one call")
        first = members[0][2]
        for _, _, member in members[1:]:
            if member["marker"] != first["marker"] or member["layer"] != first["layer"]:
                raise ValueError(f"call {call_id} crosses marker or layer identity")
        ordinals.append(call["ordinal"])
    if sorted(ordinals) != list(range(len(ordinals))):
        raise ValueError("static call ordinals must be unique and contiguous")
    return schedule, validated


def _unprofiled(trace: dict[str, Any]) -> dict[str, Any]:
    evaluation_snapshot = _verify_snapshot(
        trace["authorities"]["low_context_evaluation"], "low-context evaluation")
    report_snapshot = _verify_snapshot(trace["authorities"]["source_report"], "P2048 source report")
    evaluation = _load(Path(evaluation_snapshot["path"]), "low-context evaluation")
    report = _load(Path(report_snapshot["path"]), "P2048 source report")
    if (evaluation.get("artifact_type") != "ninfer_r9700_low_context_prefill_evaluation"
            or evaluation.get("schema_version") != 1):
        raise ValueError("low-context evaluation has the wrong schema")
    ladder = evaluation.get("ladder")
    matches = [row for row in ladder if isinstance(row, dict) and row.get("prompt_tokens") == 2048] if isinstance(ladder, list) else []
    tests = report.get("tests")
    report_matches = [row for row in tests if isinstance(row, dict) and row.get("kind") == "pp"
                      and row.get("n_prompt") == 2048] if isinstance(tests, list) else []
    if len(matches) != 1 or len(report_matches) != 1:
        raise ValueError("unprofiled authorities lack one exact P2048 row")
    speed = matches[0].get("prefill_tok_s_mean")
    report_speed = report_matches[0].get("prefill_tok_s_mean")
    seconds = report_matches[0].get("prefill_seconds_mean")
    if (type(speed) not in (int, float) or not math.isfinite(speed) or speed <= 0
            or type(report_speed) not in (int, float) or not math.isfinite(report_speed)
            or report_speed <= 0 or float(report_speed) != float(speed)
            or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
            or not math.isclose(float(speed), float(evaluation.get("observed_p2048_tok_s", -1)),
                                rel_tol=0, abs_tol=0)):
        raise ValueError("unprofiled P2048 performance authorities disagree")
    selected = trace["selected_route"].get("unprofiled_p2048")
    if (not isinstance(selected, dict)
            or _verify_snapshot(selected.get("evaluation"), "selected P2048 evaluation")
            != evaluation_snapshot
            or _verify_snapshot(selected.get("report"), "selected P2048 report")
            != report_snapshot
            or selected.get("observed_tok_s") != speed
            or selected.get("minimum_tok_s") != evaluation.get("minimum_p2048_tok_s")
            or selected.get("passes_gate") != evaluation.get("passes_p2048_gate")):
        raise ValueError("selected_route unprofiled P2048 authority differs")
    return {"prompt_tokens": 2048, "prefill_tok_s_mean": float(speed),
            "prefill_seconds_mean": float(seconds), "low_context_evaluation": evaluation_snapshot,
            "p2048_report": report_snapshot}


def reconcile(trace_path: Path, schedule_path: Path) -> dict[str, Any]:
    trace_snapshot = _snapshot(trace_path, "validated trace")
    schedule_snapshot = _snapshot(schedule_path, "static schedule")
    trace, trace_rows = _validate_trace(trace_path)
    schedule, schedule_rows = _validate_schedule(
        schedule_path, trace["workload"], trace["selected_route"], len(trace_rows))
    enriched_workload = dict(trace["selected_route"])
    artifact = enriched_workload.pop("artifact")
    executable = enriched_workload.pop("benchmark_executable")
    enriched_workload.pop("unprofiled_p2048", None)
    enriched_workload.update({"artifact_path": artifact["path"],
                              "artifact_sha256": artifact["sha256"],
                              "weights_id": artifact["weights_id"],
                              "executable_path": executable["path"],
                              "executable_sha256": executable["sha256"]})
    coverage = []
    modeled_timing = []
    modeled_inventory = []
    totals = {name: {"dispatch_count": 0, "duration_ns": 0} for name in CLASSIFICATIONS}
    for trace_row, expected in zip(trace_rows, schedule_rows, strict=True):
        for trace_name, schedule_name in (("roctx_region", "marker"), ("symbol", "symbol"),
                                          ("grid", "grid"), ("workgroup", "workgroup")):
            if trace_row[trace_name] != expected[schedule_name]:
                raise ValueError(
                    f"dispatch {trace_row['dispatch_id']} differs at {trace_name}; "
                    f"schedule index {expected['schedule_index']}")
        classification = expected["classification"]
        totals[classification]["dispatch_count"] += 1
        totals[classification]["duration_ns"] += trace_row["duration_ns"]
        row = {"dispatch_id": trace_row["dispatch_id"],
               "rocprof_dispatch_id": trace_row["rocprof_dispatch_id"],
               "schedule_index": expected["schedule_index"], "marker": expected["marker"],
               "layer": expected["layer"], "call": expected["call"],
               "symbol": trace_row["symbol"], "grid": trace_row["grid"],
               "workgroup": trace_row["workgroup"], "stream_id": trace_row.get("stream_id"),
               "resources": trace_row.get("resources"), "start_ns": trace_row["start_ns"],
               "end_ns": trace_row["end_ns"], "duration_ns": trace_row["duration_ns"],
               "classification": classification}
        if classification == "modeled":
            row.update({name: expected[name] for name in ("role", "operation", "format", "parameters")})
            modeled_timing.append({"dispatch_id": trace_row["dispatch_id"],
                                   "symbol": trace_row["symbol"], "stage": expected["marker"],
                                   "start_ns": trace_row["start_ns"], "end_ns": trace_row["end_ns"],
                                   "duration_ns": trace_row["duration_ns"]})
            modeled_inventory.append({"dispatch_id": trace_row["dispatch_id"],
                                      "symbol": trace_row["symbol"], "stage": expected["marker"],
                                      **{name: expected[name] for name in
                                         ("role", "operation", "format", "parameters")}})
        else:
            row.update({"role": expected["role"], "operation": expected["operation"],
                        "reason": expected["reason"]})
        coverage.append(row)
    trace_service = sum(row["duration_ns"] for row in trace_rows)
    if sum(value["duration_ns"] for value in totals.values()) != trace_service:
        raise ValueError("dispatch classification duration does not cover the trace")
    plan = _load(Path(trace["plan"]["path"]), "profile plan")
    sysfs_path = plan.get("required_power_profile", {}).get("sysfs_path")
    _text(sysfs_path, "profile plan power sysfs path")
    timing_source = {"trace_authority": trace_snapshot,
                     "source_matrix": trace["authorities"]["source_matrix"]}
    if trace["schema_version"] == 1:
        timing_source.update({
            "low_context_evaluation": trace["authorities"]["low_context_evaluation"],
            "benchmark_report": trace["authorities"]["source_report"],
        })
    timing = {"artifact_type": TIMING_TYPE, "schema_version": 1,
              "power_profile": {"required": "auto", "observed": "auto",
                                "rechecked_after": "auto",
                                "sysfs_path": sysfs_path},
              "source": timing_source,
              "workload": enriched_workload, "dispatches": modeled_timing}
    inventory = {"artifact_type": INVENTORY_TYPE, "schema_version": 1,
                 "timing_authority_sha256": _canonical_sha256(timing),
                 "workload": enriched_workload, "dispatches": modeled_inventory}
    result = {"artifact_type": OUTPUT_TYPE, "schema_version": SCHEMA_VERSION,
              "trace_authority": trace_snapshot, "static_schedule": schedule_snapshot,
              "schedule_sources": schedule["sources"], "workload": enriched_workload,
              "unprofiled_performance": (_unprofiled(trace)
                                           if trace["schema_version"] == 1 else None),
              "dispatches": coverage,
              "coverage": {"trace_dispatch_count": len(trace_rows),
                           "trace_service_time_ns": trace_service, **totals},
              "timing_authority": timing, "dispatch_inventory": inventory}
    if _snapshot(trace_path, "validated trace") != trace_snapshot:
        raise ValueError("validated trace changed during reconciliation")
    if _snapshot(schedule_path, "static schedule") != schedule_snapshot:
        raise ValueError("static schedule changed during reconciliation")
    for index, source in enumerate(schedule["sources"]):
        _verify_snapshot(source, f"static schedule source {index}")
    for owner in ("inputs", "authorities"):
        for name, source in trace[owner].items():
            _verify_snapshot(source, f"trace {owner} {name}")
    _verify_snapshot(trace["plan"], "trace plan")
    return result


def _publish(path: Path, value: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    published = False
    durable = False
    created_inode = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(value, indent=2) + "\n")
            output.flush(); os.fsync(output.fileno())
        temporary_stat = os.stat(temporary, follow_symlinks=False)
        created_inode = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.link(temporary, path)
        published = True
        output_stat = os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(output_stat.st_mode)
                or (output_stat.st_dev, output_stat.st_ino) != created_inode):
            raise ValueError("published dispatch reconciliation inode changed")
        if _load(path, "published reconciliation") != value:
            raise ValueError("published dispatch reconciliation differs")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        durable = True
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        if published and not durable and created_inode is not None:
            try:
                current = os.stat(path, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == created_inode:
                    path.unlink()
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-authority", required=True, type=Path)
    parser.add_argument("--static-schedule", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        value = reconcile(args.trace_authority, args.static_schedule)
        _publish(args.out, value)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote exact dispatch reconciliation to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
