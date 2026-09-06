#!/usr/bin/env python3
"""Analyze one ROCPD capture of one ordinary C1 decode round."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.analyze_whole_profile import _duration_ns, _message


MEASURED = "ninfer_bench_measured"
ORDINARY = "ninfer.decode.decode.ordinary_round payload=8192"
FORBIDDEN_ROUNDS = (".decode.mtp_round", ".decode.dflash_round")
RUNTIME_CATEGORY = "HIP_RUNTIME_API"


def _require_columns(connection: sqlite3.Connection, table: str,
                     required: set[str]) -> None:
    columns = {str(row[1]) for row in connection.execute(
        f'pragma table_info("{table}")')}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"ROCPD {table} view lacks required columns: {', '.join(missing)}")


def _interval(row: sqlite3.Row, kind: str) -> tuple[int, int, int]:
    begin, end, duration = row["start"], row["end"], row["duration"]
    if (type(begin) is not int or type(end) is not int or type(duration) is not int
            or begin < 0 or end <= begin or duration != end - begin):
        raise ValueError(f"{kind} record has inconsistent timing")
    return begin, end, duration


def _contains(parent: tuple[int, int], begin: int, end: int) -> bool:
    return parent[0] <= begin and end <= parent[1]


def _overlaps(parent: tuple[int, int], begin: int, end: int) -> bool:
    return max(parent[0], begin) < min(parent[1], end)


def _ms(value: int) -> float:
    return value / 1e6


def _aggregate_timed(rows: Iterable[sqlite3.Row], key_names: tuple[str, ...],
                     duration_prefix: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "duration": 0, "intervals": []})
    for row in rows:
        begin, end, duration = _interval(row, duration_prefix)
        key = tuple(str(row[name]) for name in key_names)
        value = grouped[key]
        value["count"] += 1
        value["duration"] += duration
        value["intervals"].append((begin, end))
    result = []
    for key, value in grouped.items():
        item: dict[str, Any] = dict(zip(key_names, key))
        item.update({
            "record_count": value["count"],
            f"summed_{duration_prefix}_duration_ns": value["duration"],
            f"summed_{duration_prefix}_duration_ms": _ms(value["duration"]),
            "interval_union_ns": _duration_ns(value["intervals"]),
            "interval_union_ms": _ms(_duration_ns(value["intervals"])),
        })
        result.append(item)
    return sorted(result, key=lambda item: (
        -item[f"summed_{duration_prefix}_duration_ns"],
        -item["interval_union_ns"],
        *(item[name] for name in key_names),
    ))


def _rank_active(rows: list[dict[str, Any]], key_names: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: (
        -item["interval_union_ns"],
        -item[next(name for name in item if name.startswith("summed_") and name.endswith("_ns"))],
        *(item[name] for name in key_names),
    ))


def _dispatch_timeline(intervals: list[tuple[int, int]], summed_duration: int) -> dict[str, Any]:
    if not intervals:
        return {
            "first_dispatch_start_ns": None, "last_dispatch_end_ns": None,
            "first_to_last_dispatch_span_ns": 0,
            "first_to_last_dispatch_span_ms": 0.0,
            "dispatch_interval_union_ns": 0, "dispatch_interval_union_ms": 0.0,
            "no_dispatch_interval_gap_ns": 0, "no_dispatch_interval_gap_ms": 0.0,
            "connected_dispatch_interval_segments": 0,
            "maximum_concurrent_dispatch_intervals": 0,
            "mean_concurrent_dispatch_intervals_while_present": None,
            "mean_concurrent_dispatch_intervals_over_span": None,
        }
    # Treat intervals as half-open: an end at T does not overlap a start at T.
    events: dict[int, int] = defaultdict(int)
    for begin, end in intervals:
        events[begin] += 1
        events[end] -= 1
    concurrency = maximum = segments = 0
    previous: int | None = None
    union = 0
    for timestamp, delta in sorted(events.items()):
        if previous is not None and timestamp > previous and concurrency > 0:
            union += timestamp - previous
        before = concurrency
        concurrency += delta
        if before == 0 and concurrency > 0:
            segments += 1
        maximum = max(maximum, concurrency)
        previous = timestamp
    if concurrency != 0 or union != _duration_ns(intervals):
        raise ValueError("kernel dispatch interval sweep is inconsistent")
    first = min(begin for begin, _ in intervals)
    last = max(end for _, end in intervals)
    span = last - first
    return {
        "first_dispatch_start_ns": first, "last_dispatch_end_ns": last,
        "first_to_last_dispatch_span_ns": span,
        "first_to_last_dispatch_span_ms": _ms(span),
        "dispatch_interval_union_ns": union,
        "dispatch_interval_union_ms": _ms(union),
        "no_dispatch_interval_gap_ns": span - union,
        "no_dispatch_interval_gap_ms": _ms(span - union),
        "connected_dispatch_interval_segments": segments,
        "maximum_concurrent_dispatch_intervals": maximum,
        "mean_concurrent_dispatch_intervals_while_present": summed_duration / union,
        "mean_concurrent_dispatch_intervals_over_span": summed_duration / span,
    }


def _load_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict)
            or report.get("artifact_type") != "ninfer_bench_report"
            or report.get("schema_version") != 20):
        raise ValueError("benchmark report must be ninfer_bench_report schema v20")
    config = report.get("config")
    tests = report.get("tests")
    if not isinstance(config, dict) or not isinstance(tests, list) or len(tests) != 1:
        raise ValueError("benchmark report must contain one configured test")
    test = tests[0]
    if not isinstance(test, dict):
        raise ValueError("benchmark report test must be an object")
    reps = test.get("reps")
    if (
        config.get("concurrency") != 1
        or config.get("spec") != "none"
        or config.get("draft_tokens") != 0
        or config.get("speculative_execution") is not False
        or config.get("dflash_verify_width") != 0
        or config.get("use_device_graph") is not True
        or config.get("decode_path") != "device_graph"
        or config.get("repetitions") != 1
        or test.get("kind") != "whole"
        or test.get("n_prompt") != 8192
        or test.get("n_gen") != 1
        or not isinstance(reps, list) or len(reps) != 1
        or not isinstance(reps[0], dict)
        or reps[0].get("decode_output_tokens") != 1
        or reps[0].get("decode_engine_tokens") != 1
    ):
        raise ValueError("benchmark report is not one ordinary C1 P8192 device-graph decode round")
    speculative = test.get("speculative")
    if not isinstance(speculative, dict) or speculative.get("enabled") is not False:
        raise ValueError("benchmark report does not explicitly disable speculative decoding")
    return config, test


def _find_database(directory: Path) -> Path:
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("ROCPD path must be a directory")
    candidates = sorted(path for path in root.rglob("*_results.db") if path.is_file())
    if len(candidates) != 1:
        raise ValueError("ROCPD directory must contain exactly one *_results.db database")
    return candidates[0].resolve()


def analyze(benchmark_report: Path, rocpd_directory: Path) -> dict[str, Any]:
    report_path = benchmark_report.resolve(strict=True)
    if not report_path.is_file():
        raise ValueError("benchmark report is not a regular file")
    config, test = _load_report(report_path)
    database = _find_database(rocpd_directory)

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        objects = {str(row[0]) for row in connection.execute(
            "select name from sqlite_master where type in ('table', 'view')")}
        required_views = {"regions", "kernels", "memory_copies", "graph_launches"}
        if not required_views <= objects:
            raise ValueError("ROCPD database lacks required trace views")
        _require_columns(connection, "regions", {
            "start", "end", "duration", "name", "category", "extdata"})
        _require_columns(connection, "kernels", {
            "start", "end", "duration", "name", "region", "dispatch_id",
            "graph_exec_id"})
        _require_columns(connection, "memory_copies", {
            "start", "end", "duration", "name", "region_name", "size",
            "graph_exec_id"})
        _require_columns(connection, "graph_launches", {
            "start", "end", "duration", "graph_exec_id", "kernel_dispatch_count"})

        regions = list(connection.execute(
            'select start, "end", duration, name, category, extdata '
            'from regions order by start, "end", name'))
        decoded_regions = []
        for row in regions:
            begin, end, _ = _interval(row, "region")
            decoded_regions.append((row, (begin, end), _message(row["extdata"])))
        measured = [interval for _, interval, message in decoded_regions
                    if message == MEASURED]
        ordinary = [interval for _, interval, message in decoded_regions
                    if message == ORDINARY]
        if len(measured) != 1:
            raise ValueError("trace must contain exactly one ninfer_bench_measured region")
        if len(ordinary) != 1:
            raise ValueError(f"trace must contain exactly one {ORDINARY} region")
        if not _contains(measured[0], *ordinary[0]):
            raise ValueError("ordinary decode region is not nested in the measured region")
        forbidden = sorted(message for _, _, message in decoded_regions
                           if any(marker in message for marker in FORBIDDEN_ROUNDS))
        if forbidden:
            raise ValueError("trace contains an MTP or DFlash decode-round marker")

        kernel_rows = list(connection.execute(
            'select dispatch_id, start, "end", duration, name, region, graph_exec_id '
            'from kernels where region = ? order by start, dispatch_id', (ORDINARY,)))
        copy_rows = list(connection.execute(
            'select start, "end", duration, name, region_name, size, graph_exec_id '
            'from memory_copies where region_name = ? order by start, id', (ORDINARY,)))
        runtime_rows = list(connection.execute(
            'select start, "end", duration, name, category from regions '
            'where category like ? order by start, "end", name',
            (f"{RUNTIME_CATEGORY}%",)))
        graph_rows = list(connection.execute(
            'select start, "end", duration, graph_exec_id, kernel_dispatch_count '
            'from graph_launches order by start, graph_exec_id'))
        kernel_associations = list(connection.execute(
            'select graph_exec_id, coalesce(region, "") as region from kernels '
            'where graph_exec_id != 0 order by dispatch_id'))
        copy_associations = list(connection.execute(
            'select graph_exec_id, coalesce(region_name, "") as region '
            'from memory_copies where graph_exec_id != 0 order by id'))
    finally:
        connection.close()

    round_interval = ordinary[0]
    contained_runtime = []
    for row in runtime_rows:
        begin, end, _ = _interval(row, "runtime")
        if _overlaps(round_interval, begin, end) and not _contains(round_interval, begin, end):
            raise ValueError("runtime record ambiguously crosses the ordinary decode boundary")
        if _contains(round_interval, begin, end):
            contained_runtime.append(row)
    contained_graphs = []
    for row in graph_rows:
        begin, end, _ = _interval(row, "graph launch")
        if _overlaps(round_interval, begin, end) and not _contains(round_interval, begin, end):
            raise ValueError("graph launch ambiguously crosses the ordinary decode boundary")
        if _contains(round_interval, begin, end):
            contained_graphs.append(row)
    if len(contained_graphs) != 1:
        raise ValueError("ordinary decode region must contain exactly one graph launch")
    graph = contained_graphs[0]
    graph_id = graph["graph_exec_id"]
    dispatch_count = graph["kernel_dispatch_count"]
    if (type(graph_id) is not int or graph_id <= 0 or type(dispatch_count) is not int
            or dispatch_count <= 0):
        raise ValueError("ordinary graph launch must have a positive ID and dispatch count")

    # The asynchronous records' nearest ROCTX association is authoritative.  Timing
    # overlap cannot pull an unmarked operation into this round.  A graph operation,
    # however, is ambiguous if its explicit association is absent or points elsewhere.
    graph_kernels = [row for row in kernel_rows if row["graph_exec_id"] == graph_id]
    if any(row["region"] != ORDINARY for row in [
            *[row for row in kernel_associations if row["graph_exec_id"] == graph_id],
            *[row for row in copy_associations if row["graph_exec_id"] == graph_id],
    ]):
        raise ValueError("ordinary graph contains an unmarked or ambiguously marked record")
    if len(graph_kernels) != dispatch_count:
        raise ValueError("graph dispatch count differs from exact ordinary-region kernels")
    all_selected_graph_records = [*kernel_rows, *copy_rows]
    for row in all_selected_graph_records:
        if row["graph_exec_id"] not in (0, graph_id):
            raise ValueError("ordinary-region record refers to a different graph launch")

    kernel_contributors = _aggregate_timed(kernel_rows, ("name",), "device")
    kernel_intervals = [(_interval(row, "kernel")[0], _interval(row, "kernel")[1])
                        for row in kernel_rows]
    kernel_sum = sum(_interval(row, "kernel")[2] for row in kernel_rows)
    kernel_timeline = _dispatch_timeline(kernel_intervals, kernel_sum)

    for row in copy_rows:
        _interval(row, "memory copy")
        if type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("memory-copy record has an invalid byte count")
    copy_contributors = _aggregate_timed(copy_rows, ("name",), "device")
    copy_intervals = [(_interval(row, "memory copy")[0],
                       _interval(row, "memory copy")[1]) for row in copy_rows]
    copy_sum = sum(_interval(row, "memory copy")[2] for row in copy_rows)
    copy_bytes: dict[str, int] = defaultdict(int)
    for row in copy_rows:
        copy_bytes[str(row["name"])] += row["size"]
    for item in copy_contributors:
        item["bytes"] = copy_bytes[item["name"]]

    runtime_contributors = _aggregate_timed(
        contained_runtime, ("category", "name"), "host")
    runtime_intervals = [(_interval(row, "runtime")[0], _interval(row, "runtime")[1])
                         for row in contained_runtime]
    runtime_sum = sum(_interval(row, "runtime")[2] for row in contained_runtime)
    graph_begin, graph_end, graph_duration = _interval(graph, "graph launch")
    measured_duration = measured[0][1] - measured[0][0]
    round_duration = round_interval[1] - round_interval[0]
    return {
        "artifact_type": "ninfer_ordinary_decode_trace_analysis",
        "schema_version": 1,
        "inputs": {
            "benchmark_report": str(report_path),
            "rocpd_directory": str(rocpd_directory.resolve()),
            "database": str(database),
        },
        "workload": {
            "kind": test["kind"], "prompt_tokens": 8192, "decode_rounds": 1,
            "concurrency": 1, "spec": "none", "decode_path": config["decode_path"],
        },
        "regions": {
            "measured_wall_duration_ns": measured_duration,
            "measured_wall_duration_ms": _ms(measured_duration),
            "ordinary_marker": ORDINARY,
            "ordinary_wall_duration_ns": round_duration,
            "ordinary_wall_duration_ms": _ms(round_duration),
        },
        "association": {
            "kernel_and_copy": "exact nearest-ROCTX region association",
            "runtime_and_graph": "strict full containment in the unique ordinary host region",
            "temporal_overlap_is_not_kernel_or_copy_association": True,
        },
        "kernels": {
            "dispatch_count": len(kernel_rows),
            "graph_dispatch_count": len(graph_kernels),
            "summed_device_duration_ns": kernel_sum,
            "summed_device_duration_ms": _ms(kernel_sum),
            "timeline": kernel_timeline,
            "contributors_by_summed_device_duration": kernel_contributors,
            "contributors_by_dispatch_interval_union": _rank_active(
                kernel_contributors, ("name",)),
        },
        "memory_copies": {
            "record_count": len(copy_rows),
            "bytes": sum(row["size"] for row in copy_rows),
            "summed_device_duration_ns": copy_sum,
            "summed_device_duration_ms": _ms(copy_sum),
            "device_interval_union_ns": _duration_ns(copy_intervals),
            "device_interval_union_ms": _ms(_duration_ns(copy_intervals)),
            "contributors": copy_contributors,
        },
        "runtime_api": {
            "record_count": len(contained_runtime),
            "summed_host_duration_ns": runtime_sum,
            "summed_host_duration_ms": _ms(runtime_sum),
            "host_interval_union_ns": _duration_ns(runtime_intervals),
            "host_interval_union_ms": _ms(_duration_ns(runtime_intervals)),
            "contributors": runtime_contributors,
        },
        "graph_launch": {
            "graph_exec_id": graph_id,
            "reported_kernel_dispatch_count": dispatch_count,
            "start_ns": graph_begin, "end_ns": graph_end,
            "host_duration_ns": graph_duration,
            "host_duration_ms": _ms(graph_duration),
        },
        "duration_semantics": {
            "summed_device_duration": "independent dispatch/copy service; intervals may overlap",
            "dispatch_interval_union": (
                "union of rocprofiler kernel-dispatch start/end intervals; it is not a GPU "
                "utilization or CU-occupancy measurement"),
            "dispatch_interval_concurrency": (
                "overlap among recorded graph-node dispatch intervals; shared queue/stream "
                "metadata identifies enqueue ownership and does not make this a serial timeline"),
            "no_dispatch_interval_gap": (
                "gaps within the first-to-last dispatch span containing no recorded kernel "
                "interval; not proof that every GPU engine was idle"),
            "wall_duration": "host ROCTX region extent only",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument("--rocpd", required=True, type=Path,
                        help="directory containing exactly one *_results.db")
    parser.add_argument("--out", type=Path,
                        help="create this JSON file instead of writing JSON to stdout")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.benchmark_report, args.rocpd)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out is None:
            print(encoded, end="")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as destination:
                destination.write(encoded)
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
