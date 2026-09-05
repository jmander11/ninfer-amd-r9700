#!/usr/bin/env python3
"""Analyze one selected-region rocprof trace into prefill attribution evidence."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MEASURED = "ninfer_bench_measured"
TEXT_CHUNK = "ninfer.prefill.prefill.chunk"
MTP_CHUNK = "ninfer.mtp.prefill.mtp_chunk"
PAYLOAD = re.compile(r" payload=(\d+)$")


def _message(extdata: str) -> str:
    try:
        value = json.loads(extdata)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("region extdata is not valid JSON") from error
    message = value.get("message") if isinstance(value, dict) else None
    return message if isinstance(message, str) else ""


def _duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    begin, end = ordered[0]
    if end < begin:
        raise ValueError("trace contains a negative interval")
    for next_begin, next_end in ordered[1:]:
        if next_end < next_begin:
            raise ValueError("trace contains a negative interval")
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def _contains(interval: tuple[int, int], begin: int, end: int) -> bool:
    return interval[0] <= begin and end <= interval[1]


def _overlaps(interval: tuple[int, int], begin: int, end: int) -> bool:
    return max(interval[0], begin) < min(interval[1], end)


def _category(region: str, begin: int, end: int, measured: tuple[int, int],
              text: list[tuple[int, int]]) -> str:
    # rocprof assigns the nearest active ROCTX range to each asynchronous GPU
    # operation. That association, rather than timestamp containment in a host
    # range, is the execution-stage authority. In particular, an operation
    # enqueued inside a selected range can finish shortly after the host pops
    # that range. Keep rejecting out-of-range work when rocprof provides no
    # selected-range association.
    if region.startswith("ninfer.mtp.prefill."):
        return "mtp_prefill"
    if region.startswith(("ninfer.attention.prefill.", "ninfer.gdn.prefill.",
                          "ninfer.post-mixer.prefill.")):
        return "base_text_prefill"
    if region.startswith("ninfer.prefill.prefill.chunk"):
        return "prefill_orchestration"
    if not _contains(measured, begin, end):
        raise ValueError("selected-region activity lies outside the measured range")
    if any(_overlaps(interval, begin, end) for interval in text):
        return "prefill_unattributed"
    return "measured_other"


def _intersections(begin: int, end: int,
                   intervals: Iterable[tuple[int, int]]) -> Iterable[tuple[int, int]]:
    for parent_begin, parent_end in intervals:
        clipped_begin, clipped_end = max(begin, parent_begin), min(end, parent_end)
        if clipped_begin < clipped_end:
            yield clipped_begin, clipped_end


def _marker_family(region: str) -> str:
    if region.startswith("ninfer.mtp.prefill."):
        return "mtp"
    if region.startswith("ninfer.attention."):
        return "attention"
    if region.startswith("ninfer.gdn."):
        return "gdn"
    if region.startswith("ninfer.post-mixer."):
        return "post_mixer"
    if region.startswith("ninfer.prefill.prefill.chunk"):
        return "prefill_orchestration"
    return "unmarked"


def _symbol_family(name: str) -> str:
    rules = (
        ("xattention_flash_consumer_kernel", "xattention_consumer"),
        ("xattention_consumer_kernel", "xattention_consumer"),
        ("xattention_rank_kernel", "xattention_rank"),
        ("xattention_pack_keys_kernel", "xattention_key_pack"),
        ("a8q4g64_linear_prefill_cta_kernel", "a8q4_prefill_cta"),
        ("a8w8g32_linear_prefill_cta_kernel", "a8w8_prefill_cta"),
        ("a8q4g64_linear_wmma32_kernel", "a8q4_matrix"),
        ("a8w8g32_linear_wmma32_kernel", "a8w8_matrix"),
        ("a8g64_quantize_activation_kernel", "a8_activation_quantize"),
        ("a8g32_quantize_activation_kernel", "a8_activation_quantize"),
        ("fused_silu_a8g64_quantize_kernel", "fused_silu_a8q4_prepare"),
        ("fused_attention_causal_kernel", "dense_attention"),
        ("residual_rmsnorm_kernel", "rmsnorm"),
        ("gated_rmsnorm_kernel", "rmsnorm"),
        ("rmsnorm_kernel", "rmsnorm"),
        ("silu_mul_strided_kernel", "silu_mul"),
        ("silu_mul_kernel", "silu_mul"),
        ("residual_add_kernel", "residual_add"),
        ("gdn", "gdn"),
    )
    for needle, family in rules:
        if needle in name:
            return family
    return "other"


def _operator_family(name: str, marker_family: str) -> str:
    """Add stage-analysis detail without changing the legacy symbol-family buckets."""

    # These kernels are defined in an anonymous namespace. rocprof's demangled
    # name therefore loses the gdn_recurrence source namespace, and the same
    # generic spelling is not sufficient evidence by itself.
    if marker_family == "gdn" and any(
        needle in name
        for needle in ("ordinary_kernel<", "snapshot_kernel<", "record_kernel<",
                       "recurrent_kernel(")
    ):
        return "gdn_recurrence"
    rules = (
        ("dense_full_score_qk", "dense_attention"),
        ("dense_full_score_maximum", "dense_attention"),
        ("dense_full_score_pv", "dense_attention"),
        ("a8g64_quantize_activation_kernel", "a8q4_activation_quantize"),
        ("a8g32_quantize_activation_kernel", "a8w8_activation_quantize"),
        ("fused_silu_a8g64_quantize_kernel", "fused_silu_a8q4_prepare"),
        ("gdn_recurrence", "gdn_recurrence"),
        ("causal_conv1d_silu_kernel", "gdn_causal_conv"),
        ("projection_conv_snapshot_kernel", "gdn_projection_conv"),
        ("projection_conv_record_kernel", "gdn_projection_conv"),
        ("control_gates_kernel", "gdn_control"),
    )
    for needle, family in rules:
        if needle in name:
            return family
    return _symbol_family(name)


def _ms(nanoseconds: int) -> float:
    return nanoseconds / 1e6


def analyze(database: Path, benchmark_report: Path) -> dict[str, Any]:
    report = json.loads(benchmark_report.read_text(encoding="utf-8"))
    if report.get("artifact_type") != "ninfer_bench_report" or report.get("schema_version") != 20:
        raise ValueError("benchmark report must be ninfer_bench_report schema v20")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1 or not isinstance(tests[0], dict):
        raise ValueError("profile benchmark report must contain exactly one measured test")
    test = tests[0]
    if test.get("kind") not in ("pp", "whole"):
        raise ValueError("profile benchmark test must be prefill-only or whole inference")
    prefill_seconds = test.get("prefill_seconds_mean")
    if not isinstance(prefill_seconds, (int, float)) or prefill_seconds <= 0:
        raise ValueError("benchmark report lacks a positive prefill duration")
    prompt_tokens = test.get("n_prompt")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ValueError("benchmark report lacks a positive prompt-token count")
    concurrency = report.get("config", {}).get("concurrency", 1)
    if not isinstance(concurrency, int) or concurrency not in range(1, 5):
        raise ValueError("benchmark report concurrency must be in [1, 4]")

    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in connection.execute(
            "select name from sqlite_master where type in ('table','view')")}
        required = {"regions", "kernels", "memory_copies"}
        if not required <= tables:
            raise ValueError("rocprof database lacks regions, kernels, or memory_copies views")
        region_rows = list(connection.execute(
            'select start, "end", extdata from regions order by start'))
        kernel_rows = list(connection.execute(
            'select start, "end", duration, name, coalesce(region, "") region '
            'from kernels order by start'))
        copy_rows = list(connection.execute(
            'select start, "end", duration, name, coalesce(region_name, "") region_name, size '
            'from memory_copies order by start'))
    finally:
        connection.close()

    regions = [((int(row["start"]), int(row["end"])), _message(row["extdata"]))
               for row in region_rows]
    measured_ranges = [interval for interval, message in regions if message == MEASURED]
    if len(measured_ranges) != 1:
        raise ValueError("trace must contain exactly one ninfer_bench_measured range")
    measured = measured_ranges[0]
    text = [interval for interval, message in regions if message.startswith(TEXT_CHUNK)]
    mtp = [interval for interval, message in regions if message.startswith(MTP_CHUNK)]
    if not text or _duration_ns(text) != sum(end - begin for begin, end in text):
        raise ValueError("Text prefill chunk ranges are absent or overlapping")
    if not all(_contains(measured, begin, end) for begin, end in text):
        raise ValueError("Text prefill chunk lies outside measured range")
    if not all(any(_contains(parent, begin, end) for parent in text) for begin, end in mtp):
        raise ValueError("MTP prefill range is not nested in a Text prefill chunk")
    chunk_messages = [message for _, message in regions if message.startswith(TEXT_CHUNK)]
    payloads = []
    for message in chunk_messages:
        match = PAYLOAD.search(message)
        if match is None:
            raise ValueError("Text prefill chunk lacks an integer payload")
        payloads.append(int(match.group(1)))
    if sum(payloads) != prompt_tokens * concurrency:
        raise ValueError("Text prefill payload sum does not match benchmark prompt tokens")
    text_wall_ns = _duration_ns(text)
    tolerance_ns = max(50_000_000, int(prefill_seconds * 1e9 * 0.001))
    if abs(text_wall_ns - int(prefill_seconds * 1e9)) > tolerance_ns:
        raise ValueError("Text prefill ranges do not match benchmark prefill duration")

    category_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    category_sum: dict[str, int] = defaultdict(int)
    category_calls: dict[str, int] = defaultdict(int)
    symbols: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    marker_families: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    symbol_families: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    operator_attribution: dict[tuple[str, str, str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for row in kernel_rows:
        begin, end = int(row["start"]), int(row["end"])
        duration = int(row["duration"])
        if duration != end - begin:
            raise ValueError("kernel duration does not match its interval")
        name, region = str(row["name"]), str(row["region"])
        category = _category(region, begin, end, measured, text)
        category_intervals[category].append((begin, end))
        category_sum[category] += duration
        category_calls[category] += 1
        marker_family = _marker_family(region)
        symbol_family = _symbol_family(name)
        for mapping, key in (
            (symbols, name), (marker_families, marker_family),
            (symbol_families, symbol_family),
        ):
            mapping[(category, key)][0] += 1
            mapping[(category, key)][1] += duration
        # Symbols identify implementation families, but only rocprof's nearest-range
        # association identifies their model stage. In particular, linear, quantize,
        # normalization, activation, and residual kernels occur in multiple stages.
        stage = marker_family if marker_family != "unmarked" else "ambiguous"
        operator_family = _operator_family(name, marker_family)
        operator_attribution[(category, stage, marker_family, operator_family)][0] += 1
        operator_attribution[(category, stage, marker_family, operator_family)][1] += duration

    def aggregate(mapping: dict[tuple[str, str], list[int]], label: str) -> list[dict[str, Any]]:
        return [
            {"execution_category": category, label: key, "calls": values[0],
             "summed_duration_ms": _ms(values[1])}
            for (category, key), values in sorted(
                mapping.items(), key=lambda item: (-item[1][1], item[0]))
        ]

    def top_per_category(mapping: dict[tuple[str, str], list[int]], label: str,
                         limit: int) -> list[dict[str, Any]]:
        result = []
        counts: dict[str, int] = defaultdict(int)
        for row in aggregate(mapping, label):
            category = row["execution_category"]
            if counts[category] < limit:
                result.append(row)
                counts[category] += 1
        return result

    copy_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    copy_kinds: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    copy_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in copy_rows:
        begin, end = int(row["start"]), int(row["end"])
        duration = int(row["duration"])
        if duration != end - begin:
            raise ValueError("memory-copy duration does not match its interval")
        category = _category(str(row["region_name"]), begin, end, measured, text)
        copy_stats[category][0] += 1
        copy_stats[category][1] += int(row["size"])
        copy_stats[category][2] += duration
        kind = str(row["name"])
        copy_kinds[(category, kind)][0] += 1
        copy_kinds[(category, kind)][1] += int(row["size"])
        copy_kinds[(category, kind)][2] += duration
        copy_intervals[category].append((begin, end))

    # The inactive gap is temporal, unlike stage attribution: include every
    # kernel overlapping Text and clip it to the Text wall interval.
    prefill_kernel_intervals = [
        clipped
        for row in kernel_rows
        for clipped in _intersections(int(row["start"]), int(row["end"]), text)
    ]
    kernel_active_ns = _duration_ns(prefill_kernel_intervals)
    unattributed_kernel_calls = category_calls.get("prefill_unattributed", 0)
    unattributed_kernel_duration = category_sum.get("prefill_unattributed", 0)
    unattributed_copy_calls = copy_stats.get("prefill_unattributed", [0])[0]
    ambiguous_operator_calls = sum(
        values[0]
        for (category, stage, _marker, _symbol), values in operator_attribution.items()
        if stage == "ambiguous" and category != "measured_other"
    )
    ambiguous_operator_duration = sum(
        values[1]
        for (category, stage, _marker, _symbol), values in operator_attribution.items()
        if stage == "ambiguous" and category != "measured_other"
    )
    return {
        "schema": "ninfer_selected_region_attribution",
        "schema_version": 1,
        "database": str(database.resolve()),
        "benchmark_report": str(benchmark_report.resolve()),
        "workload": {
            "kind": test["kind"], "prompt_tokens": prompt_tokens,
            "generated_tokens": test.get("n_gen"),
            "concurrency": concurrency,
            "prefill_chunk": report.get("config", {}).get("prefill_chunk"),
        },
        "ranges": {
            "measured_wall_ms": _ms(measured[1] - measured[0]),
            "text_prefill_chunks": len(text),
            "text_prefill_wall_ms": _ms(text_wall_ns),
            "mtp_prefill_ranges": len(mtp),
            "mtp_prefill_wall_ms": _ms(_duration_ns(mtp)),
        },
        "kernel_execution_categories": [
            {"category": category, "calls": category_calls[category],
             "independent_summed_duration_ms": _ms(category_sum[category]),
             "active_union_ms": _ms(_duration_ns(category_intervals[category]))}
            for category in sorted(category_calls)
        ],
        "prefill_stage_attribution": {
            "kernel_complete": unattributed_kernel_calls == 0,
            "unattributed_kernel_calls": unattributed_kernel_calls,
            "unattributed_kernel_summed_duration_ms": _ms(unattributed_kernel_duration),
            "memory_copy_complete": unattributed_copy_calls == 0,
            "unattributed_memory_copy_calls": unattributed_copy_calls,
            "basis": "rocprof kernel/copy region association, not temporal containment",
        },
        "prefill_no_kernel_wall_ms": _ms(text_wall_ns - kernel_active_ns),
        "prefill_kernel_active_union_ms": _ms(kernel_active_ns),
        "prefill_kernel_active_wall_fraction": kernel_active_ns / text_wall_ns,
        "prefill_no_kernel_wall_interpretation": (
            "kernel-inactive wall gap only; marker/kernel/copy trace cannot separate host work "
            "from true GPU idle"),
        "memory_copy_categories": [
            {"category": category, "calls": values[0], "bytes": values[1],
             "independent_summed_duration_ms": _ms(values[2]),
             "active_union_ms": _ms(_duration_ns(copy_intervals[category]))}
            for category, values in sorted(copy_stats.items())
        ],
        "memory_copy_kinds": [
            {"execution_category": category, "name": name, "calls": values[0],
             "bytes": values[1], "independent_summed_duration_ms": _ms(values[2])}
            for (category, name), values in sorted(
                copy_kinds.items(), key=lambda item: (-item[1][2], item[0]))
        ],
        "symbol_families": aggregate(symbol_families, "family"),
        "marker_families": aggregate(marker_families, "family"),
        "operator_attribution": [
            {
                "execution_category": category,
                "stage": stage,
                "marker_family": marker_family,
                "operator_family": operator_family,
                "calls": values[0],
                "summed_duration_ms": _ms(values[1]),
                "ambiguous": stage == "ambiguous",
            }
            for (category, stage, marker_family, operator_family), values in sorted(
                operator_attribution.items(), key=lambda item: (-item[1][1], item[0])
            )
        ],
        "operator_stage_attribution": {
            "complete": ambiguous_operator_calls == 0,
            "ambiguous_kernel_calls": ambiguous_operator_calls,
            "ambiguous_kernel_summed_duration_ms": _ms(ambiguous_operator_duration),
            "basis": (
                "rocprof nearest-range marker association; symbol names classify operators "
                "but never infer an absent stage marker"
            ),
        },
        "top_kernels": top_per_category(symbols, "name", 20),
        "duration_note": (
            "independent sums may overlap across streams; active_union is wall-time union"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise SystemExit(f"output already exists: {args.out_json}")
    try:
        result = analyze(args.database, args.benchmark_report)
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote selected-region attribution to {args.out_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
