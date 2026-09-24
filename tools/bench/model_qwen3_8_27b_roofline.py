#!/usr/bin/env python3
"""Build offline Qwen3.8-27B selected-route logical roofline evidence.

This tool deliberately does not infer model roles or tensor shapes from kernel
symbols.  It consumes one atomically published reconciliation of a validated
trace and an independently prepared static schedule.  The resulting byte rates describe
represented tensors or source-level requests, never physical cache/HBM traffic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from tools.r9700.a8q4_prefill_traffic import (
    M64_N64,
    W8_M64_N64,
    Shape,
    requested_traffic,
    requested_w8_traffic,
)


TIMING_TYPE = "ninfer_qwen3_8_27b_dispatch_timing"
INVENTORY_TYPE = "ninfer_qwen3_8_27b_dispatch_inventory"
RECONCILIATION_TYPE = "ninfer_qwen3_8_27b_dispatch_reconciliation"
TRACE_TYPE = "ninfer_r9700_selected_profile_trace"
LOW_CONTEXT_TYPE = "ninfer_r9700_low_context_prefill_evaluation"
OUTPUT_TYPE = "ninfer_qwen3_8_27b_roofline"
SCHEMA_VERSION = 1

THEORETICAL_PEAKS = {
    "advertised_memory_gbps": 640.0,
    "fp32_vector_tflops": 47.8,
    "bf16_matrix_tflops": 191.0,
    "int8_matrix_tops": 383.0,
    "int4_matrix_tops": 766.0,
}

FORMATS = {
    "q4_linear": "a8q4g64_bf16_output",
    "w8_linear": "a8w8g32_bf16_output",
    "q4_quantize": "bf16_to_a8g64",
    "w8_quantize": "bf16_to_a8g32",
    "dense_attention": "bf16q_fp8k_int4v_fp16scale_fp32_output",
    "sparse_pack": "fp8k_to_bf16_packed_k",
    "sparse_rank": "bf16q_bf16packedk_fp32_rank",
    "sparse_consumer": "bf16q_bf16packedk_int4v_fp16scale_fp32_output",
    "gdn_recurrence": "bf16_io_fp32_state",
    "gdn_control": "bf16_input_fp32_output",
    "gdn_conv": "bf16_io",
    "gated_rmsnorm": "bf16_io",
    "residual_add": "bf16_io",
    "silu_mul": "bf16_io",
    "rmsnorm": "bf16_io",
    "residual_rmsnorm": "bf16_io",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verified_snapshot(value: Any, name: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a file snapshot")
    path = Path(_text(value.get("path"), f"{name}.path")).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} is not a regular file")
    size = _integer(value.get("file_size_bytes"), f"{name}.file_size_bytes")
    digest = _sha_text(value.get("sha256"), f"{name}.sha256")
    actual = {"path": str(path), "file_size_bytes": path.stat().st_size,
              "sha256": _sha256(path)}
    if value.get("path") != actual["path"] or size != actual["file_size_bytes"] or digest != actual["sha256"]:
        raise ValueError(f"{name} path, size, or SHA-256 changed")
    return path, actual


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _schema(document: dict[str, Any], artifact_type: str, name: str) -> None:
    if (document.get("artifact_type") != artifact_type
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1):
        raise ValueError(f"{name} must be {artifact_type} schema v1")


def _sha_text(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _validate_workload(workload: dict[str, Any]) -> None:
    if workload.get("kind") not in ("pp", "whole"):
        raise ValueError("workload.kind must be pp or whole")
    _integer(workload.get("prompt_tokens"), "workload.prompt_tokens", minimum=1)
    concurrency = _integer(workload.get("concurrency"), "workload.concurrency", minimum=1)
    if concurrency > 4:
        raise ValueError("workload.concurrency must be in [1,4]")
    _integer(workload.get("prefill_chunk"), "workload.prefill_chunk", minimum=1)
    group = _integer(workload.get("kv_value_group"), "workload.kv_value_group", minimum=1)
    if group not in (16, 32):
        raise ValueError("workload.kv_value_group must be 16 or 32")
    profile = _text(workload.get("xattention_profile"), "workload.xattention_profile")
    if profile not in ("dense", "b128-s16-tau900"):
        raise ValueError("workload.xattention_profile is not a production selected-route profile")
    for name in ("artifact_path", "weights_id", "executable_path"):
        _text(workload.get(name), f"workload.{name}")
    for name in ("artifact_sha256", "executable_sha256"):
        _sha_text(workload.get(name), f"workload.{name}")


def _unique_rows(document: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    rows = document.get("dispatches")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name}.dispatches must be a nonempty array")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name}.dispatches[{index}] must be an object")
        dispatch_id = _text(row.get("dispatch_id"), f"{name}.dispatches[{index}].dispatch_id")
        if dispatch_id in result:
            raise ValueError(f"duplicate {name} dispatch_id: {dispatch_id}")
        result[dispatch_id] = row
    return result


def _wall_union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    begin, end = intervals[0]
    for next_begin, next_end in intervals[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def _base_counts() -> dict[str, Any]:
    return {
        "logical_ops": 0,
        "issued_ops": {},
        "represented_minimum_bytes": 0,
        "source_requested_bytes": None,
        "special_functions": {},
        "accounting_note": "",
    }


def _positive_parameters(params: dict[str, Any], *names: str) -> list[int]:
    return [_integer(params.get(name), name, minimum=1) for name in names]


def _linear(params: dict[str, Any], *, q4: bool) -> dict[str, Any]:
    tokens, rows, columns = _positive_parameters(params, "tokens", "rows", "columns")
    shape = Shape(tokens, rows, columns)
    traffic = (requested_traffic(shape, M64_N64) if q4
               else requested_w8_traffic(shape, W8_M64_N64))
    result = _base_counts()
    result.update({
        "logical_ops": traffic.logical_ops,
        "issued_ops": {"iu4": traffic.issued_iu4_ops} if q4
                      else {"iu8": traffic.issued_iu8_ops},
        "represented_minimum_bytes": traffic.unique_tensor_bytes,
        "source_requested_bytes": traffic.total_requested_bytes,
        "accounting_note": (
            "CTA M64xN64 source-request model; represented bytes are unique consumer tensors"
        ),
    })
    return result


def _quantize(params: dict[str, Any], *, group: int) -> dict[str, Any]:
    tokens, columns = _positive_parameters(params, "tokens", "columns")
    alignment = 64 if group == 64 else 128
    padded = math.ceil(columns / alignment) * alignment
    codes = tokens * padded
    scales = 2 * tokens * (padded // group)
    result = _base_counts()
    result.update({
        "represented_minimum_bytes": 2 * tokens * columns + codes + scales + 4,
        "source_requested_bytes": 2 * tokens * columns + codes + scales + 4,
        "accounting_note": (
            f"BF16-to-A8G{group} conversion; no artificial FLOP/TOPS convention"
        ),
    })
    return result


def _dense_attention(params: dict[str, Any]) -> dict[str, Any]:
    tokens, context_start, q_heads, kv_heads, dimension, value_group = [
        _integer(params.get(name), name, minimum=(0 if name == "context_start" else 1))
        for name in ("tokens", "context_start", "query_heads", "kv_heads", "head_dimension",
                     "value_group")
    ]
    if q_heads % kv_heads or dimension % value_group:
        raise ValueError("dense attention requires integral GQA and value-scale groups")
    pairs = tokens * context_start + tokens * (tokens + 1) // 2
    context = context_start + tokens
    represented_metadata = _integer(params.get("represented_metadata_bytes"),
                                    "represented_metadata_bytes")
    source_metadata = _integer(params.get("source_metadata_bytes"),
                               "source_metadata_bytes")
    logical_ops = 4 * q_heads * dimension * pairs
    represented = (
        2 * tokens * q_heads * dimension
        + context * kv_heads * dimension
        + context * kv_heads * dimension // 2
        + 2 * context * kv_heads * (dimension // value_group)
        + 4 * tokens * q_heads * dimension
        + represented_metadata
    )
    # The scalar streaming kernel revisits K/V for every query-head/key pair.
    # Each of its D feature threads dereferences one FP8 key byte, one packed
    # INT4 value byte (two adjacent threads repeat the represented byte), and
    # one FP16 group scale (Group adjacent threads repeat it). This is the
    # source-expression request model, before cache/coalescing effects. Q and
    # output remain one represented traversal here.
    requested = (
        2 * tokens * q_heads * dimension
        + q_heads * pairs * (dimension + dimension + 2 * dimension)
        + 4 * tokens * q_heads * dimension
        + source_metadata
    )
    result = _base_counts()
    result.update({
        "logical_ops": logical_ops,
        "issued_ops": {"fp32_vector_arithmetic": logical_ops},
        "represented_minimum_bytes": represented,
        "source_requested_bytes": requested,
        "special_functions": {"softmax_exp_arguments": q_heads * pairs},
        "accounting_note": (
            "causal QK+PV arithmetic only; softmax/reduction/synchronization excluded from FLOPs"
        ),
    })
    return result


def _sparse_pack(params: dict[str, Any]) -> dict[str, Any]:
    context, kv_heads, dimension = _positive_parameters(
        params, "context_tokens", "kv_heads", "head_dimension")
    represented_metadata = _integer(params.get("represented_metadata_bytes"),
                                    "represented_metadata_bytes")
    source_metadata = _integer(params.get("source_metadata_bytes"),
                               "source_metadata_bytes")
    payload = context * kv_heads * dimension
    result = _base_counts()
    result.update({
        "represented_minimum_bytes": payload + 2 * payload + represented_metadata,
        "source_requested_bytes": payload + 2 * payload + source_metadata,
        "accounting_note": "FP8-key read plus BF16 packed-key write; conversion is not FLOPs",
    })
    return result


def _sparse_rank(params: dict[str, Any]) -> dict[str, Any]:
    dimension, logical_pairs, wmma_tiles = _positive_parameters(
        params, "head_dimension", "logical_qk_pairs", "issued_wmma_16x16_tiles")
    represented_query, represented_keys, represented_scratch = [
        _integer(params.get(name), name)
        for name in ("represented_query_elements", "represented_packed_key_elements",
                     "represented_scratch_bytes")
    ]
    source_query, source_keys, source_scratch_read, source_scratch_write = [
        _integer(params.get(name), name)
        for name in ("source_query_elements", "source_packed_key_elements",
                     "source_scratch_read_bytes", "source_scratch_write_bytes")
    ]
    represented_metadata = _integer(params.get("represented_metadata_bytes"),
                                    "represented_metadata_bytes")
    source_metadata = _integer(params.get("source_metadata_bytes"),
                               "source_metadata_bytes")
    represented = (2 * represented_query + 2 * represented_keys + represented_scratch
                   + represented_metadata)
    requested = (2 * source_query + 2 * source_keys + source_scratch_read
                 + source_scratch_write + source_metadata)
    result = _base_counts()
    result.update({
        "logical_ops": 2 * dimension * logical_pairs,
        "issued_ops": {"bf16_matrix": 2 * 16 * 16 * dimension * wmma_tiles},
        "represented_minimum_bytes": represented,
        "source_requested_bytes": requested,
        "special_functions": {
            "estimator_exp_arguments": _integer(
                params.get("estimator_exp_arguments", 0), "estimator_exp_arguments")
        },
        "accounting_note": (
            "rank QK arithmetic and padded WMMA tiles only; sorting/control are not FLOPs"
        ),
    })
    return result


def _keep_record(keep_rows: dict[str, dict[str, Any]], dispatch_id: str,
                 tokens: int, q_heads: int) -> tuple[int, int, list[tuple[int, int]]]:
    row = keep_rows.get(dispatch_id)
    if row is None:
        raise ValueError(f"missing sparse keep-count record for {dispatch_id}")
    counts = row.get("retained_key_counts_per_query")
    tiles = row.get("consumer_tiles")
    if (not isinstance(counts, list) or len(counts) != tokens * q_heads
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                       for item in counts)):
        raise ValueError(
            f"sparse keep counts for {dispatch_id} must contain tokens*query_heads integers")
    tile_geometry: list[tuple[int, int]] = []
    if not isinstance(tiles, list) or not tiles:
        raise ValueError(f"sparse consumer tiles for {dispatch_id} must be a nonempty array")
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise ValueError(f"sparse consumer tile {dispatch_id}[{index}] must be an object")
        query_rows = _integer(tile.get("query_rows"), "query_rows", minimum=1)
        key_rows = _integer(tile.get("key_rows"), "key_rows", minimum=1)
        if query_rows > 16 or key_rows > 16:
            raise ValueError(f"sparse consumer tile {dispatch_id}[{index}] exceeds B16")
        tile_geometry.append((query_rows, key_rows))
    unique = _integer(row.get("represented_unique_kv_token_heads"),
                      "represented_unique_kv_token_heads", minimum=1)
    return sum(counts), unique, tile_geometry


def _sparse_consumer(params: dict[str, Any], dispatch_id: str,
                     keep_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tokens, q_heads, kv_heads, dimension, value_group = _positive_parameters(
        params, "tokens", "query_heads", "kv_heads", "head_dimension", "value_group")
    if q_heads % kv_heads or dimension % value_group:
        raise ValueError("sparse consumer requires integral GQA and value-scale groups")
    retained_pairs, unique_kv, tiles = _keep_record(keep_rows, dispatch_id, tokens, q_heads)
    represented_metadata = _integer(params.get("represented_metadata_bytes"),
                                    "represented_metadata_bytes")
    source_metadata = _integer(params.get("source_metadata_bytes"),
                               "source_metadata_bytes")
    q_bytes = 2 * tokens * q_heads * dimension
    key_bytes = 2 * unique_kv * dimension
    value_bytes = unique_kv * dimension // 2
    scale_bytes = 2 * unique_kv * (dimension // value_group)
    output_bytes = 4 * tokens * q_heads * dimension
    represented = (q_bytes + key_bytes + value_bytes + scale_bytes + output_bytes
                   + represented_metadata)
    # Wave zero reloads each valid Q row and packed-K row per B16 tile. All D
    # feature threads consume V and its scale once per key row and reuse it over
    # the tile's query rows. Repeated packed-byte/scale source expressions are
    # retained here; cache/coalescing effects are intentionally not inferred.
    query_source = sum(2 * query_rows * dimension for query_rows, _ in tiles)
    key_source = sum(2 * key_rows * dimension for _, key_rows in tiles)
    value_source = sum(key_rows * dimension for _, key_rows in tiles)
    scale_source = sum(2 * key_rows * dimension for _, key_rows in tiles)
    requested = (query_source + key_source + value_source + scale_source + output_bytes
                 + source_metadata)
    qk_ops = 2 * dimension * retained_pairs
    pv_ops = 2 * dimension * retained_pairs
    result = _base_counts()
    result.update({
        "logical_ops": qk_ops + pv_ops,
        "issued_ops": {
            "bf16_matrix": 2 * 16 * 16 * dimension * len(tiles),
            "fp32_vector_arithmetic": pv_ops,
        },
        "represented_minimum_bytes": represented,
        "source_requested_bytes": requested,
        "special_functions": {"softmax_exp_arguments": retained_pairs},
        "accounting_note": (
            "keep-sidecar-derived sparse QK+PV; WMMA includes every executed B16 key tile"
        ),
    })
    return result


def _gdn_recurrence(params: dict[str, Any]) -> dict[str, Any]:
    tokens, q_heads, value_heads, dimension, row_tiles = _positive_parameters(
        params, "tokens", "query_heads", "value_heads", "dimension", "row_tiles")
    if value_heads % q_heads:
        raise ValueError("GDN recurrence requires integral query/value head grouping")
    logical_ops = tokens * value_heads * dimension * (7 * dimension + 4)
    represented = (
        4 * tokens * q_heads * dimension
        + 2 * tokens * value_heads * dimension
        + 8 * tokens * value_heads
        + 8 * value_heads * dimension * dimension
        + 2 * tokens * value_heads * dimension
    )
    requested = (
        4 * tokens * value_heads * row_tiles * dimension
        + 2 * tokens * value_heads * dimension
        + 8 * tokens * value_heads * row_tiles
        + 8 * value_heads * dimension * dimension
        + 2 * tokens * value_heads * dimension
    )
    result = _base_counts()
    result.update({
        "logical_ops": logical_ops,
        "issued_ops": {"fp32_vector_arithmetic": logical_ops},
        "represented_minimum_bytes": represented,
        "source_requested_bytes": requested,
        "special_functions": {
            "semantic_exp_arguments": tokens * value_heads,
            "issued_exp_arguments": tokens * value_heads * row_tiles,
        },
        "accounting_note": (
            "state transition arithmetic only; normalization reductions/transcendentals excluded"
        ),
    })
    return result


def _gdn_control(params: dict[str, Any]) -> dict[str, Any]:
    tokens, value_heads = _positive_parameters(params, "tokens", "value_heads")
    count = tokens * value_heads
    nonlinear = _integer(params.get("softplus_nonlinear_elements"),
                         "softplus_nonlinear_elements")
    if nonlinear > count:
        raise ValueError("softplus_nonlinear_elements exceeds tokens*value_heads")
    represented = 4 * count + 8 * count + 8 * value_heads
    result = _base_counts()
    result.update({
        "represented_minimum_bytes": represented,
        "source_requested_bytes": 20 * count,
        "special_functions": {
            "exp_arguments": 2 * count + nonlinear,
            "log1p_arguments": nonlinear,
        },
        "accounting_note": "GDN control reports bytes and SFU arguments, not artificial FLOPs",
    })
    return result


def _gdn_conv(params: dict[str, Any]) -> dict[str, Any]:
    tokens, channels = _positive_parameters(params, "tokens", "channels")
    logical_ops = 8 * tokens * channels
    represented = 4 * tokens * channels + 8 * channels + 12 * channels
    result = _base_counts()
    result.update({
        "logical_ops": logical_ops,
        "issued_ops": {"fp32_vector_arithmetic": logical_ops},
        "represented_minimum_bytes": represented,
        "source_requested_bytes": represented,
        "special_functions": {"exp_arguments": tokens * channels},
        "accounting_note": "four-tap FMA arithmetic only; SiLU exponential reported separately",
    })
    return result


def _elementwise(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    elements = _integer(params.get("elements"), "elements", minimum=1)
    result = _base_counts()
    if operation == "residual_add":
        result.update(logical_ops=elements, issued_ops={"fp32_vector_arithmetic": elements},
                      represented_minimum_bytes=6 * elements,
                      source_requested_bytes=6 * elements,
                      accounting_note="one represented elementwise addition")
    elif operation == "silu_mul":
        result.update(logical_ops=4 * elements,
                      issued_ops={"fp32_vector_arithmetic": 4 * elements},
                      represented_minimum_bytes=6 * elements,
                      source_requested_bytes=6 * elements,
                      special_functions={"exp_arguments": elements},
                      accounting_note="four scalar arithmetic ops plus one exponential per element")
    elif operation in ("rmsnorm", "residual_rmsnorm", "gated_rmsnorm"):
        features = _integer(params.get("features"), "features", minimum=1)
        if elements % features:
            raise ValueError("RMSNorm elements must be divisible by features")
        rows = elements // features
        if operation == "gated_rmsnorm":
            # Per element: sumsq FMA (2), SiLU negate/add/divide (3), and
            # input*inverse*weight*SiLU (3). Per row: mean divide and epsilon add (2).
            logical_ops = 8 * elements + 2 * rows
            represented = 6 * elements + 2 * features
            requested = 8 * elements + 2 * features * rows
        else:
            logical_ops = 4 * elements
            represented = (2 * elements + 2 * features + 2 * elements
                           if operation == "rmsnorm"
                           else 4 * elements + 2 * features + 4 * elements)
            requested = (4 * elements + 2 * features * rows
                         if operation == "rmsnorm"
                         else 8 * elements + 2 * features * rows)
        special_functions = {"rsqrt_arguments": rows}
        if operation == "gated_rmsnorm":
            special_functions["exp_arguments"] = elements
        result.update(logical_ops=logical_ops,
                      issued_ops={"fp32_vector_arithmetic": logical_ops},
                      represented_minimum_bytes=represented,
                      source_requested_bytes=requested,
                      special_functions=special_functions,
                      accounting_note=(
                          "sumsq, mean/epsilon, SiLU, and gated-output scalar arithmetic; "
                          "reduction, exponential, and rsqrt reported separately"
                          if operation == "gated_rmsnorm" else
                          "sumsq/output arithmetic; reduction and rsqrt separate"))
    return result


def _counts(operation: str, params: dict[str, Any], dispatch_id: str,
            keep_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if operation == "q4_linear":
        return _linear(params, q4=True)
    if operation == "w8_linear":
        return _linear(params, q4=False)
    if operation == "q4_quantize":
        return _quantize(params, group=64)
    if operation == "w8_quantize":
        return _quantize(params, group=32)
    if operation == "dense_attention":
        return _dense_attention(params)
    if operation == "sparse_pack":
        return _sparse_pack(params)
    if operation == "sparse_rank":
        return _sparse_rank(params)
    if operation == "sparse_consumer":
        return _sparse_consumer(params, dispatch_id, keep_rows)
    if operation == "gdn_recurrence":
        return _gdn_recurrence(params)
    if operation == "gdn_control":
        return _gdn_control(params)
    if operation == "gdn_conv":
        return _gdn_conv(params)
    if operation in ("residual_add", "silu_mul", "rmsnorm", "residual_rmsnorm",
                     "gated_rmsnorm"):
        return _elementwise(operation, params)
    raise ValueError(f"unsupported operation for {dispatch_id}: {operation}")


def _rates(counts: dict[str, Any], duration_ns: int,
           probe_copy_gbps: float | None = None) -> dict[str, Any]:
    seconds = duration_ns / 1e9
    logical_ops = counts["logical_ops"]
    represented = counts["represented_minimum_bytes"]
    requested = counts["source_requested_bytes"]
    issued_rates = {name: value / seconds / 1e12
                    for name, value in counts["issued_ops"].items()}
    peak_key = {
        "iu4": "int4_matrix_tops",
        "iu8": "int8_matrix_tops",
        "bf16_matrix": "bf16_matrix_tflops",
        "fp32_vector_arithmetic": "fp32_vector_tflops",
    }
    fractions = {name: issued_rates[name] / THEORETICAL_PEAKS[peak_key[name]]
                 for name in issued_rates}
    represented_gbps = represented / seconds / 1e9
    requested_gbps = None if requested is None else requested / seconds / 1e9
    return {
        "effective_logical_tops": logical_ops / seconds / 1e12,
        "represented_minimum_gbps": represented_gbps,
        "source_requested_gbps": requested_gbps,
        "logical_arithmetic_intensity_ops_per_represented_byte": (
            None if represented == 0 else logical_ops / represented),
        "source_request_arithmetic_intensity_ops_per_byte": (
            None if requested in (None, 0) else logical_ops / requested),
        "issued_tops_by_class": issued_rates,
        "theoretical_compute_peak_fraction_by_class": fractions,
        "represented_to_advertised_memory_ratio": (
            represented_gbps / THEORETICAL_PEAKS["advertised_memory_gbps"]),
        "represented_to_supplied_copy_probe_ratio": (
            None if probe_copy_gbps is None else represented_gbps / probe_copy_gbps),
        "source_requested_to_supplied_copy_probe_ratio": (
            None if probe_copy_gbps is None or requested_gbps is None
            else requested_gbps / probe_copy_gbps),
    }


def _aggregate(rows: list[dict[str, Any]], key: str,
               probe_copy_gbps: float | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = []
    for name, members in sorted(groups.items()):
        duration = sum(item["duration_ns"] for item in members)
        wall_union = _wall_union_ns([
            (item["start_ns"], item["end_ns"]) for item in members])
        counts = _base_counts()
        counts["logical_ops"] = sum(item["counts"]["logical_ops"] for item in members)
        counts["represented_minimum_bytes"] = sum(
            item["counts"]["represented_minimum_bytes"] for item in members)
        requested = [item["counts"]["source_requested_bytes"] for item in members]
        counts["source_requested_bytes"] = (None if any(item is None for item in requested)
                                               else sum(requested))
        issued: dict[str, int] = defaultdict(int)
        special: dict[str, int] = defaultdict(int)
        for item in members:
            for kind, value in item["counts"]["issued_ops"].items():
                issued[kind] += value
            for kind, value in item["counts"]["special_functions"].items():
                special[kind] += value
        counts["issued_ops"] = dict(sorted(issued.items()))
        counts["special_functions"] = dict(sorted(special.items()))
        result.append({key: name, "dispatches": len(members),
                       "independent_device_service_time_ns": duration,
                       "device_wall_union_ns": wall_union,
                       "counts": counts,
                       "rates": _rates(counts, duration, probe_copy_gbps)})
    return result


def _reconciled_inputs(reconciliation_path: Path) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]],
        dict[str, Any], dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    reconciliation = _load(reconciliation_path)
    if (reconciliation.get("artifact_type") != RECONCILIATION_TYPE
            or type(reconciliation.get("schema_version")) is not int
            or reconciliation.get("schema_version") != 1):
        raise ValueError("reconciliation must be Qwen3.8-27B dispatch reconciliation schema v1")
    tracked: list[tuple[Path, dict[str, Any]]] = []
    for field in ("trace_authority", "static_schedule"):
        tracked.append(_verified_snapshot(reconciliation.get(field), field))
    sources = reconciliation.get("schedule_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("reconciliation.schedule_sources must be a nonempty array")
    tracked.extend(_verified_snapshot(value, f"schedule_sources[{index}]")
                   for index, value in enumerate(sources))

    trace = _load(tracked[0][0])
    if (trace.get("artifact_type") != TRACE_TYPE
            or type(trace.get("schema_version")) is not int
            or trace.get("schema_version") != 1
            or trace.get("status") != "valid_attribution_only"
            or trace.get("profile_timing_admissible") is not False):
        raise ValueError("trace authority is not validated attribution-only schema v1")
    if reconciliation.get("workload") != reconciliation.get("timing_authority", {}).get("workload"):
        raise ValueError("reconciliation and timing workload identities differ")
    workload = reconciliation.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("reconciliation workload must be an object")
    _validate_workload(workload)
    if (workload.get("kind"), workload.get("prompt_tokens"),
            workload.get("concurrency"), workload.get("xattention_profile")) != (
            "pp", 2048, 1, "dense"):
        raise ValueError("roofline reconciliation must be the dense C1 P2048 selected route")
    route = trace.get("selected_route")
    if not isinstance(route, dict):
        raise ValueError("trace authority lacks selected_route")
    expected_route = {
        "kind": workload["kind"], "prompt_tokens": workload["prompt_tokens"],
        "concurrency": workload["concurrency"], "prefill_chunk": workload["prefill_chunk"],
        "kv_value_group": workload["kv_value_group"],
        "xattention_profile": workload["xattention_profile"],
    }
    if any(route.get(key) != value for key, value in expected_route.items()):
        raise ValueError("trace selected_route differs from reconciled workload")
    artifact, executable = route.get("artifact"), route.get("benchmark_executable")
    if (not isinstance(artifact, dict) or not isinstance(executable, dict)
            or artifact.get("path") != workload["artifact_path"]
            or artifact.get("sha256") != workload["artifact_sha256"]
            or artifact.get("weights_id") != workload["weights_id"]
            or executable.get("path") != workload["executable_path"]
            or executable.get("sha256") != workload["executable_sha256"]):
        raise ValueError("trace selected artifact/executable differs from reconciled workload")

    unprofiled = reconciliation.get("unprofiled_performance")
    if not isinstance(unprofiled, dict) or unprofiled.get("prompt_tokens") != 2048:
        raise ValueError("reconciliation lacks exact unprofiled P2048 performance")
    evaluation_pair = _verified_snapshot(
        unprofiled.get("low_context_evaluation"), "unprofiled low-context evaluation")
    report_pair = _verified_snapshot(unprofiled.get("p2048_report"), "unprofiled P2048 report")
    tracked.extend((evaluation_pair, report_pair))
    trace_authorities = trace.get("authorities")
    selected_unprofiled = route.get("unprofiled_p2048")
    if (not isinstance(trace_authorities, dict)
            or trace_authorities.get("low_context_evaluation") != evaluation_pair[1]
            or trace_authorities.get("source_report") != report_pair[1]
            or not isinstance(selected_unprofiled, dict)
            or selected_unprofiled.get("evaluation") != evaluation_pair[1]
            or selected_unprofiled.get("report") != report_pair[1]):
        raise ValueError("unprofiled P2048 snapshots differ from validated trace authority")
    evaluation, report = _load(evaluation_pair[0]), _load(report_pair[0])
    if (evaluation.get("artifact_type") != LOW_CONTEXT_TYPE
            or type(evaluation.get("schema_version")) is not int
            or evaluation.get("schema_version") != 1):
        raise ValueError("unprofiled low-context evaluation has the wrong schema")
    tests = report.get("tests")
    rows = ([row for row in tests if isinstance(row, dict) and row.get("kind") == "pp"
             and row.get("n_prompt") == 2048] if isinstance(tests, list) else [])
    if len(rows) != 1 or len(tests) != 1:
        raise ValueError("unprofiled report must contain one exact P2048 row")
    observed = _number(evaluation.get("observed_p2048_tok_s"),
                       "evaluation.observed_p2048_tok_s", positive=True)
    speed = _number(unprofiled.get("prefill_tok_s_mean"),
                    "unprofiled.prefill_tok_s_mean", positive=True)
    seconds = _number(unprofiled.get("prefill_seconds_mean"),
                      "unprofiled.prefill_seconds_mean", positive=True)
    if (speed != observed
            or speed != _number(rows[0].get("prefill_tok_s_mean"),
                                "report.prefill_tok_s_mean", positive=True)
            or seconds != _number(rows[0].get("prefill_seconds_mean"),
                                  "report.prefill_seconds_mean", positive=True)):
        raise ValueError("unprofiled P2048 throughput differs across bound authorities")
    minimum = _number(evaluation.get("minimum_p2048_tok_s"),
                      "evaluation.minimum_p2048_tok_s", positive=True)
    passes = evaluation.get("passes_p2048_gate")
    if (type(passes) is not bool
            or selected_unprofiled.get("observed_tok_s") != speed
            or selected_unprofiled.get("minimum_tok_s") != minimum
            or type(selected_unprofiled.get("passes_gate")) is not bool
            or selected_unprofiled.get("passes_gate") is not passes):
        raise ValueError("validated trace selected P2048 gate differs from its authority")
    unprofiled_exact = {
        "prompt_tokens": 2048, "prefill_tok_s_mean": speed,
        "prefill_seconds_mean": seconds,
        "low_context_evaluation": evaluation_pair[1], "p2048_report": report_pair[1],
    }

    timing, inventory = reconciliation.get("timing_authority"), reconciliation.get("dispatch_inventory")
    if not isinstance(timing, dict) or not isinstance(inventory, dict):
        raise ValueError("reconciliation lacks embedded timing/inventory authorities")
    _schema(timing, TIMING_TYPE, "embedded timing authority")
    _schema(inventory, INVENTORY_TYPE, "embedded dispatch inventory")
    if inventory.get("timing_authority_sha256") != _canonical_sha256(timing):
        raise ValueError("embedded inventory does not bind the canonical timing authority")

    coverage_rows = reconciliation.get("dispatches")
    coverage = reconciliation.get("coverage")
    if not isinstance(coverage_rows, list) or not coverage_rows or not isinstance(coverage, dict):
        raise ValueError("reconciliation lacks complete dispatch coverage")
    seen: set[str] = set()
    totals = {name: {"dispatch_count": 0, "duration_ns": 0}
              for name in ("modeled", "unmodeled", "unsupported")}
    uncovered = []
    modeled_ids = set()
    trace_rows = trace.get("dispatches")
    if not isinstance(trace_rows, list) or len(trace_rows) != len(coverage_rows):
        raise ValueError("reconciliation dispatches do not cover every validated trace dispatch")
    for index, row in enumerate(coverage_rows):
        if not isinstance(row, dict):
            raise ValueError(f"reconciliation dispatch {index} must be an object")
        dispatch_id = _text(row.get("dispatch_id"), f"reconciliation dispatch {index}.dispatch_id")
        if dispatch_id in seen:
            raise ValueError(f"duplicate reconciliation dispatch_id: {dispatch_id}")
        seen.add(dispatch_id)
        trace_row = trace_rows[index]
        if not isinstance(trace_row, dict) or trace_row.get("dispatch_id") != dispatch_id:
            raise ValueError("reconciliation dispatch order/identity differs from validated trace")
        for trace_field, reconciliation_field in (
                ("rocprof_dispatch_id", "rocprof_dispatch_id"), ("symbol", "symbol"),
                ("roctx_region", "marker"), ("grid", "grid"), ("workgroup", "workgroup"),
                ("start_ns", "start_ns"), ("end_ns", "end_ns"),
                ("duration_ns", "duration_ns")):
            if trace_row.get(trace_field) != row.get(reconciliation_field):
                raise ValueError(
                    f"reconciliation dispatch {dispatch_id} differs from trace at {trace_field}")
        classification = row.get("classification")
        if classification not in totals:
            raise ValueError(f"reconciliation dispatch {dispatch_id} has invalid classification")
        duration = _integer(row.get("duration_ns"), f"{dispatch_id}.duration_ns", minimum=1)
        totals[classification]["dispatch_count"] += 1
        totals[classification]["duration_ns"] += duration
        if classification == "modeled":
            modeled_ids.add(dispatch_id)
            if "reason" in row:
                raise ValueError(f"modeled dispatch {dispatch_id} cannot have an uncovered reason")
        else:
            _text(row.get("reason"), f"{dispatch_id}.reason")
            uncovered.append(dict(row))
    trace_aggregates = trace.get("aggregates")
    trace_service = sum(value["duration_ns"] for value in totals.values())
    if (not isinstance(trace_aggregates, dict)
            or trace_aggregates.get("dispatch_count") != len(coverage_rows)
            or trace_aggregates.get("independent_device_service_time_ns") != trace_service):
        raise ValueError("validated trace aggregates differ from reconciliation coverage")
    expected_coverage = {
        "trace_dispatch_count": len(coverage_rows),
        "trace_service_time_ns": trace_service,
        **totals,
    }
    _integer(coverage.get("trace_dispatch_count"), "coverage.trace_dispatch_count")
    _integer(coverage.get("trace_service_time_ns"), "coverage.trace_service_time_ns")
    for name in totals:
        group = coverage.get(name)
        if not isinstance(group, dict):
            raise ValueError(f"coverage.{name} must be an object")
        _integer(group.get("dispatch_count"), f"coverage.{name}.dispatch_count")
        _integer(group.get("duration_ns"), f"coverage.{name}.duration_ns")
    if coverage != expected_coverage:
        raise ValueError("reconciliation coverage does not exactly partition trace dispatches")
    timing_ids = set(_unique_rows(timing, "embedded timing"))
    timing_rows = _unique_rows(timing, "embedded timing")
    inventory_rows = _unique_rows(inventory, "embedded inventory")
    inventory_ids = set(inventory_rows)
    if timing_ids != modeled_ids or inventory_ids != modeled_ids:
        raise ValueError("embedded timing/inventory must equal the modeled dispatch subset")
    coverage_by_id = {row["dispatch_id"]: row for row in coverage_rows}
    for dispatch_id in modeled_ids:
        covered = coverage_by_id[dispatch_id]
        timed, inventoried = timing_rows[dispatch_id], inventory_rows[dispatch_id]
        if any(timed.get(field) != covered.get(field) for field in
               ("symbol", "start_ns", "end_ns", "duration_ns")):
            raise ValueError(f"modeled timing row {dispatch_id} differs from trace coverage")
        if timed.get("stage") != covered.get("marker"):
            raise ValueError(f"modeled timing stage {dispatch_id} differs from trace marker")
        if any(inventoried.get(field) != covered.get(field) for field in
               ("symbol", "role", "operation", "format", "parameters")):
            raise ValueError(f"modeled inventory row {dispatch_id} differs from trace coverage")
        if inventoried.get("stage") != covered.get("marker"):
            raise ValueError(f"modeled inventory stage {dispatch_id} differs from trace marker")
    return timing, inventory, workload, uncovered, unprofiled_exact, expected_coverage, tracked


def model(reconciliation_path: Path, *, probe_path: Path | None = None) -> dict[str, Any]:
    reconciliation_snapshot = {
        "path": str(reconciliation_path.resolve(strict=True)),
        "file_size_bytes": reconciliation_path.resolve(strict=True).stat().st_size,
        "sha256": _sha256(reconciliation_path.resolve(strict=True)),
    }
    timing, inventory, workload, uncovered, unprofiled, coverage, tracked = _reconciled_inputs(
        reconciliation_path)
    _schema(timing, TIMING_TYPE, "timing authority")
    _schema(inventory, INVENTORY_TYPE, "dispatch inventory")
    power = timing.get("power_profile")
    if (not isinstance(power, dict) or power.get("required") != "auto"
            or power.get("observed") != "auto"
            or power.get("rechecked_after") != "auto"):
        raise ValueError("timing authority must record required/observed/rechecked_after auto")
    _text(power.get("sysfs_path"), "timing power_profile.sysfs_path")
    if not isinstance(workload, dict) or inventory.get("workload") != workload:
        raise ValueError("timing and inventory workload identities must match exactly")
    _validate_workload(workload)

    timing_rows = _unique_rows(timing, "timing")
    inventory_rows = _unique_rows(inventory, "inventory")
    if set(timing_rows) != set(inventory_rows):
        missing = sorted(set(timing_rows) - set(inventory_rows))
        extra = sorted(set(inventory_rows) - set(timing_rows))
        raise ValueError(f"dispatch identity mismatch; missing inventory={missing}, extra={extra}")
    if (workload["xattention_profile"] == "dense"
            and any(row.get("operation", "").startswith("sparse_")
                    for row in inventory_rows.values())):
        raise ValueError("dense workload cannot contain sparse XAttention dispatches")

    keep_rows: dict[str, dict[str, Any]] = {}
    sparse_ids = {dispatch_id for dispatch_id, row in inventory_rows.items()
                  if row.get("operation") == "sparse_consumer"}
    if sparse_ids:
        raise ValueError("schema-v1 selected-P2048 reconciliation must not contain sparse consumers")

    probe = None
    probe_copy_gbps = None
    if probe_path is not None:
        probe = _load(probe_path)
        if probe.get("artifact_type") != "ninfer_r9700_memory_probe" or probe.get("schema_version") != 1:
            raise ValueError("probe reference must be ninfer_r9700_memory_probe schema v1")
        for name in ("read_gbps", "write_gbps", "copy_gbps"):
            _number(probe.get(name), f"probe.{name}", positive=True)
        probe_copy_gbps = float(probe["copy_gbps"])

    rows = []
    intervals = []
    for dispatch_id, timing_row in timing_rows.items():
        inventory_row = inventory_rows[dispatch_id]
        for name in ("symbol", "stage"):
            expected = _text(inventory_row.get(name), f"inventory {dispatch_id}.{name}")
            if timing_row.get(name) != expected:
                raise ValueError(f"{dispatch_id} {name} differs between timing and inventory")
        begin = _integer(timing_row.get("start_ns"), f"{dispatch_id}.start_ns")
        end = _integer(timing_row.get("end_ns"), f"{dispatch_id}.end_ns", minimum=1)
        duration = _integer(timing_row.get("duration_ns"), f"{dispatch_id}.duration_ns", minimum=1)
        if end <= begin or end - begin != duration:
            raise ValueError(f"{dispatch_id} timing interval/duration is inconsistent")
        operation = _text(inventory_row.get("operation"), f"inventory {dispatch_id}.operation")
        expected_format = FORMATS.get(operation)
        if expected_format is None:
            raise ValueError(f"unsupported operation for {dispatch_id}: {operation}")
        tensor_format = _text(inventory_row.get("format"),
                              f"inventory {dispatch_id}.format")
        if tensor_format != expected_format:
            raise ValueError(
                f"inventory {dispatch_id}.format must be {expected_format} for {operation}")
        params = inventory_row.get("parameters")
        if not isinstance(params, dict):
            raise ValueError(f"inventory {dispatch_id}.parameters must be an object")
        counts = _counts(operation, params, dispatch_id, keep_rows)
        row = {
            "dispatch_id": dispatch_id,
            "symbol": timing_row["symbol"],
            "stage": timing_row["stage"],
            "operation": operation,
            "role": _text(inventory_row.get("role"), f"inventory {dispatch_id}.role"),
            "format": tensor_format,
            "parameters": params,
            "start_ns": begin,
            "end_ns": end,
            "duration_ns": duration,
            "counts": counts,
            "rates": _rates(counts, duration, probe_copy_gbps),
        }
        rows.append(row)
        intervals.append((begin, end))

    service_ns = sum(row["duration_ns"] for row in rows)
    wall_union_ns = _wall_union_ns(intervals)
    if service_ns <= 0 or wall_union_ns <= 0:
        raise ValueError("timing authority contains no positive device service")

    for path, snapshot in tracked:
        actual = {"path": str(path.resolve(strict=True)), "file_size_bytes": path.stat().st_size,
                  "sha256": _sha256(path)}
        if actual != snapshot:
            raise ValueError(f"bound authority changed while building roofline: {path}")
    if ({"path": str(reconciliation_path.resolve(strict=True)),
         "file_size_bytes": reconciliation_path.resolve(strict=True).stat().st_size,
         "sha256": _sha256(reconciliation_path.resolve(strict=True))}
            != reconciliation_snapshot):
        raise ValueError("reconciliation changed while building roofline")

    return {
        "artifact_type": OUTPUT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dispatch_reconciliation": reconciliation_snapshot,
        "workload": workload,
        "power_profile": power,
        "unprofiled_whole_p2048": {
            **unprofiled,
            "authority": "validated low-context 3/1 benchmark; never profiler timing",
        },
        "dispatch_coverage": coverage,
        "peak_references": {
            "theoretical_advertised": THEORETICAL_PEAKS,
            "supplied_memory_probe_reference": None if probe_path is None else {
                "path": str(probe_path.resolve()), "sha256": _sha256(probe_path),
                "report": probe,
            },
            "interpretation": (
                "compute fractions use theoretical issued-operation peaks; represented/probe "
                "ratios are logical reference ratios against the explicitly supplied probe, "
                "not proof of same-session collection or physical bandwidth utilization"
            ),
        },
        "physical_measurement": {
            "hbm_bytes": None,
            "hbm_bandwidth_gbps": None,
            "hbm_peak_fraction": None,
            "stall_fraction": None,
            "status": "unavailable on the retained gfx1201 ROCm counter path",
            "gl2_tcp_note": "GL2/TCP ratios remain separate relative profiler evidence",
        },
        "timing": {
            "independent_device_service_time_ns": service_ns,
            "device_wall_union_ns": wall_union_ns,
            "profiled_dispatch_count": len(rows),
            "uncovered_dispatch_count": len(uncovered),
            "note": (
                "profiler durations are dispatch attribution only; service sums modeled dispatch "
                "durations and wall union preserves their overlap"
            ),
        },
        "dispatches": sorted(rows, key=lambda row: row["dispatch_id"]),
        "uncovered_dispatches": sorted(uncovered, key=lambda row: row["dispatch_id"]),
        "aggregates_by_operation": _aggregate(rows, "operation", probe_copy_gbps),
        "aggregates_by_stage": _aggregate(rows, "stage", probe_copy_gbps),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-reconciliation", required=True, type=Path)
    parser.add_argument("--memory-probe", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise SystemExit(f"output already exists: {args.out_json}")
    try:
        result = model(args.dispatch_reconciliation, probe_path=args.memory_probe)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote Qwen3.8-27B logical roofline evidence to {args.out_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
