#!/usr/bin/env python3
"""Fail-closed comparison of the two eager layer-3 target-attention trace arms."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
from pathlib import Path

TOKEN_DOMAIN = 248077
POSITION = 129
STAGE_FIELDS = (
    ("input_x", "bf16", 5120),
    ("norm_h", "bf16", 5120),
    ("projection_q", "bf16", 6144),
    ("projection_gate", "bf16", 6144),
    ("projection_k", "bf16", 1024),
    ("projection_v", "bf16", 1024),
    ("normalized_qk", "bf16", 7168),
    ("rope_qk", "bf16", 7168),
    ("attention_fp32", "fp32", 6144),
    ("attention_bf16", "bf16", 6144),
    ("gated_attention", "bf16", 6144),
    ("residual_x", "bf16", 5120),
)
FIELD_SHAPES = {
    "input_x": [5120], "norm_h": [5120],
    "projection_q": [256, 24], "projection_gate": [256, 24],
    "projection_k": [256, 4], "projection_v": [256, 4],
    "normalized_qk": [7168], "rope_qk": [7168],
    "attention_fp32": [256, 24], "attention_bf16": [256, 24],
    "gated_attention": [256, 24], "residual_x": [5120],
    "key_fp8": [130, 4, 256], "value_int4": [130, 4, 128],
    "value_scale_fp16": [130, 4, 16],
}
STAGE_BYTES = 137216
CACHE_FIELDS = (
    ("key_fp8", "u8", 130 * 4 * 256),
    ("value_int4", "u8", 130 * 4 * 128),
    ("value_scale_fp16", "fp16", 130 * 4 * 16),
)
CACHE_BYTES = 216320
SIDECAR_BYTES = STAGE_BYTES + CACHE_BYTES
ROLES = {
    "ordinary": ("target-ordinary-frontier130", 1, 130),
    "dflash": ("target-dflash-frontier130-column0", 5, 134),
}
MANIFEST_KEYS = {
    "artifact_type", "schema_version", "diagnostic_only", "timing_evidence_eligible",
    "production_routing_authorized", "execution", "role", "call", "cache_read",
    "visibility", "provenance", "fields", "sidecar_path", "sidecar_bytes",
    "sidecar_fnv1a64", "layout",
}
CALL_KEYS = {
    "phase", "batch", "width", "selected_column", "absolute_frontier", "token",
    "cache_position", "rope_position", "text_layer", "full_attention_index",
    "sequence_batch", "sequence_width", "transaction_position_count", "live_width",
}
CACHE_KEYS = {
    "valid_for_stream", "pending", "visible_frontier", "mapped_pages", "head_dim",
    "kv_heads", "value_group", "key_layout", "value_layout", "value_scale_layout",
    "device_table_row_present", "pool_table_row_stride", "pool_table_row_count",
    "active_query_rows_present",
}
VISIBILITY_KEYS = {"row_position", "ancestor_masks", "prefix_lengths", "prefix_length_stride"}
PROVENANCE_KEYS = {
    "source_commit", "source_tree", "executable_sha256", "artifact_sha256",
    "history_sha256", "corpus_sha256", "device_index", "device_name", "architecture",
    "wave_size", "pci", "power_profile_source", "power_profile_value",
}
CLASSIFICATION = {
    "input_x": "upstream_input_difference",
    "norm_h": "input_rmsnorm_difference",
    "projection_q": "attention_projection_difference",
    "projection_gate": "attention_projection_difference",
    "projection_k": "attention_projection_difference",
    "projection_v": "attention_projection_difference",
    "normalized_qk": "qk_rmsnorm_difference",
    "rope_qk": "rope_or_position_difference",
    "attention_fp32": "full_attention_route_arithmetic_difference",
    "attention_bf16": "fp32_to_bf16_cast_difference",
    "gated_attention": "sigmoid_gate_difference",
    "residual_x": "attention_output_projection_or_residual_difference",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON value: {token}"))
    if not isinstance(value, dict):
        fail(f"manifest is not an object: {path}")
    return value


def _integer(value: object, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        fail(f"{label} is below its minimum")
    return value


def fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def _field_layout() -> list[dict]:
    fields = []
    offset = 0
    sizes = {"bf16": 2, "fp32": 4, "u8": 1, "fp16": 2}
    for name, dtype, elements in STAGE_FIELDS + CACHE_FIELDS:
        byte_count = elements * sizes[dtype]
        if name in {"normalized_qk", "rope_qk"}:
            layout = "q[256,24],then-k[256,4],each-feature-fastest"
        elif name in {item[0] for item in CACHE_FIELDS}:
            layout = "logical-position-major,head,feature-fastest"
        else:
            layout = "selected-token feature-fastest"
        fields.append({"name": name, "dtype": dtype, "shape": FIELD_SHAPES[name],
                       "layout": layout, "elements": elements, "offset": offset,
                       "bytes": byte_count})
        offset += byte_count
    if offset != SIDECAR_BYTES:
        fail("internal field layout is inconsistent")
    return fields


FIELDS = _field_layout()
FIELD_BY_NAME = {field["name"]: field for field in FIELDS}


def _validate_provenance(value: object, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != PROVENANCE_KEYS:
        fail(f"{label} provenance schema differs")
    for key in ("executable_sha256", "artifact_sha256", "history_sha256", "corpus_sha256"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            fail(f"{label} provenance {key} is invalid")
    for key in ("source_commit", "source_tree"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{40}", value[key]) is None:
            fail(f"{label} provenance {key} is invalid")
    exact = {
        "device_index": 0, "device_name": "AMD Radeon AI PRO R9700",
        "architecture": "gfx1201", "wave_size": 32, "pci": "0000:13:00.0",
        "power_profile_value": "auto",
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            fail(f"{label} provenance {key} differs")
    if not isinstance(value.get("power_profile_source"), str) or not value["power_profile_source"]:
        fail(f"{label} power profile source is invalid")
    return value


def _visible_positions(value: dict, kind: str) -> list[int]:
    if set(value) != VISIBILITY_KEYS:
        fail(f"{kind} visibility schema differs")
    row = _integer(value["row_position"], f"{kind} row_position", 0)
    if row != POSITION:
        fail(f"{kind} row position differs")
    masks = value["ancestor_masks"]
    prefixes = value["prefix_lengths"]
    stride = _integer(value["prefix_length_stride"], f"{kind} prefix stride", 0)
    if kind == "ordinary":
        if masks is not None or prefixes is not None or stride != 1:
            fail("ordinary visibility metadata differs")
        return list(range(row + 1))
    if (not isinstance(masks, list) or len(masks) != 5 or
            any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= 2**5
                for item in masks)):
        fail("DFlash ancestor masks are invalid")
    if stride not in (0, 1) or not isinstance(prefixes, list) or len(prefixes) != (1 if stride == 0 else 5):
        fail("DFlash prefix metadata is invalid")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in prefixes):
        fail("DFlash prefix values are invalid")
    if any(item != POSITION for item in prefixes):
        fail("DFlash prefix values differ from the bound history frontier")
    prefix = prefixes[0]
    mask = masks[0]
    visible = list(range(prefix))
    for bit in range(31):
        if mask & (1 << bit):
            visible.append(prefix + bit)
    if len(visible) != len(set(visible)):
        fail("DFlash visibility contains duplicates")
    visible = sorted(visible)
    frontier = ROLES[kind][2]
    if any(position >= frontier for position in visible):
        fail("DFlash visibility exceeds the pending cache frontier")
    return visible


def _decode_finite(data: bytes, dtype: str, elements: int, label: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if dtype == "bf16":
        bits = struct.unpack(f"<{elements}H", data)
        values = tuple(struct.unpack("<f", struct.pack("<I", item << 16))[0] for item in bits)
    elif dtype == "fp32":
        bits = struct.unpack(f"<{elements}I", data)
        values = tuple(struct.unpack("<f", struct.pack("<I", item))[0] for item in bits)
    elif dtype == "fp16":
        bits = struct.unpack(f"<{elements}H", data)
        values = struct.unpack(f"<{elements}e", data)
    else:
        return (), ()
    if not all(math.isfinite(item) for item in values):
        fail(f"nonfinite represented value in {label}")
    return bits, values


def _load(path: Path, kind: str) -> tuple[dict, bytes, list[int]]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        fail(f"{kind} manifest identity is invalid")
    value = _load_json(path)
    if set(value) != MANIFEST_KEYS:
        fail(f"{kind} manifest schema differs")
    role, width, frontier = ROLES[kind]
    exact = {
        "artifact_type": "ninfer_qwen3_layer3_attention_trace", "schema_version": 1,
        "diagnostic_only": True, "timing_evidence_eligible": False,
        "production_routing_authorized": False, "execution": "eager", "role": role,
        "sidecar_bytes": SIDECAR_BYTES,
        "layout": "typed selected-column layer3 attention stages then canonical logical cache positions0..129",
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            fail(f"{kind} manifest {key} differs")
    call = value.get("call")
    if not isinstance(call, dict) or set(call) != CALL_KEYS:
        fail(f"{kind} call schema differs")
    call_exact = {
        "phase": "verify", "batch": 1, "width": width, "selected_column": 0,
        "absolute_frontier": 130, "token": 96558, "cache_position": POSITION,
        "rope_position": POSITION, "text_layer": 3, "full_attention_index": 0,
        "sequence_batch": 1, "sequence_width": width,
        "transaction_position_count": width, "live_width": width,
    }
    for key, expected in call_exact.items():
        if call.get(key) != expected:
            fail(f"{kind} call {key} differs")
    cache = value.get("cache_read")
    if not isinstance(cache, dict) or set(cache) != CACHE_KEYS:
        fail(f"{kind} cache-read schema differs")
    cache_exact = {
        "valid_for_stream": True, "pending": True, "visible_frontier": frontier,
        "head_dim": 256, "kv_heads": 4, "value_group": 16,
        "key_layout": "token-fastest-head-major",
        "value_layout": "feature-fastest-page-major",
        "value_scale_layout": "feature-fastest-page-major",
        "device_table_row_present": True, "active_query_rows_present": False,
    }
    for key, expected in cache_exact.items():
        if cache.get(key) != expected:
            fail(f"{kind} cache-read {key} differs")
    mapped = _integer(cache.get("mapped_pages"), f"{kind} mapped_pages", 1)
    if mapped * 256 < frontier:
        fail(f"{kind} mapped pages do not cover the frontier")
    stride = _integer(cache.get("pool_table_row_stride"), f"{kind} table stride", 1)
    rows = _integer(cache.get("pool_table_row_count"), f"{kind} table rows", 1)
    if stride < mapped or rows < 1:
        fail(f"{kind} table geometry is invalid")
    visible = _visible_positions(value.get("visibility"), kind) if isinstance(value.get("visibility"), dict) else fail(f"{kind} visibility is invalid")
    _validate_provenance(value.get("provenance"), kind)
    if value.get("fields") != FIELDS:
        fail(f"{kind} field layout differs")
    raw_path = value.get("sidecar_path")
    sidecar = Path(raw_path) if isinstance(raw_path, str) else None
    expected_sidecar = path.with_suffix(".bin")
    if sidecar is None or sidecar != expected_sidecar or not sidecar.is_absolute() or sidecar.is_symlink() or not sidecar.is_file():
        fail(f"{kind} sidecar identity is invalid")
    data = sidecar.read_bytes()
    if len(data) != SIDECAR_BYTES or value.get("sidecar_fnv1a64") != fnv1a64(data):
        fail(f"{kind} sidecar bytes/hash differ")
    for field in FIELDS:
        begin, end = field["offset"], field["offset"] + field["bytes"]
        _decode_finite(data[begin:end], field["dtype"], field["elements"], field["name"])
    return value, data, visible


def _difference(field: dict, left: bytes, right: bytes) -> dict:
    begin, end = field["offset"], field["offset"] + field["bytes"]
    lhs, rhs = left[begin:end], right[begin:end]
    indices = [index for index, pair in enumerate(zip(lhs, rhs)) if pair[0] != pair[1]]
    detail = {"field": field["name"], "first_byte_index": indices[0],
              "mismatch_byte_count": len(indices)}
    bits_l, values_l = _decode_finite(lhs, field["dtype"], field["elements"], field["name"])
    bits_r, values_r = _decode_finite(rhs, field["dtype"], field["elements"], field["name"])
    if bits_l:
        element_bytes = 4 if field["dtype"] == "fp32" else 2
        element = indices[0] // element_bytes
        detail.update({"first_element_index": element, "left_bits": bits_l[element],
                       "right_bits": bits_r[element],
                       "maximum_absolute_difference": max(abs(a - b) for a, b in zip(values_l, values_r))})
    return detail


def analyze(ordinary_path: Path, dflash_path: Path) -> dict:
    ordinary, left, ordinary_visible = _load(ordinary_path, "ordinary")
    dflash, right, dflash_visible = _load(dflash_path, "dflash")
    if ordinary["provenance"] != dflash["provenance"]:
        fail("paired provenance differs")
    expected_visible = list(range(130))
    if ordinary_visible != expected_visible:
        fail("ordinary causal visibility differs from positions0..129")

    detail = None
    classification = "layer3_attention_internal_exact"
    for name, _, _ in STAGE_FIELDS[:8]:
        field = FIELD_BY_NAME[name]
        begin, end = field["offset"], field["offset"] + field["bytes"]
        if left[begin:end] != right[begin:end]:
            classification = CLASSIFICATION[name]
            detail = _difference(field, left, right)
            break
    if detail is None and dflash_visible != expected_visible:
        classification = "tree_visibility_difference"
        detail = {"ordinary_visible_positions": ordinary_visible,
                  "dflash_visible_positions": dflash_visible}
    if detail is None:
        for name, _, _ in CACHE_FIELDS:
            field = FIELD_BY_NAME[name]
            begin, end = field["offset"], field["offset"] + field["bytes"]
            if left[begin:end] == right[begin:end]:
                continue
            token_stride = field["bytes"] // 130
            mismatch = next(index for index, pair in enumerate(zip(left[begin:end], right[begin:end])) if pair[0] != pair[1])
            logical_position = mismatch // token_stride
            classification = ("kv_append_codec_difference" if logical_position == POSITION
                              else "prior_cache_state_difference")
            detail = _difference(field, left, right)
            detail["logical_position"] = logical_position
            break
    if detail is None:
        for name, _, _ in STAGE_FIELDS[8:]:
            field = FIELD_BY_NAME[name]
            begin, end = field["offset"], field["offset"] + field["bytes"]
            if left[begin:end] != right[begin:end]:
                classification = CLASSIFICATION[name]
                detail = _difference(field, left, right)
                break
    return {
        "artifact_type": "ninfer_qwen3_layer3_attention_trace_comparison",
        "schema_version": 1, "diagnostic_only": True,
        "timing_evidence_eligible": False, "production_routing_authorized": False,
        "classification": classification, "first_difference": detail,
        "limitations": [
            "localizes the first selected-column layer-3 attention owner but is not a numerical oracle",
            "the trace is synchronous eager-only diagnostic evidence and cannot support timing",
        ],
    }


def _write_new(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--dflash", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_new(args.output, analyze(args.ordinary, args.dflash))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
