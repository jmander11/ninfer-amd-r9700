#!/usr/bin/env python3
"""Fail-closed comparison for paired eager-only Qwen3 layer-boundary traces."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path

HIDDEN = 5120
LAYERS = 64
SNAPSHOTS = 129
SNAPSHOT_BYTES = HIDDEN * 2
PAYLOAD_BYTES = SNAPSHOTS * SNAPSHOT_BYTES
ROLES = {
    "target": (
        ("target-ordinary-frontier130", 1, 0, 130, 129),
        ("target-dflash-frontier130-column0", 5, 0, 130, 129),
    ),
    "text": (
        ("text-fresh-frontier129-column128", 129, 128, 129, 128),
        ("text-append-frontier129-column0", 1, 0, 129, 128),
    ),
}
MANIFEST_KEYS = {
    "artifact_type", "schema_version", "diagnostic_only", "timing_evidence_eligible",
    "production_routing_authorized", "execution", "role", "width", "selected_column",
    "absolute_frontier", "token", "cache_position", "rope_position", "hidden", "layers",
    "snapshot_count", "snapshot_bytes", "sidecar_path", "sidecar_bytes", "sidecar_fnv1a64",
    "layout",
}
GDN_FIELDS = (
    ("h", "bf16", 5120, 0, 10240),
    ("g", "fp32", 48, 10240, 192),
    ("beta", "fp32", 48, 10432, 192),
    ("z", "bf16", 6144, 10624, 12288),
    ("q", "bf16", 2048, 22912, 4096),
    ("k", "bf16", 2048, 27008, 4096),
    ("v", "bf16", 6144, 31104, 12288),
    ("o", "bf16", 6144, 43392, 12288),
    ("on", "bf16", 6144, 55680, 12288),
    ("x", "bf16", 5120, 67968, 10240),
)
GDN_PAYLOAD_BYTES = 78208
RECURRENT_STATE_ELEMENTS = 128 * 128 * 48
RECURRENT_STATE_BYTES = RECURRENT_STATE_ELEMENTS * 4
GDN_MANIFEST_KEYS = {
    "artifact_type", "schema_version", "diagnostic_only", "timing_evidence_eligible",
    "production_routing_authorized", "execution", "role", "width", "selected_column",
    "absolute_frontier", "token", "cache_position", "rope_position", "text_layer",
    "gdn_index", "sidecar_path", "sidecar_bytes", "sidecar_fnv1a64", "fields", "layout",
}
GDN_CLASSIFICATIONS = {
    "h": "first_difference_gdn_input_rmsnorm",
    "g": "first_difference_gdn_g_control",
    "beta": "first_difference_gdn_beta_control",
    "z": "first_difference_gdn_output_gate_projection",
    "q": "first_difference_gdn_query_projection_or_conv",
    "k": "first_difference_gdn_key_projection_or_conv",
    "v": "first_difference_gdn_value_projection_or_conv",
    "o": "first_difference_gdn_recurrence",
    "on": "first_difference_gdn_gated_rmsnorm",
    "x": "first_difference_gdn_output_projection_or_residual",
}
RECURRENT_STATE_MANIFEST_KEYS = {
    "artifact_type", "schema_version", "diagnostic_only", "timing_evidence_eligible",
    "production_routing_authorized", "execution", "role", "capture_point",
    "selected_token", "selected_cache_position", "selected_rope_position", "state_frontier",
    "linear_state_slot", "text_layer", "gdn_index", "dtype", "shape", "elements",
    "sidecar_path", "sidecar_bytes", "sidecar_fnv1a64", "layout",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def exact_json_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_value(left, right) for left, right in zip(actual, expected))
    return actual == expected


def fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def decode_gdn_values(data: bytes, dtype: str,
                      elements: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if dtype == "bf16":
        bits = struct.unpack(f"<{elements}H", data)
        values = tuple(struct.unpack("<f", struct.pack("<I", value << 16))[0]
                       for value in bits)
    elif dtype == "fp32":
        bits = struct.unpack(f"<{elements}I", data)
        values = tuple(struct.unpack("<f", struct.pack("<I", value))[0] for value in bits)
    else:
        fail(f"unsupported GDN dtype: {dtype}")
    if not all(math.isfinite(value) for value in values):
        fail(f"nonfinite represented {dtype} GDN value")
    return bits, values


def load(path: Path, expected: tuple[str, int, int, int, int]) -> tuple[dict, bytes]:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        fail(f"manifest keys differ: {path}")
    role, width, column, frontier, position = expected
    exact = {
        "artifact_type": "ninfer_qwen3_layer_boundary_trace",
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_evidence_eligible": False,
        "production_routing_authorized": False,
        "execution": "eager",
        "role": role,
        "width": width,
        "selected_column": column,
        "absolute_frontier": frontier,
        "cache_position": position,
        "rope_position": position,
        "hidden": HIDDEN,
        "layers": LAYERS,
        "snapshot_count": SNAPSHOTS,
        "snapshot_bytes": SNAPSHOT_BYTES,
        "sidecar_bytes": PAYLOAD_BYTES,
        "layout": "little-endian-u16: input, then layer0..63 post_mixer,post_mlp",
    }
    for key, expected_value in exact.items():
        if value.get(key) != expected_value:
            fail(f"manifest field differs: {path}: {key}")
    token = integer(value.get("token"), "token")
    if not 0 <= token < 248077:
        fail(f"token is outside the Text token domain: {path}")
    sidecar = Path(value.get("sidecar_path")) if isinstance(value.get("sidecar_path"), str) else None
    expected_sidecar = path.with_suffix(".bin")
    if (sidecar is None or sidecar != expected_sidecar or not sidecar.is_absolute() or
            sidecar.is_symlink() or not sidecar.is_file()):
        fail(f"sidecar identity is invalid: {path}")
    data = sidecar.read_bytes()
    if len(data) != PAYLOAD_BYTES or value.get("sidecar_fnv1a64") != fnv1a64(data):
        fail(f"sidecar bytes/hash differ: {path}")
    return value, data


def load_gdn(path: Path, expected: tuple[str, int, int, int, int]) -> tuple[dict, bytes]:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict) or set(value) != GDN_MANIFEST_KEYS:
        fail(f"GDN manifest keys differ: {path}")
    role, width, column, frontier, position = expected
    exact = {
        "artifact_type": "ninfer_qwen3_layer1_gdn_detail_trace",
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_evidence_eligible": False,
        "production_routing_authorized": False,
        "execution": "eager",
        "role": role,
        "width": width,
        "selected_column": column,
        "absolute_frontier": frontier,
        "cache_position": position,
        "rope_position": position,
        "text_layer": 1,
        "gdn_index": 1,
        "sidecar_bytes": GDN_PAYLOAD_BYTES,
        "layout": "typed selected-column layer1 GDN boundaries in field order",
    }
    for key, expected_value in exact.items():
        if value.get(key) != expected_value:
            fail(f"GDN manifest field differs: {path}: {key}")
    token = integer(value.get("token"), "token")
    if not 0 <= token < 248077:
        fail(f"token is outside the Text token domain: {path}")
    expected_fields = [
        {"name": name, "dtype": dtype, "elements": elements, "offset": offset,
         "bytes": byte_count}
        for name, dtype, elements, offset, byte_count in GDN_FIELDS
    ]
    if value.get("fields") != expected_fields:
        fail(f"GDN field layout differs: {path}")
    sidecar = Path(value.get("sidecar_path")) if isinstance(value.get("sidecar_path"), str) else None
    expected_sidecar = path.with_suffix(".bin")
    if (sidecar is None or sidecar != expected_sidecar or not sidecar.is_absolute() or
            sidecar.is_symlink() or not sidecar.is_file()):
        fail(f"GDN sidecar identity is invalid: {path}")
    data = sidecar.read_bytes()
    if len(data) != GDN_PAYLOAD_BYTES or value.get("sidecar_fnv1a64") != fnv1a64(data):
        fail(f"GDN sidecar bytes/hash differ: {path}")
    return value, data


def load_recurrent_state(path: Path, role: str) -> tuple[dict, bytes]:
    if role not in {"text-fresh-frontier129-column128", "text-append-frontier129-column0"}:
        fail("unknown recurrent-state role")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict) or set(value) != RECURRENT_STATE_MANIFEST_KEYS:
        fail(f"recurrent-state manifest keys differ: {path}")
    point = ("wide-prefill-state-after-prefix-128-before-selected-column-128"
             if role == "text-fresh-frontier129-column128" else
             "restored-append-state-before-selected-column-0")
    layer = integer(value.get("text_layer"), "recurrent-state text_layer")
    if layer not in (0, 1) or value.get("gdn_index") != layer:
        fail(f"recurrent-state layer/GDN index differs: {path}")
    exact = {
        "artifact_type": "ninfer_qwen3_gdn_recurrent_state_trace",
        "schema_version": 2, "diagnostic_only": True, "timing_evidence_eligible": False,
        "production_routing_authorized": False, "execution": "eager", "role": role,
        "capture_point": point, "selected_token": 24178, "selected_cache_position": 128,
        "selected_rope_position": 128, "state_frontier": 128, "linear_state_slot": 0,
        "text_layer": layer, "gdn_index": layer, "dtype": "fp32", "shape": [128,128,48],
        "elements": RECURRENT_STATE_ELEMENTS, "sidecar_bytes": RECURRENT_STATE_BYTES,
        "layout": "little-endian-fp32:key,value,value_head",
    }
    for key, expected in exact.items():
        if not exact_json_value(value.get(key), expected):
            fail(f"recurrent-state manifest field differs: {path}: {key}")
    sidecar = Path(value.get("sidecar_path")) if isinstance(value.get("sidecar_path"), str) else None
    expected_sidecar = path.with_suffix(".bin")
    if (sidecar is None or sidecar != expected_sidecar or not sidecar.is_absolute() or
            sidecar.is_symlink() or not sidecar.is_file()):
        fail(f"recurrent-state sidecar identity is invalid: {path}")
    data = sidecar.read_bytes()
    if len(data) != RECURRENT_STATE_BYTES or value.get("sidecar_fnv1a64") != fnv1a64(data):
        fail(f"recurrent-state sidecar bytes/hash differ: {path}")
    values = struct.unpack(f"<{RECURRENT_STATE_ELEMENTS}f", data)
    if not all(math.isfinite(item) for item in values):
        fail(f"nonfinite recurrent-state value: {path}")
    return value, data


def boundary(index: int) -> tuple[int | None, str]:
    if index == 0:
        return None, "input_embedding"
    layer = (index - 1) // 2
    return layer, "post_mixer" if index % 2 else "post_mlp"


def compare(left_path: Path, right_path: Path, diagnostic: str) -> dict:
    if diagnostic not in ROLES:
        fail("unknown diagnostic")
    left, left_data = load(left_path, ROLES[diagnostic][0])
    right, right_data = load(right_path, ROLES[diagnostic][1])
    for key in ("absolute_frontier", "token", "cache_position", "rope_position", "hidden", "layers"):
        if left[key] != right[key]:
            fail(f"paired semantic identity differs: {key}")

    first = None
    for snapshot in range(SNAPSHOTS):
        begin = snapshot * SNAPSHOT_BYTES
        end = begin + SNAPSHOT_BYTES
        if left_data[begin:end] != right_data[begin:end]:
            first = snapshot
            break
    if first is None:
        classification = "transformer_residual_stack_exact"
        detail = None
    else:
        begin = first * SNAPSHOT_BYTES
        left_values = struct.unpack_from(f"<{HIDDEN}H", left_data, begin)
        right_values = struct.unpack_from(f"<{HIDDEN}H", right_data, begin)
        mismatches = [index for index, pair in enumerate(zip(left_values, right_values))
                      if pair[0] != pair[1]]
        layer, half = boundary(first)
        classification = ("represented_input_differs" if first == 0 else
                          f"first_visible_{half}_layer_{layer}")
        index = mismatches[0]
        detail = {
            "snapshot_index": first,
            "layer": layer,
            "half": half,
            "first_hidden_index": index,
            "mismatch_count": len(mismatches),
            "left_bf16_bits": left_values[index],
            "right_bf16_bits": right_values[index],
            "preceding_boundary_exact": first == 0 or
                left_data[(first - 1) * SNAPSHOT_BYTES:first * SNAPSHOT_BYTES] ==
                right_data[(first - 1) * SNAPSHOT_BYTES:first * SNAPSHOT_BYTES],
        }
        if not detail["preceding_boundary_exact"]:
            fail("first differing boundary does not have an exact predecessor")
    return {
        "artifact_type": "ninfer_qwen3_layer_boundary_comparison",
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_evidence_eligible": False,
        "production_routing_authorized": False,
        "diagnostic": diagnostic,
        "classification": classification,
        "first_difference": detail,
        "limitations": [
            "localizes the first visible represented-BF16 residual boundary, not the primitive root cause",
            "an exact transformer residual stack leaves final RMSNorm and LM head untested",
        ],
    }


def compare_gdn(left_path: Path, right_path: Path, diagnostic: str) -> dict:
    if diagnostic not in ROLES:
        fail("unknown diagnostic")
    left, left_data = load_gdn(left_path, ROLES[diagnostic][0])
    right, right_data = load_gdn(right_path, ROLES[diagnostic][1])
    for key in ("absolute_frontier", "token", "cache_position", "rope_position",
                "text_layer", "gdn_index"):
        if left[key] != right[key]:
            fail(f"paired GDN semantic identity differs: {key}")

    detail = None
    classification = "layer1_gdn_boundaries_exact"
    for field_index, (name, dtype, elements, offset, byte_count) in enumerate(GDN_FIELDS):
        left_field = left_data[offset:offset + byte_count]
        right_field = right_data[offset:offset + byte_count]
        left_values, left_decoded = decode_gdn_values(left_field, dtype, elements)
        right_values, right_decoded = decode_gdn_values(right_field, dtype, elements)
        if left_field == right_field:
            continue
        mismatches = [index for index, pair in enumerate(zip(left_values, right_values))
                      if pair[0] != pair[1]]
        first = mismatches[0]
        classification = GDN_CLASSIFICATIONS[name]
        detail = {
            "field": name,
            "dtype": dtype,
            "first_element_index": first,
            "mismatch_count": len(mismatches),
            "left_bits": left_values[first],
            "right_bits": right_values[first],
            "left_value": left_decoded[first],
            "right_value": right_decoded[first],
            "maximum_absolute_difference": max(
                abs(left_value - right_value)
                for left_value, right_value in zip(left_decoded, right_decoded)
            ),
            "preceding_fields_exact": all(
                left_data[prior_offset:prior_offset + prior_bytes] ==
                right_data[prior_offset:prior_offset + prior_bytes]
                for _, _, _, prior_offset, prior_bytes in GDN_FIELDS[:field_index]
            ),
        }
        if not detail["preceding_fields_exact"]:
            fail("first differing GDN field does not have exact predecessors")
        break
    return {
        "artifact_type": "ninfer_qwen3_layer1_gdn_detail_comparison",
        "schema_version": 1,
        "diagnostic_only": True,
        "timing_evidence_eligible": False,
        "production_routing_authorized": False,
        "diagnostic": diagnostic,
        "classification": classification,
        "first_difference": detail,
        "limitations": [
            "captures one selected column and does not capture persistent convolution or recurrence state",
            "the first differing field localizes a stage boundary, not necessarily the primitive operation",
        ],
    }


def compare_recurrent_state(left_path: Path, right_path: Path) -> dict:
    left, left_data = load_recurrent_state(left_path, "text-fresh-frontier129-column128")
    right, right_data = load_recurrent_state(right_path, "text-append-frontier129-column0")
    if left["text_layer"] != right["text_layer"] or left["gdn_index"] != right["gdn_index"]:
        fail("paired recurrent-state layer identity differs")
    layer = left["text_layer"]
    left_bits = struct.unpack(f"<{RECURRENT_STATE_ELEMENTS}I", left_data)
    right_bits = struct.unpack(f"<{RECURRENT_STATE_ELEMENTS}I", right_data)
    mismatches = [index for index, pair in enumerate(zip(left_bits, right_bits))
                  if pair[0] != pair[1]]
    detail = None
    classification = f"layer{layer}_recurrent_prefix_state_exact"
    if mismatches:
        first = mismatches[0]
        left_values = struct.unpack(f"<{RECURRENT_STATE_ELEMENTS}f", left_data)
        right_values = struct.unpack(f"<{RECURRENT_STATE_ELEMENTS}f", right_data)
        classification = f"first_difference_layer{layer}_recurrent_prefix_state"
        detail = {
            "first_element_index": first, "mismatch_count": len(mismatches),
            "left_bits": left_bits[first], "right_bits": right_bits[first],
            "left_value": left_values[first], "right_value": right_values[first],
            "maximum_absolute_difference": max(abs(a - b)
                                                for a, b in zip(left_values, right_values)),
        }
    return {
        "artifact_type": "ninfer_qwen3_gdn_recurrent_state_comparison",
        "schema_version": 2, "diagnostic_only": True, "timing_evidence_eligible": False,
        "production_routing_authorized": False, "diagnostic": "text-prefix-state",
        "text_layer": layer, "gdn_index": layer, "classification": classification,
        "first_difference": detail,
        "limitations": [
            "compares one exact GDN FP32 state frontier and does not localize an earlier update",
            "does not authorize production routing or a performance claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--diagnostic", choices=sorted(ROLES), required=True)
    parser.add_argument("--gdn-detail", action="store_true")
    parser.add_argument("--recurrent-state", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        fail("comparison refuses to overwrite output")
    if args.gdn_detail and args.recurrent_state:
        fail("comparison mode is ambiguous")
    if args.recurrent_state:
        if args.diagnostic != "text":
            fail("recurrent-state comparison requires the Text diagnostic")
        result = compare_recurrent_state(args.left, args.right)
    else:
        result = (compare_gdn(args.left, args.right, args.diagnostic) if args.gdn_detail else
                  compare(args.left, args.right, args.diagnostic))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(args.out, flags, 0o644)
    with os.fdopen(descriptor, "w") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
