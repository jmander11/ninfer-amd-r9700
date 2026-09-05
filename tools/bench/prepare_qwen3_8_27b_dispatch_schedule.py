#!/usr/bin/env python3
"""Prepare a source/artifact-derived static dispatch schedule for selected P2048."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

from tools.artifact.container import Artifact, TensorObject
from tools.bench.reconcile_qwen3_8_27b_dispatches import (
    SCHEDULE_TYPE,
    _publish,
    _snapshot,
    _validate_trace,
)


ROOT = Path(__file__).resolve().parents[2]
_PYTHON_AUTHORITIES = (
    Path(__file__).resolve(),
    ROOT / "tools/bench/reconcile_qwen3_8_27b_dispatches.py",
    ROOT / "tools/artifact/container.py",
    ROOT / "tools/artifact/layouts.py",
    ROOT / "tools/artifact/numeric.py",
)
_CPP_AUTHORITY_ROOTS = (
    ROOT / "src/targets/qwen3_8_27b/impl/config.h",
    ROOT / "src/targets/qwen3_8_27b/impl/variant.cpp",
    ROOT / "src/targets/qwen3_8_27b/impl/r9700_full_attention.hip",
    ROOT / "src/targets/qwen3_8_27b/impl/load/bindings.cpp",
    ROOT / "src/targets/qwen3/impl/runtime/text_context_impl.h",
    ROOT / "src/ops/r9700/linear/linear_tensor_op.cpp",
    ROOT / "src/ops/r9700/linear/linear_execution.hip",
    ROOT / "src/ops/r9700/linear/fp8_activation.hip",
    ROOT / "src/ops/r9700/linear/r9700_linear.hip",
    ROOT / "src/ops/r9700/linear/r9700_q4_activation_profile.h",
    ROOT / "src/ops/r9700/linear/r9700_w8_activation_profile.h",
    ROOT / "src/ops/r9700/kv/r9700_attention_profile.h",
    ROOT / "src/ops/r9700/kv/fp8_int4_kv_attention.h",
    ROOT / "src/ops/r9700/kv/fp8_int4_kv_attention.hip",
    ROOT / "src/ops/r9700/kv/fp8_int4_kv_xattention.hip",
    ROOT / "src/ops/r9700/eager/eager_ops.hip",
    ROOT / "src/ops/r9700/gdn/gdn_ops.hip",
    ROOT / "src/ops/r9700/gdn_recurrence/gated_delta_net.hip",
)
_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def _recursive_source_authorities() -> tuple[Path, ...]:
    pending = list(_CPP_AUTHORITY_ROOTS)
    resolved: set[Path] = set(_PYTHON_AUTHORITIES)
    while pending:
        source = pending.pop().resolve(strict=True)
        if source in resolved:
            continue
        resolved.add(source)
        for include in _QUOTED_INCLUDE.findall(source.read_text(encoding="utf-8")):
            candidates = (
                source.parent / include, ROOT / "src" / include,
                ROOT / "include" / include, ROOT / "bench" / include, ROOT / include,
            )
            dependency = next((candidate for candidate in candidates if candidate.is_file()), None)
            if dependency is not None and dependency.resolve() not in resolved:
                pending.append(dependency)
    return tuple(sorted(resolved))


SOURCE_AUTHORITIES = _recursive_source_authorities()

MIXER = re.compile(
    r"^ninfer\.(attention|gdn)\.prefill\.(attention|gdn) payload=(\d+)$")
POST = re.compile(r"^ninfer\.post-mixer\.prefill\.post_mixer payload=(\d+)$")
Q4_LINEAR = "a8q4g64_linear_prefill_cta_kernel"
W8_LINEAR = "a8w8g32_linear_prefill_cta_kernel"
Q4_QUANT = "a8g64_quantize_activation_kernel"
W8_QUANT = "a8g32_quantize_activation_kernel"
DENSE = "fused_attention_causal_kernel"
FP8_QUANT = "fp8_quantize_activation_kernel"
FP8_LINEAR = "Cijk_Alik_Bljk_F8BS_"
FP8_POISON = "poison_nonfinite_output"
XATT_PACK = "xattention_pack_keys_kernel"
XATT_RANK = "xattention_rank_kernel"
XATT_CONSUMER = "xattention_flash_consumer_kernel"


def _source_marker(layer: int) -> str:
    family = "attention" if layer >= 3 and (layer - 3) % 4 == 0 else "gdn"
    return f"ninfer.{family}.prefill.{family} payload={layer}"


def _post_marker(layer: int) -> str:
    return f"ninfer.post-mixer.prefill.post_mixer payload={layer}"


def _chunks(chunk: int) -> list[int]:
    return [min(chunk, 2048 - begin) for begin in range(0, 2048, chunk)]


def _tensor(tensors: dict[str, tuple[tuple[int, ...], str]], name: str,
            shape: tuple[int, int]) -> str:
    value = tensors.get(name)
    if value is None or value[0] != shape or value[1] not in (
            "Q4G64_F16S", "W8G32_F16S", "F8E4M3_ROW_F32S"):
        raise ValueError(f"selected artifact has no supported exact matrix {name} {shape}")
    return value[1]


def _linear_kind(symbol: str) -> str | None:
    if Q4_LINEAR in symbol:
        return "Q4G64_F16S"
    if W8_LINEAR in symbol:
        return "W8G32_F16S"
    return None


def _quant_kind(symbol: str) -> str | None:
    if Q4_QUANT in symbol:
        return "Q4G64_F16S"
    if W8_QUANT in symbol:
        return "W8G32_F16S"
    return None


def _is_fp8_linear(symbol: str) -> bool:
    return FP8_LINEAR in symbol


def _assign_linear_roles(assignments: dict[int, dict[str, Any]], rows: list[dict[str, Any]],
                         indices: list[int], marker: str, roles: list[dict[str, Any]],
                         tokens: int) -> list[list[int]]:
    """Bind every projection from artifact format and exact in-marker launch order."""
    related = [index for index in indices if (
        _linear_kind(rows[index]["symbol"]) is not None
        or _quant_kind(rows[index]["symbol"]) is not None
        or FP8_QUANT in rows[index]["symbol"]
        or _is_fp8_linear(rows[index]["symbol"])
        or FP8_POISON in rows[index]["symbol"])]
    cursor = 0
    groups: list[list[int]] = []
    for role in roles:
        storage = role["storage"]
        count = 3 if storage == "F8E4M3_ROW_F32S" else 2
        group = related[cursor:cursor + count]
        cursor += count
        if len(group) != count:
            raise ValueError(f"{marker} lacks the complete {role['role']} launch sequence")
        symbols = [rows[index]["symbol"] for index in group]
        parameters = {"tokens": tokens, "rows": role["rows"], "columns": role["columns"]}
        if storage == "F8E4M3_ROW_F32S":
            if not (FP8_QUANT in symbols[0] and _is_fp8_linear(symbols[1])
                    and FP8_POISON in symbols[2]):
                raise ValueError(f"{marker} {role['role']} is not FP8 quantize/matrix/poison")
            _require_geometry(rows[group[0]], marker, role["role"] + " activation quantize",
                              {"x": tokens, "y": 1, "z": 1}, 256)
            _require_geometry(rows[group[2]], marker, role["role"] + " nonfinite poison",
                              {"x": 1, "y": 1, "z": 1}, 256)
            assignments[group[0]] = {
                "classification": "modeled", "role": role["role"] + "_activation_quantize",
                "operation": "fp8_quantize", "format": "bf16_to_fp8e4m3_row_f32scale",
                "parameters": {"tokens": tokens, "columns": role["columns"]},
            }
            assignments[group[1]] = {
                "classification": "modeled", "role": role["role"],
                "operation": "fp8_linear", "format": "fp8e4m3_row_scaled_bf16_output",
                "parameters": parameters,
            }
            assignments[group[2]] = {
                "classification": "modeled", "role": role["role"] + "_nonfinite_poison",
                "operation": "fp8_nonfinite_poison", "format": "bf16_output_status",
                "parameters": {"elements": tokens * role["rows"]},
            }
        else:
            if (_quant_kind(symbols[0]) != storage
                    or _linear_kind(symbols[1]) != storage):
                raise ValueError(f"{marker} {role['role']} kernel format differs from artifact")
            operation = "q4_linear" if storage == "Q4G64_F16S" else "w8_linear"
            group_size = 64 if operation == "q4_linear" else 32
            _require_geometry(rows[group[0]], marker, role["role"] + " activation quantize",
                              {"x": (role["columns"] // group_size) * 32,
                               "y": tokens, "z": 1}, 32)
            _require_geometry(rows[group[1]], marker, role["role"],
                              {"x": math.ceil(role["rows"] / 64) * 512,
                               "y": math.ceil(tokens / 64), "z": 1}, 512)
            assignments[group[0]] = {
                "classification": "modeled", "role": role["role"] + "_activation_quantize",
                "operation": "q4_quantize" if operation == "q4_linear" else "w8_quantize",
                "format": "bf16_to_a8g64" if operation == "q4_linear" else "bf16_to_a8g32",
                "parameters": {"tokens": tokens, "columns": role["columns"],
                               "group": group_size},
            }
            assignments[group[1]] = {
                "classification": "modeled", "role": role["role"],
                "operation": operation,
                "format": ("a8q4g64_bf16_output" if operation == "q4_linear"
                           else "a8w8g32_bf16_output"),
                "parameters": parameters,
            }
        groups.append(group)
    if cursor != len(related):
        raise ValueError(f"{marker} has extra or ambiguous low-precision linear dispatches")
    return groups


def _roles(layer: int, post: bool,
           tensors: dict[str, tuple[tuple[int, ...], str]]) -> list[dict[str, Any]]:
    prefix = f"text/layers/{layer}"
    if post:
        specs = (
            ("mlp_gate_up", f"{prefix}/mlp/gate_up", (34816, 5120)),
            ("mlp_down", f"{prefix}/mlp/down", (5120, 17408)),
        )
    elif layer >= 3 and (layer - 3) % 4 == 0:
        specs = (
            ("attention_query_key", f"{prefix}/attention/query_key", (7168, 5120)),
            ("attention_gate_value", f"{prefix}/attention/gate_value", (7168, 5120)),
            ("attention_output", f"{prefix}/attention/output", (5120, 6144)),
        )
    else:
        specs = (
            ("gdn_query_key", f"{prefix}/gdn/query_key", (4096, 5120)),
            ("gdn_value_z", f"{prefix}/gdn/value_z", (12288, 5120)),
            ("gdn_output", f"{prefix}/gdn/output", (5120, 6144)),
        )
    return [
        {"role": role, "tensor": name, "rows": shape[0], "columns": shape[1],
         "storage": _tensor(tensors, name, shape)}
        for role, name, shape in specs
    ]


def _marker_blocks(rows: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
    relevant = {_source_marker(layer) for layer in range(64)} | {
        _post_marker(layer) for layer in range(64)}
    blocks: list[tuple[str, list[int]]] = []
    for index, row in enumerate(rows):
        marker = row["roctx_region"]
        if marker not in relevant:
            continue
        if not blocks or blocks[-1][0] != marker:
            blocks.append((marker, []))
        blocks[-1][1].append(index)
    return blocks


def _assign_unique(assignments: dict[int, dict[str, Any]], rows: list[dict[str, Any]],
                   indices: list[int], needle: str, marker: str, role: str,
                   operation: str, tensor_format: str, parameters: dict[str, int],
                   expected_grid_x: int) -> int:
    def matches_symbol(symbol: str) -> bool:
        if needle == "rmsnorm_kernel":
            return re.search(r"(?:^|::)rmsnorm_kernel\(", symbol) is not None
        return needle in symbol

    matches = [index for index in indices if matches_symbol(rows[index]["symbol"])]
    if len(matches) != 1:
        raise ValueError(f"{marker} has {len(matches)} {needle} dispatches; expected exactly one")
    index = matches[0]
    if (rows[index]["grid"] != {"x": expected_grid_x, "y": 1, "z": 1}
            or rows[index]["workgroup"] != {"x": 256, "y": 1, "z": 1}):
        raise ValueError(f"{marker} {role} launch geometry differs from the source profile")
    assignments[index] = {
        "classification": "modeled", "role": role, "operation": operation,
        "format": tensor_format, "parameters": parameters,
    }
    return index


def _require_geometry(row: dict[str, Any], marker: str, role: str,
                      grid: dict[str, int], workgroup_x: int) -> None:
    if (row["grid"] != grid
            or row["workgroup"] != {"x": workgroup_x, "y": 1, "z": 1}):
        raise ValueError(f"{marker} {role} launch geometry differs from the source profile")


def build_schedule(trace: dict[str, Any], rows: list[dict[str, Any]],
                   tensors: dict[str, tuple[tuple[int, ...], str]],
                   sources: list[dict[str, Any]]) -> dict[str, Any]:
    workload = trace["workload"]
    if (workload.get("concurrency"), workload.get("prompt_tokens"),
        workload.get("generated_tokens"), workload.get("spec"),
        workload.get("draft_tokens")) != (1, 2048, 0, "none", 0):
        raise ValueError("schedule producer supports only selected C1 P2048 spec-none trace")
    if workload.get("xattention_profile") not in ("dense", "b128-s16-tau900"):
        raise ValueError("schedule producer received an unsupported attention profile")
    chunk = workload["prefill_chunk"]
    lengths = _chunks(chunk)
    expected = [marker for _ in lengths for layer in range(64)
                for marker in (_source_marker(layer), _post_marker(layer))]
    blocks = _marker_blocks(rows)
    actual = [marker for marker, _ in blocks]
    if actual != expected:
        mismatch = next((i for i, pair in enumerate(zip(actual, expected))
                         if pair[0] != pair[1]), min(len(actual), len(expected)))
        raise ValueError(
            f"trace marker order is not the static Text Program at block {mismatch}; "
            f"observed {len(actual)} blocks, expected {len(expected)}")

    assignments: dict[int, dict[str, Any]] = {}
    for block_number, (marker, indices) in enumerate(blocks):
        chunk_index = block_number // 128
        layer = (block_number % 128) // 2
        post = block_number % 2 == 1
        tokens = lengths[chunk_index]
        roles = _roles(layer, post, tensors)
        linear_groups = _assign_linear_roles(
            assignments, rows, indices, marker, roles, tokens)
        if not post and layer >= 3 and (layer - 3) % 4 == 0:
            context_start = sum(lengths[:chunk_index])
            context = context_start + tokens
            mapped_pages = math.ceil(context / 64)
            metadata = 4 * (tokens + mapped_pages)
            if workload["xattention_profile"] == "dense":
                dense = [i for i in indices if DENSE in rows[i]["symbol"]]
                if len(dense) != 1:
                    raise ValueError(f"{marker} chunk {chunk_index} has {len(dense)} dense consumers")
                dense_group = re.search(
                    r"fused_attention_causal_kernel<(16|32)u,", rows[dense[0]]["symbol"])
                if dense_group is None or int(dense_group.group(1)) != workload["kv_value_group"]:
                    raise ValueError(f"{marker} dense consumer template group differs from selected route")
                _require_geometry(rows[dense[0]], marker, "full_attention",
                                  {"x": 24 * 256, "y": tokens, "z": 1}, 256)
                assignments[dense[0]] = {
                    "classification": "modeled", "role": "full_attention",
                    "operation": "dense_attention",
                    "format": "bf16q_fp8k_int4v_fp16scale_fp32_output",
                    "parameters": {"tokens": tokens, "context_start": context_start,
                                   "query_heads": 24, "kv_heads": 4, "head_dimension": 256,
                                   "value_group": workload["kv_value_group"],
                                   "represented_metadata_bytes": metadata,
                                   "source_metadata_bytes": metadata},
                }
                attention_indices = dense
            else:
                sparse = []
                for needle, role_name, operation, tensor_format, grid in (
                    (XATT_PACK, "xattention_key_pack", "sparse_pack",
                     "fp8k_to_bf16_packed_k", {"x": 4, "y": mapped_pages, "z": 1}),
                    (XATT_RANK, "xattention_rank", "sparse_rank",
                     "bf16q_bf16packedk_fp32_rank",
                     {"x": 24, "y": math.ceil(tokens / 128), "z": 1}),
                    (XATT_CONSUMER, "xattention_consumer", "sparse_consumer",
                     "bf16q_bf16packedk_int4v_fp16scale_fp32_output",
                     {"x": 24, "y": math.ceil(tokens / 16), "z": 1}),
                ):
                    found = [i for i in indices if needle in rows[i]["symbol"]]
                    if len(found) != 1:
                        raise ValueError(f"{marker} chunk {chunk_index} has {len(found)} {role_name} dispatches")
                    _require_geometry(rows[found[0]], marker, role_name, grid, 256)
                    assignments[found[0]] = {
                        "classification": "modeled", "role": role_name,
                        "operation": operation, "format": tensor_format,
                        "parameters": {"tokens": tokens, "context_start": context_start,
                                       "context": context, "query_heads": 24,
                                       "kv_heads": 4, "head_dimension": 256,
                                       "value_group": workload["kv_value_group"],
                                       "find_block": 128, "stride": 16,
                                       "tau_permille": 900},
                    }
                    sparse.append(found[0])
                attention_indices = sparse
        if post:
            rms_index = _assign_unique(assignments, rows, indices, "rmsnorm_kernel", marker,
                           "post_mixer_rmsnorm", "rmsnorm", "bf16_io",
                           {"elements": 5120 * tokens, "features": 5120}, tokens * 256)
            silu_index = _assign_unique(assignments, rows, indices, "silu_mul_strided_kernel", marker,
                           "post_mixer_silu_mul", "silu_mul", "bf16_io",
                           {"elements": 17408 * tokens}, 17408 * tokens)
            residual_index = _assign_unique(assignments, rows, indices, "residual_add_kernel", marker,
                           "post_mixer_residual_add", "residual_add", "bf16_io",
                           {"elements": 5120 * tokens}, 5120 * tokens)
            source_order = [rms_index, *linear_groups[0], silu_index,
                            *linear_groups[1], residual_index]
        elif layer < 3 or (layer - 3) % 4 != 0:
            rms_index = _assign_unique(assignments, rows, indices, "rmsnorm_kernel", marker,
                           "gdn_input_rmsnorm", "rmsnorm", "bf16_io",
                           {"elements": 5120 * tokens, "features": 5120}, tokens * 256)
            conv_index = _assign_unique(assignments, rows, indices, "causal_conv1d_silu_kernel", marker,
                           "gdn_causal_conv", "gdn_conv", "bf16_io",
                           {"tokens": tokens, "channels": 10240}, 10240)
            recurrence_index = _assign_unique(assignments, rows, indices, "ordinary_kernel<true>", marker,
                           "gdn_recurrence", "gdn_recurrence", "bf16_io_fp32_state",
                           {"tokens": tokens, "query_heads": 16, "value_heads": 48,
                            "dimension": 128, "row_tiles": 4}, 48 * 4 * 256)
            gated_norm_index = _assign_unique(
                assignments, rows, indices, "gated_rmsnorm_kernel", marker,
                "gdn_gated_rmsnorm", "gated_rmsnorm", "bf16_io",
                {"elements": tokens * 48 * 128, "features": 128},
                tokens * 48 * 256)
            residual_index = _assign_unique(assignments, rows, indices, "residual_add_kernel", marker,
                           "gdn_output_residual_add", "residual_add", "bf16_io",
                           {"elements": 5120 * tokens}, 5120 * tokens)
            source_order = [rms_index, *linear_groups[0],
                            *linear_groups[1], conv_index, recurrence_index,
                            gated_norm_index,
                            *linear_groups[2], residual_index]
        else:
            rms_index = _assign_unique(assignments, rows, indices, "rmsnorm_kernel", marker,
                                       "attention_input_rmsnorm", "rmsnorm", "bf16_io",
                                       {"elements": 5120 * tokens, "features": 5120},
                                       tokens * 256)
            residual_index = _assign_unique(
                assignments, rows, indices, "residual_add_kernel", marker,
                "attention_output_residual_add", "residual_add", "bf16_io",
                {"elements": 5120 * tokens}, 5120 * tokens)
            source_order = [rms_index, *linear_groups[0], *linear_groups[1],
                            *attention_indices, *linear_groups[2],
                            residual_index]
        if source_order != sorted(source_order):
            raise ValueError(f"{marker} modeled dispatch order differs from the static Program")

    fp8_roles = [value["role"] for value in assignments.values()
                 if value.get("operation") == "fp8_linear"]
    allowed_fp8_roles = {
        "mlp_gate_up", "gdn_query_key", "attention_query_key", "attention_gate_value",
    }
    if fp8_roles and (set(fp8_roles) - allowed_fp8_roles
                      or len(fp8_roles) != 144 * len(lengths)):
        raise ValueError("selected artifact FP8 projection inventory is not exact four-role")
    route_artifact = trace.get("selected_route", {}).get("artifact", {})
    weights_id = route_artifact.get("weights_id") if isinstance(route_artifact, dict) else None
    hybrid = weights_id == "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
    if isinstance(weights_id, str) and bool(fp8_roles) != hybrid:
        raise ValueError("selected artifact FP8 dispatches differ from its weights profile")

    dispatches = []
    for index, row in enumerate(rows):
        marker = row["roctx_region"]
        mixer_match = MIXER.match(marker) if isinstance(marker, str) else None
        post_match = POST.match(marker) if isinstance(marker, str) else None
        matched = mixer_match or post_match
        layer = int(matched.group(3 if mixer_match else 1)) if matched else None
        entry = {
            "schedule_index": index, "marker": marker,
            "layer": {"kind": "text" if layer is not None else "orchestration",
                      "index": layer},
            "symbol": row["symbol"], "grid": row["grid"],
            "workgroup": row["workgroup"],
            "call": {"id": f"dispatch.{index}", "ordinal": index,
                     "dispatch_index": 0, "dispatch_count": 1},
        }
        entry.update(assignments.get(index, {
            "classification": "unmodeled",
            "role": "unassigned",
            "operation": "unmodeled_dispatch",
            "reason": "no exact source-derived logical roof model for this dispatch",
        }))
        dispatches.append(entry)
    return {
        "artifact_type": SCHEDULE_TYPE, "schema_version": 1,
        "workload": workload, "selected_route": trace["selected_route"],
        "dispatch_order": "trace_start_ns_then_dispatch_id",
        "sources": sources, "dispatches": dispatches,
    }


def prepare(trace_path: Path, artifact_path: Path, executable_path: Path,
            group: int, chunk: int) -> dict[str, Any]:
    trace_snapshot = _snapshot(trace_path, "validated trace")
    trace, rows = _validate_trace(trace_path)
    artifact_snapshot = _snapshot(artifact_path, "selected artifact")
    executable_snapshot = _snapshot(executable_path, "selected executable")
    route = trace["selected_route"]
    if (route["artifact"]["path"] != artifact_snapshot["path"]
            or route["artifact"]["sha256"] != artifact_snapshot["sha256"]
            or route["artifact"]["file_size_bytes"] != artifact_snapshot["file_size_bytes"]
            or route["benchmark_executable"] != executable_snapshot
            or trace["workload"]["kv_value_group"] != group
            or trace["workload"]["prefill_chunk"] != chunk):
        raise ValueError("explicit artifact/executable/group/chunk differ from validated trace")
    with Artifact.open(artifact_snapshot["path"]) as artifact:
        if artifact.identity.model_id != "qwen3.8-27b" or artifact.identity.weights_id != route[
                "artifact"]["weights_id"]:
            raise ValueError("selected artifact identity differs from trace")
        tensors = {obj.name: (obj.shape, obj.format) for obj in artifact.objects
                   if isinstance(obj, TensorObject)}
    sources = [_snapshot(path, f"source authority {path.name}") for path in SOURCE_AUTHORITIES]
    result = build_schedule(trace, rows, tensors, sources)
    missing_symbols: dict[str, int] = {}
    for row in rows:
        if row["roctx_region"] is None:
            missing_symbols[row["symbol"]] = missing_symbols.get(row["symbol"], 0) + 1
    result["unmarked_collision_evidence"] = {
        "dispatch_count": sum(missing_symbols.values()),
        "symbol_counts": dict(sorted(missing_symbols.items())),
        "disposition": "retained unmodeled; never assigned a role from symbol or time",
    }
    if _snapshot(trace_path, "validated trace") != trace_snapshot:
        raise ValueError("validated trace changed while preparing schedule")
    for source in sources:
        if _snapshot(Path(source["path"]), "source authority") != source:
            raise ValueError("source authority changed while preparing schedule")
    if (_snapshot(artifact_path, "selected artifact") != artifact_snapshot
            or _snapshot(executable_path, "selected executable") != executable_snapshot):
        raise ValueError("selected artifact or executable changed while preparing schedule")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-authority", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--kv-value-group", required=True, type=int, choices=(16, 32))
    parser.add_argument("--prefill-chunk", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    try:
        result = prepare(args.trace_authority, args.artifact, args.executable,
                         args.kv_value_group, args.prefill_chunk)
        _publish(args.out, result)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote source-derived dispatch schedule to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
