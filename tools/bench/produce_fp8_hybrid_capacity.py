#!/usr/bin/env python3
"""Produce or validate the CPU-only current hybrid P8192/G16 capacity report."""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.artifact.layouts import align_up, encoded_size
from tools.convert.qwen3_8_27b_r9700 import fp8_hybrid_inventory


SCHEMA = "ninfer.r9700.fp8-hybrid-current-capacity.v2"
HEADROOM = 1 << 30
PAGE_TOKENS = 64
PLANNER_FIELDS = (
    "target", "weights_profile", "capacity_tokens", "page_tokens", "prefill_chunk",
    "kv_value_group", "speculative_backend", "draft_tokens", "proposal_head",
    "device_graph", "concurrency", "minimum_groups", "maximum_groups",
    "minimum_sequence_bytes", "workspace_bytes", "graph_allowance_bytes",
    "request_transient_bytes", "minimum_reservation_bytes", "kv_payload_bytes",
    "kv_increment_bytes",
)
PLANNER_PROFILE = {
    "target": "qwen3_8_27b_r9700",
    "weights_profile": "R9700Q4G64Fp8FourRoleN16K16Evaluation",
    "capacity_tokens": "262144",
    "page_tokens": "64",
    "prefill_chunk": "8192",
    "kv_value_group": "16",
    "speculative_backend": "mtp",
    "draft_tokens": "3",
    "proposal_head": "optimized",
    "device_graph": "1",
}
PLANNER_INTEGER_FIELDS = PLANNER_FIELDS[2:6] + ("draft_tokens", "device_graph") + PLANNER_FIELDS[10:]


def _materialized_weights(*, mtp: bool, optimized_proposal: bool) -> tuple[int, int, int]:
    """Mirror the no-Vision target binder's feature-owned device placements."""

    cursor = 0
    payload = 0
    count = 0
    for spec in fp8_hybrid_inventory.TENSOR_SPECS:
        if spec.name.startswith("vision/"):
            continue
        if spec.name.startswith("mtp/") and not mtp:
            continue
        if spec.name.startswith("text/draft_head") and not optimized_proposal:
            continue
        encoded = encoded_size(spec.layout, spec.format, spec.shape)
        cursor = align_up(cursor, 256)
        cursor += encoded
        payload += encoded
        count += 1
    return cursor, payload, count


DENSE_FULL_HEAD_WEIGHT_BYTES, DENSE_FULL_HEAD_PAYLOAD_BYTES, DENSE_FULL_HEAD_TENSOR_COUNT = (
    _materialized_weights(mtp=False, optimized_proposal=False)
)
MTP3_OPTIMIZED_WEIGHT_BYTES, MTP3_OPTIMIZED_PAYLOAD_BYTES, MTP3_OPTIMIZED_TENSOR_COUNT = (
    _materialized_weights(mtp=True, optimized_proposal=True)
)


def _require_exact_fields(
    source: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    for key, value in expected.items():
        actual = source.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"weights report {label}.{key} differs from current authority")


def capacity_weights(source: Mapping[str, object]) -> tuple[int, int, dict[str, int]]:
    """Validate the dense observation and derive the MTP3 materialization exactly."""

    _require_exact_fields(
        source,
        {
            "artifact_type": "ninfer_bench_report",
            "schema_version": 20,
            "tool": "ninfer_bench",
        },
        "root",
    )
    config = source.get("config")
    load = source.get("load")
    memory = source.get("memory")
    environment = source.get("environment")
    if not all(isinstance(value, Mapping) for value in (config, load, memory, environment)):
        raise ValueError("weights report lacks config, load, memory, or environment authority")
    _require_exact_fields(
        load,
        {
            "target": fp8_hybrid_inventory.TARGET_KEY,
            "weights_id": fp8_hybrid_inventory.WEIGHTS_ID,
            "host_to_device_bytes": DENSE_FULL_HEAD_PAYLOAD_BYTES,
            "tensor_count": DENSE_FULL_HEAD_TENSOR_COUNT,
            "resource_count": len(fp8_hybrid_inventory.RESOURCE_SPECS),
        },
        "load",
    )
    _require_exact_fields(
        config,
        {
            "max_context": 2048,
            "concurrency": 1,
            "spec": "none",
            "draft_tokens": 0,
            "proposal_head": "full",
            "use_device_graph": True,
            "decode_path": "device_graph",
            "speculative_execution": False,
            "prefill_chunk": 4096,
            "kv_cache_format": "fp8-k-int4-v",
            "kv_value_group": 16,
            "kv_plane_layouts": {
                "key": "token-fastest-head-major",
                "value": "feature-fastest-page-major",
                "value_scale": "feature-fastest-page-major",
            },
            "q4_activation_bits": 8,
            "q4_prefill_cta_profile": "m64n128-pingpong-n16-k16-production",
            "xattention_qualification": False,
        },
        "config",
    )
    _require_exact_fields(memory, {"device": 0, "kv_cache_format": "fp8-k-int4-v"}, "memory")
    _require_exact_fields(
        environment,
        {"device_id": 0, "gpu_name": "AMD Radeon AI PRO R9700", "architecture_name": "gfx1201"},
        "environment",
    )
    weights_memory = memory.get("weights")
    if not isinstance(weights_memory, Mapping):
        raise ValueError("weights report lacks device weight allocation")
    for key in ("capacity_bytes", "used_bytes", "peak_used_bytes"):
        if type(weights_memory.get(key)) is not int:
            raise ValueError(f"weights report memory.weights.{key} is not an exact integer")
    observed = weights_memory["capacity_bytes"]
    if weights_memory["used_bytes"] != observed or weights_memory["peak_used_bytes"] != observed:
        raise ValueError("weights report device weight capacity, use, and peak differ")
    if observed != DENSE_FULL_HEAD_WEIGHT_BYTES:
        raise ValueError(
            "dense/full-head materialized weight bytes differ from the current hybrid inventory: "
            f"expected {DENSE_FULL_HEAD_WEIGHT_BYTES}, got {observed}"
        )
    available = memory.get("available_after_weights_bytes")
    if type(available) is not int:
        raise ValueError("weights report available_after_weights_bytes is not an exact integer")
    device = observed + available
    if device <= MTP3_OPTIMIZED_WEIGHT_BYTES:
        raise ValueError("device capacity does not exceed MTP3 materialized weight bytes")
    arithmetic = {
        "observed_dense_full_head_bytes": observed,
        "mtp3_optimized_bytes": MTP3_OPTIMIZED_WEIGHT_BYTES,
        "mtp3_optimized_payload_bytes": MTP3_OPTIMIZED_PAYLOAD_BYTES,
        "mtp3_optimized_tensor_count": MTP3_OPTIMIZED_TENSOR_COUNT,
        "mtp_and_optimized_draft_increment_bytes": (
            MTP3_OPTIMIZED_WEIGHT_BYTES - observed
        ),
    }
    return MTP3_OPTIMIZED_WEIGHT_BYTES, device, arithmetic


def _path_identity(path: Path) -> tuple[int, int]:
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"capacity input is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _open_regular(path: Path) -> tuple[int, tuple[int, int]]:
    source = path.expanduser().absolute()
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"capacity input is not a regular file: {source}") from error
        raise ValueError(f"capacity input cannot be opened safely: {source}: {error}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"capacity input is not a regular file: {source}")
    return descriptor, (metadata.st_dev, metadata.st_ino)


def _descriptor_bytes_and_sha256(descriptor: int) -> tuple[bytes, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks), digest.hexdigest()


def _unchanged(path: Path, descriptor: int, owner: tuple[int, int], digest: str) -> None:
    _, after = _descriptor_bytes_and_sha256(descriptor)
    if _path_identity(path.expanduser().absolute()) != owner or after != digest:
        raise ValueError(f"capacity input changed while in use: {path}")


def parse_planner(text: str) -> list[dict[str, int]]:
    reader = csv.DictReader(io.StringIO(text), strict=True)
    if tuple(reader.fieldnames or ()) != PLANNER_FIELDS:
        raise ValueError("planner capacity authority has an unexpected CSV header")
    raw_rows = list(reader)
    if len(raw_rows) != 4:
        raise ValueError("planner capacity authority lacks exact C1..4 rows")
    rows = []
    for raw in raw_rows:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("planner capacity authority contains a malformed row")
        for key, expected in PLANNER_PROFILE.items():
            if raw[key] != expected:
                raise ValueError(f"planner capacity authority {key} differs from the exact profile")
        for key in PLANNER_INTEGER_FIELDS:
            value = raw[key]
            if not value.isascii() or not value.isdecimal():
                raise ValueError(f"planner capacity authority {key} is not an exact nonnegative integer")
        rows.append({key: int(raw[key]) for key in PLANNER_FIELDS if key in PLANNER_INTEGER_FIELDS})
    if [row["concurrency"] for row in rows] != [1, 2, 3, 4]:
        raise ValueError("planner capacity authority lacks exact C1..4 rows")
    for row in rows:
        if row["minimum_groups"] != 4096 or row["kv_increment_bytes"] != 1_810_432:
            raise ValueError("planner page geometry differs from P262144/G16")
        if row["maximum_groups"] != row["concurrency"] * row["minimum_groups"]:
            raise ValueError("planner maximum groups differ from the concurrency envelope")
        if row["maximum_groups"] < row["minimum_groups"]:
            raise ValueError("planner maximum groups precede the minimum")
        expected_reservation = (
            row["minimum_sequence_bytes"] + row["workspace_bytes"]
            + row["graph_allowance_bytes"] + row["request_transient_bytes"]
        )
        if row["minimum_reservation_bytes"] != expected_reservation:
            raise ValueError("planner minimum reservation arithmetic is inconsistent")
        if row["kv_payload_bytes"] > row["minimum_sequence_bytes"]:
            raise ValueError("planner KV payload exceeds persistent sequence storage")
        if row["kv_increment_bytes"] <= 0:
            raise ValueError("planner KV increment is not positive")
    return rows


def assemble(rows: list[dict[str, int]], weights: int, device: int) -> dict[str, object]:
    if type(weights) is not int or type(device) is not int or weights < 0 or device < 0:
        raise ValueError("capacity weights and device bytes must be exact nonnegative integers")
    budget = device - weights - HEADROOM
    cells = []
    for row in rows:
        minimum = row["minimum_groups"]
        maximum = row["maximum_groups"]
        reservation = row["minimum_reservation_bytes"]
        increment = row["kv_increment_bytes"]
        if reservation > budget:
            groups = None
            resolved = None
            sequence = None
            slack = budget - reservation
        else:
            groups = min(maximum, minimum + (budget - reservation) // increment)
            resolved = reservation + (groups - minimum) * increment
            sequence = row["minimum_sequence_bytes"] + (groups - minimum) * increment
            slack = budget - resolved
        preserved = groups is not None and groups >= minimum and slack >= 0
        cells.append({**row, "resolved_groups": groups,
                      "aggregate_capacity_tokens": None if groups is None else groups * PAGE_TOKENS,
                      "resolved_sequence_bytes": sequence,
                      "resolved_reservation_bytes": resolved, "remaining_slack_bytes": slack,
                      "capacity_preserved": preserved})
    return {"schema": SCHEMA,
            "scope": {
                "target": "qwen3_8_27b_r9700",
                "weights_profile": "R9700Q4G64Fp8FourRoleN16K16Evaluation",
                "gpu_name": "AMD Radeon AI PRO R9700",
                "architecture_name": "gfx1201",
                "capacity_tokens": 262144,
                "page_tokens": PAGE_TOKENS,
                "prefill_chunk": 8192,
                "kv_cache_format": "fp8-k-int4-v",
                "kv_value_group": 16,
                "kv_plane_layouts": {
                    "key": "token-fastest-head-major",
                    "value": "feature-fastest-page-major",
                    "value_scale": "feature-fastest-page-major",
                },
                "speculative_backend": "mtp",
                "draft_tokens": 3,
                "proposal_head": "optimized",
                "device_graph": True,
                "dense_source_prefill_chunk": 4096,
                "dense_source_q4_activation_bits": 8,
                "dense_source_q4_prefill_cta_profile":
                    "m64n128-pingpong-n16-k16-production",
                "dense_source_xattention_qualification": False,
                "automatic_headroom_bytes": HEADROOM,
            },
            "weights_capacity_bytes": weights, "device_capacity_bytes": device,
            "runtime_budget_after_weights_and_headroom_bytes": budget,
            "cells": cells,
            "prior_c4_slack": {"retained_claim_bytes": 2_004_481,
                               "current_derived_bytes": cells[-1]["remaining_slack_bytes"],
                               "confirmed_exact": cells[-1]["remaining_slack_bytes"] == 2_004_481}}


def current_report(planner: Path, weight_report: Path) -> dict[str, object]:
    planner_path = planner.expanduser().absolute()
    report_path = weight_report.expanduser().absolute()
    planner_fd, planner_owner = _open_regular(planner_path)
    report_fd, report_owner = _open_regular(report_path)
    try:
        _, planner_hash = _descriptor_bytes_and_sha256(planner_fd)
        report_bytes, report_hash = _descriptor_bytes_and_sha256(report_fd)
        completed = subprocess.run(
            [f"/proc/self/fd/{planner_fd}", "--host-hybrid-capacity-csv"],
            check=True, capture_output=True, text=True, pass_fds=(planner_fd,),
        )
        source = json.loads(report_bytes)
        weights, device, materialization = capacity_weights(source)
        report = assemble(parse_planner(completed.stdout), weights, device)
        _unchanged(planner_path, planner_fd, planner_owner, planner_hash)
        _unchanged(report_path, report_fd, report_owner, report_hash)
    finally:
        os.close(report_fd)
        os.close(planner_fd)
    report["provenance"] = {
        "planner_executable": {"path": str(planner_path), "sha256": planner_hash},
        "weights_report": {"path": str(report_path), "sha256": report_hash},
        "weight_materialization": materialization,
    }
    return report


def canonical(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _durable_create(path: Path, content: str) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.pending-", dir=path.parent)
    temporary = Path(temporary_name)
    owner: tuple[int, int] | None = None
    linked = False
    committed = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            metadata = os.fstat(output.fileno())
            owner = metadata.st_dev, metadata.st_ino
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if _path_identity(temporary) != owner:
            raise ValueError(f"pending hybrid capacity report changed: {path}")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"hybrid capacity report already exists: {path}") from error
        linked = True
        if _path_identity(path) != owner:
            raise ValueError(f"published hybrid capacity report inode differs: {path}")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        committed = True
    finally:
        if linked and not committed and owner is not None:
            try:
                if _path_identity(path) == owner:
                    path.unlink()
            except FileNotFoundError:
                pass
        temporary.unlink(missing_ok=True)
        if linked or owner is not None:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--weights-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    expected = current_report(args.planner, args.weights_report)
    if args.validate:
        descriptor, owner = _open_regular(args.output)
        try:
            content, digest = _descriptor_bytes_and_sha256(descriptor)
            if json.loads(content) != expected:
                raise ValueError("hybrid capacity report differs from current planner/provenance")
            _unchanged(args.output, descriptor, owner, digest)
        finally:
            os.close(descriptor)
    else:
        _durable_create(args.output, canonical(expected))
    print(json.dumps(expected["prior_c4_slack"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
