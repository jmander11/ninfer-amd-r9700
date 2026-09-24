#!/usr/bin/env python3
"""Resolve selected FP8 dispatches through rocprof code-object URIs to captured ELF bytes."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from tools.bench.validate_fp8_gate_up_hardware_proof import _loaded_elf


FP8_LINEAR = "Cijk_Alik_Bljk_F8BS_"


def selected_loaded_fp8(database_path: Path, capture_dir: Path,
                        selected_dispatches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_ids = {row["rocprof_dispatch_id"] for row in selected_dispatches
                    if FP8_LINEAR in row["symbol"]}
    if not selected_ids:
        raise ValueError("hybrid trace contains no selected FP8 linear dispatch")
    connection = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {str(row[1]) for row in connection.execute('PRAGMA table_info("kernels")')}
        required = {"dispatch_id", "kernel_id", "name", "code_object_id", "start", "sgpr_count",
                    "vgpr_count", "accum_vgpr_count", "lds_size", "scratch_size"}
        if not required <= columns:
            raise ValueError("selected trace kernels view lacks exact FP8 resource columns")
        spill_available = {"sgpr_spill_count", "vgpr_spill_count"} <= columns
        optional_spills = (",k.sgpr_spill_count,k.vgpr_spill_count"
                           if spill_available else "")
        rows = list(connection.execute(
            "SELECT k.dispatch_id,k.kernel_id,k.name,k.code_object_id,k.sgpr_count,k.vgpr_count,"
            f"k.accum_vgpr_count,k.lds_size,k.scratch_size{optional_spills},c.uri FROM kernels k "
            "JOIN code_objects c ON c.id=k.code_object_id ORDER BY k.start,k.dispatch_id"))
    finally:
        connection.close()
    matched = [row for row in rows if row["dispatch_id"] in selected_ids]
    if {row["dispatch_id"] for row in matched} != selected_ids or len(matched) != len(selected_ids):
        raise ValueError("selected FP8 dispatch-to-code-object mapping is incomplete")
    grouped: dict[tuple[str, int, int, str], dict[str, object]] = {}
    for row in matched:
        symbol, uri = row["name"], row["uri"]
        if not isinstance(symbol, str) or FP8_LINEAR not in symbol or not isinstance(uri, str):
            raise ValueError("selected FP8 dispatch has invalid code-object identity")
        resources = {
            "sgpr_count": row["sgpr_count"],
            "architectural_vgpr_count": row["vgpr_count"],
            "accum_vgpr_count": row["accum_vgpr_count"],
            "group_segment_lds_bytes": row["lds_size"],
            "private_segment_bytes": row["scratch_size"],
            "spill_counts": {
                "available": spill_available,
                "sgpr": row["sgpr_spill_count"] if spill_available else None,
                "vgpr": row["vgpr_spill_count"] if spill_available else None,
            },
        }
        key = (symbol, int(row["kernel_id"]), int(row["code_object_id"]), uri)
        group = grouped.setdefault(key, {"dispatch_ids": [], "resources": resources})
        if group["resources"] != resources:
            raise ValueError("selected FP8 dispatch resources vary for one loaded ELF")
        group["dispatch_ids"].append(int(row["dispatch_id"]))
    identities = []
    for (symbol, kernel_id, code_object_id, uri), group in grouped.items():
        payload, code = _loaded_elf(uri, symbol, capture_dir)
        if code["sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("selected FP8 code-object extraction changed")
        identities.append({"kernel_name": symbol, "kernel_id": kernel_id,
                           "code_object_id": code_object_id,
                           "dispatch_ids": sorted(group["dispatch_ids"]),
                           "resources": group["resources"], "code_object": code})
    return sorted(identities, key=lambda item: item["kernel_name"])
