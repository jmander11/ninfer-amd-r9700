"""Resolve selected FP8 dispatches through rocprof code-object URIs to captured ELF bytes.

Also reopens retained hipBLASLt-era FP8 executed-path proofs (`proof.json`) against their trace
database and captured loaded ELF. The proof producer and its FP8 gate/up qualifier were retired
with hipBLASLt in 438f6a89; only the retained proofs are revalidated here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FP8_LINEAR = "Cijk_Alik_Bljk_F8BS_"

_FP8_RESOURCE_ENVELOPE_COMMON = {
    "sgpr_count": 128,
    "architectural_vgpr_count": 192,
    "accum_vgpr_count": 0,
    "private_segment_bytes": 0,
    "spill_counts": {"available": False, "sgpr": None, "vgpr": None},
}
FP8_PROOF_PROFILES = {
    "ninfer.r9700.fp8_gate_up_hardware_proof.v1": {
        "qualification": "gate_up",
        "resources": {**_FP8_RESOURCE_ENVELOPE_COMMON, "group_segment_lds_bytes": 25088},
    },
    "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof.v1": {
        "qualification": "attention_qk_gate_value",
        "resources": {**_FP8_RESOURCE_ENVELOPE_COMMON, "group_segment_lds_bytes": 12544},
    },
}


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as source:
        before = os.fstat(source.fileno())
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        after = os.fstat(source.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"file changed while hashing: {resolved}")
    return {"path": str(resolved), "bytes": after.st_size, "sha256": digest}


def _table(connection: sqlite3.Connection, stem: str) -> str:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (stem + "_%",),
    ).fetchall()
    names = [str(row[0]) for row in rows if re.fullmatch(re.escape(stem) + r"_[0-9a-f_]+", row[0])]
    if len(names) != 1:
        raise ValueError(f"trace must contain exactly one concrete {stem} table")
    return '"' + names[0].replace('"', '""') + '"'


def _elf_extent(data: bytes, start: int) -> int:
    if data[start : start + 6] != b"\x7fELF\x02\x01":
        raise ValueError("loaded code object is not little-endian ELF64")
    if start + 64 > len(data):
        raise ValueError("truncated ELF header")
    phoff, shoff = struct.unpack_from("<QQ", data, start + 32)
    phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", data, start + 54)
    end = max(64, phoff + phentsize * phnum, shoff + shentsize * shnum)
    if start + end > len(data) or shentsize < 64:
        raise ValueError("truncated ELF tables")
    for index in range(shnum):
        header = start + shoff + index * shentsize
        section_type = struct.unpack_from("<I", data, header + 4)[0]
        offset, size = struct.unpack_from("<QQ", data, header + 24)
        if section_type != 8:  # SHT_NOBITS has no file payload.
            end = max(end, offset + size)
    if start + end > len(data):
        raise ValueError("truncated ELF sections")
    return start + end


def _loaded_elf(
    uri: str, symbol: str, capture_dir: Path | None = None
) -> tuple[bytes, dict[str, object]]:
    parsed = urllib.parse.urlparse(uri)
    query = urllib.parse.parse_qs(parsed.fragment, strict_parsing=True)
    try:
        offset = int(query["offset"][0], 0)
        size = int(query["size"][0], 0)
    except (KeyError, ValueError) as error:
        raise ValueError(f"code-object URI lacks a valid offset/size: {uri}") from error
    if parsed.scheme == "memory":
        if capture_dir is None or not parsed.netloc.isdigit():
            raise ValueError(f"memory-backed dispatch has no exact load-time capture: {uri}")
        matches = list(capture_dir.resolve(strict=True).glob(f"{parsed.netloc}-0x{offset:x}.co"))
        if len(matches) != 1:
            raise ValueError("memory-backed dispatch must match one pointer-keyed captured ELF")
        path = matches[0].resolve(strict=True)
        source_id = _identity(path)
        data = path.read_bytes()
        if len(data) != size:
            raise ValueError("captured ELF size differs from rocprof memory URI")
        starts = [0]
        offset = 0
        size = len(data)
    elif parsed.scheme == "file":
        path = Path(urllib.parse.unquote(parsed.path)).resolve(strict=True)
        source_id = _identity(path)
        data = path.read_bytes()
        starts = []
    else:
        raise ValueError(f"unsupported dispatched code-object URI: {uri}")
    if offset < 0 or size <= 0 or offset + size > len(data):
        raise ValueError("code-object URI extent is outside its immutable source file")
    if not starts:
        cursor = offset
        while True:
            cursor = data.find(b"\x7fELF", cursor, offset + size)
            if cursor < 0:
                break
            starts.append(cursor)
            cursor += 4
    candidates: list[bytes] = []
    for start in starts:
        try:
            payload = data[start : _elf_extent(data, start)]
        except (ValueError, struct.error):
            continue
        if symbol.removesuffix(".kd").encode() in payload:
            candidates.append(payload)
    if len(candidates) != 1:
        raise ValueError(
            f"loaded extent must select one ELF containing dispatched symbol; found {len(candidates)}"
        )
    if _identity(path) != source_id:
        raise ValueError("loaded code-object source changed during extraction")
    return candidates[0], {
        "uri": uri,
        "source": source_id,
        "sha256": hashlib.sha256(candidates[0]).hexdigest(),
        "bytes": len(candidates[0]),
    }


def _resources(row: sqlite3.Row, columns: set[str]) -> dict[str, object]:
    spill_available = {"sgpr_spill_count", "vgpr_spill_count"} <= columns
    return {
        "sgpr_count": row["sgpr_count"],
        "architectural_vgpr_count": row["arch_vgpr_count"],
        "accum_vgpr_count": row["accum_vgpr_count"],
        "group_segment_lds_bytes": row["group_segment_size"],
        "private_segment_bytes": row["private_segment_size"],
        "spill_counts": {
            "available": spill_available,
            "sgpr": row["sgpr_spill_count"] if spill_available else None,
            "vgpr": row["vgpr_spill_count"] if spill_available else None,
        },
    }


def _validate_resources(
    rows: Sequence[sqlite3.Row], columns: set[str], expected: object
) -> dict[str, object]:
    encoded = {json.dumps(_resources(row, columns), sort_keys=True) for row in rows}
    if len(encoded) != 1:
        raise ValueError("FP8 calls did not retain one stable kernel resource envelope")
    resources = json.loads(next(iter(encoded)))
    if resources != expected:
        raise ValueError(
            "selected hipBLASLt specialization resources differ from the admitted envelope"
        )
    return resources


def _validate_resource_binding(rows: Sequence[sqlite3.Row], fp8: dict[str, Any]) -> None:
    code_object = fp8.get("code_object")
    expected = (
        fp8.get("kernel_id"),
        fp8.get("code_object_id"),
        code_object.get("uri") if isinstance(code_object, dict) else None,
    )
    observed = {(row["kernel_id"], row["code_object_id"], row["uri"]) for row in rows}
    if (
        type(expected[0]) is not int
        or type(expected[1]) is not int
        or not isinstance(expected[2], str)
        or observed != {expected}
    ):
        raise ValueError(
            "FP8 resource rows differ from the retained kernel/code-object/URI binding"
        )


def revalidate_fp8_resources(proof: dict[str, Any]) -> dict[str, object]:
    profile = FP8_PROOF_PROFILES.get(proof.get("schema"))
    if profile is None or proof.get("qualification") != profile["qualification"]:
        raise ValueError("FP8 proof schema/qualification does not select one resource envelope")
    fp8 = proof.get("fp8")
    database = proof.get("trace_database")
    if not isinstance(fp8, dict) or not isinstance(database, dict):
        raise ValueError("FP8 proof lacks its kernel or trace database")
    database_path = Path(str(database.get("path", "")))
    if _identity(database_path) != database:
        raise ValueError("FP8 proof trace database changed")
    connection = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        agent, code, symbol, dispatch = (
            _table(connection, stem)
            for stem in (
                "rocpd_info_agent",
                "rocpd_info_code_object",
                "rocpd_info_kernel_symbol",
                "rocpd_kernel_dispatch",
            )
        )
        gpu = connection.execute(
            f"SELECT * FROM {agent} WHERE type='GPU' AND type_index=0"
        ).fetchall()
        if (
            len(gpu) != 1
            or gpu[0]["name"] != "gfx1201"
            or gpu[0]["product_name"] != "AMD Radeon AI PRO R9700"
        ):
            raise ValueError("FP8 proof trace is not device-0 R9700/gfx1201")
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({symbol})")}
        optional_spills = (
            ",s.sgpr_spill_count,s.vgpr_spill_count"
            if {"sgpr_spill_count", "vgpr_spill_count"} <= columns
            else ""
        )
        rows = connection.execute(
            f"SELECT d.dispatch_id,s.id AS kernel_id,s.kernel_name,s.code_object_id,"
            f"s.group_segment_size,s.private_segment_size,s.sgpr_count,s.arch_vgpr_count,"
            f"s.accum_vgpr_count{optional_spills},c.uri FROM {dispatch} d "
            f"JOIN {symbol} s ON s.id=d.kernel_id JOIN {code} c ON c.id=s.code_object_id "
            f"WHERE d.agent_id=? AND s.id=? ORDER BY d.start",
            (gpu[0]["id"], fp8.get("kernel_id")),
        ).fetchall()
    finally:
        connection.close()
    if (
        len(rows) != fp8.get("dispatch_count")
        or len(rows) != 16
        or any(row["kernel_name"] != fp8.get("kernel_name") for row in rows)
    ):
        raise ValueError("FP8 proof does not reopen the exact 16 stable dispatch rows")
    _validate_resource_binding(rows, fp8)
    resources = _validate_resources(rows, columns, profile["resources"])
    if resources != fp8.get("resources"):
        raise ValueError("FP8 proof resource record differs from its trace database")
    code_object = fp8.get("code_object")
    source = code_object.get("source") if isinstance(code_object, dict) else None
    if not isinstance(source, dict):
        raise ValueError("FP8 proof lacks its captured loaded ELF")
    payload, extracted = _loaded_elf(
        str(code_object.get("uri", "")),
        str(fp8.get("kernel_name", "")),
        Path(str(source.get("path", ""))).resolve(strict=True).parent,
    )
    if extracted != code_object or hashlib.sha256(payload).hexdigest() != code_object.get("sha256"):
        raise ValueError("FP8 proof loaded ELF no longer matches its dispatch resource rows")
    return resources


def selected_loaded_fp8(
    database_path: Path, capture_dir: Path, selected_dispatches: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected_ids = {
        row["rocprof_dispatch_id"] for row in selected_dispatches if FP8_LINEAR in row["symbol"]
    }
    if not selected_ids:
        raise ValueError("hybrid trace contains no selected FP8 linear dispatch")
    connection = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {str(row[1]) for row in connection.execute('PRAGMA table_info("kernels")')}
        required = {
            "dispatch_id",
            "kernel_id",
            "name",
            "code_object_id",
            "start",
            "sgpr_count",
            "vgpr_count",
            "accum_vgpr_count",
            "lds_size",
            "scratch_size",
        }
        if not required <= columns:
            raise ValueError("selected trace kernels view lacks exact FP8 resource columns")
        spill_available = {"sgpr_spill_count", "vgpr_spill_count"} <= columns
        optional_spills = ",k.sgpr_spill_count,k.vgpr_spill_count" if spill_available else ""
        rows = list(
            connection.execute(
                "SELECT k.dispatch_id,k.kernel_id,k.name,k.code_object_id,k.sgpr_count,k.vgpr_count,"
                f"k.accum_vgpr_count,k.lds_size,k.scratch_size{optional_spills},c.uri FROM kernels k "
                "JOIN code_objects c ON c.id=k.code_object_id ORDER BY k.start,k.dispatch_id"
            )
        )
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
        identities.append(
            {
                "kernel_name": symbol,
                "kernel_id": kernel_id,
                "code_object_id": code_object_id,
                "dispatch_ids": sorted(group["dispatch_ids"]),
                "resources": group["resources"],
                "code_object": code,
            }
        )
    return sorted(identities, key=lambda item: item["kernel_name"])
