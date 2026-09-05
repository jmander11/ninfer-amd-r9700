#!/usr/bin/env python3
"""Bind a fixed matched report to dispatched gfx1201 FP8 and IU4 ISA evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.bench.validate_fp8_gate_up_comparison import validate as validate_comparison
from tools.bench.decide_fp8_post_gate_up import validate_qualifier as validate_projection


FP8_QUANTIZE = "fp8_quantize_activation_kernel"
FP8_POISON_BOUNDARIES = (
    "poison_nonfinite_output",
    "poison_nonfinite_gate_up_output",
)
Q4_KERNEL = "a8q4g64_linear_prefill_cta_kernel"

HARDWARE_PROFILES = {
    "ninfer.r9700.fp8_gate_up_hardware_proof_plan.v1": {
        "qualification": "gate_up",
        "shape": {"tokens": 2048, "rows": 34816, "columns": 5120},
        "proof_schema": "ninfer.r9700.fp8_gate_up_hardware_proof.v1",
        "resources": {
            "sgpr_count": 128, "architectural_vgpr_count": 192,
            "accum_vgpr_count": 0, "group_segment_lds_bytes": 25088,
            "private_segment_bytes": 0,
            "spill_counts": {"available": False, "sgpr": None, "vgpr": None},
        },
    },
    "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof_plan.v1": {
        "qualification": "attention_qk_gate_value",
        "shape": {"tokens": 2048, "rows": 7168, "columns": 5120},
        "proof_schema": "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof.v1",
        "resources": {
            "sgpr_count": 128, "architectural_vgpr_count": 192,
            "accum_vgpr_count": 0, "group_segment_lds_bytes": 12544,
            "private_segment_bytes": 0,
            "spill_counts": {"available": False, "sgpr": None, "vgpr": None},
        },
    },
}


def _is_fp8_poison_boundary(kernel_name: str | None) -> bool:
    return any(boundary in (kernel_name or "") for boundary in FP8_POISON_BOUNDARIES)


def _hardware_profile(plan: dict[str, Any]) -> dict[str, object]:
    profile = HARDWARE_PROFILES.get(plan.get("schema"))
    if profile is None:
        raise ValueError("unknown FP8 hardware-proof plan schema")
    # Qualification was added when attention support was introduced. Its absence in an
    # existing gate-up v1 plan retains the original gate-up meaning.
    qualification = plan.get("qualification", "gate_up")
    if qualification != profile["qualification"] or plan.get("shape") != profile["shape"]:
        raise ValueError("hardware-proof plan qualification or shape differs")
    return profile


def _validate_qualifier_report(report: dict[str, Any], qualification: str) -> None:
    if qualification == "gate_up":
        validate_comparison(report, verify_files=True)
    elif qualification == "attention_qk_gate_value":
        validate_projection(report, qualification)
    else:
        raise ValueError(f"unsupported qualification: {qualification}")


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as source:
        before = os.fstat(source.fileno())
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        after = os.fstat(source.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
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
    if data[start:start + 6] != b"\x7fELF\x02\x01":
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


def _loaded_elf(uri: str, symbol: str, capture_dir: Path | None = None) -> tuple[bytes, dict[str, object]]:
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
        matches = list(capture_dir.resolve(strict=True).glob(
            f"{parsed.netloc}-0x{offset:x}.co"))
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
            payload = data[start:_elf_extent(data, start)]
        except (ValueError, struct.error):
            continue
        if symbol.removesuffix(".kd").encode() in payload:
            candidates.append(payload)
    if len(candidates) != 1:
        raise ValueError(f"loaded extent must select one ELF containing dispatched symbol; found {len(candidates)}")
    if _identity(path) != source_id:
        raise ValueError("loaded code-object source changed during extraction")
    return candidates[0], {"uri": uri, "source": source_id,
                           "sha256": hashlib.sha256(candidates[0]).hexdigest(),
                           "bytes": len(candidates[0])}


def _disassemble(payload: bytes, symbol: str, objdump: Path) -> tuple[str, dict[str, object], dict[str, object]]:
    objdump_id = _identity(objdump)
    nm = objdump.with_name("llvm-nm")
    nm_id = _identity(nm)
    selected = symbol.removesuffix(".kd")
    with tempfile.TemporaryDirectory(prefix="ninfer-gate-up-isa-") as directory:
        code = Path(directory) / "loaded.co"
        code.write_bytes(payload)
        symbols = subprocess.run([nm_id["path"], "-n", str(code)], check=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        entries: list[tuple[int, str]] = []
        for line in symbols.stdout.splitlines():
            match = re.fullmatch(r"([0-9a-fA-F]+)\s+T\s+(.+)", line)
            if match:
                entries.append((int(match.group(1), 16), match.group(2)))
        starts = [address for address, name in entries if name == selected]
        if len(starts) != 1:
            raise ValueError("captured ELF lacks one exact global dispatched function symbol")
        start = starts[0]
        following = [address for address, _ in entries if address > start]
        if not following:
            raise ValueError("captured ELF does not bound the dispatched function")
        stop = min(following)
        command = [objdump_id["path"], "-d", "--mcpu=gfx1201",
                   f"--start-address=0x{start:x}", f"--stop-address=0x{stop:x}", str(code)]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    if not result.stdout.strip():
        raise ValueError("llvm-objdump produced no dispatched-symbol interval")
    return result.stdout, objdump_id, {"llvm_nm": nm_id, "start": start, "stop": stop}


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


def _validate_resources(rows: Sequence[sqlite3.Row], columns: set[str],
                        expected: object) -> dict[str, object]:
    encoded = {json.dumps(_resources(row, columns), sort_keys=True) for row in rows}
    if len(encoded) != 1:
        raise ValueError("FP8 calls did not retain one stable kernel resource envelope")
    resources = json.loads(next(iter(encoded)))
    if resources != expected:
        raise ValueError("selected hipBLASLt specialization resources differ from the admitted envelope")
    return resources


def _validate_resource_binding(rows: Sequence[sqlite3.Row], fp8: dict[str, Any]) -> None:
    code_object = fp8.get("code_object")
    expected = (
        fp8.get("kernel_id"), fp8.get("code_object_id"),
        code_object.get("uri") if isinstance(code_object, dict) else None,
    )
    observed = {(row["kernel_id"], row["code_object_id"], row["uri"]) for row in rows}
    if (type(expected[0]) is not int or type(expected[1]) is not int
            or not isinstance(expected[2], str) or observed != {expected}):
        raise ValueError("FP8 resource rows differ from the retained kernel/code-object/URI binding")


def revalidate_fp8_resources(proof: dict[str, Any]) -> dict[str, object]:
    profiles = [value for value in HARDWARE_PROFILES.values()
                if value["proof_schema"] == proof.get("schema")]
    if len(profiles) != 1 or proof.get("qualification") != profiles[0]["qualification"]:
        raise ValueError("FP8 proof schema/qualification does not select one resource envelope")
    profile = profiles[0]
    fp8 = proof.get("fp8")
    database = proof.get("trace_database")
    if not isinstance(fp8, dict) or not isinstance(database, dict):
        raise ValueError("FP8 proof lacks its kernel or trace database")
    database_path = Path(str(database.get("path", "")))
    if _identity(database_path) != database:
        raise ValueError("FP8 proof trace database changed")
    connection = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        agent, code, symbol, dispatch = (_table(connection, stem) for stem in (
            "rocpd_info_agent", "rocpd_info_code_object", "rocpd_info_kernel_symbol",
            "rocpd_kernel_dispatch"))
        gpu = connection.execute(
            f"SELECT * FROM {agent} WHERE type='GPU' AND type_index=0").fetchall()
        if len(gpu) != 1 or gpu[0]["name"] != "gfx1201" \
                or gpu[0]["product_name"] != "AMD Radeon AI PRO R9700":
            raise ValueError("FP8 proof trace is not device-0 R9700/gfx1201")
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({symbol})")}
        optional_spills = (
            ",s.sgpr_spill_count,s.vgpr_spill_count"
            if {"sgpr_spill_count", "vgpr_spill_count"} <= columns else "")
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
    if (len(rows) != fp8.get("dispatch_count") or len(rows) != 16
            or any(row["kernel_name"] != fp8.get("kernel_name") for row in rows)):
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
        str(code_object.get("uri", "")), str(fp8.get("kernel_name", "")),
        Path(str(source.get("path", ""))).resolve(strict=True).parent)
    if extracted != code_object or hashlib.sha256(payload).hexdigest() != code_object.get("sha256"):
        raise ValueError("FP8 proof loaded ELF no longer matches its dispatch resource rows")
    return resources


def validate(plan_path: Path, report_path: Path, trace_dir: Path, out_dir: Path,
             objdump: Path) -> dict[str, object]:
    plan_id, report_id = _identity(plan_path), _identity(report_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    profile = _hardware_profile(plan)
    qualification = str(profile["qualification"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_qualifier_report(report, qualification)
    trace_report_path = Path(str(plan.get("fresh_report", "")))
    capture_dir = Path(str(plan.get("fresh_capture_directory", "")))
    trace_report_id = _identity(trace_report_path)
    trace_report = json.loads(trace_report_path.read_text(encoding="utf-8"))
    _validate_qualifier_report(trace_report, qualification)
    if (plan.get("admission_report") != report_id or
            plan.get("fresh_trace_directory") != str(trace_dir.absolute()) or
            not capture_dir.is_absolute() or
            _identity(Path(plan.get("capture_library", {}).get("path", ""))) !=
                plan.get("capture_library") or
            plan.get("qualifier", {}).get("path") != report["provenance"]["executable_path"] or
            plan.get("qualifier", {}).get("sha256") != report["provenance"]["executable_sha256"] or
            trace_report["provenance"] != report["provenance"] or
            trace_report["algorithm"] != report["algorithm"] or
            trace_report["shape"] != report["shape"] or
            trace_report["fp8_profile"] != report["fp8_profile"] or
            trace_report["q4_control"] != report["q4_control"]):
        raise ValueError("plan does not bind the retained qualifier report")
    databases = list(trace_dir.resolve(strict=True).rglob("*_results.db"))
    if len(databases) != 1:
        raise ValueError("trace directory must contain exactly one rocprofv3 results database")
    database_id = _identity(databases[0])
    connection = sqlite3.connect(f"file:{databases[0]}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        agent, code, symbol, dispatch = (_table(connection, stem) for stem in (
            "rocpd_info_agent", "rocpd_info_code_object", "rocpd_info_kernel_symbol",
            "rocpd_kernel_dispatch"))
        gpu = connection.execute(
            f"SELECT * FROM {agent} WHERE type='GPU' AND type_index=0").fetchall()
        if len(gpu) != 1 or gpu[0]["name"] != "gfx1201" or gpu[0]["product_name"] != "AMD Radeon AI PRO R9700":
            raise ValueError("trace is not device-0 Radeon AI PRO R9700/gfx1201")
        symbol_columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({symbol})")
        }
        optional_spills = (
            ",s.sgpr_spill_count,s.vgpr_spill_count"
            if {"sgpr_spill_count", "vgpr_spill_count"} <= symbol_columns else ""
        )
        rows = connection.execute(
            f"SELECT d.dispatch_id,d.start,d.stream_id,s.id AS kernel_id,s.kernel_name,"
            f"s.display_name,s.code_object_id,s.group_segment_size,s.private_segment_size,"
            f"s.sgpr_count,s.arch_vgpr_count,s.accum_vgpr_count{optional_spills},c.uri "
            f"FROM {dispatch} d JOIN {symbol} s ON s.id=d.kernel_id "
            f"JOIN {code} c ON c.id=s.code_object_id WHERE d.agent_id=? ORDER BY d.start",
            (gpu[0]["id"],)).fetchall()
    finally:
        connection.close()
    fp8_windows: list[list[sqlite3.Row]] = []
    for index, row in enumerate(rows):
        if FP8_QUANTIZE not in (row["kernel_name"] or ""):
            continue
        finish = next((
            end for end in range(index + 1, len(rows))
            if _is_fp8_poison_boundary(rows[end]["kernel_name"])
        ), None)
        if finish is None:
            raise ValueError("FP8 quantizer dispatch lacks its status-poison boundary")
        fp8_windows.append(rows[index + 1:finish])
    if len(fp8_windows) != 16 or any(not window for window in fp8_windows):
        raise ValueError("trace does not contain the exact 16 complete FP8 calls")
    fp8_keys = {(row["kernel_id"], row["code_object_id"]) for window in fp8_windows for row in window}
    if len(fp8_keys) != 1:
        raise ValueError("FP8 calls did not dispatch one stable selected hipBLASLt specialization")
    fp8 = fp8_windows[0][0]
    fp8_resource_value = _validate_resources(
        [row for window in fp8_windows for row in window],
        symbol_columns, profile["resources"])
    q4_rows = [row for row in rows if Q4_KERNEL in (row["kernel_name"] or "")]
    if len(q4_rows) != 16 or len({(row["kernel_id"], row["code_object_id"]) for row in q4_rows}) != 1:
        raise ValueError("trace does not contain the exact 16 stable production-Q4 calls")
    q4 = q4_rows[0]
    fp8_payload, fp8_code = _loaded_elf(fp8["uri"], fp8["kernel_name"], capture_dir)
    q4_payload, q4_code = _loaded_elf(q4["uri"], q4["kernel_name"])
    fp8_isa, objdump_id, fp8_interval = _disassemble(
        fp8_payload, fp8["kernel_name"], objdump)
    q4_isa, _, q4_interval = _disassemble(q4_payload, q4["kernel_name"], objdump)
    fp8_opcodes = re.findall(r"(?m)^\s*v_(?:wmma|mfma)[^\n]*(?:fp8|f8)[^\n]*$", fp8_isa)
    q4_opcodes = re.findall(r"(?m)^\s*v_wmma_i32_16x16x32_iu4\b[^\n]*$", q4_isa)
    if not fp8_opcodes:
        raise ValueError("selected dispatched hipBLASLt symbol has no native FP8 WMMA/MFMA ISA")
    if not q4_opcodes:
        raise ValueError("dispatched production Q4 symbol has no native IU4 WMMA ISA")
    out_dir.mkdir(parents=False, exist_ok=False)
    artifacts = (("fp8.co", fp8_payload), ("q4.co", q4_payload),
                 ("fp8.s", fp8_isa.encode()), ("q4.s", q4_isa.encode()))
    artifact_ids: dict[str, object] = {}
    for name, payload in artifacts:
        path = out_dir / name
        with path.open("xb") as output:
            output.write(payload)
        artifact_ids[name] = _identity(path)
    proof = {
        "schema": profile["proof_schema"], "qualification": qualification, "pass": True,
        "plan": plan_id, "matched_report": report_id, "traced_report": trace_report_id,
        "trace_database": database_id,
        "objdump": objdump_id,
        "fp8": {"dispatch_count": 16, "kernel_id": fp8["kernel_id"],
                "code_object_id": fp8["code_object_id"],
                "kernel_name": fp8["kernel_name"], "code_object": fp8_code,
                "symbol_interval": fp8_interval,
                "native_fp8_matrix_opcode_count": len(fp8_opcodes),
                "resources": fp8_resource_value,
                "resource_binding": (
                    "all 16 rocprof kernel-symbol rows share this kernel_id/code_object_id; "
                    "that code-object URI is re-extracted and SHA-bound to fp8.co/fp8.s"
                )},
        "q4": {"dispatch_count": 16, "kernel_id": q4["kernel_id"],
               "kernel_name": q4["kernel_name"], "code_object": q4_code,
               "symbol_interval": q4_interval,
               "native_iu4_wmma_opcode_count": len(q4_opcodes)},
        "artifacts": artifact_ids,
    }
    with (out_dir / "proof.json").open("x", encoding="utf-8") as output:
        json.dump(proof, output, indent=2, sort_keys=True)
        output.write("\n")
    if revalidate_fp8_resources(proof) != fp8_resource_value:
        raise ValueError("published FP8 resource proof failed independent DB/ELF revalidation")
    if (_identity(plan_path) != plan_id or _identity(report_path) != report_id or
            _identity(trace_report_path) != trace_report_id or
            _identity(databases[0]) != database_id):
        raise ValueError("proof inputs changed during validation")
    return proof


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--objdump", type=Path,
                        default=Path("/opt/rocm/llvm/bin/llvm-objdump"))
    args = parser.parse_args(argv)
    proof = validate(args.plan, args.report, args.trace_dir, args.out_dir, args.objdump)
    print(json.dumps({"pass": proof["pass"], "proof": str((args.out_dir / 'proof.json').resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
