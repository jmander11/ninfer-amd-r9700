#!/usr/bin/env python3
"""Join a schema-v7 winner to executed dispatches and its exact gfx1201 ISA proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from tools.ppl.pareto import load_payload, validate_terminal_production_authority
from tools.bench.reconcile_qwen3_8_27b_dispatches import reconcile
from tools.bench.selected_loaded_code_objects import selected_loaded_fp8
from tools.bench.extract_embedded_code_object import extract
from tools.bench.validate_fp8_gate_up_hardware_proof import revalidate_fp8_resources
from tools.bench.run_ninfer_bench_matrix import (
    bind_n16_migration_receipt, inspect_artifact,
)


AUDIT_SCHEMA = "ninfer.r9700.twelve_candidate_hardware_path_static_audit.v1"
RECONCILIATION_SCHEMA = "ninfer_qwen3_8_27b_dispatch_reconciliation"
OUTPUT_SCHEMA = "ninfer_r9700_selected_hardware_use"
HYBRID_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
LLVM_NM = Path("/opt/rocm/llvm/bin/llvm-nm")


def _machine_interval(path: Path, symbol: str) -> dict[str, Any]:
    result = subprocess.run([str(LLVM_NM), "-S", "-n", str(path)], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    matches = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4 and fields[2] == "T" and fields[3] == symbol:
            matches.append((int(fields[0], 16), int(fields[1], 16)))
    if len(matches) != 1 or matches[0][1] <= 0:
        raise ValueError(f"embedded code object lacks one nonempty machine symbol {symbol}")
    address, size = matches[0]
    data = path.read_bytes()
    if len(data) < 64 or data[:5] != b"\x7fELF\x02":
        raise ValueError("embedded code object is not ELF64")
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    section_entry_size, section_count = struct.unpack_from("<HH", data, 58)
    if section_entry_size < 64 or section_offset + section_entry_size * section_count > len(data):
        raise ValueError("embedded code object has invalid section headers")
    extents = []
    for index in range(section_count):
        header = section_offset + index * section_entry_size
        section_address, file_offset, section_size = struct.unpack_from("<QQQ", data, header + 16)
        if (section_address <= address and address + size <= section_address + section_size
                and file_offset + address - section_address + size <= len(data)):
            begin = file_offset + address - section_address
            extents.append(data[begin:begin + size])
    if len(extents) != 1:
        raise ValueError(f"machine symbol {symbol} does not map to one file extent")
    return {"machine_bytes": size, "machine_sha256": hashlib.sha256(extents[0]).hexdigest()}


def _selected_embedded_static(executable: Path, profile: str, audit: dict[str, Any],
                              proofs: dict[str, Any], required: list[str]) -> dict[str, Any]:
    embedded = audit.get("embedded_code_objects")
    if not isinstance(embedded, dict):
        raise ValueError("static audit lacks selected embedded code-object identities")
    groups = [
        ("linear_by_profile", "q4_p2048_cta",
         ["q4_p2048_cta", "q4_wave32", "w8_p2048_cta"]),
        ("attention_by_profile", "ordinary_fp8_qk",
         ["ordinary_fp8_qk", "dense_initial_prefix_qk"]),
    ]
    if profile.startswith("xattention-"):
        groups.append(("xattention_by_profile", "xattention_rank",
                       ["xattention_rank", "xattention_flash_consumer"]))
    evidence = {}
    with tempfile.TemporaryDirectory(prefix="ninfer-selected-static-") as directory:
        for family, selector_name, names in groups:
            selector = proofs[selector_name]["code_symbol"]
            code = Path(directory) / f"{family}.co"
            extracted = extract(executable, code, code_symbol=selector)
            expected_code_sha = embedded.get(family, {}).get(profile)
            if extracted.get("code_object_sha256") != expected_code_sha:
                raise ValueError(f"selected {family} embedded ELF differs from static audit")
            intervals = {}
            for name in names:
                if name not in required and name not in ("ordinary_fp8_qk", "q4_p2048_cta"):
                    continue
                proof = proofs[name]
                symbol = (proof.get(f"g{16 if profile.endswith('g16') else 32}_code_symbol")
                          if name == "xattention_flash_consumer" else proof.get("code_symbol"))
                if not isinstance(symbol, str):
                    raise ValueError(f"static proof {name} lacks selected code symbol")
                actual = _machine_interval(code, symbol)
                expected_sha = (proof.get("g16_machine_sha256")
                                if name == "xattention_flash_consumer" and profile.endswith("g16")
                                else proof.get("g32_machine_sha256")
                                if name == "xattention_flash_consumer"
                                else proof.get("machine_sha256_all_four",
                                               proof.get("machine_sha256_both_sparse_builds")))
                if (not isinstance(expected_sha, str) or actual["machine_sha256"] != expected_sha
                        or ("machine_bytes" in proof
                            and actual["machine_bytes"] != proof["machine_bytes"])):
                    raise ValueError(f"selected {name} machine interval differs from static audit")
                intervals[name] = {"code_symbol": symbol, **actual}
            evidence[family] = {"code_object_sha256": expected_code_sha,
                                "machine_intervals": intervals}
    return evidence


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    return {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
            "sha256": sha(resolved)}


def verify_snapshot(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{label} is not a file snapshot")
    actual = snapshot(Path(value["path"]))
    size = value.get("file_size_bytes", value.get("bytes"))
    if value.get("sha256") != actual["sha256"] or size != actual["file_size_bytes"]:
        raise ValueError(f"{label} bytes changed")
    return actual


def selected_route(selection_path: Path) -> dict[str, Any]:
    selection_path = selection_path.resolve(strict=True)
    selection = load_payload(selection_path.read_text(encoding="utf-8"))
    terminal, _ = validate_terminal_production_authority(selection)
    winner = terminal["winner"]
    sources = [row for row in selection["source_provenance"] if row.get("candidate") == winner]
    if len(sources) != 1:
        raise ValueError("terminal winner lacks unique source provenance")
    source = sources[0]
    artifact = source.get("artifact")
    executable = source.get("benchmark_executable")
    if not isinstance(artifact, dict) or not isinstance(executable, dict):
        raise ValueError("terminal winner lacks artifact/executable identity")
    artifact_path = Path(str(artifact.get("path", ""))).resolve(strict=True)
    artifact_now = bind_n16_migration_receipt(
        artifact_path, inspect_artifact(artifact_path))
    executable_now = verify_snapshot(executable, "selected executable")
    if (artifact_now["sha256"] != terminal["winner_artifact"]["sha256"]
            or artifact_now.get("conversion_receipt")
            != terminal["winner_artifact"].get("conversion_receipt")):
        raise ValueError("terminal artifact digest differs from winner")
    return {
        "terminal_selection": snapshot(selection_path), "winner": winner,
        "weights_id": terminal["winner_artifact"]["weights_id"],
        "artifact": artifact_now,
        "executable": executable_now,
        "kv_value_group": terminal["winner_cache_profile"]["value_group"],
        "xattention_profile": terminal["winner_execution_profile"]["xattention_profile"],
        "prefill_chunk": selection["selected_prefill_chunk"],
    }


def _proof_file(value: object, label: str) -> None:
    verify_snapshot(value, label)


def _fp8_proof(path: Path, qualification: str, dispatches: list[dict[str, Any]],
               loaded_objects: list[dict[str, Any]],
               resource_validator: Callable[[dict[str, Any]], dict[str, object]]
               = revalidate_fp8_resources) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_schema = {
        "gate_up": "ninfer.r9700.fp8_gate_up_hardware_proof.v1",
        "attention_qk_gate_value":
            "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof.v1",
    }[qualification]
    if (value.get("schema") != expected_schema or value.get("qualification") != qualification
            or value.get("pass") is not True):
        raise ValueError(f"{qualification} FP8 proof is not admitted")
    fp8 = value.get("fp8")
    if not isinstance(fp8, dict) or type(fp8.get("native_fp8_matrix_opcode_count")) is not int \
            or fp8["native_fp8_matrix_opcode_count"] <= 0:
        raise ValueError(f"{qualification} proof lacks native FP8 matrix instructions")
    expected_resources = {
        "gate_up": {
            "sgpr_count": 128, "architectural_vgpr_count": 192,
            "accum_vgpr_count": 0, "group_segment_lds_bytes": 25088,
            "private_segment_bytes": 0,
            "spill_counts": {"available": False, "sgpr": None, "vgpr": None},
        },
        "attention_qk_gate_value": {
            "sgpr_count": 128, "architectural_vgpr_count": 192,
            "accum_vgpr_count": 0, "group_segment_lds_bytes": 12544,
            "private_segment_bytes": 0,
            "spill_counts": {"available": False, "sgpr": None, "vgpr": None},
        },
    }[qualification]
    if (fp8.get("resources") != expected_resources
            or resource_validator(value) != expected_resources):
        raise ValueError(f"{qualification} proof lacks its exact FP8 resource envelope")
    kernel = fp8.get("kernel_name")
    matching = [row for row in dispatches if isinstance(row.get("symbol"), str)
                and row["symbol"].removesuffix(".kd") == str(kernel).removesuffix(".kd")]
    required_roles = ({"mlp_gate_up", "gdn_query_key"} if qualification == "gate_up"
                      else {"attention_query_key", "attention_gate_value"})
    if (not isinstance(kernel, str)
            or not required_roles <= {row.get("role") for row in matching}):
        raise ValueError(f"{qualification} loaded-ELF symbol did not execute in selected inventory")
    expected_trace_resources = {
        "sgpr_count": expected_resources["sgpr_count"],
        "vgpr_count": expected_resources["architectural_vgpr_count"],
        "accum_vgpr_count": expected_resources["accum_vgpr_count"],
        "lds_bytes": expected_resources["group_segment_lds_bytes"],
        "scratch_bytes": expected_resources["private_segment_bytes"],
    }
    if any(not isinstance(row.get("resources"), dict)
           or any(row["resources"].get(name) != expected
                  for name, expected in expected_trace_resources.items())
           for row in matching):
        raise ValueError(f"{qualification} selected dispatch resources differ from qualified ELF")
    code = fp8.get("code_object")
    if not isinstance(code, dict) or not isinstance(code.get("source"), dict):
        raise ValueError(f"{qualification} proof lacks loaded code-object identity")
    _proof_file(code["source"], f"{qualification} loaded ELF")
    if code.get("sha256") != code["source"].get("sha256"):
        raise ValueError(f"{qualification} loaded ELF digest differs")
    selected_loaded = [item for item in loaded_objects
                       if isinstance(item, dict)
                       and str(item.get("kernel_name", "")).removesuffix(".kd")
                       == kernel.removesuffix(".kd")]
    if len(selected_loaded) != 1:
        raise ValueError(f"{qualification} selected dispatch lacks one loaded ELF identity")
    selected_code = selected_loaded[0].get("code_object")
    if (not isinstance(selected_code, dict) or selected_code.get("sha256") != code["sha256"]
            or type(selected_loaded[0].get("kernel_id")) is not int
            or type(selected_loaded[0].get("code_object_id")) is not int
            or selected_loaded[0].get("resources") != expected_resources
            or not isinstance(selected_loaded[0].get("dispatch_ids"), list)
            or not selected_loaded[0]["dispatch_ids"]):
        raise ValueError(
            f"{qualification} selected loaded ELF/resources differ from qualified ISA")
    _proof_file(selected_code.get("source"), f"{qualification} selected loaded ELF")
    for name in ("plan", "matched_report", "traced_report", "trace_database", "objdump"):
        _proof_file(value.get(name), f"{qualification} {name}")
    return {"authority": snapshot(path), "kernel_name": kernel,
            "loaded_elf_sha256": code["sha256"],
            "selected_trace_code_object": selected_loaded[0],
            "native_fp8_matrix_opcode_count": fp8["native_fp8_matrix_opcode_count"],
            "resources": expected_resources}


def verify(selection_path: Path, reconciliation_path: Path, audit_path: Path,
           fp8_paths: list[Path], *,
           route_resolver: Callable[[Path], dict[str, Any]] = selected_route,
           reconciliation_validator: Callable[[Path, Path], dict[str, Any]] = reconcile,
           loaded_fp8_validator: Callable[
               [Path, Path, list[dict[str, Any]]], list[dict[str, Any]]
           ] = selected_loaded_fp8,
           embedded_validator: Callable[
               [Path, str, dict[str, Any], dict[str, Any], list[str]], dict[str, Any]
           ] = _selected_embedded_static,
           fp8_resource_validator: Callable[[dict[str, Any]], dict[str, object]] =
           revalidate_fp8_resources,
           ) -> dict[str, Any]:
    route = route_resolver(selection_path)
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if (reconciliation.get("artifact_type") != RECONCILIATION_SCHEMA
            or reconciliation.get("schema_version") != 1):
        raise ValueError("dispatch reconciliation has the wrong schema")
    workload = reconciliation.get("workload")
    inventory = reconciliation.get("dispatch_inventory")
    if not isinstance(workload, dict) or not isinstance(inventory, dict) \
            or inventory.get("artifact_type") != "ninfer_qwen3_8_27b_dispatch_inventory" \
            or inventory.get("schema_version") != 1 or inventory.get("workload") != workload:
        raise ValueError("dispatch inventory/workload binding is malformed")
    if (workload.get("artifact_sha256"), workload.get("executable_sha256"),
            workload.get("weights_id"), workload.get("kv_value_group"),
            workload.get("xattention_profile"), workload.get("prefill_chunk")) != (
            route["artifact"]["sha256"], route["executable"]["sha256"], route["weights_id"],
            route["kv_value_group"], route["xattention_profile"], route["prefill_chunk"]):
        raise ValueError("dispatch inventory differs from terminal selected route")
    concurrency = workload.get("concurrency")
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("selected hardware inventory concurrency must be in [1,4]")
    trace_snapshot = verify_snapshot(reconciliation.get("trace_authority"), "trace authority")
    schedule_snapshot = verify_snapshot(reconciliation.get("static_schedule"), "static schedule")
    recomputed = reconciliation_validator(
        Path(trace_snapshot["path"]), Path(schedule_snapshot["path"])
    )
    if recomputed != reconciliation:
        raise ValueError("dispatch reconciliation differs from current trace/schedule validation")
    trace = json.loads(Path(trace_snapshot["path"]).read_text(encoding="utf-8"))
    trace_route = trace.get("selected_route")
    trace_inputs = trace.get("inputs")
    if (trace.get("artifact_type") != "ninfer_r9700_selected_profile_trace"
            or trace.get("schema_version") not in (1, 2)
            or trace.get("status") != "valid_attribution_only"
            or trace.get("profile_timing_admissible") is not False
            or not isinstance(trace_route, dict) or not isinstance(trace_inputs, dict)
            or trace_route.get("artifact", {}).get("sha256") != route["artifact"]["sha256"]
            or trace_route.get("benchmark_executable", {}).get("sha256")
            != route["executable"]["sha256"]
            or verify_snapshot(trace_inputs.get("terminal_selection"), "trace terminal selection")
            != route["terminal_selection"]):
        raise ValueError("reconciliation trace is not the terminal selected route")
    dispatches = inventory.get("dispatches")
    if not isinstance(dispatches, list) or not dispatches:
        raise ValueError("selected dispatch inventory is empty")
    symbols = {row.get("symbol") for row in dispatches if isinstance(row, dict)}
    if None in symbols or len(symbols) == 0:
        raise ValueError("selected dispatch inventory contains an invalid symbol")
    inventory_rows = {row.get("dispatch_id"): row.get("symbol") for row in dispatches}
    modeled = {row.get("dispatch_id"): row.get("symbol")
               for row in reconciliation.get("dispatches", [])
               if isinstance(row, dict) and row.get("classification") == "modeled"}
    if inventory_rows != modeled or None in inventory_rows:
        raise ValueError("selected inventory is not the reconciliation's exact modeled dispatch set")
    modeled_rows = reconciliation.get("dispatch_inventory", {}).get("dispatches", [])
    has_w8 = any("a8w8g32_linear_prefill_cta_kernel" in row.get("symbol", "")
                 for row in modeled_rows if isinstance(row, dict))
    has_fp8_linear = any(row.get("operation") == "fp8_linear"
                         for row in modeled_rows if isinstance(row, dict))
    if has_w8 != (route["weights_id"] == "r9700-q4-w8-mse-n16k16-eval"):
        raise ValueError("executed W8 dispatch presence differs from selected recipe")
    if has_fp8_linear != (route["weights_id"] == HYBRID_ID):
        raise ValueError("executed FP8-linear dispatch presence differs from selected recipe")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema") != AUDIT_SCHEMA
            or audit.get("status") != "static_preflight_pass_terminal_dispatch_proof_pending"):
        raise ValueError("static audit has the wrong schema")
    expected_profile = (("xattention-s16-tau900" if route["xattention_profile"] != "dense"
                         else "dense") + f"-g{route['kv_value_group']}")
    rows = [row for row in audit.get("benchmark_executables", [])
            if isinstance(row, dict) and row.get("profile") == expected_profile]
    if len(rows) != 1 or rows[0].get("sha256") != route["executable"]["sha256"]:
        raise ValueError("static audit does not bind selected executable/profile")
    proofs = audit.get("shared_symbol_proofs")
    if not isinstance(proofs, dict):
        raise ValueError("static audit lacks symbol proofs")
    required = ["q4_p2048_cta", "q4_wave32"]
    symbol_tokens = ["a8q4g64_linear_prefill_cta_kernel",
                     "a8q4g64_linear_wmma32_kernel"]
    if route["weights_id"] == "r9700-q4-w8-mse-n16k16-eval":
        required.append("w8_p2048_cta"); symbol_tokens.append("a8w8g32_linear_prefill_cta_kernel")
    if route["xattention_profile"] == "dense":
        required.append("dense_initial_prefix_qk"); symbol_tokens.append("dense_full_score_qk")
    else:
        required.extend(("xattention_rank", "xattention_flash_consumer"))
        symbol_tokens.extend(("xattention_rank_kernel", "xattention_flash_consumer_kernel"))
    for name in required:
        if name not in proofs:
            raise ValueError(f"static audit lacks selected proof {name}")
    embedded_evidence = embedded_validator(
        Path(route["executable"]["path"]), expected_profile, audit, proofs, required)
    q4 = proofs["q4_p2048_cta"]
    if (q4.get("opcode") != "v_wmma_i32_16x16x32_iu4" or q4.get("opcode_sites") != 8
            or q4.get("vgpr") != 88 or q4.get("lds_bytes") != 17152
            or q4.get("private_bytes") != 0 or q4.get("scratch_bytes") != 0):
        raise ValueError("selected Q4 CTA static proof has the wrong IU4/resources")
    q4_dispatches = [row for row in dispatches
                     if "a8q4g64_linear_prefill_cta_kernel" in row.get("symbol", "")]
    if (not q4_dispatches or any(
            row.get("resources", {}).get("vgpr_count") != q4["vgpr"]
            or row.get("resources", {}).get("lds_bytes") != q4["lds_bytes"]
            or row.get("resources", {}).get("scratch_bytes") != q4["scratch_bytes"]
            for row in q4_dispatches)):
        raise ValueError("executed Q4 CTA resources differ from selected ELF/ISA proof")
    wave = proofs["q4_wave32"]
    if (wave.get("opcode") != "v_wmma_i32_16x16x32_iu4"
            or wave.get("opcode_sites") != 4 or wave.get("vgpr") != 64
            or wave.get("lds_bytes") != 0 or wave.get("private_bytes") != 0):
        raise ValueError("selected Q4 wave32 static proof has the wrong IU4/resources")
    ordinary_qk = proofs.get("ordinary_fp8_qk")
    if (not isinstance(ordinary_qk, dict)
            or ordinary_qk.get("fp8_wmma_opcode") != "v_wmma_f32_16x16x16_fp8_fp8"
            or ordinary_qk.get("fp8_wmma_sites") != 1
            or ordinary_qk.get("packed_fp8_conversion_opcode") != "v_cvt_pk_fp8_f32"
            or ordinary_qk.get("packed_fp8_conversion_sites") != 8):
        raise ValueError("ordinary decode QK static proof lacks native FP8 WMMA/cache encode")
    if route["weights_id"] == "r9700-q4-w8-mse-n16k16-eval":
        w8 = proofs["w8_p2048_cta"]
        if (w8.get("opcode") != "v_wmma_i32_16x16x16_iu8" or w8.get("opcode_sites") != 2
                or w8.get("vgpr") != 50 or w8.get("lds_bytes") != 4352
                or w8.get("private_bytes") != 0 or w8.get("scratch_bytes") != 0):
            raise ValueError("selected W8 CTA static proof has the wrong IU8/resources")
        w8_dispatches = [row for row in dispatches
                         if "a8w8g32_linear_prefill_cta_kernel" in row.get("symbol", "")]
        if (not w8_dispatches or any(
                row.get("resources", {}).get("vgpr_count") != w8["vgpr"]
                or row.get("resources", {}).get("lds_bytes") != w8["lds_bytes"]
                or row.get("resources", {}).get("scratch_bytes") != w8["scratch_bytes"]
                for row in w8_dispatches)):
            raise ValueError("executed W8 CTA resources differ from selected ELF/ISA proof")
    if route["xattention_profile"] == "dense":
        dense = proofs["dense_initial_prefix_qk"]
        if (dense.get("opcode") != "v_wmma_f32_16x16x16_bf16"
                or dense.get("opcode_sites") != 16):
            raise ValueError("selected dense attention proof lacks BF16 WMMA")
    else:
        rank, consumer = proofs["xattention_rank"], proofs["xattention_flash_consumer"]
        selected_symbol = consumer.get(f"g{route['kv_value_group']}_code_symbol")
        if (rank.get("opcode") != "v_wmma_f32_16x16x16_bf16"
                or rank.get("opcode_sites") != 2
                or consumer.get("opcode") != "v_wmma_f32_16x16x16_bf16"
                or consumer.get("opcode_sites_each") != 16
                or not isinstance(selected_symbol, str)):
            raise ValueError("selected XAttention proof lacks its G16/G32 BF16 WMMA path")
    for token in symbol_tokens:
        if not any(token in symbol for symbol in symbols):
            raise ValueError(f"selected inventory did not execute required symbol {token}")

    hybrid = route["weights_id"] == HYBRID_ID
    if hybrid != (len(fp8_paths) == 2):
        raise ValueError("hybrid selection requires exactly two loaded-ELF FP8 proofs")
    fp8 = []
    if hybrid:
        loaded_objects = trace.get("loaded_fp8_code_objects")
        if not isinstance(loaded_objects, list) or len(loaded_objects) != 2:
            raise ValueError("hybrid trace lacks exactly two selected loaded FP8 code objects")
        capture = trace.get("code_object_capture")
        if not isinstance(capture, dict) or set(capture) != {"library", "directory", "scope"}:
            raise ValueError("hybrid trace lacks its exact code-object capture identity")
        _proof_file(capture["library"], "selected trace capture library")
        capture_dir = Path(str(capture["directory"])).resolve(strict=True)
        database = verify_snapshot(trace_inputs.get("database"), "trace database")
        if loaded_fp8_validator(Path(database["path"]), capture_dir,
                                trace.get("dispatches", [])) != loaded_objects:
            raise ValueError("selected FP8 loaded objects differ from current DB/URI capture")
        by_qualification = {}
        for path in fp8_paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            qualification = value.get("qualification")
            if qualification in by_qualification:
                raise ValueError("duplicate FP8 qualification proof")
            by_qualification[qualification] = path
        if set(by_qualification) != {"gate_up", "attention_qk_gate_value"}:
            raise ValueError("hybrid FP8 proof set is incomplete")
        fp8 = [_fp8_proof(by_qualification[name], name, dispatches, loaded_objects,
                          fp8_resource_validator)
               for name in ("gate_up", "attention_qk_gate_value")]
    elif trace.get("loaded_fp8_code_objects") not in (None, []):
        raise ValueError("non-hybrid trace must not carry loaded FP8 code objects")
    elif trace.get("code_object_capture") is not None:
        raise ValueError("non-hybrid trace must not carry code-object capture identity")

    return {
        "artifact_type": OUTPUT_SCHEMA, "schema_version": 1, "status": "passed",
        "selected_route": route,
        "authorities": {"dispatch_reconciliation": snapshot(reconciliation_path),
                        "trace": trace_snapshot,
                        "static_audit": snapshot(audit_path)},
        "executed_symbol_count": len(symbols), "required_symbol_families": symbol_tokens,
        "static_proof_names": required + ["ordinary_fp8_qk"],
        "selected_embedded_static_proofs": embedded_evidence,
        "static_only_proof_names": ["ordinary_fp8_qk"], "loaded_fp8_proofs": fp8,
        "physical_bandwidth_claim": None, "stall_freedom_claim": None,
        "note": "Static ISA and executed symbol ownership only; no physical bandwidth inference.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--reconciliation", required=True, type=Path)
    parser.add_argument("--static-audit", required=True, type=Path)
    parser.add_argument("--fp8-proof", action="append", default=[], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = args.out.parent.resolve(strict=True) / args.out.name
        if os.path.lexists(output):
            raise ValueError(f"refusing to overwrite {output}")
        value = verify(args.selection, args.reconciliation, args.static_audit, args.fp8_proof)
        revalidated = verify(args.selection, args.reconciliation, args.static_audit,
                             args.fp8_proof)
        if revalidated != value:
            raise ValueError("selected hardware-use inputs changed before publication")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        published = False
        durable = False
        created_inode = None
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2); stream.write("\n")
                stream.flush(); os.fsync(stream.fileno())
            temporary_stat = os.stat(temporary, follow_symlinks=False)
            created_inode = (temporary_stat.st_dev, temporary_stat.st_ino)
            os.link(temporary, output)
            published = True
            output_stat = os.stat(output, follow_symlinks=False)
            if (not stat.S_ISREG(output_stat.st_mode)
                    or (output_stat.st_dev, output_stat.st_ino) != created_inode):
                raise ValueError("published selected hardware-use inode changed")
            if (json.loads(output.read_text(encoding="utf-8")) != value
                    or verify(args.selection, args.reconciliation, args.static_audit,
                              args.fp8_proof) != value):
                raise ValueError("published selected hardware-use authority does not revalidate")
            directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            durable = True
        finally:
            try: os.unlink(temporary)
            except FileNotFoundError: pass
            if published and not durable and created_inode is not None:
                try:
                    current = os.stat(output, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == created_inode:
                        output.unlink()
                except FileNotFoundError:
                    pass
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
