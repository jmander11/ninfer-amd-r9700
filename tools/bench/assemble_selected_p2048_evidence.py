#!/usr/bin/env python3
"""Assemble one provenance-bound selected-P2048 operation evidence authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from tools.bench.prepare_whole_profile import DISPATCH_COUNTERS


RECONCILIATION_TYPE = "ninfer_qwen3_8_27b_dispatch_reconciliation"
ROOFLINE_TYPE = "ninfer_qwen3_8_27b_roofline"
PMC_TYPE = "ninfer_r9700_selected_profile_pmc"
STATIC_TYPE = "ninfer_r9700_operation_static_evidence"
OUTPUT_TYPE = "ninfer_r9700_selected_p2048_evidence"
SCHEMA_VERSION = 1
CLASSIFICATIONS = ("matrix", "scalar_valu", "memory_control")
RESIDENCY = ("register_reuse", "lds_reuse", "cache_streaming", "metadata_control")
OVERLAP = ("proven_dependency_safe", "not_applicable", "unproven")
WORKLOAD_FIELDS = (
    "prompt_tokens", "concurrency", "prefill_chunk", "kv_value_group",
    "xattention_profile",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved)}


def _verify_snapshot(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{label} must be a file snapshot")
    actual = _snapshot(Path(value["path"]), label)
    size = value.get("file_size_bytes", value.get("bytes"))
    if (value.get("path") != actual["path"] or value.get("sha256") != actual["sha256"]
            or type(size) is not int or size != actual["file_size_bytes"]):
        raise ValueError(f"{label} path, size, or SHA-256 changed")
    return actual


def _verify_nested_snapshots(value: Any, label: str,
                             tracked: list[tuple[Path, dict[str, Any]]]) -> None:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value and (
                "file_size_bytes" in value or "bytes" in value):
            actual = _verify_snapshot(value, label)
            tracked.append((Path(actual["path"]), actual))
            return
        for name, child in value.items():
            _verify_nested_snapshots(child, f"{label}.{name}", tracked)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_nested_snapshots(child, f"{label}[{index}]", tracked)


def _schema(value: dict[str, Any], artifact_type: str, label: str) -> None:
    if (value.get("artifact_type") != artifact_type
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1):
        raise ValueError(f"{label} must be {artifact_type} schema v1")


def _workload_matches(reconciliation: dict[str, Any], roofline: dict[str, Any],
                      pmc: dict[str, Any]) -> dict[str, Any]:
    workload = reconciliation.get("workload")
    if not isinstance(workload, dict) or roofline.get("workload") != workload:
        raise ValueError("roofline workload differs from reconciliation")
    if (workload.get("kind"), workload.get("prompt_tokens"), workload.get("concurrency"),
            workload.get("xattention_profile")) != ("pp", 2048, 1, "dense"):
        raise ValueError("evidence must describe selected dense C1 P2048")
    pmc_workload = pmc.get("workload")
    if not isinstance(pmc_workload, dict) or any(
            pmc_workload.get(field) != workload.get(field) for field in WORKLOAD_FIELDS):
        raise ValueError("PMC workload differs from reconciled selected route")
    inputs = pmc.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("PMC evidence lacks input identities")
    artifact, executable = inputs.get("artifact"), inputs.get("benchmark_executable")
    if (not isinstance(artifact, dict) or not isinstance(executable, dict)
            or artifact.get("path") != workload.get("artifact_path")
            or artifact.get("sha256") != workload.get("artifact_sha256")
            or executable.get("path") != workload.get("executable_path")
            or executable.get("sha256") != workload.get("executable_sha256")):
        raise ValueError("PMC artifact/executable differs from selected route")
    return workload


def _static_report(value: dict[str, Any], workload: dict[str, Any], label: str) -> dict[str, Any]:
    _schema(value, STATIC_TYPE, label)
    evidence_id = _text(value.get("evidence_id"), f"{label}.evidence_id")
    route = value.get("selected_route")
    expected_route = {field: workload.get(field) for field in WORKLOAD_FIELDS}
    expected_route.update({"artifact_sha256": workload.get("artifact_sha256"),
                           "executable_sha256": workload.get("executable_sha256")})
    if route != expected_route:
        raise ValueError(f"{label}.selected_route differs from selected P2048 route")
    operation = _text(value.get("operation_family"), f"{label}.operation_family")
    specialization = _text(value.get("specialization"), f"{label}.specialization")
    signatures = value.get("dispatch_signatures")
    if not isinstance(signatures, list) or not signatures:
        raise ValueError(f"{label}.dispatch_signatures must be nonempty")
    exact_signatures = []
    for index, signature in enumerate(signatures):
        if not isinstance(signature, dict) or set(signature) != {"stage", "symbol"}:
            raise ValueError(f"{label}.dispatch_signatures[{index}] must contain stage/symbol")
        stage = signature["stage"]
        if stage is not None:
            stage = _text(stage, "signature.stage")
        exact_signatures.append((stage, _text(signature["symbol"], "signature.symbol")))
    if len(set(exact_signatures)) != len(exact_signatures):
        raise ValueError(f"{label}.dispatch_signatures contains duplicates")

    hardware = value.get("intended_hardware")
    if not isinstance(hardware, dict) or hardware.get("classification") not in CLASSIFICATIONS:
        raise ValueError(f"{label}.intended_hardware classification is invalid")
    arithmetic = _text(hardware.get("arithmetic"), f"{label}.intended_hardware.arithmetic")
    opcodes = hardware.get("expected_opcodes")
    if not isinstance(opcodes, list) or not opcodes or not all(
            isinstance(item, str) and item for item in opcodes) or len(set(opcodes)) != len(opcodes):
        raise ValueError(f"{label}.expected_opcodes must be unique and nonempty")
    if hardware["classification"] == "matrix" and not any(
            "wmma" in opcode.lower() for opcode in opcodes):
        raise ValueError(f"{label} matrix specialization lacks an exact WMMA opcode")
    if hardware["classification"] != "matrix" and any(
            "wmma" in opcode.lower() for opcode in opcodes):
        raise ValueError(f"{label} non-matrix classification cannot claim WMMA")

    proof = value.get("static_proof")
    if not isinstance(proof, dict) or proof.get("status") != "passed":
        raise ValueError(f"{label}.static_proof must be passed")
    _text(proof.get("code_symbol"), f"{label}.static_proof.code_symbol")
    embedding = proof.get("executable_embedding")
    if (not isinstance(embedding, dict)
            or set(embedding) != {"offset_bytes", "file_size_bytes", "occurrence_count"}
            or _integer(embedding.get("offset_bytes"), f"{label}.embedding.offset_bytes") < 0
            or _integer(embedding.get("file_size_bytes"),
                        f"{label}.embedding.file_size_bytes", 1) < 1
            or embedding.get("occurrence_count") != 1):
        raise ValueError(f"{label} lacks an exact executable/code-object embedding proof")
    counts = proof.get("opcode_counts")
    if (not isinstance(counts, dict) or set(counts) != set(opcodes)
            or any(_integer(count, f"{label}.opcode_counts[{opcode}]", 1) < 1
                   for opcode, count in counts.items())):
        raise ValueError(f"{label}.opcode_counts must prove every expected opcode")
    resources = proof.get("resources")
    required_resources = {"lds_bytes", "vgpr_count", "private_bytes", "scratch_bytes",
                          "flat_scratch"}
    if not isinstance(resources, dict) or set(resources) != required_resources:
        raise ValueError(f"{label}.resources must contain the exact resource proof")
    for name in resources:
        _integer(resources[name], f"{label}.resources.{name}")
    if (resources["private_bytes"] != 0 or resources["scratch_bytes"] != 0
            or resources["flat_scratch"] != 0 or proof.get("zero_scratch") is not True):
        raise ValueError(f"{label} lacks zero private/scratch proof")
    inputs = proof.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
            "code_object", "assembly", "metadata", "checker", "sources"}:
        raise ValueError(
            f"{label}.static_proof.inputs must contain code_object/assembly/metadata/checker/sources")
    if not all(isinstance(inputs[name], dict)
               for name in ("code_object", "assembly", "metadata", "checker")):
        raise ValueError(f"{label}.static_proof executable inputs must be snapshots")
    if not isinstance(inputs["sources"], list) or not inputs["sources"]:
        raise ValueError(f"{label}.static_proof.inputs.sources must be nonempty")
    if embedding["file_size_bytes"] != inputs["code_object"].get("file_size_bytes"):
        raise ValueError(f"{label} code-object embedding size differs from its snapshot")
    executable_bytes = Path(workload["executable_path"]).read_bytes()
    code_bytes = Path(inputs["code_object"]["path"]).read_bytes()
    offset = embedding["offset_bytes"]
    if (executable_bytes[offset:offset + len(code_bytes)] != code_bytes
            or executable_bytes.find(code_bytes) != offset
            or executable_bytes.find(code_bytes, offset + 1) >= 0):
        raise ValueError(f"{label} code object is not uniquely embedded in the selected executable")

    memory = value.get("memory_path")
    if (not isinstance(memory, dict) or memory.get("residency") not in RESIDENCY
            or memory.get("overlap") not in OVERLAP):
        raise ValueError(f"{label}.memory_path classification is invalid")
    _text(memory.get("access_pattern"), f"{label}.memory_path.access_pattern")
    _text(memory.get("evidence"), f"{label}.memory_path.evidence")
    if memory["overlap"] == "proven_dependency_safe":
        _verify_snapshot(memory.get("overlap_evidence"),
                         f"{label}.memory_path.overlap_evidence")
    return {"evidence_id": evidence_id, "operation_family": operation,
            "specialization": specialization, "signatures": exact_signatures,
            "intended_hardware": hardware, "static_proof": proof, "memory_path": memory}


def assemble(reconciliation_path: Path, roofline_path: Path, pmc_path: Path,
             static_paths: list[Path]) -> dict[str, Any]:
    if not static_paths:
        raise ValueError(
            "at least one ninfer_r9700_operation_static_evidence schema-v1 report is required")
    input_paths = [(reconciliation_path, "dispatch reconciliation"),
                   (roofline_path, "roofline report"), (pmc_path, "PMC evidence")]
    input_snapshots = {label: _snapshot(path, label) for path, label in input_paths}
    static_snapshots = [_snapshot(path, f"static report {index}")
                        for index, path in enumerate(static_paths)]
    reconciliation = _load(reconciliation_path, "dispatch reconciliation")
    roofline = _load(roofline_path, "roofline report")
    pmc = _load(pmc_path, "PMC evidence")
    _schema(reconciliation, RECONCILIATION_TYPE, "dispatch reconciliation")
    _schema(roofline, ROOFLINE_TYPE, "roofline report")
    _schema(pmc, PMC_TYPE, "PMC evidence")
    if (pmc.get("status") != "valid_attribution_only"
            or pmc.get("profile_timing_admissible") is not False):
        raise ValueError("PMC evidence must be attribution-only")
    workload = _workload_matches(reconciliation, roofline, pmc)
    if roofline.get("dispatch_reconciliation") != input_snapshots["dispatch reconciliation"]:
        raise ValueError("roofline does not bind the exact dispatch reconciliation")
    physical = roofline.get("physical_measurement")
    if (not isinstance(physical, dict) or any(physical.get(field) is not None for field in
            ("hbm_bytes", "hbm_bandwidth_gbps", "hbm_peak_fraction", "stall_fraction"))):
        raise ValueError("roofline must retain unavailable physical HBM/stall fields as null")
    auto = reconciliation.get("timing_authority", {}).get("power_profile")
    pmc_power = pmc.get("power_profile")
    if (not isinstance(auto, dict) or any(auto.get(field) != "auto" for field in
            ("required", "observed", "rechecked_after"))
            or not isinstance(pmc_power, dict)
            or pmc_power.get("required") != "profile_standard"
            or pmc_power.get("before") != "profile_standard"
            or pmc_power.get("after") != "auto"):
        raise ValueError("auto trace and profile_standard-to-auto PMC power semantics are not exact")

    tracked: list[tuple[Path, dict[str, Any]]] = []
    for label, document in (("reconciliation", reconciliation), ("roofline", roofline),
                            ("PMC", pmc)):
        _verify_nested_snapshots(document, label, tracked)

    reports: dict[str, dict[str, Any]] = {}
    signature_owner: dict[tuple[str, str], str] = {}
    for index, (path, snapshot) in enumerate(zip(static_paths, static_snapshots, strict=True)):
        value = _load(path, f"static report {index}")
        _verify_nested_snapshots(value, f"static report {index}", tracked)
        report = _static_report(value, workload, f"static report {index}")
        evidence_id = report["evidence_id"]
        if evidence_id in reports:
            raise ValueError(f"duplicate static evidence_id: {evidence_id}")
        reports[evidence_id] = {**report, "authority": snapshot}
        for signature in report["signatures"]:
            if signature in signature_owner:
                raise ValueError(f"dispatch signature has multiple static owners: {signature}")
            signature_owner[signature] = evidence_id

    dispatches = reconciliation.get("dispatches")
    if not isinstance(dispatches, list) or not dispatches:
        raise ValueError("reconciliation dispatches must be nonempty")
    operation_totals: dict[str, dict[str, Any]] = {
        evidence_id: {"dispatch_count": 0, "duration_ns": 0,
                      "modeled_dispatch_count": 0, "uncovered_dispatch_count": 0,
                      "stages": set()}
        for evidence_id in reports
    }
    assigned = []
    for index, row in enumerate(dispatches):
        if not isinstance(row, dict):
            raise ValueError(f"reconciliation dispatch {index} is not an object")
        marker = row.get("marker")
        if marker is not None:
            marker = _text(marker, f"dispatch {index}.marker")
        signature = (marker, _text(row.get("symbol"), f"dispatch {index}.symbol"))
        evidence_id = signature_owner.get(signature)
        if evidence_id is None:
            raise ValueError(
                f"executed dispatch lacks intended-hardware static evidence: {signature}")
        report = reports[evidence_id]
        if row.get("classification") == "modeled":
            if row.get("operation") != report["operation_family"]:
                raise ValueError(f"modeled dispatch operation differs from {evidence_id}")
            operation_totals[evidence_id]["modeled_dispatch_count"] += 1
        elif row.get("classification") in ("unmodeled", "unsupported"):
            operation_totals[evidence_id]["uncovered_dispatch_count"] += 1
        else:
            raise ValueError(f"dispatch {index} has invalid reconciliation classification")
        duration = _integer(row.get("duration_ns"), f"dispatch {index}.duration_ns", 1)
        operation_totals[evidence_id]["dispatch_count"] += 1
        operation_totals[evidence_id]["duration_ns"] += duration
        operation_totals[evidence_id]["stages"].add(signature[0])
        assigned.append({"dispatch_id": row.get("dispatch_id"), "evidence_id": evidence_id,
                         "classification": row["classification"], "duration_ns": duration})
    unused = sorted(name for name, total in operation_totals.items()
                    if total["dispatch_count"] == 0)
    if unused:
        raise ValueError(f"static reports do not describe an executed dispatch: {unused}")

    roofline_ids = {row.get("dispatch_id") for row in roofline.get("dispatches", [])
                    if isinstance(row, dict)}
    uncovered_ids = {row.get("dispatch_id") for row in roofline.get("uncovered_dispatches", [])
                     if isinstance(row, dict)}
    expected_modeled = {row["dispatch_id"] for row in dispatches
                        if row.get("classification") == "modeled"}
    expected_uncovered = {row["dispatch_id"] for row in dispatches
                          if row.get("classification") != "modeled"}
    if roofline_ids != expected_modeled or uncovered_ids != expected_uncovered:
        raise ValueError("roofline modeled/uncovered dispatch coverage differs")
    if roofline.get("dispatch_coverage") != reconciliation.get("coverage"):
        raise ValueError("roofline count/duration coverage differs from reconciliation")

    stage_join = pmc.get("roctx_stage_join")
    if not isinstance(stage_join, dict) or stage_join.get("state") != "exact_same_capture":
        raise ValueError("PMC evidence lacks an exact same-capture ROCTX stage join")
    pmc_dispatches = stage_join.get("dispatches")
    if not isinstance(pmc_dispatches, list) or not pmc_dispatches:
        raise ValueError("PMC stage join must retain every captured dispatch symbol")
    pmc_signature_counts: Counter[tuple[str, str]] = Counter()
    pmc_stage_counts: Counter[str] = Counter()
    pmc_dispatch_ids: set[int] = set()
    for index, row in enumerate(pmc_dispatches):
        if not isinstance(row, dict) or set(row) != {"dispatch_id", "stage", "symbol"}:
            raise ValueError(f"PMC dispatch {index} must contain dispatch_id/stage/symbol")
        dispatch_id = _integer(row.get("dispatch_id"), f"PMC dispatch {index}.dispatch_id")
        if dispatch_id in pmc_dispatch_ids:
            raise ValueError(f"duplicate PMC dispatch_id: {dispatch_id}")
        pmc_dispatch_ids.add(dispatch_id)
        stage = _text(row.get("stage"), f"PMC dispatch {index}.stage")
        symbol = _text(row.get("symbol"), f"PMC dispatch {index}.symbol")
        signature = (stage, symbol)
        if signature not in signature_owner:
            raise ValueError(f"PMC dispatch lacks trace/static signature ownership: {signature}")
        pmc_signature_counts[signature] += 1
        pmc_stage_counts[stage] += 1
    trace_signature_counts = Counter((row["marker"], row["symbol"]) for row in dispatches)
    if any(trace_signature_counts[signature] != count
           for signature, count in pmc_signature_counts.items()):
        raise ValueError("PMC captured signature multiplicity differs from the selected trace")
    pmc_stages = stage_join.get("stages")
    if not isinstance(pmc_stages, list):
        raise ValueError("PMC stage aggregates must be an array")
    captured_stages = set(pmc_stage_counts)
    if ({row.get("stage") for row in pmc_stages if isinstance(row, dict)} != captured_stages
            or len(pmc_stages) != len(captured_stages)):
        raise ValueError("PMC stage aggregates do not exactly cover captured dispatches")
    pmc_attribution = []
    for row in pmc_stages:
        stage = _text(row.get("stage"), "PMC stage")
        evidence_ids = sorted({signature_owner[signature]
                               for signature in pmc_signature_counts if signature[0] == stage})
        dispatch_count = _integer(
            row.get("dispatch_count"), f"PMC stage {stage}.dispatch_count", 1)
        if dispatch_count != pmc_stage_counts[stage]:
            raise ValueError(f"PMC stage {stage} dispatch count differs from its exact join")
        counter_sums = row.get("counter_sums")
        if not isinstance(counter_sums, dict) or set(counter_sums) != set(DISPATCH_COUNTERS):
            raise ValueError(f"PMC stage {stage} lacks the exact counter sums")
        pmc_attribution.append({"stage": stage, "operation_evidence_ids": evidence_ids,
                                "dispatch_count": dispatch_count,
                                "counter_sums": counter_sums,
                                "gl2_hit_ratio": row.get("gl2_hit_ratio"),
                                "tcp_hit_ratio": row.get("tcp_hit_ratio"),
                                "join_basis": (
                                    "same-capture ROCTX stage plus exact profiler display-symbol "
                                    "multiplicity; never cross-capture dispatch ID")})

    for path, snapshot in tracked:
        if _snapshot(path, str(path)) != snapshot:
            raise ValueError(f"bound authority changed while assembling: {path}")
    for path, label in input_paths:
        if _snapshot(path, label) != input_snapshots[label]:
            raise ValueError(f"{label} changed while assembling")
    for path, snapshot in zip(static_paths, static_snapshots, strict=True):
        if _snapshot(path, str(path)) != snapshot:
            raise ValueError(f"static report changed while assembling: {path}")

    operation_evidence = []
    for evidence_id, report in sorted(reports.items()):
        totals = operation_totals[evidence_id]
        operation_evidence.append({
            "evidence_id": evidence_id, "authority": report["authority"],
            "operation_family": report["operation_family"],
            "specialization": report["specialization"],
            "intended_hardware": report["intended_hardware"],
            "static_proof": report["static_proof"], "memory_path": report["memory_path"],
            **{key: value for key, value in totals.items() if key != "stages"},
            "stages": sorted(totals["stages"], key=lambda value: "" if value is None else value),
        })
    has_unproven_overlap = any(
        report["memory_path"]["overlap"] == "unproven" for report in reports.values())
    return {
        "artifact_type": OUTPUT_TYPE, "schema_version": SCHEMA_VERSION,
        "status": ("assembled_with_unproven_overlap" if has_unproven_overlap else
                   "complete_with_explicit_uncovered" if expected_uncovered else
                   "complete_modeled"),
        "inputs": {**input_snapshots, "static_reports": static_snapshots},
        "selected_route": workload,
        "timing_semantics": {
            "selection_performance": {
                "power_profile": "auto", "timing_kind": "unprofiled_3_repetitions_1_warmup",
                "authority": roofline.get("unprofiled_whole_p2048"),
            },
            "dispatch_attribution": "auto trace durations; attribution only, not selection timing",
            "cache_activity": "profile_standard PMC stage ratios; attribution only",
        },
        "profiled_roofline_attribution": {
            "timing": roofline.get("timing"),
            "dispatches": roofline.get("dispatches"),
            "aggregates_by_operation": roofline.get("aggregates_by_operation"),
            "aggregates_by_stage": roofline.get("aggregates_by_stage"),
        },
        "coverage": reconciliation["coverage"], "dispatch_assignments": assigned,
        "operation_evidence": operation_evidence, "pmc_stage_attribution": pmc_attribution,
        "physical_measurement": {
            "hbm_bytes": None, "hbm_bandwidth_gbps": None,
            "hbm_peak_fraction": None, "stall_fraction": None,
            "status": "unavailable on the retained gfx1201 ROCm counter path",
            "gl2_tcp_note": "relative hit ratios only; not physical HBM or stall evidence",
        },
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(value, indent=2) + "\n")
            output.flush(); os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def validate(record_path: Path, reconciliation_path: Path, roofline_path: Path, pmc_path: Path,
             static_paths: list[Path]) -> dict[str, Any]:
    record_snapshot = _snapshot(record_path, "selected-P2048 evidence")
    record = _load(record_path, "selected-P2048 evidence")
    _schema(record, OUTPUT_TYPE, "selected-P2048 evidence")
    expected = assemble(reconciliation_path, roofline_path, pmc_path, static_paths)
    if record != expected:
        raise ValueError("selected-P2048 evidence differs from exact reconstructed authority")
    if _snapshot(record_path, "selected-P2048 evidence") != record_snapshot:
        raise ValueError("selected-P2048 evidence changed while validating")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-reconciliation", required=True, type=Path)
    parser.add_argument("--roofline", required=True, type=Path)
    parser.add_argument("--pmc", required=True, type=Path)
    parser.add_argument("--static-report", required=True, action="append", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.out is not None and os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    try:
        if args.validate is not None:
            validate(args.validate, args.dispatch_reconciliation, args.roofline, args.pmc,
                     args.static_report)
            print(f"validated selected-P2048 operation evidence: {args.validate.resolve()}")
            return 0
        value = assemble(
            args.dispatch_reconciliation, args.roofline, args.pmc, args.static_report)
        assert args.out is not None
        _publish(args.out, value)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote selected-P2048 operation evidence to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
