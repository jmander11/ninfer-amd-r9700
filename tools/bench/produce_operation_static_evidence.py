#!/usr/bin/env python3
"""Produce one strict selected-P2048 operation ISA/resource evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.r9700 import check_prefill_cta_static as cta_checker_module
from tools.r9700.check_prefill_cta_static import (
    PROFILES as CTA_PROFILES,
    _function,
    _one_integer,
    check as check_cta,
)


ARTIFACT_TYPE = "ninfer_r9700_operation_static_evidence"
SCHEMA_VERSION = 1
CLASSIFICATIONS = ("matrix", "scalar_valu", "memory_control")
RESIDENCY = ("register_reuse", "lds_reuse", "cache_streaming", "metadata_control")
OVERLAP = ("proven_dependency_safe", "not_applicable", "unproven")
CTA_RECIPES = ("q4", "w8")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved)}


def _positive(value: int, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _embedded_code_object(executable: Path, code_object: Path) -> dict[str, int]:
    executable_bytes = executable.read_bytes()
    code_bytes = code_object.read_bytes()
    if not code_bytes:
        raise ValueError("selected code object must be nonempty")
    first = executable_bytes.find(code_bytes)
    if first < 0 or executable_bytes.find(code_bytes, first + 1) >= 0:
        raise ValueError(
            "selected code object must occur exactly once byte-for-byte in the selected executable")
    return {"offset_bytes": first, "file_size_bytes": len(code_bytes),
            "occurrence_count": 1}


def _opcode_spec(value: str) -> tuple[str, int]:
    opcode, separator, count = value.rpartition("=")
    if not separator or not opcode or not re.fullmatch(r"[A-Za-z0-9_.]+", opcode):
        raise argparse.ArgumentTypeError("opcode must be NAME=POSITIVE_COUNT")
    try:
        parsed = int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("opcode count must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("opcode count must be positive")
    return opcode, parsed


def _resources(metadata_body: str) -> dict[str, int]:
    return {
        "lds_bytes": _one_integer(
            metadata_body, r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)", "LDS size"),
        "private_bytes": _one_integer(
            metadata_body, r"^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)",
            "private segment size"),
        "vgpr_count": _one_integer(
            metadata_body, r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)", "VGPR count"),
        "flat_scratch": _one_integer(
            metadata_body, r"^\s*\.set\s+\S+\.uses_flat_scratch,\s*(\d+)",
            "flat-scratch use"),
        "scratch_bytes": _one_integer(
            metadata_body, r"^;\s*ScratchSize:\s*(\d+)", "scratch size"),
    }


def produce(
    *, artifact: Path, executable: Path, code_object: Path, assembly: Path, metadata: Path,
    sources: list[Path], stage: str | list[str] | None, dispatch_symbol: str, code_symbol: str,
    operation_family: str,
    specialization: str, hardware_classification: str, arithmetic: str,
    opcodes: list[tuple[str, int]], max_lds_bytes: int, max_vgpr_count: int,
    memory_residency: str, memory_access: str, overlap: str, memory_evidence: str,
    overlap_evidence: Path | None, prefill_chunk: int, kv_value_group: int,
    xattention_profile: str, cta_recipe: str | None = None,
) -> dict[str, Any]:
    if hardware_classification not in CLASSIFICATIONS:
        raise ValueError("invalid intended-hardware classification")
    if memory_residency not in RESIDENCY or overlap not in OVERLAP:
        raise ValueError("invalid memory residency or overlap classification")
    if not all(isinstance(value, str) and value for value in
               (dispatch_symbol, code_symbol, operation_family, specialization, arithmetic, memory_access,
                memory_evidence)):
        raise ValueError("symbol, operation, specialization, arithmetic, and memory rationale are required")
    stages = stage if isinstance(stage, list) else [stage]
    if (not stages or any(item is not None and (not isinstance(item, str) or not item)
                          for item in stages)
            or len(set(stages)) != len(stages)
            or (len(stages) > 1 and None in stages)):
        raise ValueError("stages must be unique nonempty strings or one explicitly unmarked stage")
    if overlap == "proven_dependency_safe" and overlap_evidence is None:
        raise ValueError("proven dependency-safe overlap requires an explicit evidence file")
    if overlap != "proven_dependency_safe" and overlap_evidence is not None:
        raise ValueError("overlap evidence is allowed only for proven dependency-safe overlap")
    if not opcodes or len({name for name, _ in opcodes}) != len(opcodes):
        raise ValueError("expected opcodes must be nonempty and unique")
    if hardware_classification == "matrix" and not any(
            "wmma" in name.lower() for name, _ in opcodes):
        raise ValueError("matrix classification requires an exact WMMA opcode")
    if hardware_classification != "matrix" and any(
            "wmma" in name.lower() for name, _ in opcodes):
        raise ValueError("scalar/VALU or memory/control classification cannot claim WMMA")
    _positive(prefill_chunk, "prefill_chunk")
    if kv_value_group not in (16, 32):
        raise ValueError("kv_value_group must be 16 or 32")
    if xattention_profile != "dense":
        raise ValueError("selected-P2048 static evidence currently requires dense attention")
    if overlap == "proven_dependency_safe":
        if cta_recipe != "q4":
            raise ValueError(
                "proven dependency-safe overlap is currently supported only by the exact "
                "Q4 CTA checker")
        if overlap_evidence.resolve(strict=True) != assembly.resolve(strict=True):
            raise ValueError("Q4 CTA overlap evidence must be the exact checked assembly")
    _nonnegative(max_lds_bytes, "max_lds_bytes")
    _positive(max_vgpr_count, "max_vgpr_count")

    source_paths = list(sources)
    if cta_recipe is not None:
        delegated_source = Path(cta_checker_module.__file__).resolve()
        if all(path.resolve() != delegated_source for path in source_paths):
            source_paths.append(delegated_source)
    inputs = {
        "artifact": _snapshot(artifact, "artifact"),
        "executable": _snapshot(executable, "selected executable"),
        "code_object": _snapshot(code_object, "selected code object"),
        "assembly": _snapshot(assembly, "assembly"),
        "metadata": _snapshot(metadata, "metadata"),
        "checker": _snapshot(Path(__file__), "static evidence producer"),
        "sources": [_snapshot(path, f"source {index}")
                    for index, path in enumerate(source_paths)],
    }
    if not inputs["sources"]:
        raise ValueError("at least one explicit implementation source is required")
    overlap_snapshot = (_snapshot(overlap_evidence, "dependency-safe overlap evidence")
                        if overlap_evidence is not None else None)
    executable_embedding = _embedded_code_object(executable, code_object)
    assembly_text = assembly.read_text(encoding="utf-8")
    metadata_text = metadata.read_text(encoding="utf-8")
    assembly_body = _function(assembly_text, code_symbol, "assembly")
    metadata_body = _function(metadata_text, code_symbol, "metadata")
    observed = {
        opcode: len(re.findall(rf"^\s*{re.escape(opcode)}(?:\s|$)", assembly_body,
                               flags=re.MULTILINE))
        for opcode, _ in opcodes
    }
    expected = dict(opcodes)
    if observed != expected:
        raise ValueError(f"selected symbol opcode counts differ: observed={observed} expected={expected}")
    if (hardware_classification != "matrix"
            and re.search(r"^\s*v_wmma_", assembly_body,
                          flags=re.MULTILINE | re.IGNORECASE)):
        raise ValueError("non-matrix classification selected a symbol containing WMMA instructions")
    resources = _resources(metadata_body)
    if resources["lds_bytes"] > max_lds_bytes or resources["vgpr_count"] > max_vgpr_count:
        raise ValueError(
            f"selected resources exceed ceilings: LDS {resources['lds_bytes']}/{max_lds_bytes}, "
            f"VGPR {resources['vgpr_count']}/{max_vgpr_count}")
    if any(resources[name] != 0 for name in ("private_bytes", "scratch_bytes", "flat_scratch")):
        raise ValueError("selected specialization must have zero private/scratch/flat-scratch")

    if cta_recipe is not None:
        if cta_recipe not in CTA_RECIPES:
            raise ValueError("CTA recipe must be q4 or w8")
        profile = CTA_PROFILES[cta_recipe]
        if (code_symbol != profile.production_symbol or hardware_classification != "matrix"
                or expected != {profile.opcode: profile.opcode_count}
                or max_lds_bytes != profile.lds_ceiling
                or max_vgpr_count != profile.vgpr_ceiling):
            raise ValueError("CTA declaration differs from the exact challenger profile")
        delegated = check_cta(cta_recipe, "lds-scope", assembly, metadata)
        if (delegated["opcode_count"] != observed[profile.opcode]
                or delegated["lds_bytes"] != resources["lds_bytes"]
                or delegated["vgpr_count"] != resources["vgpr_count"]):
            raise ValueError("CTA checker and producer observations differ")

    # Rehash every explicit input after all parsing and delegated validation.
    for name in ("artifact", "executable", "code_object", "assembly", "metadata", "checker"):
        if _snapshot(Path(inputs[name]["path"]), name) != inputs[name]:
            raise ValueError(f"{name} changed while producing static evidence")
    for index, snapshot in enumerate(inputs["sources"]):
        if _snapshot(Path(snapshot["path"]), f"source {index}") != snapshot:
            raise ValueError(f"source {index} changed while producing static evidence")
    if (overlap_snapshot is not None
            and _snapshot(Path(overlap_snapshot["path"]), "dependency-safe overlap evidence")
            != overlap_snapshot):
        raise ValueError("dependency-safe overlap evidence changed while producing static evidence")

    selected_route = {
        "prompt_tokens": 2048, "concurrency": 1, "prefill_chunk": prefill_chunk,
        "kv_value_group": kv_value_group, "xattention_profile": xattention_profile,
        "artifact_sha256": inputs["artifact"]["sha256"],
        "executable_sha256": inputs["executable"]["sha256"],
    }
    proof_inputs = {name: inputs[name] for name in
                    ("code_object", "assembly", "metadata", "checker", "sources")}
    memory_path = {"residency": memory_residency, "access_pattern": memory_access,
                   "overlap": overlap, "evidence": memory_evidence}
    if overlap_evidence is not None:
        memory_path["overlap_evidence"] = overlap_snapshot
    return {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "evidence_id": specialization, "selected_route": selected_route,
        "operation_family": operation_family, "specialization": specialization,
        "dispatch_signatures": [{"stage": item, "symbol": dispatch_symbol}
                                for item in stages],
        "intended_hardware": {"classification": hardware_classification,
                              "arithmetic": arithmetic,
                              "expected_opcodes": [name for name, _ in opcodes]},
        "static_proof": {"status": "passed", "code_symbol": code_symbol,
                         "executable_embedding": executable_embedding,
                         "opcode_counts": observed,
                         "resources": resources, "zero_scratch": True,
                         "inputs": proof_inputs},
        "memory_path": memory_path,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("artifact", "executable", "code-object", "assembly", "metadata"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", type=Path)
    marker = parser.add_mutually_exclusive_group(required=True)
    marker.add_argument("--stage", action="append")
    marker.add_argument("--unmarked", action="store_true")
    parser.add_argument("--dispatch-symbol", required=True)
    parser.add_argument("--code-symbol", required=True)
    parser.add_argument("--operation-family", required=True)
    parser.add_argument("--specialization", required=True)
    parser.add_argument("--hardware-class", required=True, choices=CLASSIFICATIONS)
    parser.add_argument("--arithmetic", required=True)
    parser.add_argument("--opcode", required=True, action="append", type=_opcode_spec)
    parser.add_argument("--max-lds-bytes", required=True, type=int)
    parser.add_argument("--max-vgpr-count", required=True, type=int)
    parser.add_argument("--memory-residency", required=True, choices=RESIDENCY)
    parser.add_argument("--memory-access", required=True)
    parser.add_argument("--overlap", required=True, choices=OVERLAP)
    parser.add_argument("--memory-evidence", required=True)
    parser.add_argument("--overlap-evidence", type=Path)
    parser.add_argument("--prefill-chunk", required=True, type=int)
    parser.add_argument("--kv-value-group", required=True, type=int, choices=(16, 32))
    parser.add_argument("--xattention-profile", required=True, choices=("dense",))
    parser.add_argument("--cta-recipe", choices=CTA_RECIPES)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    try:
        result = produce(
            artifact=args.artifact, executable=args.executable, code_object=args.code_object,
            assembly=args.assembly,
            metadata=args.metadata, sources=args.source,
            stage=None if args.unmarked else args.stage,
            dispatch_symbol=args.dispatch_symbol, code_symbol=args.code_symbol,
            operation_family=args.operation_family, specialization=args.specialization,
            hardware_classification=args.hardware_class, arithmetic=args.arithmetic,
            opcodes=args.opcode, max_lds_bytes=args.max_lds_bytes,
            max_vgpr_count=args.max_vgpr_count, memory_residency=args.memory_residency,
            memory_access=args.memory_access, overlap=args.overlap,
            memory_evidence=args.memory_evidence, overlap_evidence=args.overlap_evidence,
            prefill_chunk=args.prefill_chunk, kv_value_group=args.kv_value_group,
            xattention_profile=args.xattention_profile, cta_recipe=args.cta_recipe)
        _publish(args.out, result)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote selected-operation static evidence to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
