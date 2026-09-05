#!/usr/bin/env python3
"""Recompute and atomically join every final-artifact cutover admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from tools.bench.assemble_dflash_selection import assemble as assemble_dflash
from tools.bench.focused_verification_io import validate_report as validate_focused
from tools.bench.produce_mtp_shortlist_head_evidence import produce as produce_mtp
from tools.bench.prepare_selected_converter_preflight import revalidate as revalidate_converter
from tools.bench.select_prefill_chunk import validate_selection_record
from tools.bench.validate_low_context_prefill import validate_ladder
from tools.bench.validate_selected_niah import validate as validate_niah
from tools.bench.validate_selected_vision_diagnostic import validate as validate_vision
from tools.bench.verify_selected_hardware_use import verify as verify_hardware
from tools.ppl.assemble_pareto import _campaign_quality_candidate
from tools.ppl.pareto import load_payload, validate_terminal_production_authority
from tools.ppl.validate_selected_exact_token import validate as validate_exact_token


REPO = Path(__file__).resolve().parents[2]
PRODUCT_CONCURRENCIES = [1, 2, 3, 4]
MIXED = "r9700-q4-w8-mse-n16k16-eval"
HYBRID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
ARTIFACT_TYPE = "ninfer_r9700_final_artifact_cutover_admission"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_regular(path: Path, label: str) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def snapshot(path: Path, label: str) -> dict[str, Any]:
    value = load_regular(path, label)
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path), "value": value}


def identity(snapshot_value: dict[str, Any]) -> dict[str, str]:
    return {"path": snapshot_value["path"], "sha256": snapshot_value["sha256"]}


def file_identity(path: Path, label: str) -> dict[str, str]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def validate_prepared_closure(plan_path: Path) -> dict[str, str]:
    closure = plan_path.parent / "prepared.sha256"
    if not os.path.lexists(closure):
        raise ValueError("cutover prepared closure is missing")
    file_identity(closure, "cutover prepared closure")
    entries: dict[str, str] = {}
    for line in closure.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or Path(parts[1]).is_absolute() \
                or ".." in Path(parts[1]).parts or parts[1] in entries:
            raise ValueError("cutover prepared closure is malformed")
        entries[parts[1]] = parts[0]
    required = {
        str(Path(__file__).resolve().relative_to(REPO)),
        str(plan_path.resolve(strict=True).relative_to(REPO)),
        str((plan_path.parent / "run.sh").resolve(strict=True).relative_to(REPO)),
    }
    if not required <= set(entries):
        raise ValueError("cutover prepared closure lacks its executable package")
    for name, digest in entries.items():
        path = REPO / name
        if file_identity(path, f"closure input {name}")["sha256"] != digest:
            raise ValueError(f"cutover prepared closure bytes changed: {name}")
    return {"path": str(closure.resolve(strict=True)), "sha256": sha(closure)}


def selected_route(selection: dict[str, Any], terminal: dict[str, Any]) -> tuple[dict, dict]:
    winner = terminal["winner"]
    rows = [row for row in selection["source_provenance"] if row.get("candidate") == winner]
    if len(rows) != 1:
        raise ValueError("terminal winner lacks one source-provenance row")
    source = rows[0]
    return ({
        "winner": winner,
        "weights_id": terminal["winner_artifact"]["weights_id"],
        "artifact": source["artifact"],
        "executable": source["benchmark_executable"],
        "cache_group": terminal["winner_cache_profile"]["value_group"],
        "attention_profile": terminal["winner_execution_profile"]["xattention_profile"],
        "prefill_chunk": selection["selected_prefill_chunk"],
    }, source)


def _physical_matches(actual: object, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    return all(actual.get(key) == expected.get(key) for key in ("sha256", "file_size_bytes"))


def require_route(label: str, value: dict[str, Any], route: dict[str, Any], *, dense=False) -> None:
    selected = value.get("selected_route", value.get("terminal_route", value))
    if not isinstance(selected, dict):
        raise ValueError(f"{label} lacks a selected route")
    artifact = selected.get("artifact")
    executable = selected.get("benchmark_executable", selected.get("executable", selected.get("benchmark")))
    cache = selected.get("cache_profile")
    execution = selected.get("execution_profile")
    observed_group = (
        cache.get("value_group") if isinstance(cache, dict)
        else selected.get("kv_value_group", selected.get("value_group", selected.get("selected_cache_group")))
    )
    observed_attention = (
        execution.get("xattention_profile") if isinstance(execution, dict)
        else selected.get("xattention_profile", selected.get("attention_profile"))
    )
    observed_chunk = selected.get("selected_prefill_chunk", selected.get("prefill_chunk"))
    if selected.get("winner", route["winner"]) != route["winner"]:
        raise ValueError(f"{label} winner differs")
    if artifact is not None and not _physical_matches(artifact, route["artifact"]):
        raise ValueError(f"{label} artifact differs")
    if executable is not None and not _physical_matches(executable, route["executable"]):
        raise ValueError(f"{label} executable differs")
    if observed_group is not None and observed_group != route["cache_group"]:
        raise ValueError(f"{label} cache group differs")
    expected_attention = "dense" if dense else route["attention_profile"]
    if observed_attention is not None and observed_attention != expected_attention:
        raise ValueError(f"{label} attention profile differs")
    if observed_chunk is not None and observed_chunk != route["prefill_chunk"]:
        raise ValueError(f"{label} prefill chunk differs")


def validate_conditionals(route: dict[str, Any], hardware: dict[str, Any]) -> dict[str, bool]:
    authorities = hardware.get("authorities")
    if not isinstance(authorities, dict):
        raise ValueError("hardware-use authority lacks conditional proof identities")
    mixed_proof = authorities.get("mixed_mtp_bulk_w8")
    fp8_proofs = hardware.get("loaded_fp8_proofs", [])
    mixed = route["weights_id"] == MIXED
    hybrid = route["weights_id"] == HYBRID
    if mixed != isinstance(mixed_proof, dict):
        raise ValueError("mixed MTP-bulk W8 proof conditional differs")
    if hybrid != (isinstance(fp8_proofs, list) and len(fp8_proofs) == 2):
        raise ValueError("four-role loaded-FP8 proof conditional differs")
    if not hybrid and fp8_proofs not in (None, []):
        raise ValueError("non-hybrid route carries loaded-FP8 proof")
    return {"mixed_mtp_bulk_w8": mixed, "four_role_loaded_fp8": hybrid,
            "low_context_dense_control": True}


def validate_quality_map(path: Path, selection: dict, chunk: int) -> dict:
    value = load_regular(path, "quality authority map")
    authorities = value.get("authorities")
    if (
        value.get("artifact_type") != "ninfer_r9700_terminal_quality_authority_map"
        or value.get("schema_version") != 1
        or value.get("selected_prefill_chunk") != chunk
        or value.get("concurrency") != 1
        or not isinstance(authorities, dict)
        or len(authorities) != 6
    ):
        raise ValueError("quality authority map has invalid identity/inventory")
    expected = {
        (source["artifact"]["weights_id"], source["quality"]["representation"]["xattention_profile"]):
            (source["quality"]["path"], source["quality"]["sha256"])
        for source in selection["source_provenance"]
    }
    if len(expected) != 6:
        raise ValueError("schema-v7 source provenance lacks six quality authorities")
    observed = set()
    for item in authorities.values():
        authority_path = Path(item.get("path", "")) if isinstance(item, dict) else Path()
        campaign = load_regular(authority_path, "quality campaign")
        if item.get("sha256") != sha(authority_path):
            raise ValueError("quality campaign bytes changed")
        matches = [key for key, bound in expected.items() if bound == (str(authority_path.resolve()), item["sha256"])]
        if len(matches) != 1:
            raise ValueError("quality map campaign is not bound by schema-v7")
        weights_id, profile = matches[0]
        for group in (16, 32):
            _campaign_quality_candidate(campaign, weights_id, group, chunk)
        observed.add((weights_id, profile))
    if observed != set(expected):
        raise ValueError("quality map does not cover all six recipe/profile authorities")
    return value


def matrix_inventory(selection: dict) -> list[dict[str, Any]]:
    inventory = []
    for source in selection["source_provenance"]:
        matrices = source.get("matrices")
        if not isinstance(matrices, dict) or set(matrices) != {"pareto-capacity", "pareto-whole"}:
            raise ValueError("schema-v7 candidate lacks capacity and whole matrices")
        for preset, bound in matrices.items():
            path = Path(bound.get("path", ""))
            manifest = load_regular(path, f"{preset} matrix")
            if bound.get("sha256") != sha(path) or manifest.get("concurrency") != PRODUCT_CONCURRENCIES:
                raise ValueError(f"{preset} matrix bytes or C1..4 inventory differ")
            if manifest.get("artifact") != source["artifact"] or manifest.get("bench") != source["benchmark_executable"]:
                raise ValueError(f"{preset} matrix physical identity differs")
            inventory.append({"candidate": source["candidate"], "preset": preset,
                              "manifest": {"path": str(path.resolve()), "sha256": sha(path)},
                              "concurrency": PRODUCT_CONCURRENCIES})
    if len(inventory) != 24:
        raise ValueError("cutover requires twelve capacity and twelve whole matrices")
    return inventory


def revalidate_mtp(path: Path) -> dict:
    value = load_regular(path, "MTP shortlist-head evidence")
    route = value.get("selected_route", {})
    trace = value.get("executed_trace", {})
    proof = value.get("static_proof", {})
    rebuilt = produce(
        plan_path=Path(trace["plan"]["path"]), benchmark_report=Path(trace["benchmark_report"]["path"]),
        trace_database=Path(trace["database"]["path"]), power_before=Path(trace["power_before"]["path"]),
        power_after=Path(trace["power_after"]["path"]),
        terminal_selection=Path(route["terminal_selection"]["path"]),
        artifact=Path(route["artifact"]["path"]), executable=Path(route["benchmark_executable"]["path"]),
        code_object=Path(proof["code_object"]["path"]), dispatch_symbol=trace["display_symbol"],
        stages=trace["stages"],
    )
    if rebuilt != value:
        raise ValueError("MTP shortlist-head evidence does not revalidate")
    return value


def revalidate_hardware(path: Path, selection_path: Path) -> dict:
    value = load_regular(path, "selected hardware-use authority")
    authorities = value.get("authorities", {})
    fp8 = [Path(row["authority"]["path"]) for row in value.get("loaded_fp8_proofs", [])]
    mixed = authorities.get("mixed_mtp_bulk_w8")
    rebuilt = verify_hardware(
        selection_path, Path(authorities["dispatch_reconciliation"]["path"]),
        Path(authorities["static_audit"]["path"]), Path(authorities["mtp_shortlist_head"]["path"]),
        fp8, Path(mixed["path"]) if isinstance(mixed, dict) else None,
    )
    if rebuilt != value:
        raise ValueError("selected hardware-use authority does not revalidate")
    return value


def revalidate_dflash(path: Path) -> dict:
    value = load_regular(path, "DFlash selection")
    capacities, paretos = [], []
    for row in value.get("candidates", []):
        name = row.get("name", "")
        if not isinstance(name, str) or not name.startswith("k") or "-w" not in name:
            raise ValueError("DFlash selection has malformed K/W candidate")
        k, w = (int(part) for part in name[1:].split("-w", 1))
        capacities.append((k, w, Path(row["capacity_matrix"]["path"]).parent))
        if isinstance(row.get("pareto_matrix"), dict):
            paretos.append((k, w, Path(row["pareto_matrix"]["path"]).parent))
    rebuilt = assemble_dflash(
        Path(value["selected_base"]["path"]), Path(value["conversion_report"]["path"]),
        Path(value["shortlist"]["path"]).parent, capacities, paretos,
    )
    if rebuilt != value:
        raise ValueError("DFlash selection does not revalidate")
    return value


def validate_converter_preflight(path: Path, selection: Path,
                                 route: dict[str, Any]) -> dict[str, Any]:
    converter = revalidate_converter(path, selection)
    converter_route = converter.get("selected_route", {})
    if (converter_route.get("winner") != route["winner"]
            or converter_route.get("weights_id") != route["weights_id"]
            or not _physical_matches(converter_route.get("evaluation_artifact"), route["artifact"])
            or converter.get("status") != "passed_no_artifact_no_device"):
        raise ValueError("converter preflight does not bind the selected route")
    return converter


def assemble(plan_path: Path) -> dict[str, Any]:
    plan_snapshot = snapshot(plan_path, "cutover admission plan")
    prepared_closure = validate_prepared_closure(plan_path)
    plan = plan_snapshot["value"]
    paths = {
        name: (Path(value) if Path(value).is_absolute() else REPO / value)
        for name, value in plan.get("inputs", {}).items()
    }
    required = {"selection", "prefill_chunk", "quality_map", "exact_plan", "exact_campaign",
                "exact_admission", "mtp", "low_manifest", "low_admission", "niah_plan",
                "niah_root", "niah_admission", "vision_plan", "vision_root",
                "vision_admission", "focused_closure", "focused", "hardware", "dflash",
                "converter_preflight"}
    if set(paths) != required:
        raise ValueError("cutover admission plan input inventory differs")
    for name in ("niah_root", "vision_root"):
        root_metadata = paths[name].lstat()
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError(f"{name} is not a real directory")
    initial_inputs = {
        name: file_identity(path, name) for name, path in paths.items()
        if name not in {"niah_root", "vision_root"}
    }

    selection_snapshot = snapshot(paths["selection"], "schema-v7 selection")
    selection = selection_snapshot["value"]
    terminal, _ = validate_terminal_production_authority(load_payload(json.dumps(selection)))
    if terminal.get("production_status") != "selected_route_pending_shortlist_head_trace_and_niah":
        raise ValueError("schema-v7 winner is not eligible for final physical gates")
    route, winner_source = selected_route(selection, terminal)
    converter = validate_converter_preflight(
        paths["converter_preflight"], paths["selection"], route)
    chunk = validate_selection_record(paths["prefill_chunk"])
    if chunk["selected_prefill_chunk"] != route["prefill_chunk"]:
        raise ValueError("prefill-chunk and schema-v7 decisions differ")
    validate_quality_map(paths["quality_map"], selection, route["prefill_chunk"])
    matrices = matrix_inventory(selection)

    exact = validate_exact_token(paths["exact_plan"], paths["exact_campaign"])
    if exact != load_regular(paths["exact_admission"], "exact-token admission"):
        raise ValueError("exact-token admission does not revalidate")
    require_route("exact-token", exact, route)

    mtp = revalidate_mtp(paths["mtp"])
    require_route("MTP shortlist head", mtp, route)

    dense_sources = [source for source in selection["source_provenance"]
                     if source["artifact"]["weights_id"] == route["weights_id"]
                     and source["cache_value_group"] == route["cache_group"]
                     and source["quality"]["representation"]["xattention_profile"] == "dense"]
    if len(dense_sources) != 1:
        raise ValueError("schema-v7 lacks one winner-matched dense control")
    dense_route = {**route, "artifact": dense_sources[0]["artifact"],
                   "executable": dense_sources[0]["benchmark_executable"], "attention_profile": "dense"}
    low_record = load_regular(paths["low_admission"], "low-context admission")
    low = validate_ladder(paths["low_manifest"], 2000.0,
                          Path(dense_route["executable"]["path"]),
                          Path(dense_route["artifact"]["path"]), paths["selection"])
    low_control = low.get("dense_control", {})
    if (low != low_record or low.get("passes_p2048_gate") is not True
            or not _physical_matches(low.get("artifact"), dense_route["artifact"])
            or not _physical_matches(low.get("bench"), dense_route["executable"])
            or low.get("expected_kv_value_group") != route["cache_group"]
            or low.get("selected_prefill_chunk") != route["prefill_chunk"]
            or low_control.get("winner") != route["winner"]
            or not _physical_matches(low_control.get("executable"), dense_route["executable"])):
        raise ValueError("low-context dense-control P2048 >=2000 tok/s admission does not revalidate")

    niah = validate_niah(paths["niah_plan"], paths["niah_root"])
    if niah != load_regular(paths["niah_admission"], "NIAH admission"):
        raise ValueError("NIAH admission does not revalidate")
    require_route("NIAH", niah, route)

    vision = validate_vision(paths["vision_plan"], paths["vision_root"])
    if vision != load_regular(paths["vision_admission"], "Vision diagnostic completion"):
        raise ValueError("Vision diagnostic completion does not revalidate")
    if vision.get("status") != "complete_diagnostic_no_numeric_threshold":
        raise ValueError("Vision diagnostic has not completed")
    require_route("Vision diagnostic", vision, route)

    focused_record = load_regular(paths["focused"], "focused verification")
    focused_route = validate_focused(focused_record, paths["focused_closure"])
    require_route("focused verification", focused_route, route)

    hardware = revalidate_hardware(paths["hardware"], paths["selection"])
    require_route("hardware use", hardware, route)
    conditional_proofs = validate_conditionals(route, hardware)

    dflash = revalidate_dflash(paths["dflash"])
    base = dflash.get("selected_base", {})
    if (base.get("sha256") != selection_snapshot["sha256"]
            or base.get("cache_group") != route["cache_group"]
            or base.get("text_prefill_attention_profile") != route["attention_profile"]
            or base.get("prefill_chunk") != route["prefill_chunk"]
            or base.get("weight_recipe") != terminal["winner_artifact"]):
        raise ValueError("DFlash companion does not bind the same base selection")

    input_snapshots = {
        name: file_identity(path, name) for name, path in paths.items()
        if name not in {"niah_root", "vision_root"}
    }
    if input_snapshots != initial_inputs:
        raise ValueError("cutover admission input changed during joined validation")
    return {
        "artifact_type": ARTIFACT_TYPE, "schema_version": 1, "status": "passed",
        "selected_route": route, "terminal_selection": identity(selection_snapshot),
        "inputs": input_snapshots, "matrix_inventory": matrices,
        "quality_authority_count": 6, "capacity_matrix_count": 12,
        "whole_matrix_count": 12, "concurrency": PRODUCT_CONCURRENCIES,
        "conditional_proofs": conditional_proofs,
        "converter_preflight": input_snapshots["converter_preflight"],
        "cutover_plan": identity(plan_snapshot), "prepared_closure": prepared_closure,
    }


def publish(plan: Path, output: Path) -> dict[str, Any]:
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite cutover admission output: {output}")
    value = assemble(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    owner = None
    published = False
    durable = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno()); owner = (metadata.st_dev, metadata.st_ino)
            json.dump(value, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, output); published = True
        current = os.lstat(output)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != owner:
            raise ValueError("published cutover admission inode differs")
        if load_regular(output, "published cutover admission") != value or assemble(plan) != value:
            raise ValueError("published cutover admission does not revalidate")
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
        durable = True
        return value
    finally:
        try:
            current = os.lstat(temporary)
            if owner is not None and (current.st_dev, current.st_ino) == owner: temporary.unlink()
        except FileNotFoundError: pass
        if published and not durable and owner is not None:
            try:
                current = os.lstat(output)
                if (current.st_dev, current.st_ino) == owner: output.unlink()
            except FileNotFoundError: pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try: publish(args.plan, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
