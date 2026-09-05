#!/usr/bin/env python3
"""Assemble and select provenance-bound DFlash2 K/W campaign evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.run_ninfer_bench_matrix import (
    MATRIX_SCHEMA_VERSION,
    PRODUCT_CONCURRENCIES,
    build_cases,
    file_sha256,
    load_bench_report,
    report_rows,
    validate_automatic_feasibility,
    validate_bound_diagnostic,
    validate_hybrid_shared_workspace_authority,
    write_dflash_determinism,
    write_dflash_greedy_parity,
    write_dflash_quality_evidence,
    write_dflash_shortlist,
)
from tools.ppl.assemble_pareto import _missing_capacity_provenance
from tools.ppl.assemble_pareto import _manifest_prefill_chunk
from tools.ppl.pareto import validate_terminal_production_authority


ARTIFACT_TYPE = "ninfer_r9700_dflash_selection"
SCHEMA_VERSION = 2
RULE = "maximin_whole_then_capacity_then_acceptance_then_canonical_kw_v2"
DFLASH_RECIPE_ID = "r9700-dflash2-all-q4g64-n16k16-bf16-codebook-eval-v1"
DFLASH_COMPANION_BY_BASE = {
    "r9700-q4g64-n16k16-eval": "r9700-q4g64-n16k16-dflash2-q4-eval",
    "r9700-q4-w8-mse-n16k16-eval": "r9700-q4-w8-mse-n16k16-dflash2-q4-eval",
    "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval",
}


def _hybrid_base_authority(artifact: dict[str, Any]) -> dict[str, Any] | None:
    if artifact.get("weights_id") != "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        return None
    receipt = artifact.get("conversion_receipt")
    fields = ("recipe_id", "selection_sha256", "object_plan_sha256",
              "source_index_sha256", "source_ranking_sha256")
    if (not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str)
            or not isinstance(receipt.get("sha256"), str)
            or any(not isinstance(receipt.get(name), str) for name in fields)):
        raise ValueError("selected hybrid base lacks its conversion authority")
    return {"receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
            **{name: receipt[name] for name in fields}}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root is not an object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_physical_identity(value: object, label: str, *, artifact: bool = False) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} provenance is not an object")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ValueError(f"{label} provenance lacks a path")
    if type(value.get("file_size_bytes")) is not int or value["file_size_bytes"] <= 0:
        raise ValueError(f"{label} provenance lacks a positive byte size")
    if not _valid_sha256(value.get("sha256")):
        raise ValueError(f"{label} provenance lacks a SHA-256")
    if artifact and (
        not isinstance(value.get("model_id"), str) or not value["model_id"]
        or not isinstance(value.get("weights_id"), str) or not value["weights_id"]
    ):
        raise ValueError(f"{label} provenance lacks artifact identity")


def _matrix(root: Path, preset: str, k: int | None = None, w: int | None = None,
            *, prefill_chunk: int, allow_failures: bool = False) -> dict:
    path = root / "manifest.json"
    value = _load(path)
    if (
        value.get("artifact_type") != "ninfer_bench_matrix_run"
        or value.get("schema_version") != MATRIX_SCHEMA_VERSION
        or value.get("preset") != preset
        or value.get("dry_run") is not False
        or value.get("failures")
        or (not allow_failures and (root / "failures.json").is_file())
    ):
        raise ValueError(f"{path} is not a physical schema-v{MATRIX_SCHEMA_VERSION} {preset} matrix")
    if k is not None and (
        value.get("dflash_draft_tokens") != k
        or value.get("dflash_verify_width") != w
    ):
        raise ValueError(f"{path} does not bind DFlash K{k}/W{w}")
    try:
        manifest_prefill_chunk = _manifest_prefill_chunk(value, preset)
    except ValueError as error:
        raise ValueError(
            f"{path} does not bind selected prefill chunk {prefill_chunk}"
        ) from error
    if (
        value.get("selected_prefill_chunk") != prefill_chunk
        or manifest_prefill_chunk != prefill_chunk
    ):
        raise ValueError(f"{path} does not bind selected prefill chunk {prefill_chunk}")
    return value


def _same_campaign(
    manifest: dict, artifact: dict, bench: dict, group: int, text_prefill_profile: str,
    prefill_chunk: int, hybrid_authority: dict[str, Any] | None,
) -> None:
    _validate_physical_identity(manifest.get("artifact"), "matrix artifact", artifact=True)
    _validate_physical_identity(manifest.get("bench"), "matrix benchmark")
    if manifest.get("artifact") != artifact or manifest.get("bench") != bench:
        raise ValueError("DFlash campaign artifact or benchmark executable changed")
    if manifest.get("expected_kv_value_group") != group:
        raise ValueError("DFlash campaign cache group differs from selected base")
    if (
        manifest.get("expected_q4_activation_bits") != 8
        or manifest.get("expected_w8_activation_bits") != 8
        or manifest.get("expected_fp8_qk_wmma_enabled") is not True
        or manifest.get("expected_xattention_profile") != text_prefill_profile
    ):
        raise ValueError(
            "DFlash campaign execution profile is not the selected Text-prefill/A8/FP8-QK profile"
        )
    if hybrid_authority is not None:
        if manifest.get("required_candidate_identity") != "fp8-hybrid-selection-authority":
            raise ValueError("hybrid DFlash campaign lacks its candidate identity")
        observed = validate_hybrid_shared_workspace_authority(
            manifest.get("hybrid_shared_workspace_authority"), [prefill_chunk]
        )
        if observed != hybrid_authority:
            raise ValueError("hybrid DFlash campaign planner differs from selected base")
    elif (
        manifest.get("required_candidate_identity") is not None
        or manifest.get("hybrid_shared_workspace_authority") is not None
    ):
        raise ValueError("non-hybrid DFlash campaign carries hybrid authority")


def _selected_hybrid_authority(
    base: dict[str, Any], base_recipe: dict[str, Any], prefill_chunk: int,
) -> dict[str, Any] | None:
    hybrid = base_recipe.get("weights_id") == "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
    if not hybrid:
        return None
    winner = base["terminal_production_selection"]["winner"]
    sources = [
        source for source in base.get("source_provenance", [])
        if isinstance(source, dict) and source.get("candidate") == winner
    ]
    if len(sources) != 1:
        raise ValueError("selected DFlash base lacks unique source provenance")
    bindings = sources[0].get("matrices")
    if not isinstance(bindings, dict) or set(bindings) != {"pareto-capacity", "pareto-whole"}:
        raise ValueError("selected DFlash base lacks exact matrix provenance")
    authorities = []
    for preset in ("pareto-capacity", "pareto-whole"):
        binding = bindings[preset]
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise ValueError("selected DFlash matrix provenance is malformed")
        path = Path(binding["path"]).resolve(strict=True)
        if file_sha256(path) != binding.get("sha256"):
            raise ValueError("selected DFlash matrix provenance changed")
        manifest = _load(path)
        if manifest.get("required_candidate_identity") != "fp8-hybrid-selection-authority":
            raise ValueError("selected hybrid base lacks its candidate identity")
        authorities.append(validate_hybrid_shared_workspace_authority(
            manifest.get("hybrid_shared_workspace_authority"), [prefill_chunk]
        ))
    if authorities[0] != authorities[1]:
        raise ValueError("selected hybrid base matrices bind different planners")
    return authorities[0]


def _records(root: Path, manifest: dict, preset: str, k: int, w: int,
             prefill_chunk: int, *, allow_missing: bool = False) -> dict[int, list[dict]]:
    cases = {
        (case.suite, case.name): case
        for case in build_cases(preset, k, w, production_prefill_chunk=prefill_chunk)
    }
    expected_c = list(PRODUCT_CONCURRENCIES) if preset != "dflash-shortlist" else [1]
    if manifest.get("concurrency") != expected_c:
        raise ValueError(f"{root} does not bind the required concurrency set")
    expected = {
        (case.suite, case.name, concurrency)
        for case in cases.values()
        for concurrency in expected_c
        if not case.concurrency_one_only or concurrency == 1
    }
    records = manifest.get("commands")
    if not isinstance(records, list):
        raise ValueError(f"{root} does not retain the exact {preset} point set")
    actual = []
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("suite"), str)
            or not isinstance(record.get("case"), str)
            or type(record.get("concurrency")) is not int
            or not isinstance(record.get("report"), str) or not record["report"]
            or not isinstance(record.get("command"), list)
            or any(not isinstance(part, str) for part in record["command"])
            or (cases.get((record.get("suite"), record.get("case")), None) is not None
                and cases[(record["suite"], record["case"])].diagnostic
                and (not isinstance(record.get("bound_diagnostic"), str)
                     or not record["bound_diagnostic"]))
        ):
            raise ValueError(f"{root} has a malformed {preset} command record")
        actual.append((record["suite"], record["case"], record["concurrency"]))
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(f"{root} does not retain the exact {preset} point set")
    reports: dict[int, list[dict]] = {}
    for record in records:
        case = cases[(record["suite"], record["case"])]
        path = Path(record["report"])
        if allow_missing and not path.is_file():
            continue
        report = load_bench_report(
            path, manifest["expected_kv_value_group"], 8, 8, True,
            record["concurrency"], manifest["artifact"], record["command"], case,
            manifest["expected_xattention_profile"],
        )
        reports.setdefault(record["concurrency"], []).append(report)
        if case.diagnostic:
            validate_bound_diagnostic(
                Path(record["bound_diagnostic"]), artifact=manifest["artifact"],
                bench=manifest["bench"], report_path=path, case=case,
                concurrency=record["concurrency"],
            )
    if not allow_missing and sorted(reports) != expected_c:
        raise ValueError(f"{root} has an incomplete concurrency set")
    return reports


def _selected_base(path: Path) -> tuple[int, str, dict, dict]:
    value = _load(path)
    validate_terminal_production_authority(value)
    if value.get("artifact_type") != "ninfer_r9700_pareto_comparison" or value.get("schema_version") != 7:
        raise ValueError("base selection is not a schema-v7 Pareto record")
    profile_selection = value.get("same_recipe_static_profile_selection")
    chosen = value.get("terminal_production_selection")
    selections = (
        profile_selection.get("selections") if isinstance(profile_selection, dict) else None
    )
    if (
        value.get("single_static_profile_selection_required") is not True
        or not isinstance(profile_selection, dict)
        or profile_selection.get("rule") != "same_recipe_static_profile_maximin_v2"
        or not isinstance(selections, list)
        or not selections
        or any(not isinstance(selection, dict) for selection in selections)
        or not isinstance(chosen, dict)
        or chosen.get("rule") !=
        "global_maximin_whole_then_capacity_then_quality_then_canonical_v1"
    ):
        raise ValueError("base selection has no required terminal production decision")
    winner = chosen.get("winner")
    profile_winners = [selection.get("winner") for selection in selections]
    if (
        any(not isinstance(name, str) or not name for name in profile_winners)
        or chosen.get("eligible_profile_winners") != sorted(profile_winners)
        or winner not in profile_winners
    ):
        raise ValueError("terminal production decision does not bind per-recipe winners")
    rows = [row for row in value.get("candidates", []) if row.get("name") == winner]
    if len(rows) != 1 or winner not in value.get("frontier", []):
        raise ValueError("base cache winner is not a unique retained frontier candidate")
    recipe = chosen.get("winner_artifact")
    if (
        not isinstance(recipe, dict)
        or recipe.get("kind") != "artifact"
        or not isinstance(recipe.get("weights_id"), str)
        or not recipe["weights_id"]
        or not _valid_sha256(recipe.get("sha256"))
    ):
        raise ValueError("base selection lacks provenance-bound artifact recipe identity")
    if rows[0].get("weight_recipe") != recipe:
        raise ValueError("terminal winner artifact differs from its candidate provenance")
    cache_profile = rows[0].get("cache_profile")
    execution_profile = rows[0].get("execution_profile")
    if (
        not isinstance(cache_profile, dict)
        or type(cache_profile.get("value_group")) is not int
        or cache_profile["value_group"] <= 0
    ):
        raise ValueError("base cache winner lacks a valid value-group profile")
    if (
        not isinstance(execution_profile, dict)
        or execution_profile.get("xattention_profile")
        not in ("dense", "b128-s16-tau900")
        or chosen.get("winner_cache_profile") != cache_profile
        or chosen.get("winner_execution_profile") != execution_profile
    ):
        raise ValueError("base winner lacks a provenance-bound static execution profile")
    return (
        cache_profile["value_group"], execution_profile["xattention_profile"], recipe, value,
    )


def _validate_shortlist_output(
    root: Path, manifest: dict, shortlist: dict, artifact: dict, bench: dict,
    prefill_chunk: int,
) -> None:
    cases = {
        (case.suite, case.name): case
        for case in build_cases(
            "dflash-shortlist", production_prefill_chunk=prefill_chunk
        )
    }
    rows = []
    for record in manifest["commands"]:
        rows.extend(report_rows(
            Path(record["report"]), cases[(record["suite"], record["case"])],
            manifest["expected_kv_value_group"], 8, 8, True, record["concurrency"],
            artifact, record["command"],
        ))
    parity_path = root / "greedy-token-parity.json"
    parity = _load(parity_path)
    if (
        parity.get("artifact_type") != "ninfer_dflash_ordinary_greedy_parity"
        or parity.get("schema_version") != 1
        or parity.get("artifact") != artifact
        or parity.get("benchmark_executable") != bench
        or parity.get("pass") is not True
    ):
        raise ValueError("DFlash shortlist parity evidence is invalid")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        recomputed_parity, parity_failures = write_dflash_greedy_parity(
            temporary, manifest["commands"], artifact=artifact, bench=bench,
        )
        if parity_failures or recomputed_parity != parity:
            raise ValueError("DFlash shortlist parity differs from recomputed raw reports")
        recomputed, failures = write_dflash_shortlist(
            temporary, manifest["commands"], rows, recomputed_parity,
            artifact=artifact, bench=bench,
        )
    if failures:
        raise ValueError("DFlash shortlist cannot be recomputed from retained evidence")
    recomputed["ordinary_control_parity"]["path"] = str(parity_path)
    if recomputed != shortlist:
        raise ValueError("DFlash shortlist differs from recomputed retained evidence")


def _auxiliary(
    root: Path, manifest: dict, artifact: dict, bench: dict, k: int, w: int,
) -> dict:
    parity_path = root / "greedy-token-parity.json"
    determinism_path = root / "dflash-proposal-determinism.json"
    quality_path = root / "dflash-generated-quality.json"
    parity, determinism, quality = map(_load, (parity_path, determinism_path, quality_path))
    for value in (parity, determinism, quality):
        if value.get("artifact") != artifact or value.get("benchmark_executable") != bench:
            raise ValueError("DFlash auxiliary evidence provenance mismatch")
        if value.get("pass") is not True:
            raise ValueError("DFlash auxiliary evidence did not pass")
    comparisons = parity.get("comparisons")
    if (
        parity.get("artifact_type") != "ninfer_dflash_ordinary_greedy_parity"
        or parity.get("schema_version") != 1
        or not isinstance(comparisons, list)
        or len(comparisons) != len(PRODUCT_CONCURRENCIES)
        or {row.get("concurrency") for row in comparisons} != set(PRODUCT_CONCURRENCIES)
        or any(row.get("draft_tokens") != k or row.get("dflash_verify_width") != w
               or row.get("exact") is not True for row in comparisons)
    ):
        raise ValueError("DFlash ordinary-output parity is incomplete")
    det_rows = determinism.get("comparisons")
    if (
        determinism.get("artifact_type") != "ninfer_dflash_proposal_determinism"
        or determinism.get("schema_version") != 1
        or not isinstance(det_rows, list) or len(det_rows) != 1
        or det_rows[0].get("exact") is not True
    ):
        raise ValueError("DFlash repeated proposal/target determinism is incomplete")
    if (
        quality.get("artifact_type") != "ninfer_dflash_generated_quality_evidence"
        or quality.get("schema_version") != 1
        or quality.get("target_output_gate", {}).get("pass") is not True
        or quality.get("proposal_gate", {}).get("pass") is not True
        or quality.get("target_output_gate", {}).get("evidence") != _identity(parity_path)
        or quality.get("proposal_gate", {}).get("determinism_evidence") != _identity(determinism_path)
    ):
        raise ValueError("DFlash generated-quality gate is incomplete or rebound")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        recomputed_parity, parity_failures = write_dflash_greedy_parity(
            temporary, manifest["commands"], artifact=artifact, bench=bench,
        )
        recomputed_determinism, determinism_failures = write_dflash_determinism(
            temporary, manifest["commands"], artifact=artifact, bench=bench,
        )
        recomputed_quality, quality_failures = write_dflash_quality_evidence(
            temporary, manifest["commands"], recomputed_parity, recomputed_determinism,
            artifact=artifact, bench=bench,
        )
    recomputed_quality["target_output_gate"]["evidence"]["path"] = str(parity_path)
    recomputed_quality["proposal_gate"]["determinism_evidence"]["path"] = str(
        determinism_path
    )
    if parity_failures or determinism_failures or quality_failures:
        raise ValueError("DFlash auxiliary gates fail when recomputed from retained evidence")
    if (
        recomputed_parity != parity
        or recomputed_determinism != determinism
        or recomputed_quality != quality
    ):
        raise ValueError("DFlash auxiliary evidence differs from recomputed raw reports")
    return {
        "parity": _identity(parity_path), "determinism": _identity(determinism_path),
        "generated_quality": _identity(quality_path),
    }


def _dominates(left: dict, right: dict) -> bool:
    dimensions = (
        [("whole", cell) for cell in left["whole"]]
        + [("capacity", cell) for cell in left["capacity"]]
        + [("acceptance", cell) for cell in left["acceptance"]]
    )
    return all(left[kind][cell] >= right[kind][cell] for kind, cell in dimensions) and any(
        left[kind][cell] > right[kind][cell] for kind, cell in dimensions
    )


def _normalized(name: str, values: dict[str, dict], frontier: list[str]) -> dict:
    result = {}
    for kind in ("whole", "capacity", "acceptance"):
        ratios = {}
        for cell in values[name][kind]:
            best = max(values[other][kind][cell] for other in frontier)
            ratios[cell] = values[name][kind][cell] / best if best else 1.0
        result[f"{kind}_ratio_by_cell"] = ratios
        result[f"minimum_{kind}_ratio"] = min(ratios.values())
    return result


def assemble(
    base_selection: Path, conversion_report: Path, shortlist_root: Path,
    capacity_inputs: Sequence[tuple[int, int, Path]],
    pareto_inputs: Sequence[tuple[int, int, Path]],
) -> dict[str, Any]:
    group, text_prefill_profile, base_recipe, base = _selected_base(base_selection)
    prefill_chunk = base["selected_prefill_chunk"]
    hybrid_authority = _selected_hybrid_authority(base, base_recipe, prefill_chunk)
    shortlist_path = shortlist_root / "dflash-shortlist.json"
    shortlist = _load(shortlist_path)
    manifest = _matrix(
        shortlist_root, "dflash-shortlist", prefill_chunk=prefill_chunk
    )
    artifact, bench = shortlist.get("artifact"), shortlist.get("benchmark_executable")
    if shortlist.get("artifact_type") != "ninfer_dflash_shortlist" or shortlist.get("schema_version") != 1 or shortlist.get("pass") is not True:
        raise ValueError("DFlash shortlist is not a passing schema-v1 record")
    if not isinstance(artifact, dict) or not isinstance(bench, dict):
        raise ValueError("DFlash shortlist lacks physical provenance")
    _validate_physical_identity(artifact, "DFlash artifact", artifact=True)
    _validate_physical_identity(bench, "DFlash benchmark")
    _same_campaign(
        manifest, artifact, bench, group, text_prefill_profile,
        prefill_chunk, hybrid_authority,
    )
    _records(shortlist_root, manifest, "dflash-shortlist", 1, 2, prefill_chunk)
    _validate_shortlist_output(
        shortlist_root, manifest, shortlist, artifact, bench, prefill_chunk
    )

    conversion = _load(conversion_report)
    expected_companion = DFLASH_COMPANION_BY_BASE.get(base_recipe.get("weights_id"))
    winner_sources = [
        source for source in base.get("source_provenance", [])
        if isinstance(source, dict)
        and source.get("candidate") == base["terminal_production_selection"]["winner"]
    ]
    if len(winner_sources) != 1:
        raise ValueError("selected DFlash base lacks unique source provenance")
    expected_base_authority = _hybrid_base_authority(winner_sources[0].get("artifact", {}))
    if (
        expected_companion is None
        or conversion.get("target_key") != "qwen3_8_27b_r9700"
        or conversion.get("recipe_id") != DFLASH_RECIPE_ID
        or conversion.get("status") != "registered-evaluation-only"
        or conversion.get("weight_recipe_selected") is not False
        or conversion.get("identity", {}).get("model_id") != artifact.get("model_id")
        or conversion.get("base", {}).get("identity", {}).get("weights_id") != base_recipe.get("weights_id")
        or conversion.get("base", {}).get("sha256") != base_recipe.get("sha256")
        or conversion.get("base", {}).get("authority") != expected_base_authority
        or conversion.get("base", {}).get("payload_copy") != "byte_exact"
        or conversion.get("artifact", {}).get("sha256") != artifact.get("sha256")
        or conversion.get("artifact", {}).get("bytes") != artifact.get("file_size_bytes")
        or not isinstance(conversion.get("artifact", {}).get("path"), str)
        or Path(conversion["artifact"]["path"]).resolve()
        != Path(artifact["path"]).resolve()
        or conversion.get("identity", {}).get("weights_id") != artifact.get("weights_id")
        or artifact.get("weights_id") != expected_companion
        or conversion.get("dflash_recipe", {}).get("matrix_format") != "Q4G64_F16S"
        or conversion.get("dflash_recipe", {}).get("selector_codebook_format") != "BF16"
        or conversion.get("dflash_recipe", {}).get("objects") != 66
        or conversion.get("dflash_recipe", {}).get("source_tensors") != 81
        or conversion.get("dflash_recipe", {}).get("format_counts")
        != {"BF16": 34, "Q4G64_F16S": 32}
        or conversion.get("dflash_recipe", {}).get("format_encoded_bytes")
        != {"BF16": 254_814_720, "Q4G64_F16S": 954_654_720}
        or conversion.get("dflash_recipe", {}).get("tensor_encoded_bytes") != 1_209_469_440
        or conversion.get("dflash_recipe", {}).get("runtime_repack") is not False
    ):
        raise ValueError("DFlash companion conversion does not bind the selected base/artifact")

    frontier_rows = shortlist.get("non_dominated_candidates")
    if not isinstance(frontier_rows, list) or not frontier_rows:
        raise ValueError("DFlash shortlist has no retained frontier")
    frontier_profiles = {
        (row.get("draft_tokens"), row.get("verify_width_resolved")): row
        for row in frontier_rows if isinstance(row, dict)
    }
    if (
        len(frontier_profiles) != len(frontier_rows)
        or any(type(k) is not int or type(w) is not int or k < 1 or w < 2
               for k, w in frontier_profiles)
    ):
        raise ValueError("DFlash shortlist frontier has duplicate or malformed K/W profiles")
    capacities = {(k, w): root for k, w, root in capacity_inputs}
    paretos = {(k, w): root for k, w, root in pareto_inputs}
    if len(capacities) != len(capacity_inputs) or set(capacities) != set(frontier_profiles):
        raise ValueError("capacity inputs must exactly cover the shortlist K/W frontier")
    if (
        len(paretos) != len(pareto_inputs)
        or not set(paretos).issubset(frontier_profiles)
    ):
        raise ValueError("Pareto inputs must be unique retained-shortlist K/W profiles")

    candidates, objective_values = [], {}
    for profile in sorted(frontier_profiles):
        k, w = profile
        root = capacities[profile]
        cap_manifest = _matrix(
            root, "dflash-capacity", k, w, prefill_chunk=prefill_chunk,
            allow_failures=True,
        )
        _same_campaign(
            cap_manifest, artifact, bench, group, text_prefill_profile,
            prefill_chunk, hybrid_authority,
        )
        cap_reports = _records(
            root, cap_manifest, "dflash-capacity", k, w, prefill_chunk,
            allow_missing=True,
        )
        failures = _missing_capacity_provenance(root, cap_manifest)
        capacity = {}
        for concurrency, reports in cap_reports.items():
            if len(reports) != 1:
                raise ValueError(f"K{k}/W{w} C{concurrency} has duplicate capacity reports")
            classified = validate_automatic_feasibility(reports[0])
            if classified["measurement_kind"] != "resolved_effective_maximum":
                raise ValueError(f"K{k}/W{w} C{concurrency} capacity is not an exact maximum")
            capacity[f"c{concurrency}"] = classified["resolved_effective_maximum_tokens"]
        eligible = len(capacity) == len(PRODUCT_CONCURRENCIES) and not failures
        if eligible != (profile in paretos):
            raise ValueError("Pareto inputs must exactly cover capacity-eligible K/W profiles")
        row = {
            "draft_tokens": k, "verify_width": w,
            "capacity_eligible": eligible, "capacity_tokens_by_concurrency": capacity,
            "capacity_failures": failures,
            "capacity_matrix": _identity(root / "manifest.json"),
        }
        if eligible:
            proot = paretos[profile]
            pmanifest = _matrix(
                proot, "dflash-pareto", k, w, prefill_chunk=prefill_chunk
            )
            _same_campaign(
                pmanifest, artifact, bench, group, text_prefill_profile,
                prefill_chunk, hybrid_authority,
            )
            _records(proot, pmanifest, "dflash-pareto", k, w, prefill_chunk)
            aux = _auxiliary(proot, pmanifest, artifact, bench, k, w)
            rows = []
            cases = {
                (case.suite, case.name): case
                for case in build_cases(
                    "dflash-pareto", k, w, production_prefill_chunk=prefill_chunk
                )
            }
            for record in pmanifest["commands"]:
                rows.extend(report_rows(
                    Path(record["report"]), cases[(record["suite"], record["case"])],
                    group, 8, 8, True, record["concurrency"], artifact, record["command"],
                    text_prefill_profile,
                ))
            whole = {
                f"{item['label']}_c{item['concurrency']}": item["whole_output_tok_s_mean"]
                for item in rows if item["suite"] == "dflash_pareto_whole_inference"
            }
            acceptance = {
                f"{item['label']}_c{item['concurrency']}": item["spec_acceptance_length"]
                for item in rows if item["suite"] == "dflash_pareto_decode"
            }
            if (
                len(whole) != 2 * len(PRODUCT_CONCURRENCIES)
                or len(acceptance) != 2 * len(PRODUCT_CONCURRENCIES)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value <= 0 for value in whole.values())
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value < 1 or value > k + 1 for value in acceptance.values())
            ):
                raise ValueError(f"K{k}/W{w} lacks complete positive whole/acceptance cells")
            objective_values[f"k{k}-w{w}"] = {
                "whole": whole, "capacity": capacity, "acceptance": acceptance,
            }
            row.update({
                "pareto_matrix": _identity(proot / "manifest.json"),
                "objectives": objective_values[f"k{k}-w{w}"],
                **aux,
            })
        candidates.append(row)

    eligible_names = sorted(objective_values)
    if not eligible_names:
        raise ValueError("no shortlist-frontier K/W profile is capacity eligible")
    reference = objective_values[eligible_names[0]]
    for name in eligible_names[1:]:
        if any(set(objective_values[name][kind]) != set(reference[kind])
               for kind in ("whole", "capacity", "acceptance")):
            raise ValueError("eligible DFlash profiles have different objective cell sets")
    frontier = [name for name in eligible_names if not any(
        other != name and _dominates(objective_values[other], objective_values[name])
        for other in eligible_names
    )]
    normalized = {name: _normalized(name, objective_values, frontier) for name in frontier}
    remaining, decisive = frontier, "canonical_kw"
    for stage, metric in (
        ("maximin_whole_inference_throughput", "minimum_whole_ratio"),
        ("maximin_resolved_capacity", "minimum_capacity_ratio"),
        ("maximin_acceptance_length", "minimum_acceptance_ratio"),
    ):
        best = max(normalized[name][metric] for name in remaining)
        narrowed = [name for name in remaining if normalized[name][metric] == best]
        if len(narrowed) == 1:
            remaining, decisive = narrowed, stage
            break
        remaining = narrowed
    winner = min(remaining, key=lambda name: tuple(map(int, name[1:].split("-w"))))
    return {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "selection_rule": RULE, "selected_cache_group": group,
        "selected_prefill_chunk": prefill_chunk,
        "selected_base": {
            **_identity(base_selection),
            "cache_group": group,
            "text_prefill_attention_profile": text_prefill_profile,
            "prefill_chunk": prefill_chunk,
            "dflash_proposal_attention_profile": "dense",
            "target_verification_attention_profile": "dense",
            "weight_recipe": base_recipe,
        },
        "conversion_report": _identity(conversion_report),
        "artifact": artifact, "benchmark_executable": bench,
        "required_candidate_identity": (
            "fp8-hybrid-selection-authority" if hybrid_authority is not None else None
        ),
        "hybrid_shared_workspace_authority": hybrid_authority,
        "shortlist": _identity(shortlist_path), "candidates": candidates,
        "eligible_frontier": frontier, "normalized_objectives": normalized,
        "winner": winner, "decisive_stage": decisive,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--shortlist-dir", type=Path, required=True)
    parser.add_argument("--capacity", action="append", nargs=3, metavar=("K", "W", "DIR"), required=True)
    parser.add_argument("--pareto", action="append", nargs=3, metavar=("K", "W", "DIR"), default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    result = assemble(
        args.base_selection.resolve(), args.conversion_report.resolve(), args.shortlist_dir.resolve(),
        [(int(k), int(w), Path(root).resolve()) for k, w, root in args.capacity],
        [(int(k), int(w), Path(root).resolve()) for k, w, root in args.pareto],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.out.name}.", dir=args.out.parent)
    published = False
    durable = False
    created_inode = None
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary_stat = os.stat(temporary, follow_symlinks=False)
        created_inode = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.link(temporary, args.out)
        published = True
        output_stat = os.stat(args.out, follow_symlinks=False)
        if (not stat.S_ISREG(output_stat.st_mode)
                or (output_stat.st_dev, output_stat.st_ino) != created_inode):
            raise ValueError("published DFlash selection inode changed")
        if (_load(args.out) != result or assemble(
                args.base_selection.resolve(), args.conversion_report.resolve(),
                args.shortlist_dir.resolve(),
                [(int(k), int(w), Path(root).resolve()) for k, w, root in args.capacity],
                [(int(k), int(w), Path(root).resolve()) for k, w, root in args.pareto],
        ) != result):
            raise ValueError("published DFlash selection does not revalidate")
        directory = os.open(args.out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        durable = True
    finally:
        Path(temporary).unlink(missing_ok=True)
        if published and not durable and created_inode is not None:
            try:
                current = os.stat(args.out, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == created_inode:
                    args.out.unlink()
            except FileNotFoundError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
