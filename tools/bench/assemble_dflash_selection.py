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
    R9700_POWER_PROFILE,
    build_cases,
    dflash_shortlist_profiles,
    file_sha256,
    load_bench_report,
    report_rows,
    validate_automatic_feasibility,
    validate_bound_diagnostic,
    validate_hybrid_shared_workspace_authority,
    base_hybrid_shared_workspace_authority,
    write_dflash_determinism,
    write_dflash_greedy_parity,
    write_dflash_quality_evidence,
    write_dflash_shortlist,
    DFLASH_COMPANIONS,
    DFLASH_PRODUCTION_PROFILES,
    DFLASH_SHORTLIST_SCHEMA_VERSION,
    inspect_artifact,
    inspect_executable,
    bind_n16_migration_receipt,
    require_dflash_companion,
)
from tools.bench.prefill_chunk_authority import validate_prefill_chunk_authority
from tools.convert.qwen3_8_27b_r9700 import dflash2_matrix_recipes
from tools.ppl.assemble_pareto import _missing_capacity_provenance
from tools.ppl.assemble_pareto import _manifest_prefill_chunk
from tools.ppl.pareto import validate_terminal_production_authority


ARTIFACT_TYPE = "ninfer_r9700_dflash_selection"
SCHEMA_VERSION = 4
RULE = "qualified_recipe_kw_primary_c1_and_per_concurrency_maximin_v4"
MATERIAL_SPEEDUP = 1.02
UNCERTAINTY_SIGMAS = 2.0
RECIPES = tuple(recipe.key for recipe in dflash2_matrix_recipes.RECIPES)




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
        or (not allow_failures and value.get("failures"))
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
    dflash_widths: Sequence[int] = (),
) -> None:
    _validate_physical_identity(manifest.get("artifact"), "matrix artifact", artifact=True)
    _validate_physical_identity(manifest.get("bench"), "matrix benchmark")
    if manifest.get("artifact") != artifact or manifest.get("bench") != bench:
        raise ValueError("DFlash campaign artifact or benchmark executable changed")
    if manifest.get("preset") in {"dflash-shortlist", "dflash-pareto"} and (
        manifest.get("power_profile") != {
            "required": "auto",
            "sysfs_path": str(R9700_POWER_PROFILE),
            "observed": "auto",
            "rechecked_after": "auto",
        }
    ):
        raise ValueError("DFlash timing campaign does not bind stable auto power")
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
            manifest.get("hybrid_shared_workspace_authority"), [prefill_chunk], dflash_widths
        )
        build = observed["build"]
        bench_root = Path(bench["path"]).resolve().parent.parent
        if (
            Path(build["root"]).resolve() != bench_root
            or Path(build["cmake_cache"]["path"]).resolve()
            != bench_root / "CMakeCache.txt"
            or Path(build["compile_commands"]["path"]).resolve()
            != bench_root / "compile_commands.json"
        ):
            raise ValueError("hybrid planner does not bind the benchmark build root")
        _validate_physical_identity(build["cmake_cache"], "hybrid CMake cache")
        _validate_physical_identity(build["compile_commands"], "hybrid compile commands")
        projected = base_hybrid_shared_workspace_authority(observed)
        if any(projected[key] != hybrid_authority[key] for key in (
            "maximum_concurrency", "prefill_chunks", "inventories_by_prefill_chunk",
        )):
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
             prefill_chunk: int, *, allow_missing: bool = False,
             required_concurrency: Sequence[int] = PRODUCT_CONCURRENCIES) -> dict[int, list[dict]]:
    cases = {
        (case.suite, case.name): case
        for case in build_cases(preset, k, w, production_prefill_chunk=prefill_chunk)
    }
    expected_c = list(required_concurrency) if preset != "dflash-shortlist" else [1]
    if (not expected_c or sorted(set(expected_c)) != expected_c
            or any(type(c) is not int or c not in PRODUCT_CONCURRENCIES for c in expected_c)):
        raise ValueError("invalid declared concurrency set")
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
            manifest["expected_xattention_profile"],
        ))
    parity_path = root / "greedy-token-parity.json"
    parity = _load(parity_path)
    if (
        parity.get("artifact_type") != "ninfer_dflash_ordinary_greedy_parity"
        or parity.get("schema_version") != 2
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
    required_concurrency: Sequence[int] = PRODUCT_CONCURRENCIES,
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
        or parity.get("schema_version") != 2
        or not isinstance(comparisons, list)
        or len(comparisons) != 2 * len(required_concurrency)
        or {(row.get("phase"), row.get("concurrency")) for row in comparisons}
        != {(phase, concurrency) for phase in ("decode", "whole")
            for concurrency in required_concurrency}
        or any(row.get("draft_tokens") != k or row.get("dflash_verify_width") != w
               or row.get("includes_seed") is not (row.get("phase") == "whole")
               or row.get("exact") is not True
               for row in comparisons)
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
        or quality.get("target_output_gate", {}).get("concurrency") != list(required_concurrency)
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
            required_concurrency=required_concurrency,
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


def _matched_ordinary_speed_gate(rows: Sequence[dict[str, Any]],
                                 required_concurrency: Sequence[int] = PRODUCT_CONCURRENCIES) -> dict[str, Any]:
    """Require material, uncertainty-separated whole and decode wins at every product cell."""

    suites = (
        "dflash_pareto_whole_inference", "dflash_pareto_whole_control",
        "dflash_pareto_decode", "dflash_pareto_control",
    )
    by_suite: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    metric = {
        "dflash_pareto_whole_inference": ("whole_output_tok_s_mean", "whole_output_tok_s_stddev"),
        "dflash_pareto_whole_control": ("whole_output_tok_s_mean", "whole_output_tok_s_stddev"),
        "dflash_pareto_decode": ("decode_output_tok_s_mean", "decode_output_tok_s_stddev"),
        "dflash_pareto_control": ("decode_output_tok_s_mean", "decode_output_tok_s_stddev"),
    }
    for suite in suites:
        selected = [row for row in rows if row.get("suite") == suite]
        indexed: dict[tuple[int, int, int], dict[str, Any]] = {}
        for row in selected:
            key_values = (row.get("n_prompt"), row.get("n_gen"), row.get("concurrency"))
            mean_field, stddev_field = metric[suite]
            if (
                any(type(value) is not int or value <= 0 for value in key_values)
                or key_values[2] not in PRODUCT_CONCURRENCIES
                or type(row.get("requested_output_tokens")) is not int
                or row["requested_output_tokens"] <= 0
                or type(row.get(mean_field)) not in (int, float)
                or isinstance(row.get(mean_field), bool)
                or not math.isfinite(row[mean_field]) or row[mean_field] <= 0
                or type(row.get(stddev_field)) not in (int, float)
                or isinstance(row.get(stddev_field), bool)
                or not math.isfinite(row[stddev_field]) or row[stddev_field] < 0
            ):
                raise ValueError(f"{suite} has malformed matched speed evidence")
            key = (key_values[0], key_values[1], key_values[2])
            if key in indexed:
                raise ValueError(f"{suite} has duplicate matched speed cells")
            indexed[key] = row
        by_suite[suite] = indexed

    expected = {
        (prompt, 256, concurrency)
        for prompt in (8192, 32768)
        for concurrency in required_concurrency
    }
    if any(set(by_suite[suite]) != expected for suite in suites):
        raise ValueError("DFlash matched speed gate lacks exact declared 8K/32K concurrency cells")

    raw_ratios: dict[str, dict[str, float]] = {"whole": {}, "decode": {}}
    conservative: dict[str, dict[str, float]] = {"whole": {}, "decode": {}}
    matched = {}
    for phase, candidate_suite, control_suite in (
        ("whole", "dflash_pareto_whole_inference", "dflash_pareto_whole_control"),
        ("decode", "dflash_pareto_decode", "dflash_pareto_control"),
    ):
        mean_field, stddev_field = metric[candidate_suite]
        for prompt, generated, concurrency in sorted(expected):
            candidate = by_suite[candidate_suite][(prompt, generated, concurrency)]
            control = by_suite[control_suite][(prompt, generated, concurrency)]
            if candidate["requested_output_tokens"] != control["requested_output_tokens"]:
                raise ValueError("DFlash matched speed cell changes requested output tokens")
            cell = f"p{prompt}_g{generated}_c{concurrency}"
            candidate_lower = candidate[mean_field] - UNCERTAINTY_SIGMAS * candidate[stddev_field]
            control_upper = control[mean_field] + UNCERTAINTY_SIGMAS * control[stddev_field]
            if candidate_lower <= 0 or control_upper <= 0:
                raise ValueError("DFlash matched speed uncertainty bound is nonpositive")
            raw_ratios[phase][cell] = candidate[mean_field] / control[mean_field]
            conservative[phase][cell] = candidate_lower / control_upper
            if not all(math.isfinite(value) for value in (
                raw_ratios[phase][cell], conservative[phase][cell]
            )):
                raise ValueError("DFlash matched speed ratio is not finite")
            matched[f"{phase}_{cell}"] = {
                "dflash_mean": candidate[mean_field], "dflash_stddev": candidate[stddev_field],
                "ordinary_mean": control[mean_field], "ordinary_stddev": control[stddev_field],
            }
    minimum_raw = {phase: min(values.values()) for phase, values in raw_ratios.items()}
    minimum_conservative = {
        phase: min(values.values()) for phase, values in conservative.items()
    }
    passed = all(
        minimum_raw[phase] >= MATERIAL_SPEEDUP and minimum_conservative[phase] > 1.0
        for phase in ("whole", "decode")
    )
    return {
        "criterion": (
            "at least 1.02x raw mean speedup and a strictly positive two-standard-deviation "
            "lower-bound speedup for both exact whole-request and isolated decode controls at "
            "every declared 8K/32K concurrency cell"
        ),
        "material_speedup": MATERIAL_SPEEDUP,
        "uncertainty_sigmas": UNCERTAINTY_SIGMAS,
        "required_concurrency": list(required_concurrency),
        "required_prompt_tokens": [8192, 32768],
        "required_generated_tokens": 256,
        "raw_mean_speedup_by_phase_and_cell": raw_ratios,
        "conservative_speedup_by_phase_and_cell": conservative,
        "matched_means_by_phase_and_cell": matched,
        "minimum_raw_mean_speedup": minimum_raw,
        "minimum_conservative_speedup": minimum_conservative,
        "pass": passed,
    }


def selected_base_route(selection: Path) -> dict:
    """Resolve actual terminal base and chunk authorities, never a filename convention."""
    group, profile, recipe, base = _selected_base(selection)
    chunk_binding = base["prefill_chunk_selection"]
    chunk, _record = validate_prefill_chunk_authority(Path(chunk_binding["path"]))
    if (chunk["sha256"] != chunk_binding["sha256"]
            or chunk["selected_prefill_chunk"] != base["selected_prefill_chunk"]):
        raise ValueError("terminal base differs from its selected chunk authority")
    winner = base["terminal_production_selection"]["winner"]
    sources = [source for source in base["source_provenance"]
               if source.get("candidate") == winner]
    if len(sources) != 1:
        raise ValueError("terminal base lacks unique artifact provenance")
    retained = sources[0]["artifact"]
    artifact_path = Path(retained["path"])
    artifact = bind_n16_migration_receipt(artifact_path, inspect_artifact(artifact_path))
    if (artifact != retained or artifact["weights_id"] != recipe["weights_id"]
            or artifact["sha256"] != recipe["sha256"]):
        raise ValueError("terminal base artifact or conversion receipt changed")
    return {"terminal_selection": _identity(selection), "winner": winner,
            "base_artifact": artifact, "cache_group": group,
            "base_benchmark": sources[0]["benchmark_executable"],
            "text_prefill_attention_profile": profile,
            "selected_prefill_chunk": chunk["selected_prefill_chunk"],
            "prefill_chunk_authority": chunk,
            "hybrid_base_authority": _selected_hybrid_authority(base, recipe, chunk["selected_prefill_chunk"])}


def benchmark_profile(path: Path, group: int, profile: str) -> dict:
    """Bind a fresh evaluator build to the selected static configuration."""
    path = path.resolve(strict=True)
    if path.name != "ninfer_bench" or path.parent.name != "bench":
        raise ValueError("DFlash benchmark must identify an explicit build root")
    cache = path.parent.parent / "CMakeCache.txt"
    values = {}
    for line in cache.read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("#", "//")):
            name, value = line.split("=", 1)
            values[name.split(":", 1)[0]] = value
    expected = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
                "NINFER_R9700_KV_VALUE_GROUP": str(group),
                "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
                "NINFER_R9700_W8_ACTIVATION_BITS": "8",
                "NINFER_R9700_FP8_QK_WMMA": "1",
                "NINFER_R9700_XATTENTION_QUALIFICATION":
                    "ON" if profile == "b128-s16-tau900" else "OFF"}
    if profile == "b128-s16-tau900":
        expected.update({"NINFER_R9700_XATTENTION_STRIDE": "16",
                         "NINFER_R9700_XATTENTION_TAU_PERMILLE": "900"})
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("DFlash evaluator build differs from selected base configuration")
    return {"benchmark": inspect_executable(path), "cmake_cache": _identity(cache),
            "configuration": expected}


def candidate_key(recipe: str, k: int, w: int) -> str:
    if recipe not in RECIPES or (k, w) not in DFLASH_PRODUCTION_PROFILES:
        raise ValueError("unsupported recipe/K/W; expected owned recipe and K4/W5 or K5/W6")
    return f"{recipe}/k{k}-w{w}"


def recipe_evidence(route: dict, recipe: str, conversion: Path, root: Path) -> dict:
    if recipe not in RECIPES:
        raise ValueError("unknown DFlash recipe")
    chunk = route["selected_prefill_chunk"]
    manifest = _matrix(root, "dflash-shortlist", prefill_chunk=chunk)
    shortlist = _load(root / "dflash-shortlist.json")
    artifact, bench = shortlist.get("artifact"), shortlist.get("benchmark_executable")
    if (shortlist.get("artifact_type") != "ninfer_dflash_shortlist"
            or shortlist.get("schema_version") != DFLASH_SHORTLIST_SCHEMA_VERSION
            or shortlist.get("pass") is not True):
        raise ValueError("DFlash recipe lacks passing two-width shortlist evidence")
    _validate_physical_identity(artifact, "companion", artifact=True)
    _validate_physical_identity(bench, "benchmark")
    actual = require_dflash_companion(Path(artifact["path"]), inspect_artifact(Path(artifact["path"])))
    if (actual != artifact or DFLASH_COMPANIONS[artifact["weights_id"]] !=
            (route["base_artifact"]["weights_id"], recipe)
            or actual["dflash_base_artifact"] != route["base_artifact"]
            or actual["dflash_conversion_report"] != _identity(conversion.resolve())):
        raise ValueError("recipe evidence changed artifact, recipe, base, or conversion receipt")
    build = benchmark_profile(Path(bench["path"]), route["cache_group"],
                              route["text_prefill_attention_profile"])
    if build["benchmark"] != bench:
        raise ValueError("DFlash benchmark changed")
    if bench["sha256"] == route["base_benchmark"]["sha256"]:
        raise ValueError("DFlash evaluator must be rebuilt with recipe-aware companion admission")
    _same_campaign(manifest, artifact, bench, route["cache_group"],
                   route["text_prefill_attention_profile"], chunk,
                   route["hybrid_base_authority"], [5, 6])
    _records(root, manifest, "dflash-shortlist", 4, 5, chunk)
    _validate_shortlist_output(root, manifest, shortlist, artifact, bench, chunk)
    profiles = {(row["profile"]["draft_tokens_requested"],
                 row["profile"]["verify_width_resolved"])
                for row in shortlist["candidates"] if row.get("valid_for_ranking")}
    if profiles != set(DFLASH_PRODUCTION_PROFILES):
        raise ValueError("recipe shortlist does not retain both qualified production widths")
    return {"recipe": recipe, "artifact": artifact, "benchmark": bench, "build": build,
            "conversion_report": _identity(conversion), "shortlist": _identity(root / "dflash-shortlist.json")}


def capacity_cells(route: dict, evidence: dict, k: int, w: int, root: Path) -> dict:
    chunk = route["selected_prefill_chunk"]
    manifest = _matrix(root, "dflash-capacity", k, w, prefill_chunk=chunk, allow_failures=True)
    _same_campaign(manifest, evidence["artifact"], evidence["benchmark"], route["cache_group"],
                   route["text_prefill_attention_profile"], chunk,
                   route["hybrid_base_authority"], [w])
    reports = _records(root, manifest, "dflash-capacity", k, w, chunk, allow_missing=True)
    failures = _missing_capacity_provenance(root, manifest)
    failed = {row["concurrency"]: row for row in failures}
    if len(failed) != len(failures) or set(reports) & set(failed):
        raise ValueError("capacity cell has duplicate or conflicting outcomes")
    if set(reports) | set(failed) != set(PRODUCT_CONCURRENCIES):
        raise ValueError("every capacity cell needs either a valid report or a bound failure")
    cells = {}
    for concurrency in PRODUCT_CONCURRENCIES:
        if concurrency in failed:
            cells[str(concurrency)] = {"eligible": False, "exclusion": failed[concurrency]}
            continue
        if len(reports[concurrency]) != 1:
            raise ValueError("capacity cell has duplicate reports")
        classified = validate_automatic_feasibility(reports[concurrency][0])
        if classified["measurement_kind"] != "resolved_effective_maximum":
            raise ValueError("capacity cell is not an exact resolved maximum")
        tokens = classified["resolved_effective_maximum_tokens"]
        required_tokens = concurrency * ((32768 + 256 + 2 * w + 63) // 64) * 64
        cells[str(concurrency)] = {"eligible": tokens >= required_tokens, "tokens": tokens,
            **({"exclusion": {"status": "insufficient_32k_generation_capacity",
                              "required_tokens": required_tokens, "resolved_tokens": tokens}}
               if tokens < required_tokens else {})}
    return {"matrix": _identity(root / "manifest.json"), "cells": cells,
            "eligible_concurrency": [c for c in PRODUCT_CONCURRENCIES if cells[str(c)]["eligible"]]}


def performance_cells(route: dict, evidence: dict, k: int, w: int, root: Path,
                      concurrency: Sequence[int]) -> dict:
    """Declared capacity decisions define completeness; successful reports never define it."""
    concurrency = list(concurrency)
    if not concurrency or 1 not in concurrency:
        raise ValueError("DFlash advancement requires eligible C1")
    chunk = route["selected_prefill_chunk"]
    manifest = _matrix(root, "dflash-pareto", k, w, prefill_chunk=chunk)
    _same_campaign(manifest, evidence["artifact"], evidence["benchmark"], route["cache_group"],
                   route["text_prefill_attention_profile"], chunk,
                   route["hybrid_base_authority"], [w])
    _records(root, manifest, "dflash-pareto", k, w, chunk, required_concurrency=concurrency)
    aux = _auxiliary(root, manifest, evidence["artifact"], evidence["benchmark"], k, w, concurrency)
    cases = {(case.suite, case.name): case for case in
             build_cases("dflash-pareto", k, w, production_prefill_chunk=chunk)}
    rows = []
    for record in manifest["commands"]:
        rows.extend(report_rows(Path(record["report"]), cases[(record["suite"], record["case"])],
            route["cache_group"], 8, 8, True, record["concurrency"], evidence["artifact"],
            record["command"], route["text_prefill_attention_profile"]))
    gates = {str(c): _matched_ordinary_speed_gate(
        [row for row in rows if row["concurrency"] == c], [c]) for c in concurrency}
    objectives = {}
    for c in concurrency:
        whole = {row["label"]: row["whole_output_tok_s_mean"] for row in rows
                 if row["suite"] == "dflash_pareto_whole_inference" and row["concurrency"] == c}
        acceptance = {row["label"]: row["spec_acceptance_length"] for row in rows
                      if row["suite"] == "dflash_pareto_decode" and row["concurrency"] == c}
        if (len(whole) != 2 or len(acceptance) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in whole.values())
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value < 1 or value > k + 1 for value in acceptance.values())):
            raise ValueError("declared performance cell lacks complete finite whole/acceptance evidence")
        objectives[str(c)] = {"whole": whole, "acceptance": acceptance}
    return {"matrix": _identity(root / "manifest.json"), "declared_concurrency": concurrency,
            "matched_speed_by_concurrency": gates, "objectives": objectives, **aux}


def _rank_cells(values: dict[str, dict]) -> dict:
    if not values:
        return {"eligible_frontier": [], "winner": None, "normalized_objectives": {}}
    frontier = sorted(name for name in values if not any(
        other != name and _dominates(values[other], values[name]) for other in values))
    normalized = {name: _normalized(name, values, frontier) for name in frontier}
    remaining, decisive = frontier, "canonical_recipe_kw"
    for stage, metric in (("maximin_whole_throughput", "minimum_whole_ratio"),
                          ("maximin_capacity", "minimum_capacity_ratio"),
                          ("maximin_acceptance", "minimum_acceptance_ratio")):
        best = max(normalized[name][metric] for name in remaining)
        remaining = [name for name in remaining if normalized[name][metric] == best]
        if len(remaining) == 1:
            decisive = stage
            break
    return {"eligible_frontier": frontier, "winner": min(remaining),
            "decisive_stage": decisive, "normalized_objectives": normalized}


def assemble(base_selection: Path,
             recipe_inputs: Sequence[tuple[str, Path, Path]],
             capacity_inputs: Sequence[tuple[str, int, int, Path]],
             c1_inputs: Sequence[tuple[str, int, int, Path]],
             pareto_inputs: Sequence[tuple[str, int, int, Path]]) -> dict[str, Any]:
    route = selected_base_route(base_selection)
    recipes = {recipe: recipe_evidence(route, recipe, conversion, root)
               for recipe, conversion, root in recipe_inputs}
    if len(recipes) != len(recipe_inputs) or set(recipes) != set(RECIPES):
        raise ValueError("selection requires all three recipe comparisons exactly once")
    if len({json.dumps(value["benchmark"], sort_keys=True) for value in recipes.values()}) != 1:
        raise ValueError("recipe comparison must use one fresh benchmark executable")
    def index(inputs):
        result = {candidate_key(recipe, k, w): root for recipe, k, w, root in inputs}
        if len(result) != len(inputs):
            raise ValueError("duplicate recipe/K/W evidence")
        return result
    capacities, c1, paretos = map(index, (capacity_inputs, c1_inputs, pareto_inputs))
    expected = {candidate_key(recipe, k, w) for recipe in RECIPES
                for k, w in DFLASH_PRODUCTION_PROFILES}
    if set(capacities) != expected or not set(c1) <= expected or not set(paretos) <= expected:
        raise ValueError("evidence does not cover the exact recipe/two-width candidate set")
    capacity_evidence, screens = {}, {}
    for recipe in RECIPES:
        for k, w in DFLASH_PRODUCTION_PROFILES:
            key = candidate_key(recipe, k, w)
            capacity_evidence[key] = capacity_cells(route, recipes[recipe], k, w, capacities[key])
            if 1 in capacity_evidence[key]["eligible_concurrency"]:
                if key not in c1:
                    raise ValueError("capacity-eligible C1 lacks its required material-win screen")
                screens[key] = performance_cells(route, recipes[recipe], k, w, c1[key], [1])
    has_material_k4 = any(key.endswith("/k4-w5") and
        screen["matched_speed_by_concurrency"]["1"]["pass"] for key, screen in screens.items())
    candidates, expected_c1, expected_pareto = [], set(screens), set()
    values = {str(c): {} for c in PRODUCT_CONCURRENCIES}
    for recipe in RECIPES:
        for k, w in DFLASH_PRODUCTION_PROFILES:
            key = candidate_key(recipe, k, w)
            capacity = capacity_evidence[key]
            row = {"key": key, "recipe": recipe, "draft_tokens": k, "verify_width": w,
                   "capacity": capacity, "qualified_concurrency": [], "exclusions": {
                       c: cell["exclusion"] for c, cell in capacity["cells"].items()
                       if not cell["eligible"]}}
            declared = capacity["eligible_concurrency"]
            if 1 in declared:
                screen = screens[key]
                row["c1_screen"] = screen
                if screen["matched_speed_by_concurrency"]["1"]["pass"] and has_material_k4:
                    expected_pareto.add(key)
                    if key not in paretos:
                        raise ValueError("C1 material survivor lacks declared-concurrency followup")
                    performance = performance_cells(route, recipes[recipe], k, w, paretos[key], declared)
                    row["performance"] = performance
                    for c in declared:
                        gate = performance["matched_speed_by_concurrency"][str(c)]
                        if gate["pass"]:
                            row["qualified_concurrency"].append(c)
                            values[str(c)][key] = {
                                **performance["objectives"][str(c)],
                                "capacity": {"tokens": capacity["cells"][str(c)]["tokens"]}}
                        else:
                            row["exclusions"][str(c)] = {"status": "no_material_matched_speed", "gate": gate}
                else:
                    for c in declared:
                        row["exclusions"][str(c)] = {"status": "C1_material_gate_failed"
                            if not screen["matched_speed_by_concurrency"]["1"]["pass"]
                            else "no_material_K4_C1_route"}
            else:
                for c in declared:
                    row["exclusions"][str(c)] = {"status": "C1_capacity_gate_failed"}
            candidates.append(row)
    if set(c1) != expected_c1 or set(paretos) != expected_pareto:
        raise ValueError("followup evidence differs from declared capacity/C1 advancement decisions")
    ranked = {c: _rank_cells(cells) for c, cells in values.items()}
    winner = ranked["1"]["winner"]
    winner_row = next((row for row in candidates if row["key"] == winner), None)
    return {"artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
            "status": "evaluation_complete" if winner else "no_qualified_C1_winner",
            "selection_rule": RULE, "production_selected": False,
            "scope": "primary C1 evaluation recommendation; per-C evidence is not runtime recipe switching",
            "selected_base": route, "recipes": recipes, "candidates": candidates,
            "winner": winner, "per_concurrency": ranked,
            "winner_qualified_concurrency": winner_row["qualified_concurrency"] if winner_row else [],
            "winner_exclusions": winner_row["exclusions"] if winner_row else {}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--recipe", action="append", nargs=3,
                        metavar=("RECIPE", "CONVERSION", "SHORTLIST"), required=True)
    for name in ("capacity", "c1", "pareto"):
        parser.add_argument("--" + name, action="append", nargs=4,
                            metavar=("RECIPE", "K", "W", "DIR"), default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    recipes = [(key, Path(conversion).resolve(), Path(root).resolve())
               for key, conversion, root in args.recipe]
    def inputs(name):
        return [(key, int(k), int(w), Path(root).resolve())
                for key, k, w, root in getattr(args, name)]
    def result():
        return assemble(args.base_selection.resolve(), recipes, inputs("capacity"),
                        inputs("c1"), inputs("pareto"))
    value = result()
    from tools.bench.prefill_chunk_authority import durable_create_json
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Recompute before publication, so a changed input never leaves a selected authority.
    if result() != value:
        raise ValueError("DFlash inputs changed during selection")
    durable_create_json(args.out, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
