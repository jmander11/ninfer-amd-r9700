#!/usr/bin/env python3
"""Classify qualified R9700 model profiles on a quality/speed/capacity frontier.

The input must contain complete whole-inference throughput for an identical set
of named workloads. Scorer ``score_seconds`` is deliberately not an objective:
it measures a teacher-forced diagnostic program, not the Engine workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ppl.run import QUALITY_TIERS
from tools.bench.run_ninfer_bench_matrix import PRODUCTION_PREFILL_CHUNKS


ARTIFACT_TYPE = "ninfer_r9700_pareto_comparison"
SCHEMA_VERSION = 7
INPUT_ARTIFACT_TYPE = "ninfer_r9700_pareto_input"
INPUT_SCHEMA_VERSION = 4
SELECTION_RULE = "same_recipe_static_profile_maximin_v2"
TERMINAL_SELECTION_RULE = (
    "global_maximin_whole_then_capacity_then_quality_then_canonical_v1"
)
XATTENTION_PROFILES = ("dense", "b128-s16-tau900")
TERMINAL_RECIPE_PROFILES = {
    "r9700-q4g64-n16k16-eval": "all-q4g64-v1",
    "r9700-q4-w8-mse-n16k16-eval": "mixed-q4g64-w8g32-mse-v1",
    "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        "four-role-rowwise-f8e4m3-all-other-q4g64-v0",
}


def _shortlist_head_precision_gate(value: object) -> dict:
    if not isinstance(value, dict) or (
        value.get("head_format") != "Q4G64_F16S"
        or value.get("activation_profile") != "A8G64"
        or type(value.get("all_repetitions_minimum")) is not bool
        or value.get("counter_semantics")
        != ("per-repetition counters sum request-lane rounds; every whole g256 MTP3 "
            "cell requires concurrency * 64 rounds")
    ):
        raise ValueError("candidate lacks the exact Q4 shortlist-head round gate")
    cells = value.get("cells")
    expected_keys = {
        f"whole-pp{prompt}+tg256_c{concurrency}"
        for prompt in (8192, 32768) for concurrency in range(1, 5)
    }
    if not isinstance(cells, dict) or set(cells) != expected_keys:
        raise ValueError("Q4 shortlist-head round gate lacks the exact 8K/32K C=1..4 cells")
    all_minimum = True
    for key, cell in cells.items():
        concurrency = int(key.rsplit("_c", 1)[1])
        expected_rounds = 64 * concurrency
        repetitions = cell.get("repetitions") if isinstance(cell, dict) else None
        if (
            cell.get("generated_tokens_per_lane") != 256
            or cell.get("draft_window") != 3
            or cell.get("minimum_rounds_per_lane") != 64
            or cell.get("concurrency") != concurrency
            or cell.get("expected_rounds_per_repetition") != expected_rounds
            or cell.get("expected_rounds_all_repetitions") != expected_rounds * 3
            or not isinstance(repetitions, list) or len(repetitions) != 3
        ):
            raise ValueError(f"Q4 shortlist-head round gate cell {key} is malformed")
        observed = 0
        cell_minimum = True
        for index, repetition in enumerate(repetitions):
            if not isinstance(repetition, dict) or (
                repetition.get("repetition") != index
                or type(repetition.get("observed_rounds")) is not int
                or repetition["observed_rounds"] < expected_rounds
                or repetition.get("expected_minimum_rounds") != expected_rounds
                or repetition.get("minimum_met")
                is not (repetition["observed_rounds"] == expected_rounds)
            ):
                raise ValueError(f"Q4 shortlist-head round gate cell {key} repetition differs")
            observed += repetition["observed_rounds"]
            cell_minimum = cell_minimum and repetition["minimum_met"]
        if (
            cell.get("observed_rounds_all_repetitions") != observed
            or cell.get("all_repetitions_minimum") is not cell_minimum
        ):
            raise ValueError(f"Q4 shortlist-head round gate cell {key} aggregate differs")
        all_minimum = all_minimum and cell_minimum
    expected_status = (
        "q4_round_gate_pass_pending_trace"
        if all_minimum else "conditional_head_precision_required"
    )
    if (
        value["all_repetitions_minimum"] is not all_minimum
        or value.get("status") != expected_status
    ):
        raise ValueError("Q4 shortlist-head round gate outcome differs from its repetitions")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _weight_storage_profile(identity: object) -> str | None:
    if not isinstance(identity, dict) or identity.get("kind") != "artifact":
        return None
    return TERMINAL_RECIPE_PROFILES.get(identity.get("weights_id"))


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _quality_objective(quality: object, label: str) -> tuple[dict | None, list[str]]:
    reasons: list[str] = []
    if not isinstance(quality, dict):
        reasons.append(f"missing_quality_measurement:{label}")
        quality = {}
    tier = quality.get("tier")
    if tier not in QUALITY_TIERS:
        reasons.append(f"missing_explicit_quality_tier:{label}")
    mean_delta = quality.get("mean_nll_delta")
    scored_positions = quality.get("scored_positions")
    new_severe_positions = quality.get("new_severe_positions")
    complete = quality.get("complete_finite_aligned") is True
    if not complete:
        reasons.append(f"quality_sidecars_not_complete_finite_aligned:{label}")
    if not _finite_number(mean_delta):
        reasons.append(f"missing_finite_mean_nll_delta:{label}")
    if type(scored_positions) is not int or scored_positions <= 0:
        reasons.append(f"missing_positive_scored_positions:{label}")
    if type(new_severe_positions) is not int or new_severe_positions < 0:
        reasons.append(f"missing_nonnegative_new_severe_positions:{label}")

    computed_eligible = False
    severe_rate = None
    severe_budget = None
    if (
        tier in QUALITY_TIERS
        and _finite_number(mean_delta)
        and type(scored_positions) is int
        and scored_positions > 0
        and type(new_severe_positions) is int
        and new_severe_positions >= 0
        and complete
    ):
        spec = QUALITY_TIERS[tier]
        severe_rate = new_severe_positions / scored_positions
        severe_budget = max(
            spec["minimum_new_severe_budget"],
            math.ceil(spec["maximum_new_severe_rate"] * scored_positions),
        )
        computed_eligible = (
            float(mean_delta) <= spec["maximum_mean_nll_delta"]
            and new_severe_positions <= severe_budget
        )
        if not computed_eligible:
            reasons.append(f"quality_guardrails_not_met:{label}")
    if quality.get("eligible") is not computed_eligible:
        reasons.append(f"declared_quality_eligibility_mismatch:{label}")

    if reasons:
        return None, reasons
    return {
        "quality_tier": tier,
        "mean_nll_delta": float(mean_delta),
        "new_severe_position_rate": severe_rate,
        "new_severe_positions": new_severe_positions,
        "scored_positions": scored_positions,
        "new_severe_position_budget": severe_budget,
    }, []


def _capacity_objective(capacity: object, label: str) -> tuple[int | None, list[str]]:
    if not isinstance(capacity, dict):
        return None, [f"missing_resolved_effective_maximum_capacity:{label}"]
    if capacity.get("measurement_kind") != "resolved_effective_maximum":
        return None, [f"capacity_is_not_resolved_effective_maximum:{label}"]
    if capacity.get("binding_constraint") not in ("device_memory", "model_context"):
        return None, [f"invalid_effective_capacity_constraint:{label}"]
    tokens = capacity.get("tokens")
    if type(tokens) is not int or tokens <= 0:
        return None, [f"missing_positive_capacity_tokens:{label}"]
    return tokens, []


def _candidate_objectives(
    record: dict,
    required_workloads: list[str],
    required_quality_cells: list[str],
    required_capacity_cells: list[str],
) -> tuple[dict | None, list[str]]:
    reasons: list[str] = []
    if required_quality_cells:
        supplied_quality = record.get("quality_cells")
        if not isinstance(supplied_quality, dict):
            supplied_quality = {}
        quality_cells: dict[str, dict] = {}
        for cell in required_quality_cells:
            objective, cell_reasons = _quality_objective(supplied_quality.get(cell), cell)
            reasons.extend(cell_reasons)
            if objective is not None:
                quality_cells[cell] = objective
    else:
        objective, cell_reasons = _quality_objective(record.get("quality"), "quality")
        reasons.extend(reason.replace(":quality", "") for reason in cell_reasons)
        quality_cells = {"quality": objective} if objective is not None else {}

    if required_capacity_cells:
        supplied_capacity = record.get("capacity_by_cell")
        if not isinstance(supplied_capacity, dict):
            supplied_capacity = {}
        capacity_cells: dict[str, int] = {}
        for cell in required_capacity_cells:
            capacity, cell_reasons = _capacity_objective(supplied_capacity.get(cell), cell)
            reasons.extend(cell_reasons)
            if capacity is not None:
                capacity_cells[cell] = capacity
    else:
        capacity, cell_reasons = _capacity_objective(record.get("capacity"), "capacity")
        reasons.extend(reason.replace(":capacity", "") for reason in cell_reasons)
        capacity_cells = {}
        if capacity is not None:
            capacity_cells["capacity"] = capacity

    speeds = record.get("whole_inference_tokens_per_second")
    if not isinstance(speeds, dict):
        speeds = {}
    for workload in required_workloads:
        value = speeds.get(workload)
        if not _finite_number(value) or (_finite_number(value) and float(value) <= 0.0):
            reasons.append(f"missing_positive_whole_inference_speed:{workload}")

    if reasons:
        return None, reasons
    return {
        "quality_cells": quality_cells,
        "capacity_tokens_by_cell": capacity_cells,
        "whole_inference_tokens_per_second": {
            workload: float(speeds[workload]) for workload in required_workloads
        },
    }, []


def dominates(first: dict, second: dict, required_workloads: list[str]) -> bool:
    """Return whether first is no worse everywhere and strictly better somewhere."""

    quality_cells = first["quality_cells"].keys()
    capacity_cells = first["capacity_tokens_by_cell"].keys()
    no_worse = (
        all(
            first["quality_cells"][cell][metric]
            <= second["quality_cells"][cell][metric]
            for cell in quality_cells
            for metric in ("mean_nll_delta", "new_severe_position_rate")
        )
        and all(
            first["capacity_tokens_by_cell"][cell]
            >= second["capacity_tokens_by_cell"][cell]
            for cell in capacity_cells
        )
        and all(
            first["whole_inference_tokens_per_second"][workload]
            >= second["whole_inference_tokens_per_second"][workload]
            for workload in required_workloads
        )
    )
    strictly_better = (
        any(
            first["quality_cells"][cell][metric]
            < second["quality_cells"][cell][metric]
            for cell in quality_cells
            for metric in ("mean_nll_delta", "new_severe_position_rate")
        )
        or any(
            first["capacity_tokens_by_cell"][cell]
            > second["capacity_tokens_by_cell"][cell]
            for cell in capacity_cells
        )
        or any(
            first["whole_inference_tokens_per_second"][workload]
            > second["whole_inference_tokens_per_second"][workload]
            for workload in required_workloads
        )
    )
    return no_worse and strictly_better


def _recipe_identity(
    record: dict, provenance_by_candidate: dict[str, list[dict]]
) -> dict | None:
    """Return a stable weight-recipe identity without deriving it from a candidate name."""

    matches = provenance_by_candidate.get(record["name"], [])
    if matches:
        if len(matches) != 1:
            return None
        artifact = matches[0].get("artifact")
        if not isinstance(artifact, dict):
            return None
        weights_id = artifact.get("weights_id")
        sha256 = artifact.get("sha256")
        if not isinstance(weights_id, str) or not weights_id:
            return None
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            return None
        return {"kind": "artifact", "weights_id": weights_id, "sha256": sha256}

    declared = record.get("weight_recipe")
    if declared is not None:
        if not isinstance(declared, str) or not declared:
            raise ValueError(f"candidate {record['name']} has an invalid weight_recipe")
        return {"kind": "declared", "value": declared}
    return None


def _recipe_key(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _static_profile_tuple(cache_profile: dict, execution_profile: dict) -> tuple:
    layouts = cache_profile["plane_layouts"]
    return (
        cache_profile["value_group"], layouts["key"], layouts["value"],
        layouts["value_scale"], execution_profile["q4_activation_bits"],
        execution_profile["w8_activation_bits"], execution_profile["fp8_qk_wmma_profile"],
        execution_profile["xattention_profile"],
    )


def _normalized_selection_values(
    name: str,
    objectives: dict[str, dict],
    frontier_names: list[str],
    workloads: list[str],
) -> dict:
    objective = objectives[name]
    speed_best = {
        workload: max(
            objectives[other]["whole_inference_tokens_per_second"][workload]
            for other in frontier_names
        )
        for workload in workloads
    }
    capacity_cells = objective["capacity_tokens_by_cell"].keys()
    capacity_best = {
        cell: max(
            objectives[other]["capacity_tokens_by_cell"][cell]
            for other in frontier_names
        )
        for cell in capacity_cells
    }
    speed_ratios = {
        workload: objective["whole_inference_tokens_per_second"][workload]
        / speed_best[workload]
        for workload in workloads
    }
    capacity_ratios = {
        cell: objective["capacity_tokens_by_cell"][cell] / capacity_best[cell]
        for cell in capacity_cells
    }
    quality_fractions: dict[str, dict[str, float]] = {}
    for cell, quality in objective["quality_cells"].items():
        tier = QUALITY_TIERS[quality["quality_tier"]]
        quality_fractions[cell] = {
            "mean_nll_delta": quality["mean_nll_delta"]
            / tier["maximum_mean_nll_delta"],
            "new_severe_positions": quality["new_severe_positions"]
            / quality["new_severe_position_budget"],
        }
    return {
        "throughput_ratio_by_workload": speed_ratios,
        "minimum_throughput_ratio": min(speed_ratios.values()),
        "capacity_ratio_by_cell": capacity_ratios,
        "minimum_capacity_ratio": min(capacity_ratios.values()),
        "quality_budget_fraction_by_cell": quality_fractions,
        "worst_quality_budget_fraction": max(
            fraction
            for cell in quality_fractions.values()
            for fraction in cell.values()
        ),
    }


def _select_cache_profiles(
    results: list[dict], objectives: dict[str, dict], workloads: list[str],
    *, include_singleton: bool = False,
) -> dict:
    frontier_results = [result for result in results if result["pareto"]]
    selection_pool = (
        [result for result in results if result["name"] in objectives]
        if include_singleton else frontier_results
    )
    groups: dict[str, list[dict]] = {}
    identities: dict[str, dict] = {}
    for result in selection_pool:
        identity = result["weight_recipe"]
        if identity is None:
            continue
        key = _recipe_key(identity)
        identities[key] = identity
        groups.setdefault(key, []).append(result)

    selections: list[dict] = []
    collapsed_names: set[str] = set()
    selection_frontier_names: set[str] = {
        result["name"] for result in selection_pool if result["weight_recipe"] is None
    }
    for key in sorted(groups):
        candidates = groups[key]
        group = [
            result for result in candidates
            if not any(
                other["name"] != result["name"]
                and dominates(
                    objectives[other["name"]], objectives[result["name"]], workloads
                )
                for other in candidates
            )
        ]
        selection_frontier_names.update(result["name"] for result in group)
        if len(group) < 2 and not include_singleton:
            continue
        names = sorted(result["name"] for result in group)
        normalized = {
            name: _normalized_selection_values(name, objectives, names, workloads)
            for name in names
        }
        remaining = names
        stages = (
            ("maximin_whole_inference_throughput", "minimum_throughput_ratio", max),
            ("maximin_resolved_capacity", "minimum_capacity_ratio", max),
            ("minimum_quality_budget_consumption", "worst_quality_budget_fraction", min),
        )
        decisive_stage = (
            "sole_pareto_frontier_candidate" if len(names) == 1
            else "canonical_static_profile_identity"
        )
        if len(names) > 1:
            for stage, metric, chooser in stages:
                best = chooser(normalized[name][metric] for name in remaining)
                narrowed = [name for name in remaining if normalized[name][metric] == best]
                if len(narrowed) == 1:
                    remaining = narrowed
                    decisive_stage = stage
                    break
                remaining = narrowed
        if len(remaining) > 1:
            by_name = {result["name"]: result for result in group}
            winner = min(
                remaining,
                key=lambda name: (
                    _static_profile_tuple(
                        by_name[name]["cache_profile"], by_name[name]["execution_profile"]
                    ),
                    name,
                ),
            )
        else:
            winner = remaining[0]
        collapsed_names.update(names)
        canonical_profiles = {
            result["name"]: list(_static_profile_tuple(
                result["cache_profile"], result["execution_profile"]
            ))
            for result in group
        }
        winner_result = next(result for result in group if result["name"] == winner)
        selections.append({
            "weight_recipe": identities[key],
            "frontier_candidates": names,
            "winner": winner,
            "winner_cache_profile": winner_result["cache_profile"],
            "winner_execution_profile": winner_result["execution_profile"],
            "decisive_stage": decisive_stage,
            "rationale": {
                "ordered_stages": [
                    "maximin_whole_inference_throughput",
                    "maximin_resolved_capacity",
                    "minimum_quality_budget_consumption",
                    "canonical_static_profile_identity",
                ],
                "decisive_stage": decisive_stage,
                "canonical_static_profile_tuple_by_candidate": canonical_profiles,
            },
            "normalized_objectives": normalized,
        })
    return {
        "rule": SELECTION_RULE,
        "selections": selections,
        "not_collapsed": sorted(selection_frontier_names - collapsed_names),
    }


def _terminal_production_selection(
    results: list[dict], objectives: dict[str, dict], workloads: list[str],
    profile_selection: dict,
) -> dict:
    """Choose one production artifact/profile from the retained per-recipe winners."""

    winners = sorted(
        selection["winner"] for selection in profile_selection["selections"]
    )
    if not winners:
        raise ValueError("terminal production selection has no eligible recipe winner")
    by_name = {result["name"]: result for result in results}
    normalized = {
        name: _normalized_selection_values(name, objectives, winners, workloads)
        for name in winners
    }
    remaining = winners
    decisive_stage = "canonical_artifact_and_static_profile_identity"
    for stage, metric, chooser in (
        ("maximin_whole_inference_throughput", "minimum_throughput_ratio", max),
        ("maximin_resolved_capacity", "minimum_capacity_ratio", max),
        ("minimum_quality_budget_consumption", "worst_quality_budget_fraction", min),
    ):
        best = chooser(normalized[name][metric] for name in remaining)
        narrowed = [name for name in remaining if normalized[name][metric] == best]
        if len(narrowed) == 1:
            remaining = narrowed
            decisive_stage = stage
            break
        remaining = narrowed
    if len(remaining) == 1:
        winner = remaining[0]
    else:
        winner = min(
            remaining,
            key=lambda name: (
                _recipe_key(by_name[name]["weight_recipe"]),
                _static_profile_tuple(
                    by_name[name]["cache_profile"], by_name[name]["execution_profile"]
                ),
                name,
            ),
        )
    selected = by_name[winner]
    shortlist_head_gate = selected["shortlist_head_precision_gate"]
    if not isinstance(shortlist_head_gate, dict):
        raise ValueError("terminal production winner lacks shortlist-head precision evidence")
    return {
        "rule": TERMINAL_SELECTION_RULE,
        "eligible_profile_winners": winners,
        "winner": winner,
        "winner_artifact": selected["weight_recipe"],
        "winner_cache_profile": selected["cache_profile"],
        "winner_execution_profile": selected["execution_profile"],
        "shortlist_head_precision_status": shortlist_head_gate["status"],
        "production_status": (
            "selected_route_pending_shortlist_head_trace_and_niah"
            if shortlist_head_gate["all_repetitions_minimum"]
            else "conditional_head_precision_required"
        ),
        "shortlist_head_precision_gate": shortlist_head_gate,
        "decisive_stage": decisive_stage,
        "normalized_objectives": normalized,
        "rationale": {
            "ordered_stages": [
                "maximin_whole_inference_throughput",
                "maximin_resolved_capacity",
                "minimum_quality_budget_consumption",
                "canonical_artifact_and_static_profile_identity",
            ],
            "exact_measured_means_without_noise_tolerance": True,
        },
    }


def classify(payload: dict) -> dict:
    if (
        payload.get("artifact_type") != INPUT_ARTIFACT_TYPE
        or payload.get("schema_version") != INPUT_SCHEMA_VERSION
    ):
        raise ValueError("Pareto input must be ninfer_r9700_pareto_input schema v4")
    workloads = payload.get("required_speed_workloads")
    if (
        not isinstance(workloads, list)
        or not workloads
        or any(not isinstance(name, str) or not name for name in workloads)
        or len(set(workloads)) != len(workloads)
    ):
        raise ValueError("required_speed_workloads must be a nonempty unique string list")
    records = payload.get("candidates")
    if not isinstance(records, list) or not records:
        raise ValueError("candidates must be a nonempty list")
    require_single_selection = payload.get("require_single_static_profile_selection", False)
    if type(require_single_selection) is not bool:
        raise ValueError("require_single_static_profile_selection must be boolean")
    selected_prefill_chunk = payload.get("selected_prefill_chunk")
    if selected_prefill_chunk is not None and selected_prefill_chunk not in PRODUCTION_PREFILL_CHUNKS:
        raise ValueError("selected_prefill_chunk is unsupported")
    if require_single_selection and selected_prefill_chunk is None:
        raise ValueError("static profile selection requires one explicit selected_prefill_chunk")
    prefill_chunk_selection = payload.get("prefill_chunk_selection")
    if require_single_selection and (
        not isinstance(prefill_chunk_selection, dict)
        or not isinstance(prefill_chunk_selection.get("path"), str)
        or not _valid_sha256(prefill_chunk_selection.get("sha256"))
        or prefill_chunk_selection.get("selection_rule")
        != "global_maximin_normalized_prefill_then_workspace_then_smaller_chunk_v2"
        or prefill_chunk_selection.get("selected_prefill_chunk") != selected_prefill_chunk
    ):
        raise ValueError("static profile selection lacks its global prefill-chunk authority")
    quality_cells = payload.get("required_quality_cells", [])
    capacity_cells = payload.get("required_capacity_cells", [])
    for label, values in (
        ("required_quality_cells", quality_cells),
        ("required_capacity_cells", capacity_cells),
    ):
        if (
            not isinstance(values, list)
            or any(not isinstance(name, str) or not name for name in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{label} must be a unique string list")
    invalid_concurrency = set()
    for name in (*workloads, *capacity_cells):
        match = re.search(r"(?:_c|^c)(\d+)$", name, re.IGNORECASE)
        if match is not None and int(match.group(1)) not in (1, 2, 3, 4):
            invalid_concurrency.add(int(match.group(1)))
    if invalid_concurrency:
        raise ValueError(
            "Pareto selection inputs are limited to product concurrency C=1..4"
        )

    provenance_by_candidate: dict[str, list[dict]] = {}
    provenance = payload.get("source_provenance", [])
    if isinstance(provenance, list):
        for item in provenance:
            if isinstance(item, dict) and isinstance(item.get("candidate"), str):
                provenance_by_candidate.setdefault(item["candidate"], []).append(item)

    names: list[str] = []
    objectives: dict[str, dict] = {}
    results: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each candidate must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("candidate names must be nonempty and unique")
        names.append(name)
        if require_single_selection and record.get("prefill_chunk") != selected_prefill_chunk:
            raise ValueError(
                f"candidate {name} does not bind selected prefill chunk {selected_prefill_chunk}"
            )
        supplied_objective_keys = []
        for field in ("whole_inference_tokens_per_second", "capacity_by_cell"):
            values = record.get(field)
            if isinstance(values, dict):
                supplied_objective_keys.extend(values)
        if any(
            (match := re.search(r"(?:_c|^c)(\d+)$", key, re.IGNORECASE)) is not None
            and int(match.group(1)) not in (1, 2, 3, 4)
            for key in supplied_objective_keys
        ):
            raise ValueError(
                f"candidate {name} contains a non-product C5+ objective cell"
            )
        cache_profile = record.get("cache_profile")
        execution_profile = record.get("execution_profile")
        plane_layouts = (
            cache_profile.get("plane_layouts") if isinstance(cache_profile, dict) else None
        )
        if (
            not isinstance(cache_profile, dict)
            or cache_profile.get("value_group") not in (16, 32)
            or not isinstance(plane_layouts, dict)
            or set(plane_layouts) != {"key", "value", "value_scale"}
            or any(not isinstance(value, str) or not value for value in plane_layouts.values())
        ):
            raise ValueError(f"candidate {name} has no complete cache_profile identity")
        if (
            not isinstance(execution_profile, dict)
            or set(execution_profile) != {
                "q4_activation_bits", "w8_activation_bits", "fp8_qk_wmma_profile",
                "xattention_profile",
            }
            or execution_profile.get("q4_activation_bits") != 8
            or execution_profile.get("w8_activation_bits") != 8
            or execution_profile.get("fp8_qk_wmma_profile")
            != "t1-ge64-t2-ge320-t3plus-stream-v1"
            or execution_profile.get("xattention_profile")
            not in ("dense", "b128-s16-tau900")
        ):
            raise ValueError(f"candidate {name} has no complete execution_profile identity")
        objective, reasons = _candidate_objectives(
            record, workloads, quality_cells, capacity_cells
        )
        shortlist_head_gate = (
            _shortlist_head_precision_gate(record.get("shortlist_head_precision_gate"))
            if require_single_selection else None
        )
        if objective is not None:
            objectives[name] = objective
        weight_recipe = _recipe_identity(record, provenance_by_candidate)
        weight_storage_profile = _weight_storage_profile(weight_recipe)
        if require_single_selection and weight_storage_profile is None:
            raise ValueError(f"candidate {name} has an unsupported terminal weight recipe")
        results.append({
            "name": name,
            "cache_profile": cache_profile,
            "execution_profile": execution_profile,
            "prefill_chunk": record.get("prefill_chunk"),
            "weight_recipe": weight_recipe,
            "weight_storage_profile": weight_storage_profile,
            "shortlist_head_precision_gate": shortlist_head_gate,
            "comparable": objective is not None,
            "reasons": reasons,
        })

    for result in results:
        name = result["name"]
        if name not in objectives:
            result.update({"pareto": False, "dominated_by": []})
            continue
        dominated_by = sorted(
            other
            for other in objectives
            if other != name and dominates(objectives[other], objectives[name], workloads)
        )
        result.update({
            "pareto": not dominated_by,
            "dominated_by": dominated_by,
            "objectives": objectives[name],
        })

    results.sort(key=lambda result: result["name"])
    if require_single_selection:
        expected_recipe_ids = set(TERMINAL_RECIPE_PROFILES)
        expected_profiles = {
            (16, "dense"), (32, "dense"),
            (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
        }
        recipe_groups: dict[str, list[dict]] = {}
        for result in results:
            identity = result["weight_recipe"]
            if not isinstance(identity, dict) or identity.get("kind") != "artifact":
                raise ValueError(
                    "static XAttention selection requires provenance-bound artifacts"
                )
            recipe_groups.setdefault(_recipe_key(identity), []).append(result)
        if not recipe_groups:
            raise ValueError("static XAttention selection requires candidate recipes")
        artifact_keys_by_weights_id: dict[str, set[str]] = {}
        for key, group in recipe_groups.items():
            identity = group[0]["weight_recipe"]
            artifact_keys_by_weights_id.setdefault(identity["weights_id"], set()).add(key)
        if any(len(keys) != 1 for keys in artifact_keys_by_weights_id.values()):
            raise ValueError(
                "static XAttention selection requires one artifact hash per weights_id"
            )
        if (
            set(artifact_keys_by_weights_id) != expected_recipe_ids
            or len(results) != len(expected_recipe_ids) * len(expected_profiles)
        ):
            raise ValueError(
                "static XAttention selection requires exactly all three product recipe "
                "profile quartets"
            )
        for group in recipe_groups.values():
            actual_profiles = {
                (result["cache_profile"]["value_group"],
                 result["execution_profile"]["xattention_profile"])
                for result in group
            }
            if len(group) != len(expected_profiles) or actual_profiles != expected_profiles:
                raise ValueError(
                    "static XAttention selection requires dense/sparse G16/G32 "
                    "controls for every candidate artifact"
                )
    selection = _select_cache_profiles(
        results, objectives, workloads, include_singleton=require_single_selection
    )
    if require_single_selection and (
        not selection["selections"] or selection["not_collapsed"]
    ):
        raise ValueError("static XAttention selection left an uncollapsed frontier profile")
    terminal_selection = (
        _terminal_production_selection(results, objectives, workloads, selection)
        if require_single_selection else None
    )
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "required_speed_workloads": workloads,
        "required_quality_cells": quality_cells or ["quality"],
        "required_capacity_cells": capacity_cells or ["capacity"],
        "objective_directions": {
            "quality_cells.mean_nll_delta": "minimize_each_required_cell",
            "quality_cells.new_severe_position_rate": "minimize_each_required_cell",
            "whole_inference_tokens_per_second": "maximize_each_required_workload",
            "capacity_tokens_by_cell": "maximize_each_required_cell",
        },
        "score_seconds_is_objective": False,
        "single_static_profile_selection_required": require_single_selection,
        "selected_prefill_chunk": selected_prefill_chunk,
        "prefill_chunk_selection": prefill_chunk_selection,
        "frontier": sorted(result["name"] for result in results if result["pareto"]),
        "same_recipe_static_profile_selection": selection,
        "terminal_production_selection": terminal_selection,
        "source_provenance": provenance,
        "candidates": results,
    }


def validate_terminal_production_authority(value: object) -> tuple[dict, dict]:
    """Strictly recompute a complete three-recipe schema-v7 terminal decision."""

    if not isinstance(value, dict) or (
        value.get("artifact_type") != ARTIFACT_TYPE
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("single_static_profile_selection_required") is not True
    ):
        raise ValueError("static profile authority is not a required schema-v7 decision")
    candidates = value.get("candidates")
    workloads = value.get("required_speed_workloads")
    selected_prefill_chunk = value.get("selected_prefill_chunk")
    prefill_chunk_selection = value.get("prefill_chunk_selection")
    source_provenance = value.get("source_provenance")
    input_authority = value.get("pareto_input")
    if (
        not isinstance(candidates, list) or len(candidates) != 12
        or not isinstance(workloads, list) or not workloads
        or selected_prefill_chunk not in PRODUCTION_PREFILL_CHUNKS
        or not isinstance(prefill_chunk_selection, dict)
        or not isinstance(prefill_chunk_selection.get("path"), str)
        or not _valid_sha256(prefill_chunk_selection.get("sha256"))
        or prefill_chunk_selection.get("selection_rule")
        != "global_maximin_normalized_prefill_then_workspace_then_smaller_chunk_v2"
        or prefill_chunk_selection.get("selected_prefill_chunk") != selected_prefill_chunk
        or any(not isinstance(row, dict) for row in candidates)
        or not isinstance(source_provenance, list)
        or len(source_provenance) != 12
        or not isinstance(input_authority, dict)
        or set(input_authority) != {"path", "sha256"}
        or not isinstance(input_authority.get("path"), str)
        or not _valid_sha256(input_authority.get("sha256"))
    ):
        raise ValueError("schema-v7 authority lacks the exact twelve candidate controls")
    names = [row.get("name") for row in candidates]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 12:
        raise ValueError("schema-v7 authority has malformed candidate names")
    objectives: dict[str, dict] = {}
    recipe_profiles: dict[str, set[tuple[int, str]]] = {}
    recipe_identities: dict[str, dict] = {}
    provenance_by_candidate: dict[str, list[dict]] = {}
    for source in source_provenance:
        if isinstance(source, dict) and isinstance(source.get("candidate"), str):
            provenance_by_candidate.setdefault(source["candidate"], []).append(source)
    if len(provenance_by_candidate) != 12:
        raise ValueError("schema-v7 authority has malformed source provenance")
    for row in candidates:
        recipe = row.get("weight_recipe")
        cache = row.get("cache_profile")
        execution = row.get("execution_profile")
        objective = row.get("objectives")
        if (
            row.get("comparable") is not True or row.get("reasons") != []
            or not isinstance(objective, dict)
            or not isinstance(recipe, dict) or recipe.get("kind") != "artifact"
            or not isinstance(recipe.get("weights_id"), str)
            or row.get("weight_storage_profile")
            != TERMINAL_RECIPE_PROFILES.get(recipe.get("weights_id"))
            or not isinstance(recipe.get("sha256"), str)
            or len(recipe["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in recipe["sha256"])
            or not isinstance(cache, dict) or cache.get("value_group") not in (16, 32)
            or not isinstance(execution, dict)
            or execution.get("xattention_profile") not in XATTENTION_PROFILES
            or row.get("prefill_chunk") != selected_prefill_chunk
            or _recipe_identity(row, provenance_by_candidate) != recipe
        ):
            raise ValueError("schema-v7 authority has malformed candidate provenance/objectives")
        weights_id = recipe["weights_id"]
        prior = recipe_identities.setdefault(weights_id, recipe)
        if prior != recipe:
            raise ValueError("schema-v7 authority binds multiple hashes to one weights_id")
        recipe_profiles.setdefault(weights_id, set()).add((
            cache["value_group"], execution["xattention_profile"],
        ))
        objectives[row["name"]] = objective
    expected_recipes = set(TERMINAL_RECIPE_PROFILES)
    expected_profiles = {
        (16, "dense"), (32, "dense"),
        (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
    }
    if set(recipe_profiles) != expected_recipes or any(
        profiles != expected_profiles for profiles in recipe_profiles.values()
    ):
        raise ValueError("schema-v7 authority lacks all three complete recipe profile quartets")
    recomputed_frontier = []
    for row in candidates:
        dominated_by = sorted(
            other for other in objectives
            if other != row["name"]
            and dominates(objectives[other], objectives[row["name"]], workloads)
        )
        if row.get("dominated_by") != dominated_by or row.get("pareto") != (not dominated_by):
            raise ValueError("schema-v7 candidate Pareto status does not recompute exactly")
        if not dominated_by:
            recomputed_frontier.append(row["name"])
    actual_frontier = sorted(recomputed_frontier)
    if value.get("frontier") != actual_frontier or not actual_frontier:
        raise ValueError("schema-v7 authority frontier disagrees with candidate rows")
    expected_profile_selection = _select_cache_profiles(
        candidates, objectives, workloads, include_singleton=True
    )
    if (
        len(expected_profile_selection["selections"]) != 3
        or expected_profile_selection["not_collapsed"]
        or value.get("same_recipe_static_profile_selection") != expected_profile_selection
    ):
        raise ValueError("schema-v7 per-recipe selection does not recompute exactly")
    expected_terminal = _terminal_production_selection(
        candidates, objectives, workloads, expected_profile_selection
    )
    if value.get("terminal_production_selection") != expected_terminal:
        raise ValueError("schema-v7 terminal production selection does not recompute exactly")
    winner = expected_terminal["winner"]
    selected = next(row for row in candidates if row["name"] == winner)
    if winner not in actual_frontier:
        raise ValueError("schema-v7 terminal winner is not on the global frontier")
    input_path = Path(input_authority["path"])
    if not input_path.is_file() or _file_sha256(input_path) != input_authority["sha256"]:
        raise ValueError("schema-v7 Pareto input authority bytes changed")
    rebuilt = classify(load_payload(input_path.read_text(encoding="utf-8")))
    rebuilt["pareto_input"] = input_authority
    if rebuilt != value:
        raise ValueError("schema-v7 authority differs from its bound Pareto input")
    return expected_terminal, selected


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def load_payload(text: str) -> dict:
    source = json.loads(text, object_pairs_hook=_unique_json_object)
    if not isinstance(source, dict):
        raise ValueError("Pareto input root must be an object")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        input_path = args.input.resolve()
        source = load_payload(input_path.read_text(encoding="utf-8"))
        result = classify(source)
        result["pareto_input"] = {
            "path": str(input_path), "sha256": _file_sha256(input_path),
        }
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
