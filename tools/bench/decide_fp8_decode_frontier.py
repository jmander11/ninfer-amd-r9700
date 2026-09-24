#!/usr/bin/env python3
"""Gate the selective FP8 frontier on exact product-reachable decode schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "ninfer.r9700.fp8-selective-decode-decision.v1"
ATTRIBUTION_SCHEMA = "ninfer.r9700.fp8-selective-q4-decode-attribution.v1"
INDEX_SCHEMA = "ninfer.r9700.fp8-selective-decode-qualifier-index.v1"
QUALIFIER_SCHEMA = "ninfer.r9700.fp8_projection_decode_qualification.v1"
ROLES = {
    "text.mlp.gate_up": (34816, 5120, 64),
    "text.attention.query_key": (7168, 5120, 16),
    "text.attention.gate_value": (7168, 5120, 16),
    "text.gdn.query_key": (4096, 5120, 48),
}
SHAPES = {
    "gate_up": (34816, 5120),
    "attention_qk_gate_value": (7168, 5120),
    "gdn_query_key": (4096, 5120),
}
ROLE_SHAPE = {
    "text.mlp.gate_up": "gate_up",
    "text.attention.query_key": "attention_qk_gate_value",
    "text.attention.gate_value": "attention_qk_gate_value",
    "text.gdn.query_key": "gdn_query_key",
}
TOKEN_EXTENTS = (1, 2, 3, 4, 8, 12, 16)
MATERIAL_ROUND_REGRESSION_FRACTION = 0.01


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_bound(path: Path, expected: str) -> dict[str, object]:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{path}: malformed SHA-256")
    if sha256_file(path) != expected:
        raise ValueError(f"{path}: SHA-256 differs")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{path}: root is not an object")
    return result


def expected_schedules() -> list[dict[str, object]]:
    rows = []
    for batch in range(1, 5):
        rows.append({"schedule_id": f"ordinary_c{batch}", "mode": "ordinary",
                     "batch": batch, "width": 1, "tokens": batch})
    for batch in range(1, 5):
        rows.append({"schedule_id": f"mtp3_verify_c{batch}", "mode": "mtp3_verify",
                     "batch": batch, "width": 4, "tokens": 4 * batch})
    return rows


def expected_qualifications() -> set[tuple[str, int]]:
    return {(shape, tokens) for shape in SHAPES for tokens in TOKEN_EXTENTS}


def _live_provenance(report: Mapping[str, object], label: str) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label}: provenance absent")
    for stem in ("source", "executable"):
        path = provenance.get(f"{stem}_path")
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError(f"{label}: {stem} path absent")
        if provenance.get(f"{stem}_sha256") != sha256_file(Path(path)):
            raise ValueError(f"{label}: live {stem} hash differs")


def _median14(values: object, label: str) -> float:
    if (not isinstance(values, list) or len(values) != 14 or
            any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values)):
        raise ValueError(f"{label}: expected 14 positive finite samples")
    ordered = sorted(float(x) for x in values)
    return 0.5 * (ordered[6] + ordered[7])


def validate_qualifier(report: Mapping[str, object], shape_id: str, tokens: int) -> dict[str, float]:
    rows, columns = SHAPES[shape_id]
    qualification_id = f"{shape_id}_t{tokens}"
    if (report.get("schema") != QUALIFIER_SCHEMA or report.get("schema_version") != 1 or
            report.get("artifact_type") != "ninfer_r9700_fp8_projection_decode_comparison" or
            report.get("qualification_id") != qualification_id or report.get("pass") is not True or
            report.get("shape") != {"tokens": tokens, "rows": rows, "columns": columns} or
            report.get("fp8_profile") != "E4M3-outer-vec32f-hipBLASLt-top-supported" or
            report.get("q4_control") != "A8Q4G64-decode-production"):
        raise ValueError(f"{qualification_id}: schema or identity differs")
    _live_provenance(report, qualification_id)
    hardware = report.get("hardware")
    power = report.get("power_profile")
    if (not isinstance(hardware, Mapping) or hardware.get("device") != "AMD Radeon AI PRO R9700" or
            hardware.get("architecture") != "gfx1201" or hardware.get("wave_size") != 32 or
            hardware.get("vendor_id") != "0x1002" or hardware.get("device_id") != "0x7551"):
        raise ValueError(f"{qualification_id}: hardware differs")
    if not isinstance(power, Mapping) or any(power.get(x) != "auto" for x in ("required", "before", "after")):
        raise ValueError(f"{qualification_id}: auto power identity differs")
    if any(report.get(field) is not True for field in (
        "direct_weight_binding", "outer_vector_scales", "nonfinite_status_poisoning",
        "q4_nonfinite_status_poisoning", "no_clobber_rejection",
    )):
        raise ValueError(f"{qualification_id}: correctness gate failed")
    token_probes = sorted({0, tokens // 2, tokens - 1})
    oracle = report.get("represented_format_axis_oracle")
    if (not isinstance(oracle, Mapping) or oracle.get("probe_count") != len(token_probes) * 3 or
            oracle.get("tokens") != token_probes or oracle.get("rows") != [0, rows // 2, rows - 1] or
            type(oracle.get("fp8_max_bf16_steps")) is not int or
            not 0 <= oracle["fp8_max_bf16_steps"] <= 1 or
            type(oracle.get("q4_max_bf16_steps")) is not int or
            not 0 <= oracle["q4_max_bf16_steps"] <= 2):
        raise ValueError(f"{qualification_id}: axis oracle differs")
    fp8 = _median14(report.get("fp8_complete_ms_samples"), qualification_id)
    q4 = _median14(report.get("q4_complete_ms_samples"), qualification_id)
    speedup = q4 / fp8
    if (report.get("balanced_interleaved_pairs") != 7 or
            report.get("trial_order") != "7 paired repetitions, each fp8,q4 then q4,fp8" or
            type(report.get("fp8_complete_median_ms")) not in (int, float) or
            type(report.get("q4_complete_median_ms")) not in (int, float) or
            type(report.get("fp8_over_q4_speedup")) not in (int, float) or
            not math.isclose(float(report["fp8_complete_median_ms"]), fp8, rel_tol=1e-6) or
            not math.isclose(float(report["q4_complete_median_ms"]), q4, rel_tol=1e-6) or
            not math.isclose(float(report["fp8_over_q4_speedup"]), speedup, rel_tol=1e-6)):
        raise ValueError(f"{qualification_id}: balanced timing differs")
    algorithm = report.get("algorithm")
    workspace = report.get("selected_workspace_bytes")
    if (not isinstance(algorithm, Mapping) or type(workspace) is not int or workspace < 0 or
            algorithm.get("selected_workspace_bytes") != workspace or
            workspace > int(algorithm.get("algorithm_max_workspace_bytes", -1)) or
            int(algorithm.get("heuristic_returned", 0)) <= 0):
        raise ValueError(f"{qualification_id}: algorithm workspace differs")
    return {"fp8_complete_median_ms": fp8, "q4_complete_median_ms": q4,
            "fp8_over_q4_time_ratio": fp8 / q4, "fp8_over_q4_speedup": speedup}


def validate_attribution(report: Mapping[str, object]) -> list[dict[str, object]]:
    if (report.get("schema") != ATTRIBUTION_SCHEMA or report.get("schema_version") != 1 or
            report.get("status") != "complete" or
            report.get("product") != {"maximum_concurrency": 4, "draft_tokens": 3,
                                       "device_graph": True} or
            report.get("mtp_draft_selected_object_calls_per_round") != 0):
        raise ValueError("decode attribution identity differs")
    _live_provenance(report, "decode attribution")
    schedules = report.get("schedules")
    if not isinstance(schedules, list) or len(schedules) != 8:
        raise ValueError("decode attribution lacks eight schedules")
    expected = {row["schedule_id"]: row for row in expected_schedules()}
    observed = {row.get("schedule_id"): row for row in schedules if isinstance(row, Mapping)}
    if set(observed) != set(expected):
        raise ValueError("decode attribution schedule set differs")
    result = []
    for schedule_id, identity in expected.items():
        row = observed[schedule_id]
        if any(row.get(field) != identity[field] for field in ("mode", "batch", "width", "tokens")):
            raise ValueError(f"{schedule_id}: schedule geometry differs")
        calls = row.get("selected_role_call_counts")
        services = row.get("selected_role_q4_service_ns")
        if calls != {role: count for role, (*_, count) in ROLES.items()} or not isinstance(services, Mapping):
            raise ValueError(f"{schedule_id}: selected role call counts differ")
        if set(services) != set(ROLES) or any(type(value) is not int or value <= 0 for value in services.values()):
            raise ValueError(f"{schedule_id}: Q4 service map differs")
        selected = sum(int(value) for value in services.values())
        whole = row.get("whole_round_ns")
        if row.get("selected_q4_service_ns") != selected or type(whole) is not int or whole < selected:
            raise ValueError(f"{schedule_id}: Q4 or whole-round service sum differs")
        result.append(dict(row))
    return result


def _validate_control(report: Mapping[str, object], speculative: bool) -> dict[str, float]:
    config = report.get("config")
    tests = report.get("tests")
    if (report.get("schema_version") != 20 or not isinstance(config, Mapping) or
            config.get("concurrency") != 1 or config.get("use_device_graph") is not True or
            not isinstance(tests, list) or not tests):
        raise ValueError("decode whole control identity differs")
    row = next((x for x in tests if x.get("n_prompt") == 8192 and x.get("n_gen") == 256), None)
    if not isinstance(row, Mapping):
        raise ValueError("decode whole control lacks P8192/G256")
    if speculative:
        if config.get("spec") != "mtp" or config.get("draft_tokens") != 3 or any(
            rep.get("speculative", {}).get("rounds") != 64 for rep in row.get("reps", [])
        ):
            raise ValueError("MTP3 control does not contain exact 64-round repetitions")
        rounds = 64
    else:
        if config.get("spec") != "none" or config.get("draft_tokens") != 0:
            raise ValueError("ordinary control is speculative")
        rounds = 256
    return {"decode_seconds": float(row["decode_seconds_mean"]),
            "decode_tok_s": float(row["decode_output_tok_s_mean"]), "rounds": rounds}


def decide(*, prefill: Mapping[str, object], quality: Mapping[str, object],
           mtp_control: Mapping[str, object], ordinary_control: Mapping[str, object],
           attribution: Mapping[str, object], qualifiers: Mapping[tuple[str, int], Mapping[str, object]],
           provenance: Mapping[str, object]) -> dict[str, object]:
    if (prefill.get("schema") != "ninfer.r9700.fp8-post-gate-up-decision.v1" or
            prefill.get("decision", {}).get("verdict") != "proceed" or
            prefill.get("capacity", {}).get("all_cells_preserve_minimum_startup") is not True):
        raise ValueError("prefill/capacity gate is not admitted")
    if (quality.get("schema") != "ninfer.qwen3_8_27b.selective-e4m3-vs-q4g64-sampled-quality.v1" or
            quality.get("selection") != {"roles": sorted(ROLES), "object_count": 144} or
            quality.get("aggregate", {}).get("nonfinite_values") != 0 or
            quality.get("comparative_object_counts", {}).get("relative_l2", {}).get("e4m3_worse") != 0 or
            quality.get("comparative_object_counts", {}).get("max_abs", {}).get("e4m3_worse") != 0):
        raise ValueError("selective source-quality gate differs")
    schedules = validate_attribution(attribution)
    timings = {key: validate_qualifier(qualifiers[key], *key) for key in expected_qualifications()}
    reasons = []
    rows = []
    for row in schedules:
        tokens = int(row["tokens"])
        projected = 0.0
        for role, service in row["selected_role_q4_service_ns"].items():
            ratio = timings[(ROLE_SHAPE[role], tokens)]["fp8_over_q4_time_ratio"]
            projected += int(service) * ratio
        q4 = int(row["selected_q4_service_ns"])
        whole = int(row["whole_round_ns"])
        projected_whole = whole - q4 + projected
        regression = (projected_whole - whole) / whole
        if projected > q4:
            reasons.append(f"{row['schedule_id']}_selected_service_regresses")
        if regression > MATERIAL_ROUND_REGRESSION_FRACTION:
            reasons.append(f"{row['schedule_id']}_whole_round_regresses_over_1pct")
        rows.append({**row, "projected_selected_fp8_service_ns": projected,
                     "projected_whole_round_ns": projected_whole,
                     "projected_whole_round_change_fraction": regression})
    ordinary = _validate_control(ordinary_control, False)
    mtp = _validate_control(mtp_control, True)
    ordinary_delta = next(x for x in rows if x["schedule_id"] == "ordinary_c1")["projected_whole_round_ns"] - next(x for x in rows if x["schedule_id"] == "ordinary_c1")["whole_round_ns"]
    mtp_delta = next(x for x in rows if x["schedule_id"] == "mtp3_verify_c1")["projected_whole_round_ns"] - next(x for x in rows if x["schedule_id"] == "mtp3_verify_c1")["whole_round_ns"]
    projected_ordinary_s = ordinary["decode_seconds"] + ordinary["rounds"] * ordinary_delta / 1e9
    projected_mtp_s = mtp["decode_seconds"] + mtp["rounds"] * mtp_delta / 1e9
    if projected_ordinary_s > ordinary["decode_seconds"]:
        reasons.append("ordinary_c1_whole_decode_regresses")
    if projected_mtp_s > mtp["decode_seconds"]:
        reasons.append("mtp3_c1_whole_decode_regresses")
    return {
        "schema": SCHEMA, "status": "complete",
        "coverage": {"maximum_concurrency": 4, "ordinary_token_extents": [1, 2, 3, 4],
                     "mtp3_verify_token_extents": [4, 8, 12, 16],
                     "mtp3_draft_selected_object_calls_per_round": 0,
                     "unique_qualifier_token_extents": list(TOKEN_EXTENTS),
                     "fixed_roles": {role: {"rows": rows_, "columns": columns, "calls_per_round": calls}
                                     for role, (rows_, columns, calls) in ROLES.items()}},
        "qualifications": {f"{shape}_t{tokens}": timings[(shape, tokens)]
                           for shape, tokens in sorted(timings)},
        "schedules": rows,
        "whole_decode_controls": {
            "ordinary_c1": {**ordinary, "projected_decode_seconds": projected_ordinary_s,
                            "projected_decode_tok_s": 256 / projected_ordinary_s},
            "mtp3_c1": {**mtp, "projected_decode_seconds": projected_mtp_s,
                        "projected_decode_tok_s": 256 / projected_mtp_s},
        },
        "retained_gates": {"prefill_capacity": True, "source_quality": True},
        "decision": {"verdict": "proceed" if not reasons else "reject",
                     "gate": "no selected-service regression at any reachable schedule, no >1% whole-round regression, and no C1 whole-decode regression",
                     "reasons": sorted(set(reasons))},
        "provenance": dict(provenance),
    }


def canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill-decision", "quality", "mtp-control", "ordinary-control", "attribution", "qualifier-index"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ("prefill_decision", "quality", "mtp_control", "ordinary_control", "attribution", "qualifier_index")
    paths = {name: getattr(args, name) for name in names}
    expected = {name: getattr(args, f"{name}_sha256") for name in names}
    loaded = {name: load_bound(path, expected[name]) for name, path in paths.items()}
    index = loaded["qualifier_index"]
    if index.get("schema") != INDEX_SCHEMA or index.get("schema_version") != 1:
        raise ValueError("qualifier index schema differs")
    entries = index.get("qualifiers")
    if not isinstance(entries, list):
        raise ValueError("qualifier index lacks entries")
    qualifiers = {}
    qualifier_provenance = {}
    for entry in entries:
        key = (entry.get("shape_id"), entry.get("tokens"))
        if key in qualifiers or key not in expected_qualifications():
            raise ValueError("qualifier index contains duplicate or unknown extent")
        path = Path(entry["path"])
        qualifiers[key] = load_bound(path, entry["sha256"])
        qualifier_provenance[f"{key[0]}_t{key[1]}"] = {"path": str(path), "sha256": entry["sha256"]}
    if set(qualifiers) != expected_qualifications():
        raise ValueError("qualifier index does not contain the exact 21 reports")
    provenance = {name: {"path": str(path), "sha256": expected[name]} for name, path in paths.items()}
    provenance["qualifiers"] = qualifier_provenance
    report = decide(prefill=loaded["prefill_decision"], quality=loaded["quality"],
                    mtp_control=loaded["mtp_control"], ordinary_control=loaded["ordinary_control"],
                    attribution=loaded["attribution"], qualifiers=qualifiers, provenance=provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_json(report))
    print(json.dumps(report["decision"], sort_keys=True))
    return 0 if report["decision"]["verdict"] == "proceed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
