#!/usr/bin/env python3
"""Project the P2048 post-hybrid bottleneck from retained measurements.

This is deliberately an attribution bridge, not a benchmark substitute.  It combines the
current all-Q4 selected-region trace with complete-op Q4/FP8 measurements for the four selected
roles and keeps cross-run estimates visibly separate from measured whole-inference deltas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROLE_SPECS = (
    ("mlp_gate_up", 64, "post_mixer"),
    ("attention_query_key_and_gate_value", 32, "attention"),
    ("gdn_query_key", 48, "gdn"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_test_seconds(report: dict[str, Any], label: str) -> float:
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        raise ValueError(f"{label} must contain exactly one benchmark test")
    test = tests[0]
    if (test.get("kind"), test.get("n_prompt"), test.get("n_gen")) != ("pp", 2048, 0):
        raise ValueError(f"{label} is not the required P2048 prefill-only workload")
    return float(test["prefill_seconds_mean"])


def _operator(trace: dict[str, Any], stage: str, family: str) -> dict[str, Any]:
    matches = [
        item
        for item in trace["operator_attribution"]
        if item["execution_category"] == "base_text_prefill"
        and item["stage"] == stage
        and item["operator_family"] == family
    ]
    if len(matches) != 1:
        raise ValueError(f"trace must contain exactly one {stage}/{family} attribution")
    return matches[0]


def project(
    trace_path: Path,
    baseline_path: Path,
    hybrid_path: Path,
    role_paths: list[Path],
    floor_tok_s: float,
) -> dict[str, Any]:
    if len(role_paths) != len(ROLE_SPECS):
        raise ValueError("exactly three role reports are required")
    trace = _load(trace_path)
    baseline = _load(baseline_path)
    hybrid = _load(hybrid_path)
    roles = [_load(path) for path in role_paths]

    workload = trace["workload"]
    expected_workload = {
        "kind": "pp",
        "prompt_tokens": 2048,
        "generated_tokens": 0,
        "concurrency": 1,
        "prefill_chunk": 4096,
    }
    if workload != expected_workload:
        raise ValueError("trace workload differs from exact C1/P2048/chunk4096 contract")
    baseline_ms = 1000.0 * _one_test_seconds(baseline, "baseline")
    hybrid_ms = 1000.0 * _one_test_seconds(hybrid, "hybrid")
    target_ms = 1000.0 * 2048.0 / floor_tok_s

    role_projection = []
    for spec, report in zip(ROLE_SPECS, roles, strict=True):
        name, calls, stage = spec
        shape = report.get("shape", {})
        if report.get("pass") is not True or shape.get("tokens") != 2048:
            raise ValueError(f"{name} report did not pass at T=2048")
        q4_each = float(report["q4_complete_median_ms"])
        fp8_each = float(report["fp8_complete_median_ms"])
        role_projection.append(
            {
                "role": name,
                "stage": stage,
                "semantic_calls": calls,
                "shape": shape,
                "q4_complete_median_ms_per_call": q4_each,
                "fp8_complete_median_ms_per_call": fp8_each,
                "q4_projected_ms": calls * q4_each,
                "fp8_projected_ms": calls * fp8_each,
                "projected_saving_ms": calls * (q4_each - fp8_each),
            }
        )

    selected_q4_ms = sum(item["q4_projected_ms"] for item in role_projection)
    selected_fp8_ms = sum(item["fp8_projected_ms"] for item in role_projection)
    projected_role_saving_ms = selected_q4_ms - selected_fp8_ms

    stage_q4 = {
        stage: _operator(trace, stage, "a8q4_prefill_cta")
        for stage in ("post_mixer", "attention", "gdn")
    }
    remaining_q4 = [
        {
            "role": "mlp_down",
            "stage": "post_mixer",
            "semantic_calls": 64,
            "cross_run_projected_ms": float(stage_q4["post_mixer"]["summed_duration_ms"])
            - role_projection[0]["q4_projected_ms"],
        },
        {
            "role": "attention_output",
            "stage": "attention",
            "semantic_calls": 16,
            "cross_run_projected_ms": float(stage_q4["attention"]["summed_duration_ms"])
            - role_projection[1]["q4_projected_ms"],
        },
        {
            "role": "gdn_value_z_and_output",
            "stage": "gdn",
            "semantic_calls": 96,
            "cross_run_projected_ms": float(stage_q4["gdn"]["summed_duration_ms"])
            - role_projection[2]["q4_projected_ms"],
        },
    ]
    if any(item["cross_run_projected_ms"] < 0.0 for item in remaining_q4):
        raise ValueError("standalone selected-role estimate exceeds its traced stage total")

    dense = _operator(trace, "attention", "other")
    recurrence = _operator(trace, "gdn", "gdn_recurrence")
    dense_plus_recurrence_ms = float(dense["summed_duration_ms"]) + float(
        recurrence["summed_duration_ms"]
    )
    gap_ms = hybrid_ms - target_ms
    whole_observed_saving_ms = baseline_ms - hybrid_ms
    model_residual_ms = whole_observed_saving_ms - projected_role_saving_ms

    all_q4_ms = sum(float(item["summed_duration_ms"]) for item in stage_q4.values())
    remaining_q4_ms = sum(item["cross_run_projected_ms"] for item in remaining_q4)
    base_text = next(
        item
        for item in trace["kernel_execution_categories"]
        if item["category"] == "base_text_prefill"
    )
    other_service_ms = (
        float(base_text["independent_summed_duration_ms"])
        - all_q4_ms
        - float(dense["summed_duration_ms"])
        - float(recurrence["summed_duration_ms"])
    )

    input_paths = [trace_path, baseline_path, hybrid_path, *role_paths]
    return {
        "schema": "ninfer.r9700.post_hybrid_prefill_projection.v1",
        "scope": "bounded_cross_run_projection_not_a_measurement",
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in input_paths
        ],
        "workload": expected_workload,
        "floor": {
            "prefill_tok_s": floor_tok_s,
            "target_wall_ms": target_ms,
            "measured_pre_relocation_hybrid_wall_ms": hybrid_ms,
            "remaining_wall_gap_ms": gap_ms,
        },
        "whole_observation": {
            "all_q4_wall_ms": baseline_ms,
            "pre_relocation_hybrid_wall_ms": hybrid_ms,
            "observed_saving_ms": whole_observed_saving_ms,
            "standalone_role_projected_saving_ms": projected_role_saving_ms,
            "unassigned_cross_run_delta_ms": model_residual_ms,
        },
        "selected_fp8_roles": {
            "semantic_calls": 144,
            "roles": role_projection,
            "projected_q4_ms": selected_q4_ms,
            "projected_fp8_ms": selected_fp8_ms,
            "projected_saving_ms": projected_role_saving_ms,
            "disposition": "already selected; remeasure in one post-relocation trace, do not resweep",
        },
        "remaining_q4_roles": {
            "semantic_calls": 176,
            "trace_stage_q4_ms": all_q4_ms,
            "cross_run_projected_ms": remaining_q4_ms,
            "roles": remaining_q4,
        },
        "dense_attention": {
            "current_trace_kernel_calls": int(dense["calls"]),
            "semantic_layer_calls": 16,
            "summed_duration_ms": float(dense["summed_duration_ms"]),
            "route": "dense production",
        },
        "xattention": {
            "current_trace_calls": 0,
            "projected_p2048_ms": None,
            "disposition": "excluded: compile-isolated and no retained matched P2048 timing",
        },
        "gdn_recurrence": {
            "semantic_calls": int(recurrence["calls"]),
            "summed_duration_ms": float(recurrence["summed_duration_ms"]),
        },
        "other_base_text_kernel_service": {
            "summed_duration_ms": other_service_ms,
            "note": "remainder after traced Q4, dense-attention, and recurrence buckets",
        },
        "minimum_unselected_two_family_bound": {
            "families": ["dense_attention", "gdn_recurrence"],
            "semantic_calls": 16 + int(recurrence["calls"]),
            "removable_upper_bound_ms": dense_plus_recurrence_ms,
            "fraction_that_must_be_removed": gap_ms / dense_plus_recurrence_ms,
            "reason": "neither family alone reaches the remaining wall gap; together they do",
        },
        "one_fresh_profile_decision": {
            "required": "one post-relocation selected-region C1/P2048/chunk4096 dense trace",
            "exact_buckets": [
                "144 selected FP8 complete calls split by the three measured role shapes",
                "176 remaining Q4 calls split as MLP-down 64, attention-output 16, GDN value-z/output 96",
                "dense full-score attention: 16 semantic layer calls",
                "GDN recurrence: 48 semantic calls",
                "other kernel service and kernel-inactive wall",
            ],
            "decision_rule": (
                "Use measured post-relocation times to rank these buckets. Do not open an XAttention "
                "or operator sweep from this projection; XAttention requires a matched P2048 whole candidate."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--gate-up", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--gdn", type=Path, required=True)
    parser.add_argument("--floor-tok-s", type=float, default=2000.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = project(
        args.trace,
        args.baseline,
        args.hybrid,
        [args.gate_up, args.attention, args.gdn],
        args.floor_tok_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
