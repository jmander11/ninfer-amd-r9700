#!/usr/bin/env python3
"""Validate the sole unpooled CU-mask repeat with separate 15%/10% bounds."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "r9700-n16k16-q4-cu-mask-causal-prepare-20260905" / "validate.py"
SPEC = importlib.util.spec_from_file_location("cu_mask_base_validate", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen base CU-mask validator")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

ARM_IDS = BASE.ARM_IDS
FRACTIONS = BASE.FRACTIONS
MASKS = BASE.MASKS
NOMINAL_TOLERANCE = 0.15
CURVE_TOLERANCE = 0.10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_plan(plan: dict[str, Any], *, verify_inputs: bool) -> None:
    BASE.validate_plan(plan, verify_inputs=verify_inputs)
    repeat = plan.get("repeat_contract", {})
    _require(repeat.get("index") == 1 and repeat.get("pooling") == "none" and
             "terminal miss" in repeat.get("terminal", "") and
             "no further tolerance change, pooling, or repeat" in repeat.get("terminal", ""),
             "wrong one-repeat/no-pooling contract")
    _require(plan.get("peak_probe", {}).get("iu4_mask_response_relative_tolerance") ==
             NOMINAL_TOLERANCE, "nominal IU4 guard must be exactly 15%")
    _require(plan.get("classification", {}).get("relative_curve_tolerance") ==
             CURVE_TOLERANCE, "causal curve tolerance must remain exactly 10%")
    peak = plan.get("peak_probe", {})
    _require((peak.get("seconds"), peak.get("trials"), peak.get("code_gib")) ==
             (0.2, 5, 0.25), "peak-control workload changed")


def classify(arms: dict[str, dict[str, float]]) -> dict[str, Any]:
    _require(set(arms) == set(ARM_IDS), "classification requires exactly all three arms")
    full = arms["full64"]
    curves: dict[str, dict[str, float]] = {}
    for arm in ARM_IDS:
        values = arms[arm]
        _require(all(isinstance(values.get(k), (int, float)) and
                     not isinstance(values.get(k), bool) and
                     math.isfinite(values[k]) and values[k] > 0
                     for k in ("iu4", "stream", "q4_ms")),
                 f"{arm} metrics must be positive and finite")
        curves[arm] = {"iu4": values["iu4"] / full["iu4"],
                       "stream": values["stream"] / full["stream"],
                       "production": full["q4_ms"] / values["q4_ms"]}
    for arm in ("half32", "quarter16"):
        _require(BASE._relative_error(curves[arm]["iu4"], FRACTIONS[arm]) <=
                 NOMINAL_TOLERANCE + 1.0e-12,
                 f"{arm} IU4 control does not validate the requested CU mask within 15%")
    compute = all(BASE._relative_error(curves[a]["production"], curves[a]["iu4"])
                  <= CURVE_TOLERANCE + 1.0e-12 for a in ("half32", "quarter16"))
    stream = all(BASE._relative_error(curves[a]["production"], curves[a]["stream"])
                 <= CURVE_TOLERANCE + 1.0e-12 for a in ("half32", "quarter16"))
    if compute and not stream:
        result = "cu_local_service"
    elif stream and not compute:
        result = "shared_memory_service"
    else:
        result = "mixed_or_inconclusive"
    return {"classification": result, "curves_relative_to_full64": curves,
            "nominal_iu4_guard_relative_tolerance": NOMINAL_TOLERANCE,
            "causal_curve_relative_tolerance": CURVE_TOLERANCE,
            "within_ten_percent": {"iu4": compute, "stream": stream},
            "repeat_index": 1, "pooled_with_prior": False,
            "further_repeat_authorized": False}


def validate(plan_path: Path, output: Path) -> dict[str, Any]:
    plan = BASE._load(plan_path)
    validate_plan(plan, verify_inputs=True)
    package = plan_path.resolve().parent
    arm_metrics: dict[str, dict[str, float]] = {}
    evidence: dict[str, Any] = {}
    for arm in ARM_IDS:
        for when in ("before", "after"):
            _require((package / f"power-{arm}-{when}.txt").read_text().strip() == "auto",
                     f"{arm} power {when} is not auto")
        peak_path = package / f"peak-{arm}.json"
        benchmark_path = package / f"benchmark-{arm}.json"
        database_path = package / f"raw-{arm}/cu-mask-{arm}_results.db"
        peak = BASE._validate_peak(BASE._load(peak_path), plan)
        BASE._validate_benchmark(BASE._load(benchmark_path), plan, package, arm)
        trace = BASE._validate_trace(database_path,
                                     BASE._expected_command(plan, package, arm),
                                     MASKS[arm], plan)
        arm_metrics[arm] = {**peak, "q4_ms": trace["summed_duration_ms"]}
        evidence[arm] = {"hsa_cu_mask": MASKS[arm],
                         "active_cus": int(64 * FRACTIONS[arm]),
                         "peak_probe_sha256": BASE._sha256(peak_path),
                         "benchmark_report_sha256": BASE._sha256(benchmark_path),
                         **peak, **trace}
    decision = classify(arm_metrics)
    payload = {"schema": BASE.RESULT_SCHEMA, "status": "valid",
               "decision": decision, "workload": plan["workload"], "arms": evidence,
               "plan": {"path": str(plan_path.resolve()),
                        "sha256": BASE._sha256(plan_path)},
               "repeat_contract": plan["repeat_contract"],
               "interpretation": "Independent terminal CU-mask repeat; no pooled estimate and no further retry."}
    BASE._publish(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _require(args.output is None, "--preflight does not accept --output")
        validate_plan(BASE._load(args.plan), verify_inputs=True)
        print("CU-mask causal repeat1 preflight: PASS")
        return 0
    _require(args.output is not None, "--output is required unless --preflight is used")
    result = validate(args.plan, args.output)
    print(f"CU-mask causal repeat1: {result['decision']['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
