#!/usr/bin/env python3
"""Recompute and fail closed on a retained K5120 rows1..4 RMSNorm report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "ninfer.r9700.rmsnorm-k5120-rows4-qualification.v2"
POWER_PATH = "/sys/class/drm/card2/device/power_dpm_force_performance_level"
UNCERTAINTY = 3.0 * 1.4826
ITERATIONS = 8192
CALLS = 129
ORACLE = "independent FP64 complete represented-BF16 RMSNorm formula; BF16 output boundary"
CRITERION = ("maximum 2 BF16 representable steps; absolute and relative-L2 values are "
             "diagnostics")
STREAM = "one owned nondefault stream; both routes and events ordered on it"
METHOD = "9 alternating paired trials; raw event intervals"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(lhs: float, rhs: float) -> bool:
    return math.isclose(lhs, rhs, rel_tol=1.0e-9, abs_tol=1.0e-12)


def number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label)
    result = float(value)
    require(math.isfinite(result), label)
    return result


def validate(report: dict[str, Any]) -> dict[str, float | bool]:
    require(report.get("schema") == SCHEMA and report.get("status") == "measured",
            "schema/status")
    command = report.get("command", [])
    require(isinstance(command, list) and len(command) == 4 and
            command[1:3] == ["--benchmark", "--out-json"] and
            all(isinstance(item, str) and item for item in command), "command")
    device = report.get("device", {})
    require(device.get("ordinal") == 0 and device.get("architecture") == "gfx1201" and
            device.get("wave_size") == 32 and "R9700" in device.get("name", ""),
            "device identity")
    require(type(device.get("runtime_version")) is int and
            type(device.get("driver_version")) is int, "runtime identity")
    power = report.get("power_profile", {})
    require(power == {"path": POWER_PATH, "required": "auto", "before_device": "auto",
                      "after_numerical": "auto", "after_timing": "auto"},
            "power checkpoints")
    correctness = report.get("correctness", {})
    require(correctness.get("oracle") == ORACLE and correctness.get("criterion") == CRITERION and
            correctness.get("rows") == [1, 2, 3, 4] and
            correctness.get("fixtures") == ["ordinary", "mixed_magnitude", "zero"] and
            correctness.get("unit_offset") == [False, True] and
            correctness.get("eps") == [1e-8, 1e-6, 1e-4] and
            correctness.get("maximum_bf16_steps_allowed") == 2 and
            correctness.get("cases_per_route") == 72 and
            correctness.get("graph_replays") == 2 and
            correctness.get("output_canaries") is True and
            correctness.get("malformed_cases") == 11 and
            correctness.get("malformed_case_names") == [
                "null_input", "null_weight", "null_output", "rows_zero", "rows_five",
                "eps_zero", "eps_negative", "eps_nan", "eps_positive_infinity",
                "eps_negative_infinity", "null_stream"],
            "correctness inventory")
    for route in ("incumbent", "candidate"):
        error = correctness.get(route, {})
        require(type(error.get("maximum_bf16_steps")) is int and
                0 <= error["maximum_bf16_steps"] <= 2, f"{route} BF16 criterion")
        require(number(error.get("maximum_absolute_error"), f"{route} max abs") >= 0.0,
                f"{route} negative max abs")
        require(number(error.get("maximum_relative_l2"), f"{route} relative L2") >= 0.0,
                f"{route} negative relative L2")
    timing = report.get("timing", {})
    require(timing.get("iterations_per_interval") == ITERATIONS and
            timing.get("minimum_interval_ms") == 10 and
            timing.get("uncertainty") == "4.4478*MAD" and
            timing.get("stream") == STREAM and timing.get("method") == METHOD,
            "timing contract")
    rows = report.get("rows", [])
    require(isinstance(rows, list) and [row.get("rows") for row in rows] == [1, 2, 3, 4],
            "row inventory")
    minimum_saving = math.inf
    all_passed = True
    for row in rows:
        require(row.get("ordinary_calls_per_round") == CALLS, "production call weight")
        trials = row.get("trials", [])
        require(isinstance(trials, list) and len(trials) == 9, "raw trial count")
        incumbent: list[float] = []
        candidate: list[float] = []
        for index, trial in enumerate(trials):
            require(trial.get("index") == index and trial.get("order") ==
                    ("incumbent,candidate" if index % 2 == 0 else "candidate,incumbent"),
                    "interleave order")
            incumbent_interval = number(trial.get("incumbent_interval_ms"), "incumbent interval")
            candidate_interval = number(trial.get("candidate_interval_ms"), "candidate interval")
            require(incumbent_interval >= 10.0 and candidate_interval >= 10.0,
                    "short timing interval")
            incumbent.append(incumbent_interval / ITERATIONS)
            candidate.append(candidate_interval / ITERATIONS)
        im, cm = statistics.median(incumbent), statistics.median(candidate)
        imad = statistics.median(abs(value - im) for value in incumbent)
        cmad = statistics.median(abs(value - cm) for value in candidate)
        iu, cu = UNCERTAINTY * imad, UNCERTAINTY * cmad
        require(im > iu, "nonpositive incumbent lower bound")
        ratio = (cm + cu) / (im - iu)
        saving = CALLS * ((im - iu) - (cm + cu))
        passed = ratio <= 1.01 and saving >= 5.0
        expected = {"incumbent_median_ms": im, "candidate_median_ms": cm,
                    "incumbent_mad_ms": imad, "candidate_mad_ms": cmad,
                    "incumbent_uncertainty_ms": iu, "candidate_uncertainty_ms": cu,
                    "robust_upper_ratio": ratio, "robust_round_saving_lower_ms": saving}
        for key, value in expected.items():
            require(close(number(row.get(key), key), value), f"inconsistent {key}")
        require(row.get("passed") is passed, "inconsistent row verdict")
        minimum_saving = min(minimum_saving, saving)
        all_passed = all_passed and passed
    decision = report.get("decision", {})
    require(close(number(decision.get("maximum_robust_ratio"), "ratio limit"), 1.01) and
            close(number(decision.get("minimum_round_saving_lower_ms"), "saving floor"), 5.0) and
            close(number(decision.get("minimum_observed_round_saving_lower_ms"),
                         "minimum saving"), minimum_saving) and
            decision.get("all_rows_passed") is all_passed and
            decision.get("accepted") is all_passed, "decision recomputation")
    require(all_passed, "report decision rejected")
    return {"accepted": all_passed, "minimum_round_saving_lower_ms": minimum_saving}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
        require(isinstance(payload, dict), "report root")
        result = validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
