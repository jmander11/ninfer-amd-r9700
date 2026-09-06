#!/usr/bin/env python3
"""Validate and summarize the one exact standalone MLP-down T5 qualification cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-dflash-mlp-down-t5-qualification-20260906"
EXPECTED_BINARY = ROOT / "tools/r9700/build/a8q4_dflash_mlp_down_small_t_qual"
EXPECTED_BINARY_SHA = "560ad12b7dace9817a6a056a9241fe59dfd599a14a7dc1fdc37eb9ecbfe33f47"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite numeric report value")
    return float(value)


def close(actual: object, expected: float, label: str) -> None:
    if not math.isclose(finite(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"reported {label} differs from raw timing arms")


def arm(timing: dict, name: str) -> list[float]:
    values = timing.get(name)
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError(f"invalid timing arm: {name}")
    result = [finite(value) for value in values]
    if any(value <= 0 for value in result):
        raise ValueError(f"nonpositive timing arm: {name}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.report != PACKAGE / "cell-t5.json" or args.summary != PACKAGE / "summary.json":
        raise ValueError("output paths differ from the immutable package contract")
    if args.summary.exists():
        raise ValueError("summary output already exists")
    report = load(args.report)
    if report.get("schema") != "ninfer.r9700.a8q4-dflash-mlp-down-small-t-cell.v1":
        raise ValueError("cell schema differs")
    if report.get("production_dispatch_changed") is not False or report.get("shape") != {
        "rows": 5120, "columns": 17408, "tokens": 5
    }:
        raise ValueError("cell scope or production isolation differs")
    if report.get("scope") != (
        "standalone Q4 MLP-down N5120/K17408 at exact DFlash T5 only; no production routing"
    ):
        raise ValueError("cell claim scope differs")
    hardware = report.get("hardware", {})
    if hardware.get("device") != "AMD Radeon AI PRO R9700" or hardware.get("architecture") != "gfx1201" or hardware.get("pci_vendor_device") != "1002:7551" or hardware.get("integrated") is not False or hardware.get("wavefront_width") != 32 or hardware.get("power_profile_before_after") != "auto" or re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", hardware.get("pci_bus_id", "")) is None:
        raise ValueError("hardware/power identity differs")
    numeric = report.get("numeric", {})
    candidate_steps = numeric.get("candidate_maximum_bf16_steps")
    incumbent_steps = numeric.get("incumbent_maximum_bf16_steps")
    if numeric.get("oracle") != "independent FP64 represented A8G64 x Q4G64 formula" or numeric.get("maximum_bf16_steps_allowed") != 2 or isinstance(candidate_steps, bool) or not isinstance(candidate_steps, int) or candidate_steps < 0 or candidate_steps > 2 or isinstance(incumbent_steps, bool) or not isinstance(incumbent_steps, int) or incumbent_steps < 0 or incumbent_steps > 2:
        raise ValueError("numerical oracle contract failed")
    timing = report.get("timing", {})
    if timing.get("method") != "unprofiled HIP events" or timing.get("iterations_per_sample") != 10 or timing.get("forward_reverse_pairs") != 7:
        raise ValueError("timing method differs")
    incumbent_forward = arm(timing, "incumbent_forward_ms")
    candidate_forward = arm(timing, "candidate_forward_ms")
    candidate_reverse = arm(timing, "candidate_reverse_ms")
    incumbent_reverse = arm(timing, "incumbent_reverse_ms")
    incumbent_balanced = [(a + b) * 0.5 for a, b in zip(incumbent_forward, incumbent_reverse)]
    candidate_balanced = [(a + b) * 0.5 for a, b in zip(candidate_forward, candidate_reverse)]
    forward_ratios = [a / b for a, b in zip(candidate_forward, incumbent_forward)]
    reverse_ratios = [a / b for a, b in zip(candidate_reverse, incumbent_reverse)]
    for name, expected in (("incumbent_balanced_ms", incumbent_balanced), ("candidate_balanced_ms", candidate_balanced), ("forward_candidate_over_incumbent", forward_ratios), ("reverse_candidate_over_incumbent", reverse_ratios)):
        published = arm(timing, name)
        for index, (actual, wanted) in enumerate(zip(published, expected)):
            close(actual, wanted, f"{name}[{index}]")
    decision = report.get("decision", {})
    if decision.get("criterion") != "both launch-order medians faster, two-standard-error paired ratio upper below one, order ratio delta at most 0.02":
        raise ValueError("decision criterion differs")
    accepted = decision.get("accepted")
    if not isinstance(accepted, bool) or report.get("status") != ("passed" if accepted else "rejected"):
        raise ValueError("decision/status mismatch")
    ratios = forward_ratios + reverse_ratios
    derived = {
        "incumbent_median_ms": statistics.median(incumbent_balanced),
        "candidate_median_ms": statistics.median(candidate_balanced),
        "incumbent_forward_median_ms": statistics.median(incumbent_forward),
        "candidate_forward_median_ms": statistics.median(candidate_forward),
        "incumbent_reverse_median_ms": statistics.median(incumbent_reverse),
        "candidate_reverse_median_ms": statistics.median(candidate_reverse),
        "forward_ratio_median": statistics.median(forward_ratios),
        "reverse_ratio_median": statistics.median(reverse_ratios),
        "paired_ratio_mean": sum(ratios) / len(ratios),
        "paired_ratio_standard_deviation": statistics.stdev(ratios),
    }
    derived["paired_ratio_upper"] = derived["paired_ratio_mean"] + 2.0 * derived["paired_ratio_standard_deviation"] / math.sqrt(len(ratios))
    derived["order_ratio_delta"] = abs(derived["forward_ratio_median"] - derived["reverse_ratio_median"])
    derived["maximum_order_ratio_delta"] = 0.02
    derived["candidate_over_incumbent"] = derived["candidate_median_ms"] / derived["incumbent_median_ms"]
    for name, expected in derived.items():
        close(decision.get(name), expected, name)
    computed = derived["candidate_median_ms"] < derived["incumbent_median_ms"] and derived["candidate_forward_median_ms"] < derived["incumbent_forward_median_ms"] and derived["candidate_reverse_median_ms"] < derived["incumbent_reverse_median_ms"] and derived["forward_ratio_median"] < 1.0 and derived["reverse_ratio_median"] < 1.0 and derived["paired_ratio_upper"] < 1.0 and derived["order_ratio_delta"] <= 0.02
    if accepted != computed:
        raise ValueError("reported decision differs from exact qualification gates")
    executable = report.get("executable", {})
    if executable != {"path": str(EXPECTED_BINARY), "sha256": EXPECTED_BINARY_SHA}:
        raise ValueError("executed binary identity differs")
    expected_sources = {
        "kernel": "692471102a3c31e0a71e923c6fd5044717bef3afa8ef60a7cbcc5a8d22bd0a52",
        "contract": "41bf31711813bfc5919961447dd540ed3f009d99a5c665826b555e995ab03a51",
        "qualifier": "fd2f93b3f0214c50d926e94f5e5e10928d00fc40e24f1b873214cc0ae0d6ec1a",
        "shared_harness": "69af9a07e18bf331a175dfae6c02fb12db71baebad5e63552bf5744116612486",
    }
    if report.get("sources") != expected_sources:
        raise ValueError("runtime source receipts differ")
    if not isinstance(numeric.get("pairwise_bit_exact_diagnostic"), bool):
        raise ValueError("pairwise diagnostic is not boolean")
    summary = {
        "schema": "ninfer.r9700.dflash-mlp-down-t5-qualification-summary.v1",
        "status": "accepted" if accepted else "rejected",
        "production_eligible": False,
        "production_dispatch_changed": False,
        "scope": {"rows": 5120, "columns": 17408, "tokens": 5, "cells": 1},
        "t6": "prior-static-rejection-not-executed; exact retained T5 assembly contains no T6 kernel",
        "decision": decision,
        "limitations": [
            "standalone synthetic represented-input qualification only",
            "does not establish whole-DFlash or production performance",
            "does not authorize production routing",
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
