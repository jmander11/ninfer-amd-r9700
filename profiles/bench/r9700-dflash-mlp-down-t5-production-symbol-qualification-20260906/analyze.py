#!/usr/bin/env python3
"""Fail-closed validator for the post-integration production-symbol T5 cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-dflash-mlp-down-t5-production-symbol-qualification-20260906"
BINARY = ROOT / "tools/r9700/build/a8q4_dflash_mlp_down_small_t_qual"
BINARY_SHA = "2add29d2c5065ff22215a7ffd76fd80ce65020fe6c99ebd80ba58bfde45d95b6"
SOURCES = {
    "kernel": "d27f969b2e0e760d276798daf51c3cc5acbf9e056ffb92badbbc604ee9b6052d",
    "contract": "d22c780e18964c8206c1718a2478732766814fbf6e5471995fb2086e86c99bd2",
    "qualifier": "75edc21e02d9c87af959a973be7a794dbd8f23f80a6e039bbdcb783dfdf4a3b1",
    "shared_harness": "69af9a07e18bf331a175dfae6c02fb12db71baebad5e63552bf5744116612486",
}
CRITERION = (
    "both launch-order medians faster, two-standard-error paired ratio upper below one, "
    "order ratio delta at most 0.02"
)


def load(path: Path) -> dict:
    def pairs(items: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("report is not a JSON object")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} is not finite numeric")
    return float(value)


def arm(timing: dict, name: str) -> list[float]:
    values = timing.get(name)
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError(f"invalid timing arm: {name}")
    result = [finite(value, name) for value in values]
    if any(value <= 0 for value in result):
        raise ValueError(f"nonpositive timing arm: {name}")
    return result


def close(actual: object, expected: float, label: str) -> None:
    if not math.isclose(finite(actual, label), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"reported {label} differs from raw timing")


def validate(report: dict) -> dict:
    if report.get("schema") != "ninfer.r9700.a8q4-dflash-mlp-down-small-t-cell.v1" or report.get("status") not in ("passed", "rejected"):
        raise ValueError("report schema/status differs")
    if report.get("production_dispatch_changed") is not False or report.get("scope") != "standalone Q4 MLP-down N5120/K17408 at exact DFlash T5 only; no production routing" or report.get("shape") != {"rows": 5120, "columns": 17408, "tokens": 5}:
        raise ValueError("report scope differs")
    hardware = report.get("hardware", {})
    if hardware.get("device") != "AMD Radeon AI PRO R9700" or hardware.get("architecture") != "gfx1201" or hardware.get("pci_vendor_device") != "1002:7551" or hardware.get("integrated") is not False or hardware.get("wavefront_width") != 32 or hardware.get("power_profile_before_after") != "auto" or re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", hardware.get("pci_bus_id", "")) is None:
        raise ValueError("hardware identity differs")
    numeric = report.get("numeric", {})
    if numeric.get("oracle") != "independent FP64 represented A8G64 x Q4G64 formula" or numeric.get("maximum_bf16_steps_allowed") != 2:
        raise ValueError("numeric oracle differs")
    for name in ("candidate_maximum_bf16_steps", "incumbent_maximum_bf16_steps"):
        value = numeric.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2:
            raise ValueError("numeric error bound failed")
    if not isinstance(numeric.get("pairwise_bit_exact_diagnostic"), bool):
        raise ValueError("pairwise diagnostic is not boolean")
    if report.get("executable") != {"path": str(BINARY), "sha256": BINARY_SHA} or report.get("sources") != SOURCES:
        raise ValueError("executable/source identity differs")
    timing = report.get("timing", {})
    if timing.get("method") != "unprofiled HIP events" or timing.get("iterations_per_sample") != 10 or timing.get("forward_reverse_pairs") != 7:
        raise ValueError("timing contract differs")
    incumbent_forward = arm(timing, "incumbent_forward_ms")
    candidate_forward = arm(timing, "candidate_forward_ms")
    candidate_reverse = arm(timing, "candidate_reverse_ms")
    incumbent_reverse = arm(timing, "incumbent_reverse_ms")
    incumbent_balanced = [(a + b) * 0.5 for a, b in zip(incumbent_forward, incumbent_reverse, strict=True)]
    candidate_balanced = [(a + b) * 0.5 for a, b in zip(candidate_forward, candidate_reverse, strict=True)]
    forward_ratios = [a / b for a, b in zip(candidate_forward, incumbent_forward, strict=True)]
    reverse_ratios = [a / b for a, b in zip(candidate_reverse, incumbent_reverse, strict=True)]
    for name, expected in (("incumbent_balanced_ms", incumbent_balanced), ("candidate_balanced_ms", candidate_balanced), ("forward_candidate_over_incumbent", forward_ratios), ("reverse_candidate_over_incumbent", reverse_ratios)):
        published = arm(timing, name)
        for index, (actual, wanted) in enumerate(zip(published, expected, strict=True)):
            close(actual, wanted, f"{name}[{index}]")
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
        "paired_ratio_mean": statistics.mean(ratios),
        "paired_ratio_standard_deviation": statistics.stdev(ratios),
    }
    derived["paired_ratio_upper"] = derived["paired_ratio_mean"] + 2.0 * derived["paired_ratio_standard_deviation"] / math.sqrt(len(ratios))
    derived["order_ratio_delta"] = abs(derived["forward_ratio_median"] - derived["reverse_ratio_median"])
    derived["maximum_order_ratio_delta"] = 0.02
    derived["candidate_over_incumbent"] = derived["candidate_median_ms"] / derived["incumbent_median_ms"]
    decision = report.get("decision", {})
    if decision.get("criterion") != CRITERION:
        raise ValueError("decision criterion differs")
    for name, expected in derived.items():
        close(decision.get(name), expected, name)
    computed = derived["candidate_median_ms"] < derived["incumbent_median_ms"] and derived["candidate_forward_median_ms"] < derived["incumbent_forward_median_ms"] and derived["candidate_reverse_median_ms"] < derived["incumbent_reverse_median_ms"] and derived["forward_ratio_median"] < 1.0 and derived["reverse_ratio_median"] < 1.0 and derived["paired_ratio_upper"] < 1.0 and derived["order_ratio_delta"] <= 0.02
    accepted = decision.get("accepted")
    if not isinstance(accepted, bool) or accepted != computed or report.get("status") != ("passed" if computed else "rejected"):
        raise ValueError("decision/status differs from raw timing")
    return {"accepted": accepted, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.report != PACKAGE / "cell-t5-production.json" or args.summary != PACKAGE / "summary.json" or args.summary.exists():
        raise ValueError("paths are not fresh immutable package outputs")
    result = validate(load(args.report))
    summary = {"artifact_type": "ninfer_r9700_dflash_mlp_down_t5_production_symbol_qualification", "schema_version": 1, "status": "accepted" if result["accepted"] else "rejected", "production_eligible": False, "production_routing_authorized": False, "source_commit": "d1a9b6fb33a843363fba3ad46d6569071df1ab66", "shape": {"tokens": 5, "rows": 5120, "columns": 17408}, "decision": result["decision"], "limitations": ["standalone represented-input qualification only", "does not establish whole-DFlash performance or token parity", "does not authorize production routing"]}
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
