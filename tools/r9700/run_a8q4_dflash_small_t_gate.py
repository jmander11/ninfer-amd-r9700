#!/usr/bin/env python3
"""Extend the retained C1 DFlash winner across exact flattened C2..4 widths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

try:
    from .check_a8q4_dflash_small_t_static import check as check_static
except ImportError:
    from check_a8q4_dflash_small_t_static import check as check_static

SHAPES = ((34816, 5120),)
TOKENS = (8, 10, 12, 15, 16, 18, 20, 24)
WIDTH_ROLES = {
    8: ("K4_C2_proposal",),
    10: ("K5_C2_proposal", "W5_C2_verification"),
    12: ("K4_C3_proposal", "W6_C2_verification"),
    15: ("K5_C3_proposal", "W5_C3_verification"),
    16: ("K4_C4_proposal",),
    18: ("W6_C3_verification",),
    20: ("K5_C4_proposal", "W5_C4_verification"),
    24: ("W6_C4_verification",),
}
PRIOR_ELIGIBLE = (
    (34816, 5120, 4), (34816, 5120, 5), (34816, 5120, 6),
)
PRIOR_SUMMARY_SHA256 = "27cba6590e4272908b30e5a1ad6073610288c6e26b27100f5ad662d25648b122"
CELL_SCHEMA = "ninfer.r9700.a8q4-dflash-small-t-cell.v3"
GATE_SCHEMA = "ninfer.r9700.a8q4-dflash-small-t-gate.v3"
SCOPE = "N34816/K5120 flattened K4/W5 and K5/W6 widths through C4 only"
ORACLE = "independent FP64 represented A8G64 x Q4G64 formula"
CRITERION = (
    "both launch-order medians faster, two-standard-error paired ratio upper below one, "
    "order ratio delta at most 0.02"
)
SOURCE_PATHS = {
    "kernel": Path("src/ops/r9700/linear/r9700_linear.hip"),
    "contract": Path("src/ops/r9700/linear/r9700_linear.h"),
    "qualifier": Path("tools/r9700/a8q4_dflash_small_t_qual.hip"),
}
PACKAGE_PATHS = (
    *SOURCE_PATHS.values(),
    Path("tools/r9700/run_a8q4_dflash_small_t_gate.py"),
    Path("tools/r9700/check_a8q4_dflash_small_t_static.py"),
    Path("tools/r9700/Makefile"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"small-T cell has nonnumeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"small-T cell has nonfinite {name}")
    return result


def _close(actual: object, expected: float, name: str) -> None:
    value = _finite_number(actual, name)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"small-T cell has inconsistent {name}")


def _samples(timing: dict, name: str) -> list[float]:
    values = timing.get(name)
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError(f"small-T cell has invalid {name}")
    result = [_finite_number(value, name) for value in values]
    if any(value <= 0.0 for value in result):
        raise ValueError(f"small-T cell has nonpositive {name}")
    return result


def validate_cell(
    report: object,
    rows: int,
    columns: int,
    tokens: int,
    expected_executable: dict[str, str],
    expected_sources: dict[str, str],
) -> dict:
    """Recompute the evidence and return the independently eligible cell."""
    if not isinstance(report, dict) or report.get("schema") != CELL_SCHEMA:
        raise ValueError("small-T cell has the wrong schema")
    if report.get("shape") != {"rows": rows, "columns": columns, "tokens": tokens}:
        raise ValueError("small-T cell has the wrong shape")
    if report.get("scope") != SCOPE or report.get("production_dispatch_changed") is not False:
        raise ValueError("small-T cell has the wrong scope")
    hardware = report.get("hardware")
    if not isinstance(hardware, dict) or set(hardware) != {
        "device", "architecture", "pci_bus_id", "pci_vendor_device", "integrated",
        "wavefront_width", "power_profile_before_after",
    }:
        raise ValueError("small-T cell has incomplete hardware identity")
    if (hardware["device"] != "AMD Radeon AI PRO R9700" or
            hardware["architecture"] != "gfx1201" or
            hardware["pci_vendor_device"] != "1002:7551" or
            hardware["integrated"] is not False or hardware["wavefront_width"] != 32 or
            hardware["power_profile_before_after"] != "auto" or
            not isinstance(hardware["pci_bus_id"], str) or
            re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                         hardware["pci_bus_id"]) is None):
        raise ValueError("small-T cell has the wrong hardware identity")
    if report.get("executable") != expected_executable or \
            report.get("sources") != expected_sources:
        raise ValueError("small-T cell is not bound to the preflight identities")

    numeric = report.get("numeric")
    if not isinstance(numeric, dict) or numeric.get("oracle") != ORACLE or \
            numeric.get("maximum_bf16_steps_allowed") != 2:
        raise ValueError("small-T cell lacks the direct oracle contract")
    for route in ("candidate", "incumbent"):
        value = numeric.get(f"{route}_maximum_bf16_steps")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2:
            raise ValueError("small-T cell lacks direct oracle admission")
    if not isinstance(numeric.get("pairwise_bit_exact_diagnostic"), bool):
        raise ValueError("small-T cell lacks the pairwise diagnostic")

    timing = report.get("timing")
    if not isinstance(timing, dict) or timing.get("method") != "unprofiled HIP events" or \
            timing.get("iterations_per_sample") != 10 or \
            timing.get("forward_reverse_pairs") != 7:
        raise ValueError("small-T cell has the wrong timing contract")
    incumbent_forward = _samples(timing, "incumbent_forward_ms")
    candidate_forward = _samples(timing, "candidate_forward_ms")
    candidate_reverse = _samples(timing, "candidate_reverse_ms")
    incumbent_reverse = _samples(timing, "incumbent_reverse_ms")
    incumbent_balanced = [
        (forward + reverse) * 0.5
        for forward, reverse in zip(incumbent_forward, incumbent_reverse)
    ]
    candidate_balanced = [
        (forward + reverse) * 0.5
        for forward, reverse in zip(candidate_forward, candidate_reverse)
    ]
    forward_ratios = [
        candidate / incumbent
        for candidate, incumbent in zip(candidate_forward, incumbent_forward)
    ]
    reverse_ratios = [
        candidate / incumbent
        for candidate, incumbent in zip(candidate_reverse, incumbent_reverse)
    ]
    for name, expected in (
        ("incumbent_balanced_ms", incumbent_balanced),
        ("candidate_balanced_ms", candidate_balanced),
        ("forward_candidate_over_incumbent", forward_ratios),
        ("reverse_candidate_over_incumbent", reverse_ratios),
    ):
        retained = _samples(timing, name)
        for index, (actual, value) in enumerate(zip(retained, expected)):
            _close(actual, value, f"{name}[{index}]")

    incumbent_median = statistics.median(incumbent_balanced)
    candidate_median = statistics.median(candidate_balanced)
    incumbent_forward_median = statistics.median(incumbent_forward)
    candidate_forward_median = statistics.median(candidate_forward)
    incumbent_reverse_median = statistics.median(incumbent_reverse)
    candidate_reverse_median = statistics.median(candidate_reverse)
    forward_ratio_median = statistics.median(forward_ratios)
    reverse_ratio_median = statistics.median(reverse_ratios)
    ratios = forward_ratios + reverse_ratios
    ratio_mean = statistics.mean(ratios)
    ratio_standard_deviation = statistics.stdev(ratios)
    ratio_upper = ratio_mean + 2.0 * ratio_standard_deviation / math.sqrt(len(ratios))
    order_ratio_delta = abs(forward_ratio_median - reverse_ratio_median)
    maximum_order_ratio_delta = 0.02
    accepted = (
        candidate_median < incumbent_median and
        candidate_forward_median < incumbent_forward_median and
        candidate_reverse_median < incumbent_reverse_median and
        forward_ratio_median < 1.0 and reverse_ratio_median < 1.0 and
        ratio_upper < 1.0 and order_ratio_delta <= maximum_order_ratio_delta
    )
    expected_decision = {
        "incumbent_median_ms": incumbent_median,
        "candidate_median_ms": candidate_median,
        "candidate_over_incumbent": candidate_median / incumbent_median,
        "incumbent_forward_median_ms": incumbent_forward_median,
        "candidate_forward_median_ms": candidate_forward_median,
        "incumbent_reverse_median_ms": incumbent_reverse_median,
        "candidate_reverse_median_ms": candidate_reverse_median,
        "forward_ratio_median": forward_ratio_median,
        "reverse_ratio_median": reverse_ratio_median,
        "paired_ratio_mean": ratio_mean,
        "paired_ratio_standard_deviation": ratio_standard_deviation,
        "paired_ratio_upper": ratio_upper,
        "order_ratio_delta": order_ratio_delta,
        "maximum_order_ratio_delta": maximum_order_ratio_delta,
    }
    decision = report.get("decision")
    if not isinstance(decision, dict) or decision.get("criterion") != CRITERION or \
            decision.get("accepted") is not accepted:
        raise ValueError("small-T cell decision is inconsistent")
    for name, expected in expected_decision.items():
        _close(decision.get(name), expected, f"decision.{name}")
    status = report.get("status")
    if status not in ("passed", "rejected") or (status == "passed") is not accepted:
        raise ValueError("small-T cell status is inconsistent")
    return {"report": report, "eligible": accepted}


def _committed_package(repo: Path) -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    hashes: dict[str, str] = {}
    for relative in PACKAGE_PATHS:
        current = (repo / relative).read_bytes()
        retained = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=repo, check=True,
            capture_output=True,
        ).stdout
        if current != retained:
            raise RuntimeError(f"qualification package is not committed: {relative}")
        hashes[relative.as_posix()] = sha256_bytes(current)
    return commit, hashes


def validate_prior_summary(report: object) -> None:
    if not isinstance(report, dict) or \
            report.get("schema") != "ninfer.r9700.a8q4-dflash-small-t-gate.v2" or \
            report.get("status") != "passed" or report.get("complete_screen") is not True or \
            report.get("routing_authorized") is not False or \
            report.get("required_cell_count") != 39:
        raise ValueError("retained C1 screen is not the closed schema-v2 authority")
    eligible = report.get("eligible_cells")
    expected = [
        {"rows": rows, "columns": columns, "tokens": tokens}
        for rows, columns, tokens in PRIOR_ELIGIBLE
    ]
    if eligible != expected or not isinstance(report.get("forbidden_cells"), list) or \
            len(report["forbidden_cells"]) != 36 or \
            not isinstance(report.get("cells"), list) or len(report["cells"]) != 39 or \
            not all(isinstance(cell, dict) and cell.get("evidence_valid") is True
                    for cell in report["cells"]):
        raise ValueError("retained C1 screen has the wrong exact eligibility set")


def _write_exclusive(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=Path("tools/r9700/build/a8q4_dflash_small_t_qual"))
    parser.add_argument("--assembly", type=Path,
                        default=Path("tools/r9700/build/q4g64_linear.s"))
    parser.add_argument("--c1-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve(strict=True)
    assembly = args.assembly.resolve(strict=True)
    prior_summary = args.c1_summary.resolve(strict=True)
    if sha256(prior_summary) != PRIOR_SUMMARY_SHA256:
        raise RuntimeError("retained C1 summary has the wrong SHA-256")
    validate_prior_summary(json.loads(prior_summary.read_text()))
    commit, package_hashes = _committed_package(repo)
    source_hashes = {name: package_hashes[path.as_posix()]
                     for name, path in SOURCE_PATHS.items()}
    executable_identity = {"path": str(binary), "sha256": sha256(binary)}
    static_result = check_static(assembly.read_text())
    preflight = {
        "schema": "ninfer.r9700.a8q4-dflash-small-t-preflight.v1",
        "git_commit": commit,
        "package_sha256": package_hashes,
        "executable": executable_identity,
        "assembly": {"path": str(assembly), "sha256": sha256(assembly)},
        "static": {str(tokens): values for tokens, values in static_result.items()},
        "retained_c1_screen": {
            "path": str(prior_summary), "sha256": PRIOR_SUMMARY_SHA256,
            "eligible_cells": [
                {"rows": rows, "columns": columns, "tokens": tokens}
                for rows, columns, tokens in PRIOR_ELIGIBLE
            ],
        },
    }
    if args.output_dir.exists():
        raise SystemExit("output directory must be fresh")
    args.output_dir.mkdir(parents=True)
    _write_exclusive(args.output_dir / "preflight.json", preflight)

    cells = []
    for rows, columns in SHAPES:
        for tokens in TOKENS:
            stem = f"n{rows}-k{columns}-t{tokens}"
            name = f"{stem}.json"
            path = args.output_dir / name
            command = [str(binary), str(rows), str(columns), str(tokens),
                       "--out-json", str(path)]
            process = subprocess.run(command, check=False, capture_output=True, text=True)
            process_name = f"{stem}.process.json"
            _write_exclusive(args.output_dir / process_name, {
                "command": command,
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            })
            cell = {
                "rows": rows, "columns": columns, "tokens": tokens,
                "eligible": False, "evidence_valid": False, "report": name,
                "process": process_name, "applicability": list(WIDTH_ROLES[tokens]),
            }
            try:
                if process.returncode not in (0, 1) or not path.is_file():
                    raise ValueError("cell process did not retain a decision")
                def reject_constant(item: str) -> object:
                    raise ValueError(f"nonfinite JSON constant {item}")
                value = json.loads(path.read_text(), parse_constant=reject_constant)
                validated = validate_cell(
                    value, rows, columns, tokens, executable_identity, source_hashes
                )
                if (process.returncode == 0) is not validated["eligible"]:
                    raise ValueError("cell exit status disagrees with recomputed decision")
                cell.update({
                    "eligible": validated["eligible"],
                    "evidence_valid": True,
                    "status": value["status"],
                    "sha256": sha256(path),
                    "candidate_over_incumbent": value["decision"]["candidate_over_incumbent"],
                    "paired_ratio_upper": value["decision"]["paired_ratio_upper"],
                    "order_ratio_delta": value["decision"]["order_ratio_delta"],
                })
            except (OSError, json.JSONDecodeError, ValueError) as error:
                cell["status"] = "invalid"
                cell["error"] = str(error)
                if path.is_file():
                    cell["sha256"] = sha256(path)
            cells.append(cell)

    new_eligible = [
        {"rows": cell["rows"], "columns": cell["columns"], "tokens": cell["tokens"]}
        for cell in cells if cell["eligible"]
    ]
    forbidden = [
        {"rows": cell["rows"], "columns": cell["columns"], "tokens": cell["tokens"]}
        for cell in cells if not cell["eligible"]
    ]
    complete = len(cells) == len(SHAPES) * len(TOKENS) and all(
        cell["evidence_valid"] for cell in cells
    )
    passed = complete
    combined_eligible = [
        {"rows": rows, "columns": columns, "tokens": tokens}
        for rows, columns, tokens in PRIOR_ELIGIBLE
    ] + new_eligible
    summary = {
        "schema": GATE_SCHEMA,
        "status": "passed" if passed else "rejected",
        "scope": SCOPE,
        "production_dispatch_changed": False,
        "preflight": {"report": "preflight.json", "sha256": sha256(args.output_dir / "preflight.json")},
        "required_shapes": [list(shape) for shape in SHAPES],
        "required_tokens": list(TOKENS),
        "width_applicability": {str(tokens): list(WIDTH_ROLES[tokens]) for tokens in TOKENS},
        "required_cell_count": len(SHAPES) * len(TOKENS),
        "complete_screen": complete,
        "retained_c1_eligible_cells": combined_eligible[:len(PRIOR_ELIGIBLE)],
        "new_eligible_cells": new_eligible,
        "eligible_cells": combined_eligible,
        "forbidden_cells": forbidden,
        "routing_authorized": False,
        "routing_requirement": (
            "Only eligible exact cells may enter later whole-DFlash A/B; every forbidden or "
            "unlisted shape/width remains on the incumbent"
        ),
        "cells": cells,
    }
    _write_exclusive(args.output_dir / "summary.json", summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
