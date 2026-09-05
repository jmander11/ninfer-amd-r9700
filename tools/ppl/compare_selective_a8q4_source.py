#!/usr/bin/env python3
"""Validate the paired activation-inclusive selective G64/G128 8K source gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl.compare_q4_group_source import (
    _against, _load_bf16, _sidecar_values, _source_key,
)
from tools.ppl.q4_group_source_diagnostic import _atomic_new, sha256_file
from tools.ppl.selective_a8q4_source_diagnostic import (
    ARTIFACT_TYPE, SCHEMA_VERSION, TOKENS, _implementation, _profile_contract, _scope,
)


COMPARISON_TYPE = "ninfer_qwen3_8_selective_a8g128_q4g128_source_comparison"
EXPECTED_BUDGET = 11
SOURCE_SCREEN = REPO / "profiles/bench/r9700-selective-q4g128-mse-source-screen-20260905.json"


def _close(actual, expected: float) -> bool:
    return type(actual) in (int, float) and math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _validate_exact_scope(report: dict, path: Path) -> None:
    if report.get("matrix_scope") != _scope():
        raise ValueError(f"{path}: activation-inclusive selected-row scope differs")


def _load_score(path: Path, profile: str, bf16_path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("artifact_type") != ARTIFACT_TYPE
            or report.get("schema_version") != SCHEMA_VERSION
            or report.get("status") != "diagnostic_represented_formula_not_product_ppl"
            or report.get("profile") != profile
            or report.get("formula") != _profile_contract(profile)
            or report.get("implementation_sha256") != _implementation()):
        raise ValueError(f"{path}: activation-inclusive score identity differs")
    expected_workload = {"tokens": TOKENS, "skip": "half", "prefill_chunk": 4096,
                         "schedule": "prefill", "device": 0, "scored_positions": 4095}
    if report.get("workload") != expected_workload:
        raise ValueError(f"{path}: activation-inclusive workload differs")
    _validate_exact_scope(report, path)
    if report.get("bf16_authority") != {
        "path": str(bf16_path.resolve()), "sha256": sha256_file(bf16_path),
    }:
        raise ValueError(f"{path}: BF16 authority binding differs")
    if report.get("sampled_source_gate") != {
        "path": str(SOURCE_SCREEN.resolve()), "sha256": sha256_file(SOURCE_SCREEN),
    }:
        raise ValueError(f"{path}: sampled source gate binding differs")
    nlls = _sidecar_values(path, ".nllf32", "f")
    argmax = _sidecar_values(path, ".argmaxi32", "i")
    if len(nlls) != 4095 or len(argmax) != 4095 or not all(map(math.isfinite, nlls)):
        raise ValueError(f"{path}: activation-inclusive sidecar inventory differs")
    for kind, suffix in (("nll", ".nllf32"), ("argmax", ".argmaxi32")):
        target = path.with_suffix(suffix)
        if report.get("sidecars", {}).get(kind) != {
            "path": target.name, "sha256": sha256_file(target),
        }:
            raise ValueError(f"{path}: {kind} sidecar binding differs")
    result = report.get("result", {})
    exact = {"tokens_scored": 4095, "argmax_tokens": 4095, "non_finite": 0,
             "terrible_tokens": sum(value >= 10.0 for value in nlls)}
    derived = {"sum_nll": sum(nlls), "mean_nll": sum(nlls) / len(nlls),
               "max_nll": max(nlls), "ppl": math.exp(sum(nlls) / len(nlls))}
    if any(result.get(key) != value for key, value in exact.items()) \
            or any(not _close(result.get(key), value) for key, value in derived.items()):
        raise ValueError(f"{path}: result summary differs from sidecars")
    duration = result.get("score_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: score duration is invalid")
    return report, nlls, argmax


def compare(control_path: Path, candidate_path: Path, bf16_path: Path) -> dict[str, object]:
    control, control_nll, control_argmax = _load_score(
        control_path, "a8g64-q4g64-control", bf16_path
    )
    candidate, candidate_nll, candidate_argmax = _load_score(
        candidate_path, "a8g128-q4g128-mse", bf16_path
    )
    bf16, bf16_nll, bf16_argmax = _load_bf16(bf16_path)
    if (_source_key(control) != _source_key(candidate)
            or control["execution"] != candidate["execution"]
            or control["matrix_scope"] != candidate["matrix_scope"]
            or control["sampled_source_gate"] != candidate["sampled_source_gate"]):
        raise ValueError("activation-inclusive paired score identity differs")
    bf16_key = _source_key({"source": {
        "config_sha256": bf16["source_config_sha256"],
        "index_sha256": bf16["source_index_sha256"],
        "shards_sha256": bf16["source_shards_sha256"],
        "corpus_ids_sha256": bf16["corpus_ids_sha256"],
    }})
    if _source_key(control) != bf16_key:
        raise ValueError("activation-inclusive scores and BF16 source/corpus differ")
    control_gate = _against(bf16_nll, control_nll, bf16_argmax, control_argmax)
    candidate_gate = _against(bf16_nll, candidate_nll, bf16_argmax, candidate_argmax)
    paired = _against(control_nll, candidate_nll, control_argmax, candidate_argmax)
    if control.get("against_bf16") != control_gate or candidate.get("against_bf16") != candidate_gate:
        raise ValueError("activation-inclusive retained gate differs from sidecars")
    if (control_gate["new_severe_position_budget"] != EXPECTED_BUDGET
            or candidate_gate["new_severe_position_budget"] != EXPECTED_BUDGET):
        raise ValueError("activation-inclusive severe-position budget differs")
    passed = (control_gate["mean_nll_delta_pass"]
              and control_gate["new_severe_positions_pass"]
              and candidate_gate["mean_nll_delta_pass"]
              and candidate_gate["new_severe_positions_pass"])
    return {
        "artifact_type": COMPARISON_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "activation_inclusive_source_gate_not_product_admission",
        "inputs": {
            "control": {"path": str(control_path.resolve()),
                        "sha256": sha256_file(control_path)},
            "candidate": {"path": str(candidate_path.resolve()),
                          "sha256": sha256_file(candidate_path)},
            "bf16": {"path": str(bf16_path.resolve()), "sha256": sha256_file(bf16_path)},
        },
        "control_against_bf16": control_gate,
        "candidate_against_bf16": candidate_gate,
        "candidate_minus_control": paired,
        "acceptance": {"maximum_mean_nll_delta": candidate_gate["maximum_mean_nll_delta"],
                       "maximum_new_severe_positions": EXPECTED_BUDGET,
                       "severe_threshold_nll": 10.0,
                       "control_and_candidate_must_pass_direct_bf16_gate": True,
                       "paired_candidate_minus_control_is_diagnostic": True},
        "pass": passed,
        "next_if_pass": "one disconnected A8G128-by-Q4G128 kernel qualification",
        "limitations": [
            "This is the complete represented source/reference formula, not a product kernel.",
            "No artifact, runtime format, 32K score, or production admission is authorized.",
        ],
    }


def validate_comparison(path: Path, control_path: Path, candidate_path: Path,
                        bf16_path: Path) -> dict[str, object]:
    retained = json.loads(path.read_text(encoding="utf-8"))
    expected = compare(control_path, candidate_path, bf16_path)
    if retained != expected:
        raise ValueError("retained activation-inclusive comparison differs")
    return retained


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        report = compare(args.control, args.candidate, args.bf16)
        _atomic_new(args.out, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        return 0 if report["pass"] else 1
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"compare-selective-a8q4-source: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
