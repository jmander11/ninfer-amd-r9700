#!/usr/bin/env python3
"""Validate and compare the paired selective-G64/G128 source-only 8K scores."""

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
from tools.ppl.selective_q4g128_source_diagnostic import (
    ARTIFACT_TYPE, SCHEMA_VERSION, TOKENS, _gate, _implementation, _quantization,
)


COMPARISON_TYPE = "ninfer_qwen3_8_selective_q4g128_source_comparison"
EXPECTED_NEW_SEVERE_BUDGET = 11


def _close(actual: object, expected: float) -> bool:
    return type(actual) in (int, float) and math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _load_score(path: Path, codec: str, bf16_path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("artifact_type") != ARTIFACT_TYPE
            or report.get("schema_version") != SCHEMA_VERSION
            or report.get("status") != "diagnostic_weight_codec_only_not_product_ppl"
            or report.get("codec_profile") != codec):
        raise ValueError(f"{path}: selective source score identity differs")
    if report.get("implementation_sha256") != _implementation():
        raise ValueError(f"{path}: selective source score implementation differs")
    if report.get("quantization") != _quantization(codec):
        raise ValueError(f"{path}: selective source score quantization differs")
    expected_workload = {
        "tokens": TOKENS, "skip": "half", "prefill_chunk": 4096,
        "schedule": "prefill", "device": 0,
    }
    if report.get("workload") != expected_workload:
        raise ValueError(f"{path}: selective source score workload differs")
    if report.get("bf16_authority") != {
        "path": str(bf16_path.resolve()), "sha256": sha256_file(bf16_path),
    }:
        raise ValueError(f"{path}: BF16 authority binding differs")
    nlls = _sidecar_values(path, ".nllf32", "f")
    argmax = _sidecar_values(path, ".argmaxi32", "i")
    if len(nlls) != TOKENS // 2 - 1 or len(argmax) != len(nlls):
        raise ValueError(f"{path}: selective score sidecar cardinality differs")
    if not all(math.isfinite(float(value)) for value in nlls):
        raise ValueError(f"{path}: selective score NLL sidecar is nonfinite")
    for kind, suffix in (("nll", ".nllf32"), ("argmax", ".argmaxi32")):
        target = path.with_suffix(suffix)
        if report.get("sidecars", {}).get(kind) != {
            "path": target.name, "sha256": sha256_file(target),
        }:
            raise ValueError(f"{path}: {kind} sidecar binding differs")
    result = report.get("result", {})
    expected_result = {
        "tokens_scored": len(nlls),
        "argmax_tokens": len(argmax),
        "non_finite": 0,
        "terrible_tokens": sum(value >= 10.0 for value in nlls),
        "sum_nll": sum(nlls),
        "mean_nll": sum(nlls) / len(nlls),
        "max_nll": max(nlls),
        "ppl": math.exp(sum(nlls) / len(nlls)),
    }
    for key in ("tokens_scored", "argmax_tokens", "non_finite", "terrible_tokens"):
        if result.get(key) != expected_result[key]:
            raise ValueError(f"{path}: retained result {key} differs from sidecars")
    for key in ("sum_nll", "mean_nll", "max_nll", "ppl"):
        if not _close(result.get(key), expected_result[key]):
            raise ValueError(f"{path}: retained result {key} differs from sidecars")
    duration = result.get("score_seconds")
    if type(duration) not in (int, float) or not math.isfinite(float(duration)) \
            or float(duration) <= 0:
        raise ValueError(f"{path}: retained score duration is invalid")
    return report, nlls, argmax


def compare(control_path: Path, candidate_path: Path, bf16_path: Path) -> dict[str, object]:
    control, control_nll, control_argmax = _load_score(
        control_path, "q4g64-absmax", bf16_path
    )
    candidate, candidate_nll, candidate_argmax = _load_score(
        candidate_path, "q4g128-mse", bf16_path
    )
    bf16, bf16_nll, bf16_argmax = _load_bf16(bf16_path)
    if (_source_key(control) != _source_key(candidate)
            or control.get("workload") != candidate.get("workload")
            or control.get("execution") != candidate.get("execution")
            or control.get("matrix_scope") != candidate.get("matrix_scope")
            or control.get("sampled_source_gate") != candidate.get("sampled_source_gate")):
        raise ValueError("selective G64/G128 paired score identity differs")
    bf16_key = _source_key({"source": {
        "config_sha256": bf16["source_config_sha256"],
        "index_sha256": bf16["source_index_sha256"],
        "shards_sha256": bf16["source_shards_sha256"],
        "corpus_ids_sha256": bf16["corpus_ids_sha256"],
    }})
    if _source_key(control) != bf16_key:
        raise ValueError("selective scores and BF16 authority source/corpus differ")
    control_gate = _gate(control_nll, control_argmax, bf16_nll, bf16_argmax)
    candidate_gate = _gate(candidate_nll, candidate_argmax, bf16_nll, bf16_argmax)
    if control.get("quality_gate") != control_gate:
        raise ValueError("selective G64 retained quality gate differs from sidecars")
    if candidate.get("quality_gate") != candidate_gate:
        raise ValueError("selective G128 retained quality gate differs from sidecars")
    paired = _against(control_nll, candidate_nll, control_argmax, candidate_argmax)
    if (control_gate["against_bf16"]["new_severe_position_budget"]
            != EXPECTED_NEW_SEVERE_BUDGET
            or candidate_gate["against_bf16"]["new_severe_position_budget"]
            != EXPECTED_NEW_SEVERE_BUDGET):
        raise ValueError("selective source severe-position budget differs from exact 8K gate")
    passed = control_gate["pass"] is True and candidate_gate["pass"] is True
    return {
        "artifact_type": COMPARISON_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_weight_codec_gate_not_product_admission",
        "inputs": {
            "q4g64_control": {
                "path": str(control_path.resolve()), "sha256": sha256_file(control_path),
            },
            "q4g128_candidate": {
                "path": str(candidate_path.resolve()), "sha256": sha256_file(candidate_path),
            },
            "bf16": {"path": str(bf16_path.resolve()), "sha256": sha256_file(bf16_path)},
        },
        "workload": control["workload"],
        "q4g64_control_against_bf16": control_gate["against_bf16"],
        "q4g128_candidate_against_bf16": candidate_gate["against_bf16"],
        "q4g128_minus_q4g64": paired,
        "pass": passed,
        "acceptance": {
            "candidate_mean_nll_delta_maximum": candidate_gate["maximum_mean_nll_delta"],
            "severe_threshold_nll": 10.0,
            "candidate_new_severe_positions_maximum": EXPECTED_NEW_SEVERE_BUDGET,
            "paired_candidate_minus_control_is_diagnostic": True,
        },
        "limitations": [
            "Only the exact 160 selected roles are quantized; all other weights stay BF16.",
            "Activations and matrix arithmetic remain BF16, not product A8Q4.",
            "No A8G128 activation grouping is created, executed, or validated.",
            "Pass permits only subsequent format/kernel feasibility work.",
        ],
    }


def validate_comparison(path: Path, control_path: Path, candidate_path: Path,
                        bf16_path: Path) -> dict[str, object]:
    retained = json.loads(path.read_text(encoding="utf-8"))
    expected = compare(control_path, candidate_path, bf16_path)
    if retained != expected:
        raise ValueError("retained selective Q4G128 comparison differs from current inputs")
    return retained


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4g64", type=Path, required=True)
    parser.add_argument("--q4g128", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        report = compare(args.q4g64, args.q4g128, args.bf16)
        _atomic_new(args.out, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        return 0 if report["pass"] else 1
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"compare-selective-q4g128-source: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
