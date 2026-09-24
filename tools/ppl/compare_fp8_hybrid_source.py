#!/usr/bin/env python3
"""Compare one canonical FP8/Q4 source-codec diagnostic with retained BF16."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.convert.qwen3_8_27b_r9700 import fp8_hybrid_decision, fp8_hybrid_inventory
from tools.convert.qwen3_8_27b_r9700.fp8_hybrid_decision import DECISION
from tools.ppl.compare_q4_group_source import (
    _against, _exact_int, _load_bf16, _sidecar_values, _source_key,
)
from tools.ppl.fp8_hybrid_source_diagnostic import (
    ARTIFACT_TYPE, COMPARISON_TYPE, QUANTIZATION, SCHEMA_VERSION, TOKENS,
    _matrix_scope,
)
from tools.ppl.q4_group_source_diagnostic import (
    TERRIBLE_NLL, _atomic_new, sha256_file,
)


def _load_hybrid(path: Path, tokens: int = TOKENS) -> tuple[dict, list[float], list[int]]:
    report = json.loads(path.read_text())
    identity = report.get("identity")
    expected_identity = {
        "weights_id": DECISION.weights_id,
        "recipe_id": DECISION.recipe_id,
        "selection_sha256": DECISION.selection_sha256,
    }
    if (report.get("artifact_type") != ARTIFACT_TYPE
            or report.get("schema_version") != SCHEMA_VERSION
            or report.get("status") != "diagnostic_weight_codec_only_not_product_ppl"
            or identity != expected_identity
            or report.get("format_counts") != fp8_hybrid_inventory.FORMAT_COUNTS
            or report.get("quantization") != QUANTIZATION
            or report.get("matrix_scope") != _matrix_scope()):
        raise ValueError(f"{path}: wrong hybrid diagnostic identity")
    implementation = report.get("implementation", {})
    diagnostic_path = REPO / "tools/ppl/fp8_hybrid_source_diagnostic.py"
    if (implementation.get("diagnostic") != sha256_file(diagnostic_path)
            or implementation.get("selection_authority")
            != sha256_file(fp8_hybrid_decision._authority_path())
            or implementation.get("bf16_backend")
            != sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/backend.py")
            or implementation.get("bf16_protocol")
            != sha256_file(REPO / "tools/reference/qwen3_8_27b_bf16/protocol.py")):
        raise ValueError(f"{path}: stale hybrid diagnostic implementation")
    if report.get("workload", {}).get("tokens") != tokens:
        raise ValueError(f"{path}: hybrid diagnostic is not the exact {tokens}-token workload")
    nlls = _sidecar_values(path, ".nllf32", "f")
    argmax = _sidecar_values(path, ".argmaxi32", "i")
    for kind, suffix in (("nll", ".nllf32"), ("argmax", ".argmaxi32")):
        target = path.with_suffix(suffix)
        if report["sidecars"][kind] != {
            "path": target.name, "sha256": sha256_file(target),
        }:
            raise ValueError(f"{path}: {kind} sidecar identity mismatch")
    count = _exact_int(report["result"].get("tokens_scored"), "tokens_scored")
    if (count != len(nlls)
            or _exact_int(report["result"].get("argmax_tokens"), "argmax_tokens")
            != len(argmax)):
        raise ValueError(f"{path}: sidecar cardinality mismatch")
    if not nlls or not all(math.isfinite(value) for value in nlls):
        raise ValueError(f"{path}: NLL sidecar is empty or nonfinite")
    derived = {
        "non_finite": 0,
        "terrible_tokens": sum(value >= TERRIBLE_NLL for value in nlls),
        "sum_nll": sum(nlls),
        "mean_nll": sum(nlls) / len(nlls),
        "max_nll": max(nlls),
    }
    for key, value in derived.items():
        actual = report["result"].get(key)
        if actual != value and not (
            type(value) is float and type(actual) is float
            and math.isclose(actual, value, rel_tol=1e-7, abs_tol=1e-7)
        ):
            raise ValueError(f"{path}: result {key} does not match raw sidecar")
    return report, nlls, argmax


def compare(hybrid_path: Path, bf16_path: Path) -> dict:
    hybrid, candidate_nll, candidate_argmax = _load_hybrid(hybrid_path)
    bf16, reference_nll, reference_argmax = _load_bf16(bf16_path)
    bf16_source = _source_key({"source": {
        "config_sha256": bf16["source_config_sha256"],
        "index_sha256": bf16["source_index_sha256"],
        "shards_sha256": bf16["source_shards_sha256"],
        "corpus_ids_sha256": bf16["corpus_ids_sha256"],
    }})
    if _source_key(hybrid) != bf16_source:
        raise ValueError("hybrid diagnostic and BF16 source/corpus identities differ")
    if (bf16.get("prompt_tokens") != TOKENS
            or bf16.get("schedule") != "prefill"
            or bf16.get("skip_tokens") != TOKENS // 2):
        raise ValueError("BF16 input is not the exact retained 8K prefill/half cell")
    against = _against(
        reference_nll, candidate_nll, reference_argmax, candidate_argmax,
    )
    return {
        "artifact_type": COMPARISON_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_weight_codec_gate_not_product_admission",
        "identity": hybrid["identity"],
        "inputs": {
            "hybrid": {"path": str(hybrid_path.resolve()),
                       "sha256": sha256_file(hybrid_path)},
            "bf16": {"path": str(bf16_path.resolve()), "sha256": sha256_file(bf16_path)},
        },
        "workload": hybrid["workload"],
        "against_bf16": against,
        "pass": against["mean_nll_delta_pass"] and against["new_severe_positions_pass"],
        "limitations": [
            "Weights use the canonical E4M3/Q4 selection, but activations and matmul remain BF16.",
            "Argmax flips are diagnostics and are not a lossy-codec admission gate.",
            "A pass is source-codec evidence, not product PPL or production promotion.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        report = compare(args.hybrid, args.bf16)
        _atomic_new(args.out, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        return 0 if report["pass"] else 1
    except (FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"compare-fp8-hybrid-source: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
