#!/usr/bin/env python3
"""Validate product/BF16 and, when available, source-to-product FP8/Q4 effects."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl import run
from tools.ppl.compare_fp8_hybrid_source import _load_hybrid
from tools.ppl.compare_q4_group_source import _load_bf16, _source_key
from tools.ppl.q4_group_source_diagnostic import _atomic_new, sha256_file
from tools.ppl.validate_fp8_hybrid_greedy import (
    _candidate_sidecars, validate as validate_product,
)

ARTIFACT_TYPE = "ninfer_qwen3_8_fp8_hybrid_execution_localization"
SCHEMA_VERSION = 1
THRESHOLD = run.TERRIBLE_NLL
SUPPORTED_TOKENS = (run.DEFAULT_TOKENS, run.LONG_TOKENS)


def _input(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _pair(reference_nll: list[float], candidate_nll: list[float],
          reference_argmax: list[int], candidate_argmax: list[int]) -> dict:
    if (len(reference_nll) != len(candidate_nll)
            or len(reference_argmax) != len(candidate_argmax)
            or len(reference_nll) != len(reference_argmax)
            or not reference_nll):
        raise ValueError("localization sidecars are not nonempty and exactly aligned")
    if not all(math.isfinite(value) for value in (*reference_nll, *candidate_nll)):
        raise ValueError("localization NLL sidecars contain non-finite values")
    summary = run.paired_delta_summary(candidate_nll, reference_nll)
    assert summary is not None
    severe = run.severe_position_stats(
        candidate_nll, reference_nll, threshold=THRESHOLD,
        maximum_new_rate=run.QUALITY_TIERS["capacity-speed"]["maximum_new_severe_rate"],
        minimum_budget=0,
    )
    return {
        "reference_mean_nll": sum(reference_nll) / len(reference_nll),
        "candidate_mean_nll": sum(candidate_nll) / len(candidate_nll),
        "mean_nll_delta": sum(c - r for r, c in zip(reference_nll, candidate_nll))
        / len(reference_nll),
        "paired_delta_se": run.paired_delta_se(candidate_nll, reference_nll),
        **summary,
        **severe,
        **run.exact_argmax_stats(candidate_argmax, reference_argmax),
        "argmax_identity_is_gate": False,
    }


def _greedy_attribution(bf16: list[int], source: list[int], product: list[int]) -> dict:
    if len(bf16) != len(source) or len(bf16) != len(product):
        raise ValueError("greedy attribution sidecars lost alignment")
    categories = {
        "all_three_equal": 0,
        "source_only_differs_from_bf16_product_repairs": 0,
        "product_only_differs_from_bf16_runtime_introduces": 0,
        "both_differ_from_bf16_same_prediction": 0,
        "both_differ_from_bf16_different_predictions": 0,
    }
    for reference, source_token, product_token in zip(bf16, source, product):
        source_differs = source_token != reference
        product_differs = product_token != reference
        if not source_differs and not product_differs:
            categories["all_three_equal"] += 1
        elif source_differs and not product_differs:
            categories["source_only_differs_from_bf16_product_repairs"] += 1
        elif not source_differs and product_differs:
            categories["product_only_differs_from_bf16_runtime_introduces"] += 1
        elif source_token == product_token:
            categories["both_differ_from_bf16_same_prediction"] += 1
        else:
            categories["both_differ_from_bf16_different_predictions"] += 1
    return {"positions": len(bf16), **categories}


def _severe_attribution(bf16: list[float], source: list[float], product: list[float]) -> dict:
    bf16_set = {index for index, value in enumerate(bf16) if value >= THRESHOLD}
    source_set = {index for index, value in enumerate(source) if value >= THRESHOLD}
    product_set = {index for index, value in enumerate(product) if value >= THRESHOLD}
    source_new = source_set - bf16_set
    product_new = product_set - bf16_set
    return {
        "threshold_nll": THRESHOLD,
        "bf16_positions": sorted(bf16_set),
        "source_codec_positions": sorted(source_set),
        "product_positions": sorted(product_set),
        "new_vs_bf16_common": sorted(source_new & product_new),
        "new_vs_bf16_source_only_product_repairs": sorted(source_new - product_new),
        "new_vs_bf16_product_only_runtime_introduces": sorted(product_new - source_new),
    }


def compare(source_path: Path | None, product_campaign_path: Path, artifact_path: Path,
            bf16_campaign_path: Path, repeat_path: Path, tokens: int) -> dict:
    if tokens not in SUPPORTED_TOKENS:
        raise ValueError(f"unsupported localization token length: {tokens}")
    if source_path is not None:
        source_path = source_path.resolve(strict=True)
    product_campaign_path = product_campaign_path.resolve(strict=True)
    artifact_path = artifact_path.resolve(strict=True)
    bf16_campaign_path = bf16_campaign_path.resolve(strict=True)
    repeat_path = repeat_path.resolve(strict=True)

    product_validation = validate_product(
        product_campaign_path, artifact_path, bf16_campaign_path, repeat_path, tokens
    )
    product_campaign = json.loads(product_campaign_path.read_text(encoding="utf-8"))
    product_cells = [
        cell for cell in product_campaign["cells"]
        if cell.get("scheme") == "r9700-g16" and cell.get("schedule") == "prefill"
        and cell.get("prompt_tokens") == tokens
    ]
    if len(product_cells) != 1:
        raise ValueError(f"product campaign lacks one exact {tokens}-token G16 prefill cell")
    product_path, product_nll, product_argmax = _candidate_sidecars(product_cells[0])

    bf16_path = Path(product_validation["bf16_cell"]["path"])
    bf16_report, bf16_nll, bf16_argmax = _load_bf16(bf16_path)
    bf16_key = _source_key({"source": {
        "config_sha256": bf16_report["source_config_sha256"],
        "index_sha256": bf16_report["source_index_sha256"],
        "shards_sha256": bf16_report["source_shards_sha256"],
        "corpus_ids_sha256": bf16_report["corpus_ids_sha256"],
    }})
    if (
        bf16_report.get("prompt_tokens") != tokens
        or bf16_report.get("skip_tokens") != tokens // 2
        or len(product_nll) != tokens // 2 - 1
        or len(bf16_nll) != tokens // 2 - 1
    ):
        raise ValueError("product/BF16 scored-position domains differ")
    artifact = product_validation["candidate_artifact"]
    identity = {
        "weights_id": artifact["weights_id"],
        "recipe_id": artifact["conversion_receipt"]["recipe_id"],
        "selection_sha256": artifact["conversion_receipt"]["selection_sha256"],
    }
    product_vs_bf16 = _pair(bf16_nll, product_nll, bf16_argmax, product_argmax)
    report = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "diagnostic_runtime_increment_not_admission_gate"
            if source_path is not None
            else "validated_product_vs_bf16_no_matching_source_codec"
        ),
        "identity": identity,
        "inputs": {
            "source_codec": _input(source_path) if source_path is not None else None,
            "product_campaign": _input(product_campaign_path),
            "product_artifact": _input(artifact_path),
            "product_cell": _input(product_path),
            "bf16_campaign": _input(bf16_campaign_path),
            "bf16_repeat_comparison": _input(repeat_path),
            "bf16_cell": _input(bf16_path),
        },
        "workload": {
            "tokens": tokens, "skip_tokens": tokens // 2,
            "scored_positions": len(bf16_nll), "schedule": "prefill",
            "prefill_chunk": 4096,
        },
        "product_vs_bf16": product_vs_bf16,
        "source_localization_available": source_path is not None,
        "gate_thresholds_changed": False,
    }
    if source_path is None:
        report["limitations"] = [
            "No same-length source-codec sidecar was supplied, so this report validates product versus BF16 but makes no source-to-product attribution.",
            "An 8K source-codec row cannot be compared positionally with this 32K workload.",
            "Argmax differences are diagnostic and do not alter the existing quality gate.",
        ]
        return report

    source_report, source_nll, source_argmax = _load_hybrid(source_path, tokens)
    if _source_key(source_report) != bf16_key:
        raise ValueError("source-codec and product BF16 source/corpus identities differ")
    if (
        source_report.get("workload") != {
            "tokens": tokens, "skip": "half", "prefill_chunk": 4096,
            "schedule": "prefill", "device": 0,
        }
        or len(source_nll) != tokens // 2 - 1
        or len(source_argmax) != tokens // 2 - 1
    ):
        raise ValueError("source/product/BF16 scored-position domains differ")
    if source_report["identity"] != identity:
        raise ValueError("source-codec and product artifact identities differ")
    report.update({
        "source_codec_vs_bf16": _pair(
            bf16_nll, source_nll, bf16_argmax, source_argmax
        ),
        "product_minus_source_codec": _pair(
            source_nll, product_nll, source_argmax, product_argmax
        ),
        "greedy_attribution": _greedy_attribution(
            bf16_argmax, source_argmax, product_argmax
        ),
        "severe_attribution": _severe_attribution(bf16_nll, source_nll, product_nll),
        "limitations": [
            "The source-codec route fixes represented weights while retaining BF16 activations and matrix arithmetic.",
            "Product-minus-source therefore localizes the collective runtime increment: activation quantization, product matrix arithmetic/reduction, FP8-K/INT4-V cache effects, and fused implementation differences are not separable from these sidecars.",
            "All argmax and product-minus-source severe transitions are diagnostic and do not alter either existing quality gate.",
        ],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-codec", type=Path)
    parser.add_argument("--product-campaign", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bf16-campaign", type=Path, required=True)
    parser.add_argument("--bf16-repeat-comparison", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=SUPPORTED_TOKENS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        report = compare(
            args.source_codec, args.product_campaign, args.artifact,
            args.bf16_campaign, args.bf16_repeat_comparison, args.tokens,
        )
        _atomic_new(
            args.out, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
        )
        return 0
    except (FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"compare-fp8-hybrid-execution: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
