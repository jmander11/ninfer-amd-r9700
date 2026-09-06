#!/usr/bin/env python3
"""Reopen the retained quality evidence that can still contribute to terminal selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl import assemble_pareto, run


BF16 = REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json"
REPEAT = REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json"
ALL_Q4_DENSE = REPO / "profiles/ppl/xattention-dense-q4g64-v3-rebase-20260904/results.json"
HYBRID = (
    (8192, REPO / "profiles/ppl/fp8-hybrid-product-8k-20260904/results.json"),
    (32768, REPO / "profiles/ppl/fp8-hybrid-product-32k-20260904/results.json"),
)
HYBRID_WEIGHTS = Path("out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-eval.ninfer")
G16 = Path("build-r9700-dense-selection-g16/apps/ninfer-ppl")
G32 = Path("build-r9700-dense-selection-g32/apps/ninfer-ppl")
IDS = REPO / "tools/ppl/corpus.ids"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value


def main() -> None:
    repeat = run.validate_bf16_repeat_comparison(REPEAT, BF16)
    dense = load(ALL_Q4_DENSE)
    for group in (16, 32):
        assemble_pareto._campaign_quality_candidate(
            dense, "r9700-q4g64-eval", group, 4096
        )

    hybrid_rows = []
    for tokens, campaign_path in HYBRID:
        campaign = load(campaign_path)
        artifact = {
            **campaign.get("candidate_artifact"),
            "g16_path": str(HYBRID_WEIGHTS.resolve()),
            "g32_path": str(HYBRID_WEIGHTS.resolve()),
        }
        corpus = campaign.get("corpus")
        scorers = {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": run.file_sha256(path),
            }
            for name, path in (("r9700-g16", G16), ("r9700-g32", G32))
        }
        imported = run.load_reused_candidate_cells(
            campaign_path,
            selected_candidates=["r9700-g16", "r9700-g32"],
            candidate_artifact=artifact,
            profile_weights={
                "r9700-g16": HYBRID_WEIGHTS, "r9700-g32": HYBRID_WEIGHTS,
            },
            profile_bins={"r9700-g16": G16, "r9700-g32": G32},
            scorer_provenance=scorers,
            ids=IDS,
            corpus_provenance=corpus,
            lengths=[tokens],
            skip="half",
            prefill_chunk=4096,
            device=0,
            expected_q4_activation_bits=8,
            expected_w8_activation_bits=8,
            expected_fp8_qk_wmma=True,
            expected_xattention_profile="dense",
            allow_partial=True,
        )
        cell, path = imported[("r9700-g16", tokens)]
        hybrid_rows.append({
            "prompt_tokens": tokens,
            "cell": str(path),
            "nll_sha256": cell["nll_sha256"],
            "argmax_sha256": cell["argmax_sha256"],
        })

    print(json.dumps({
        "bf16_chunk4096": repeat,
        "all_q4_dense_chunk4096": {
            "path": str(ALL_Q4_DENSE),
            "sha256": run.file_sha256(ALL_Q4_DENSE),
            "groups": [16, 32],
            "lengths": [8192, 32768],
        },
        "four_role_dense_chunk4096_partial_only": {
            "groups": [16],
            "lengths": [8192, 32768],
            "cells": hybrid_rows,
            "terminal_authority": False,
            "reason": "G32 is absent and the two lengths are held by separate campaigns",
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
