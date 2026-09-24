#!/usr/bin/env python3
"""Validate the exact four-role N16/K16 A4 numerical-first campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path


REPO = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
OUTPUT = REPO / "profiles/ppl/four-role-n16k16-a4-8k-20260905"
ARTIFACT = REPO / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer"
SCORER = REPO / "build-r9700-a4q4-n16k16/apps/ninfer-ppl"
IDS = REPO / "tools/ppl/corpus.ids"
BF16 = REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/results.json"
REPEAT = REPO / "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-repeat-comparison-20260904.json"
WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
ARTIFACT_SHA256 = "040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2"
SCORER_SHA256 = "55e5dfb93f385f76f1a8d41b00f53e4790a72795f333ca6d70b170d4acf7fc38"
BF16_SHA256 = "9b402ad02be0f27d80530bd1eba4ead7c0179c658aa9654d43ca5b0fb9ca41f0"
REPEAT_SHA256 = "9b13916a2a3ca8b8e5d01352f98a8306414f4e13a79989c19b1d1248d5eb00de"
IDS_SHA256 = "aa1ba6d9932a4bcefdfd1c395bde73911daeba083c6b45d353a421dcd1eae077"
MEAN_LIMIT = 0.04879016416943205
SEVERE_LIMIT = 11


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_f32(path: Path) -> list[float]:
    payload = path.read_bytes()
    if len(payload) % 4:
        raise ValueError(f"unaligned FP32 sidecar: {path}")
    return [value[0] for value in struct.iter_unpack("<f", payload)]


def _load_i32(path: Path) -> list[int]:
    payload = path.read_bytes()
    if len(payload) % 4:
        raise ValueError(f"unaligned I32 sidecar: {path}")
    return [value[0] for value in struct.iter_unpack("<i", payload)]


def validate(output: Path = OUTPUT) -> dict:
    if output.is_symlink() or not output.is_dir():
        raise ValueError("campaign output must be a real directory")
    expected_files = {
        "8192.prefill.r9700-g16.json",
        "8192.prefill.r9700-g16.nllf32",
        "8192.prefill.r9700-g16.argmaxi32",
        "results.json",
        "results.md",
    }
    actual_files = {entry.name for entry in output.iterdir()}
    if actual_files != expected_files or any(entry.is_symlink() for entry in output.iterdir()):
        raise ValueError("campaign output file inventory differs")

    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    expected_top = {
        "artifact_type": "ninfer_r9700_ppl_campaign",
        "schema_version": 6,
        "model_id": "qwen3.8-27b",
        "reference_weights_id": "bf16-source",
        "lengths": [8192],
        "skip": "half",
        "prefill_chunk": 4096,
        "schedules": ["prefill"],
        "spec": "none",
        "draft_tokens": 0,
        "q4_activation_bits": 4,
        "w8_activation_bits": 8,
        "fp8_qk_wmma_enabled": True,
        "xattention_profile": "dense",
        "quality_tier": "capacity-speed",
        "gates": {"r9700-g16": MEAN_LIMIT},
        "required_candidate_identity": "fp8-hybrid-selection-authority",
    }
    for key, expected in expected_top.items():
        if report.get(key) != expected:
            raise ValueError(f"campaign {key} differs")
    if report.get("weights_inputs") != {
        "bf16-reference": "/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16",
        "r9700-g16": str(ARTIFACT),
    }:
        raise ValueError("campaign weight inputs differ")
    artifact = report.get("candidate_artifact", {})
    if any((
        artifact.get("path") != str(ARTIFACT),
        artifact.get("bytes") != 21_553_549_312,
        artifact.get("sha256") != ARTIFACT_SHA256,
        artifact.get("model_id") != "qwen3.8-27b",
        artifact.get("weights_id") != WEIGHTS_ID,
        artifact.get("conversion_receipt", {}).get("sha256")
            != "2c5cb9f405ba581f685a96b1a2965ad3ff49b31280ebdd15d19158b6f4779c59",
        artifact.get("conversion_receipt", {}).get("recipe_id")
            != "r9700-q4g64-f8e4m3-four-role-n16k16-eval-v1",
    )):
        raise ValueError("campaign artifact identity differs")
    scorer = report.get("scorers", {}).get("r9700-g16", {})
    if scorer.get("path") != str(SCORER) or scorer.get("sha256") != SCORER_SHA256:
        raise ValueError("campaign scorer identity differs")
    corpus = report.get("corpus", {})
    if corpus.get("path") != str(IDS) or corpus.get("ids_sha256") != IDS_SHA256:
        raise ValueError("campaign corpus identity differs")
    reused = report.get("reused_bf16_campaign", {})
    repeat = report.get("bf16_repeat_comparison", {})
    if Path(str(reused.get("path"))).resolve() != BF16 or reused.get("sha256") != BF16_SHA256:
        raise ValueError("campaign BF16 authority differs")
    if Path(str(repeat.get("path"))).resolve() != REPEAT or repeat.get("sha256") != REPEAT_SHA256:
        raise ValueError("campaign BF16 repeat proof differs")

    cells = report.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ValueError("campaign must contain exactly BF16 and A4 cells")
    candidates = [cell for cell in cells if cell.get("scheme") == "r9700-g16"]
    if len(candidates) != 1:
        raise ValueError("campaign must contain exactly one A4 cell")
    cell = candidates[0]
    expected_cell = {
        "weights_id": WEIGHTS_ID,
        "schedule": "prefill",
        "spec": "none",
        "draft_tokens": 0,
        "prefill_chunk": 4096,
        "prompt_tokens": 8192,
        "skip_tokens": 4096,
        "tokens_scored": 4095,
        "argmax_tokens": 4095,
        "non_finite": 0,
        "q4_activation_bits": 4,
        "w8_activation_bits": 8,
        "fp8_qk_wmma_enabled": True,
        "xattention_qualification": False,
        "complete_finite_aligned": True,
        "new_severe_position_budget": SEVERE_LIMIT,
    }
    for key, expected in expected_cell.items():
        if cell.get(key) != expected:
            raise ValueError(f"A4 cell {key} differs")
    raw_path = output / "8192.prefill.r9700-g16.json"
    nll_path = raw_path.with_suffix(".nllf32")
    argmax_path = raw_path.with_suffix(".argmaxi32")
    if cell.get("nll_sha256") != _sha256(nll_path) or cell.get("argmax_sha256") != _sha256(argmax_path):
        raise ValueError("A4 sidecar hash differs")
    nll = _load_f32(nll_path)
    argmax = _load_i32(argmax_path)
    if len(nll) != 4095 or len(argmax) != 4095 or not all(math.isfinite(value) for value in nll):
        raise ValueError("A4 sidecars are incomplete, nonfinite, or misaligned")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    for key, value in raw.items():
        if cell.get(key) != value:
            raise ValueError(f"A4 raw report field differs: {key}")
    delta = cell.get("delta_mean_nll")
    severe = cell.get("new_severe_positions")
    admitted = (
        isinstance(delta, (int, float)) and math.isfinite(delta) and delta <= MEAN_LIMIT
        and type(severe) is int and severe <= SEVERE_LIMIT
        and cell.get("new_severe_positions_pass") is True
        and cell.get("quality_eligible") is True
        and cell.get("pass") is True
        and report.get("pass") is True
    )
    return {"admitted": admitted, "mean_nll_delta": delta, "new_severe_positions": severe}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = validate(args.output)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"A4 numerical gate validation failed: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0 if result["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
