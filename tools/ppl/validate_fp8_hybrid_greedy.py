#!/usr/bin/env python3
"""Validate the post-PPL greedy-token diagnostic for the selected FP8/Q4 hybrid."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl import run

OUTPUT_TYPE = "ninfer_qwen3_8_fp8_hybrid_greedy_diagnostic"
OUTPUT_SCHEMA = 1
CANDIDATE = "r9700-g16"


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} root must be an object: {path}")
    return value


def _binding(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": run.file_sha256(resolved)}


def _require_binding(value: object, expected: dict, label: str) -> None:
    if not isinstance(value, dict):
        raise SystemExit(f"hybrid campaign lacks {label}")
    raw_path = value.get("path")
    if (
        not isinstance(raw_path, str)
        or Path(raw_path).resolve() != Path(expected["path"]).resolve()
        or value.get("sha256") != expected["sha256"]
    ):
        raise SystemExit(f"hybrid campaign {label} differs from the required authority")


def _validate_scorer(value: object, label: str) -> tuple[Path, dict]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise SystemExit(f"hybrid campaign lacks the {label} scorer identity")
    path = Path(value["path"]).resolve(strict=True)
    expected = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": run.file_sha256(path),
    }
    if value != expected:
        raise SystemExit(f"hybrid campaign {label} scorer bytes changed")
    return path, expected


def _candidate_sidecars(cell: dict) -> tuple[Path, list[float], list[int]]:
    path = run._command_output_path(cell)
    raw = _read_json(path, "candidate scorer report")
    if (
        any(field not in raw for field in run.CANDIDATE_SCORER_REPORT_FIELDS)
        or any(raw[field] != cell.get(field) for field in run.CANDIDATE_SCORER_REPORT_FIELDS)
    ):
        raise SystemExit("candidate raw scorer fields differ from the hybrid campaign")
    nll_path = path.with_suffix(".nllf32")
    argmax_path = path.with_suffix(".argmaxi32")
    if (
        not nll_path.is_file()
        or not argmax_path.is_file()
        or run.file_sha256(nll_path) != cell.get("nll_sha256")
        or run.file_sha256(argmax_path) != cell.get("argmax_sha256")
    ):
        raise SystemExit("candidate sidecar hashes differ from the hybrid campaign")
    nll = run.load_nlls(path)
    argmax = run.load_argmax(path)
    if not run.cell_ok(cell, nll, argmax) or not all(
        0 <= token < run.TOKEN_DOMAIN for token in argmax
    ):
        raise SystemExit("candidate sidecars are not complete, finite, aligned, and in-domain")
    return path, nll, argmax


def validate(
    campaign_path: Path,
    artifact_path: Path,
    bf16_campaign_path: Path,
    repeat_path: Path,
    tokens: int,
) -> dict:
    campaign_path = campaign_path.resolve(strict=True)
    artifact_path = artifact_path.resolve(strict=True)
    bf16_campaign_path = bf16_campaign_path.resolve(strict=True)
    repeat_path = repeat_path.resolve(strict=True)
    campaign = _read_json(campaign_path, "hybrid PPL campaign")

    gate = run.QUALITY_TIERS["capacity-speed"]["maximum_mean_nll_delta"]
    expected_top = {
        "artifact_type": run.CAMPAIGN_ARTIFACT_TYPE,
        "schema_version": run.CAMPAIGN_SCHEMA_VERSION,
        "model_id": run.MODEL_ID,
        "reference_weights_id": run.BF16_WEIGHTS_ID,
        "required_candidate_identity": "fp8-hybrid-selection-authority",
        "lengths": [tokens],
        "skip": "half",
        "prefill_chunk": 4096,
        "schedules": ["prefill"],
        "spec": "none",
        "draft_tokens": 0,
        "q4_activation_bits": 8,
        "w8_activation_bits": 8,
        "fp8_qk_wmma_enabled": True,
        "xattention_profile": "dense",
        "baseline": run.BASELINE,
        "gates": {CANDIDATE: gate},
        "quality_tier": "capacity-speed",
        "pass": True,
    }
    for field, expected in expected_top.items():
        if campaign.get(field) != expected:
            raise SystemExit(
                f"hybrid PPL campaign {field}={campaign.get(field)!r}; expected {expected!r}"
            )

    live_artifact = run.inspect_candidate_artifact(artifact_path)
    run.require_fp8_hybrid_candidate(live_artifact)
    if campaign.get("candidate_artifact") != live_artifact:
        raise SystemExit("hybrid PPL campaign artifact/receipt binding differs from live bytes")

    bf16_binding = _binding(bf16_campaign_path)
    repeat_binding = _binding(repeat_path)
    _require_binding(campaign.get("reused_bf16_campaign"), bf16_binding, "BF16 campaign")
    validated_repeat = run.validate_bf16_repeat_comparison(
        repeat_path, bf16_campaign_path
    )
    campaign_repeat = campaign.get("bf16_repeat_comparison")
    if campaign_repeat != validated_repeat:
        raise SystemExit("hybrid PPL campaign repeat proof differs from the live authority")

    corpus = campaign.get("corpus")
    weights = campaign.get("weights_inputs")
    scorers = campaign.get("scorers")
    if (
        not isinstance(corpus, dict)
        or not isinstance(corpus.get("path"), str)
        or not isinstance(weights, dict)
        or not isinstance(weights.get(run.BASELINE), str)
        or not isinstance(weights.get(CANDIDATE), str)
        or not isinstance(scorers, dict)
    ):
        raise SystemExit("hybrid PPL campaign lacks corpus/weight/scorer provenance")
    ids = Path(corpus["path"]).resolve(strict=True)
    if run.validate_corpus(ids, tokens) != corpus:
        raise SystemExit("hybrid PPL campaign corpus provenance changed")
    bf16_weights = Path(weights[run.BASELINE]).resolve(strict=True)
    candidate_weights = Path(weights[CANDIDATE]).resolve(strict=True)
    if candidate_weights != artifact_path:
        raise SystemExit("hybrid PPL campaign candidate path differs from the required artifact")
    bf16_scorer, bf16_scorer_identity = _validate_scorer(
        scorers.get(run.BASELINE), "BF16"
    )
    candidate_scorer, _ = _validate_scorer(scorers.get(CANDIDATE), "candidate")

    reused = run.load_reused_bf16_cells(
        bf16_campaign_path,
        bf16_weights=bf16_weights,
        bf16_scorer=bf16_scorer,
        scorer_identity=bf16_scorer_identity,
        ids=ids,
        corpus_provenance=corpus,
        lengths=[tokens],
        skip="half",
        prefill_chunk=4096,
        device=0,
    )
    bf16_cell, bf16_path = reused[tokens]
    bf16_argmax = run.load_argmax(bf16_path)

    cells = campaign.get("cells")
    if not isinstance(cells, list):
        raise SystemExit("hybrid PPL campaign cells must be an array")
    candidates = [
        cell for cell in cells
        if isinstance(cell, dict)
        and cell.get("scheme") == CANDIDATE
        and cell.get("prompt_tokens") == tokens
        and cell.get("schedule") == "prefill"
    ]
    references = [
        cell for cell in cells
        if isinstance(cell, dict)
        and cell.get("scheme") == run.BASELINE
        and cell.get("prompt_tokens") == tokens
        and cell.get("schedule") == "prefill"
    ]
    if len(candidates) != 1 or len(references) != 1 or len(cells) != 2:
        raise SystemExit("hybrid PPL campaign is not the exact BF16/G16 prefill pair")
    candidate = candidates[0]
    reference = references[0]
    _require_binding(reference.get("reused_bf16_campaign"), bf16_binding, "cell BF16 campaign")
    if any(reference.get(field) != bf16_cell.get(field) for field in run.BF16_SCORER_REPORT_FIELDS):
        raise SystemExit("hybrid PPL reference cell differs from the retained BF16 authority")
    for field in ("nll_sha256", "argmax_sha256", "complete_finite_aligned"):
        if reference.get(field) != bf16_cell.get(field):
            raise SystemExit(
                f"hybrid PPL reference cell differs from retained BF16 sidecar field {field}"
            )
    run.validate_cell_report(
        candidate,
        profile_name=CANDIDATE,
        weights=artifact_path,
        ids=ids,
        expected_weights_id=live_artifact["weights_id"],
        schedule="prefill",
        skip="half",
        tokens=tokens,
        prefill_chunk=4096,
        extra=[],
        expected_q4_activation_bits=8,
        expected_w8_activation_bits=8,
        expected_fp8_qk_wmma=True,
        expected_xattention_profile="dense",
    )
    candidate_path, candidate_nll, candidate_argmax = _candidate_sidecars(candidate)
    if not all(math.isfinite(value) for value in candidate_nll):
        raise SystemExit("candidate NLL sidecar is non-finite")
    stats = run.exact_argmax_stats(candidate_argmax, bf16_argmax)
    for field, expected in stats.items():
        if candidate.get(field) != expected:
            raise SystemExit(f"hybrid PPL campaign retained a stale greedy statistic: {field}")
    if candidate.get("argmax_identity_is_gate") is not False:
        raise SystemExit("lossy hybrid greedy identity must remain diagnostic-only")

    return {
        "artifact_type": OUTPUT_TYPE,
        "schema_version": OUTPUT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": run.MODEL_ID,
        "workload": {
            "prompt_tokens": tokens,
            "skip": "half",
            "scored_positions": len(candidate_argmax),
            "schedule": "prefill",
            "spec": "none",
            "prefill_chunk": 4096,
        },
        "hybrid_campaign": _binding(campaign_path),
        "candidate_artifact": live_artifact,
        "conversion_receipt": live_artifact["conversion_receipt"],
        "bf16_campaign": bf16_binding,
        "bf16_repeat_comparison": repeat_binding,
        "candidate_scorer": _binding(candidate_scorer),
        "candidate_cell": {
            "path": str(candidate_path.resolve()),
            "json_sha256": run.file_sha256(candidate_path),
            "argmax_sha256": candidate["argmax_sha256"],
        },
        "bf16_cell": {
            "path": str(bf16_path.resolve()),
            "json_sha256": run.file_sha256(bf16_path),
            "argmax_sha256": bf16_cell["argmax_sha256"],
        },
        "greedy": {**stats, "argmax_identity_is_gate": False},
        "valid": True,
    }


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing diagnostic: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bf16-campaign", type=Path, required=True)
    parser.add_argument("--bf16-repeat-comparison", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(run.DEFAULT_TOKENS, run.LONG_TOKENS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.campaign,
        args.artifact,
        args.bf16_campaign,
        args.bf16_repeat_comparison,
        args.tokens,
    )
    _write_new(args.out, result)
    print(f"hybrid greedy diagnostic: {args.out}")
    print(
        f"  compared={result['greedy']['argmax_compared']} "
        f"mismatches={result['greedy']['argmax_mismatches']} "
        f"flip_rate={result['greedy']['argmax_flip_rate']:.9f}"
    )


if __name__ == "__main__":
    main()
