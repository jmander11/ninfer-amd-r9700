#!/usr/bin/env python3
"""Validate exact fresh-process repeats of the deterministic BF16 authority."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ppl import run


ARTIFACT_TYPE = run.BF16_REPEAT_ARTIFACT_TYPE
SCHEMA_VERSION = run.BF16_REPEAT_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    return run.file_sha256(path)


def _flag(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise ValueError(f"BF16 command must contain exactly one {name}")
    index = command.index(name)
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise ValueError(f"BF16 command has malformed {name}")
    return command[index + 1]


def _require_distinct_campaigns(
    first: Path, second: Path, output: Path | None = None,
) -> None:
    first_resolved = first.resolve(strict=True)
    second_resolved = second.resolve(strict=True)
    if first_resolved == second_resolved or first_resolved.samefile(second_resolved):
        raise ValueError("BF16 repeats must be two distinct fresh campaign paths")
    if output is None:
        return
    output_resolved = output.resolve()
    for campaign in (first_resolved, second_resolved):
        aliases_campaign = output_resolved == campaign
        if output_resolved.exists():
            aliases_campaign = aliases_campaign or output_resolved.samefile(campaign)
        if aliases_campaign:
            raise ValueError("BF16 repeat output must not alias an input campaign")


def load_campaign(path: Path) -> tuple[dict, dict[int, tuple[dict, Path]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    if (
        payload.get("artifact_type") != run.CAMPAIGN_ARTIFACT_TYPE
        or payload.get("schema_version") != run.CAMPAIGN_SCHEMA_VERSION
        or payload.get("pass") is not True
        or payload.get("candidate_artifact") is not None
        or set(payload.get("weights_inputs", {})) != {run.BASELINE}
        or set(payload.get("scorers", {})) != {run.BASELINE}
        or payload.get("schedules") != ["prefill"]
        or payload.get("spec") != "none"
        or payload.get("draft_tokens") != 0
    ):
        raise ValueError(f"{path} is not a reference-only deterministic prefill campaign")
    lengths = payload.get("lengths")
    if lengths != [run.DEFAULT_TOKENS, run.LONG_TOKENS]:
        raise ValueError(f"{path} must contain the matched 8K/32K authority")
    campaign_cells = payload.get("cells")
    if not isinstance(campaign_cells, list):
        raise ValueError(f"{path} lacks the exact reference-cell inventory")
    reference_cells = [
        cell for cell in campaign_cells
        if isinstance(cell, dict) and cell.get("scheme") == run.BASELINE
    ]
    if len(reference_cells) != len(lengths) or len(campaign_cells) != len(lengths):
        raise ValueError(f"{path} lacks the exact reference-cell inventory")
    command = reference_cells[0].get("command")
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise ValueError(f"{path} has a malformed BF16 scorer command")
    scorer = Path(command[0])
    if not scorer.is_file():
        raise ValueError(f"{path} BF16 scorer is missing: {scorer}")
    scorer_identity = {
        "path": str(scorer.resolve()),
        "bytes": scorer.stat().st_size,
        "sha256": _sha256(scorer),
    }
    if payload["scorers"][run.BASELINE] != scorer_identity:
        raise ValueError(f"{path} BF16 scorer bytes differ from the campaign")
    ids = Path(_flag(command, "--ids"))
    try:
        corpus = run.validate_corpus(ids, max(lengths))
    except SystemExit as error:
        raise ValueError(f"{path} BF16 corpus does not revalidate: {error}") from error
    if payload.get("corpus") != corpus:
        raise ValueError(f"{path} BF16 corpus provenance differs from its command inputs")
    try:
        cells = run.load_reused_bf16_cells(
            path,
            bf16_weights=Path(_flag(command, "--weights")),
            bf16_scorer=Path(command[0]),
            scorer_identity=scorer_identity,
            ids=ids,
            corpus_provenance=corpus,
            lengths=lengths,
            skip=payload["skip"],
            prefill_chunk=payload["prefill_chunk"],
            device=int(_flag(command, "--device")),
        )
    except SystemExit as error:
        raise ValueError(f"invalid BF16 campaign {path}: {error}") from error
    return payload, cells


def _validate_sidecar_content(label: str, tokens: int, cell: dict, path: Path) -> None:
    nlls = run.load_nlls(path)
    argmax = run.load_argmax(path)
    if not run.cell_ok(cell, nlls, argmax):
        raise ValueError(f"{label} BF16 {tokens}-token sidecars have invalid content")
    total = math.fsum(nlls)
    mean = total / len(nlls)
    expected = {
        "sum_nll": total,
        "mean_nll": mean,
        "max_nll": max(nlls),
        "ppl": math.exp(mean),
        "terrible_tokens": sum(value >= cell["terrible_nll"] for value in nlls),
    }
    differing = [field for field, value in expected.items() if cell.get(field) != value]
    if differing:
        raise ValueError(
            f"{label} BF16 {tokens}-token NLL aggregates differ from sidecar: "
            + ", ".join(differing)
        )


def compare(
    first_payload: dict,
    first_cells: dict[int, tuple[dict, Path]],
    second_payload: dict,
    second_cells: dict[int, tuple[dict, Path]],
) -> dict:
    identity_fields = (
        "model_id", "reference_weights_id", "reference_source", "reference_execution",
        "corpus", "lengths", "skip", "prefill_chunk", "schedules", "spec",
        "draft_tokens", "terrible_nll",
    )
    missing_identity = [
        field for field in identity_fields
        if field not in first_payload or field not in second_payload
    ]
    if missing_identity:
        raise ValueError(
            "BF16 repeat campaign identity is missing: " + ", ".join(missing_identity)
        )
    differing_identity = [
        field for field in identity_fields
        if first_payload[field] != second_payload[field]
    ]
    if "scorers" not in first_payload or "scorers" not in second_payload:
        raise ValueError("BF16 repeat campaign identity is missing: scorers")
    if first_payload["scorers"] != second_payload["scorers"]:
        differing_identity.append("scorers")
    if differing_identity:
        raise ValueError(
            "BF16 repeat campaign identity differs: " + ", ".join(differing_identity)
        )

    rows = []
    exact = True
    for tokens in first_payload["lengths"]:
        if tokens not in first_cells or tokens not in second_cells:
            raise ValueError(f"BF16 repeat lacks the {tokens}-token cell")
        first_cell, first_path = first_cells[tokens]
        second_cell, second_path = second_cells[tokens]
        for label, cell in (("first", first_cell), ("second", second_cell)):
            missing = [field for field in run.BF16_SCORER_REPORT_FIELDS if field not in cell]
            if missing:
                raise ValueError(
                    f"{label} BF16 {tokens}-token cell lacks semantic fields: "
                    + ", ".join(missing)
                )
            expected_elements = run.expected_tokens_scored(cell)
            if (
                expected_elements <= 0
                or cell["tokens_scored"] != expected_elements
                or cell["argmax_tokens"] != expected_elements
            ):
                raise ValueError(f"{label} BF16 {tokens}-token cell has inconsistent lengths")
            expected_bytes = expected_elements * 4
            for suffix in (".nllf32", ".argmaxi32"):
                sidecar = first_path.with_suffix(suffix) if label == "first" \
                    else second_path.with_suffix(suffix)
                if not sidecar.is_file() or sidecar.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"{label} BF16 {tokens}-token {suffix} sidecar has invalid length"
                    )
            _validate_sidecar_content(
                label, tokens, cell, first_path if label == "first" else second_path
            )
        semantic_fields = {
            field: first_cell[field]
            for field in run.BF16_SCORER_REPORT_FIELDS
        }
        semantic_exact = all(
            first_cell[field] == second_cell[field]
            for field in run.BF16_SCORER_REPORT_FIELDS
        )
        nll_hashes = (_sha256(first_path.with_suffix(".nllf32")),
                      _sha256(second_path.with_suffix(".nllf32")))
        argmax_hashes = (_sha256(first_path.with_suffix(".argmaxi32")),
                         _sha256(second_path.with_suffix(".argmaxi32")))
        row_exact = semantic_exact and nll_hashes[0] == nll_hashes[1] \
            and argmax_hashes[0] == argmax_hashes[1]
        exact = exact and row_exact
        rows.append({
            "prompt_tokens": tokens,
            "exact": row_exact,
            "semantic_fields_exact": semantic_exact,
            "semantic_fields": semantic_fields,
            "nll_sha256": {"first": nll_hashes[0], "second": nll_hashes[1]},
            "argmax_sha256": {"first": argmax_hashes[0], "second": argmax_hashes[1]},
        })
    return {"exact": exact, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _require_distinct_campaigns(args.first, args.second, args.out)
    first_payload, first_cells = load_campaign(args.first)
    second_payload, second_cells = load_campaign(args.second)
    result = compare(first_payload, first_cells, second_payload, second_cells)
    result.update({
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "quality_evidence": False,
        "inputs": {
            "first": {"path": str(args.first.resolve()), "sha256": _sha256(args.first)},
            "second": {"path": str(args.second.resolve()), "sha256": _sha256(args.second)},
        },
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"BF16 repeat comparison: exact={result['exact']} ({args.out})")
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
