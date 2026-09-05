#!/usr/bin/env python3
"""Reconcile completed FP8-hybrid product quality with current CPU capacity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.bench import produce_fp8_hybrid_capacity as capacity_tool
from tools.ppl import run as ppl


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _output_path(cell: dict) -> Path:
    command = cell.get("command")
    if not isinstance(command, list) or command.count("--out-json") != 1:
        raise ValueError("quality cell lacks one retained output command")
    value = Path(command[command.index("--out-json") + 1])
    return value if value.is_absolute() else REPO / value


def validate_campaign(path: Path, tokens: int, artifact: dict | None) -> tuple[dict, dict]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "artifact_type": ppl.CAMPAIGN_ARTIFACT_TYPE,
        "schema_version": ppl.CAMPAIGN_SCHEMA_VERSION,
        "lengths": [tokens],
        "schedules": ["prefill"],
        "spec": "none",
        "quality_tier": "capacity-speed",
        "required_candidate_identity": "fp8-hybrid-selection-authority",
        "pass": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{tokens}-token campaign {key} differs")
    if payload.get("gates") != {"r9700-g16": math.log(1.05)}:
        raise ValueError(f"{tokens}-token campaign does not use the capacity-speed NLL gate")

    recorded_artifact = payload.get("candidate_artifact")
    if not isinstance(recorded_artifact, dict):
        raise ValueError(f"{tokens}-token campaign lacks artifact provenance")
    if artifact is None:
        artifact = ppl.inspect_candidate_artifact(Path(recorded_artifact["path"]))
        ppl.require_fp8_hybrid_candidate(artifact)
    if recorded_artifact != artifact:
        raise ValueError(f"{tokens}-token campaign artifact or receipt is not current")

    scorers = payload.get("scorers")
    if not isinstance(scorers, dict) or set(scorers) != {ppl.BASELINE, "r9700-g16"}:
        raise ValueError(f"{tokens}-token campaign scorer set differs")
    for name, identity in scorers.items():
        scorer = Path(identity["path"]).resolve(strict=True)
        current = {"path": str(scorer), "bytes": scorer.stat().st_size, "sha256": sha256(scorer)}
        if identity != current:
            raise ValueError(f"{tokens}-token {name} scorer changed")
    for label in ("reused_bf16_campaign", "bf16_repeat_comparison"):
        identity = payload.get(label)
        if not isinstance(identity, dict):
            raise ValueError(f"{tokens}-token campaign lacks {label} provenance")
        authority = Path(identity["path"])
        authority = authority if authority.is_absolute() else REPO / authority
        if sha256(authority.resolve(strict=True)) != identity.get("sha256"):
            raise ValueError(f"{tokens}-token {label} bytes changed")

    cells = payload.get("cells")
    baseline = [cell for cell in cells if cell.get("scheme") == ppl.BASELINE]
    candidate = [cell for cell in cells if cell.get("scheme") == "r9700-g16"]
    if len(baseline) != 1 or len(candidate) != 1:
        raise ValueError(f"{tokens}-token campaign does not have one aligned pair")
    baseline_cell, candidate_cell = baseline[0], candidate[0]
    for cell in (baseline_cell, candidate_cell):
        output = _output_path(cell)
        nll_path = output.with_suffix(".nllf32")
        argmax_path = output.with_suffix(".argmaxi32")
        if sha256(nll_path) != cell.get("nll_sha256") or sha256(argmax_path) != cell.get("argmax_sha256"):
            raise ValueError(f"{tokens}-token sidecar digest differs")
        raw = json.loads(output.read_text(encoding="utf-8"))
        fields = (
            ppl.BF16_SCORER_REPORT_FIELDS
            if cell is baseline_cell else ppl.CANDIDATE_SCORER_REPORT_FIELDS
        )
        if any(raw.get(name) != cell.get(name) for name in fields):
            raise ValueError(f"{tokens}-token raw scorer report differs from its campaign")

    baseline_nll = ppl.load_nlls(_output_path(baseline_cell))
    candidate_nll = ppl.load_nlls(_output_path(candidate_cell))
    baseline_argmax = ppl.load_argmax(_output_path(baseline_cell))
    candidate_argmax = ppl.load_argmax(_output_path(candidate_cell))
    if not candidate_nll or len(candidate_nll) != len(baseline_nll):
        raise ValueError(f"{tokens}-token NLL sidecars are empty or unaligned")
    if len(candidate_argmax) != len(candidate_nll) or len(baseline_argmax) != len(candidate_nll):
        raise ValueError(f"{tokens}-token argmax sidecars are unaligned")
    if not all(math.isfinite(value) for value in (*baseline_nll, *candidate_nll)):
        raise ValueError(f"{tokens}-token NLL sidecars contain non-finite values")

    delta = sum(candidate_nll) / len(candidate_nll) - sum(baseline_nll) / len(baseline_nll)
    gate = math.log(1.05)
    severe = ppl.severe_position_stats(
        candidate_nll, baseline_nll, threshold=ppl.TERRIBLE_NLL,
        maximum_new_rate=ppl.QUALITY_TIERS["capacity-speed"]["maximum_new_severe_rate"],
        minimum_budget=ppl.QUALITY_TIERS["capacity-speed"]["minimum_new_severe_budget"],
    )
    for key, value in severe.items():
        if candidate_cell.get(key) != value:
            raise ValueError(f"{tokens}-token severe-position field {key} differs")
    if not math.isclose(candidate_cell.get("delta_mean_nll"), delta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{tokens}-token mean-NLL delta differs from sidecars")
    if delta > gate or severe["new_severe_positions_pass"] is not True:
        raise ValueError(f"{tokens}-token quality gate failed")
    return artifact, {
        "campaign": {"path": str(resolved), "sha256": sha256(resolved)},
        "tokens_scored": len(candidate_nll),
        "delta_mean_nll": delta,
        "maximum_delta_mean_nll": gate,
        "new_severe_positions": severe["new_severe_positions"],
        "new_severe_position_budget": severe["new_severe_position_budget"],
        "new_severe_position_rate": severe["new_severe_position_rate"],
        "maximum_new_severe_rate": severe["maximum_new_severe_rate"],
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-8k", type=Path, required=True)
    parser.add_argument("--quality-32k", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--weights-report", type=Path, required=True)
    parser.add_argument("--p2048-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    artifact, quality_8k = validate_campaign(args.quality_8k, 8192, None)
    _, quality_32k = validate_campaign(args.quality_32k, 32768, artifact)

    capacity_path = args.capacity.resolve(strict=True)
    retained_capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    current_capacity = capacity_tool.current_report(
        args.planner, args.weights_report
    )
    if retained_capacity != current_capacity:
        raise SystemExit("capacity report differs from the current CPU planner/provenance")
    capacity_cells = current_capacity["cells"]
    if [cell["concurrency"] for cell in capacity_cells] != [1, 2, 3, 4] or not all(
        cell["capacity_preserved"] for cell in capacity_cells
    ):
        raise SystemExit("current capacity evidence does not admit exact C1..4")

    p2048_path = args.p2048_report.resolve(strict=True)
    p2048 = json.loads(p2048_path.read_text(encoding="utf-8"))
    tests = p2048.get("tests")
    if not isinstance(tests, list) or len(tests) != 1 or tests[0].get("n_prompt") != 2048:
        raise SystemExit("P2048 report does not contain the bounded C1 prefill observation")
    throughput = tests[0].get("prefill_tok_s_mean")
    if not isinstance(throughput, (int, float)) or not math.isfinite(throughput):
        raise SystemExit("P2048 report has no finite prefill throughput")
    floor = 2000.0

    result = {
        "schema": "ninfer.r9700.fp8-hybrid-quality-capacity-admission.v1",
        "artifact": artifact,
        "quality": {"8k": quality_8k, "32k": quality_32k, "pass": True},
        "capacity": {
            "report": {"path": str(capacity_path), "sha256": sha256(capacity_path)},
            "concurrency": [1, 2, 3, 4],
            "remaining_slack_bytes": [cell["remaining_slack_bytes"] for cell in capacity_cells],
            "pass": True,
        },
        "quality_capacity_admitted": True,
        "performance": {
            "p2048_report": {"path": str(p2048_path), "sha256": sha256(p2048_path)},
            "observed_prefill_tok_s": throughput,
            "required_floor_tok_s": floor,
            "floor_pass": throughput >= floor,
            "prepared_broad_c1_4_matrices_launched": False,
        },
        "performance_promotion_admitted": False,
        "remaining_before_performance_promotion": [
            "raise and freshly remeasure current-production C1/P2048 prefill at or above 2000 tok/s",
            "then run the prepared provenance-bound C1..4 whole-inference and native-capacity matrices",
            "complete selected-artifact decode graph/eager and speculative execution evidence",
        ],
        "decision": "quality and current C1..4 capacity pass; performance promotion remains blocked by the P2048 floor",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"quality_capacity_admitted": True, "performance_promotion_admitted": False,
                      "p2048_tok_s": throughput}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
