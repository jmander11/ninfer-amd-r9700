#!/usr/bin/env python3
"""Validate the bounded post-PPL FP8-hybrid execution-equivalence gate.

The GPU-producing campaign remains tools/ppl/run.py.  This validator only reopens
its evidence, revalidates the selected artifact/receipt and scorer identities, and
queries the compiled CPU width authority for the C=1..4 descriptor inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.ppl import run as ppl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_execution_coverage(payload: dict) -> dict[str, dict]:
    expected_root = {
        "artifact_type": ppl.CAMPAIGN_ARTIFACT_TYPE,
        "schema_version": ppl.CAMPAIGN_SCHEMA_VERSION,
        "required_candidate_identity": "fp8-hybrid-selection-authority",
        "lengths": [ppl.DEFAULT_TOKENS],
        "schedules": ["decode"],
        "spec": "mtp",
        "draft_tokens": ppl.DEFAULT_DRAFT_TOKENS,
        "execution_parity_max_abs_nll": 0.0,
        "pass": True,
    }
    for key, value in expected_root.items():
        if payload.get(key) != value:
            raise ValueError(f"execution campaign {key}={payload.get(key)!r}; expected {value!r}")

    wanted = {
        "mtp3_graph": ("mtp", 3, True, None),
        "mtp3_eager": ("mtp", 3, False, "device_graph_parity"),
        "ordinary_graph": ("none", 0, True, "spec_parity"),
        "mtp4_graph": ("mtp", 4, True, "draft_window_parity"),
    }
    candidates = [cell for cell in payload.get("cells", []) if cell.get("scheme") != ppl.BASELINE]
    found: dict[str, dict] = {}
    for label, (spec, draft, graph, parity_key) in wanted.items():
        matches = [
            cell for cell in candidates
            if cell.get("prompt_tokens") == ppl.DEFAULT_TOKENS
            and cell.get("schedule") == "decode"
            and cell.get("spec") == spec
            and cell.get("draft_tokens") == draft
            and cell.get("device_graph") is graph
        ]
        if len(matches) != 1:
            raise ValueError(f"execution campaign requires exactly one {label} candidate cell")
        cell = matches[0]
        if cell.get("pass") is not True or cell.get("quality_eligible") is not True:
            raise ValueError(f"execution campaign {label} cell did not pass")
        if parity_key is not None:
            parity = cell.get(parity_key)
            if not isinstance(parity, dict) or parity.get("pass") is not True:
                raise ValueError(f"execution campaign {label} lacks passing {parity_key}")
            if (parity.get("max_abs_nll_gate") != 0.0
                    or parity.get("complete_finite_aligned") is not True
                    or parity.get("argmax_exact") is not True):
                raise ValueError(f"execution campaign {label} parity is not exact and complete")
        found[label] = cell
    return found


def parse_width_inventory(raw: str, expected: tuple[int, int, int, int]) -> list[int]:
    rows = list(csv.DictReader(io.StringIO(raw)))
    if not rows:
        raise ValueError("compiled prepared-width inventory is empty")
    widths: list[int] = []
    for row in rows:
        observed = tuple(int(row[name]) for name in (
            "prefill", "max_concurrency", "mtp_width", "dflash_width"
        ))
        if observed != expected:
            raise ValueError(f"compiled prepared-width inventory profile {observed}; expected {expected}")
        widths.append(int(row["prepared_width"]))
    if widths != sorted(set(widths)):
        raise ValueError("compiled prepared-width inventory is not sorted and unique")
    return widths


def query_widths(
    tool: Path, prefill: int, concurrency: int, mtp_width: int,
    dflash_width: int = 0,
) -> list[int]:
    process = subprocess.run(
        [str(tool), "--host-hybrid-widths-csv", str(prefill), str(concurrency),
         str(mtp_width), str(dflash_width)],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return parse_width_inventory(
        process.stdout, (prefill, concurrency, mtp_width, dflash_width)
    )


def validate_current_identity(payload: dict) -> tuple[dict, dict]:
    recorded = payload.get("candidate_artifact")
    if not isinstance(recorded, dict):
        raise ValueError("execution campaign lacks candidate artifact provenance")
    path = Path(str(recorded.get("path", "")))
    current = ppl.inspect_candidate_artifact(path)
    ppl.require_fp8_hybrid_candidate(current)
    if current != recorded:
        raise ValueError("hybrid artifact or adjacent conversion receipt changed after campaign")

    scorers = payload.get("scorers")
    if not isinstance(scorers, dict) or set(scorers) != {ppl.BASELINE, "r9700-g16"}:
        raise ValueError("execution campaign must bind exactly BF16 and selected G16 scorers")
    for label, record in scorers.items():
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        expected = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        if record != expected:
            raise ValueError(f"execution campaign {label} scorer identity changed")
    return current, scorers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--width-tool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("execution campaign root must be an object")
    try:
        cells = validate_execution_coverage(payload)
        artifact, scorers = validate_current_identity(payload)
        prefill = int(payload.get("prefill_chunk", 0))
        inventories = {
            "ordinary": query_widths(args.width_tool.resolve(strict=True), prefill, 4, 0),
            "mtp3": query_widths(args.width_tool.resolve(strict=True), prefill, 4, 4),
            "mtp4": query_widths(args.width_tool.resolve(strict=True), prefill, 4, 5),
        }
        expected = {
            "ordinary": [1, 2, 3, 4, prefill],
            "mtp3": [1, 2, 3, 4, 8, 12, 16, prefill],
            "mtp4": [1, 2, 3, 4, 5, 10, 15, 20, prefill],
        }
        if inventories != expected:
            raise ValueError(f"compiled prepared-width inventories differ: {inventories!r}")
    except (OSError, ValueError, subprocess.CalledProcessError, SystemExit) as error:
        raise SystemExit(f"FP8 hybrid execution gate is invalid: {error}") from error

    evidence = {
        "artifact_type": "ninfer_fp8_hybrid_execution_gate",
        "schema_version": 1,
        "campaign": {"path": str(args.results.resolve()), "sha256": sha256(args.results)},
        "artifact": artifact,
        "scorers": scorers,
        "prepared_width_authority": {
            "tool": {"path": str(args.width_tool.resolve()), "sha256": sha256(args.width_tool)},
            "maximum_concurrency": 4,
            "inventories": inventories,
        },
        "covered_candidate_cells": sorted(cells),
        "criterion": "exact C1 graph/eager, MTP/ordinary, and MTP3/MTP4 parity; compiled C1..4 widths",
        "pass": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"FP8 hybrid execution gate: PASS ({args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
