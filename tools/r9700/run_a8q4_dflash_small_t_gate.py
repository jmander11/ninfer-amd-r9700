#!/usr/bin/env python3
"""Run the exact 39-cell C1 DFlash small-T screen and retain every outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SHAPES = (
    (4096, 5120), (5120, 6144), (7168, 5120), (12288, 5120),
    (34816, 5120), (5120, 17408), (248320, 5120), (131072, 5120),
    (5120, 25600), (6144, 5120), (5120, 4096), (1280, 5120), (256, 5120),
)
TOKENS = (4, 5, 6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cell(report: object, rows: int, columns: int, tokens: int) -> dict:
    if not isinstance(report, dict) or report.get("schema") != \
            "ninfer.r9700.a8q4-dflash-small-t-cell.v1":
        raise ValueError("small-T cell has the wrong schema")
    if report.get("shape") != {"rows": rows, "columns": columns, "tokens": tokens}:
        raise ValueError("small-T cell has the wrong shape")
    if report.get("production_dispatch_changed") is not False:
        raise ValueError("small-T cell claims a production dispatch change")
    status = report.get("status")
    decision = report.get("decision")
    numeric = report.get("numeric")
    if status not in ("passed", "rejected") or not isinstance(decision, dict) or \
            decision.get("accepted") is not (status == "passed"):
        raise ValueError("small-T cell status/decision is inconsistent")
    if not isinstance(numeric, dict) or numeric.get("maximum_bf16_steps_allowed") != 2 or \
            not isinstance(numeric.get("candidate_maximum_bf16_steps"), int) or \
            not isinstance(numeric.get("incumbent_maximum_bf16_steps"), int) or \
            numeric["candidate_maximum_bf16_steps"] > 2 or \
            numeric["incumbent_maximum_bf16_steps"] > 2:
        raise ValueError("small-T cell lacks direct oracle admission")
    for name in ("incumbent_forward_ms", "candidate_forward_ms", "candidate_reverse_ms",
                 "incumbent_reverse_ms", "incumbent_balanced_ms", "candidate_balanced_ms",
                 "forward_candidate_over_incumbent", "reverse_candidate_over_incumbent"):
        values = report.get("timing", {}).get(name)
        if not isinstance(values, list) or len(values) != 7 or \
                any(not isinstance(value, (int, float)) or value <= 0 for value in values):
            raise ValueError(f"small-T cell has invalid {name}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=Path("tools/r9700/build/a8q4_dflash_small_t_qual"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if args.output_dir.exists():
        raise SystemExit("output directory must be fresh")
    args.output_dir.mkdir(parents=True)

    cells = []
    for rows, columns in SHAPES:
        for tokens in TOKENS:
            name = f"n{rows}-k{columns}-t{tokens}.json"
            path = args.output_dir / name
            command = [str(binary), str(rows), str(columns), str(tokens),
                       "--out-json", str(path)]
            process = subprocess.run(command, check=False)
            if process.returncode not in (0, 1) or not path.is_file():
                raise RuntimeError(f"small-T cell failed without a retained decision: {command}")
            report = validate_cell(json.loads(path.read_text()), rows, columns, tokens)
            if (process.returncode == 0) != (report["status"] == "passed"):
                raise RuntimeError("small-T cell exit status disagrees with its report")
            cells.append({
                "rows": rows, "columns": columns, "tokens": tokens,
                "status": report["status"], "report": name, "sha256": sha256(path),
                "candidate_over_incumbent": report["decision"]["candidate_over_incumbent"],
                "paired_ratio_upper": report["decision"]["paired_ratio_upper"],
                "order_ratio_delta": report["decision"]["order_ratio_delta"],
            })

    passed = all(cell["status"] == "passed" for cell in cells)
    summary = {
        "schema": "ninfer.r9700.a8q4-dflash-small-t-gate.v1",
        "status": "passed" if passed else "rejected",
        "scope": "C1 T4/T5/T6 only; flattened C2..4 widths remain unqualified",
        "production_dispatch_changed": False,
        "binary": {"path": str(binary), "sha256": sha256(binary)},
        "required_shapes": [list(shape) for shape in SHAPES],
        "required_tokens": list(TOKENS),
        "required_cell_count": len(SHAPES) * len(TOKENS),
        "all_required_cells_pass": passed,
        "cells": cells,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
