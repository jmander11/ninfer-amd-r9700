#!/usr/bin/env python3
"""Resume-safe runner for the paired selective weight-only Q4G128 source gate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from tools.ppl.compare_selective_q4g128_source import (
    _load_score, validate_comparison,
)


REPO = Path(__file__).resolve().parents[2]
PYTHON = Path("/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python")
WEIGHTS = Path("/ssdpool2nvme/local_llm/models/qwen3.8-27b-bf16")
IDS = REPO / "tools/ppl/corpus.ids"
SCREEN = REPO / "profiles/bench/r9700-selective-q4g128-mse-source-screen-20260905.json"
BF16 = REPO / (
    "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/"
    "8192.prefill.bf16-reference.json"
)
OUTPUT = REPO / "profiles/ppl/selective-q4g128-source-8k-20260905"
ARMS = {
    "q4g64-absmax": OUTPUT / "q4g64-control.json",
    "q4g128-mse": OUTPUT / "q4g128-candidate.json",
}
COMPARISON = OUTPUT / "comparison.json"


def _triplet(path: Path) -> frozenset[Path]:
    return frozenset((path, path.with_suffix(".nllf32"), path.with_suffix(".argmaxi32")))


def _ensure_namespace() -> None:
    if OUTPUT.is_symlink():
        raise ValueError("selective Q4G128 output directory must not be a symlink")
    if OUTPUT.exists():
        if not OUTPUT.is_dir():
            raise ValueError("selective Q4G128 output namespace is not a directory")
    else:
        OUTPUT.mkdir()
    allowed = frozenset((COMPARISON,)).union(*(_triplet(path) for path in ARMS.values()))
    observed = frozenset(OUTPUT.iterdir())
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(f"unexpected selective Q4G128 output entries: {sorted(map(str, unexpected))}")


def _arm_state(profile: str, path: Path) -> str:
    present = frozenset(item for item in _triplet(path) if item.exists() or item.is_symlink())
    if not present:
        return "missing"
    if present != _triplet(path):
        raise ValueError(f"{profile}: partial selective Q4G128 output triplet")
    if any(item.is_symlink() or not item.is_file() for item in present):
        raise ValueError(f"{profile}: selective Q4G128 output triplet is not regular files")
    _load_score(path, profile, BF16)
    return "complete"


def _score_command(profile: str, path: Path) -> list[str]:
    return [
        str(PYTHON), "-m", "tools.ppl.selective_q4g128_source_diagnostic",
        "--weights", str(WEIGHTS), "--ids", str(IDS), "--codec", profile,
        "--source-screen", str(SCREEN), "--bf16", str(BF16), "--device", "0",
        "--out", str(path),
    ]


def run() -> int:
    os.chdir(REPO)
    _ensure_namespace()
    for profile, path in ARMS.items():
        if _arm_state(profile, path) == "missing":
            subprocess.run(_score_command(profile, path), check=True)
            if _arm_state(profile, path) != "complete":
                raise RuntimeError(f"{profile}: scorer returned without complete evidence")
    if COMPARISON.exists() or COMPARISON.is_symlink():
        if COMPARISON.is_symlink() or not COMPARISON.is_file():
            raise ValueError("selective Q4G128 comparison is not a regular file")
        report = validate_comparison(
            COMPARISON, ARMS["q4g64-absmax"], ARMS["q4g128-mse"], BF16
        )
    else:
        command = [
            str(PYTHON), "-m", "tools.ppl.compare_selective_q4g128_source",
            "--q4g64", str(ARMS["q4g64-absmax"]),
            "--q4g128", str(ARMS["q4g128-mse"]),
            "--bf16", str(BF16), "--out", str(COMPARISON),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"selective Q4G128 comparator failed: {completed.returncode}")
        report = validate_comparison(
            COMPARISON, ARMS["q4g64-absmax"], ARMS["q4g128-mse"], BF16
        )
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        return run()
    except (FileExistsError, OSError, RuntimeError, ValueError,
            subprocess.SubprocessError) as error:
        print(f"run-selective-q4g128-source-gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
