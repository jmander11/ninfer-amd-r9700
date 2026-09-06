#!/usr/bin/env python3
"""Dependency-light immutable-output preflight for the FP8 T5/T6 package."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(
    "/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/"
    "r9700-fp8-target-gate-up-t5t6-qualification-20260906")
OUTPUTS = (
    "regression.stdout", "regression.stderr", "regression.exit",
    "benchmark.stdout", "benchmark.stderr", "benchmark.exit", "static.stdout",
    "hardware.txt", "power-before.txt", "power-after.txt", "cell-t5.json",
    "cell-t6.json", "summary.json", "result.sha256",
)


def require_fresh_outputs(package: Path) -> None:
    for name in OUTPUTS:
        path = package / name
        if path.exists() or path.is_symlink():
            raise ValueError(f"immutable output path already exists: {path}")


def main() -> int:
    require_fresh_outputs(PACKAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
