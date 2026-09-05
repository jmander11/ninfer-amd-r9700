#!/usr/bin/env python3
"""Fail-closed filesystem boundaries for terminal quality recovery."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.reference.qwen3_8_27b_bf16.protocol import validate_checkpoint_files


def require_preflight(checkpoint: Path, outputs: list[Path]) -> None:
    validate_checkpoint_files(checkpoint.resolve(strict=True))
    for output in outputs:
        if os.path.lexists(output):
            raise ValueError(f"quality recovery output namespace is occupied: {output}")


def publish(pending: Path, final: Path) -> None:
    if pending.parent.resolve(strict=True) != final.parent.resolve(strict=True):
        raise ValueError("pending and final authority must share one directory")
    if pending.is_symlink() or not pending.is_file():
        raise ValueError("pending authority is not a regular file")
    if os.path.lexists(final):
        raise ValueError("final quality authority already exists")
    os.link(pending, final)
    descriptor = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    pending.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--checkpoint", required=True, type=Path)
    preflight.add_argument("--absent", action="append", default=[], type=Path)
    publication = subparsers.add_parser("publish")
    publication.add_argument("--pending", required=True, type=Path)
    publication.add_argument("--final", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            require_preflight(args.checkpoint, args.absent)
        else:
            publish(args.pending, args.final)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
