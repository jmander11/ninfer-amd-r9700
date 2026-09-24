#!/usr/bin/env python3
"""Exclusively publish and revalidate the selected low-context authority."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from tools.bench.validate_low_context_prefill import validate_ladder


def _inode(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"low-context publication input is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _unlink_if_owned(path: Path, inode: tuple[int, int]) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if (metadata.st_dev, metadata.st_ino) == inode:
        path.unlink()


def publish(
    pending: Path,
    output: Path,
    manifest: Path,
    executable: Path,
    artifact: Path,
    selection: Path,
    minimum_p2048_tok_s: float,
    transient: Path | None = None,
) -> None:
    if os.path.lexists(output):
        raise ValueError(f"refusing occupied low-context authority path: {output}")
    pending_inode = _inode(pending)
    transient_inode = _inode(transient) if transient is not None else None
    actual = json.loads(pending.read_text(encoding="utf-8"))
    expected = validate_ladder(
        manifest, minimum_p2048_tok_s, executable, artifact, selection
    )
    if actual != expected:
        raise ValueError("pending low-context authority differs from fresh validation")

    published = False
    durable = False
    try:
        descriptor = os.open(pending, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(pending, output, follow_symlinks=False)
        published = True
        if _inode(output) != pending_inode:
            raise ValueError("published low-context authority inode changed")
        published_value = json.loads(output.read_text(encoding="utf-8"))
        if published_value != validate_ladder(
            manifest, minimum_p2048_tok_s, executable, artifact, selection
        ):
            raise ValueError("published low-context authority differs from fresh validation")
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        durable = True
        _unlink_if_owned(pending, pending_inode)
        if transient is not None and transient_inode is not None:
            _unlink_if_owned(transient, transient_inode)
    finally:
        if not durable:
            if published:
                _unlink_if_owned(output, pending_inode)
            _unlink_if_owned(pending, pending_inode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--min-p2048-tok-s", required=True, type=float)
    parser.add_argument("--transient", type=Path)
    args = parser.parse_args()
    try:
        publish(
            args.pending,
            args.out,
            args.manifest,
            args.executable,
            args.artifact,
            args.selection,
            args.min_p2048_tok_s,
            args.transient,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
