#!/usr/bin/env python3
"""Create-only durable publication for the selected prefill-chunk authority."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from tools.bench.select_prefill_chunk import validate_selection_record


def _identity(path: Path) -> tuple[int, int]:
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"prefill-chunk selection is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _remove_owned(path: Path, owned: tuple[int, int] | None) -> None:
    if owned is None:
        return
    try:
        if _identity(path) == owned:
            path.unlink()
    except FileNotFoundError:
        pass


def publish(pending: Path, published: Path) -> None:
    if os.path.lexists(published):
        raise ValueError(f"prefill-chunk selection namespace already exists: {published}")
    if pending.is_symlink() or not pending.is_file():
        raise ValueError(f"pending prefill-chunk selection is not a regular file: {pending}")

    owned_pending = _identity(pending)
    with pending.open("rb") as source:
        opened = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino) != owned_pending:
            raise ValueError("pending prefill-chunk selection inode changed before fsync")
        os.fsync(source.fileno())
    if _identity(pending) != owned_pending:
        raise ValueError("pending prefill-chunk selection inode changed after fsync")

    pending_value = validate_selection_record(pending)
    if _identity(pending) != owned_pending:
        raise ValueError("pending prefill-chunk selection inode changed during validation")

    owned_published: tuple[int, int] | None = None
    durable = False
    try:
        os.link(pending, published)
        owned_published = _identity(published)
        if owned_published != owned_pending:
            raise ValueError("published prefill-chunk selection does not alias its pending inode")
        published_value = validate_selection_record(published)
        if published_value != pending_value or _identity(published) != owned_pending:
            raise ValueError("published prefill-chunk selection changed during validation")
        descriptor = os.open(published.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _identity(published) != owned_pending:
            raise ValueError("published prefill-chunk selection changed after directory fsync")
        durable = True
    finally:
        if not durable:
            _remove_owned(published, owned_published)

    _remove_owned(pending, owned_pending)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pending", type=Path)
    parser.add_argument("published", type=Path)
    args = parser.parse_args()
    publish(args.pending, args.published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
