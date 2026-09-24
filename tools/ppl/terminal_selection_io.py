#!/usr/bin/env python3
"""Exclusive publication helpers for the terminal schema-v7 selection authority."""

from __future__ import annotations

import json
import hashlib
import os
import stat
from pathlib import Path

from tools.ppl.pareto import validate_terminal_production_authority


def require_absent(paths: list[Path]) -> None:
    occupied = [str(path) for path in paths if os.path.lexists(path)]
    if occupied:
        raise ValueError("terminal selection namespace already exists: " + ", ".join(occupied))


def publish(
    pending_input: Path,
    published_input: Path,
    pending_result: Path,
    published_result: Path,
    *,
    guard_path: Path | None = None,
    guard_sha256: str | None = None,
) -> None:
    for path in (pending_input, pending_result):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"terminal selection pending artifact is not a regular file: {path}")
    require_absent([published_input, published_result])

    if (guard_path is None) != (guard_sha256 is None):
        raise ValueError("terminal selection publication guard is incomplete")

    def guard_unchanged() -> None:
        if guard_path is None:
            return
        with guard_path.open("rb") as source:
            observed = hashlib.file_digest(source, "sha256").hexdigest()
        if observed != guard_sha256:
            raise ValueError("terminal selection publication guard changed")

    def identity(path: Path) -> tuple[int, int]:
        metadata = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"terminal selection publication is not regular: {path}")
        return metadata.st_dev, metadata.st_ino

    def remove_owned(path: Path, owned: tuple[int, int] | None) -> None:
        if owned is None:
            return
        try:
            if identity(path) == owned:
                path.unlink()
        except FileNotFoundError:
            pass

    pending_input_inode = identity(pending_input)
    pending_result_inode = identity(pending_result)
    for path, owned in (
        (pending_input, pending_input_inode), (pending_result, pending_result_inode)
    ):
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if (opened.st_dev, opened.st_ino) != owned:
                raise ValueError("terminal selection pending inode changed before fsync")
            os.fsync(source.fileno())
        if identity(path) != owned:
            raise ValueError("terminal selection pending inode changed after fsync")
    guard_unchanged()

    input_inode = None
    result_inode = None
    durable = False
    try:
        os.link(pending_input, published_input)
        input_inode = identity(published_input)
        if input_inode != pending_input_inode:
            raise ValueError("published Pareto input does not alias its owned pending inode")
        value = json.loads(pending_result.read_text(encoding="utf-8"))
        validate_terminal_production_authority(value)
        os.link(pending_result, published_result)
        result_inode = identity(published_result)
        if result_inode != pending_result_inode:
            raise ValueError("published Pareto result does not alias its owned pending inode")
        if (
            published_input.read_bytes() != pending_input.read_bytes()
            or published_result.read_bytes() != pending_result.read_bytes()
        ):
            raise ValueError("terminal selection publication bytes changed")
        published_value = json.loads(published_result.read_text(encoding="utf-8"))
        validate_terminal_production_authority(published_value)
        if published_value != value:
            raise ValueError("published terminal selection differs from pending result")
        guard_unchanged()
        for parent in {published_input.parent.resolve(), published_result.parent.resolve()}:
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        guard_unchanged()
        durable = True
    finally:
        if not durable:
            remove_owned(published_result, result_inode)
            remove_owned(published_input, input_inode)

    remove_owned(pending_input, pending_input_inode)
    remove_owned(pending_result, pending_result_inode)
