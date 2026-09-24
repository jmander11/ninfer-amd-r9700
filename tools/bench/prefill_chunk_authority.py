"""Stable inspection and create-only publication for prefill-chunk authorities."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def _identity(path: Path) -> tuple[int, int]:
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"prefill-chunk authority is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def validate_prefill_chunk_authority(
    path: Path, selected_chunk: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one semantic selection record while holding its original inode open."""

    source = path.expanduser()
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"prefill-chunk authority is not a regular file: {source}") from error
        raise ValueError(f"prefill-chunk authority cannot be opened safely: {source}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"prefill-chunk authority is not a regular file: {source}")
        owner = metadata.st_dev, metadata.st_ino
        before = _descriptor_sha256(descriptor)
        resolved = source.resolve(strict=True)
        from tools.bench.select_prefill_chunk import validate_selection_record
        selection = validate_selection_record(resolved)
        after = _descriptor_sha256(descriptor)
        if _identity(source) != owner or after != before:
            raise ValueError("prefill-chunk authority changed during validation")
    finally:
        os.close(descriptor)
    selected = selection.get("selected_prefill_chunk")
    if type(selected) is not int:
        raise ValueError("prefill-chunk authority lacks an integer selection")
    if selected_chunk is not None and selected != selected_chunk:
        raise ValueError(
            "prefill-chunk authority selection differs from the requested matrix chunk"
        )
    return ({
        "path": str(resolved),
        "sha256": before,
        "artifact_type": selection.get("artifact_type"),
        "schema_version": selection.get("schema_version"),
        "base_chunk_profile": selection.get("base_chunk_profile"),
        "selected_prefill_chunk": selected,
    }, selection)


def inspect_prefill_chunk_authority(path: Path, selected_chunk: int) -> dict[str, Any]:
    return validate_prefill_chunk_authority(path, selected_chunk)[0]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_create_json(path: Path, payload: object) -> None:
    """Publish JSON atomically without overwriting any existing directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.pending-", dir=path.parent)
    temporary = Path(temporary_name)
    pending_owner: tuple[int, int] | None = None
    linked = False
    committed = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            metadata = os.fstat(output.fileno())
            pending_owner = metadata.st_dev, metadata.st_ino
            output.write(json.dumps(payload, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if _identity(temporary) != pending_owner:
            raise ValueError(f"pending prefill-chunk selection changed: {path}")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"prefill-chunk selection namespace already exists: {path}") from error
        linked = True
        if _identity(path) != pending_owner:
            raise ValueError(f"published prefill-chunk selection inode differs: {path}")
        _fsync_directory(path.parent)
        committed = True
    finally:
        if linked and not committed and pending_owner is not None:
            try:
                if _identity(path) == pending_owner:
                    path.unlink()
            except FileNotFoundError:
                pass
        temporary.unlink(missing_ok=True)
        if linked or pending_owner is not None:
            _fsync_directory(path.parent)
