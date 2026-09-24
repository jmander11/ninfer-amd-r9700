#!/usr/bin/env python3
"""Safely extract a HIP fatbin or one symbol-selected inner gfx1201 device ELF."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_OBJCOPY = Path("/opt/rocm/llvm/bin/llvm-objcopy")
OFFLOAD_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def _snapshot(path: Path, label: str) -> tuple[Path, int, str, int, int, int, int]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    with resolved.open("rb") as source:
        before = os.fstat(source.fileno())
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        after = os.fstat(source.fileno())
    current = resolved.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode,
                              value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError(f"{label} changed while being snapshotted")
    return (resolved, after.st_size, digest, after.st_dev, after.st_ino,
            after.st_mode, after.st_mtime_ns)


def _require_private_regular(path: Path, source: Path, label: str) -> None:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label} is not a private regular file")
    if os.path.samefile(path, source):
        raise ValueError(f"{label} aliases the input executable")


def _publish(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _select_device_code_object(fatbin: bytes, code_symbol: str) -> tuple[bytes, str, int]:
    if not code_symbol or "\x00" in code_symbol:
        raise ValueError("code symbol must be a nonempty text symbol")
    candidates: list[tuple[bytes, str, int]] = []
    position = 0
    while True:
        position = fatbin.find(OFFLOAD_MAGIC, position)
        if position < 0:
            break
        try:
            count = struct.unpack_from("<Q", fatbin, position + len(OFFLOAD_MAGIC))[0]
            if not 1 <= count <= 1024:
                raise ValueError
            cursor = position + len(OFFLOAD_MAGIC) + 8
            entries = []
            for _ in range(count):
                offset, size, identifier_size = struct.unpack_from("<QQQ", fatbin, cursor)
                cursor += 24
                if identifier_size > 4096 or cursor + identifier_size > len(fatbin):
                    raise ValueError
                identifier = fatbin[cursor:cursor + identifier_size].decode("utf-8")
                cursor += identifier_size
                start, end = position + offset, position + offset + size
                # Clang represents the host member as a legal zero-sized entry. Device
                # candidates still have to begin with ELF below; only malformed extents fail here.
                if start < cursor or end < start or end > len(fatbin):
                    raise ValueError
                entries.append((identifier, start, end))
            ordered_extents = sorted((start, end) for _, start, end in entries)
            if any(left[1] > right[0] for left, right in zip(
                ordered_extents, ordered_extents[1:]
            )):
                raise ValueError
        except (UnicodeDecodeError, ValueError, struct.error):
            position += len(OFFLOAD_MAGIC)
            continue
        for identifier, start, end in entries:
            payload = fatbin[start:end]
            if (
                "amdgcn-amd-amdhsa" in identifier
                and identifier.rsplit("--", 1)[-1].split(":", 1)[0] == "gfx1201"
                and payload.startswith(b"\x7fELF")
                and code_symbol.encode("utf-8") in payload
            ):
                candidates.append((payload, identifier, start))
        position += len(OFFLOAD_MAGIC)
    if len(candidates) != 1:
        raise ValueError(
            f"HIP fatbin must contain exactly one gfx1201 code object with {code_symbol}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def extract(
    executable: Path,
    output: Path,
    *,
    objcopy: Path = DEFAULT_OBJCOPY,
    code_symbol: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, str | int | None]:
    before = _snapshot(executable, "input executable")
    source = before[0]
    output_absolute = output.expanduser().absolute()
    if os.path.lexists(output_absolute):
        try:
            aliases_source = os.path.samefile(source, output_absolute)
        except OSError:
            aliases_source = False
        if aliases_source:
            raise ValueError("output and input executable must not be the same inode")
        raise ValueError(f"refusing to overwrite existing output: {output_absolute}")
    if output_absolute.resolve(strict=False) == source:
        raise ValueError("output and input executable must be distinct paths")
    objcopy_snapshot = _snapshot(objcopy, "llvm-objcopy")
    selected_objcopy = objcopy_snapshot[0]

    output_absolute.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-hip-fatbin.",
                                     dir=output_absolute.parent) as directory:
        temporary_root = Path(directory)
        section = temporary_root / "embedded.hip_fatbin"
        rewritten_elf = temporary_root / "discarded-output.elf"
        command = [str(selected_objcopy), "--dump-section", f".hip_fatbin={section}",
                   str(source), str(rewritten_elf)]
        try:
            runner(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"llvm-objcopy extraction failed: {detail}") from error
        if not rewritten_elf.exists():
            raise ValueError("llvm-objcopy did not create the required distinct output ELF")
        _require_private_regular(rewritten_elf, source, "llvm-objcopy output ELF")
        if not section.exists():
            raise ValueError("input executable has no nonempty .hip_fatbin section")
        _require_private_regular(section, source, "extracted .hip_fatbin")
        if os.path.samefile(section, rewritten_elf):
            raise ValueError("extracted .hip_fatbin aliases the llvm-objcopy output ELF")
        if section.stat().st_size == 0:
            raise ValueError("input executable has no nonempty .hip_fatbin section")
        payload = section.read_bytes()

    bundle_target = None
    bundle_offset = None
    if code_symbol is not None:
        payload, bundle_target, bundle_offset = _select_device_code_object(payload, code_symbol)

    after = _snapshot(source, "input executable after extraction")
    if after != before:
        raise ValueError("input executable changed during code-object extraction")
    if source.read_bytes().count(payload) != 1:
        raise ValueError("extracted HIP fatbin does not occur exactly once in the input executable")
    if _snapshot(source, "input executable before publication") != before:
        raise ValueError("input executable changed before code-object publication")
    if _snapshot(selected_objcopy, "llvm-objcopy after extraction") != objcopy_snapshot:
        raise ValueError("llvm-objcopy changed during code-object extraction")
    _publish(output_absolute, payload)
    return {
        "executable_path": str(source),
        "executable_size_bytes": before[1],
        "executable_sha256": before[2],
        "executable_device": before[3],
        "executable_inode": before[4],
        "code_object_path": str(output_absolute.resolve(strict=True)),
        "code_object_size_bytes": len(payload),
        "code_object_sha256": hashlib.sha256(payload).hexdigest(),
        "section": ".hip_fatbin",
        "bundle_target": bundle_target,
        "bundle_offset_bytes": bundle_offset,
        "code_symbol": code_symbol,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--objcopy", type=Path, default=DEFAULT_OBJCOPY)
    parser.add_argument(
        "--code-symbol",
        help="publish the unique gfx1201 inner ELF containing this exact code symbol",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = extract(
            args.executable, args.out, objcopy=args.objcopy, code_symbol=args.code_symbol,
        )
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"executable sha256: {result['executable_sha256']}")
    print(f"code object sha256: {result['code_object_sha256']}")
    kind = "gfx1201 device code object" if args.code_symbol else "embedded HIP fatbin"
    print(f"wrote {kind} to {result['code_object_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
