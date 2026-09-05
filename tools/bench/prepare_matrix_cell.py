#!/usr/bin/env python3
"""Create one prepared matrix cell, or exactly validate an existing one.

This is intentionally a preparation-only wrapper.  The wrapped command must
contain one ``--output-dir`` argument and is responsible for creating a fresh
prepared matrix.  When that output already exists, the wrapper runs the same
command against a private temporary root and compares the complete prepared
tree after translating only that root and ignoring the inherently variable
manifest creation timestamp.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Sequence


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _output_argument(command: Sequence[str]) -> tuple[int, Path]:
    positions = [index for index, value in enumerate(command) if value == "--output-dir"]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError("prepared matrix command must contain exactly one --output-dir value")
    index = positions[0] + 1
    return index, _lexical_absolute(Path(command[index]))


def _tree(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in directory.iterdir():
            mode = entry.lstat().st_mode
            relative = str(entry.relative_to(root))
            if stat.S_ISDIR(mode):
                result[relative] = "directory"
                pending.append(entry)
            elif stat.S_ISREG(mode):
                result[relative] = "file"
            else:
                raise ValueError(f"prepared matrix contains a nonregular entry: {entry}")
    return result


def _translated(value, source: str, destination: str):
    if isinstance(value, str):
        return value.replace(source, destination)
    if isinstance(value, list):
        return [_translated(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {key: _translated(item, source, destination) for key, item in value.items()}
    return value


def validate_existing(actual: Path, expected: Path) -> None:
    actual = _lexical_absolute(actual)
    expected = _lexical_absolute(expected)
    for path, label in ((actual, "existing"), (expected, "derived")):
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError(f"{label} prepared matrix root is not a real directory: {path}")
    actual_tree = _tree(actual)
    expected_tree = _tree(expected)
    if actual_tree != expected_tree:
        raise ValueError("existing prepared matrix has partial or unexpected filesystem entries")
    if actual_tree.get("manifest.json") != "file" or actual_tree.get("commands.sh") != "file":
        raise ValueError("derived prepared matrix lacks its manifest or commands")

    actual_manifest = json.loads((actual / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = json.loads((expected / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(actual_manifest, dict) or not isinstance(expected_manifest, dict):
        raise ValueError("prepared matrix manifest is not an object")
    if (not isinstance(actual_manifest.get("created_at_utc"), str)
            or not actual_manifest["created_at_utc"]
            or not isinstance(expected_manifest.get("created_at_utc"), str)
            or not expected_manifest["created_at_utc"]):
        raise ValueError("prepared matrix manifest lacks its creation timestamp")
    actual_manifest.pop("created_at_utc", None)
    expected_manifest.pop("created_at_utc", None)
    expected_manifest = _translated(expected_manifest, str(expected), str(actual))
    if actual_manifest != expected_manifest:
        raise ValueError("existing prepared matrix manifest differs from the derived contract")

    expected_commands = (expected / "commands.sh").read_text(encoding="utf-8").replace(
        str(expected), str(actual)
    )
    if (actual / "commands.sh").read_text(encoding="utf-8") != expected_commands:
        raise ValueError("existing prepared matrix commands differ from the derived contract")


def prepare_or_validate(output: Path, command: Sequence[str]) -> str:
    if not command:
        raise ValueError("prepared matrix command is empty")
    output = _lexical_absolute(output)
    output_index, command_output = _output_argument(command)
    if command_output != output:
        raise ValueError("wrapper output and command --output-dir differ")
    if os.path.lexists(output):
        if not stat.S_ISDIR(output.lstat().st_mode):
            raise ValueError(f"existing prepared matrix root is not a real directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.derive.", dir=output.parent))
        expected = temporary / "expected"
        derived = list(command)
        derived[output_index] = str(expected)
        try:
            subprocess.run(derived, check=True)
            validate_existing(output, expected)
        finally:
            shutil.rmtree(temporary)
        return "validated"
    subprocess.run(list(command), check=True)
    if not os.path.lexists(output) or not stat.S_ISDIR(output.lstat().st_mode):
        raise ValueError("prepare command did not create the expected real output directory")
    return "created"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        disposition = prepare_or_validate(args.output_dir, command)
    except (json.JSONDecodeError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"prepare-matrix-cell: {error}", file=__import__("sys").stderr)
        return 2
    print(f"{disposition} prepared matrix: {_lexical_absolute(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
