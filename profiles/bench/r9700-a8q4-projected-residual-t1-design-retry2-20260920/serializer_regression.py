#!/usr/bin/env python3
"""Regression for the report serializer's embedded multiline compile receipts."""
from __future__ import annotations

import json
from pathlib import Path

from prepare import open_exclusive


TEXT = ('{\n  "argv": ["hipcc", "quote=\\\"", "path=\\\\tmp"],\r\n'
        '  "compiler": "clang\tversion"\n}\n'
        + "all JSON controls: " + "".join(chr(value) for value in range(32))
        + '\nUTF-8: café λ; quotes: "; backslash: \\; slash: /')


def regression_command(executable: Path, directory: Path) -> list[str]:
    fixture = directory / "serializer-input.txt"
    with open_exclusive(fixture) as stream:
        stream.write(TEXT)
    return [str(executable), "--serialize-json-text", str(fixture)]


def reject_constant(value: str):
    raise ValueError(f"non-JSON constant: {value}")


def check_output(payload: bytes) -> None:
    decoded = json.loads(payload.decode("utf-8"), strict=True,
                         parse_constant=reject_constant)
    if decoded != {"text": TEXT}:
        raise RuntimeError("report serializer did not preserve every input character")
