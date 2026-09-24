#!/usr/bin/env python3
"""Fail fast unless a benchmark artifact has the exact N16/K16 Q4 identity."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Sequence

from tools.ppl.run import inspect_candidate_artifact


MAGIC = b"NINFER\x00\x02"
PREFIX = struct.Struct("<8sQ")
MODEL_ID = "qwen3.8-27b"
Q4_FORMAT = "Q4G64_F16S"
Q4_LAYOUT = "r9700-q4g64-n16-k16-v1"
WEIGHTS_IDS = (
    "r9700-q4g64-n16k16-eval",
    "r9700-q4-w8-mse-n16k16-eval",
    "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
)


def validate(path: Path, weights_id: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required N16/K16 artifact is unavailable as a regular file: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(PREFIX.size)
        if len(prefix) != PREFIX.size:
            raise ValueError(f"artifact has a truncated prefix: {path}")
        magic, directory_bytes = PREFIX.unpack(prefix)
        if magic != MAGIC or not 0 < directory_bytes <= 64 * 1024 * 1024:
            raise ValueError(f"artifact has an invalid NInfer v2 prefix: {path}")
        try:
            directory = json.loads(stream.read(directory_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"artifact directory is invalid: {path}") from error
    if directory.get("identity") != {"model_id": MODEL_ID, "weights_id": weights_id}:
        raise ValueError(f"artifact identity is not {MODEL_ID}/{weights_id}: {path}")
    objects = directory.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"artifact object inventory is invalid: {path}")
    q4 = [item for item in objects if isinstance(item, dict) and item.get("format") == Q4_FORMAT]
    if not q4 or any(item.get("layout") != Q4_LAYOUT for item in q4):
        raise ValueError(f"artifact does not use only {Q4_LAYOUT} for Q4G64: {path}")
    try:
        artifact = inspect_candidate_artifact(path)
    except SystemExit as error:
        raise ValueError(str(error)) from error
    if (artifact.get("weights_id") != weights_id
            or not isinstance(artifact.get("conversion_receipt"), dict)):
        raise ValueError(f"artifact lacks its exact N16 migration receipt: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--weights-id", choices=WEIGHTS_IDS, required=True)
    args = parser.parse_args(argv)
    try:
        validate(args.artifact, args.weights_id)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
