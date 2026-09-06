#!/usr/bin/env python3
"""Fail-closed filesystem boundaries for terminal quality recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.reference.qwen3_8_27b_bf16.protocol import validate_checkpoint_files


EXPECTED_AUTHORITIES = {
    "ALL_Q4_DENSE_QUALITY": ("r9700-q4g64-n16k16-eval", "dense"),
    "ALL_Q4_XATTENTION_QUALITY": ("r9700-q4g64-n16k16-eval", "b128-s16-tau900"),
    "MIXED_DENSE_QUALITY": ("r9700-q4-w8-mse-n16k16-eval", "dense"),
    "MIXED_XATTENTION_QUALITY": (
        "r9700-q4-w8-mse-n16k16-eval", "b128-s16-tau900",
    ),
    "FOUR_ROLE_DENSE_QUALITY": (
        "r9700-q4g64-f8e4m3-four-role-n16k16-eval", "dense",
    ),
    "FOUR_ROLE_XATTENTION_QUALITY": (
        "r9700-q4g64-f8e4m3-four-role-n16k16-eval", "b128-s16-tau900",
    ),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_authority_map(path: Path) -> dict[str, Any]:
    """Reopen the complete six-authority map and its exact N16 campaign identities."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("quality authority map is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("quality authority map is not valid JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {
            "artifact_type", "schema_version", "selected_prefill_chunk",
            "selected_prefill_chunk_authority", "concurrency", "authorities",
        }
        or value.get("artifact_type") != "ninfer_r9700_terminal_quality_authority_map"
        or value.get("schema_version") != 2
        or value.get("selected_prefill_chunk") not in (1024, 2048, 4096, 8192)
        or value.get("concurrency") != 1
        or not isinstance(value.get("authorities"), dict)
        or set(value["authorities"]) != set(EXPECTED_AUTHORITIES)
    ):
        raise ValueError("quality authority map schema is invalid")

    from tools.bench.prefill_chunk_authority import validate_prefill_chunk_authority
    chunk = value["selected_prefill_chunk"]
    selected = value.get("selected_prefill_chunk_authority")
    if not isinstance(selected, dict) or set(selected) != {"path", "sha256"}:
        raise ValueError("quality authority map chunk binding is invalid")
    selected_path = Path(selected.get("path", ""))
    summary, _ = validate_prefill_chunk_authority(selected_path, chunk)
    if selected != {"path": summary["path"], "sha256": summary["sha256"]}:
        raise ValueError("quality authority map chunk binding changed")

    from tools.ppl.assemble_pareto import _campaign_quality_candidate
    for name, (weights_id, profile) in EXPECTED_AUTHORITIES.items():
        entry = value["authorities"][name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "artifact"}:
            raise ValueError(f"quality authority {name} schema is invalid")
        campaign_path = Path(entry.get("path", ""))
        if (
            not campaign_path.is_absolute() or campaign_path.is_symlink()
            or not campaign_path.is_file() or str(campaign_path.resolve()) != str(campaign_path)
            or entry.get("sha256") != _sha256(campaign_path)
        ):
            raise ValueError(f"quality authority {name} campaign changed")
        try:
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"quality authority {name} campaign is invalid JSON") from error
        sources = []
        for group in (16, 32):
            _, source = _campaign_quality_candidate(campaign, weights_id, group, chunk)
            sources.append(source)
        identities = [{
            "weights_id": source.get("weights_id"),
            "sha256": source.get("sha256"),
            "file_size_bytes": source.get("file_size_bytes"),
            "conversion_receipt": source.get("conversion_receipt"),
        } for source in sources]
        identity = identities[0]
        if (
            identities[0] != identities[1]
            or sources[0].get("representation", {}).get("xattention_profile") != profile
            or not isinstance(identity["conversion_receipt"], dict)
            or entry.get("artifact") != identity
        ):
            raise ValueError(f"quality authority {name} N16 artifact binding changed")
    return value


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
    validate_authority_map(pending)
    os.link(pending, final)
    try:
        validate_authority_map(final)
        descriptor = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if pending.samefile(final):
            final.unlink()
        raise
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
