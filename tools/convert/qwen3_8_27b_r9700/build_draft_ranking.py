#!/usr/bin/env python3
"""Build the Qwen3.8 draft-head frequency row from validated token corpora."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence


MODEL_ID = "qwen3.8-27b"
TOKENIZER_MODEL_ID = "Qwen/Qwen3.8-27B"
VOCAB_ROWS = 248320
TOKENIZER_ID_COUNT = 248077
I64_MAX = (1 << 63) - 1
RANKING_BYTES = VOCAB_ROWS * 8


@dataclass(frozen=True, slots=True)
class CorpusCounts:
    ids_path: Path
    manifest_path: Path
    ids_sha256: str
    manifest_sha256: str
    token_count: int
    source: str
    counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RankingProvenance:
    ranking_path: Path
    sidecar_path: Path
    ranking_sha256: str
    sidecar_sha256: str
    total_tokens: int
    distinct_token_ids: int
    corpora: tuple[CorpusCounts, ...]


def manifest_path_for(ids_path: Path) -> Path:
    if ids_path.suffix != ".ids":
        raise ValueError(f"corpus must have an .ids suffix: {ids_path}")
    return ids_path.with_name(ids_path.stem + ".manifest.json")


def _required_count(manifest: dict[str, object], path: Path) -> int:
    declared: list[int] = []
    for key in ("tokens", "token_count"):
        if key not in manifest:
            continue
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer")
        declared.append(value)
    if not declared:
        raise ValueError(f"{path}: manifest must declare tokens or token_count")
    if any(value != declared[0] for value in declared[1:]):
        raise ValueError(f"{path}: tokens and token_count disagree")
    return declared[0]


def _tiling_warning(manifest: dict[str, object]) -> str | None:
    for key in ("tiled", "is_tiled", "repeated", "throughput_padding"):
        if manifest.get(key) not in (None, False, 0, "", "none", "false"):
            return key
    if manifest.get("tiling") not in (None, False, 0, "", "none", "false"):
        return "tiling"
    warning = " ".join(
        str(manifest.get(key, "")) for key in ("note", "notes", "description")
    ).lower()
    if re.search(r"\b(tiled?|tiling|repeat(?:ed|s|ing|ition)?)\b", warning):
        return "manifest note"
    return None


def _load_manifest(path: Path) -> tuple[dict[str, object], str]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"manifest not found: {path}") from error
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON manifest: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest root must be an object")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_manifest(manifest: dict[str, object], path: Path) -> tuple[int, str]:
    tiled = _tiling_warning(manifest)
    if tiled is not None:
        raise ValueError(
            f"{path}: {tiled} marks repeated/tiled throughput padding; "
            "including it would bias draft frequency counts"
        )
    required = {
        "artifact_type": "ninfer_ppl_corpus",
        "schema_version": 2,
        "model_id": MODEL_ID,
        "add_special_tokens": False,
        "chat_template": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{path}: {key} must be {expected!r}, got {manifest.get(key)!r}"
            )
    tokenizer_model_id = manifest.get("tokenizer_model_id")
    if tokenizer_model_id is not None and tokenizer_model_id != TOKENIZER_MODEL_ID:
        raise ValueError(
            f"{path}: tokenizer_model_id must be {TOKENIZER_MODEL_ID!r}, "
            f"got {tokenizer_model_id!r}"
        )
    digest = manifest.get("ids_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{path}: ids_sha256 must be 64 lowercase hexadecimal digits")
    return _required_count(manifest, path), digest


def _count_ids(path: Path) -> tuple[list[int], int, str]:
    counts = [0] * VOCAB_ROWS
    digest = hashlib.sha256()
    token_count = 0
    current: int | None = None
    offset = 0

    def finish_token() -> None:
        nonlocal current, token_count
        if current is None:
            return
        counts[current] += 1
        token_count += 1
        current = None

    try:
        source = path.open("rb")
    except FileNotFoundError as error:
        raise ValueError(f"corpus not found: {path}") from error
    with source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
            for byte in chunk:
                if 48 <= byte <= 57:
                    current = (0 if current is None else current) * 10 + byte - 48
                    if current >= TOKENIZER_ID_COUNT:
                        raise ValueError(
                            f"{path}: token ID at byte {offset} is outside "
                            f"0..{TOKENIZER_ID_COUNT - 1}"
                        )
                elif byte in b" \t\r\n\v\f":
                    finish_token()
                else:
                    raise ValueError(
                        f"{path}: byte {offset} is not an ASCII decimal digit or whitespace"
                    )
                offset += 1
    finish_token()
    return counts, token_count, digest.hexdigest()


def load_corpus(ids_path: str | Path) -> CorpusCounts:
    ids = Path(ids_path)
    manifest_path = manifest_path_for(ids)
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    expected_count, expected_digest = _validate_manifest(manifest, manifest_path)
    counts, token_count, ids_sha256 = _count_ids(ids)
    if ids_sha256 != expected_digest:
        raise ValueError(
            f"{ids}: SHA-256 {ids_sha256} does not match manifest {expected_digest}"
        )
    if token_count != expected_count:
        raise ValueError(
            f"{ids}: parsed {token_count} tokens, manifest declares {expected_count}"
        )
    return CorpusCounts(
        ids_path=ids.resolve(),
        manifest_path=manifest_path.resolve(),
        ids_sha256=ids_sha256,
        manifest_sha256=manifest_sha256,
        token_count=token_count,
        source=str(manifest.get("source", "unspecified")),
        counts=tuple(counts),
    )


def combine_corpora(corpora: Sequence[CorpusCounts]) -> list[int]:
    if not corpora:
        raise ValueError("at least one --corpus is required")
    paths = [corpus.ids_path for corpus in corpora]
    if len(paths) != len(set(paths)):
        raise ValueError("the same corpus must not be supplied more than once")
    totals = [0] * VOCAB_ROWS
    for corpus in corpora:
        for token_id, count in enumerate(corpus.counts):
            total = totals[token_id] + count
            if total > I64_MAX:
                raise OverflowError(f"token {token_id} exceeds signed I64 frequency range")
            totals[token_id] = total
    return totals


def provenance_path_for(ranking_path: str | Path) -> Path:
    return Path(str(Path(ranking_path)) + ".provenance.json")


def _required_report_int(report: dict[str, object], key: str, path: Path) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: {key} must be a non-negative integer")
    return value


def validate_ranking_provenance(
    ranking_path: str | Path,
) -> RankingProvenance:
    """Re-derive and validate a ranking before artifact conversion.

    The sidecar is not treated as an assertion to trust: every named corpus is
    revalidated against its own manifest, then counted again to prove the row.
    """

    ranking = Path(ranking_path).resolve()
    sidecar = provenance_path_for(ranking).resolve()
    try:
        ranking_payload = ranking.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"ranking not found: {ranking}") from error
    if len(ranking_payload) != RANKING_BYTES:
        raise ValueError(
            f"{ranking}: expected exactly one {RANKING_BYTES}-byte I64 row, "
            f"got {len(ranking_payload)} bytes"
        )
    ranking_sha256 = hashlib.sha256(ranking_payload).hexdigest()
    report, sidecar_sha256 = _load_manifest(sidecar)
    expected_scalars: dict[str, object] = {
        "artifact_type": "ninfer_qwen3_8_draft_ranking",
        "schema_version": 1,
        "model_id": MODEL_ID,
        "vocab_rows": VOCAB_ROWS,
        "tokenizer_id_count": TOKENIZER_ID_COUNT,
        "rows": 1,
        "dtype": "little-endian signed I64",
        "ranking_bytes": RANKING_BYTES,
        "ranking_sha256": ranking_sha256,
    }
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            raise ValueError(
                f"{sidecar}: {key} must be {expected!r}, got {report.get(key)!r}"
            )

    entries = report.get("corpora")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{sidecar}: corpora must be a non-empty array")
    corpora: list[CorpusCounts] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{sidecar}: corpora[{index}] must be an object")
        ids_value = entry.get("ids_path")
        manifest_value = entry.get("manifest_path")
        if not isinstance(ids_value, str) or not isinstance(manifest_value, str):
            raise ValueError(
                f"{sidecar}: corpora[{index}] paths must be strings"
            )
        corpus = load_corpus(ids_value)
        expected_entry: dict[str, object] = {
            "ids_path": str(corpus.ids_path),
            "manifest_path": str(corpus.manifest_path),
            "ids_sha256": corpus.ids_sha256,
            "manifest_sha256": corpus.manifest_sha256,
            "tokens": corpus.token_count,
            "source": corpus.source,
        }
        for key, expected in expected_entry.items():
            actual = entry.get(key)
            if actual != expected:
                raise ValueError(
                    f"{sidecar}: corpora[{index}].{key} must be "
                    f"{expected!r}, got {actual!r}"
                )
        if Path(manifest_value).resolve() != corpus.manifest_path:
            raise ValueError(
                f"{sidecar}: corpora[{index}].manifest_path is not the sibling manifest"
            )
        corpora.append(corpus)

    totals = combine_corpora(corpora)
    expected_payload = _ranking_payload(totals)
    if ranking_payload != expected_payload:
        raise ValueError(f"{ranking}: frequency row does not match the named corpora")
    total_tokens = sum(corpus.token_count for corpus in corpora)
    distinct = sum(count != 0 for count in totals)
    if _required_report_int(report, "total_tokens", sidecar) != total_tokens:
        raise ValueError(f"{sidecar}: total_tokens does not match the named corpora")
    if _required_report_int(report, "distinct_token_ids", sidecar) != distinct:
        raise ValueError(f"{sidecar}: distinct_token_ids does not match the ranking")
    return RankingProvenance(
        ranking_path=ranking,
        sidecar_path=sidecar,
        ranking_sha256=ranking_sha256,
        sidecar_sha256=sidecar_sha256,
        total_tokens=total_tokens,
        distinct_token_ids=distinct,
        corpora=tuple(corpora),
    )


def _ranking_payload(counts: Sequence[int]) -> bytes:
    if len(counts) != VOCAB_ROWS:
        raise ValueError(f"frequency row has {len(counts)} entries, expected {VOCAB_ROWS}")
    words = array("q", counts)
    if words.itemsize != 8:
        raise RuntimeError("host signed long long is not 64 bits")
    if sys.byteorder != "little":
        words.byteswap()
    payload = words.tobytes()
    if len(payload) != RANKING_BYTES:
        raise RuntimeError("encoded ranking is not exactly one I64 vocabulary row")
    return payload


def build_ranking(
    corpus_paths: Sequence[str | Path], out_path: str | Path,
) -> tuple[Path, Path]:
    output = Path(out_path)
    sidecar = provenance_path_for(output)
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    corpora = [load_corpus(path) for path in corpus_paths]
    totals = combine_corpora(corpora)
    payload = _ranking_payload(totals)
    ranking_sha256 = hashlib.sha256(payload).hexdigest()
    total_tokens = sum(corpus.token_count for corpus in corpora)
    if sum(totals) != total_tokens:
        raise RuntimeError("combined frequency sum differs from validated token count")

    report = {
        "artifact_type": "ninfer_qwen3_8_draft_ranking",
        "schema_version": 1,
        "model_id": MODEL_ID,
        "vocab_rows": VOCAB_ROWS,
        "tokenizer_id_count": TOKENIZER_ID_COUNT,
        "rows": 1,
        "dtype": "little-endian signed I64",
        "ranking_bytes": len(payload),
        "ranking_sha256": ranking_sha256,
        "total_tokens": total_tokens,
        "distinct_token_ids": sum(count != 0 for count in totals),
        "combination": "unweighted token-frequency sum; each validated non-tiled corpus once",
        "special_ids": (
            "not forced by this builder; tokenizer special-ID inclusion remains converter-owned"
        ),
        "corpora": [
            {
                "ids_path": str(corpus.ids_path),
                "manifest_path": str(corpus.manifest_path),
                "ids_sha256": corpus.ids_sha256,
                "manifest_sha256": corpus.manifest_sha256,
                "tokens": corpus.token_count,
                "source": corpus.source,
            }
            for corpus in corpora
        ],
    }
    sidecar_payload = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    try:
        for destination, data in ((output, payload), (sidecar, sidecar_payload)):
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=destination.name + ".", delete=False
            ) as handle:
                handle.write(data)
                temporary.append(Path(handle.name))
        temporary[0].replace(output)
        temporary[1].replace(sidecar)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return output, sidecar


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", action="append", required=True, type=Path,
        help="Qwen3.8 .ids corpus; its sibling <stem>.manifest.json is mandatory",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output, sidecar = build_ranking(args.corpus, args.out)
    except (FileExistsError, OSError, OverflowError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {output} and {sidecar}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
