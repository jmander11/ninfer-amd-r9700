#!/usr/bin/env python3
"""Run the native NInfer product performance matrix.

The matrix is intentionally layered instead of fully factorial:

* k=3 is the primary MTP path to evaluate.
* k=0 and k=5 are baseline/max-window controls.
* k=0..5 is swept on representative context-decode cases.
* Device Graph is compared only for decode-bearing tests.
* Prefill-only tests sweep length and chunk size, but not graph on/off.
* The concurrency preset decomposes primary MTP3 prefill and decode at C=1..4.
* The DFlash shortlist preset compares explicit K=1..11/W/topology profiles at 8K.
* The DFlash Pareto preset binds one explicit K/W profile to matched speed,
  acceptance, proposal/selector diagnostics, and exact ordinary-token parity.
* The Pareto feasibility preset proves the required 32K workload fits at every
  fixed concurrency and identifies whether memory or the logical context ceiling bounded it.
* The Pareto capacity preset resolves the exact effective maximum at the model-native
  per-request context ceiling.
* The Pareto whole preset measures fresh-prompt makespan and retains the separately
  timed prefill/decode phases from those same repetitions. It is the complete speed
  authority for static-profile selection; the Pareto phase preset is diagnostic only.

Raw ninfer_bench reports stay under profiles/bench. This script writes a
descriptive manifest, exact commands, per-case logs, raw JSON reports, and a flat
summary CSV/JSON that is easy to compare across runs.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import shlex
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.matrix_contract import (
    MATRIX_SCHEMA_VERSION,
    PRODUCT_CONCURRENCIES,
    PRODUCTION_PREFILL_CHUNKS,
    R9700_KV_PLANE_LAYOUTS,
)
from tools.bench.prefill_chunk_authority import inspect_prefill_chunk_authority
from tools.ppl import run as ppl_run
from tools.ppl.validate_fp8_hybrid_execution_gate import query_widths

DEFAULT_BENCH = REPO_ROOT / "build-r9700/bench/ninfer_bench"
DEFAULT_CORPUS = REPO_ROOT / "bench/fixtures/bench_corpus.ids"

PREFILL_LENGTHS_CORE = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PREFILL_LENGTHS_FULL_EXTRA = (32768, 65536)
PREFILL_CHUNKS = (128, 256, 512, 1024, 2048, 4096)
PRODUCTION_PREFILL_PROMPTS = (8192, 32768)
LOW_CONTEXT_PREFILL_PROMPTS = (128, 512, 1024, 2048, 4096)
R9700_POWER_PROFILE = Path(
    "/sys/class/drm/card2/device/power_dpm_force_performance_level"
)
PURE_DECODE_GENS = (16, 64, 128, 512, 2048)
CONTEXT_CORE = ((512, 512), (2048, 512), (8192, 512))
CONTEXT_FULL_EXTRA = ((32768, 256), (65536, 128))
PRIMARY_KS = (0, 3, 5)
SWEEP_KS = (0, 1, 2, 3, 4, 5)
REPORT_SCHEMA_VERSION = 20
REPORT_ARTIFACT_TYPE = "ninfer_bench_report"
REPORT_TOOL = "ninfer_bench"
DFLASH_SHORTLIST_SCHEMA_VERSION = 1
DFLASH_SHORTLIST_K_ORDER = (1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6)
MODEL_ID = "qwen3.8-27b"
TARGET_ID = "qwen3_8_27b_r9700"
FP8_QK_WMMA_PROFILE = "t1-ge64-t2-ge320-t3plus-stream-v1"
FP8_QK_WMMA_T1_MIN_CONTEXT = 64
FP8_QK_WMMA_T2_MIN_CONTEXT = 320
XATTENTION_PROFILES = ("dense", "b128-s16-tau900")
BENCHMARK_PENDING_TIMEOUT_MS = 0xFFFFFFFF
NINFER_MAGIC = b"NINFER\x00\x02"
NINFER_PREFIX = struct.Struct("<8sQ")
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
KV_PAGE_TOKENS = 64
AUTOMATIC_KV_HEADROOM_BYTES = 1024 * 1024 * 1024
MODEL_NATIVE_CONTEXT = 262144
PUBLIC_TOKEN_DOMAIN = 248077
DFLASH_CAMPAIGN_WEIGHTS_IDS = frozenset({
    "r9700-q4g64-n16k16-dflash2-q4-eval",
    "r9700-q4-w8-mse-n16k16-dflash2-q4-eval",
    "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval",
})
DFLASH_CAMPAIGN_PRESETS = frozenset({
    "dflash-shortlist", "dflash-pareto", "dflash-feasibility", "dflash-capacity",
})
HYBRID_DFLASH_PRESETS = frozenset({
    "dflash-shortlist", "dflash-pareto", "dflash-capacity",
})
HYBRID_BASE_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
HYBRID_DFLASH_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval"
DFLASH_RECIPE_ID = "r9700-dflash2-all-q4g64-n16k16-bf16-codebook-eval-v1"
DFLASH_SOURCE_RECEIPT = {
    "config_sha256": "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980",
    "readme_sha256": "0c06405ffff835f4da26115114a6dd7bb4a8b8a6881c17edd3a1086a99281269",
    "tensor_count": 81,
    "safetensors_bytes": 3_848_817_896,
    "safetensors_sha256": "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
}
PARETO_CAMPAIGN_PRESETS = frozenset({
    "pareto", "pareto-whole", "pareto-feasibility", "pareto-capacity",
})
SELECTED_PREFILL_CHUNK_PRESETS = (
    PARETO_CAMPAIGN_PRESETS | DFLASH_CAMPAIGN_PRESETS | {"low-context-prefill"}
)
POWER_BOUND_PRESETS = frozenset({
    "prefill-chunk", "low-context-prefill", "ordinary-diagnostic", "pareto-whole",
    "dflash-shortlist", "dflash-pareto",
})
POWER_RECHECK_PRESETS = frozenset({
    "ordinary-diagnostic", "pareto-whole", "prefill-chunk", "low-context-prefill",
    "dflash-shortlist", "dflash-pareto",
})
AT_FDCWD = -100
RENAME_EXCHANGE = 2


@dataclasses.dataclass(frozen=True)
class BenchCase:
    suite: str
    name: str
    args: tuple[str, ...]
    repetitions: int
    warmup: int
    notes: str = ""
    retain_token_ids: bool = False
    parity_role: str | None = None
    diagnostic: bool = False
    concurrency_one_only: bool = False


def csv_list(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def pair_list(values: Iterable[tuple[int, int]]) -> str:
    return ";".join(f"{p},{g}" for p, g in values)


def require_auto_power_profile(path: Path | None = None) -> str:
    """Fail closed before a chunk-speed campaign if the R9700 is not in auto mode."""

    profile_path = R9700_POWER_PROFILE if path is None else path
    try:
        observed = profile_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"cannot read R9700 power profile {profile_path}: {error}") from error
    if observed != "auto":
        raise ValueError(
            f"prefill-chunk requires stable R9700 power profile auto; observed {observed!r} "
            f"at {profile_path}"
        )
    return observed


def mtp_args(k: int) -> tuple[str, ...]:
    args = ("--spec", "mtp", "--draft-tokens", str(k))
    return (*args, "--lm-head-draft") if k > 0 else args


def dflash_args(k: int, verify_width: int) -> tuple[str, ...]:
    args = ("--spec", "dflash", "--draft-tokens", str(k), "--lm-head-draft")
    if verify_width:
        args = (*args, "--dflash-verify-width", str(verify_width))
    return args


def resolved_dflash_verify_width(draft_tokens: int, requested_width: int) -> int:
    if draft_tokens == 0:
        return 0
    if requested_width:
        return requested_width
    if draft_tokens <= 5 or draft_tokens >= 8:
        return draft_tokens + 1
    return 12


def resolved_dflash_topology(draft_tokens: int, verify_width: int) -> str:
    if not 1 <= draft_tokens <= 11:
        raise ValueError("DFlash topology requires K in [1, 11]")
    if not 2 <= verify_width <= 16:
        raise ValueError("DFlash topology requires W in [2, 16]")
    proposal = "single-block" if draft_tokens <= 7 else "two-block"
    verification = (
        "packed-tree" if draft_tokens <= 7 and verify_width != draft_tokens + 1 else "chain"
    )
    return f"{proposal}-{verification}"


def dflash_shortlist_profiles() -> list[dict[str, Any]]:
    profiles = []
    for k in range(1, 12):
        width = resolved_dflash_verify_width(k, 0)
        profiles.append({
            "draft_tokens_requested": k,
            "verify_width_requested": width,
            "verify_width_resolved": width,
            "topology": resolved_dflash_topology(k, width),
            "proposal_head": "optimized",
        })
    return profiles


def dflash_shortlist_frontier(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the multi-objective frontier without hiding speed/quality tradeoffs."""

    dimensions = (
        ("target_equivalent_generated_tok_s", True),
        ("accepted_tokens_per_round", True),
        ("fallback_rate_per_attempt", False),
        ("first_reject_rate_per_proposal_hop", False),
    )

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        no_worse = True
        strictly_better = False
        for key, maximize in dimensions:
            left_value = float(left[key])
            right_value = float(right[key])
            no_worse = no_worse and (
                left_value >= right_value if maximize else left_value <= right_value
            )
            strictly_better = strictly_better or (
                left_value > right_value if maximize else left_value < right_value
            )
        return no_worse and strictly_better

    eligible = [candidate for candidate in candidates if candidate.get("valid_for_ranking")]
    return [
        candidate
        for candidate in eligible
        if not any(
            other is not candidate and dominates(other, candidate) for other in eligible
        )
    ]


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def count_corpus_tokens(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"corpus file not found: {path}")
    return len(path.read_text(encoding="utf-8").split())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inode(path: Path) -> tuple[int, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"runner-owned output is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_owned(path: Path, owner: tuple[int, int]) -> bool:
    try:
        if _inode(path) != owner:
            return False
    except FileNotFoundError:
        return False
    path.unlink()
    return True


def remove_runner_owned_file(path: Path) -> None:
    try:
        owner = _inode(path)
    except FileNotFoundError:
        return
    if not _unlink_if_owned(path, owner):
        raise ValueError(f"runner-owned output changed before cleanup: {path}")
    _fsync_directory(path.parent)


def _rename_exchange(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(
        AT_FDCWD, os.fsencode(left), AT_FDCWD, os.fsencode(right), RENAME_EXCHANGE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left} <-> {right}")


def durable_replace_bytes(path: Path, payload: bytes) -> None:
    """Durably publish runner-owned bytes without clobbering a raced foreign inode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    previous_descriptor: int | None = None
    try:
        previous_descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        previous = None
    except OSError as error:
        raise ValueError(f"runner-owned output cannot be opened safely: {path}: {error}") from error
    else:
        metadata = os.fstat(previous_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(previous_descriptor)
            previous_descriptor = None
            raise ValueError(f"runner-owned output is not a regular file: {path}")
        previous = metadata.st_dev, metadata.st_ino
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.pending-", dir=path.parent
        )
    except Exception:
        if previous_descriptor is not None:
            os.close(previous_descriptor)
        raise
    temporary = Path(temporary_name)
    pending_owner: tuple[int, int] | None = None
    published = False
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as output:
            pending_owner = (os.fstat(output.fileno()).st_dev, os.fstat(output.fileno()).st_ino)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if _inode(temporary) != pending_owner:
            raise ValueError(f"runner-owned pending output changed before publication: {path}")
        if previous is None:
            os.link(temporary, path)
            if _inode(path) != pending_owner:
                raise ValueError(f"runner-owned publication inode differs: {path}")
            published = True
        else:
            _rename_exchange(temporary, path)
            published = True
            if _inode(path) != pending_owner or _inode(temporary) != previous:
                if _inode(path) == pending_owner:
                    _rename_exchange(temporary, path)
                    published = False
                raise ValueError(f"runner-owned output changed during publication: {path}")
        _fsync_directory(path.parent)
        committed = True
        if not _unlink_if_owned(temporary, previous if previous is not None else pending_owner):
            raise ValueError(f"runner-owned pending inode changed before cleanup: {path}")
        _fsync_directory(path.parent)
    except Exception:
        if published and not committed and pending_owner is not None:
            if previous is not None:
                try:
                    if _inode(path) == pending_owner and _inode(temporary) == previous:
                        _rename_exchange(temporary, path)
                        published = False
                except (FileNotFoundError, ValueError, OSError):
                    pass
            else:
                _unlink_if_owned(path, pending_owner)
        if pending_owner is not None:
            _unlink_if_owned(temporary, pending_owner)
        raise
    finally:
        if previous_descriptor is not None:
            os.close(previous_descriptor)


def durable_replace_text(path: Path, payload: str) -> None:
    durable_replace_bytes(path, payload.encode("utf-8"))


def durable_replace_json(path: Path, payload: object) -> None:
    durable_replace_text(path, json.dumps(payload, indent=2) + "\n")


def manifest_owned_path(root: Path, value: object, label: str) -> Path:
    """Resolve one manifest path without allowing it to escape its campaign directory."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"matrix manifest has an invalid {label} path")
    owner = root.expanduser().resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = owner / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(owner)
    except ValueError as error:
        raise ValueError(
            f"matrix manifest {label} path is outside its output directory: {value}"
        ) from error
    return resolved


def validate_manifest_output_ownership(root: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("commands")
    if not isinstance(records, list):
        return
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"matrix manifest has an invalid commands[{index}] record")
        report = manifest_owned_path(root, record.get("report"), f"commands[{index}].report")
        command = record.get("command")
        if not isinstance(command, list) or any(not isinstance(part, str) for part in command):
            raise ValueError(f"matrix manifest has an invalid commands[{index}].command")
        output_indices = [position for position, part in enumerate(command) if part == "--output-file"]
        if len(output_indices) != 1 or output_indices[0] + 1 >= len(command):
            raise ValueError(f"matrix manifest commands[{index}] has invalid --output-file")
        command_output = manifest_owned_path(
            root, command[output_indices[0] + 1], f"commands[{index}].command --output-file"
        )
        if command_output != report:
            raise ValueError(f"matrix manifest commands[{index}] output path differs from report")
        for field in ("raw_diagnostic", "bound_diagnostic"):
            value = record.get(field)
            if value is not None:
                manifest_owned_path(root, value, f"commands[{index}].{field}")
        environment = record.get("environment")
        if isinstance(environment, dict) and "NINFER_DFLASH_CANDIDATE_STATS_OUT" in environment:
            raw = manifest_owned_path(
                root, environment["NINFER_DFLASH_CANDIDATE_STATS_OUT"],
                f"commands[{index}].environment diagnostic output",
            )
            if raw != manifest_owned_path(
                root, record.get("raw_diagnostic"), f"commands[{index}].raw_diagnostic"
            ):
                raise ValueError(
                    f"matrix manifest commands[{index}] diagnostic output path differs"
                )


def inspect_executable(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"benchmark executable not found: {resolved}")
    return {
        "path": str(resolved),
        "file_size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def inspect_artifact(path: Path) -> dict[str, Any]:
    """Bind a benchmark campaign to exact NInfer bytes and their declared identity."""

    resolved = path.expanduser().resolve()
    with resolved.open("rb") as source:
        prefix = source.read(NINFER_PREFIX.size)
        if len(prefix) != NINFER_PREFIX.size:
            raise SystemExit(f"artifact has a truncated prefix: {resolved}")
        magic, directory_bytes = NINFER_PREFIX.unpack(prefix)
        if magic != NINFER_MAGIC:
            raise SystemExit(f"artifact is not NInfer v2: {resolved}")
        if directory_bytes == 0 or directory_bytes > MAX_DIRECTORY_BYTES:
            raise SystemExit(
                f"artifact has invalid directory size {directory_bytes}: {resolved}"
            )
        raw_directory = source.read(directory_bytes)
    try:
        directory = json.loads(raw_directory)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"artifact directory is invalid: {resolved}: {error}") from error
    identity = directory.get("identity") if isinstance(directory, dict) else None
    if not isinstance(identity, dict):
        raise SystemExit(f"artifact has no identity: {resolved}")
    model_id = identity.get("model_id")
    weights_id = identity.get("weights_id")
    if model_id != MODEL_ID or not isinstance(weights_id, str) or not weights_id:
        raise SystemExit(
            f"artifact identity must be {MODEL_ID}/<weights-id>, got "
            f"{model_id!r}/{weights_id!r}: {resolved}"
        )
    objects = directory.get("objects")
    if not isinstance(objects, list):
        raise SystemExit(f"artifact object inventory is invalid: {resolved}")
    for obj in objects:
        if (isinstance(obj, dict) and obj.get("format") == "Q4G64_F16S" and
                obj.get("layout") != "r9700-q4g64-n16-k16-v1"):
            raise SystemExit(
                f"artifact contains retired non-N16/K16 Q4 storage: {resolved}"
            )
    return {
        "path": str(resolved),
        "file_size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
        "model_id": model_id,
        "weights_id": weights_id,
    }


def _artifact_object_count(path: Path) -> int:
    with path.resolve().open("rb") as source:
        prefix = source.read(NINFER_PREFIX.size)
        if len(prefix) != NINFER_PREFIX.size:
            raise SystemExit(f"artifact has a truncated prefix: {path}")
        magic, directory_bytes = NINFER_PREFIX.unpack(prefix)
        if magic != NINFER_MAGIC or not 0 < directory_bytes <= MAX_DIRECTORY_BYTES:
            raise SystemExit(f"artifact has an invalid directory: {path}")
        try:
            directory = json.loads(source.read(directory_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"artifact directory is invalid: {path}: {error}") from error
    objects = directory.get("objects") if isinstance(directory, dict) else None
    if not isinstance(objects, list):
        raise SystemExit(f"artifact object inventory is invalid: {path}")
    return len(objects)


def _receipt_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label} lacks a path")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else REPO_ROOT / candidate).resolve(strict=True)


def _hybrid_base_authority(receipt: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "recipe_id", "selection_sha256", "object_plan_sha256",
        "source_index_sha256", "source_ranking_sha256",
    )
    if (
        not isinstance(receipt.get("path"), str)
        or not isinstance(receipt.get("sha256"), str)
        or any(not isinstance(receipt.get(field), str) for field in fields)
    ):
        raise SystemExit("hybrid base lacks its exact conversion authority")
    return {
        "receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
        **{field: receipt[field] for field in fields},
    }


def require_fp8_hybrid_dflash_companion(
    path: Path, artifact: dict[str, Any]
) -> dict[str, Any]:
    """Bind a four-role DFlash companion back to the exact selected hybrid base."""

    if artifact.get("weights_id") != HYBRID_DFLASH_WEIGHTS_ID:
        raise SystemExit("hybrid DFlash evidence requires the four-role DFlash companion")
    report_path = Path(str(path.resolve()) + ".conversion.json")
    if report_path.is_symlink() or not report_path.is_file():
        raise SystemExit("hybrid DFlash companion lacks a regular conversion report")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"hybrid DFlash conversion report is invalid: {error}") from error
    base = report.get("base") if isinstance(report, dict) else None
    converted = report.get("artifact") if isinstance(report, dict) else None
    recipe = report.get("dflash_recipe") if isinstance(report, dict) else None
    source = report.get("dflash_source") if isinstance(report, dict) else None
    identity = report.get("identity") if isinstance(report, dict) else None
    if not all(isinstance(value, dict) for value in (base, converted, recipe, source, identity)):
        raise SystemExit("hybrid DFlash conversion report is incomplete")
    base_path = _receipt_path(base.get("path"), "hybrid DFlash base")
    base_artifact = inspect_artifact(base_path)
    hybrid = ppl_run.inspect_candidate_artifact(base_path, digest=base_artifact["sha256"])
    ppl_run.require_fp8_hybrid_candidate(hybrid)
    if (
        hybrid["path"] != base_artifact["path"]
        or hybrid["bytes"] != base_artifact["file_size_bytes"]
        or hybrid["sha256"] != base_artifact["sha256"]
        or hybrid["weights_id"] != HYBRID_BASE_WEIGHTS_ID
    ):
        raise SystemExit("hybrid DFlash base inspection differs from benchmark provenance")
    base_artifact = {**base_artifact, "conversion_receipt": hybrid["conversion_receipt"]}
    expected_authority = _hybrid_base_authority(hybrid["conversion_receipt"])
    if (
        report.get("status") != "registered-evaluation-only"
        or report.get("target_key") != TARGET_ID
        or report.get("recipe_id") != DFLASH_RECIPE_ID
        or report.get("weight_recipe_selected") is not False
        or identity.get("model_id") != MODEL_ID
        or identity.get("weights_id") != artifact["weights_id"]
        or base.get("identity") != {
            "model_id": MODEL_ID, "weights_id": HYBRID_BASE_WEIGHTS_ID,
        }
        or base.get("bytes") != base_artifact["file_size_bytes"]
        or base.get("sha256") != base_artifact["sha256"]
        or base.get("payload_copy") != "byte_exact"
        or base.get("authority") != expected_authority
        or _receipt_path(converted.get("path"), "hybrid DFlash artifact") != path.resolve()
        or converted.get("bytes") != artifact["file_size_bytes"]
        or converted.get("sha256") != artifact["sha256"]
        or converted.get("projected_bytes") != artifact["file_size_bytes"]
        or type(converted.get("projected_device_arena_bytes")) is not int
        or converted["projected_device_arena_bytes"] <= 0
        or _artifact_object_count(path) != 1190
        or recipe.get("matrix_format") != "Q4G64_F16S"
        or recipe.get("activation_profile") != "compile_selected_adaptive_A8G64"
        or recipe.get("selector_codebook_format") != "BF16"
        or recipe.get("objects") != 66
        or recipe.get("source_tensors") != 81
        or recipe.get("format_counts") != {"BF16": 34, "Q4G64_F16S": 32}
        or recipe.get("format_encoded_bytes")
        != {"BF16": 254_814_720, "Q4G64_F16S": 954_654_720}
        or recipe.get("tensor_encoded_bytes") != 1_209_469_440
        or recipe.get("runtime_repack") is not False
        or any(source.get(key) != value for key, value in DFLASH_SOURCE_RECEIPT.items())
    ):
        raise SystemExit("hybrid DFlash companion does not bind the exact hybrid base")
    return {
        **artifact,
        "dflash_conversion_report": {
            "path": str(report_path), "sha256": file_sha256(report_path),
        },
        "hybrid_base_artifact": base_artifact,
    }


def require_fp8_hybrid_artifact(
    path: Path, artifact: dict[str, Any], preset: str | None = None,
) -> dict[str, Any]:
    """Attach the current authority-bound hybrid receipt to benchmark provenance."""

    if preset in HYBRID_DFLASH_PRESETS:
        return require_fp8_hybrid_dflash_companion(path, artifact)
    hybrid = ppl_run.inspect_candidate_artifact(path, digest=artifact["sha256"])
    ppl_run.require_fp8_hybrid_candidate(hybrid)
    if (
        hybrid["path"] != artifact["path"]
        or hybrid["bytes"] != artifact["file_size_bytes"]
        or hybrid["sha256"] != artifact["sha256"]
        or hybrid["model_id"] != artifact["model_id"]
        or hybrid["weights_id"] != artifact["weights_id"]
    ):
        raise SystemExit("hybrid artifact inspection differs from benchmark provenance")
    return {**artifact, "conversion_receipt": hybrid["conversion_receipt"]}


def bind_n16_migration_receipt(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Attach the exact migration authority for each selectable N16 base recipe."""
    if artifact.get("weights_id") not in ppl_run.N16_MIGRATION_PROFILES:
        return artifact
    inspected = ppl_run.inspect_candidate_artifact(path, digest=artifact["sha256"])
    if (inspected.get("path") != artifact.get("path")
            or inspected.get("bytes") != artifact.get("file_size_bytes")
            or any(inspected.get(key) != artifact.get(key) for key in (
                "sha256", "model_id", "weights_id",
            )) or not isinstance(inspected.get("conversion_receipt"), dict)):
        raise SystemExit("N16 migration receipt inspection differs from benchmark provenance")
    return {**artifact, "conversion_receipt": inspected["conversion_receipt"]}


def validate_fp8_hybrid_performance_contract(args: argparse.Namespace) -> None:
    if not args.require_fp8_hybrid:
        return
    allowed = {
        "prefill-chunk", "low-context-prefill", "pareto-whole", "pareto-capacity",
        *HYBRID_DFLASH_PRESETS,
    }
    if args.preset not in allowed:
        raise SystemExit(
            "--require-fp8-hybrid supports only prefill-chunk, low-context-prefill, "
            "pareto-whole, pareto-capacity, dflash-shortlist, dflash-pareto, or "
            "dflash-capacity"
        )
    if args.preset == "prefill-chunk":
        if args.concurrency != [1]:
            raise SystemExit("hybrid prefill-chunk evidence requires exactly C=1")
        if not args.prefill_chunk or any(
            chunk not in PRODUCTION_PREFILL_CHUNKS for chunk in args.prefill_chunk
        ):
            raise SystemExit("hybrid prefill-chunk evidence has an unsupported chunk")
    elif args.preset in ("low-context-prefill", "dflash-shortlist"):
        if args.concurrency != [1]:
            raise SystemExit(f"hybrid {args.preset} evidence requires exactly C=1")
        if (
            not args.prefill_chunk
            or len(args.prefill_chunk) != 1
            or args.prefill_chunk[0] not in PRODUCTION_PREFILL_CHUNKS
        ):
            raise SystemExit(f"hybrid {args.preset} requires one selected prefill chunk")
    else:
        if args.concurrency != [1, 2, 3, 4]:
            raise SystemExit("hybrid performance evidence requires exactly C=1,2,3,4")
        if (
            not args.prefill_chunk
            or len(args.prefill_chunk) != 1
            or args.prefill_chunk[0] not in PRODUCTION_PREFILL_CHUNKS
        ):
            raise SystemExit("hybrid performance evidence requires one selected prefill chunk")
    if args.expected_kv_value_group not in (16, 32):
        raise SystemExit("hybrid performance evidence requires a product G16 or G32 build")
    if args.expected_xattention_profile not in ("dense", "b128-s16-tau900"):
        raise SystemExit("hybrid performance evidence requires a product attention profile")
    if args.suite or args.limit is not None or args.repetitions is not None or args.warmup is not None:
        raise SystemExit("hybrid performance evidence requires the complete fixed preset")
    if args.hybrid_width_tool is None:
        raise SystemExit("--require-fp8-hybrid requires --hybrid-width-tool")


def validate_post_chunk_capacity_contract(args: argparse.Namespace) -> None:
    """Keep the terminal capacity gate on its exact four-concurrency product geometry."""

    if not args.require_post_chunk_capacity:
        return
    if args.preset != "pareto-capacity":
        raise SystemExit("--require-post-chunk-capacity requires --preset pareto-capacity")
    if args.concurrency != list(PRODUCT_CONCURRENCIES):
        raise SystemExit("post-chunk capacity requires exactly C=1,2,3,4")
    if args.device != 0:
        raise SystemExit("post-chunk capacity is fixed to device 0")
    if args.prefill_chunk_authority is None:
        raise SystemExit("post-chunk capacity requires --prefill-chunk-authority")
    if args.expected_kv_value_group not in (16, 32):
        raise SystemExit("post-chunk capacity requires a product G16 or G32 build")
    if args.expected_xattention_profile not in XATTENTION_PROFILES:
        raise SystemExit("post-chunk capacity requires a product attention profile")
    if (
        args.suite
        or args.limit is not None
        or args.repetitions is not None
        or args.warmup is not None
    ):
        raise SystemExit("post-chunk capacity requires the complete fixed preset")


def validate_hybrid_shared_workspace_authority(
    authority: object, prefill_chunks: Sequence[int],
) -> dict[str, Any]:
    """Validate exact host-planner shared-workspace widths for every measured chunk."""

    chunks = sorted(set(prefill_chunks))
    expected_by_chunk = {
        str(chunk): {
            "ordinary": [1, 2, 3, 4, chunk],
            "mtp3": [1, 2, 3, 4, 8, 12, 16, chunk],
        }
        for chunk in chunks
    }
    if (
        not isinstance(authority, dict)
        or set(authority) != {
            "tool", "maximum_concurrency", "prefill_chunks", "inventories_by_prefill_chunk",
        }
        or authority.get("maximum_concurrency") != 4
        or authority.get("prefill_chunks") != chunks
        or authority.get("inventories_by_prefill_chunk") != expected_by_chunk
        or not isinstance(authority.get("tool"), dict)
        or not isinstance(authority["tool"].get("path"), str)
        or type(authority["tool"].get("file_size_bytes")) is not int
        or authority["tool"]["file_size_bytes"] <= 0
        or not isinstance(authority["tool"].get("sha256"), str)
        or len(authority["tool"]["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in authority["tool"]["sha256"])
    ):
        raise ValueError("hybrid shared-workspace width authority differs")
    return authority


def build_hybrid_shared_workspace_authority(
    width_tool: Path, prefill_chunks: Sequence[int],
) -> dict[str, Any]:
    chunks = sorted(set(prefill_chunks))
    inventories: dict[str, dict[str, list[int]]] = {}
    for chunk in chunks:
        observed = {
            "ordinary": query_widths(width_tool, chunk, 4, 0),
            "mtp3": query_widths(width_tool, chunk, 4, 4),
        }
        expected = {
            "ordinary": [1, 2, 3, 4, chunk],
            "mtp3": [1, 2, 3, 4, 8, 12, 16, chunk],
        }
        if observed != expected:
            raise SystemExit(
                f"hybrid shared-workspace width authority differs at chunk {chunk}: {observed!r}"
            )
        inventories[str(chunk)] = observed
    authority = {
        "tool": inspect_executable(width_tool),
        "maximum_concurrency": 4,
        "prefill_chunks": chunks,
        "inventories_by_prefill_chunk": inventories,
    }
    validate_hybrid_shared_workspace_authority(authority, chunks)
    return authority


def validate_dflash_campaign_artifact(
    preset: str, artifact: dict[str, Any], *, dry_run: bool
) -> None:
    """Reject physical DFlash campaigns for base recipes excluded by native capacity."""

    if dry_run or preset not in DFLASH_CAMPAIGN_PRESETS:
        return
    weights_id = artifact.get("weights_id")
    if weights_id not in DFLASH_CAMPAIGN_WEIGHTS_IDS:
        raise SystemExit(
            f"--preset {preset} requires a registered DFlash companion of one of the three "
            f"terminal base recipes; got {MODEL_ID}/{weights_id}."
        )


def add_repetition_args(
    base_args: list[str], case: BenchCase, repetitions_override: int | None,
    warmup_override: int | None,
) -> list[str]:
    repetitions = repetitions_override if repetitions_override is not None else case.repetitions
    warmup = warmup_override if warmup_override is not None else case.warmup
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError(f"benchmark case {case.name} has invalid repetitions")
    if type(warmup) is not int or warmup < 0:
        raise ValueError(f"benchmark case {case.name} has invalid warmup")
    return [*base_args, "-r", str(repetitions), "--warmup", str(warmup)]


def build_cases(
    preset: str, dflash_draft_tokens: int | None = None, dflash_verify_width: int = 0,
    prefill_chunks: Sequence[int] = PRODUCTION_PREFILL_CHUNKS,
    prefill_prompt: int = 8192,
    production_prefill_chunk: int = 4096,
) -> list[BenchCase]:
    if preset == "smoke":
        return [
            BenchCase("prefill_length", "prefill_p128_k0", ("-p", "128", *mtp_args(0)), 1, 0),
            BenchCase("pure_decode", "tg8_k3_graph", ("-n", "8", *mtp_args(3)), 1, 0),
            BenchCase(
                "context_decode",
                "ctx_p128_g8_k3_graph",
                ("-pg", "128,8", "--max-ctx", "256", *mtp_args(3)),
                1,
                0,
            ),
        ]

    if preset == "concurrency":
        return [
            BenchCase(
                "concurrent_prefill",
                "prefill_lengths_k3",
                ("-p", "512,2048,8192", *mtp_args(3)),
                3,
                1,
                "concurrent prefill throughput at representative prompt lengths",
            ),
            BenchCase(
                "concurrent_decode",
                "pure_decode_g512_k3_graph",
                ("-n", "512", *mtp_args(3)),
                3,
                1,
                "saturated decode from the one-token seed",
            ),
            BenchCase(
                "concurrent_decode",
                "context_decode_k3_graph",
                ("-pg", "512,512;2048,512;8192,512", *mtp_args(3)),
                3,
                1,
                "batched decode at representative context offsets",
            ),
        ]

    if preset == "prefill-chunk":
        return [
            BenchCase(
                "production_prefill_chunk",
                f"prefill_p{prefill_prompt}_chunk{chunk}_ordinary",
                ("-p", str(prefill_prompt), "--prefill-chunk", str(chunk),
                 "--draft-tokens", "0"),
                3,
                1,
                (
                    "C1 spec-none ordinary Text-prefill chunk selection"
                ),
                concurrency_one_only=True,
            )
            for chunk in prefill_chunks
        ]

    if preset == "low-context-prefill":
        return [
            BenchCase(
                "low_context_prefill",
                f"prefill_p{prompt}_dense_none",
                (
                    "-p", str(prompt), "--prefill-chunk", str(production_prefill_chunk),
                    "--draft-tokens", "0",
                ),
                3,
                1,
                "C1 dense prefill-only low-context performance ladder",
                concurrency_one_only=True,
            )
            for prompt in LOW_CONTEXT_PREFILL_PROMPTS
        ]

    if preset == "ordinary-diagnostic":
        return [
            BenchCase(
                "ordinary_decode",
                "whole_p8192_g256_none_graph",
                ("--whole-pg", "8192,256", "--prefill-chunk", "4096",
                 "--draft-tokens", "0"),
                3,
                1,
                "C1 8K ordinary non-speculative prefill/decode diagnostic",
                concurrency_one_only=True,
            )
        ]

    if preset == "pareto":
        return [
            BenchCase(
                "pareto_prefill",
                "prefill_p8192_p32768_k3",
                ("-p", "8192,32768", "--prefill-chunk", str(production_prefill_chunk),
                 *mtp_args(3)),
                3,
                1,
                "matched 8K/32K production-profile prefill throughput",
            ),
            BenchCase(
                "pareto_decode",
                "context_p8192_p32768_g256_k3_graph",
                ("-pg", "8192,256;32768,256", "--prefill-chunk",
                 str(production_prefill_chunk),
                 *mtp_args(3)),
                3,
                1,
                "matched 8K/32K production-profile decode and MTP acceptance",
            ),
        ]

    if preset == "pareto-whole":
        return [
            BenchCase(
                "pareto_whole_inference",
                "whole_p8192_p32768_g256_ordinary_graph",
                ("--whole-pg", "8192,256;32768,256", "--prefill-chunk",
                 str(production_prefill_chunk), "--draft-tokens", "0"),
                3,
                1,
                "ranking spec-none ordinary fresh-prompt 8K/32K whole-inference throughput",
                parity_role="ordinary",
            ),
        ]

    if preset == "pareto-feasibility":
        return [
            BenchCase(
                "pareto_workload_feasibility",
                "workload_feasibility_mtp3",
                ("-p", "128", "--prefill-chunk", str(production_prefill_chunk),
                 "--max-ctx", "33030", "--kv-capacity", "auto",
                 *mtp_args(3)),
                1,
                0,
                "required 32K workload feasibility with standard headroom and MTP3",
            ),
        ]

    if preset == "pareto-capacity":
        return [
            BenchCase(
                "pareto_effective_capacity",
                "effective_capacity_ordinary",
                ("-p", "128", "--prefill-chunk", str(production_prefill_chunk),
                 "--max-ctx", str(MODEL_NATIVE_CONTEXT),
                 "--kv-capacity", "auto", "--draft-tokens", "0"),
                1,
                0,
                "resolved ordinary effective maximum at the model-native context ceiling",
            ),
        ]

    if preset == "dflash-feasibility":
        if dflash_draft_tokens is None:
            raise ValueError("dflash-feasibility requires an explicit DFlash draft window")
        profile = dflash_args(dflash_draft_tokens, dflash_verify_width)
        resolved_width = resolved_dflash_verify_width(
            dflash_draft_tokens, dflash_verify_width
        )
        capacity_context = 32768 + 256 + 2 * resolved_width
        return [
            BenchCase(
                "dflash_workload_feasibility",
                "workload_feasibility_dflash",
                ("-p", "128", "--prefill-chunk", str(production_prefill_chunk),
                 "--max-ctx", str(capacity_context),
                 "--kv-capacity", "auto", *profile),
                1,
                0,
                "required 32K workload feasibility with standard headroom and selected DFlash",
            ),
        ]

    if preset == "dflash-capacity":
        if dflash_draft_tokens is None:
            raise ValueError("dflash-capacity requires an explicit DFlash draft window")
        profile = dflash_args(dflash_draft_tokens, dflash_verify_width)
        return [
            BenchCase(
                "dflash_effective_capacity",
                "effective_capacity_dflash",
                ("-p", "128", "--prefill-chunk", str(production_prefill_chunk),
                 "--max-ctx", str(MODEL_NATIVE_CONTEXT),
                 "--kv-capacity", "auto", *profile),
                1,
                0,
                "resolved effective maximum at the model-native ceiling with selected DFlash",
            ),
        ]

    if preset == "dflash-shortlist":
        control = BenchCase(
            suite="dflash_shortlist_control",
            name="context_p8192_g256_ordinary_graph",
            args=("-pg", "8192,256", "--prefill-chunk", str(production_prefill_chunk),
                  *mtp_args(0), "--retain-token-ids"),
            repetitions=2,
            warmup=1,
            notes="shared ordinary greedy control for every shortlisted K",
            retain_token_ids=True,
            parity_role="ordinary",
            concurrency_one_only=True,
        )
        candidates: list[BenchCase] = []
        diagnostics: list[BenchCase] = []
        for k in DFLASH_SHORTLIST_K_ORDER:
            width = resolved_dflash_verify_width(k, 0)
            topology = resolved_dflash_topology(k, width)
            profile = dflash_args(k, width)
            candidates.append(
                BenchCase(
                    suite="dflash_shortlist_decode",
                    name=f"context_p8192_g256_dflash_k{k}_w{width}",
                    args=("-pg", "8192,256", "--prefill-chunk",
                          str(production_prefill_chunk), *profile, "--retain-token-ids"),
                    repetitions=2,
                    warmup=1,
                    notes=f"8K shortlist performance and acceptance; {topology}",
                    retain_token_ids=True,
                    parity_role="dflash",
                    concurrency_one_only=True,
                )
            )
            diagnostics.append(
                BenchCase(
                    suite="dflash_shortlist_repair_diagnostic",
                    name=f"context_p8192_g256_dflash_k{k}_w{width}_diagnostic",
                    args=("-pg", "8192,256", "--prefill-chunk",
                          str(production_prefill_chunk), *profile, "--no-device-graph"),
                    repetitions=1,
                    warmup=0,
                    notes=(f"syncing first-reject/repair diagnostic; {topology}; "
                           "timings are discarded"),
                    diagnostic=True,
                    concurrency_one_only=True,
                )
            )
        # All performance rows precede the synchronizing diagnostics. The balanced K order avoids
        # assigning a monotonic thermal/order bias to increasing K.
        return [control, *candidates, *diagnostics]

    if preset == "dflash-pareto":
        if dflash_draft_tokens is None:
            raise ValueError("dflash-pareto requires an explicit DFlash draft window")
        profile = dflash_args(dflash_draft_tokens, dflash_verify_width)
        return [
            BenchCase(
                suite="dflash_pareto_prefill",
                name="prefill_p8192_p32768_dflash",
                args=("-p", "8192,32768", "--prefill-chunk",
                      str(production_prefill_chunk), *profile),
                repetitions=3,
                warmup=1,
                notes="matched 8K/32K DFlash-enabled prefill throughput",
            ),
            BenchCase(
                suite="dflash_pareto_decode",
                name="context_p8192_p32768_g256_dflash_graph",
                args=("-pg", "8192,256;32768,256", "--prefill-chunk",
                      str(production_prefill_chunk), *profile, "--retain-token-ids"),
                repetitions=3,
                warmup=1,
                notes="matched 8K/32K DFlash decode, acceptance, and greedy outputs",
                retain_token_ids=True,
                parity_role="dflash",
            ),
            BenchCase(
                suite="dflash_pareto_control",
                name="context_p8192_p32768_g256_ordinary_graph",
                args=("-pg", "8192,256;32768,256", "--prefill-chunk",
                      str(production_prefill_chunk), *mtp_args(0), "--retain-token-ids"),
                repetitions=3,
                warmup=1,
                notes="ordinary greedy control over the identical artifact and prompts",
                retain_token_ids=True,
                parity_role="ordinary",
            ),
            BenchCase(
                suite="dflash_pareto_whole_inference",
                name="whole_p8192_p32768_g256_dflash_graph",
                args=("--whole-pg", "8192,256;32768,256", "--prefill-chunk",
                      str(production_prefill_chunk), *profile),
                repetitions=3,
                warmup=1,
                notes="matched fresh-prompt DFlash whole-inference makespan and output throughput",
            ),
            *[
                BenchCase(
                    suite="dflash_selector_diagnostic",
                    name=f"context_p8192_g256_dflash_eager_diagnostic_{repeat}",
                    args=("-pg", "8192,256", "--prefill-chunk",
                          str(production_prefill_chunk), *profile, "--no-device-graph",
                          "--retain-token-ids"),
                    repetitions=1,
                    warmup=0,
                    notes=("repeated syncing proposal/selector trace; exact proposal and target "
                           "tokens are quality evidence; timings are discarded"),
                    retain_token_ids=True,
                    diagnostic=True,
                    concurrency_one_only=True,
                )
                for repeat in ("a", "b")
            ],
        ]

    include_full = preset == "full"
    prefill_lengths = PREFILL_LENGTHS_CORE + (PREFILL_LENGTHS_FULL_EXTRA if include_full else ())
    context_pairs = CONTEXT_CORE + (CONTEXT_FULL_EXTRA if include_full else ())
    sweep_pairs = ((2048, 512),) + (((32768, 256),) if include_full else ())

    cases: list[BenchCase] = []

    for k in PRIMARY_KS:
        cases.append(
            BenchCase(
                "prefill_length",
                f"prefill_lengths_k{k}",
                ("-p", csv_list(prefill_lengths), *mtp_args(k)),
                3,
                1,
                "prefill length curve",
            )
        )

    for k in PRIMARY_KS:
        for chunk in PREFILL_CHUNKS:
            cases.append(
                BenchCase(
                    "prefill_chunk",
                    f"prefill_p8192_chunk{chunk}_k{k}",
                    (
                        "-p",
                        "8192",
                        "--prefill-chunk",
                        str(chunk),
                        *mtp_args(k),
                    ),
                    3,
                    1,
                    "workspace and chunk-size sensitivity",
                )
            )

    for k in PRIMARY_KS:
        for graph in (True, False):
            graph_suffix = "graph" if graph else "eager"
            args = ["-n", csv_list(PURE_DECODE_GENS), *mtp_args(k)]
            if not graph:
                args.append("--no-device-graph")
            cases.append(
                BenchCase(
                    "pure_decode",
                    f"tg_lengths_k{k}_{graph_suffix}",
                    tuple(args),
                    5,
                    1,
                    "pure decode throughput; tg seeds only one token",
                )
            )

    for k in PRIMARY_KS:
        for graph in (True, False):
            graph_suffix = "graph" if graph else "eager"
            args = ["-pg", pair_list(context_pairs), *mtp_args(k)]
            if not graph:
                args.append("--no-device-graph")
            cases.append(
                BenchCase(
                    "context_decode",
                    f"context_decode_k{k}_{graph_suffix}",
                    tuple(args),
                    3,
                    1,
                    "decode at real context offsets",
                )
            )

    for k in SWEEP_KS:
        cases.append(
            BenchCase(
                "mtp_sweep",
                f"mtp_sweep_k{k}_graph",
                ("-pg", pair_list(sweep_pairs), *mtp_args(k)),
                3,
                1,
                "primary MTP draft-window sweep",
            )
        )

    for k, prompt in ((3, 8174), (5, 8170)):
        for graph in (True, False):
            graph_suffix = "graph" if graph else "eager"
            args = [
                "-pg",
                f"{prompt},12",
                "--max-ctx",
                "8192",
                *mtp_args(k),
            ]
            if not graph:
                args.append("--no-device-graph")
            cases.append(
                BenchCase(
                    "tail_stress",
                    f"tail_k{k}_{graph_suffix}",
                    tuple(args),
                    3,
                    1,
                    "near-capacity fallback stress",
                )
            )

    return cases


def filtered_cases(cases: list[BenchCase], suites: Sequence[str], limit: int | None) -> list[BenchCase]:
    selected = cases
    if suites:
        allowed = set(suites)
        selected = [case for case in selected if case.suite in allowed]
    if limit is not None:
        selected = selected[:limit]
    return selected


def max_prompt_in_cases(cases: Sequence[BenchCase]) -> int:
    max_prompt = 0
    for case in cases:
        args = list(case.args)
        for flag in ("-p", "--n-prompt"):
            if flag in args:
                raw = args[args.index(flag) + 1]
                max_prompt = max(max_prompt, *(int(piece) for piece in raw.split(",")))
        for flag in ("-pg", "--prompt-gen", "--whole-pg"):
            if flag in args:
                raw = args[args.index(flag) + 1]
                for pair in raw.split(";"):
                    p, _ = pair.split(",", 1)
                    max_prompt = max(max_prompt, int(p))
    return max_prompt


def expected_tests(case: BenchCase) -> list[dict[str, Any]]:
    args = list(case.args)
    expected: list[dict[str, Any]] = []
    for flag in ("-p", "--n-prompt"):
        if flag in args:
            for raw in args[args.index(flag) + 1].split(","):
                prompt = int(raw)
                expected.append(
                    {"label": f"pp{prompt}", "kind": "pp", "n_prompt": prompt, "n_gen": 0}
                )
    for flag in ("-n", "--n-gen"):
        if flag in args:
            for raw in args[args.index(flag) + 1].split(","):
                generated = int(raw)
                expected.append(
                    {"label": f"tg{generated}", "kind": "tg", "n_prompt": 0,
                     "n_gen": generated}
                )
    for flag in ("-pg", "--prompt-gen", "--whole-pg"):
        if flag in args:
            for raw in args[args.index(flag) + 1].split(";"):
                prompt_text, generated_text = raw.split(",", 1)
                prompt, generated = int(prompt_text), int(generated_text)
                whole = flag == "--whole-pg"
                expected.append(
                    {"label": (f"whole-pp{prompt}+tg{generated}" if whole else
                               f"pp{prompt}+tg{generated}"),
                     "kind": "whole" if whole else "pp+tg",
                     "n_prompt": prompt, "n_gen": generated}
                )
    if not expected:
        raise ValueError(f"benchmark case {case.name!r} declares no tests")
    return expected


def _finite_number(value: object, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if positive and number <= 0.0:
        return False
    if nonnegative and number < 0.0:
        return False
    return True


def _nonnegative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _case_option(case: BenchCase, name: str) -> str | None:
    args = list(case.args)
    return args[args.index(name) + 1] if name in args else None


def validate_automatic_feasibility(report: dict[str, Any]) -> dict[str, Any]:
    """Validate auto sizing and classify feasibility or the effective native-context maximum."""

    config = report.get("config", {})
    if config.get("kv_cache_format") != "fp8-k-int4-v":
        raise ValueError("benchmark report has invalid kv_cache_format")
    if report.get("memory", {}).get("kv_cache_format") != "fp8-k-int4-v":
        raise ValueError("benchmark memory report has invalid kv_cache_format")
    memory = report.get("memory", {})
    graph_allowance = memory.get("device_graph_allowance_bytes")
    graph_observed = memory.get("device_graph_observed_bytes")
    if type(graph_allowance) is not int or graph_allowance <= 0:
        raise ValueError("automatic-capacity report has invalid Device Graph allowance")
    if (
        type(graph_observed) is not int
        or graph_observed < 0
        or graph_observed > graph_allowance
    ):
        raise ValueError("automatic-capacity report has invalid Device Graph observed allocation")
    max_context = config.get("max_context")
    concurrency = config.get("concurrency")
    if type(max_context) is not int or max_context <= 0:
        raise ValueError("automatic-capacity report has invalid max_context")
    if type(concurrency) is not int or concurrency not in PRODUCT_CONCURRENCIES:
        raise ValueError("automatic-capacity report has invalid concurrency")
    if memory.get("max_context") != max_context:
        raise ValueError("automatic-capacity memory/config max_context mismatch")

    pages = memory.get("kv_capacity_page_groups")
    maximum_pages = memory.get("kv_capacity_max_page_groups")
    resolved_tokens = memory.get("kv_capacity")
    logical_pages = (max_context + KV_PAGE_TOKENS - 1) // KV_PAGE_TOKENS
    minimum_pages = max(logical_pages, concurrency)
    expected_maximum = logical_pages * concurrency
    if type(pages) is not int or not minimum_pages <= pages <= expected_maximum:
        raise ValueError("automatic-capacity report has invalid resolved page groups")
    if maximum_pages != expected_maximum:
        raise ValueError("automatic-capacity report has invalid maximum page groups")
    if resolved_tokens != pages * KV_PAGE_TOKENS:
        raise ValueError("automatic-capacity report token/page resolution is inconsistent")

    minimum_bytes = memory.get("minimum_runtime_reservation_bytes")
    increment_bytes = memory.get("kv_capacity_increment_bytes")
    reservation_bytes = memory.get("runtime_reservation_bytes")
    available_bytes = memory.get("available_after_weights_bytes")
    headroom_bytes = memory.get("kv_capacity_headroom_bytes")
    slack_bytes = memory.get("planned_slack_bytes")
    for key, value in (
        ("minimum_runtime_reservation_bytes", minimum_bytes),
        ("kv_capacity_increment_bytes", increment_bytes),
        ("runtime_reservation_bytes", reservation_bytes),
        ("available_after_weights_bytes", available_bytes),
        ("planned_slack_bytes", slack_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"automatic-capacity report has invalid {key}")
    if headroom_bytes != AUTOMATIC_KV_HEADROOM_BYTES:
        raise ValueError("automatic-capacity report does not retain the standard 1 GiB headroom")
    if pages < expected_maximum and increment_bytes <= 0:
        raise ValueError("automatic-capacity report has no positive page-group increment")
    expected_reservation = minimum_bytes + (pages - minimum_pages) * increment_bytes
    if reservation_bytes != expected_reservation:
        raise ValueError("automatic-capacity report reservation curve is inconsistent")
    if available_bytes - reservation_bytes != slack_bytes or slack_bytes < headroom_bytes:
        raise ValueError("automatic-capacity report headroom/slack accounting is inconsistent")
    if pages < expected_maximum and slack_bytes - headroom_bytes >= increment_bytes:
        raise ValueError("automatic-capacity report did not resolve the maximum feasible page group")
    effective_maximum = max_context == MODEL_NATIVE_CONTEXT
    if increment_bytes == 0:
        # C=1 has no expandable point inside the addressable curve: M_min == M_max. The
        # resolver proves the configured context fits but exposes no byte stride from which a
        # device-memory maximum could be established.
        return {
            "measurement_kind": (
                "resolved_effective_maximum" if effective_maximum
                else "required_workload_feasibility"
            ),
            "binding_constraint": (
                "model_context" if effective_maximum else "logical_context_ceiling"
            ),
            "uncensored": effective_maximum,
            "resolved_effective_maximum_tokens": (
                pages * KV_PAGE_TOKENS if effective_maximum else None
            ),
            "memory_limited_capacity_tokens": None,
            "logical_capacity_tokens": pages * KV_PAGE_TOKENS,
        }
    else:
        unconstrained_pages = minimum_pages + (
            available_bytes - headroom_bytes - minimum_bytes
        ) // increment_bytes
    expected_pages = min(unconstrained_pages, expected_maximum)
    if pages != expected_pages:
        raise ValueError("automatic-capacity report does not match the resolver equation")
    if effective_maximum:
        constraint = "device_memory" if pages < expected_maximum else "model_context"
        measurement_kind = "resolved_effective_maximum"
    elif unconstrained_pages < expected_maximum:
        constraint = "device_memory"
        measurement_kind = "required_workload_feasibility"
    elif unconstrained_pages > expected_maximum:
        constraint = "logical_context_ceiling"
        measurement_kind = "required_workload_feasibility"
    else:
        constraint = "device_memory_and_logical_context_ceiling"
        measurement_kind = "required_workload_feasibility"
    return {
        "measurement_kind": measurement_kind,
        "binding_constraint": constraint,
        "uncensored": effective_maximum or constraint != "logical_context_ceiling",
        "resolved_effective_maximum_tokens": (
            pages * KV_PAGE_TOKENS if effective_maximum else None
        ),
        "memory_limited_capacity_tokens": (
            pages * KV_PAGE_TOKENS if constraint == "device_memory" else None
        ),
        "logical_capacity_tokens": pages * KV_PAGE_TOKENS,
    }


def validate_case_profile(config: dict[str, Any], case: BenchCase) -> None:
    requested_spec = _case_option(case, "--spec") or "mtp"
    requested_drafts = int(_case_option(case, "--draft-tokens") or "0")
    expected_spec = requested_spec if requested_drafts else "none"
    requested_verify = int(_case_option(case, "--dflash-verify-width") or "0")
    expected_verify = (
        resolved_dflash_verify_width(requested_drafts, requested_verify)
        if expected_spec == "dflash" else 0
    )
    expected_head = "optimized" if "--lm-head-draft" in case.args else "full"
    expected_graph = "--no-device-graph" not in case.args
    expected = {
        "spec": expected_spec,
        "draft_tokens": requested_drafts,
        "speculative_execution": requested_drafts > 0,
        "dflash_verify_width_requested": requested_verify,
        "dflash_verify_width": expected_verify,
        "proposal_head": expected_head,
        "use_device_graph": expected_graph,
        "retain_token_ids": case.retain_token_ids,
        "repetitions": case.repetitions,
        "warmup": case.warmup,
    }
    requested_prefill_chunk = _case_option(case, "--prefill-chunk")
    if requested_prefill_chunk is not None:
        expected["prefill_chunk"] = int(requested_prefill_chunk)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"benchmark report {key}={config.get(key)!r}; expected {value!r} for {case.name}"
            )


def validate_whole_ordinary_command(case: BenchCase, command: Sequence[str] | None) -> None:
    """Require base-selection whole timing to be the exact spec-none ordinary route."""

    if case.suite != "pareto_whole_inference":
        return
    if command is None:
        raise ValueError("pareto-whole validation requires the exact benchmark command")
    if command.count("--draft-tokens") != 1:
        raise ValueError("pareto-whole command requires exactly one --draft-tokens")
    index = command.index("--draft-tokens")
    if index + 1 >= len(command) or command[index + 1] != "0":
        raise ValueError("pareto-whole command requires --draft-tokens 0")
    if "--spec" in command or "--lm-head-draft" in command:
        raise ValueError("pareto-whole command must be spec-none ordinary without a draft head")


def _validate_speculative(spec: object, *, enabled: bool, draft_window: int,
                          require_sample: bool, require_accepted: bool,
                          label: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"benchmark test {label} has no speculative statistics")
    for key in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps"):
        if not _nonnegative_integer(spec.get(key)):
            raise ValueError(f"benchmark test {label} has invalid speculative {key}")
    if spec.get("enabled") is not enabled:
        raise ValueError(f"benchmark test {label} has inconsistent speculative enabled flag")
    if spec.get("draft_window") != (draft_window if enabled else 0):
        raise ValueError(f"benchmark test {label} has inconsistent speculative draft_window")
    rounds = spec["rounds"]
    drafted = spec["drafted_tokens"]
    accepted = spec["accepted_tokens"]
    if accepted > drafted:
        raise ValueError(f"benchmark test {label} accepted more tokens than it drafted")
    accepted_per_position = spec.get("accepted_per_position")
    if not isinstance(accepted_per_position, list) or any(
        not _nonnegative_integer(value) for value in accepted_per_position
    ):
        raise ValueError(f"benchmark test {label} has invalid accepted_per_position")
    if enabled:
        if require_sample and (rounds == 0 or drafted == 0):
            raise ValueError(f"benchmark test {label} has no speculative acceptance sample")
        if len(accepted_per_position) != draft_window or sum(accepted_per_position) != accepted:
            raise ValueError(f"benchmark test {label} has incoherent acceptance positions")
        if require_accepted and accepted == 0:
            raise ValueError(f"benchmark test {label} has no accepted speculative tokens")
        if rounds == 0 and drafted == 0:
            if accepted != 0 or spec["fallback_steps"] != 0:
                raise ValueError(f"benchmark test {label} has acceptance without a draft round")
            if spec.get("acceptance_rate") is not None or spec.get("acceptance_length") is not None:
                raise ValueError(f"benchmark test {label} has rates without an acceptance sample")
        elif rounds == 0 or drafted == 0:
            raise ValueError(f"benchmark test {label} has partial speculative counters")
        else:
            expected_rate = accepted / drafted
            expected_length = 1.0 + accepted / rounds
            if not _finite_number(spec.get("acceptance_rate"), nonnegative=True) or not math.isclose(
                float(spec["acceptance_rate"]), expected_rate, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"benchmark test {label} has incoherent acceptance_rate")
            if not _finite_number(spec.get("acceptance_length"), positive=True) or not math.isclose(
                float(spec["acceptance_length"]), expected_length, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"benchmark test {label} has incoherent acceptance_length")
    elif (
        rounds != 0
        or drafted != 0
        or accepted != 0
        or spec["fallback_steps"] != 0
        or accepted_per_position
        or spec.get("acceptance_rate") is not None
        or spec.get("acceptance_length") is not None
    ):
        raise ValueError(f"benchmark test {label} has speculative data while disabled")


def validate_report_tests(report: dict[str, Any], case: BenchCase) -> None:
    expected = expected_tests(case)
    tests = report.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("benchmark report has no test rows")
    actual_geometry = [
        {key: test.get(key) for key in ("label", "kind", "n_prompt", "n_gen")}
        if isinstance(test, dict) else None
        for test in tests
    ]
    if actual_geometry != expected:
        raise ValueError(
            f"benchmark report test geometry {actual_geometry!r}; expected {expected!r}"
        )
    config = report["config"]
    repetitions = config.get("repetitions")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("benchmark report has invalid repetitions")
    draft_window = config.get("draft_tokens")
    if not _nonnegative_integer(draft_window):
        raise ValueError("benchmark report has invalid draft_tokens")
    for test, geometry in zip(tests, expected, strict=True):
        label = geometry["label"]
        has_prefill = geometry["kind"] in ("pp", "pp+tg", "whole")
        has_decode = geometry["kind"] in ("tg", "pp+tg", "whole")
        wanted_outputs = geometry["n_gen"] + 1 if has_decode else 1
        if test.get("requested_output_tokens") != wanted_outputs:
            raise ValueError(f"benchmark test {label} has wrong requested_output_tokens")
        for key in ("workspace_peak_bytes", "workspace_allocator_peak_bytes"):
            if not _nonnegative_integer(test.get(key)):
                raise ValueError(f"benchmark test {label} has invalid {key}")
        if not _finite_number(test.get("total_seconds_mean"), positive=True):
            raise ValueError(f"benchmark test {label} has no positive total timing")
        if has_prefill:
            for key in ("prefill_seconds_mean", "prefill_tok_s_mean"):
                if not _finite_number(test.get(key), positive=True):
                    raise ValueError(f"benchmark test {label} has no positive {key}")
        if has_decode:
            for key in (
                "decode_seconds_mean", "decode_output_tok_s_mean", "decode_engine_tok_s_mean"
            ):
                if not _finite_number(test.get(key), positive=True):
                    raise ValueError(f"benchmark test {label} has no positive {key}")
        if geometry["kind"] == "whole" and not _finite_number(
            test.get("whole_output_tok_s_mean"), positive=True
        ):
            raise ValueError(f"benchmark test {label} has no positive whole_output_tok_s_mean")
        reps = test.get("reps")
        if not isinstance(reps, list) or len(reps) != repetitions:
            raise ValueError(f"benchmark test {label} does not retain every repetition")
        for rep in reps:
            if not isinstance(rep, dict):
                raise ValueError(f"benchmark test {label} has a non-object repetition")
            retained = rep.get("generated_token_ids_by_lane")
            if case.retain_token_ids:
                concurrency = config.get("concurrency")
                if not isinstance(retained, list) or len(retained) != concurrency:
                    raise ValueError(
                        f"benchmark test {label} does not retain every concurrency lane"
                    )
                if any(
                    not isinstance(lane, list)
                    or len(lane) != wanted_outputs
                    or any(type(token) is not int or token < 0 for token in lane)
                    for lane in retained
                ):
                    raise ValueError(f"benchmark test {label} has invalid retained token IDs")
            elif retained is not None:
                raise ValueError(f"benchmark test {label} unexpectedly retains token IDs")
        _validate_speculative(
            test.get("speculative"),
            enabled=draft_window > 0,
            draft_window=draft_window,
            require_sample=has_decode,
            require_accepted=case.parity_role == "mtp",
            label=label,
        )


def load_bench_report(
    report_path: Path,
    expected_kv_value_group: int | None = None,
    expected_q4_activation_bits: int | None = None,
    expected_w8_activation_bits: int | None = None,
    expected_fp8_qk_wmma: bool | None = None,
    expected_concurrency: int | None = None,
    expected_artifact: dict[str, Any] | None = None,
    expected_command: Sequence[str] | None = None,
    expected_case: BenchCase | None = None,
    expected_xattention_profile: str = "dense",
) -> dict[str, Any]:
    if expected_xattention_profile not in XATTENTION_PROFILES:
        raise ValueError(
            f"unsupported XAttention profile {expected_xattention_profile!r}; "
            f"expected one of {XATTENTION_PROFILES!r}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("benchmark report root must be an object")
    identity = (
        report.get("schema_version"),
        report.get("artifact_type"),
        report.get("tool"),
    )
    expected = (REPORT_SCHEMA_VERSION, REPORT_ARTIFACT_TYPE, REPORT_TOOL)
    if identity != expected:
        raise ValueError(
            "unsupported benchmark report identity: "
            f"schema_version={identity[0]!r}, artifact_type={identity[1]!r}, "
            f"tool={identity[2]!r}; expected {expected!r}"
        )
    config = report.get("config", {})
    value_group = config.get("kv_value_group")
    if value_group not in (16, 32):
        raise ValueError(f"benchmark report has invalid kv_value_group={value_group!r}")
    if expected_kv_value_group is not None and value_group != expected_kv_value_group:
        raise ValueError(
            f"benchmark report kv_value_group={value_group}; "
            f"expected compiled G{expected_kv_value_group}"
        )
    plane_layouts = config.get("kv_plane_layouts")
    if plane_layouts != R9700_KV_PLANE_LAYOUTS:
        raise ValueError(
            f"benchmark report kv_plane_layouts={plane_layouts!r}; "
            f"expected compiled {R9700_KV_PLANE_LAYOUTS!r}"
        )
    activation_bits = config.get("q4_activation_bits")
    if activation_bits not in (4, 8):
        raise ValueError(
            f"benchmark report has invalid q4_activation_bits={activation_bits!r}"
        )
    if (
        expected_q4_activation_bits is not None
        and activation_bits != expected_q4_activation_bits
    ):
        raise ValueError(
            f"benchmark report q4_activation_bits={activation_bits}; "
            f"expected compiled A{expected_q4_activation_bits}"
        )
    q4_prefill_cta_profile = config.get("q4_prefill_cta_profile")
    if q4_prefill_cta_profile not in (
        "m64n128-pingpong-n16-k16-scalar-base-production",
    ):
        raise ValueError(
            "benchmark report has invalid q4_prefill_cta_profile="
            f"{q4_prefill_cta_profile!r}"
        )
    w8_activation_bits = config.get("w8_activation_bits")
    if w8_activation_bits not in (8, 16):
        raise ValueError(
            f"benchmark report has invalid w8_activation_bits={w8_activation_bits!r}"
        )
    if (
        expected_w8_activation_bits is not None
        and w8_activation_bits != expected_w8_activation_bits
    ):
        raise ValueError(
            f"benchmark report w8_activation_bits={w8_activation_bits}; "
            f"expected compiled A{expected_w8_activation_bits}"
        )
    fp8_qk_wmma = config.get("fp8_qk_wmma_enabled")
    if type(fp8_qk_wmma) is not bool:
        raise ValueError(
            "benchmark report has invalid fp8_qk_wmma_enabled="
            f"{fp8_qk_wmma!r}"
        )
    if expected_fp8_qk_wmma is not None and fp8_qk_wmma is not expected_fp8_qk_wmma:
        raise ValueError(
            f"benchmark report fp8_qk_wmma_enabled={fp8_qk_wmma}; "
            f"expected {expected_fp8_qk_wmma}"
        )
    expected_attention_profile = {
        "fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
        "fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
        "fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
    }
    for key, value in expected_attention_profile.items():
        if config.get(key) != value:
            raise ValueError(f"benchmark report {key}={config.get(key)!r}; expected {value!r}")
    expected_xattention = (
        {"xattention_qualification": False}
        if expected_xattention_profile == "dense"
        else {
            "xattention_qualification": True,
            "xattention_profile": "b128-s16-tau900",
            "xattention_find_block": 128,
            "xattention_stride": 16,
            "xattention_tau_permille": 900,
        }
    )
    for key, value in expected_xattention.items():
        if config.get(key) != value:
            raise ValueError(
                f"benchmark report {key}={config.get(key)!r}; expected {value!r}"
            )
    if expected_xattention_profile == "dense":
        stale = {
            "xattention_profile", "xattention_find_block", "xattention_stride",
            "xattention_tau_permille",
        }.intersection(config)
        if stale:
            raise ValueError(
                "dense benchmark report retains XAttention fields: "
                + ", ".join(sorted(stale))
            )
    if config.get("pending_timeout_ms") != BENCHMARK_PENDING_TIMEOUT_MS:
        raise ValueError(
            f"benchmark report pending_timeout_ms={config.get('pending_timeout_ms')!r}; "
            f"expected benchmark-owned {BENCHMARK_PENDING_TIMEOUT_MS}"
        )
    if config.get("pending_deadline") != "unbounded":
        raise ValueError(
            f"benchmark report pending_deadline={config.get('pending_deadline')!r}; "
            "expected benchmark-owned unbounded admission deadline"
        )
    concurrency = config.get("concurrency")
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency not in PRODUCT_CONCURRENCIES
    ):
        raise ValueError(f"benchmark report has invalid concurrency={concurrency!r}")
    if expected_concurrency is not None and concurrency != expected_concurrency:
        raise ValueError(
            f"benchmark report concurrency={concurrency}; expected C={expected_concurrency}"
        )
    if expected_artifact is not None:
        artifact_report = report.get("artifact", {})
        artifact_path = artifact_report.get("path")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError("benchmark report has no artifact path")
        actual_artifact = Path(artifact_path).expanduser().resolve()
        wanted_artifact = Path(expected_artifact["path"]).expanduser().resolve()
        if actual_artifact != wanted_artifact:
            raise ValueError(
                f"benchmark report artifact={actual_artifact}; expected {wanted_artifact}"
            )
        if artifact_report.get("file_size_bytes") != expected_artifact["file_size_bytes"]:
            raise ValueError("benchmark report artifact size does not match selected bytes")
        load = report.get("load", {})
        if load.get("target") != TARGET_ID:
            raise ValueError(f"benchmark report target={load.get('target')!r}; expected {TARGET_ID}")
        if load.get("weights_id") != expected_artifact["weights_id"]:
            raise ValueError("benchmark report weights_id does not match selected artifact")
    if expected_command is not None:
        wanted_command = " ".join(str(part) for part in expected_command)
        if report.get("command") != wanted_command:
            raise ValueError("benchmark report command does not match this matrix point")
    if expected_case is not None:
        validate_whole_ordinary_command(expected_case, expected_command)
        validate_case_profile(config, expected_case)
        validate_report_tests(report, expected_case)
        expected_capacity_mode = (
            "auto" if _case_option(expected_case, "--kv-capacity") == "auto" else "explicit"
        )
        if report.get("memory", {}).get("kv_capacity_mode") != expected_capacity_mode:
            raise ValueError(
                "benchmark report kv_capacity_mode="
                f"{report.get('memory', {}).get('kv_capacity_mode')!r}; "
                f"expected {expected_capacity_mode!r}"
            )
        if expected_capacity_mode == "auto":
            validate_automatic_feasibility(report)
    return report


def report_rows(
    report_path: Path,
    case: BenchCase,
    expected_kv_value_group: int | None = None,
    expected_q4_activation_bits: int | None = None,
    expected_w8_activation_bits: int | None = None,
    expected_fp8_qk_wmma: bool | None = None,
    expected_concurrency: int | None = None,
    expected_artifact: dict[str, Any] | None = None,
    expected_command: Sequence[str] | None = None,
    expected_xattention_profile: str = "dense",
) -> list[dict[str, Any]]:
    report = load_bench_report(
        report_path,
        expected_kv_value_group,
        expected_q4_activation_bits,
        expected_w8_activation_bits,
        expected_fp8_qk_wmma,
        expected_concurrency,
        expected_artifact,
        expected_command,
        case,
        expected_xattention_profile,
    )
    config = report.get("config", {})
    load = report.get("load", {})
    memory = report.get("memory", {})
    weights_memory = memory.get("weights", {})
    sequence_memory = memory.get("sequence", {})
    workspace_memory = memory.get("workspace", {})
    request_transient_memory = memory.get("request_transient", {})
    environment = report.get("environment", {})
    capacity_evidence = (
        validate_automatic_feasibility(report)
        if memory.get("kv_capacity_mode") == "auto"
        else None
    )
    rows = []
    for test in report.get("tests", []):
        speculative = test.get("speculative", {})
        performance_eligible = not case.diagnostic
        draft_tokens = config.get("draft_tokens")
        verify_width = config.get("dflash_verify_width")
        topology = (
            resolved_dflash_topology(draft_tokens, verify_width)
            if config.get("spec") == "dflash"
            and type(draft_tokens) is int
            and type(verify_width) is int
            else None
        )
        row = {
            "suite": case.suite,
            "case": case.name,
            "performance_eligible": performance_eligible,
            "diagnostic_timings_discarded": case.diagnostic,
            "parity_role": case.parity_role,
            "report": str(report_path),
            "label": test.get("label"),
            "kind": test.get("kind"),
            "n_prompt": test.get("n_prompt"),
            "n_gen": test.get("n_gen"),
            "requested_output_tokens": test.get("requested_output_tokens"),
            "target": load.get("target"),
            "weights_id": load.get("weights_id"),
            "artifact_path": report.get("artifact", {}).get("path"),
            "artifact_file_size_bytes": report.get("artifact", {}).get("file_size_bytes"),
            "artifact_sha256": (
                expected_artifact.get("sha256") if expected_artifact is not None else None
            ),
            "max_context": config.get("max_context"),
            "kv_capacity": memory.get("kv_capacity"),
            "kv_capacity_mode": memory.get("kv_capacity_mode"),
            "kv_capacity_page_groups": memory.get("kv_capacity_page_groups"),
            "kv_capacity_max_page_groups": memory.get("kv_capacity_max_page_groups"),
            "kv_capacity_measurement_kind": (
                capacity_evidence["measurement_kind"] if capacity_evidence else None
            ),
            "kv_capacity_binding_constraint": (
                capacity_evidence["binding_constraint"] if capacity_evidence else None
            ),
            "kv_capacity_uncensored": (
                capacity_evidence["uncensored"] if capacity_evidence else None
            ),
            "memory_limited_capacity_tokens": (
                capacity_evidence["memory_limited_capacity_tokens"] if capacity_evidence else None
            ),
            "resolved_effective_maximum_tokens": (
                capacity_evidence["resolved_effective_maximum_tokens"]
                if capacity_evidence else None
            ),
            "prefill_chunk": config.get("prefill_chunk"),
            "concurrency": config.get("concurrency"),
            "pending_timeout_ms": config.get("pending_timeout_ms"),
            "kv_cache_format": config.get("kv_cache_format"),
            "kv_value_group": config.get("kv_value_group"),
            "kv_plane_layouts": config.get("kv_plane_layouts"),
            "q4_activation_bits": config.get("q4_activation_bits"),
            "q4_prefill_cta_profile": config.get("q4_prefill_cta_profile"),
            "w8_activation_bits": config.get("w8_activation_bits"),
            "fp8_qk_wmma_enabled": config.get("fp8_qk_wmma_enabled"),
            "fp8_qk_wmma_profile": config.get("fp8_qk_wmma_profile"),
            "fp8_qk_wmma_t1_min_context": config.get("fp8_qk_wmma_t1_min_context"),
            "fp8_qk_wmma_t2_min_context": config.get("fp8_qk_wmma_t2_min_context"),
            "xattention_qualification": config.get("xattention_qualification"),
            "xattention_profile": config.get("xattention_profile"),
            "xattention_find_block": config.get("xattention_find_block"),
            "xattention_stride": config.get("xattention_stride"),
            "xattention_tau_permille": config.get("xattention_tau_permille"),
            "spec": config.get("spec"),
            "draft_tokens": config.get("draft_tokens"),
            "dflash_verify_width_requested": config.get("dflash_verify_width_requested"),
            "dflash_verify_width": config.get("dflash_verify_width"),
            "dflash_topology": topology,
            "proposal_head": config.get("proposal_head"),
            "decode_path": config.get("decode_path"),
            "decode_graph_primed": config.get("decode_graph_prime", {}).get("primed"),
            "decode_graph_prime_output_tokens": config.get("decode_graph_prime", {}).get(
                "output_tokens"
            ),
            "repetitions": config.get("repetitions"),
            "warmup": config.get("warmup"),
            "retain_token_ids": config.get("retain_token_ids"),
            "load_seconds": load.get("load_seconds") if performance_eligible else None,
            "upload_seconds": load.get("upload_seconds") if performance_eligible else None,
            "artifact_bytes_read": load.get("artifact_bytes_read"),
            "host_to_device_bytes": load.get("host_to_device_bytes"),
            "peak_staging_bytes": load.get("peak_staging_bytes"),
            "kv_payload_bytes": memory.get("kv_payload_bytes"),
            "weights_capacity_bytes": weights_memory.get("capacity_bytes"),
            "sequence_capacity_bytes": sequence_memory.get("capacity_bytes"),
            "workspace_capacity_bytes": workspace_memory.get("capacity_bytes"),
            "request_transient_capacity_bytes": request_transient_memory.get("capacity_bytes"),
            "device_graph_allowance_bytes": memory.get("device_graph_allowance_bytes"),
            "device_graph_observed_bytes": memory.get("device_graph_observed_bytes"),
            "minimum_runtime_reservation_bytes": memory.get("minimum_runtime_reservation_bytes"),
            "kv_capacity_increment_bytes": memory.get("kv_capacity_increment_bytes"),
            "runtime_reservation_bytes": memory.get("runtime_reservation_bytes"),
            "available_after_weights_bytes": memory.get("available_after_weights_bytes"),
            "available_after_startup_bytes": memory.get("available_after_startup_bytes"),
            "kv_capacity_headroom_bytes": memory.get("kv_capacity_headroom_bytes"),
            "planned_slack_bytes": memory.get("planned_slack_bytes"),
            "workspace_peak_bytes": test.get("workspace_peak_bytes"),
            "workspace_allocator_peak_bytes": test.get("workspace_allocator_peak_bytes"),
            "prefill_tok_s_mean": test.get("prefill_tok_s_mean") if performance_eligible else None,
            "prefill_tok_s_stddev": (test.get("prefill_tok_s_stddev")
                                      if performance_eligible else None),
            "decode_output_tok_s_mean": (test.get("decode_output_tok_s_mean")
                                          if performance_eligible else None),
            "decode_output_tok_s_stddev": (test.get("decode_output_tok_s_stddev")
                                            if performance_eligible else None),
            "decode_engine_tok_s_mean": (test.get("decode_engine_tok_s_mean")
                                          if performance_eligible else None),
            "decode_engine_tok_s_stddev": (test.get("decode_engine_tok_s_stddev")
                                            if performance_eligible else None),
            "whole_output_tok_s_mean": (test.get("whole_output_tok_s_mean")
                                          if performance_eligible else None),
            "whole_output_tok_s_stddev": (test.get("whole_output_tok_s_stddev")
                                            if performance_eligible else None),
            "prepare_seconds_mean": (test.get("prepare_seconds_mean")
                                      if performance_eligible else None),
            "prefill_seconds_mean": (test.get("prefill_seconds_mean")
                                      if performance_eligible else None),
            "decode_seconds_mean": (test.get("decode_seconds_mean")
                                     if performance_eligible else None),
            "total_seconds_mean": (test.get("total_seconds_mean")
                                    if performance_eligible else None),
            "spec_acceptance_rate": speculative.get("acceptance_rate"),
            "spec_acceptance_length": speculative.get("acceptance_length"),
            "spec_rounds": speculative.get("rounds"),
            "spec_drafted_tokens": speculative.get("drafted_tokens"),
            "spec_accepted_tokens": speculative.get("accepted_tokens"),
            "spec_fallback_steps": speculative.get("fallback_steps"),
            "spec_accepted_per_position": json.dumps(
                speculative.get("accepted_per_position", []), separators=(",", ":")
            ),
            "gpu_name": environment.get("gpu_name"),
            "architecture_name": environment.get("architecture_name"),
            "hip_runtime_version": environment.get("hip_runtime_version"),
            "hip_driver_version": environment.get("hip_driver_version"),
            "device_id": environment.get("device_id"),
        }
        rows.append(row)
    return rows


def write_summary(rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        durable_replace_text(out_dir / "summary.csv", "")
        durable_replace_text(out_dir / "summary.json", "[]\n")
        return
    fieldnames = list(rows[0].keys())
    csv_output = io.StringIO(newline="")
    writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    durable_replace_text(out_dir / "summary.csv", csv_output.getvalue())
    durable_replace_json(out_dir / "summary.json", rows)


def validate_dflash_diagnostic_raw(
    path: Path, draft_tokens: int, verify_width: int | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DFlash diagnostic root must be an object")
    if (
        payload.get("artifact_type") != "ninfer_dflash_proposal_selector_raw"
        or payload.get("schema_version") != 2
        or payload.get("diagnostic_only") is not True
        or payload.get("timing_eligible") is not False
    ):
        raise ValueError("unsupported DFlash diagnostic identity")
    logits = payload.get("logits")
    proposal = payload.get("proposal")
    reject = payload.get("reject")
    by_depth = payload.get("by_depth")
    if not all(isinstance(section, dict) for section in (logits, proposal, reject, by_depth)):
        raise ValueError("DFlash diagnostic is missing a counter section")
    if (
        logits.get("drafts") != draft_tokens
        or not _nonnegative_integer(logits.get("rows"))
        or logits.get("rows") == 0
    ):
        raise ValueError("DFlash diagnostic logits profile does not match the requested draft K")
    for key in ("elements", "nan", "inf"):
        if not _nonnegative_integer(logits.get(key)):
            raise ValueError(f"DFlash diagnostic has invalid logits {key}")
    if (
        logits["elements"] != logits["rows"] * draft_tokens
        or logits["nan"] != 0
        or logits["inf"] != 0
    ):
        raise ValueError("DFlash diagnostic logits must be nonempty and finite")
    if not _finite_number(logits.get("finite_min")) or not _finite_number(logits.get("finite_max")):
        raise ValueError("DFlash diagnostic lacks a finite logits range")
    counter_keys = (
        "hops", "hits", "in_tree", "in_top16", "in_top64", "in_top256",
        "in_draft_head", "absent_from_draft_head",
    )
    if any(not _nonnegative_integer(proposal.get(key)) for key in counter_keys):
        raise ValueError("DFlash diagnostic has invalid proposal counters")
    if proposal["hops"] == 0:
        raise ValueError("DFlash diagnostic contains no proposal hops")
    if proposal["hits"] > proposal["hops"]:
        raise ValueError("DFlash diagnostic hits exceed hops")
    if not (
        proposal["hits"] <= proposal["in_tree"] <= proposal["hops"]
        and proposal["in_top16"] <= proposal["in_top64"]
        <= proposal["in_top256"] <= proposal["hops"]
    ):
        raise ValueError("DFlash diagnostic proposal membership counters are incoherent")
    if proposal["in_draft_head"] + proposal["absent_from_draft_head"] != proposal["hops"]:
        raise ValueError("DFlash diagnostic draft-head membership is incoherent")
    reject_keys = (
        "count", "in_tree", "in_top16", "in_top64", "in_top256", "in_draft_head",
        "absent_from_draft_head",
    )
    if any(not _nonnegative_integer(reject.get(key)) for key in reject_keys):
        raise ValueError("DFlash diagnostic has invalid reject counters")
    if reject["count"] > proposal["hops"]:
        raise ValueError("DFlash diagnostic rejects exceed hops")
    if not (
        reject["in_tree"] <= reject["count"]
        and reject["in_top16"] <= reject["in_top64"]
        <= reject["in_top256"] <= reject["count"]
    ):
        raise ValueError("DFlash diagnostic reject membership counters are incoherent")
    if reject["in_draft_head"] + reject["absent_from_draft_head"] != reject["count"]:
        raise ValueError("DFlash diagnostic reject membership is incoherent")
    for key in ("hops", "hits", "top16", "top256", "rejects"):
        values = by_depth.get(key)
        if (
            not isinstance(values, list)
            or len(values) != draft_tokens
            or any(not _nonnegative_integer(value) for value in values)
        ):
            raise ValueError(f"DFlash diagnostic has invalid by-depth {key}")
    if sum(by_depth["hops"]) != proposal["hops"]:
        raise ValueError("DFlash diagnostic by-depth hops are incoherent")
    if sum(by_depth["hits"]) != proposal["hits"]:
        raise ValueError("DFlash diagnostic by-depth hits are incoherent")
    if sum(by_depth["top16"]) != proposal["in_top16"]:
        raise ValueError("DFlash diagnostic by-depth top16 is incoherent")
    if sum(by_depth["top256"]) != proposal["in_top256"]:
        raise ValueError("DFlash diagnostic by-depth top256 is incoherent")
    if sum(by_depth["rejects"]) != reject["count"]:
        raise ValueError("DFlash diagnostic by-depth rejects are incoherent")
    trace = payload.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("DFlash diagnostic has no per-round proposal trace")
    traced_hops = 0
    for index, round_trace in enumerate(trace):
        if not isinstance(round_trace, dict) or round_trace.get("round_index") != index:
            raise ValueError("DFlash diagnostic round indexes are not contiguous")
        proposal_ids = round_trace.get("proposal_ids")
        parent_index = round_trace.get("parent_index")
        target_tokens = round_trace.get("target_licensed_tokens")
        if (
            not isinstance(proposal_ids, list)
            or not 2 <= len(proposal_ids) <= 16
            or (verify_width is not None and len(proposal_ids) != verify_width)
            or any(type(token) is not int or not 0 <= token < PUBLIC_TOKEN_DOMAIN
                   for token in proposal_ids)
            or not isinstance(parent_index, list)
            or len(parent_index) != len(proposal_ids)
            or parent_index[0] != -1
            or any(type(parent) is not int or not 0 <= parent < child
                   for child, parent in enumerate(parent_index[1:], start=1))
            or not isinstance(target_tokens, list)
            or not 1 <= len(target_tokens) <= len(proposal_ids)
            or any(type(token) is not int or not 0 <= token < PUBLIC_TOKEN_DOMAIN
                   for token in target_tokens)
        ):
            raise ValueError("DFlash diagnostic has an invalid aligned round trace")
        traced_hops += len(target_tokens)
    if traced_hops != proposal["hops"]:
        raise ValueError("DFlash diagnostic trace hops are incoherent")
    return payload


def bind_dflash_diagnostic(
    raw_path: Path,
    evidence_path: Path,
    *,
    artifact: dict[str, Any],
    bench: dict[str, Any],
    report_path: Path,
    case: BenchCase,
    concurrency: int,
) -> dict[str, Any]:
    draft_tokens = int(_case_option(case, "--draft-tokens") or "0")
    requested_width = int(_case_option(case, "--dflash-verify-width") or "0")
    resolved_width = resolved_dflash_verify_width(draft_tokens, requested_width)
    raw = validate_dflash_diagnostic_raw(raw_path, draft_tokens, resolved_width)
    report_config = json.loads(report_path.read_text(encoding="utf-8")).get("config", {})
    compiled_profile = {
        key: report_config.get(key)
        for key in (
            "kv_cache_format", "kv_value_group", "q4_activation_bits",
            "q4_prefill_cta_profile", "w8_activation_bits",
            "fp8_qk_wmma_enabled", "fp8_qk_wmma_profile",
            "fp8_qk_wmma_t1_min_context", "fp8_qk_wmma_t2_min_context",
        )
    }
    if any(value is None for value in compiled_profile.values()):
        raise ValueError("DFlash diagnostic benchmark report lacks compiled profile provenance")
    evidence = {
        "artifact_type": "ninfer_dflash_proposal_selector_evidence",
        "schema_version": 2,
        "diagnostic_only": True,
        "timing_eligible": False,
        "artifact": artifact,
        "benchmark_executable": bench,
        "profile": {
            "spec": "dflash",
            "draft_tokens": draft_tokens,
            "dflash_verify_width_requested": int(
                _case_option(case, "--dflash-verify-width") or "0"
            ),
            "dflash_verify_width": resolved_dflash_verify_width(
                draft_tokens, int(_case_option(case, "--dflash-verify-width") or "0")
            ),
            "proposal_head": "optimized" if "--lm-head-draft" in case.args else "full",
            "use_device_graph": "--no-device-graph" not in case.args,
            "concurrency": concurrency,
            "compiled": compiled_profile,
        },
        "benchmark_report": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "raw_diagnostic": {
            "path": str(raw_path),
            "sha256": file_sha256(raw_path),
            "report": raw,
        },
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    durable_replace_json(evidence_path, evidence)
    return evidence


def validate_bound_diagnostic(
    evidence_path: Path,
    *,
    artifact: dict[str, Any],
    bench: dict[str, Any],
    report_path: Path,
    case: BenchCase,
    concurrency: int,
) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        not isinstance(evidence, dict)
        or evidence.get("artifact_type") != "ninfer_dflash_proposal_selector_evidence"
        or evidence.get("schema_version") != 2
        or evidence.get("artifact") != artifact
        or evidence.get("benchmark_executable") != bench
    ):
        raise ValueError("DFlash diagnostic provenance does not match this matrix")
    report = evidence.get("benchmark_report", {})
    raw = evidence.get("raw_diagnostic", {})
    if report != {"path": str(report_path), "sha256": file_sha256(report_path)}:
        raise ValueError("DFlash diagnostic benchmark report identity changed")
    raw_path = Path(raw.get("path", ""))
    if raw.get("sha256") != file_sha256(raw_path):
        raise ValueError("DFlash raw diagnostic identity changed")
    expected_profile = {
        "spec": "dflash",
        "draft_tokens": int(_case_option(case, "--draft-tokens") or "0"),
        "dflash_verify_width_requested": int(
            _case_option(case, "--dflash-verify-width") or "0"
        ),
        "dflash_verify_width": resolved_dflash_verify_width(
            int(_case_option(case, "--draft-tokens") or "0"),
            int(_case_option(case, "--dflash-verify-width") or "0"),
        ),
        "proposal_head": "optimized" if "--lm-head-draft" in case.args else "full",
        "use_device_graph": "--no-device-graph" not in case.args,
        "concurrency": concurrency,
        "compiled": {
            key: json.loads(report_path.read_text(encoding="utf-8")).get("config", {}).get(key)
            for key in (
                "kv_cache_format", "kv_value_group", "q4_activation_bits", "w8_activation_bits",
                "fp8_qk_wmma_enabled", "fp8_qk_wmma_profile",
                "fp8_qk_wmma_t1_min_context", "fp8_qk_wmma_t2_min_context",
            )
        },
    }
    if evidence.get("profile") != expected_profile:
        raise ValueError("DFlash diagnostic execution profile changed")
    validate_dflash_diagnostic_raw(
        raw_path, expected_profile["draft_tokens"], expected_profile["dflash_verify_width"]
    )
    return evidence


def write_dflash_determinism(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    *,
    artifact: dict[str, Any],
    bench: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare two isolated syncing runs at the complete aligned proposal/target boundary."""

    failures: list[dict[str, Any]] = []
    diagnostics = sorted(
        (record for record in records if record.get("suite") == "dflash_selector_diagnostic"),
        key=lambda record: record["case"],
    )
    comparisons: list[dict[str, Any]] = []
    if len(diagnostics) != 2:
        failures.append({"error": f"expected two repeated DFlash diagnostics, found {len(diagnostics)}"})
    else:
        loaded: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for record in diagnostics:
            evidence_path = Path(record["bound_diagnostic"])
            report_path = Path(record["report"])
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_identity = evidence.get("benchmark_report")
            raw_identity = evidence.get("raw_diagnostic")
            embedded_raw = evidence.get("raw_diagnostic", {}).get("report")
            if (
                evidence.get("artifact_type")
                != "ninfer_dflash_proposal_selector_evidence"
                or evidence.get("schema_version") != 2
                or evidence.get("artifact") != artifact
                or evidence.get("benchmark_executable") != bench
                or report_identity
                != {"path": str(report_path), "sha256": file_sha256(report_path)}
                or not isinstance(raw_identity, dict)
                or not isinstance(embedded_raw, dict)
            ):
                raise ValueError("DFlash determinism evidence provenance is invalid")
            raw_path = Path(raw_identity.get("path", ""))
            if raw_identity.get("sha256") != file_sha256(raw_path):
                raise ValueError("DFlash determinism raw diagnostic identity changed")
            raw = validate_dflash_diagnostic_raw(
                raw_path, evidence["profile"]["draft_tokens"],
                evidence["profile"].get("dflash_verify_width"),
            )
            if raw != embedded_raw:
                raise ValueError("DFlash determinism embedded trace differs from raw bytes")
            loaded.append((evidence, report, raw))
        first_evidence, first_report, first_raw = loaded[0]
        second_evidence, second_report, second_raw = loaded[1]
        first_tokens = [rep["generated_token_ids_by_lane"]
                        for test in first_report["tests"] for rep in test["reps"]]
        second_tokens = [rep["generated_token_ids_by_lane"]
                         for test in second_report["tests"] for rep in test["reps"]]
        profile_exact = first_evidence["profile"] == second_evidence["profile"]
        proposal_trace_exact = first_raw["trace"] == second_raw["trace"]
        target_output_exact = first_tokens == second_tokens
        comparison = {
            "profile_exact": profile_exact,
            "proposal_trace_exact": proposal_trace_exact,
            "target_output_exact": target_output_exact,
            "round_count": len(first_raw["trace"]),
            "aligned_target_hops": sum(
                len(round_trace["target_licensed_tokens"])
                for round_trace in first_raw["trace"]
            ),
            "first_evidence": {
                "path": diagnostics[0]["bound_diagnostic"],
                "sha256": file_sha256(Path(diagnostics[0]["bound_diagnostic"])),
            },
            "second_evidence": {
                "path": diagnostics[1]["bound_diagnostic"],
                "sha256": file_sha256(Path(diagnostics[1]["bound_diagnostic"])),
            },
            "exact": profile_exact and proposal_trace_exact and target_output_exact,
        }
        comparisons.append(comparison)
        if not comparison["exact"]:
            failures.append({"error": "repeated DFlash proposal/target trace differs", **comparison})
    payload = {
        "artifact_type": "ninfer_dflash_proposal_determinism",
        "schema_version": 1,
        "criterion": (
            "exact per-round proposal ids, parent topology, aligned licensed target tokens, "
            "and final generated target tokens across two isolated executions"
        ),
        "artifact": artifact,
        "benchmark_executable": bench,
        "comparisons": comparisons,
        "pass": not failures and len(comparisons) == 1,
    }
    durable_replace_json(out_dir / "dflash-proposal-determinism.json", payload)
    return payload, failures


def write_dflash_quality_evidence(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    parity: dict[str, Any],
    determinism: dict[str, Any],
    *,
    artifact: dict[str, Any],
    bench: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Assemble the generated-output DFlash quality gate from bound raw evidence."""

    failures: list[dict[str, Any]] = []
    if parity.get("artifact") != artifact or parity.get("benchmark_executable") != bench:
        failures.append({"error": "ordinary/DFlash parity provenance mismatch"})
    if determinism.get("artifact") != artifact or determinism.get("benchmark_executable") != bench:
        failures.append({"error": "proposal determinism provenance mismatch"})

    parity_rows = parity.get("comparisons")
    if not isinstance(parity_rows, list):
        parity_rows = []
    observed_concurrency = {
        row.get("concurrency") for row in parity_rows
        if isinstance(row, dict) and row.get("exact") is True
    }
    parity_complete = (
        parity.get("pass") is True
        and len(parity_rows) == len(PRODUCT_CONCURRENCIES)
        and observed_concurrency == set(PRODUCT_CONCURRENCIES)
    )
    if not parity_complete:
        failures.append({"error": "exact ordinary/DFlash target-output parity is incomplete"})

    determinism_rows = determinism.get("comparisons")
    deterministic = (
        determinism.get("pass") is True
        and isinstance(determinism_rows, list)
        and len(determinism_rows) == 1
        and determinism_rows[0].get("exact") is True
    )
    if not deterministic:
        failures.append({"error": "repeated proposal/target trace determinism is incomplete"})

    diagnostic_records = sorted(
        (record for record in records if record.get("suite") == "dflash_selector_diagnostic"),
        key=lambda record: record.get("case", ""),
    )
    diagnostics: list[dict[str, Any]] = []
    if len(diagnostic_records) != 2:
        failures.append({"error": f"expected two bound selector diagnostics, found {len(diagnostic_records)}"})
    else:
        expected_profile: dict[str, Any] | None = None
        for record in diagnostic_records:
            try:
                evidence_path = Path(record["bound_diagnostic"])
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                if (
                    evidence.get("artifact_type") != "ninfer_dflash_proposal_selector_evidence"
                    or evidence.get("schema_version") != 2
                    or evidence.get("artifact") != artifact
                    or evidence.get("benchmark_executable") != bench
                ):
                    raise ValueError("bound diagnostic provenance mismatch")
                profile = evidence.get("profile")
                if not isinstance(profile, dict):
                    raise ValueError("bound diagnostic profile is missing")
                if expected_profile is None:
                    expected_profile = profile
                elif profile != expected_profile:
                    raise ValueError("repeated diagnostic profiles differ")
                raw_identity = evidence.get("raw_diagnostic")
                if not isinstance(raw_identity, dict):
                    raise ValueError("bound diagnostic raw identity is missing")
                report_path = Path(record["report"])
                if evidence.get("benchmark_report") != {
                    "path": str(report_path),
                    "sha256": file_sha256(report_path),
                }:
                    raise ValueError("bound diagnostic benchmark report identity changed")
                raw_path = Path(raw_identity.get("path", ""))
                if raw_identity.get("sha256") != file_sha256(raw_path):
                    raise ValueError("bound diagnostic raw bytes changed")
                raw = validate_dflash_diagnostic_raw(
                    raw_path, profile["draft_tokens"], profile["dflash_verify_width"]
                )
                if raw_identity.get("report") != raw:
                    raise ValueError("bound diagnostic embedded raw report differs")
                diagnostics.append({
                    "evidence": {"path": str(evidence_path), "sha256": file_sha256(evidence_path)},
                    "raw": {"path": str(raw_path), "sha256": raw_identity["sha256"]},
                    "profile": profile,
                    "finite_logit_elements": raw["logits"]["elements"],
                    "proposal_rounds": len(raw["trace"]),
                    "aligned_target_hops": sum(
                        len(round_trace["target_licensed_tokens"])
                        for round_trace in raw["trace"]
                    ),
                })
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
                failures.append({"error": f"invalid bound selector diagnostic: {error}"})

        if expected_profile is not None:
            expected_k = expected_profile.get("draft_tokens")
            expected_width = expected_profile.get("dflash_verify_width")
            if any(
                not isinstance(row, dict)
                or row.get("draft_tokens") != expected_k
                or row.get("dflash_verify_width") != expected_width
                for row in parity_rows
            ):
                failures.append({"error": "target-output parity and proposal profiles differ"})

    payload = {
        "artifact_type": "ninfer_dflash_generated_quality_evidence",
        "schema_version": 1,
        "artifact": artifact,
        "benchmark_executable": bench,
        "scope": "generated DFlash proposal and target-output behavior",
        "target_output_gate": {
            "criterion": "exact ordinary/DFlash generated tokens for every repetition and lane",
            "concurrency": list(PRODUCT_CONCURRENCIES),
            "evidence": {
                "path": str(out_dir / "greedy-token-parity.json"),
                "sha256": file_sha256(out_dir / "greedy-token-parity.json"),
            },
            "pass": parity_complete,
        },
        "proposal_gate": {
            "criterion": (
                "exact aligned proposal IDs, rooted parent topology, licensed target tokens, "
                "and final target output across two isolated runs"
            ),
            "diagnostics": diagnostics,
            "determinism_evidence": {
                "path": str(out_dir / "dflash-proposal-determinism.json"),
                "sha256": file_sha256(out_dir / "dflash-proposal-determinism.json"),
            },
            "pass": deterministic and len(diagnostics) == 2,
        },
        "nll_diagnostic": {
            "status": "not_applicable",
            "reason": (
                "the DFlash companion changes proposal execution, not the teacher-forced target "
                "distribution; base-artifact BF16-source NLL is gated by the paired PPL campaign"
            ),
        },
        "pass": not failures,
    }
    durable_replace_json(out_dir / "dflash-generated-quality.json", payload)
    return payload, failures


def write_dflash_greedy_parity(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    *,
    artifact: dict[str, Any],
    bench: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controls: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for record in records:
        if record["parity_role"] != "ordinary":
            continue
        concurrency = record["concurrency"]
        if concurrency in controls:
            failures.append({"concurrency": concurrency, "error": "duplicate ordinary control"})
        else:
            controls[concurrency] = record
    candidates = [record for record in records if record["parity_role"] == "dflash"]
    candidates.sort(key=lambda record: (record["concurrency"], record.get("case", "")))
    comparisons: list[dict[str, Any]] = []
    candidate_concurrency = {record["concurrency"] for record in candidates}
    for concurrency in sorted(set(controls) - candidate_concurrency):
        failures.append({"concurrency": concurrency, "error": "ordinary control has no DFlash pair"})
    for candidate_record in candidates:
        concurrency = candidate_record["concurrency"]
        if concurrency not in controls:
            failures.append({
                "concurrency": concurrency,
                "candidate_report": candidate_record["report"],
                "error": "missing ordinary control",
            })
            continue
        control_path = Path(controls[concurrency]["report"])
        candidate_path = Path(candidate_record["report"])
        control = json.loads(control_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_config = candidate.get("config", {})
        draft_tokens = candidate_config.get("draft_tokens")
        verify_width = candidate_config.get("dflash_verify_width")
        if type(draft_tokens) is not int or type(verify_width) is not int:
            raise ValueError("DFlash parity report lacks integer K/W profile")
        mismatch_count = 0
        compared_tokens = 0
        first_mismatch: dict[str, Any] | None = None
        if len(control.get("tests", [])) != len(candidate.get("tests", [])):
            mismatch_count = 1
            first_mismatch = {"reason": "test count differs"}
        else:
            for control_test, candidate_test in zip(control["tests"], candidate["tests"], strict=True):
                geometry = tuple(control_test.get(key) for key in ("label", "kind", "n_prompt", "n_gen"))
                candidate_geometry = tuple(
                    candidate_test.get(key) for key in ("label", "kind", "n_prompt", "n_gen")
                )
                if geometry != candidate_geometry:
                    mismatch_count += 1
                    first_mismatch = first_mismatch or {"reason": "test geometry differs"}
                    continue
                control_runs = [rep["generated_token_ids_by_lane"] for rep in control_test["reps"]]
                candidate_runs = [rep["generated_token_ids_by_lane"] for rep in candidate_test["reps"]]
                expected = control_runs[0]
                all_runs = [("ordinary", i, run) for i, run in enumerate(control_runs)] + [
                    ("dflash", i, run) for i, run in enumerate(candidate_runs)
                ]
                for route, repetition, lanes in all_runs:
                    for lane, (wanted, actual) in enumerate(zip(expected, lanes, strict=True)):
                        compared_tokens += len(wanted)
                        for position, (left, right) in enumerate(zip(wanted, actual, strict=True)):
                            if left != right:
                                mismatch_count += 1
                                first_mismatch = first_mismatch or {
                                    "test": geometry[0], "route": route, "repetition": repetition,
                                    "lane": lane, "position": position,
                                    "ordinary_token": left, "observed_token": right,
                                }
        exact = mismatch_count == 0
        comparison = {
            "concurrency": concurrency,
            "draft_tokens": draft_tokens,
            "dflash_verify_width": verify_width,
            "dflash_topology": resolved_dflash_topology(draft_tokens, verify_width),
            "ordinary_report": {"path": str(control_path), "sha256": file_sha256(control_path)},
            "dflash_report": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
            "ordinary_profile": {
                key: control.get("config", {}).get(key)
                for key in (
                    "spec", "draft_tokens", "dflash_verify_width_requested",
                    "dflash_verify_width", "proposal_head", "use_device_graph", "concurrency",
                    "kv_value_group", "q4_activation_bits", "q4_prefill_cta_profile",
                    "w8_activation_bits",
                    "fp8_qk_wmma_profile",
                )
            },
            "dflash_profile": {
                key: candidate.get("config", {}).get(key)
                for key in (
                    "spec", "draft_tokens", "dflash_verify_width_requested",
                    "dflash_verify_width", "proposal_head", "use_device_graph", "concurrency",
                    "kv_value_group", "q4_activation_bits", "q4_prefill_cta_profile",
                    "w8_activation_bits",
                    "fp8_qk_wmma_profile",
                )
            },
            "compared_tokens": compared_tokens,
            "mismatch_count": mismatch_count,
            "exact": exact,
            "first_mismatch": first_mismatch,
        }
        comparisons.append(comparison)
        if not exact:
            failures.append({"error": "greedy tokens differ", **comparison})
    payload = {
        "artifact_type": "ninfer_dflash_ordinary_greedy_parity",
        "schema_version": 1,
        "criterion": "exact token identity for every retained repetition and concurrency lane",
        "artifact": artifact,
        "benchmark_executable": bench,
        "comparisons": comparisons,
        "pass": not failures,
    }
    durable_replace_json(out_dir / "greedy-token-parity.json", payload)
    return payload, failures


def write_dflash_shortlist(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    parity: dict[str, Any],
    *,
    artifact: dict[str, Any],
    bench: dict[str, Any],
    provenance_stable: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write the cross-K decision artifact from independently retained evidence."""

    failures: list[dict[str, Any]] = []
    if not provenance_stable:
        failures.append({"error": "artifact or benchmark executable changed during campaign"})
    parity_by_k = {
        comparison.get("draft_tokens"): comparison
        for comparison in parity.get("comparisons", [])
        if comparison.get("concurrency") == 1
    }
    performance_by_k = {
        row.get("draft_tokens"): row
        for row in rows
        if row.get("suite") == "dflash_shortlist_decode"
        and row.get("concurrency") == 1
        and row.get("label") == "pp8192+tg256"
    }
    diagnostic_by_k = {
        record.get("dflash_draft_tokens"): record
        for record in records
        if record.get("suite") == "dflash_shortlist_repair_diagnostic"
        and record.get("concurrency") == 1
    }
    controls = [
        row for row in rows
        if row.get("suite") == "dflash_shortlist_control"
        and row.get("concurrency") == 1
        and row.get("label") == "pp8192+tg256"
    ]
    if len(controls) != 1:
        failures.append({"error": f"expected one ordinary control row, found {len(controls)}"})

    candidates: list[dict[str, Any]] = []
    for profile in dflash_shortlist_profiles():
        k = profile["draft_tokens_requested"]
        row = performance_by_k.get(k)
        comparison = parity_by_k.get(k)
        diagnostic_record = diagnostic_by_k.get(k)
        reasons: list[str] = []
        if row is None:
            reasons.append("missing performance row")
        elif (
            row.get("dflash_verify_width_requested") != profile["verify_width_requested"]
            or row.get("dflash_verify_width") != profile["verify_width_resolved"]
            or row.get("dflash_topology") != profile["topology"]
            or row.get("proposal_head") != "optimized"
            or row.get("decode_path") != "dflash_device_graph"
        ):
            reasons.append("performance execution profile mismatch")
        if comparison is None:
            reasons.append("missing ordinary-control parity row")
        elif not comparison.get("exact"):
            reasons.append("ordinary-control greedy token mismatch")

        diagnostic_summary: dict[str, Any] | None = None
        if diagnostic_record is None:
            reasons.append("missing first-reject diagnostic")
        else:
            try:
                evidence_path = Path(diagnostic_record["bound_diagnostic"])
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                if (
                    evidence.get("artifact_type")
                    != "ninfer_dflash_proposal_selector_evidence"
                    or evidence.get("schema_version") != 2
                    or evidence.get("artifact") != artifact
                    or evidence.get("benchmark_executable") != bench
                    or evidence.get("timing_eligible") is not False
                ):
                    raise ValueError("diagnostic provenance mismatch")
                report_path = Path(diagnostic_record["report"])
                if evidence.get("benchmark_report") != {
                    "path": str(report_path),
                    "sha256": file_sha256(report_path),
                }:
                    raise ValueError("diagnostic benchmark report identity changed")
                evidence_profile = evidence.get("profile", {})
                if (
                    evidence_profile.get("draft_tokens") != k
                    or evidence_profile.get("dflash_verify_width_requested")
                    != profile["verify_width_requested"]
                    or evidence_profile.get("dflash_verify_width")
                    != profile["verify_width_resolved"]
                    or evidence_profile.get("proposal_head") != "optimized"
                    or evidence_profile.get("use_device_graph") is not False
                ):
                    raise ValueError("diagnostic execution profile mismatch")
                raw_identity = evidence["raw_diagnostic"]
                raw_path = Path(raw_identity["path"])
                if raw_identity.get("sha256") != file_sha256(raw_path):
                    raise ValueError("diagnostic raw bytes changed")
                raw = validate_dflash_diagnostic_raw(
                    raw_path, k, profile["verify_width_resolved"]
                )
                if raw_identity.get("report") != raw:
                    raise ValueError("diagnostic embedded report differs from raw bytes")
                proposal_hops = raw["proposal"]["hops"]
                first_rejects = raw["reject"]["count"]
                if not _nonnegative_integer(proposal_hops) or proposal_hops == 0:
                    raise ValueError("diagnostic has no proposal hops")
                if not _nonnegative_integer(first_rejects) or first_rejects > proposal_hops:
                    raise ValueError("diagnostic has invalid first-reject count")
                diagnostic_summary = {
                    "timing_eligible": False,
                    "timings_discarded": True,
                    "evidence": {
                        "path": str(evidence_path),
                        "sha256": file_sha256(evidence_path),
                    },
                    "proposal_hops": proposal_hops,
                    "first_rejects": first_rejects,
                    "first_reject_rate_per_proposal_hop": first_rejects / proposal_hops,
                    "proposal": raw["proposal"],
                    "reject": raw["reject"],
                    "by_depth": raw["by_depth"],
                }
            except (
                json.JSONDecodeError, KeyError, OSError, TypeError, ValueError,
                ZeroDivisionError,
            ):
                reasons.append("invalid first-reject diagnostic")

        performance: dict[str, Any] | None = None
        scalar_metrics: dict[str, Any] = {}
        if row is not None:
            rounds = row.get("spec_rounds")
            drafted = row.get("spec_drafted_tokens")
            accepted = row.get("spec_accepted_tokens")
            fallback = row.get("spec_fallback_steps")
            speed = row.get("decode_engine_tok_s_mean")
            if (
                not _nonnegative_integer(rounds) or rounds == 0
                or not _nonnegative_integer(drafted) or drafted == 0
                or not _nonnegative_integer(accepted)
                or accepted > drafted
                or not _nonnegative_integer(fallback)
                or not _finite_number(speed, positive=True)
            ):
                reasons.append("invalid performance or speculative counters")
            else:
                attempts = rounds + fallback
                if attempts == 0:
                    reasons.append("no speculative attempts")
                else:
                    scalar_metrics = {
                        "target_equivalent_generated_tok_s": speed,
                        "accepted_tokens_per_round": accepted / rounds,
                        "fallback_rate_per_attempt": fallback / attempts,
                        "first_reject_rate_per_proposal_hop": (
                            diagnostic_summary["first_reject_rate_per_proposal_hop"]
                            if diagnostic_summary is not None else None
                        ),
                    }
                    performance = {
                        "benchmark_report": {
                            "path": row["report"],
                            "sha256": file_sha256(Path(row["report"])),
                        },
                        "prompt_tokens": row["n_prompt"],
                        "requested_generated_tokens": row["n_gen"],
                        "repetitions": row["repetitions"],
                        "warmup": row["warmup"],
                        "accepted_tokens_per_round": scalar_metrics[
                            "accepted_tokens_per_round"
                        ],
                        "target_equivalent_generated_tok_s": speed,
                        "output_generated_tok_s": row["decode_output_tok_s_mean"],
                        "acceptance_rate": row["spec_acceptance_rate"],
                        "acceptance_length": row["spec_acceptance_length"],
                        "rounds": rounds,
                        "drafted_tokens": drafted,
                        "accepted_tokens": accepted,
                        "fallback_steps": fallback,
                        "fallback_rate_per_attempt": scalar_metrics[
                            "fallback_rate_per_attempt"
                        ],
                        "decode_seconds_mean": row["decode_seconds_mean"],
                        "total_seconds_mean": row["total_seconds_mean"],
                        "memory": {
                            key: row[key]
                            for key in (
                                "kv_capacity", "kv_payload_bytes", "weights_capacity_bytes",
                                "sequence_capacity_bytes", "workspace_capacity_bytes",
                                "request_transient_capacity_bytes", "device_graph_allowance_bytes",
                                "workspace_peak_bytes", "workspace_allocator_peak_bytes",
                            )
                        },
                    }

        exact_parity = comparison is not None and comparison.get("exact") is True
        valid = (
            exact_parity and not reasons
            and diagnostic_summary is not None and performance is not None
        )
        candidate = {
            "profile": profile,
            "exact_ordinary_control_parity": exact_parity,
            "valid_for_ranking": valid,
            "exclusion_reasons": reasons,
            "ordinary_control_parity": comparison,
            "performance": performance,
            "first_reject_repair_diagnostic": diagnostic_summary,
            **scalar_metrics,
        }
        candidates.append(candidate)
        if reasons:
            failures.append({"draft_tokens": k, "errors": reasons})

    valid_candidates = [candidate for candidate in candidates if candidate["valid_for_ranking"]]
    ranked = sorted(
        valid_candidates,
        key=lambda candidate: candidate["target_equivalent_generated_tok_s"],
        reverse=True,
    )
    ranking = [
        {
            "speed_rank": index,
            "draft_tokens": candidate["profile"]["draft_tokens_requested"],
            "verify_width_requested": candidate["profile"]["verify_width_requested"],
            "verify_width_resolved": candidate["profile"]["verify_width_resolved"],
            "topology": candidate["profile"]["topology"],
            **{
                key: candidate[key]
                for key in (
                    "target_equivalent_generated_tok_s", "accepted_tokens_per_round",
                    "fallback_rate_per_attempt", "first_reject_rate_per_proposal_hop",
                )
            },
        }
        for index, candidate in enumerate(ranked, start=1)
    ]
    frontier = dflash_shortlist_frontier(valid_candidates)
    frontier_rows = [
        next(row for row in ranking if row["draft_tokens"] == candidate["profile"]["draft_tokens_requested"])
        for candidate in frontier
    ]
    frontier_rows.sort(key=lambda row: row["draft_tokens"])
    payload = {
        "artifact_type": "ninfer_dflash_shortlist",
        "schema_version": DFLASH_SHORTLIST_SCHEMA_VERSION,
        "artifact": artifact,
        "benchmark_executable": bench,
        "workload": {
            "route": "public Engine",
            "concurrency": 1,
            "prompt_tokens": 8192,
            "requested_generated_tokens": 256,
            "proposal_head": "optimized",
            "performance_repetitions": 2,
            "performance_warmup": 1,
            "diagnostic_repetitions": 1,
            "diagnostic_warmup": 0,
        },
        "metric_definitions": {
            "target_equivalent_generated_tok_s": (
                "mean per-repetition (rounds + accepted_tokens + fallback_steps) / "
                "decode_seconds from ninfer_bench"
            ),
            "accepted_tokens_per_round": "accepted_tokens / speculative rounds",
            "fallback_rate_per_attempt": "fallback_steps / (rounds + fallback_steps)",
            "first_reject_rate_per_proposal_hop": (
                "syncing diagnostic first rejects / diagnostic proposal hops; never timing evidence"
            ),
        },
        "ordinary_control": controls[0] if len(controls) == 1 else None,
        "ordinary_control_parity": {
            "path": str(out_dir / "greedy-token-parity.json"),
            "sha256": file_sha256(out_dir / "greedy-token-parity.json"),
            "pass": parity.get("pass"),
        },
        "candidates": candidates,
        "speed_ranking_exact_parity_only": ranking,
        "non_dominated_candidates": frontier_rows,
        "followup": {
            "concurrency": list(PRODUCT_CONCURRENCIES),
            "candidates": [row["draft_tokens"] for row in frontier_rows],
            "selection_rule": (
                "retain the multi-objective frontier; do not collapse acceptance and speed "
                "to one scalar"
            ),
        },
        "pass": not failures and provenance_stable and len(valid_candidates) == 11,
    }
    durable_replace_json(out_dir / "dflash-shortlist.json", payload)
    return payload, failures


def run_command(
    command: Sequence[str], stdout_path: Path, stderr_path: Path,
    environment: dict[str, str] | None = None,
) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stdout=stdout,
            stderr=stderr,
            env=None if environment is None else {**os.environ, **environment},
            check=False,
        )
    return process.returncode


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    artifact: dict[str, Any],
    bench: dict[str, Any],
    cases: Sequence[BenchCase],
    commands: Sequence[dict[str, Any]],
) -> None:
    manifest = {
        "artifact_type": "ninfer_bench_matrix_run",
        "schema_version": MATRIX_SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "preset": args.preset,
        "base_ranking_profile": (
            "spec-none-ordinary" if args.preset == "pareto-whole" else None
        ),
        "base_capacity_profile": (
            "spec-none-ordinary" if args.preset == "pareto-capacity" else None
        ),
        "base_chunk_profile": (
            "spec-none-ordinary" if args.preset == "prefill-chunk" else None
        ),
        "mtp_diagnostic_only": args.preset == "pareto",
        "dflash_draft_tokens": args.dflash_draft_tokens,
        "dflash_verify_width_requested": args.dflash_verify_width,
        "dflash_verify_width": (
            resolved_dflash_verify_width(args.dflash_draft_tokens, args.dflash_verify_width)
            if args.dflash_draft_tokens is not None else 0
        ),
        "dflash_shortlist_profiles": (
            dflash_shortlist_profiles() if args.preset == "dflash-shortlist" else []
        ),
        "repo_root": str(REPO_ROOT),
        "bench": bench,
        "artifact": artifact,
        "required_candidate_identity": (
            "fp8-hybrid-selection-authority" if args.require_fp8_hybrid else None
        ),
        "hybrid_shared_workspace_authority": args.hybrid_width_authority,
        "corpus": str(args.corpus),
        "corpus_tokens": count_corpus_tokens(args.corpus),
        "corpus_sha256": (
            file_sha256(args.corpus) if args.preset in POWER_BOUND_PRESETS else None
        ),
        "dry_run": args.dry_run,
        "prepare_only": args.prepare_only,
        "selected_prefill_chunk": (
            args.prefill_chunk[0] if args.preset in SELECTED_PREFILL_CHUNK_PRESETS else None
        ),
        **(
            {"prefill_chunk_authority": args.prefill_chunk_authority_record}
            if args.prefill_chunk_authority_record is not None else {}
        ),
        **({"post_chunk_capacity_gate": True} if args.require_post_chunk_capacity else {}),
        "power_profile": (
            {
                "required": "auto",
                "sysfs_path": str(R9700_POWER_PROFILE),
                "observed": args.power_profile_observed,
                **(
                    {"rechecked_after": args.power_profile_rechecked_after}
                    if args.preset in POWER_RECHECK_PRESETS else {}
                ),
            }
            if args.preset in POWER_BOUND_PRESETS else None
        ),
        "expected_kv_value_group": args.expected_kv_value_group,
        "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
        "expected_q4_activation_bits": args.expected_q4_activation_bits,
        "expected_w8_activation_bits": args.expected_w8_activation_bits,
        "expected_fp8_qk_wmma_enabled": bool(args.expected_fp8_qk_wmma),
        "expected_fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
        "expected_fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
        "expected_fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
        "expected_xattention_profile": args.expected_xattention_profile,
        "concurrency": list(args.concurrency),
        "resume": args.resume,
        "case_count": len(cases),
        "point_count": len(commands),
        "commands": list(commands),
        "notes": [
            "MTP rows are optional diagnostics and never base chunk/capacity/whole ranking inputs.",
            "Use context_decode and mtp_sweep rows only for MTP diagnostics.",
            "tg rows use a one-token seed and report G decode tokens after the begin token.",
            "DFlash proposal diagnostics synchronize device-to-host copies and are never timing evidence.",
            "DFlash greedy parity requires exact ordinary/DFlash token IDs for every repetition and lane.",
            "DFlash shortlist ranking admits exact-parity rows only and retains a multi-objective frontier.",
        ],
    }
    durable_replace_json(out_dir / "manifest.json", manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=(
            "smoke", "prefill-chunk", "low-context-prefill", "pareto", "pareto-whole", "pareto-feasibility",
            "pareto-capacity", "ordinary-diagnostic",
            "dflash-shortlist", "dflash-pareto", "dflash-feasibility", "dflash-capacity",
            "concurrency",
            "core", "full",
        ),
        default="core",
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument(
        "--weights", type=Path, required=True, help="exact .ninfer artifact passed to the bench"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--prefill-chunk",
        action="append",
        type=int,
        choices=PRODUCTION_PREFILL_CHUNKS,
        metavar="TOKENS",
        help=("chunk candidate for --preset prefill-chunk (default: 1024,2048,4096,8192); "
              "every Pareto/DFlash campaign preset requires one selected value"),
    )
    parser.add_argument(
        "--prefill-chunk-authority",
        type=Path,
        help="validated production chunk-selection record bound into this matrix manifest",
    )
    parser.add_argument(
        "--prefill-prompt",
        type=int,
        choices=PRODUCTION_PREFILL_PROMPTS,
        default=8192,
        help="prompt length for --preset prefill-chunk (default: 8192)",
    )
    parser.add_argument(
        "--dflash-draft-tokens", type=int, choices=range(1, 12),
        help="required startup-fixed DFlash K for DFlash Pareto/capacity presets",
    )
    parser.add_argument(
        "--dflash-verify-width", type=int, choices=(0, *range(2, 17)), default=0,
        help="DFlash verify W for DFlash presets (0 = package default; explicit values must be 2..16)",
    )
    parser.add_argument(
        "--concurrency",
        action="append",
        type=int,
        metavar="N",
        help="fixed Engine concurrency; repeat to sweep (default: 1)",
    )
    parser.add_argument(
        "--expected-kv-value-group",
        type=int,
        choices=(16, 32),
        help="reject reports not emitted by the intended compiled G16/G32 profile",
    )
    parser.add_argument(
        "--expected-q4-activation-bits",
        type=int,
        choices=(4, 8),
        default=8,
        help="reject reports not emitted by the intended compiled A4/A8 Q4 profile (default: 8)",
    )
    parser.add_argument(
        "--expected-w8-activation-bits",
        type=int,
        choices=(8, 16),
        default=8,
        help="reject reports not emitted by the intended W8 A8/BF16 profile (default: 8)",
    )
    parser.add_argument(
        "--expected-fp8-qk-wmma",
        type=int,
        choices=(0, 1),
        default=1,
        help="require the selected T1/T2 FP8-Q/K attention profile (default: 1)",
    )
    parser.add_argument(
        "--expected-xattention-profile",
        choices=XATTENTION_PROFILES,
        default="dense",
        help="require the compile-bound dense or B128/S16/tau900 Text-prefill profile",
    )
    parser.add_argument("--suite", action="append", default=[], help="suite to run; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N selected cases")
    parser.add_argument("--repetitions", type=int, default=None, help="override all case repetitions")
    parser.add_argument("--warmup", type=int, default=None, help="override all case warmup repetitions")
    parser.add_argument("--dry-run", action="store_true", help="write commands but do not execute")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="inspect and hash real inputs and write the executable command matrix without running it",
    )
    parser.add_argument(
        "--require-fp8-hybrid", action="store_true",
        help="require the authority-bound hybrid artifact/receipt and fixed C1..4 gate geometry",
    )
    parser.add_argument(
        "--require-post-chunk-capacity", action="store_true",
        help="require the exact receipt-bound twelve-matrix C1..4 capacity-gate contract",
    )
    parser.add_argument(
        "--hybrid-width-tool", type=Path,
        help="host-only runtime planner executable from the same shared-workspace build",
    )
    parser.add_argument("--resume", action="store_true", help="skip cases with an existing valid JSON report")
    parser.add_argument(
        "--no-build", action="store_true", help="do not build build-r9700/bench/ninfer_bench"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.repetitions is not None and args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise SystemExit("--warmup must be nonnegative")
    if args.device < 0:
        raise SystemExit("--device must be nonnegative")
    if args.dry_run and args.prepare_only:
        raise SystemExit("--dry-run and --prepare-only are mutually exclusive")
    if args.require_fp8_hybrid and args.dry_run:
        raise SystemExit("--require-fp8-hybrid needs --prepare-only or a real run")
    if args.prepare_only and args.resume:
        raise SystemExit("--prepare-only requires a fresh output directory")
    if args.preset in ("dflash-pareto", "dflash-feasibility", "dflash-capacity") and args.dflash_draft_tokens is None:
        raise SystemExit(f"--preset {args.preset} requires --dflash-draft-tokens")
    if args.preset not in ("dflash-pareto", "dflash-feasibility", "dflash-capacity") and (
        args.dflash_draft_tokens is not None or args.dflash_verify_width != 0
    ):
        raise SystemExit("DFlash K/W options require a DFlash Pareto/capacity preset")
    args.concurrency = args.concurrency or [1]
    if len(args.concurrency) != len(set(args.concurrency)):
        raise SystemExit("duplicate --concurrency value")
    if any(concurrency not in PRODUCT_CONCURRENCIES for concurrency in args.concurrency):
        raise SystemExit("--concurrency must be in [1, 4]")
    validate_fp8_hybrid_performance_contract(args)
    validate_post_chunk_capacity_contract(args)
    if args.preset == "dflash-shortlist":
        if args.concurrency != [1]:
            raise SystemExit("--preset dflash-shortlist is the fixed C=1 shortlist")
        if args.suite or args.limit is not None:
            raise SystemExit("--preset dflash-shortlist requires its complete fixed case set")
        if args.repetitions is not None or args.warmup is not None:
            raise SystemExit("--preset dflash-shortlist uses fixed 2/1 performance and 1/0 diagnostic repetitions")
    if args.preset in SELECTED_PREFILL_CHUNK_PRESETS and (
        args.prefill_chunk is None or len(args.prefill_chunk) != 1
    ):
        raise SystemExit(f"--preset {args.preset} requires exactly one --prefill-chunk")
    if args.preset == "prefill-chunk":
        if args.concurrency != [1]:
            raise SystemExit("--preset prefill-chunk is fixed at C=1")
        if args.suite or args.limit is not None:
            raise SystemExit("--preset prefill-chunk requires its complete candidate set")
        if args.repetitions is not None or args.warmup is not None:
            raise SystemExit("--preset prefill-chunk uses fixed 3/1 repetitions")
        if args.prefill_chunk and len(args.prefill_chunk) != len(set(args.prefill_chunk)):
            raise SystemExit("duplicate --prefill-chunk value")
        selected_chunks = args.prefill_chunk or list(PRODUCTION_PREFILL_CHUNKS)
        if args.prefill_prompt == 8192 and set(selected_chunks) != set(PRODUCTION_PREFILL_CHUNKS):
            raise SystemExit(
                "8K prefill-chunk screening requires exactly 1024,2048,4096,8192"
            )
        if args.prefill_prompt == 32768 and len(selected_chunks) != 2:
            raise SystemExit("32K prefill-chunk confirmation requires exactly two finalists")
    elif args.preset == "low-context-prefill":
        if args.concurrency != [1]:
            raise SystemExit("--preset low-context-prefill is fixed at C=1")
        if args.device != 0:
            raise SystemExit("--preset low-context-prefill is fixed to device 0")
        if args.expected_kv_value_group is None:
            raise SystemExit("--preset low-context-prefill requires --expected-kv-value-group")
        if args.expected_xattention_profile != "dense":
            raise SystemExit("--preset low-context-prefill requires the dense attention profile")
        if args.suite or args.limit is not None or args.resume:
            raise SystemExit("--preset low-context-prefill requires one fresh complete ladder")
        if args.repetitions is not None or args.warmup is not None:
            raise SystemExit("--preset low-context-prefill uses fixed 3/1 repetitions")
        if args.prefill_prompt != 8192:
            raise SystemExit("--prefill-prompt requires --preset prefill-chunk")
    elif args.preset == "ordinary-diagnostic":
        if args.concurrency != [1]:
            raise SystemExit("--preset ordinary-diagnostic is fixed at C=1")
        if args.device != 0:
            raise SystemExit("--preset ordinary-diagnostic is fixed to device 0")
        if args.suite or args.limit is not None or args.resume:
            raise SystemExit("--preset ordinary-diagnostic requires one fresh complete run")
        if args.repetitions is not None or args.warmup is not None:
            raise SystemExit("--preset ordinary-diagnostic uses fixed 3/1 repetitions")
        if args.prefill_chunk is not None or args.prefill_prompt != 8192:
            raise SystemExit("--preset ordinary-diagnostic has fixed 8K/chunk4096 geometry")
    elif args.preset in (
        "pareto", "pareto-whole", "pareto-feasibility", "pareto-capacity",
        *DFLASH_CAMPAIGN_PRESETS,
    ):
        if args.prefill_chunk is not None and len(args.prefill_chunk) != 1:
            raise SystemExit(f"--preset {args.preset} accepts exactly one --prefill-chunk")
        if args.prefill_prompt != 8192:
            raise SystemExit("--prefill-prompt requires --preset prefill-chunk")
    elif args.prefill_chunk is not None or args.prefill_prompt != 8192:
        raise SystemExit("prefill chunk options are not valid for this preset")

    args.bench = args.bench.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()

    args.prefill_chunk_authority_record = None
    if args.prefill_chunk_authority is not None:
        if (
            args.preset not in SELECTED_PREFILL_CHUNK_PRESETS
            or args.prefill_chunk is None
            or len(args.prefill_chunk) != 1
        ):
            raise SystemExit(
                "--prefill-chunk-authority requires a selected-chunk campaign with one chunk"
            )
        try:
            args.prefill_chunk_authority_record = inspect_prefill_chunk_authority(
                args.prefill_chunk_authority, args.prefill_chunk[0]
            )
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error

    if not args.dry_run and not args.weights.is_file():
        raise SystemExit(f"weights file not found: {args.weights}")
    artifact_provenance = (
        {"path": str(args.weights), "exists": args.weights.is_file()}
        if args.dry_run
        else bind_n16_migration_receipt(args.weights, inspect_artifact(args.weights))
    )
    if args.require_fp8_hybrid:
        if artifact_provenance.get("weights_id") == HYBRID_BASE_WEIGHTS_ID:
            ppl_run.require_fp8_hybrid_candidate({
                **artifact_provenance,
                "bytes": artifact_provenance["file_size_bytes"],
            })
        else:
            artifact_provenance = require_fp8_hybrid_artifact(
                args.weights, artifact_provenance, args.preset
            )
    validate_dflash_campaign_artifact(
        args.preset, artifact_provenance, dry_run=args.dry_run
    )
    corpus_tokens = count_corpus_tokens(args.corpus)

    args.power_profile_observed = None
    args.power_profile_rechecked_after = None
    if args.preset in POWER_BOUND_PRESETS:
        if args.dry_run or args.prepare_only:
            args.power_profile_observed = (
                "not-checked-prepare-only" if args.prepare_only else "not-checked-dry-run"
            )
        else:
            try:
                args.power_profile_observed = require_auto_power_profile()
            except ValueError as error:
                raise SystemExit(str(error)) from error

    try:
        all_cases = build_cases(
            args.preset, args.dflash_draft_tokens, args.dflash_verify_width,
            args.prefill_chunk or PRODUCTION_PREFILL_CHUNKS, args.prefill_prompt,
            args.prefill_chunk[0] if args.prefill_chunk else 4096,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cases = filtered_cases(all_cases, args.suite, args.limit)
    if not cases:
        raise SystemExit("selected matrix is empty")
    args.hybrid_width_authority = None
    if args.require_fp8_hybrid:
        width_tool = args.hybrid_width_tool.expanduser().resolve(strict=True)
        args.hybrid_width_authority = build_hybrid_shared_workspace_authority(
            width_tool, args.prefill_chunk or PRODUCTION_PREFILL_CHUNKS,
        )
    max_prompt = max_prompt_in_cases(cases)
    if max_prompt > corpus_tokens:
        raise SystemExit(
            f"selected matrix needs prompt length {max_prompt}, but corpus has {corpus_tokens} tokens"
        )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = REPO_ROOT / "profiles/bench" / f"ninfer-{args.preset}-{utc_stamp()}"
    out_dir = out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    if out_dir.is_symlink():
        raise SystemExit(f"output directory must not be a symlink: {out_dir}")
    out_dir = out_dir.resolve()
    prior_manifest_path = out_dir / "manifest.json"
    prior_manifest: dict[str, Any] | None = None
    if not args.resume and out_dir.exists():
        raise SystemExit(
            f"output directory already exists: {out_dir}; use a fresh directory or --resume"
        )
    if args.resume:
        try:
            _inode(prior_manifest_path)
        except (FileNotFoundError, ValueError):
            raise SystemExit(
                f"--resume requires the existing schema-v{MATRIX_SCHEMA_VERSION} matrix manifest"
            )
        try:
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"existing matrix manifest is invalid: {error}") from error
        if (
            not isinstance(prior_manifest, dict)
            or prior_manifest.get("schema_version") != MATRIX_SCHEMA_VERSION
            or prior_manifest.get("artifact_type") != "ninfer_bench_matrix_run"
        ):
            raise SystemExit(
                f"--resume requires the existing schema-v{MATRIX_SCHEMA_VERSION} matrix manifest"
            )
        if prior_manifest.get("artifact") != artifact_provenance:
            raise SystemExit("--resume artifact bytes or identity differ from the existing matrix")
        try:
            validate_manifest_output_ownership(out_dir, prior_manifest)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    json_dir = out_dir / "json"
    log_dir = out_dir / "logs"
    diagnostic_dir = out_dir / "diagnostics"
    json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    command_records: list[dict[str, Any]] = []
    commands_sh: list[str] = []
    points = [
        (case, concurrency)
        for case in cases
        for concurrency in args.concurrency
        if not case.concurrency_one_only or concurrency == 1
    ]
    for case, concurrency in points:
        report_path = json_dir / case.suite / f"c{concurrency}" / f"{case.name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        base_args = [
            str(args.bench),
            "--weights",
            str(args.weights),
            "--corpus",
            str(args.corpus),
            "--device",
            str(args.device),
            "--concurrency",
            str(concurrency),
            *case.args,
            "--output",
            "json",
            "--output-file",
            str(report_path),
        ]
        command = add_repetition_args(base_args, case, args.repetitions, args.warmup)
        raw_diagnostic = (
            diagnostic_dir / f"{case.name}.c{concurrency}.raw.json" if case.diagnostic else None
        )
        bound_diagnostic = (
            diagnostic_dir / f"{case.name}.c{concurrency}.json" if case.diagnostic else None
        )
        environment = (
            {
                "NINFER_DFLASH_CANDIDATE_STATS": "1",
                "NINFER_DFLASH_CANDIDATE_STATS_OUT": str(raw_diagnostic),
            }
            if raw_diagnostic is not None
            else {}
        )
        point_draft_tokens = int(_case_option(case, "--draft-tokens") or "0")
        point_requested_width = int(_case_option(case, "--dflash-verify-width") or "0")
        point_resolved_width = (
            resolved_dflash_verify_width(point_draft_tokens, point_requested_width)
            if _case_option(case, "--spec") == "dflash" else 0
        )
        command_records.append(
            {
                "suite": case.suite,
                "case": case.name,
                "concurrency": concurrency,
                "report": str(report_path),
                "notes": case.notes,
                "performance_eligible": not case.diagnostic,
                "parity_role": case.parity_role,
                "dflash_draft_tokens": point_draft_tokens,
                "dflash_verify_width_requested": point_requested_width,
                "dflash_verify_width_resolved": point_resolved_width,
                "dflash_topology": (
                    resolved_dflash_topology(point_draft_tokens, point_resolved_width)
                    if point_resolved_width else None
                ),
                "environment": environment,
                "raw_diagnostic": str(raw_diagnostic) if raw_diagnostic is not None else None,
                "bound_diagnostic": str(bound_diagnostic) if bound_diagnostic is not None else None,
                "command": command,
            }
        )
        commands_sh.append(
            shell_join(["env", *(f"{key}={value}" for key, value in environment.items()), *command])
        )
    if prior_manifest is not None:
        expected_resume = {
            "preset": args.preset,
            "dflash_draft_tokens": args.dflash_draft_tokens,
            "dflash_verify_width_requested": args.dflash_verify_width,
            "dflash_verify_width": (
                resolved_dflash_verify_width(args.dflash_draft_tokens, args.dflash_verify_width)
                if args.dflash_draft_tokens is not None else 0
            ),
            "expected_kv_value_group": args.expected_kv_value_group,
            "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "expected_q4_activation_bits": args.expected_q4_activation_bits,
            "expected_w8_activation_bits": args.expected_w8_activation_bits,
            "expected_fp8_qk_wmma_enabled": bool(args.expected_fp8_qk_wmma),
            "expected_xattention_profile": args.expected_xattention_profile,
            "required_candidate_identity": (
                "fp8-hybrid-selection-authority" if args.require_fp8_hybrid else None
            ),
            "hybrid_shared_workspace_authority": args.hybrid_width_authority,
            "selected_prefill_chunk": (
                args.prefill_chunk[0]
                if args.preset in SELECTED_PREFILL_CHUNK_PRESETS else None
            ),
            "corpus_sha256": (
                file_sha256(args.corpus) if args.preset in POWER_BOUND_PRESETS else None
            ),
            "power_profile": (
                {
                    "required": "auto",
                    "sysfs_path": str(R9700_POWER_PROFILE),
                    "observed": args.power_profile_observed,
                }
                if args.preset in POWER_BOUND_PRESETS else None
            ),
            "concurrency": list(args.concurrency),
            "commands": command_records,
        }
        if args.prefill_chunk_authority_record is not None:
            expected_resume["prefill_chunk_authority"] = args.prefill_chunk_authority_record
        if args.require_post_chunk_capacity:
            expected_resume["post_chunk_capacity_gate"] = True
        for key, value in expected_resume.items():
            if key == "power_profile" and prior_manifest.get("prepare_only") is True:
                continue
            prior_value = prior_manifest.get(key)
            if key == "power_profile" and isinstance(prior_value, dict):
                prior_value = {
                    name: prior_value.get(name)
                    for name in ("required", "sysfs_path", "observed")
                }
            if prior_value != value:
                raise SystemExit(f"--resume matrix contract changed: {key}")

    commands_preamble = "#!/usr/bin/env bash\nset -euo pipefail\n"
    if args.preset in POWER_BOUND_PRESETS:
        commands_preamble += (
            f"test \"$(cat {shlex.quote(str(R9700_POWER_PROFILE))})\" = auto\n"
        )
    commands_epilogue = ""
    if args.preset in POWER_RECHECK_PRESETS:
        commands_epilogue = (
            f"\ntest \"$(cat {shlex.quote(str(R9700_POWER_PROFILE))})\" = auto\n"
        )
    commands_text = (
        commands_preamble + "\n" + "\n\n".join(commands_sh) + "\n"
        + commands_epilogue
    )
    durable_replace_text(out_dir / "commands.sh", commands_text)

    if args.dry_run:
        bench_provenance = {"path": str(args.bench), "exists": args.bench.is_file()}
        write_manifest(
            out_dir, args, artifact_provenance, bench_provenance, cases, command_records
        )
        print(f"wrote dry-run matrix to {out_dir}")
        print(f"cases: {len(cases)}; points: {len(points)}")
        return 0

    if args.prepare_only:
        if not args.bench.is_file():
            raise SystemExit(f"benchmark executable not found: {args.bench}")
        bench_provenance = inspect_executable(args.bench)
        write_manifest(
            out_dir, args, artifact_provenance, bench_provenance, cases, command_records
        )
        print(f"wrote provenance-bound command matrix to {out_dir}")
        print(f"cases: {len(cases)}; points: {len(points)}; no benchmark executed")
        return 0

    failures: list[dict[str, Any]] = []
    if not args.no_build:
        build_stdout = log_dir / "build.stdout.txt"
        build_stderr = log_dir / "build.stderr.txt"
        rc = run_command(
            [
                "cmake",
                "--build",
                str(REPO_ROOT / "build-r9700"),
                "--parallel",
                "--target",
                "ninfer_bench",
            ],
            build_stdout,
            build_stderr,
        )
        if rc != 0:
            failures.append(
                {
                    "case": "build",
                    "returncode": rc,
                    "stdout": str(build_stdout),
                    "stderr": str(build_stderr),
                }
            )
            durable_replace_json(out_dir / "failures.json", failures)
            print(f"build failed; see {build_stderr}", file=sys.stderr)
            return 1

    if not args.bench.is_file():
        raise SystemExit(f"bench binary not found after build: {args.bench}")
    bench_provenance = inspect_executable(args.bench)
    if args.resume and prior_manifest is not None and prior_manifest.get("bench") != bench_provenance:
        raise SystemExit("--resume benchmark executable bytes differ from the existing matrix")
    write_manifest(out_dir, args, artifact_provenance, bench_provenance, cases, command_records)

    for index, (record, point) in enumerate(zip(command_records, points, strict=True), start=1):
        case, concurrency = point
        report_path = Path(record["report"])
        report_exists = os.path.lexists(report_path)
        if report_exists:
            try:
                _inode(report_path)
            except ValueError as error:
                raise SystemExit(str(error)) from error
        if args.resume and report_exists:
            try:
                load_bench_report(
                    report_path,
                    args.expected_kv_value_group,
                    args.expected_q4_activation_bits,
                    args.expected_w8_activation_bits,
                    bool(args.expected_fp8_qk_wmma),
                    concurrency,
                    artifact_provenance,
                    record["command"],
                    case,
                    args.expected_xattention_profile,
                )
                if case.diagnostic:
                    validate_bound_diagnostic(
                        Path(record["bound_diagnostic"]),
                        artifact=artifact_provenance,
                        bench=bench_provenance,
                        report_path=report_path,
                        case=case,
                        concurrency=concurrency,
                    )
                print(
                    f"[{index}/{len(points)}] skip {case.name} C={concurrency} "
                    "(existing report)"
                )
                continue
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                remove_runner_owned_file(report_path)

        stdout_path = log_dir / f"{case.suite}.{case.name}.c{concurrency}.stdout.txt"
        stderr_path = log_dir / f"{case.suite}.{case.name}.c{concurrency}.stderr.txt"
        print(f"[{index}/{len(points)}] run {case.suite}/{case.name} C={concurrency}")
        raw_diagnostic = record.get("raw_diagnostic")
        bound_diagnostic = record.get("bound_diagnostic")
        if raw_diagnostic:
            remove_runner_owned_file(Path(raw_diagnostic))
        if bound_diagnostic:
            remove_runner_owned_file(Path(bound_diagnostic))
        rc = run_command(
            record["command"], stdout_path, stderr_path, record.get("environment") or None
        )
        if rc != 0:
            failures.append(
                {
                    "suite": case.suite,
                    "case": case.name,
                    "concurrency": concurrency,
                    "returncode": rc,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "command": record["command"],
                }
            )
            print(f"  failed with rc={rc}; see {stderr_path}", file=sys.stderr)
            continue
        if not report_path.is_file():
            failures.append(
                {
                    "suite": case.suite,
                    "case": case.name,
                    "concurrency": concurrency,
                    "returncode": rc,
                    "error": "report file was not created",
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "command": record["command"],
                }
            )
        elif case.diagnostic:
            try:
                bind_dflash_diagnostic(
                    Path(record["raw_diagnostic"]),
                    Path(record["bound_diagnostic"]),
                    artifact=artifact_provenance,
                    bench=bench_provenance,
                    report_path=report_path,
                    case=case,
                    concurrency=concurrency,
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                failures.append(
                    {
                        "suite": case.suite,
                        "case": case.name,
                        "concurrency": concurrency,
                        "error": f"failed to bind DFlash diagnostic: {exc}",
                    }
                )

    rows: list[dict[str, Any]] = []
    for record, point in zip(command_records, points, strict=True):
        case, concurrency = point
        report_path = Path(record["report"])
        if not report_path.is_file():
            continue
        try:
            rows.extend(
                report_rows(
                    report_path,
                    case,
                    args.expected_kv_value_group,
                    args.expected_q4_activation_bits,
                    args.expected_w8_activation_bits,
                    bool(args.expected_fp8_qk_wmma),
                    concurrency,
                    artifact_provenance,
                    record["command"],
                    args.expected_xattention_profile,
                )
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "suite": case.suite,
                    "case": case.name,
                    "concurrency": concurrency,
                    "report": str(report_path),
                    "error": f"failed to parse report: {exc}",
                }
            )

    finished_artifact = bind_n16_migration_receipt(
        args.weights, inspect_artifact(args.weights)
    )
    if args.require_fp8_hybrid:
        if finished_artifact.get("weights_id") == HYBRID_BASE_WEIGHTS_ID:
            ppl_run.require_fp8_hybrid_candidate({
                **finished_artifact, "bytes": finished_artifact["file_size_bytes"],
            })
        else:
            finished_artifact = require_fp8_hybrid_artifact(
                args.weights, finished_artifact, args.preset
            )
    if finished_artifact != artifact_provenance:
        failures.append(
            {
                "case": "artifact_provenance",
                "error": "artifact bytes or identity changed during the matrix",
                "started": artifact_provenance,
                "finished": finished_artifact,
            }
        )
    finished_bench = inspect_executable(args.bench)
    if finished_bench != bench_provenance:
        failures.append(
            {
                "case": "benchmark_provenance",
                "error": "benchmark executable bytes changed during the matrix",
                "started": bench_provenance,
                "finished": finished_bench,
            }
        )
    if args.prefill_chunk_authority_record is not None:
        try:
            finished_chunk_authority = inspect_prefill_chunk_authority(
                Path(args.prefill_chunk_authority_record["path"]),
                args.prefill_chunk_authority_record["selected_prefill_chunk"],
            )
        except (OSError, ValueError) as error:
            failures.append({
                "case": "prefill_chunk_authority",
                "error": f"selected-chunk authority failed final revalidation: {error}",
            })
        else:
            if finished_chunk_authority != args.prefill_chunk_authority_record:
                failures.append({
                    "case": "prefill_chunk_authority",
                    "error": "selected-chunk authority changed during the matrix",
                    "started": args.prefill_chunk_authority_record,
                    "finished": finished_chunk_authority,
                })
    if args.preset in POWER_RECHECK_PRESETS:
        try:
            args.power_profile_rechecked_after = require_auto_power_profile()
        except ValueError as error:
            failures.append({"case": "power_profile", "error": str(error)})
            args.power_profile_rechecked_after = "not-auto-or-unreadable"
        write_manifest(out_dir, args, artifact_provenance, bench_provenance,
                       cases, command_records)

    parity_payload: dict[str, Any] | None = None
    if args.preset in ("dflash-pareto", "dflash-shortlist"):
        parity_records = [record for record in command_records if record["parity_role"] is not None]
        try:
            parity_payload, parity_failures = write_dflash_greedy_parity(
                out_dir,
                parity_records,
                artifact=artifact_provenance,
                bench=bench_provenance,
            )
            failures.extend({"case": "dflash_greedy_parity", **failure} for failure in parity_failures)
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            failures.append(
                {"case": "dflash_greedy_parity", "error": f"failed to compare tokens: {exc}"}
            )

    determinism_payload: dict[str, Any] | None = None
    if args.preset == "dflash-pareto":
        try:
            determinism_payload, determinism_failures = write_dflash_determinism(
                out_dir, command_records, artifact=artifact_provenance, bench=bench_provenance
            )
            failures.extend(
                {"case": "dflash_proposal_determinism", **failure}
                for failure in determinism_failures
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            failures.append({
                "case": "dflash_proposal_determinism",
                "error": f"failed to compare repeated proposal traces: {exc}",
            })

    if (
        args.preset == "dflash-pareto"
        and parity_payload is not None
        and determinism_payload is not None
    ):
        try:
            _, quality_failures = write_dflash_quality_evidence(
                out_dir,
                command_records,
                parity_payload,
                determinism_payload,
                artifact=artifact_provenance,
                bench=bench_provenance,
            )
            failures.extend(
                {"case": "dflash_generated_quality", **failure}
                for failure in quality_failures
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            failures.append({
                "case": "dflash_generated_quality",
                "error": f"failed to assemble generated quality evidence: {exc}",
            })

    if args.preset == "dflash-shortlist" and parity_payload is not None:
        try:
            _, shortlist_failures = write_dflash_shortlist(
                out_dir,
                command_records,
                rows,
                parity_payload,
                artifact=artifact_provenance,
                bench=bench_provenance,
                provenance_stable=(
                    finished_artifact == artifact_provenance
                    and finished_bench == bench_provenance
                ),
            )
            failures.extend(
                {"case": "dflash_shortlist", **failure}
                for failure in shortlist_failures
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            failures.append(
                {"case": "dflash_shortlist", "error": f"failed to rank shortlist: {exc}"}
            )

    write_summary(rows, out_dir)
    if failures:
        durable_replace_json(out_dir / "failures.json", failures)
        print(f"completed with {len(failures)} failure(s); see {out_dir / 'failures.json'}")
        return 1

    # A resumed campaign may have repaired every point from an earlier failed attempt. Keep the
    # directory's terminal status unambiguous for downstream provenance assemblers.
    try:
        remove_runner_owned_file(out_dir / "failures.json")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"completed {len(points)} points across {len(cases)} cases")
    print(f"summary: {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
