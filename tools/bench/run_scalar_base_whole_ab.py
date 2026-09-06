#!/usr/bin/env python3
"""Run the fixed production/scalar-base whole-P2048 qualification A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import stat
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONTROL_BINARY = ROOT / "build-r9700-scalar-base-control/bench/ninfer_bench"
CANDIDATE_BINARY = ROOT / "build-r9700-scalar-base-qualification/bench/ninfer_bench"
CONTROL_BUILD_DIRECTORY = ROOT / "build-r9700-scalar-base-control"
CANDIDATE_BUILD_DIRECTORY = ROOT / "build-r9700-scalar-base-qualification"
OPERATOR_PRODUCT_EXECUTABLE = (
    CANDIDATE_BUILD_DIRECTORY / "tests/ninfer_r9700_a8q4_scalar_base_product_qual"
)
OPERATOR_AUTHORITY = (
    ROOT / "profiles/bench/"
    "r9700-a8q4-n16k16-scalar-base-product-p2048-ab-20260905.json"
)
OPERATOR_AUTHORITY_SHA256 = (
    "325cad2e3c53b620f2864014bd16d3f4e510fdeb846b19a2c33ad31ee7c28fd8"
)
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-eval.ninfer"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
REPORT = ROOT / "profiles/bench/r9700-scalar-base-production-p2048-c1-ab-20260905.json"
RAW_DIRECTORY = REPORT.with_suffix("")
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")

REPORT_SCHEMA = "ninfer_r9700_scalar_base_whole_ab"
REPORT_SCHEMA_VERSION = 1
BENCH_SCHEMA_VERSION = 20
ROBUST_FACTOR = 4.4478
PAIR_ORDER = ("AB", "BA") * 4
ROLE_ORDER = tuple(role for pair in PAIR_ORDER for role in pair)
CONTROL_PROFILE = "m64n128-pingpong-n16-k16-production"
CANDIDATE_PROFILE = "m64n128-pingpong-n16-k16-scalar-base-qualification"
MODEL_ID = "qwen3.8-27b"
TARGET_ID = "qwen3_8_27b_r9700"
WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
ARTIFACT_SHA256 = "040c6e7ed29c856718a638c00181975710d987b7d5f49f4cafbdf68911f7e7d2"
ARTIFACT_SIZE_BYTES = 21_553_549_312
LINEAR_SOURCE = ROOT / "src/ops/r9700/linear/r9700_linear.hip"
BUILD_SOURCE_PATHS = (
    ROOT / "CMakeLists.txt",
    ROOT / "src/ops/r9700/linear/r9700_linear.h",
    LINEAR_SOURCE,
    ROOT / "src/ops/r9700/linear/r9700_q4_activation_profile.h",
)
MATCHED_CACHE_OPTIONS = {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_HIP_ARCHITECTURES": "gfx1201",
    "NINFER_BUILD_BENCHMARKS": "ON",
    "NINFER_R9700_FP8_QK_WMMA": "1",
    "NINFER_R9700_KV_VALUE_GROUP": "16",
    "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
    "NINFER_R9700_W8_ACTIVATION_BITS": "8",
    "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
    "NINFER_R9700_XATTENTION_STRIDE": "16",
    "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000",
}
NINFER_PREFIX = struct.Struct("<8sQ")
NINFER_MAGIC = b"NINFER\x00\x02"
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
KV_PLANES = {
    "key": "token-fastest-head-major",
    "value": "feature-fastest-page-major",
    "value_scale": "feature-fastest-page-major",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_identity(path: Path, *, executable: bool = False) -> dict[str, object]:
    absolute = Path(os.path.abspath(path))
    try:
        metadata = os.lstat(absolute)
    except FileNotFoundError as error:
        raise ValueError(f"required input is missing: {absolute}") from error
    if not stat.S_ISREG(metadata.st_mode) or absolute.is_symlink():
        raise ValueError(f"required input is not a real regular file: {absolute}")
    try:
        absolute.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"required input is outside the repository root: {absolute}") from error
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_uid, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    with absolute.open("rb") as source:
        opened_before = os.fstat(source.fileno())
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        opened_after = os.fstat(source.fileno())
    current = os.lstat(absolute)
    if not (
        identity(metadata) == identity(opened_before) == identity(opened_after)
        == identity(current)
    ):
        raise ValueError(f"required input changed while being identified: {absolute}")
    if current.st_uid != os.getuid():
        raise ValueError(f"required input is not owned by the invoking repository owner: {absolute}")
    if executable and not os.access(absolute, os.X_OK):
        raise ValueError(f"required executable is not executable: {absolute}")
    return {
        "path": str(absolute.relative_to(ROOT)),
        "sha256": digest,
        "file_size_bytes": current.st_size,
        "device": current.st_dev,
        "inode": current.st_ino,
        "uid": current.st_uid,
        "mode": stat.S_IMODE(current.st_mode),
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }


def inspect_artifact(path: Path) -> dict[str, object]:
    identity = _regular_identity(path)
    if (identity["file_size_bytes"], identity["sha256"]) != (
        ARTIFACT_SIZE_BYTES, ARTIFACT_SHA256
    ):
        raise ValueError("selected artifact bytes differ from the fixed P2048 authority")
    with path.open("rb") as source:
        opened_before = os.fstat(source.fileno())
        if (opened_before.st_dev, opened_before.st_ino, opened_before.st_size,
            opened_before.st_mtime_ns, opened_before.st_ctime_ns) != (
            identity["device"], identity["inode"], identity["file_size_bytes"],
            identity["mtime_ns"], identity["ctime_ns"]
        ):
            raise ValueError("selected artifact changed before directory inspection")
        prefix = source.read(NINFER_PREFIX.size)
        if len(prefix) != NINFER_PREFIX.size:
            raise ValueError("selected artifact has a truncated prefix")
        magic, directory_bytes = NINFER_PREFIX.unpack(prefix)
        if magic != NINFER_MAGIC or not 0 < directory_bytes <= MAX_DIRECTORY_BYTES:
            raise ValueError("selected artifact is not a valid NInfer v2 artifact")
        raw_directory = source.read(directory_bytes)
        if len(raw_directory) != directory_bytes:
            raise ValueError("selected artifact has a truncated directory")
        opened_after = os.fstat(source.fileno())
    if (opened_after.st_dev, opened_after.st_ino, opened_after.st_size,
        opened_after.st_mtime_ns, opened_after.st_ctime_ns) != (
        identity["device"], identity["inode"], identity["file_size_bytes"],
        identity["mtime_ns"], identity["ctime_ns"]
    ):
        raise ValueError("selected artifact changed during directory inspection")
    try:
        directory = json.loads(raw_directory)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selected artifact directory is invalid") from error
    declared = directory.get("identity") if isinstance(directory, dict) else None
    if not isinstance(declared, dict) or declared.get("model_id") != MODEL_ID or declared.get(
        "weights_id"
    ) != WEIGHTS_ID:
        raise ValueError("selected artifact identity differs from the fixed P2048 authority")
    return {**identity, "model_id": MODEL_ID, "weights_id": WEIGHTS_ID}


def _stable_text(path: Path, label: str) -> tuple[str, dict[str, object]]:
    before = _regular_identity(path)
    text = path.read_text(encoding="utf-8")
    after = _regular_identity(path)
    if after != before:
        raise ValueError(f"{label} changed while being read")
    return text, before


def _compile_entry(
    build_directory: Path,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    database_path = build_directory / "compile_commands.json"
    text, identity = _stable_text(database_path, "build compile database")
    database = json.loads(text)
    if not isinstance(database, list):
        raise ValueError("build compile database root is not an array")
    matches = [
        row for row in database
        if isinstance(row, dict)
        and Path(str(row.get("file", ""))).resolve() == LINEAR_SOURCE.resolve()
    ]
    if len(matches) != 1 or set(matches[0]) != {"directory", "command", "file", "output"}:
        raise ValueError("build compile database lacks one exact linear source command")
    entry = matches[0]
    if (
        Path(str(entry["directory"])).resolve() != build_directory.resolve()
        or Path(str(entry["file"])).resolve() != LINEAR_SOURCE.resolve()
        or not isinstance(entry["command"], str)
        or not entry["command"]
        or not isinstance(entry["output"], str)
        or not entry["output"]
    ):
        raise ValueError("linear source compile identity differs")
    return entry, identity, database


def _cache_values(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    values: dict[str, str] = {}
    text, identity = _stable_text(path, "build CMake cache")
    for line in text.splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        values[key_type.split(":", 1)[0]] = value
    return values, identity


def _normalized_compile(entry: dict[str, object], build_directory: Path) -> list[str]:
    flag = "-DNINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION=1"
    return [
        token.replace(str(build_directory.resolve()), "$BUILD")
        for token in shlex.split(str(entry["command"]))
        if token != flag
    ]


def _normalized_product_database(
    database: list[dict[str, object]], build_directory: Path
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for entry in database:
        if not isinstance(entry, dict) or set(entry) != {"directory", "command", "file", "output"}:
            raise ValueError("compile database entry shape differs")
        source = Path(str(entry["file"])).resolve()
        try:
            relative = source.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("compile database source is outside repository") from error
        if relative.parts[0] not in ("src", "bench"):
            continue
        result.append({
            "source": str(relative),
            "command": _normalized_compile(entry, build_directory),
            "output": str(entry["output"]).replace(str(build_directory.resolve()), "$BUILD"),
        })
    result.sort(key=lambda row: (str(row["source"]), str(row["output"])))
    if not result:
        raise ValueError("compile database lacks product sources")
    return result


def build_receipts() -> dict[str, object]:
    sources = {
        str(path.relative_to(ROOT)): _regular_identity(path)
        for path in BUILD_SOURCE_PATHS
    }
    builds: dict[str, dict[str, object]] = {}
    entries: dict[str, dict[str, object]] = {}
    normalized_databases: dict[str, list[dict[str, object]]] = {}
    for role, build_directory, binary, qualification in (
        ("control", CONTROL_BUILD_DIRECTORY, CONTROL_BINARY, False),
        ("candidate", CANDIDATE_BUILD_DIRECTORY, CANDIDATE_BINARY, True),
    ):
        if not build_directory.is_dir() or build_directory.is_symlink():
            raise ValueError(f"{role} build directory is invalid")
        cache = build_directory / "CMakeCache.txt"
        database = build_directory / "compile_commands.json"
        cache_values, cache_identity = _cache_values(cache)
        expected_qualification = "ON" if qualification else "OFF"
        if (
            any(cache_values.get(key) != value
                for key, value in MATCHED_CACHE_OPTIONS.items())
            or cache_values.get("NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION")
            != expected_qualification
        ):
            raise ValueError(f"{role} CMake cache differs from the matched build contract")
        entry, database_identity, database_entries = _compile_entry(build_directory)
        tokens = shlex.split(str(entry["command"]))
        flag = "-DNINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION=1"
        if (tokens.count(flag) == 1) != qualification:
            raise ValueError(f"{role} qualification compile flag differs")
        entries[role] = entry
        normalized_databases[role] = _normalized_product_database(
            database_entries, build_directory
        )
        builds[role] = {
            "build_directory": str(build_directory.relative_to(ROOT)),
            "binary": _regular_identity(binary, executable=True),
            "cmake_cache": cache_identity,
            "compile_database": database_identity,
            "linear_source_compile": entry,
            "scalar_base_qualification": qualification,
            "matched_cache_options": {
                **MATCHED_CACHE_OPTIONS,
                "NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION": expected_qualification,
            },
        }
        if qualification:
            builds[role]["operator_product_executable"] = _regular_identity(
                OPERATOR_PRODUCT_EXECUTABLE, executable=True
            )
    if _normalized_compile(entries["control"], CONTROL_BUILD_DIRECTORY) != _normalized_compile(
        entries["candidate"], CANDIDATE_BUILD_DIRECTORY
    ):
        raise ValueError("control and candidate linear compiles differ beyond qualification flag")
    if normalized_databases["control"] != normalized_databases["candidate"]:
        raise ValueError("control and candidate product compiles differ beyond qualification flag")
    normalized_sha = hashlib.sha256(json.dumps(
        normalized_databases["control"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "sources": sources,
        "control": builds["control"],
        "candidate": builds["candidate"],
        "only_compile_difference": "NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION=1",
        "normalized_product_compile_database_sha256": normalized_sha,
    }


def inspect_operator_authority(builds: dict[str, object]) -> dict[str, object]:
    identity = _regular_identity(OPERATOR_AUTHORITY)
    if identity["sha256"] != OPERATOR_AUTHORITY_SHA256:
        raise ValueError("operator authority SHA-256 differs")
    raw = OPERATOR_AUTHORITY.read_text(encoding="utf-8")
    if _regular_identity(OPERATOR_AUTHORITY) != identity:
        raise ValueError("operator authority changed while being validated")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != (
        "ninfer.r9700.a8q4-n16k16-scalar-base-product-gate.v1"
    ):
        raise ValueError("operator authority schema differs")
    decision = value.get("decision")
    cells = decision.get("cells") if isinstance(decision, dict) else None
    if (
        not isinstance(decision, dict)
        or decision.get("admission_pass") is not True
        or not _finite(decision.get("robust_weighted_saving_lower_ms"))
        or decision["robust_weighted_saving_lower_ms"] < 5.0
        or decision.get("required_robust_weighted_saving_lower_ms") != 5.0
        or decision.get("maximum_every_cell_robust_ratio_upper") != 1.01
        or not isinstance(cells, list)
        or not cells
        or any(
            not isinstance(cell, dict)
            or not _finite(cell.get("robust_ratio_upper"))
            or cell["robust_ratio_upper"] > 1.01
            for cell in cells
        )
    ):
        raise ValueError("operator authority admission decision differs")
    static = value.get("static")
    fields = static.get("fields") if isinstance(static, dict) else None
    candidate = builds.get("candidate")
    sources = builds.get("sources")
    if not isinstance(candidate, dict) or not isinstance(sources, dict):
        raise ValueError("candidate build receipt is incomplete")
    product_executable = candidate.get("operator_product_executable")
    source = sources.get(str(LINEAR_SOURCE.relative_to(ROOT)))
    profile_path = ROOT / "src/ops/r9700/linear/r9700_q4_activation_profile.h"
    profile = sources.get(str(profile_path.relative_to(ROOT)))
    if (
        not isinstance(fields, dict)
        or not isinstance(product_executable, dict)
        or not isinstance(source, dict)
        or not isinstance(profile, dict)
        or fields.get("product_executable_sha256") != product_executable.get("sha256")
        or fields.get("source_sha256") != source.get("sha256")
        or fields.get("profile_sha256") != profile.get("sha256")
    ):
        raise ValueError("operator authority product identities differ from candidate")
    return {**identity, "schema": value["schema"], "admission_pass": True}


def input_identities() -> dict[str, dict[str, object]]:
    builds = build_receipts()
    return {
        "build_receipts": builds,
        "operator_authority": inspect_operator_authority(builds),
        "artifact": inspect_artifact(ARTIFACT),
        "corpus": _regular_identity(CORPUS),
        "runner": _regular_identity(Path(__file__).resolve()),
    }


def power_identity() -> dict[str, object]:
    command = [
        "/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname", "--showprofile",
        "--showperflevel", "--json",
    ]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        raise ValueError("rocm-smi identity query failed")
    try:
        payload = json.loads(process.stdout)
        level = POWER.read_text(encoding="utf-8").strip()
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("R9700 power identity is unavailable") from error
    card = payload.get("card0") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"card0"}
        or not isinstance(card, dict)
        or card.get("Card Series") != "AMD Radeon AI PRO R9700"
        or card.get("GFX Version") != "gfx1201"
        or card.get("Performance Level") != "auto"
        or level != "auto"
    ):
        raise ValueError("device 0 must be the exact R9700/gfx1201 auto-power target")
    return {
        "command": command,
        "device": {"ordinal": 0, "name": card["Card Series"], "architecture": card["GFX Version"]},
        "rocm_smi_performance_level": card["Performance Level"],
        "sysfs_path": str(POWER),
        "sysfs_value": level,
    }


def command_for(role: str) -> list[str]:
    if role not in ("A", "B"):
        raise ValueError(f"invalid A/B role: {role}")
    binary = CONTROL_BINARY if role == "A" else CANDIDATE_BINARY
    return [
        str(binary.relative_to(ROOT)),
        "--weights", str(ARTIFACT.relative_to(ROOT)),
        "--corpus", str(CORPUS.relative_to(ROOT)),
        "--device", "0",
        "--concurrency", "1",
        "-p", "2048",
        "--prefill-chunk", "4096",
        "--spec", "mtp",
        "--draft-tokens", "0",
        "--retain-token-ids",
        "--output", "json",
        "-r", "1",
        "--warmup", "1",
    ]


def expected_bench_stderr() -> str:
    return (
        f"[ninfer_bench] loading {ARTIFACT.relative_to(ROOT)} "
        "(max_context=2048, concurrency=1, kv_format=fp8-k-int4-v)\n"
        "[ninfer_bench] test 1/1 pp2048: warmup=1 reps=1\n"
    )


def validate_bench_stderr(value: object) -> None:
    if type(value) is not str or value != expected_bench_stderr():
        raise ValueError("ninfer_bench stderr protocol differs")


def _finite(value: object, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _require_exact_keys(value: dict, expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} field inventory differs from schema v20")


def validate_bench_report(value: object, role: str, command: Sequence[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("benchmark report root is not an object")
    _require_exact_keys(
        value,
        {"schema_version", "artifact_type", "tool", "command", "environment", "artifact", "load", "memory", "config", "tests"},
        "benchmark report",
    )
    if (
        value.get("schema_version") != BENCH_SCHEMA_VERSION
        or value.get("artifact_type") != "ninfer_bench_report"
        or value.get("tool") != "ninfer_bench"
        or value.get("command") != " ".join(command)
    ):
        raise ValueError("benchmark report identity or exact command differs")
    environment = value.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("benchmark report environment is not an object")
    _require_exact_keys(
        environment,
        {"gpu_name", "architecture_name", "hip_runtime_version", "hip_driver_version", "device_id"},
        "benchmark environment",
    )
    if (
        environment.get("gpu_name"), environment.get("architecture_name"), environment.get("device_id")
    ) != ("AMD Radeon AI PRO R9700", "gfx1201", 0):
        raise ValueError("benchmark report device identity differs")
    for key in ("hip_runtime_version", "hip_driver_version"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"benchmark report lacks {key}")
    artifact = value.get("artifact")
    if not isinstance(artifact, dict) or artifact != {
        "path": str(ARTIFACT.relative_to(ROOT)),
        "file_size_bytes": ARTIFACT_SIZE_BYTES,
    }:
        raise ValueError("benchmark report artifact differs")
    load = value.get("load")
    if not isinstance(load, dict):
        raise ValueError("benchmark report load identity is not an object")
    _require_exact_keys(
        load,
        {"target", "weights_id", "load_seconds", "upload_seconds", "artifact_bytes_read", "host_to_device_bytes", "peak_staging_bytes", "tensor_count", "resource_count"},
        "benchmark load",
    )
    if load.get("target") != TARGET_ID or load.get("weights_id") != WEIGHTS_ID:
        raise ValueError("benchmark report loaded identity differs")
    for key in ("load_seconds", "upload_seconds"):
        if not _finite(load.get(key)):
            raise ValueError(f"benchmark report has invalid {key}")
    for key in ("artifact_bytes_read", "host_to_device_bytes", "peak_staging_bytes", "tensor_count", "resource_count"):
        if type(load.get(key)) is not int or load[key] < 0:
            raise ValueError(f"benchmark report has invalid {key}")
    config = value.get("config")
    expected_profile = CONTROL_PROFILE if role == "A" else CANDIDATE_PROFILE
    expected_config = {
        "max_context": 2048,
        "prefill_chunk": 4096,
        "kv_cache_format": "fp8-k-int4-v",
        "kv_value_group": 16,
        "kv_plane_layouts": KV_PLANES,
        "q4_activation_bits": 8,
        "q4_prefill_cta_profile": expected_profile,
        "w8_activation_bits": 8,
        "fp8_qk_wmma_enabled": True,
        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
        "fp8_qk_wmma_t1_min_context": 64,
        "fp8_qk_wmma_t2_min_context": 320,
        "xattention_qualification": False,
        "concurrency": 1,
        "pending_timeout_ms": 0xFFFFFFFF,
        "pending_deadline": "unbounded",
        "spec": "none",
        "draft_tokens": 0,
        "speculative_execution": False,
        "dflash_verify_width_requested": 0,
        "dflash_verify_width": 0,
        "proposal_head": "full",
        "use_device_graph": True,
        "retain_token_ids": True,
        "decode_path": "device_graph",
        "decode_graph_prime": {"primed": False, "output_tokens": 0},
        "repetitions": 1,
        "warmup": 1,
        "corpus_path": str(CORPUS.relative_to(ROOT)),
        "corpus_tokens": len(CORPUS.read_text(encoding="utf-8").split()),
    }
    if config != expected_config:
        raise ValueError("benchmark report semantic configuration differs")
    memory = value.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("benchmark report memory is not an object")
    _require_exact_keys(
        memory,
        {
            "device", "max_context", "kv_capacity_mode", "kv_capacity",
            "kv_capacity_page_groups", "kv_capacity_max_page_groups", "kv_cache_format",
            "weights", "sequence", "workspace", "request_transient",
            "minimum_runtime_reservation_bytes", "kv_capacity_increment_bytes",
            "runtime_reservation_bytes", "available_after_weights_bytes",
            "available_after_startup_bytes", "kv_capacity_headroom_bytes", "planned_slack_bytes",
            "device_graph_allowance_bytes", "device_graph_observed_bytes", "kv_payload_bytes",
        },
        "benchmark memory",
    )
    if any(
        memory.get(key) != expected
        for key, expected in {
            "device": 0, "max_context": 2048, "kv_capacity_mode": "explicit",
            "kv_capacity": 2048, "kv_cache_format": "fp8-k-int4-v",
        }.items()
    ):
        raise ValueError("benchmark report memory contract differs")
    for arena_name in ("weights", "sequence", "workspace", "request_transient"):
        arena = memory.get(arena_name)
        if not isinstance(arena, dict):
            raise ValueError(f"benchmark memory {arena_name} arena is invalid")
        _require_exact_keys(arena, {"capacity_bytes", "used_bytes", "peak_used_bytes"}, f"benchmark memory {arena_name}")
        if any(type(arena.get(key)) is not int or arena[key] < 0 for key in arena):
            raise ValueError(f"benchmark memory {arena_name} arena contains invalid bytes")
    for key in (
        "kv_capacity_page_groups", "kv_capacity_max_page_groups",
        "minimum_runtime_reservation_bytes", "kv_capacity_increment_bytes",
        "runtime_reservation_bytes", "available_after_weights_bytes",
        "available_after_startup_bytes", "kv_capacity_headroom_bytes", "planned_slack_bytes",
        "device_graph_allowance_bytes", "device_graph_observed_bytes", "kv_payload_bytes",
    ):
        if type(memory.get(key)) is not int or memory[key] < 0:
            raise ValueError(f"benchmark memory has invalid {key}")
    tests = value.get("tests")
    if not isinstance(tests, list) or len(tests) != 1 or not isinstance(tests[0], dict):
        raise ValueError("benchmark report must contain exactly one P2048 test")
    test = tests[0]
    _require_exact_keys(
        test,
        {
            "label", "kind", "n_prompt", "n_gen", "requested_output_tokens",
            "prefill_tok_s_mean", "prefill_tok_s_stddev",
            "decode_output_tok_s_mean", "decode_output_tok_s_stddev",
            "decode_engine_tok_s_mean", "decode_engine_tok_s_stddev",
            "whole_output_tok_s_mean", "whole_output_tok_s_stddev",
            "prepare_seconds_mean", "prepare_seconds_stddev",
            "prefill_seconds_mean", "prefill_seconds_stddev",
            "decode_seconds_mean", "decode_seconds_stddev",
            "total_seconds_mean", "total_seconds_stddev",
            "workspace_peak_bytes", "workspace_allocator_peak_bytes", "speculative", "reps",
        },
        "benchmark P2048 test",
    )
    if {key: test.get(key) for key in ("label", "kind", "n_prompt", "n_gen", "requested_output_tokens")} != {
        "label": "pp2048", "kind": "pp", "n_prompt": 2048, "n_gen": 0,
        "requested_output_tokens": 1,
    }:
        raise ValueError("benchmark report test geometry differs")
    for key in ("workspace_peak_bytes", "workspace_allocator_peak_bytes"):
        if type(test.get(key)) is not int or test[key] < 0:
            raise ValueError(f"benchmark report has invalid {key}")
    for key in ("prefill_tok_s_mean", "prefill_seconds_mean", "total_seconds_mean"):
        if not _finite(test.get(key), positive=True):
            raise ValueError(f"benchmark report has invalid {key}")
    for key in ("prefill_tok_s_stddev", "prepare_seconds_mean", "prepare_seconds_stddev", "prefill_seconds_stddev", "total_seconds_stddev"):
        if not _finite(test.get(key)):
            raise ValueError(f"benchmark report has invalid {key}")
    for key in (
        "decode_output_tok_s_mean", "decode_output_tok_s_stddev",
        "decode_engine_tok_s_mean", "decode_engine_tok_s_stddev",
        "whole_output_tok_s_mean", "whole_output_tok_s_stddev",
        "decode_seconds_mean", "decode_seconds_stddev",
    ):
        if test.get(key) is not None:
            raise ValueError(f"prefill-only benchmark unexpectedly reports {key}")
    reps = test.get("reps")
    if not isinstance(reps, list) or len(reps) != 1 or not isinstance(reps[0], dict):
        raise ValueError("benchmark report must retain exactly one measured repetition")
    rep = reps[0]
    _require_exact_keys(
        rep,
        {"generated_output_tokens", "decode_output_tokens", "decode_engine_tokens", "generated_token_ids_by_lane", "timings", "speculative"},
        "benchmark repetition",
    )
    if rep.get("generated_output_tokens") != 1 or rep.get("decode_output_tokens") is not None or rep.get(
        "decode_engine_tokens"
    ) is not None:
        raise ValueError("benchmark repetition output geometry differs")
    tokens = rep.get("generated_token_ids_by_lane")
    if (
        not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], list)
        or len(tokens[0]) != 1 or type(tokens[0][0]) is not int or tokens[0][0] < 0
    ):
        raise ValueError("benchmark repetition lacks its exact semantic output token")
    timings = rep.get("timings")
    if not isinstance(timings, dict):
        raise ValueError("benchmark repetition timings are not an object")
    _require_exact_keys(
        timings,
        {"prepare_seconds", "vision_seconds", "prefill_seconds", "decode_seconds", "total_seconds"},
        "benchmark repetition timings",
    )
    if any(
        not _finite(timings.get(key), positive=key in ("prefill_seconds", "total_seconds"))
        for key in ("prepare_seconds", "vision_seconds", "prefill_seconds", "decode_seconds", "total_seconds")
    ):
        raise ValueError("benchmark repetition timings are invalid")
    if timings["vision_seconds"] != 0 or timings["decode_seconds"] != 0:
        raise ValueError("P2048 prefill repetition performed non-prefill model work")
    expected_aggregates = {
        "prepare_seconds_mean": timings["prepare_seconds"],
        "prefill_seconds_mean": timings["prefill_seconds"],
        "total_seconds_mean": timings["total_seconds"],
        "prefill_tok_s_mean": 2048.0 / float(timings["prefill_seconds"]),
    }
    if any(
        not math.isclose(float(test[key]), float(expected), rel_tol=1e-8, abs_tol=1e-10)
        for key, expected in expected_aggregates.items()
    ) or any(test[key] != 0 for key in (
        "prefill_tok_s_stddev", "prepare_seconds_stddev", "prefill_seconds_stddev",
        "total_seconds_stddev",
    )):
        raise ValueError("single-repetition benchmark aggregates differ from the retained timing")
    speculative = rep.get("speculative")
    expected_speculative = {
        "enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
        "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
        "acceptance_length": None, "accepted_per_position": [],
    }
    if speculative != expected_speculative or test.get("speculative") != expected_speculative:
        raise ValueError("benchmark report contains speculative work")
    return {
        "prefill_ms": float(timings["prefill_seconds"]) * 1000.0,
        "total_ms": float(timings["total_seconds"]) * 1000.0,
        "tokens": tokens,
        "workspace_peak_bytes": test["workspace_peak_bytes"],
        "workspace_allocator_peak_bytes": test["workspace_allocator_peak_bytes"],
        "environment": environment,
    }


def mad(values: Sequence[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def decide(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    if len(samples) != 16 or [sample.get("role") for sample in samples] != list(ROLE_ORDER):
        raise ValueError("whole A/B sample order differs from AB,BA repeated four times")
    controls: list[float] = []
    candidates: list[float] = []
    candidate_prefill: list[float] = []
    token_vectors = [sample.get("tokens") for sample in samples]
    if any(
        not isinstance(tokens, list) or len(tokens) != 1
        or not isinstance(tokens[0], list) or len(tokens[0]) != 1
        or type(tokens[0][0]) is not int or tokens[0][0] < 0
        for tokens in token_vectors
    ):
        raise ValueError("whole A/B sample lacks one exact semantic token vector")
    exact_parity = all(tokens == token_vectors[0] for tokens in token_vectors)
    workspace_equal = True
    environment_equal = True
    pairs: list[dict[str, object]] = []
    for index in range(0, len(samples), 2):
        first, second = samples[index:index + 2]
        control, candidate = (first, second) if first["role"] == "A" else (second, first)
        a = float(control["total_ms"])
        b = float(candidate["total_ms"])
        if not _finite(a, positive=True) or not _finite(b, positive=True):
            raise ValueError("whole A/B timing is invalid")
        candidate_prefill_ms = float(candidate["prefill_ms"])
        if not _finite(candidate_prefill_ms, positive=True):
            raise ValueError("whole A/B prefill diagnostic is invalid")
        controls.append(a)
        candidates.append(b)
        candidate_prefill.append(candidate_prefill_ms)
        pair_parity = control["tokens"] == candidate["tokens"] == token_vectors[0]
        pair_workspace = (
            control["workspace_peak_bytes"] == candidate["workspace_peak_bytes"]
            and control["workspace_allocator_peak_bytes"] == candidate["workspace_allocator_peak_bytes"]
        )
        pair_environment = control["environment"] == candidate["environment"]
        exact_parity &= pair_parity
        workspace_equal &= pair_workspace
        environment_equal &= pair_environment
        pairs.append({
            "order": str(first["role"]) + str(second["role"]),
            "control_total_ms": a,
            "candidate_total_ms": b,
            "total_saving_ms": a - b,
            "candidate_over_control_total": b / a,
            "exact_output_parity": pair_parity,
            "workspace_equal": pair_workspace,
            "environment_equal": pair_environment,
        })
    control_median = statistics.median(controls)
    candidate_median = statistics.median(candidates)
    control_mad = mad(controls)
    candidate_mad = mad(candidates)
    ratios = [b / a for a, b in zip(controls, candidates, strict=True)]
    savings = [a - b for a, b in zip(controls, candidates, strict=True)]
    ratio_median = statistics.median(ratios)
    ratio_mad = mad(ratios)
    saving_median = statistics.median(savings)
    saving_mad = mad(savings)
    ratio_upper = ratio_median + ROBUST_FACTOR * ratio_mad
    saving_lower = saving_median - ROBUST_FACTOR * saving_mad
    stable = (
        control_mad / control_median <= 0.02
        and candidate_mad / candidate_median <= 0.02
        and (max(controls) - min(controls)) / control_median <= 0.06
        and (max(candidates) - min(candidates)) / candidate_median <= 0.06
    )
    no_large_pair_regression = all(b <= 1.05 * a for a, b in zip(controls, candidates, strict=True))
    promoted = (
        stable and ratio_upper < 1.0 and no_large_pair_regression and exact_parity
        and workspace_equal and environment_equal and saving_lower >= 10.0
    )
    candidate_prefill_median = statistics.median(candidate_prefill)
    return {
        "classification": "promote" if promoted else ("inconclusive" if not stable else "reject"),
        "promotion_pass": promoted,
        "pair_order": list(PAIR_ORDER),
        "sample_count_per_role": 8,
        "robust_mad_factor": ROBUST_FACTOR,
        "control_total_median_ms": control_median,
        "control_total_mad_ms": control_mad,
        "candidate_total_median_ms": candidate_median,
        "candidate_total_mad_ms": candidate_mad,
        "paired_ratio_median": ratio_median,
        "paired_ratio_mad": ratio_mad,
        "robust_ratio_upper": ratio_upper,
        "required_robust_ratio_upper_exclusive": 1.0,
        "paired_total_saving_median_ms": saving_median,
        "paired_total_saving_mad_ms": saving_mad,
        "robust_whole_saving_lower_ms": saving_lower,
        "required_robust_whole_saving_lower_ms": 10.0,
        "stability_pass": stable,
        "no_candidate_sample_over_paired_control_1p05": no_large_pair_regression,
        "exact_semantic_output_parity": exact_parity,
        "semantic_token_vector": token_vectors[0],
        "workspace_identity_pass": workspace_equal,
        "environment_identity_pass": environment_equal,
        "all_16_token_vectors_identical": exact_parity,
        "floor_diagnostic_candidate_median_ms_max": 1024.0,
        "floor_diagnostic_candidate_prefill_median_ms": candidate_prefill_median,
        "floor_diagnostic_pass": candidate_prefill_median <= 1024.0,
        "pairs": pairs,
    }


def _inode(path: Path) -> tuple[int, int, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"owned output is not regular: {path}")
    return metadata.st_dev, metadata.st_ino, metadata.st_uid


def _unlink_if_owned(path: Path, owner: tuple[int, int, int]) -> bool:
    try:
        if _inode(path) != owner:
            return False
    except FileNotFoundError:
        return False
    path.unlink()
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes(path: Path, payload: bytes, validator: Callable[[bytes], None]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"create-only output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"output parent is not a real directory: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    temporary = Path(temporary_name)
    owner: tuple[int, int, int] | None = None
    linked = False
    try:
        metadata = os.fstat(descriptor)
        owner = metadata.st_dev, metadata.st_ino, metadata.st_uid
        if metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
            raise ValueError("pending output ownership is invalid")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short output write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        if _inode(temporary) != owner:
            raise ValueError("pending output inode changed")
        validator(payload)
        os.link(temporary, path)
        linked = True
        if _inode(path) != owner or path.read_bytes() != payload:
            raise ValueError("published output differs")
        validator(path.read_bytes())
        _fsync_directory(path.parent)
        if not _unlink_if_owned(temporary, owner):
            raise ValueError("pending output changed before cleanup")
        _fsync_directory(path.parent)
    except Exception:
        if linked and owner is not None:
            _unlink_if_owned(path, owner)
        raise
    finally:
        os.close(descriptor)
        if owner is not None:
            _unlink_if_owned(temporary, owner)


def publish_json(path: Path, value: object, validator: Callable[[object], None]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def validate_bytes(raw: bytes) -> None:
        validator(json.loads(raw))

    publish_bytes(path, payload, validate_bytes)


def _raw_path(index: int, role: str) -> Path:
    return RAW_DIRECTORY / f"sample-{index + 1:02d}-{role.lower()}.json"


def validate_final_report(
    value: object, expected_identities: dict[str, dict[str, object]] | None = None
) -> None:
    if not isinstance(value, dict) or value.get("schema") != REPORT_SCHEMA or value.get(
        "schema_version"
    ) != REPORT_SCHEMA_VERSION or value.get("status") != "measured":
        raise ValueError("whole A/B aggregate identity differs")
    _require_exact_keys(
        value,
        {
            "schema", "schema_version", "status", "repository_root", "pair_order",
            "per_process", "input_identities", "power_before", "power_after", "samples",
            "decision",
        },
        "whole A/B aggregate",
    )
    if value.get("repository_root") != str(ROOT) or value.get("pair_order") != list(PAIR_ORDER):
        raise ValueError("whole A/B aggregate route/order differs")
    if value.get("per_process") != {"measured_repetitions": 1, "discarded_warmups": 1}:
        raise ValueError("whole A/B aggregate per-process timing contract differs")
    expected_power_command = [
        "/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname", "--showprofile",
        "--showperflevel", "--json",
    ]
    for label in ("power_before", "power_after"):
        power = value.get(label)
        if not isinstance(power, dict) or power != {
            "command": expected_power_command,
            "device": {"ordinal": 0, "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201"},
            "rocm_smi_performance_level": "auto",
            "sysfs_path": str(POWER),
            "sysfs_value": "auto",
        }:
            raise ValueError(f"whole A/B aggregate {label} differs")
    identities = value.get("input_identities")
    current_identities = input_identities() if expected_identities is None else expected_identities
    if not isinstance(identities, dict) or identities != current_identities:
        raise ValueError("whole A/B aggregate input identity changed")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != len(ROLE_ORDER):
        raise ValueError("whole A/B aggregate sample inventory differs")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("whole A/B aggregate sample is not an object")
        role = ROLE_ORDER[index]
        raw_path = _raw_path(index, role)
        command = command_for(role)
        raw_identity = _regular_identity(raw_path)
        if raw_identity["mode"] != 0o444 or raw_identity["uid"] != os.getuid():
            raise ValueError("whole A/B raw report mode or owner differs")
        raw_value = json.loads(raw_path.read_text(encoding="utf-8"))
        if _regular_identity(raw_path) != raw_identity:
            raise ValueError("whole A/B raw report changed while being revalidated")
        measured = validate_bench_report(raw_value, role, command)
        expected_sample = {
            "index": index + 1, "role": role, "command": command,
            "raw_report": raw_identity, **measured,
        }
        if sample != expected_sample:
            raise ValueError("whole A/B sample differs from its raw authority")
    if value.get("decision") != decide(samples):
        raise ValueError("whole A/B aggregate decision differs")


def run_campaign() -> dict[str, object]:
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f"runner must be invoked from the repository root: {ROOT}")
    if os.path.lexists(REPORT) or os.path.lexists(RAW_DIRECTORY):
        raise ValueError("whole A/B output namespace must be fresh")
    if not REPORT.parent.is_dir() or REPORT.parent.is_symlink():
        raise ValueError("whole A/B report parent must be a real directory")
    started = input_identities()
    power_before = power_identity()
    RAW_DIRECTORY.mkdir(mode=0o755)
    directory_owner = os.lstat(RAW_DIRECTORY)
    samples: list[dict[str, object]] = []
    raw_owners: dict[Path, tuple[int, int, int]] = {}
    try:
        for index, role in enumerate(ROLE_ORDER):
            command = command_for(role)
            process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            if process.returncode != 0:
                raise RuntimeError(f"whole A/B {role} sample {index + 1} failed: {process.stderr}")
            try:
                validate_bench_stderr(process.stderr)
            except ValueError as error:
                raise RuntimeError(
                    f"whole A/B {role} sample {index + 1} stderr protocol differs: "
                    f"{process.stderr}"
                ) from error
            try:
                raw_value = json.loads(process.stdout)
            except json.JSONDecodeError as error:
                raise ValueError(f"whole A/B {role} sample {index + 1} emitted invalid JSON") from error
            measured = validate_bench_report(raw_value, role, command)
            raw_path = _raw_path(index, role)
            publish_json(raw_path, raw_value, lambda value, r=role, c=command: validate_bench_report(value, r, c))
            raw_owners[raw_path] = _inode(raw_path)
            samples.append({
                "index": index + 1,
                "role": role,
                "command": command,
                "raw_report": {
                    **_regular_identity(raw_path),
                },
                **measured,
            })
        power_after = power_identity()
        if power_after["device"] != power_before["device"] or power_after["sysfs_value"] != "auto":
            raise ValueError("R9700 identity or auto-power state changed during whole A/B")
        finished = input_identities()
        if finished != started:
            raise ValueError("whole A/B input bytes or inode identities changed during measurement")
        report = {
            "schema": REPORT_SCHEMA,
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "measured",
            "repository_root": str(ROOT),
            "pair_order": list(PAIR_ORDER),
            "per_process": {"measured_repetitions": 1, "discarded_warmups": 1},
            "input_identities": started,
            "power_before": power_before,
            "power_after": power_after,
            "samples": samples,
            "decision": decide(samples),
        }
        publish_json(REPORT, report, lambda value: validate_final_report(value, started))
        return report
    except Exception:
        for path, owner in raw_owners.items():
            _unlink_if_owned(path, owner)
        if os.path.lexists(RAW_DIRECTORY):
            current = os.lstat(RAW_DIRECTORY)
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino, current.st_uid)
                == (directory_owner.st_dev, directory_owner.st_ino, directory_owner.st_uid)
                and not any(RAW_DIRECTORY.iterdir())
            ):
                RAW_DIRECTORY.rmdir()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path(os.path.abspath(args.output))
    if output != REPORT:
        parser.error(f"--output must be the fixed authority path {REPORT}")
    try:
        report = run_campaign()
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"scalar-base-whole-ab: {error}\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0 if report["decision"]["promotion_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
