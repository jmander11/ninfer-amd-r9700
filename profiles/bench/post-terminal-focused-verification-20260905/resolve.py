#!/usr/bin/env python3
"""Resolve the exact schema-v7 winner and its compile-matched verification build."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.bench.run_ninfer_bench_matrix import (
    MATRIX_SCHEMA_VERSION,
    PRODUCT_CONCURRENCIES,
    file_sha256,
    inspect_artifact,
    bind_n16_migration_receipt,
    inspect_executable,
    require_fp8_hybrid_artifact,
    validate_hybrid_shared_workspace_authority,
)
from tools.bench.focused_verification_io import unlink_if_owned
from tools.ppl.pareto import load_payload, validate_terminal_production_authority

HYBRID_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"


def is_hybrid_weights(weights_id: object) -> bool:
    return weights_id == HYBRID_WEIGHTS_ID


def inspect_test_python(launcher: Path) -> dict:
    """Validate an explicit interpreter without resolving away its venv launcher."""
    if not launcher.is_absolute() or not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError("TEST_PYTHON must be an existing executable absolute path")
    probe = r"""
import importlib.metadata
import json
import os
import sys

import pytest
import safetensors
import torch

print(json.dumps({
    "runtime_executable": sys.executable,
    "python_version": ".".join(map(str, sys.version_info[:3])),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("pytest", "torch", "safetensors")
    },
}))
"""
    try:
        completed = subprocess.run(
            [str(launcher), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        identity = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(
            "TEST_PYTHON cannot import pytest, Torch, and safetensors"
        ) from error
    required = {"runtime_executable", "python_version", "prefix", "base_prefix", "packages"}
    if set(identity) != required or set(identity.get("packages", {})) != {
        "pytest", "torch", "safetensors"
    }:
        raise ValueError("TEST_PYTHON returned malformed interpreter identity")
    prefix = Path(identity["prefix"])
    pyvenv = prefix / "pyvenv.cfg"
    return {
        "launcher_path": str(launcher),
        "runtime_executable": identity["runtime_executable"],
        "executable_sha256": file_sha256(launcher),
        "python_version": identity["python_version"],
        "prefix": identity["prefix"],
        "base_prefix": identity["base_prefix"],
        "pyvenv_cfg": (
            {"path": str(pyvenv), "sha256": file_sha256(pyvenv)} if pyvenv.is_file() else None
        ),
        "packages": identity["packages"],
    }


def cache_value(cache: Path, name: str) -> str:
    prefix = f"{name}:"
    rows = [line for line in cache.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]
    if len(rows) != 1 or "=" not in rows[0]:
        raise ValueError(f"selected build lacks unique CMake cache value {name}")
    return rows[0].split("=", 1)[1]


def resolve(selection_path: Path, test_python: Path | None = None) -> dict:
    test_python_identity = inspect_test_python(test_python) if test_python is not None else None
    selection_path = selection_path.resolve(strict=True)
    selection_raw = selection_path.read_bytes()
    selection_sha = file_sha256(selection_path)
    value = load_payload(selection_raw.decode("utf-8"))
    terminal, selected = validate_terminal_production_authority(value)
    winner = terminal["winner"]
    candidates = [row for row in value["candidates"] if row.get("name") == winner]
    sources = [row for row in value["source_provenance"] if row.get("candidate") == winner]
    if len(candidates) != 1 or len(sources) != 1:
        raise ValueError("terminal winner lacks unique candidate/source provenance")
    candidate, source = candidates[0], sources[0]
    recipe = terminal["winner_artifact"]
    cache_profile = terminal["winner_cache_profile"]
    execution = terminal["winner_execution_profile"]
    chunk = value["selected_prefill_chunk"]
    if (
        candidate.get("weight_recipe") != recipe
        or candidate.get("cache_profile") != cache_profile
        or candidate.get("execution_profile") != execution
        or selected.get("name") != winner
    ):
        raise ValueError("terminal winner tuple differs from its selected candidate")

    bindings = source.get("matrices")
    if not isinstance(bindings, dict) or set(bindings) != {"pareto-capacity", "pareto-whole"}:
        raise ValueError("terminal winner lacks exact capacity/whole matrix bindings")
    manifests = {}
    bound_paths = []
    for preset in ("pareto-capacity", "pareto-whole"):
        binding = bindings[preset]
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise ValueError("terminal winner matrix binding is malformed")
        path = Path(binding["path"]).resolve(strict=True)
        if file_sha256(path) != binding.get("sha256"):
            raise ValueError("terminal winner matrix bytes changed")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("artifact_type") != "ninfer_bench_matrix_run"
            or manifest.get("schema_version") != MATRIX_SCHEMA_VERSION
            or manifest.get("preset") != preset
            or manifest.get("concurrency") != list(PRODUCT_CONCURRENCIES)
            or manifest.get("selected_prefill_chunk") != chunk
            or manifest.get("expected_kv_value_group") != cache_profile["value_group"]
            or manifest.get("expected_xattention_profile") != execution["xattention_profile"]
            or manifest.get("artifact") != source.get("artifact")
            or manifest.get("bench") != source.get("benchmark_executable")
        ):
            raise ValueError("terminal winner matrix differs from selected route")
        manifests[preset] = manifest
        bound_paths.append((path, binding["sha256"]))
    if manifests["pareto-capacity"]["artifact"] != manifests["pareto-whole"]["artifact"]:
        raise ValueError("terminal winner matrices bind different artifacts")
    if manifests["pareto-capacity"]["bench"] != manifests["pareto-whole"]["bench"]:
        raise ValueError("terminal winner matrices bind different executables")

    manifest = manifests["pareto-whole"]
    artifact_path = Path(manifest["artifact"]["path"]).resolve(strict=True)
    bench_path = Path(manifest["bench"]["path"]).resolve(strict=True)
    artifact = bind_n16_migration_receipt(artifact_path, inspect_artifact(artifact_path))
    hybrid = is_hybrid_weights(recipe["weights_id"])
    planner = None
    if hybrid:
        artifact = require_fp8_hybrid_artifact(artifact_path, artifact)
        authorities = []
        for matrix in manifests.values():
            if matrix.get("required_candidate_identity") != "fp8-hybrid-selection-authority":
                raise ValueError("selected hybrid matrix lacks its candidate authority")
            authorities.append(validate_hybrid_shared_workspace_authority(
                matrix.get("hybrid_shared_workspace_authority"), [chunk]
            ))
        if authorities[0] != authorities[1]:
            raise ValueError("terminal winner matrices bind different hybrid planners")
        planner_path = Path(authorities[0]["tool"]["path"]).resolve(strict=True)
        if inspect_executable(planner_path) != authorities[0]["tool"]:
            raise ValueError("terminal winner hybrid planner bytes changed")
        planner = authorities[0]["tool"]
    elif any(
        matrix.get("required_candidate_identity") is not None
        or matrix.get("hybrid_shared_workspace_authority") is not None
        for matrix in manifests.values()
    ):
        raise ValueError("selected non-hybrid route carries hybrid authority")
    if artifact != manifest["artifact"] or inspect_executable(bench_path) != manifest["bench"]:
        raise ValueError("selected artifact or benchmark bytes changed")

    if bench_path.name != "ninfer_bench" or bench_path.parent.name != "bench":
        raise ValueError("selected benchmark path does not identify its build tree")
    build = bench_path.parent.parent
    cmake_cache = build / "CMakeCache.txt"
    ctest_file = build / "CTestTestfile.cmake"
    if not cmake_cache.is_file() or not ctest_file.is_file():
        raise ValueError("selected benchmark build lacks CMake/CTest identity")
    sparse = execution["xattention_profile"] == "b128-s16-tau900"
    expected_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_GENERATOR": "Ninja",
        "NINFER_R9700_KV_VALUE_GROUP": str(cache_profile["value_group"]),
        "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
        "NINFER_R9700_W8_ACTIVATION_BITS": "8",
        "NINFER_R9700_FP8_QK_WMMA": "1",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "ON" if sparse else "OFF",
        "NINFER_R9700_XATTENTION_STRIDE": "16",
        "NINFER_R9700_XATTENTION_TAU_PERMILLE": "900",
    }
    for name, expected in expected_cache.items():
        if cache_value(cmake_cache, name) != expected:
            raise ValueError(f"selected build {name} differs from terminal execution profile")
    if file_sha256(selection_path) != selection_sha:
        raise ValueError("terminal selection changed while resolving verification route")
    if any(file_sha256(path) != digest for path, digest in bound_paths):
        raise ValueError("terminal matrix changed while resolving verification route")
    route = {
        "artifact_type": "ninfer_r9700_post_terminal_verification_route",
        "schema_version": 1,
        "terminal_selection": {"path": str(selection_path), "sha256": selection_sha},
        "winner": winner,
        "artifact": {**artifact, "path": str(artifact_path)},
        "benchmark": {**manifest["bench"], "path": str(bench_path)},
        "source_matrices": {
            preset: {"path": str(path), "sha256": digest}
            for preset, (path, digest) in zip(
                ("pareto-capacity", "pareto-whole"), bound_paths, strict=True
            )
        },
        "build_directory": str(build.resolve()),
        "build_identity": {
            "cmake_cache": {"path": str(cmake_cache), "sha256": file_sha256(cmake_cache)},
            "ctest_root": {"path": str(ctest_file), "sha256": file_sha256(ctest_file)},
        },
        "cache_profile": cache_profile,
        "execution_profile": execution,
        "selected_prefill_chunk": chunk,
        "hybrid_width_tool": planner,
        "maximum_runtime_concurrency": 4,
    }
    if test_python_identity is not None:
        route["test_python"] = test_python_identity
    return route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--test-python", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    try:
        value = resolve(args.selection, args.test_python)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    owner = None
    try:
        with args.out.open("x", encoding="utf-8") as output:
            metadata = os.fstat(output.fileno())
            owner = (metadata.st_dev, metadata.st_ino)
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        print(f"{owner[0]}:{owner[1]}")
    except Exception:
        if owner is not None:
            unlink_if_owned(args.out, owner)
        raise


if __name__ == "__main__":
    main()
