#!/usr/bin/env python3
"""Build the selected marker-bearing profiler binary and publish its derivation receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from tools.bench.prepare_selected_decode_memory_profile import (
    MARKER_SOURCE, PROFILE_BUILD_DIR, PROFILE_BUILD_PRODUCER,
    PROFILE_BUILD_RECEIPT_TYPE, PROFILE_ROUTE_RESOLVER, lexical_absolute,
    profile_build_commands, sha, validate_profile_build_receipt,
)
from tools.bench.run_ninfer_bench_matrix import inspect_executable
from tools.ppl.pareto import load_payload, validate_terminal_production_authority

REPO = Path(__file__).resolve().parents[2]
EXPECTED_RECEIPT = (
    REPO / "profiles/rocprof/selected-ordinary-decode-profile-build-20260905/receipt.json"
)
FINALIST_TOKEN = "prefill-chunk-selection-pipeline-20260905/run-finalists.sh"


def snapshot(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "file_size_bytes": resolved.stat().st_size,
            "sha256": sha(resolved)}


def finalist_campaign_is_live() -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if FINALIST_TOKEN in command and "--execute-gpu-campaign" in command:
            return True
    return False


def resolve_route(selection: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ninfer_selected_profile_route",
                                                  PROFILE_ROUTE_RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load selected route resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(selection)


def publish(path: Path, value: dict, validator: Callable[[Path], None]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    published = False
    durable = False
    inode = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_stat = os.stat(temporary, follow_symlinks=False)
        inode = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.link(temporary, path)
        published = True
        current = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != inode:
            raise ValueError("published profile-build receipt inode changed")
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("published profile-build receipt bytes changed")
        validator(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        durable = True
    finally:
        Path(temporary).unlink(missing_ok=True)
        if published and not durable and inode is not None:
            try:
                current = os.stat(path, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == inode:
                    path.unlink()
            except FileNotFoundError:
                pass


def build(selection_path: Path, output: Path) -> dict:
    selection_path = selection_path.resolve(strict=True)
    output = lexical_absolute(output)
    if output != EXPECTED_RECEIPT or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("profile-build receipt must use its fixed prepared package path")
    if os.path.lexists(output):
        raise ValueError("profile-build receipt namespace is already occupied")
    if finalist_campaign_is_live():
        raise ValueError("live 32K finalist campaign forbids instrumentation rebuild")
    if os.path.lexists(PROFILE_BUILD_DIR):
        raise ValueError("distinct instrumentation build directory is not fresh")
    route = resolve_route(selection_path)
    selection_raw = selection_path.read_bytes()
    selection = load_payload(selection_raw.decode("utf-8"))
    terminal, _ = validate_terminal_production_authority(selection)
    whole = route["source_matrices"]["pareto-whole"]
    manifest_path = Path(whole["path"]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compiled = {
        "q4_activation_bits": 8, "w8_activation_bits": 8,
        "fp8_qk_wmma_enabled": True,
        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
        "fp8_qk_wmma_t1_min_context": 64, "fp8_qk_wmma_t2_min_context": 320,
        "q4_prefill_cta_profile":
            "m64n128-pingpong-n16-k16-scalar-base-production",
    }
    selected_route = {
        "winner": terminal["winner"], "artifact": terminal["winner_artifact"],
        "cache_profile": terminal["winner_cache_profile"],
        "execution_profile": terminal["winner_execution_profile"],
        "prefill_chunk": selection["selected_prefill_chunk"],
    }
    sparse = selected_route["execution_profile"]["xattention_profile"] != "dense"
    definitions = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
        "NINFER_BUILD_BENCHMARKS": "ON",
        "NINFER_R9700_KV_VALUE_GROUP": str(selected_route["cache_profile"]["value_group"]),
        "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
        "NINFER_R9700_Q4_PREFILL_PINGPONG_QUALIFICATION": "OFF",
        "NINFER_R9700_W8_ACTIVATION_BITS": "8", "NINFER_R9700_FP8_QK_WMMA": "1",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "ON" if sparse else "OFF",
        "NINFER_R9700_XATTENTION_STRIDE": "16",
        "NINFER_R9700_XATTENTION_TAU_PERMILLE": "900",
    }
    commands = profile_build_commands(definitions)
    subprocess.run(commands["configure"], cwd=REPO, check=True)
    subprocess.run(commands["build"], cwd=REPO, check=True)
    binary = PROFILE_BUILD_DIR / "bench/ninfer_bench"
    cache = PROFILE_BUILD_DIR / "CMakeCache.txt"
    compile_database = PROFILE_BUILD_DIR / "compile_commands.json"
    marker = snapshot(MARKER_SOURCE)
    receipt = {
        "artifact_type": PROFILE_BUILD_RECEIPT_TYPE, "schema_version": 1,
        "status": "passed", "purpose": "profiler_attribution_only",
        "profile_timing_admissible": False,
        "producer": snapshot(PROFILE_BUILD_PRODUCER),
        "route_resolver": snapshot(PROFILE_ROUTE_RESOLVER),
        "terminal_selection": {"path": str(selection_path),
                               "sha256": hashlib.sha256(selection_raw).hexdigest()},
        "source_matrix": {"path": str(manifest_path), "sha256": whole["sha256"]},
        "artifact": manifest["artifact"],
        "terminal_timing_executable": manifest["bench"],
        "ordinary_round_marker_source": marker,
        "selected_route": selected_route, "compiled_route": compiled,
        "hybrid_shared_workspace_authority": manifest.get(
            "hybrid_shared_workspace_authority"),
        "instrumentation_executable": inspect_executable(binary),
        "build_directory": str(PROFILE_BUILD_DIR), "cmake_cache": snapshot(cache),
        "compile_database": snapshot(compile_database),
        "compile_definitions": definitions, "build_commands": commands,
    }
    def validate_published(path: Path) -> None:
        validated, _, _ = validate_profile_build_receipt(
            path, selection_path=selection_path,
            selection_sha256=receipt["terminal_selection"]["sha256"],
            manifest_path=manifest_path, manifest_sha256=whole["sha256"],
            artifact=manifest["artifact"], terminal_executable=manifest["bench"],
            marker_source=marker, selected_route=selected_route,
            compiled_route=compiled,
            hybrid_workspace_authority=manifest.get("hybrid_shared_workspace_authority"),
        )
        if validated != receipt:
            raise ValueError("profile-build receipt changed during publication")

    publish(output, receipt, validate_published)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        build(args.selection, args.out)
    except (OSError, subprocess.CalledProcessError, UnicodeError, json.JSONDecodeError,
            KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
