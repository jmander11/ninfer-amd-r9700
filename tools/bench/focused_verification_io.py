#!/usr/bin/env python3
"""Publication contract for the selected-route focused GPU verification."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from tools.ppl.pareto import load_payload, validate_terminal_production_authority


ARTIFACT_TYPE = "ninfer_r9700_post_terminal_focused_verification"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_absent(paths: list[Path]) -> None:
    occupied = [str(path) for path in paths if os.path.lexists(path)]
    if occupied:
        raise ValueError("focused-verification namespace already exists: " + ", ".join(occupied))


def inode_identity(path: Path) -> tuple[int, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"focused-verification owned path is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def unlink_if_owned(path: Path, owner: tuple[int, int]) -> bool:
    try:
        current = inode_identity(path)
    except FileNotFoundError:
        return False
    if current != owner:
        return False
    path.unlink()
    return True


def validate_report(value: object, closure: Path) -> dict:
    if not isinstance(value, dict) or (
        value.get("artifact_type") != ARTIFACT_TYPE
        or value.get("schema_version") != 1
        or value.get("status") != "passed"
        or value.get("mode") != "selected_route_gpu_correctness_integration"
        or value.get("maximum_runtime_concurrency") != 4
    ):
        raise ValueError("focused-verification report has invalid identity/status")
    route = value.get("route")
    closure_identity = value.get("prepared_closure")
    checks = value.get("checks")
    if (
        not isinstance(route, dict)
        or route.get("artifact_type") != "ninfer_r9700_post_terminal_verification_route"
        or route.get("maximum_runtime_concurrency") != 4
        or not isinstance(closure_identity, dict)
        or closure_identity.get("path") != str(closure.resolve())
        or closure_identity.get("sha256") != file_sha256(closure)
        or not isinstance(checks, dict)
        or checks.get("native_host_contracts") != 15
        or checks.get("external_schema_contracts") != 8
        or checks.get("selected_device_qualifiers") not in (29, 30)
    ):
        raise ValueError("focused-verification report lacks exact route/check provenance")
    selection = route.get("terminal_selection")
    artifact = route.get("artifact")
    benchmark = route.get("benchmark")
    matrices = route.get("source_matrices")
    interpreter = route.get("test_python")
    build_identity = route.get("build_identity")
    if (
        not isinstance(selection, dict)
        or not isinstance(artifact, dict)
        or not isinstance(benchmark, dict)
        or not isinstance(matrices, dict)
        or set(matrices) != {"pareto-capacity", "pareto-whole"}
        or not isinstance(interpreter, dict)
        or not isinstance(build_identity, dict)
        or set(build_identity) != {"cmake_cache", "ctest_root"}
    ):
        raise ValueError("focused-verification route lacks selected build identities")
    identities = [selection, artifact, benchmark, *matrices.values(), *build_identity.values()]
    for identity in identities:
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError("focused-verification route contains malformed file identity")
        path = Path(identity["path"])
        if not path.is_file() or identity.get("sha256") != file_sha256(path):
            raise ValueError("focused-verification route file identity changed")
    launcher = Path(interpreter.get("launcher_path", ""))
    if (
        not launcher.is_absolute()
        or not launcher.is_file()
        or not os.access(launcher, os.X_OK)
        or interpreter.get("executable_sha256") != file_sha256(launcher)
    ):
        raise ValueError("focused-verification interpreter identity changed")
    terminal, selected = validate_terminal_production_authority(
        load_payload(Path(selection["path"]).read_text(encoding="utf-8"))
    )
    if (
        selection.get("sha256") != file_sha256(Path(selection["path"]))
        or route.get("winner") != terminal["winner"]
        or route.get("cache_profile") != terminal["winner_cache_profile"]
        or route.get("execution_profile") != terminal["winner_execution_profile"]
        or route.get("selected_prefill_chunk") != selected.get("prefill_chunk")
    ):
        raise ValueError("focused-verification route differs from terminal winner")
    planner = route.get("hybrid_width_tool")
    if planner is not None:
        planner_path = Path(planner.get("path", "")) if isinstance(planner, dict) else Path()
        if not planner_path.is_file() or planner.get("sha256") != file_sha256(planner_path):
            raise ValueError("focused-verification hybrid planner identity changed")
    return route


def publish_success(
    route_path: Path,
    closure: Path,
    pending: Path,
    output: Path,
    device_qualifiers: int,
) -> None:
    require_absent([pending, output])
    route = json.loads(route_path.read_text(encoding="utf-8"))
    value = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "status": "passed",
        "mode": "selected_route_gpu_correctness_integration",
        "route": route,
        "prepared_closure": {
            "path": str(closure.resolve()),
            "sha256": file_sha256(closure),
        },
        "checks": {
            "native_host_contracts": 15,
            "external_schema_contracts": 8,
            "selected_device_qualifiers": device_qualifiers,
        },
        "maximum_runtime_concurrency": 4,
    }
    validate_report(value, closure)
    pending_owner: tuple[int, int] | None = None
    output_owner: tuple[int, int] | None = None
    try:
        with pending.open("x", encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            pending_owner = (metadata.st_dev, metadata.st_ino)
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, output)
        if inode_identity(pending) != pending_owner or inode_identity(output) != pending_owner:
            raise ValueError("focused-verification hard-link publication inode differs")
        output_owner = pending_owner
        validate_report(json.loads(output.read_text(encoding="utf-8")), closure)
        if not unlink_if_owned(pending, pending_owner):
            raise ValueError("focused-verification pending inode changed before cleanup")
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if pending_owner is not None:
            unlink_if_owned(pending, pending_owner)
        if output_owner is not None:
            unlink_if_owned(output, output_owner)
        raise
