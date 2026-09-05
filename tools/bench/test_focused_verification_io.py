import json
import sys
import tempfile
from pathlib import Path

import pytest

from tools.bench import focused_verification_io as publication


def identity(path: Path) -> dict:
    path.write_text("{}\n", encoding="utf-8")
    return {"path": str(path), "sha256": publication.file_sha256(path)}


def route(root: Path) -> dict:
    selection = identity(root / "selection.json")
    return {
        "artifact_type": "ninfer_r9700_post_terminal_verification_route",
        "terminal_selection": selection,
        "winner": "winner",
        "artifact": identity(root / "artifact.ninfer"),
        "benchmark": identity(root / "ninfer_bench"),
        "source_matrices": {
            "pareto-capacity": identity(root / "capacity.json"),
            "pareto-whole": identity(root / "whole.json"),
        },
        "build_identity": {
            "cmake_cache": identity(root / "CMakeCache.txt"),
            "ctest_root": identity(root / "CTestTestfile.cmake"),
        },
        "cache_profile": {"value_group": 16},
        "execution_profile": {"xattention_profile": "dense"},
        "selected_prefill_chunk": 4096,
        "hybrid_width_tool": None,
        "test_python": {
            "launcher_path": sys.executable,
            "executable_sha256": publication.file_sha256(Path(sys.executable)),
        },
        "maximum_runtime_concurrency": 4,
    }


def test_dangling_namespace_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "result.json"
        output.symlink_to(Path(directory) / "missing")
        with pytest.raises(ValueError, match="namespace already exists"):
            publication.require_absent([output])


def test_success_is_exclusively_published_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        route_path = root / "route.json"
        closure = root / "prepared.sha256"
        pending = root / "result.pending.json"
        output = root / "result.json"
        selected_route = route(root)
        route_path.write_text(json.dumps(selected_route), encoding="utf-8")
        closure.write_text("closure\n", encoding="utf-8")
        monkeypatch.setattr(
            publication,
            "validate_terminal_production_authority",
            lambda value: (
                {
                    "winner": "winner",
                    "winner_cache_profile": {"value_group": 16},
                    "winner_execution_profile": {"xattention_profile": "dense"},
                },
                {"prefill_chunk": 4096},
            ),
        )
        publication.publish_success(route_path, closure, pending, output, 29)
        assert publication.validate_report(
            json.loads(output.read_text()), closure
        ) == selected_route
        assert not pending.exists()


def test_existing_output_is_preserved() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        route_path = root / "route.json"
        closure = root / "prepared.sha256"
        pending = root / "result.pending.json"
        output = root / "result.json"
        route_path.write_text(json.dumps(route(root)), encoding="utf-8")
        closure.write_text("closure\n", encoding="utf-8")
        output.write_text("owned\n", encoding="utf-8")
        with pytest.raises(ValueError, match="namespace already exists"):
            publication.publish_success(route_path, closure, pending, output, 29)
        assert output.read_text() == "owned\n"


def test_owned_cleanup_preserves_replacement_inode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "route.json"
        path.write_text("owned\n", encoding="utf-8")
        owner = publication.inode_identity(path)
        replacement = Path(directory) / "replacement.json"
        replacement.write_text("replacement\n", encoding="utf-8")
        replacement.replace(path)
        assert not publication.unlink_if_owned(path, owner)
        assert path.read_text(encoding="utf-8") == "replacement\n"


def test_publication_failure_preserves_replacement_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        route_path = root / "route.json"
        closure = root / "prepared.sha256"
        pending = root / "result.pending.json"
        output = root / "result.json"
        route_path.write_text(json.dumps(route(root)), encoding="utf-8")
        closure.write_text("closure\n", encoding="utf-8")
        monkeypatch.setattr(
            publication,
            "validate_terminal_production_authority",
            lambda value: (
                {
                    "winner": "winner",
                    "winner_cache_profile": {"value_group": 16},
                    "winner_execution_profile": {"xattention_profile": "dense"},
                },
                {"prefill_chunk": 4096},
            ),
        )
        real_validate = publication.validate_report
        calls = 0

        def replace_after_publication(value, bound_closure):
            nonlocal calls
            calls += 1
            if calls == 2:
                output.unlink()
                output.write_text("replacement\n", encoding="utf-8")
                raise RuntimeError("late validation failure")
            return real_validate(value, bound_closure)

        monkeypatch.setattr(publication, "validate_report", replace_after_publication)
        with pytest.raises(RuntimeError, match="late validation failure"):
            publication.publish_success(route_path, closure, pending, output, 29)
        assert not pending.exists()
        assert output.read_text(encoding="utf-8") == "replacement\n"


def test_pending_is_inode_owned_before_write_fsync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        route_path = root / "route.json"
        closure = root / "prepared.sha256"
        pending = root / "result.pending.json"
        output = root / "result.json"
        route_path.write_text(json.dumps(route(root)), encoding="utf-8")
        closure.write_text("closure\n", encoding="utf-8")
        monkeypatch.setattr(
            publication,
            "validate_terminal_production_authority",
            lambda value: (
                {
                    "winner": "winner",
                    "winner_cache_profile": {"value_group": 16},
                    "winner_execution_profile": {"xattention_profile": "dense"},
                },
                {"prefill_chunk": 4096},
            ),
        )
        monkeypatch.setattr(publication.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync")))
        with pytest.raises(OSError, match="fsync"):
            publication.publish_success(route_path, closure, pending, output, 29)
        assert not pending.exists()
        assert not output.exists()
