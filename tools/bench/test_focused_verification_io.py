import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


def terminal_authority(_value):
    return (
        {
            "winner": "winner",
            "winner_cache_profile": {"value_group": 16},
            "winner_execution_profile": {"xattention_profile": "dense"},
        },
        {"prefill_chunk": 4096},
    )


class FocusedVerificationIoTest(unittest.TestCase):
    def test_dangling_namespace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.symlink_to(Path(directory) / "missing")
            with self.assertRaisesRegex(ValueError, "namespace already exists"):
                publication.require_absent([output])

    def test_success_is_exclusively_published_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_path = root / "route.json"
            closure = root / "prepared.sha256"
            pending = root / "result.pending.json"
            output = root / "result.json"
            selected_route = route(root)
            route_path.write_text(json.dumps(selected_route), encoding="utf-8")
            closure.write_text("closure\n", encoding="utf-8")
            with patch.object(
                publication, "validate_terminal_production_authority", terminal_authority
            ):
                publication.publish_success(route_path, closure, pending, output, 29)
                self.assertEqual(
                    publication.validate_report(json.loads(output.read_text()), closure),
                    selected_route,
                )
            self.assertFalse(pending.exists())

    def test_existing_output_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_path = root / "route.json"
            closure = root / "prepared.sha256"
            pending = root / "result.pending.json"
            output = root / "result.json"
            route_path.write_text(json.dumps(route(root)), encoding="utf-8")
            closure.write_text("closure\n", encoding="utf-8")
            output.write_text("owned\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "namespace already exists"):
                publication.publish_success(route_path, closure, pending, output, 29)
            self.assertEqual(output.read_text(), "owned\n")

    def test_owned_cleanup_preserves_replacement_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.json"
            path.write_text("owned\n", encoding="utf-8")
            owner = publication.inode_identity(path)
            replacement = Path(directory) / "replacement.json"
            replacement.write_text("replacement\n", encoding="utf-8")
            replacement.replace(path)
            self.assertFalse(publication.unlink_if_owned(path, owner))
            self.assertEqual(path.read_text(encoding="utf-8"), "replacement\n")

    def test_publication_failure_preserves_replacement_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_path = root / "route.json"
            closure = root / "prepared.sha256"
            pending = root / "result.pending.json"
            output = root / "result.json"
            route_path.write_text(json.dumps(route(root)), encoding="utf-8")
            closure.write_text("closure\n", encoding="utf-8")
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

            with (
                patch.object(
                    publication, "validate_terminal_production_authority", terminal_authority
                ),
                patch.object(publication, "validate_report", replace_after_publication),
                self.assertRaisesRegex(RuntimeError, "late validation failure"),
            ):
                publication.publish_success(route_path, closure, pending, output, 29)
            self.assertFalse(pending.exists())
            self.assertEqual(output.read_text(encoding="utf-8"), "replacement\n")

    def test_pending_is_inode_owned_before_write_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_path = root / "route.json"
            closure = root / "prepared.sha256"
            pending = root / "result.pending.json"
            output = root / "result.json"
            route_path.write_text(json.dumps(route(root)), encoding="utf-8")
            closure.write_text("closure\n", encoding="utf-8")
            with (
                patch.object(
                    publication, "validate_terminal_production_authority", terminal_authority
                ),
                patch.object(
                    publication.os,
                    "fsync",
                    lambda _fd: (_ for _ in ()).throw(OSError("fsync")),
                ),
                self.assertRaisesRegex(OSError, "fsync"),
            ):
                publication.publish_success(route_path, closure, pending, output, 29)
            self.assertFalse(pending.exists())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
