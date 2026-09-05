import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bench import prefill_chunk_selection_io as publication


class PrefillChunkSelectionIoTest(unittest.TestCase):
    def test_post_chunk_publisher_uses_repo_module_invocation(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        script = (
            repo / "profiles/bench/post-chunk-twelve-candidate-20260905/"
            "finalize-selection-and-prepare.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m tools.bench.prefill_chunk_selection_io", script)
        result = subprocess.run(
            [sys.executable, "-m", "tools.bench.prefill_chunk_selection_io", "--help"],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _paths(self, root: Path) -> tuple[Path, Path]:
        pending = root / "selection.pending.json"
        published = root / "selection.json"
        pending.write_text("{}\n", encoding="utf-8")
        return pending, published

    def test_success_is_create_only_and_removes_owned_pending_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pending, published = self._paths(Path(directory))
            with mock.patch.object(publication, "validate_selection_record", return_value={}):
                publication.publish(pending, published)
            self.assertFalse(pending.exists())
            self.assertEqual(published.read_text(encoding="utf-8"), "{}\n")

    def test_existing_dangling_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pending, published = self._paths(Path(directory))
            published.symlink_to(Path(directory) / "missing")
            with self.assertRaisesRegex(ValueError, "namespace already exists"):
                publication.publish(pending, published)
            self.assertTrue(pending.exists())
            self.assertTrue(published.is_symlink())

    def test_failed_readback_removes_only_owned_published_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pending, published = self._paths(Path(directory))
            calls = 0

            def replace_then_fail(_path: Path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    published.unlink()
                    published.write_text("foreign\n", encoding="utf-8")
                    raise ValueError("readback failed")
                return {}

            with mock.patch.object(
                publication, "validate_selection_record", side_effect=replace_then_fail
            ), self.assertRaisesRegex(ValueError, "readback failed"):
                publication.publish(pending, published)
            self.assertTrue(pending.exists())
            self.assertEqual(published.read_text(encoding="utf-8"), "foreign\n")

    def test_success_does_not_remove_replaced_pending_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pending, published = self._paths(Path(directory))
            calls = 0

            def replace_pending_on_readback(_path: Path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    pending.unlink()
                    pending.write_text("foreign pending\n", encoding="utf-8")
                return {}

            with mock.patch.object(
                publication,
                "validate_selection_record",
                side_effect=replace_pending_on_readback,
            ):
                publication.publish(pending, published)
            self.assertEqual(pending.read_text(encoding="utf-8"), "foreign pending\n")
            self.assertEqual(published.read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()
