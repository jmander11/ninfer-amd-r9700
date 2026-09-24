import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.ppl import terminal_selection_io as publication


class TerminalSelectionIoTest(unittest.TestCase):
    def test_dangling_symlink_is_not_an_absent_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "authority.json"
            output.symlink_to(Path(directory) / "missing.json")
            with self.assertRaisesRegex(ValueError, "namespace already exists"):
                publication.require_absent([output])

    def test_publish_is_exclusive_and_removes_pending_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_input = root / "input.pending.json"
            pending_result = root / "result.pending.json"
            published_input = root / "input.json"
            published_result = root / "result.json"
            pending_input.write_text("{}\n", encoding="utf-8")
            pending_result.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(publication, "validate_terminal_production_authority"):
                publication.publish(
                    pending_input, published_input, pending_result, published_result
                )
            self.assertEqual(published_input.read_text(encoding="utf-8"), "{}\n")
            self.assertEqual(published_result.read_text(encoding="utf-8"), "{}\n")
            self.assertFalse(pending_input.exists())
            self.assertFalse(pending_result.exists())

    def test_validation_failure_rolls_back_only_published_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_input = root / "input.pending.json"
            pending_result = root / "result.pending.json"
            published_input = root / "input.json"
            published_result = root / "result.json"
            pending_input.write_text("{}\n", encoding="utf-8")
            pending_result.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                publication,
                "validate_terminal_production_authority",
                side_effect=ValueError("invalid authority"),
            ):
                with self.assertRaisesRegex(ValueError, "invalid authority"):
                    publication.publish(
                        pending_input, published_input, pending_result, published_result
                    )
            self.assertFalse(published_input.exists())
            self.assertFalse(published_result.exists())
            self.assertTrue(pending_input.exists())
            self.assertTrue(pending_result.exists())

    def test_existing_result_is_preserved_and_nothing_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_input = root / "input.pending.json"
            pending_result = root / "result.pending.json"
            published_input = root / "input.json"
            published_result = root / "result.json"
            pending_input.write_text("{}\n", encoding="utf-8")
            pending_result.write_text("{}\n", encoding="utf-8")
            published_result.write_text("owned elsewhere\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "namespace already exists"):
                publication.publish(
                    pending_input, published_input, pending_result, published_result
                )
            self.assertFalse(published_input.exists())
            self.assertEqual(
                published_result.read_text(encoding="utf-8"), "owned elsewhere\n"
            )

    def test_guard_change_rejects_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_input = root / "input.pending.json"
            pending_result = root / "result.pending.json"
            published_input = root / "input.json"
            published_result = root / "result.json"
            guard = root / "chunk.json"
            pending_input.write_text("{}\n", encoding="utf-8")
            pending_result.write_text("{}\n", encoding="utf-8")
            guard.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "guard changed"):
                publication.publish(
                    pending_input, published_input, pending_result, published_result,
                    guard_path=guard, guard_sha256="0" * 64,
                )
            self.assertFalse(published_input.exists())
            self.assertFalse(published_result.exists())

    def test_late_failure_does_not_remove_replacement_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_input = root / "input.pending.json"
            pending_result = root / "result.pending.json"
            published_input = root / "input.json"
            published_result = root / "result.json"
            pending_input.write_text("{}\n", encoding="utf-8")
            pending_result.write_text("{}\n", encoding="utf-8")
            calls = 0

            def replace_on_readback(_value):
                nonlocal calls
                calls += 1
                if calls == 2:
                    published_input.unlink()
                    published_input.write_text("foreign\n", encoding="utf-8")
                    raise ValueError("late validation failed")

            with mock.patch.object(
                publication, "validate_terminal_production_authority",
                side_effect=replace_on_readback,
            ), self.assertRaisesRegex(ValueError, "late validation failed"):
                publication.publish(
                    pending_input, published_input, pending_result, published_result
                )
            self.assertEqual(published_input.read_text(encoding="utf-8"), "foreign\n")
            self.assertFalse(published_result.exists())

    def test_success_does_not_remove_replaced_pending_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_input = root / "input.pending.json"
            pending_result = root / "result.pending.json"
            published_input = root / "input.json"
            published_result = root / "result.json"
            pending_input.write_text("{}\n", encoding="utf-8")
            pending_result.write_text("{}\n", encoding="utf-8")
            calls = 0

            def replace_pending_on_readback(_value):
                nonlocal calls
                calls += 1
                if calls == 2:
                    pending_input.unlink()
                    pending_input.write_text("foreign pending\n", encoding="utf-8")

            with mock.patch.object(
                publication, "validate_terminal_production_authority",
                side_effect=replace_pending_on_readback,
            ):
                publication.publish(
                    pending_input, published_input, pending_result, published_result
                )
            self.assertEqual(pending_input.read_text(encoding="utf-8"), "foreign pending\n")
            self.assertFalse(pending_result.exists())
            self.assertEqual(published_input.read_text(encoding="utf-8"), "{}\n")
            self.assertEqual(published_result.read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()
