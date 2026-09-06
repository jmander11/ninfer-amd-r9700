#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight  # noqa: E402


class PreflightTest(unittest.TestCase):
    def test_accepts_fresh_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            preflight.require_fresh_outputs(Path(directory))

    def test_rejects_regular_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / preflight.OUTPUTS[0]).write_text("occupied")
            with self.assertRaisesRegex(ValueError, "already exists"):
                preflight.require_fresh_outputs(root)

    def test_rejects_dangling_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / preflight.OUTPUTS[-1]).symlink_to(root / "missing-target")
            self.assertFalse((root / preflight.OUTPUTS[-1]).exists())
            self.assertTrue((root / preflight.OUTPUTS[-1]).is_symlink())
            with self.assertRaisesRegex(ValueError, "already exists"):
                preflight.require_fresh_outputs(root)


if __name__ == "__main__":
    unittest.main()
