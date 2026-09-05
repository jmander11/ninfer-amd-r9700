#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ppl.quality_recovery_io import publish, require_preflight


class QualityRecoveryIoTest(unittest.TestCase):
    def test_preflight_rejects_dangling_output_and_validates_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"; checkpoint.mkdir()
            output = root / "output"; output.symlink_to(root / "missing")
            with patch("tools.ppl.quality_recovery_io.validate_checkpoint_files") as validate:
                with self.assertRaisesRegex(ValueError, "occupied"):
                    require_preflight(checkpoint, [output])
            validate.assert_called_once_with(checkpoint.resolve())

    def test_publish_is_exclusive_and_removes_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = root / "pending.json"; pending.write_bytes(b"authority")
            final = root / "final.json"
            publish(pending, final)
            self.assertEqual(final.read_bytes(), b"authority")
            self.assertFalse(pending.exists())
            new_pending = root / "pending.json"; new_pending.write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "already exists"):
                publish(new_pending, final)
            self.assertEqual(final.read_bytes(), b"authority")

    def test_publish_rejects_symlink_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"; source.write_bytes(b"authority")
            pending = root / "pending"; pending.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                publish(pending, root / "final")


if __name__ == "__main__":
    unittest.main()
