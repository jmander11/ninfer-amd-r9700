from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench import publish_low_context_prefill as publication

REPO = Path(__file__).resolve().parents[2]


def _arguments(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = tuple(
        root / name
        for name in (
            "pending.json",
            "output.json",
            "manifest.json",
            "bench",
            "artifact",
            "selection.json",
        )
    )
    for path in paths[2:]:
        path.write_text("source\n", encoding="utf-8")
    paths[0].write_text(json.dumps({"passes_p2048_gate": True}), encoding="utf-8")
    return paths


class PublishLowContextPrefillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_publishes_exact_inode_and_removes_pending(self) -> None:
        pending, output, manifest, executable, artifact, selection = _arguments(self.tmp_path)
        expected = {"passes_p2048_gate": True}
        with patch.object(publication, "validate_ladder", lambda *_args: expected):
            publication.publish(pending, output, manifest, executable, artifact, selection, 2000.0)
        self.assertEqual(json.loads(output.read_text()), expected)
        self.assertFalse(pending.exists())

    def test_preserves_occupied_or_dangling_output(self) -> None:
        pending, output, manifest, executable, artifact, selection = _arguments(self.tmp_path)
        output.symlink_to(self.tmp_path / "missing")
        with (
            patch.object(
                publication, "validate_ladder", lambda *_args: {"passes_p2048_gate": True}
            ),
            self.assertRaisesRegex(ValueError, "occupied"),
        ):
            publication.publish(pending, output, manifest, executable, artifact, selection, 2000.0)
        self.assertTrue(output.is_symlink())
        self.assertTrue(pending.exists())

    def test_post_publish_validation_failure_rolls_back_owned_inode(self) -> None:
        pending, output, manifest, executable, artifact, selection = _arguments(self.tmp_path)
        calls = 0

        def validate(*_args):
            nonlocal calls
            calls += 1
            return {"passes_p2048_gate": True} if calls == 1 else {"passes_p2048_gate": False}

        with (
            patch.object(publication, "validate_ladder", validate),
            self.assertRaisesRegex(ValueError, "published"),
        ):
            publication.publish(pending, output, manifest, executable, artifact, selection, 2000.0)
        self.assertFalse(output.exists())
        self.assertFalse(pending.exists())

    def test_package_uses_importable_module_publisher(self) -> None:
        command = subprocess.run(
            [sys.executable, "-m", "tools.bench.publish_low_context_prefill", "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(command.returncode, 0, command.stderr)
        script = (
            REPO / "profiles/bench/low-context-selected-ladder-20260905/run-and-publish.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m tools.bench.publish_low_context_prefill", script)
        self.assertNotIn('python3 "$publisher"', script)


if __name__ == "__main__":
    unittest.main()
