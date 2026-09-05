"""Dependency-light tests for the standalone GDN determinism probe contract."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from . import gdn_determinism_probe as probe


class GdnDeterminismProbeTest(unittest.TestCase):
    def test_defaults_and_explicit_routes(self) -> None:
        defaults = probe.parse_options(["--out-json", "report.json"])
        self.assertEqual(defaults.rows, 4096)
        self.assertEqual(defaults.repeats, 3)
        self.assertEqual(defaults.routes, ("chunk", "fused-recurrent"))
        self.assertFalse(defaults.deterministic_algorithms)
        selected = probe.parse_options([
            "--rows", "128", "--repeats", "4", "--device", "2",
            "--routes", "chunk,fused-recurrent", "--deterministic-algorithms",
            "--out-json", "-",
        ])
        self.assertEqual(
            selected,
            probe.Options(128, 4, 2, ("chunk", "fused-recurrent"), True, "-"),
        )

    def test_rejects_invalid_geometry_and_routes(self) -> None:
        for argv in (
            ["--rows", "0", "--out-json", "x"],
            ["--repeats", "1", "--out-json", "x"],
            ["--device", "-1", "--out-json", "x"],
            ["--rows", "257", "--routes", "chunk,project-naive", "--out-json", "x"],
        ):
            with self.assertRaises(ValueError):
                probe.parse_options(argv)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                probe.parse_options(["--routes", "chunk,chunk", "--out-json", "x"])
            with self.assertRaises(SystemExit):
                probe.parse_options(["--routes", "unknown", "--out-json", "x"])

    def test_atomic_report_serialization(self) -> None:
        payload = {
            "artifact_type": probe.SCHEMA,
            "schema_version": probe.SCHEMA_VERSION,
            "rows": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "report.json"
            probe.write_report(str(path), payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(path.parent.glob("*.tmp-*")), [])
        with self.assertRaises(ValueError):
            probe.write_report("ignored.json", {"artifact_type": "wrong", "schema_version": 1})

    def test_python_tree_identity_ignores_non_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            first = probe.python_tree_sha256(root)
            (root / "nested" / "module.pyc").write_bytes(b"mutable cache")
            self.assertEqual(probe.python_tree_sha256(root), first)
            (root / "nested" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(probe.python_tree_sha256(root), first)

    def test_distribution_record_is_dependency_light_and_bound(self) -> None:
        record = probe.distribution_record("this-package-does-not-exist-ninfer")
        self.assertEqual(
            record,
            {"version": None, "record_sha256": None, "direct_url_sha256": None},
        )
        installed = probe.distribution_record("pip")
        self.assertIsInstance(installed["version"], str)
        if installed["record_sha256"] is not None:
            self.assertRegex(installed["record_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
