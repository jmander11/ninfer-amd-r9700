#!/usr/bin/env python3
"""CPU-only contract tests for the DFlash K4/W5 after-parity screen analyzer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import analyze  # noqa: E402


class TestK4W5ScreenAnalyzer(unittest.TestCase):
    def test_command_base_decode(self) -> None:
        cmd = analyze.command("base-decode", PACKAGE / "results/base-decode.json")
        self.assertIn("--spec", cmd)
        self.assertIn("mtp", cmd)
        self.assertIn("--draft-tokens", cmd)
        self.assertIn("--no-device-graph", cmd)
        self.assertIn("--whole-pg", cmd)
        self.assertIn("129,27", cmd)
        self.assertIn("-r", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--warmup", cmd)
        self.assertIn("1", cmd)

    def test_command_dflash_k4w5(self) -> None:
        cmd = analyze.command("dflash-k4w5", PACKAGE / "results/dflash-k4w5.json")
        self.assertIn("--spec", cmd)
        self.assertIn("dflash", cmd)
        self.assertIn("--draft-tokens", cmd)
        self.assertIn("4", cmd)
        self.assertIn("--dflash-verify-width", cmd)
        self.assertIn("5", cmd)
        self.assertIn("--no-device-graph", cmd)
        self.assertIn("--whole-pg", cmd)
        self.assertIn("129,27", cmd)

    def test_commands_differ_in_spec(self) -> None:
        base = analyze.command("base-decode", PACKAGE / "results/base-decode.json")
        dflash = analyze.command("dflash-k4w5", PACKAGE / "results/dflash-k4w5.json")
        self.assertIn("mtp", base)
        self.assertIn("dflash", dflash)
        self.assertIn("5", dflash)  # verify width

    def test_token_domain(self) -> None:
        self.assertEqual(analyze.TOKEN_DOMAIN, 248077)

    def test_base_decode_reference(self) -> None:
        self.assertEqual(analyze.BASE_DECODE_TOK_S, 27.05729956)
        self.assertEqual(analyze.BREAK_EVEN_MS_PER_ROUND, 57.6915)


if __name__ == "__main__":
    unittest.main(verbosity=2)
