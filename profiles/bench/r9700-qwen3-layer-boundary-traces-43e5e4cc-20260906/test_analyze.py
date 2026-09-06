"""CPU contract tests for the 43e5e4cc layer-boundary evidence analyzer."""

import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("layer_boundary_analyze", PATH)
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class AnalyzeTest(unittest.TestCase):
    def test_exact_retained_output_authority(self):
        for stem in ("target-ordinary", "target-dflash", "text-fresh", "text-append"):
            self.assertEqual(ANALYZE.validate_output_tokens(stem, [96558, 96917]),
                             [96558, 96917])
            with self.assertRaisesRegex(RuntimeError, "output sequence"):
                ANALYZE.validate_output_tokens(stem, [24178, 96917])

    def test_rejects_bool_and_domain(self):
        with self.assertRaisesRegex(RuntimeError, "domain"):
            ANALYZE.validate_output_tokens("target-ordinary", [True, 96917])
        with self.assertRaisesRegex(RuntimeError, "domain"):
            ANALYZE.validate_output_tokens("target-ordinary", [96558, 248077])

    def test_exact_commands(self):
        path = ANALYZE.RESULTS / "target-dflash.json"
        command = ANALYZE.command("target-dflash", path)
        self.assertIn("--whole-pg", command)
        self.assertIn("--spec", command)
        self.assertEqual(command[command.index("--dflash-verify-width") + 1], "5")
        append = ANALYZE.command("text-append", ANALYZE.RESULTS / "text-append.json")
        self.assertIn("--isolate-prompt-decode", append)
        self.assertIn("128,1", append)

    def test_decode_path_is_role_exact(self):
        ANALYZE.validate_decode_path({"decode_path": "eager"}, False, "target-ordinary")
        ANALYZE.validate_decode_path({"decode_path": "dflash_eager"}, True, "target-dflash")
        with self.assertRaisesRegex(RuntimeError, "decode path"):
            ANALYZE.validate_decode_path({"decode_path": "eager"}, True, "target-dflash")

    def test_trace_input_token_is_role_exact(self):
        for stem in ("target-ordinary", "target-dflash"):
            self.assertEqual(ANALYZE.validate_trace_input_token(stem, 96558), 96558)
            with self.assertRaisesRegex(RuntimeError, "selected input token"):
                ANALYZE.validate_trace_input_token(stem, 24178)
        for stem in ("text-fresh", "text-append"):
            self.assertEqual(ANALYZE.validate_trace_input_token(stem, 24178), 24178)
            with self.assertRaisesRegex(RuntimeError, "selected input token"):
                ANALYZE.validate_trace_input_token(stem, 96558)


if __name__ == "__main__":
    unittest.main()
