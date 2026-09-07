"""CPU contract tests for the 6fe53d53 Text layer-boundary evidence analyzer."""

import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("text_layer_boundary_analyze", PATH)
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class AnalyzeTest(unittest.TestCase):
    def test_exact_commands(self):
        fresh = ANALYZE.command("text-fresh", ANALYZE.RESULTS / "text-fresh.json")
        self.assertIn("--whole-pg", fresh)
        self.assertEqual(fresh[fresh.index("--whole-pg") + 1], "129,1")
        self.assertEqual(fresh[fresh.index("--corpus") + 1], str(ANALYZE.HISTORY))
        self.assertNotIn("--isolate-prompt-decode", fresh)
        self.assertIn("--output-file", fresh)
        append = ANALYZE.command("text-append", ANALYZE.RESULTS / "text-append.json")
        self.assertIn("-pg", append)
        self.assertEqual(append[append.index("-pg") + 1], "128,1")
        self.assertEqual(append[append.index("--corpus") + 1], str(ANALYZE.CORPUS))
        self.assertIn("--isolate-prompt-decode", append)

    def test_trace_environment_is_role_exact(self):
        self.assertEqual(
            ANALYZE.trace_environment("text-fresh"),
            {"NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": "text-fresh-frontier129-column128",
             "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST":
                 str(ANALYZE.RESULTS / "text-fresh.trace.json"),
             "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR":
                 str(ANALYZE.RESULTS / "text-fresh.trace.bin")})
        self.assertEqual(
            ANALYZE.trace_environment("text-append"),
            {"NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": "text-append-frontier129-column0",
             "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST":
                 str(ANALYZE.RESULTS / "text-append.trace.json"),
             "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR":
                 str(ANALYZE.RESULTS / "text-append.trace.bin")})

    def test_retained_tokens_are_exact(self):
        self.assertEqual(ANALYZE.RETAINED_TOKENS, [96558, 96917])
        for token in ANALYZE.RETAINED_TOKENS:
            self.assertTrue(0 <= token < ANALYZE.TOKEN_DOMAIN)

    def test_config_key_set_matches_combined_build(self):
        self.assertEqual(len(ANALYZE.CONFIG_KEYS), 35)
        for key in ("dflash_mlp_down_t5_candidate", "dflash_rmsnorm_rows56_candidate",
                    "attention_parity_candidate"):
            self.assertIn(key, ANALYZE.CONFIG_KEYS)

    def test_device_staging_bytes(self):
        self.assertEqual(ANALYZE.DEVICE_STAGING_BYTES, 1320972)


if __name__ == "__main__":
    unittest.main()
