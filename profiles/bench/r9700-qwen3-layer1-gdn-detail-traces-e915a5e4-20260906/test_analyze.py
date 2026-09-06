#!/usr/bin/env python3
"""CPU contract tests for the e915a5e4 layer-1 GDN trace analyzer."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("gdn_detail_analyze", PATH)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


class AnalyzerContractTest(unittest.TestCase):
    def test_exact_four_commands(self):
        for stem in analyze.ROLES:
            command = analyze.command(stem, Path("/tmp/unused.json"))
            self.assertEqual(command[command.index("--concurrency") + 1], "1")
            self.assertIn("--no-device-graph", command)
            self.assertIn("--retain-token-ids", command)
            self.assertEqual(command[-4:], ["-r", "1", "--warmup", "0"])
        self.assertIn("--spec", analyze.command("target-dflash", Path("/tmp/unused")))
        self.assertIn("--isolate-prompt-decode",
                      analyze.command("text-append", Path("/tmp/unused")))

    def test_comparator_is_exact_gdn_mode(self):
        output = analyze.RESULTS / "target-gdn-comparison.json"
        command = analyze.comparator_command("target", "target-ordinary", "target-dflash", output)
        self.assertIn("--gdn-detail", command)
        self.assertEqual(command[command.index("--left") + 1],
                         str(analyze.RESULTS / "target-ordinary.gdn.json"))
        self.assertEqual(command[command.index("--right") + 1],
                         str(analyze.RESULTS / "target-dflash.gdn.json"))

    def test_speculative_accounting_accepts_exact_forms(self):
        ordinary = {"enabled":False,"draft_window":0,"rounds":0,"drafted_tokens":0,
                    "accepted_tokens":0,"fallback_steps":0,"acceptance_rate":None,
                    "acceptance_length":None,"accepted_per_position":[]}
        dflash = {"enabled":True,"draft_window":4,"rounds":0,"drafted_tokens":0,
                  "accepted_tokens":0,"fallback_steps":1,"acceptance_rate":None,
                  "acceptance_length":None,"accepted_per_position":[0,0,0,0]}
        analyze.validate_speculative(ordinary, False, "ordinary")
        analyze.validate_speculative(dflash, True, "dflash")

    def test_speculative_accounting_rejects_mutations(self):
        value = {"enabled":True,"draft_window":4,"rounds":0,"drafted_tokens":0,
                 "accepted_tokens":0,"fallback_steps":2,"acceptance_rate":None,
                 "acceptance_length":None,"accepted_per_position":[0,0,0,0]}
        with self.assertRaisesRegex(RuntimeError, "accounting"):
            analyze.validate_speculative(value, True, "dflash")
        value["fallback_steps"] = 1
        value["accepted_per_position"] = [0,0,0]
        with self.assertRaisesRegex(RuntimeError, "zero-draft"):
            analyze.validate_speculative(value, True, "dflash")

    def test_positive_speculative_accounting_is_recomputed(self):
        value = {"enabled":True,"draft_window":4,"rounds":1,"drafted_tokens":4,
                 "accepted_tokens":0,"fallback_steps":0,"acceptance_rate":0.0,
                 "acceptance_length":1.0,"accepted_per_position":[0,0,0,0]}
        analyze.validate_speculative(value, True, "dflash")
        for key, replacement in (("accepted_per_position", [1,0,0,0]),
                                 ("acceptance_rate", 0.25), ("acceptance_length", 2.0)):
            with self.subTest(key=key):
                mutated = dict(value)
                mutated[key] = replacement
                with self.assertRaisesRegex(RuntimeError, "positive speculative"):
                    analyze.validate_speculative(mutated, True, "dflash")

    def test_unknown_scope_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unknown arm"):
            analyze.command("other", Path("/tmp/unused"))
        with self.assertRaisesRegex(RuntimeError, "unknown comparison"):
            analyze.comparator_command("other", "a", "b", Path("/tmp/unused"))

    def test_report_contract_accepts_bound_shape_and_rejects_extra_config(self):
        predecessor = Path(
            "/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/"
            "r9700-qwen3-layer-boundary-traces-43e5e4cc-20260906/results/target-ordinary.json")
        with tempfile.TemporaryDirectory() as raw:
            prior_results = analyze.RESULTS
            try:
                analyze.RESULTS = Path(raw)
                values = {}
                for stem in analyze.ROLES:
                    value = json.loads((predecessor.parent / f"{stem}.json").read_text())
                    value["config"]["dflash_mlp_down_t5_candidate"] = True
                    value["config"]["dflash_rmsnorm_rows56_candidate"] = True
                    report = analyze.RESULTS / f"{stem}.json"
                    value["command"] = " ".join(analyze.command(stem, report))
                    report.write_text(json.dumps(value))
                    _, tokens = analyze.validate_report(stem)
                    self.assertEqual(tokens, [96558, 96917])
                    values[stem] = value
                value = values["target-ordinary"]
                report = analyze.RESULTS / "target-ordinary.json"
                value["config"]["unexpected"] = 1
                report.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError, "configuration"):
                    analyze.validate_report("target-ordinary")
            finally:
                analyze.RESULTS = prior_results


if __name__ == "__main__":
    unittest.main()
