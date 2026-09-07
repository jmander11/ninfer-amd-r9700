#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("recurrent_state_analyze", PACKAGE / "analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)
PRIOR = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/r9700-qwen3-layer1-gdn-detail-traces-e915a5e4-20260906/results")


class AnalyzeTest(unittest.TestCase):
    def report_fixture(self, directory: Path, stem: str) -> dict:
        value = json.loads((PRIOR / f"{stem}.json").read_text())
        value["command"] = " ".join(analyze.command(stem))
        for key in ("dflash_mlp_down_t5_candidate", "dflash_rmsnorm_rows56_candidate"):
            value["config"][key] = False
        (directory / f"{stem}.json").write_text(json.dumps(value))
        return value

    def with_results(self, directory: Path):
        class Restore:
            def __enter__(inner):
                inner.old = analyze.RESULTS
                analyze.RESULTS = directory

            def __exit__(inner, *_):
                analyze.RESULTS = inner.old
        return Restore()

    def test_validates_both_exact_report_roles(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.report_fixture(directory, "text-fresh")
            self.report_fixture(directory, "text-append")
            with self.with_results(directory):
                self.assertEqual(analyze.validate_report("text-fresh")[1], [96558, 96917])
                self.assertEqual(analyze.validate_report("text-append")[1], [96558, 96917])

    def test_rejects_selector_token_and_speculative_mutations(self):
        mutations = (
            lambda value: value["config"].__setitem__("dflash_rmsnorm_rows56_candidate", True),
            lambda value: value["tests"][0]["reps"][0]["generated_token_ids_by_lane"][0].__setitem__(1, 0),
            lambda value: value["tests"][0]["speculative"].__setitem__("rounds", 1),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                value = self.report_fixture(directory, "text-fresh")
                mutate(value)
                (directory / "text-fresh.json").write_text(json.dumps(value))
                with self.with_results(directory), self.assertRaises(RuntimeError):
                    analyze.validate_report("text-fresh")

    def test_expected_state_classification_and_mutation(self):
        value = {
            "classification": "first_difference_layer1_recurrent_prefix_state",
            "first_difference": {"first_element_index": 599, "mismatch_count": 402925,
                "left_bits": 950422399, "right_bits": 950422400,
                "maximum_absolute_difference": 0.000046528875827789307},
        }
        analyze.validate_expected_state_comparison(value)
        for key, replacement in (("first_element_index", 600), ("mismatch_count", 1),
                                 ("maximum_absolute_difference", 0.0)):
            mutated = copy.deepcopy(value)
            mutated["first_difference"][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "classification"):
                analyze.validate_expected_state_comparison(mutated)

    def test_expected_gdn_classification_and_mutation(self):
        value = {"classification": "first_difference_gdn_recurrence",
                 "first_difference": {"field": "o", "first_element_index": 154,
                     "mismatch_count": 106, "left_bits": 14405, "right_bits": 14406,
                     "maximum_absolute_difference": 0.000003814697265625}}
        analyze.validate_expected_gdn_comparison(value)
        value["first_difference"]["field"] = "v"
        with self.assertRaisesRegex(RuntimeError, "GDN output"):
            analyze.validate_expected_gdn_comparison(value)


if __name__ == "__main__":
    unittest.main()
