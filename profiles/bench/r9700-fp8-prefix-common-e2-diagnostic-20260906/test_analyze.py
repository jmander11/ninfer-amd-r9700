#!/usr/bin/env python3
import copy
import importlib.util
import unittest
from pathlib import Path
from unittest import mock


PATH = Path(__file__).resolve().parent / "analyze.py"
SPEC = importlib.util.spec_from_file_location("fp8_e2_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzeTest(unittest.TestCase):
    def test_retained_results(self) -> None:
        summary = MODULE.analyze()
        self.assertEqual(summary["decision"], "e2_operator_qualified_whole_text_pending")
        self.assertEqual(summary["prefix_mismatch_count"], 0)
        self.assertFalse(summary["timing_evidence_eligible"])

    def test_timing_mutation_rejected(self) -> None:
        original = MODULE.load_json
        def changed(path: Path):
            value = original(path)
            if path.name == "e2-selector1.json":
                value = copy.deepcopy(value)
                value["cells"][0]["common_algorithm_trials"][0]["balanced_timings"][0]["selected_median_ms"] += 1
            return value
        with mock.patch.object(MODULE, "load_json", side_effect=changed):
            with self.assertRaises(ValueError):
                MODULE.analyze()

    def test_correctness_mutation_rejected(self) -> None:
        original = MODULE.load_json
        def changed(path: Path):
            value = original(path)
            if path.name == "catalog-selector1-preliminary.json":
                value = copy.deepcopy(value)
                value["cells"][0]["common_algorithm_trials"][0]["eligible"] = False
            return value
        with mock.patch.object(MODULE, "load_json", side_effect=changed):
            with self.assertRaises(ValueError):
                MODULE.analyze()

    def test_plan_semantic_mutation_rejected(self) -> None:
        original = MODULE.load_json
        def changed(path: Path):
            value = original(path)
            if path.name == "plan.json":
                value = copy.deepcopy(value)
                value["provenance_limitations"] = value["provenance_limitations"][:-1]
            return value
        with mock.patch.object(MODULE, "load_json", side_effect=changed):
            with self.assertRaises(ValueError):
                MODULE.analyze()


if __name__ == "__main__":
    unittest.main()
