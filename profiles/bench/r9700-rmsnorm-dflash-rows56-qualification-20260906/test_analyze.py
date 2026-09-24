import copy
import importlib.util
import math
from pathlib import Path
import unittest

PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("rows56_analyze", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(ratio5=0.8, ratio6=1.1):
    cells = []
    for rows, ratio in ((5, ratio5), (6, ratio6)):
        incumbent = [0.1 + i * 0.0001 for i in range(7)]
        challenger = [x * ratio for x in incumbent]
        cells.append({"features": 5120, "rows": rows, "unit_offset": True,
                      "epsilon": 1e-6, "iterations_per_sample": 256,
                      "forward_reverse_pairs": 7,
                      "incumbent_forward_ms": incumbent,
                      "challenger_forward_ms": challenger,
                      "challenger_reverse_ms": challenger,
                      "incumbent_reverse_ms": incumbent})
    return {"schema": "ninfer.r9700.rmsnorm-k5120-rows56-cell.v1",
            "production_dispatch_changed": False,
            "numeric": {"oracle": "independent FP64 represented-BF16 RMSNorm formula",
                        "maximum_bf16_steps_allowed": 2,
                        "incumbent_maximum_bf16_steps": 2,
                        "challenger_maximum_bf16_steps": 2}, "cells": cells}


class AnalyzeTest(unittest.TestCase):
    def test_exact_cell_selection(self):
        result = MODULE.analyze(report())
        self.assertEqual(result["eligible_rows"], [5])
        self.assertEqual(result["forbidden_rows"], [6])

    def test_rejects_scope_mutation(self):
        value = report(); value["cells"][0]["rows"] = 4
        with self.assertRaises(ValueError): MODULE.analyze(value)

    def test_rejects_nonfinite_raw_sample(self):
        value = report(); value["cells"][0]["challenger_forward_ms"][0] = math.inf
        with self.assertRaises(ValueError): MODULE.analyze(value)

    def test_rejects_missing_sample(self):
        value = report(); value["cells"][0]["incumbent_reverse_ms"].pop()
        with self.assertRaises(ValueError): MODULE.analyze(value)

    def test_rejects_oracle_failure(self):
        value = report(); value["numeric"]["challenger_maximum_bf16_steps"] = 3
        with self.assertRaises(ValueError): MODULE.analyze(value)

    def test_order_instability_forbids_cell(self):
        value = report(0.8, 0.8)
        value["cells"][0]["challenger_reverse_ms"] = [x * 0.85 for x in value["cells"][0]["incumbent_reverse_ms"]]
        result = MODULE.analyze(value)
        self.assertNotIn(5, result["eligible_rows"])


if __name__ == "__main__": unittest.main()
