import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.r9700 import run_a8q4_n16k16_scalar_base_gate as gate


def raw(control=4.0, candidate=3.95, spread=0.0):
    cells = []
    for name, rows, columns, calls in gate.SHAPES:
        a = [128 * (control + (index - 3.5) * spread) for index in range(8)]
        b = [128 * (candidate + (index - 3.5) * spread) for index in range(8)]
        cells.append({
            "name": name, "tokens": 2048, "rows": rows, "columns": columns,
            "calls": calls, "iterations_per_sample": 128,
            "control_interval_samples_ms": a, "candidate_interval_samples_ms": b,
        })
    return {
        "schema": gate.MEASUREMENT_SCHEMA,
        "device": {"ordinal": 0, "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201"},
        "actual_product_wrappers": True,
        "warmup_order": ["A", "B", "B", "A"],
        "pair_order": ["AB", "BA"] * 4,
        "cells": cells,
    }


class ProductGateTest(unittest.TestCase):
    def test_five_ms_weighted_lower_admits(self):
        result = gate.decision(raw())
        self.assertEqual(result["classification"], "admitted")
        self.assertGreaterEqual(result["robust_weighted_saving_lower_ms"], 5.0)
        self.assertEqual(result["required_robust_weighted_saving_lower_ms"], 5.0)

    def test_stable_weighted_miss_rejects(self):
        result = gate.decision(raw(candidate=3.98))
        self.assertEqual(result["classification"], "rejected")
        self.assertFalse(result["admission_pass"])

    def test_every_cell_robust_upper_is_enforced(self):
        value = raw()
        value["cells"][0]["candidate_interval_samples_ms"] = [4.05 * 128] * 8
        result = gate.decision(value)
        self.assertEqual(result["classification"], "rejected")
        self.assertGreater(result["cells"][0]["robust_ratio_upper"], 1.01)

    def test_unstable_measurement_is_inconclusive(self):
        value = raw()
        value["cells"][1]["candidate_interval_samples_ms"] = [3.95 * 128] * 7 + [5.0 * 128]
        self.assertEqual(gate.decision(value)["classification"], "inconclusive")

    def test_v2_and_standalone_measurements_are_rejected(self):
        value = raw(); value["schema"] = "ninfer.r9700.a8q4-n16k16-scalar-base-measurement.v2"
        with self.assertRaisesRegex(RuntimeError, "product-wrapper v3"):
            gate.decision(value)
        value = raw(); value["actual_product_wrappers"] = False
        with self.assertRaisesRegex(RuntimeError, "product-wrapper v3"):
            gate.decision(value)

    def test_wrong_cell_or_pair_inventory_is_rejected(self):
        value = raw(); value["cells"].pop()
        with self.assertRaisesRegex(RuntimeError, "cell count"):
            gate.decision(value)
        value = raw(); value["pair_order"] = ["AB"] * 8
        with self.assertRaisesRegex(RuntimeError, "balanced order"):
            gate.decision(value)

    def test_parse_static_requires_loaded_object_hashes(self):
        base = {
            "logical_vgpr": 88, "architectural_vgpr": 88, "lds_bytes": 17152,
            "compiler_occupancy_waves": 16, "iu4": 8, "qualified_tuples": 32,
            "activation_code_max_voffset": 71303164, "activation_code_max_final_byte": 71303167,
            "weight_code_max_voffset": 89128952, "weight_code_max_final_byte": 89128959,
            "activation_scale_max_voffset": 4456446, "activation_scale_max_final_byte": 4456447,
            "weight_scale_max_voffset": 5570558, "weight_scale_max_final_byte": 5570559,
            "source_sha256": "1" * 64, "profile_sha256": "2" * 64,
            "assembly_sha256": "3" * 64, "product_executable_sha256": "4" * 64,
            "loaded_code_object_sha256": "5" * 64, "loaded_disassembly_sha256": "6" * 64,
            "loaded_notes_sha256": "7" * 64, "scratch": 0, "spills": 0, "scalar_bases": 5,
        }
        stdout = "PASS " + " ".join(f"{key}={value}" for key, value in base.items()) + "\n"
        hashes = {gate.SOURCE: "1" * 64, gate.PROFILE: "2" * 64,
                  gate.ASSEMBLY: "3" * 64, gate.PRODUCT_QUALIFIER: "4" * 64}
        with mock.patch.object(gate, "sha256", side_effect=lambda path: hashes[path]):
            self.assertEqual(gate.parse_static(stdout)["loaded_code_object_sha256"], "5" * 64)
            malformed = stdout.replace("loaded_notes_sha256=" + "7" * 64 + " ", "")
            with self.assertRaisesRegex(RuntimeError, "loaded_notes_sha256"):
                gate.parse_static(malformed)

    def test_publication_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("foreign\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fresh"):
                gate.publish(path, {}, {})
            self.assertEqual(path.read_text(encoding="utf-8"), "foreign\n")

    def test_main_runs_bound_shape_and_product_regressions_before_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            report_path.parent.mkdir(exist_ok=True)
            calls = []
            static_stdout = "PASS placeholder\n"
            static_fields = {
                "loaded_code_object_sha256": "5" * 64,
                "loaded_disassembly_sha256": "6" * 64,
                "loaded_notes_sha256": "7" * 64,
                "product_executable_sha256": "4" * 64,
            }
            records = []

            def checked(command, pattern):
                calls.append(list(command))
                value = {"command": list(command), "returncode": 0, "stdout": "PASS\n", "stderr": ""}
                records.append(value)
                return value

            run_results = [
                subprocess.CompletedProcess([], 0, static_stdout, ""),
                subprocess.CompletedProcess([], 0, json.dumps(raw()), ""),
            ]
            power = {"device": {"ordinal": 0}, "sysfs_value": "auto"}
            captured = {}

            def publish(path, value, hashes):
                captured.update(value)

            previous = Path.cwd(); os.chdir(root)
            try:
                with mock.patch.object(gate, "ROOT", root), mock.patch.object(gate, "BUILD", root / "build"), \
                     mock.patch.object(gate, "REPORT", report_path), \
                     mock.patch.object(gate, "snapshot_inputs", return_value={"input": "hash"}), \
                     mock.patch.object(gate, "validate_build_contract"), \
                     mock.patch.object(gate, "run", side_effect=run_results), \
                     mock.patch.object(gate, "parse_static", return_value=static_fields), \
                     mock.patch.object(gate, "checked", side_effect=checked), \
                     mock.patch.object(gate, "power", return_value=power), \
                     mock.patch.object(gate, "publish", side_effect=publish):
                    self.assertEqual(gate.main(["--benchmark", "--output", str(report_path)]), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(calls, [
                [str(gate.BOUNDS_TEST)],
                [str(gate.SHAPE_QUALIFIER), "--prefill-cta-scalar-base-regression"],
                [str(gate.PRODUCT_QUALIFIER), "--regression"],
            ])
            self.assertEqual(captured["schema"], gate.SCHEMA)
            self.assertEqual(captured["measurement"]["schema"], gate.MEASUREMENT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
