#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cu_mask_validate", HERE / "validate.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metrics(q4_half: float, q4_quarter: float,
            stream_half: float = 0.95, stream_quarter: float = 0.85):
    return {
        "full64": {"iu4": 400.0, "stream": 600.0, "q4_ms": 100.0},
        "half32": {"iu4": 200.0, "stream": 600.0 * stream_half,
                   "q4_ms": 100.0 / q4_half},
        "quarter16": {"iu4": 100.0, "stream": 600.0 * stream_quarter,
                      "q4_ms": 100.0 / q4_quarter},
    }


class TestCuMaskValidator(unittest.TestCase):
    def test_live_plan_contract(self):
        plan = json.loads((HERE / "plan.json").read_text())
        MODULE.validate_plan(plan, verify_inputs=False)

    def test_classifies_cu_local_curve(self):
        result = MODULE.classify(metrics(0.51, 0.26))
        self.assertEqual(result["classification"], "cu_local_service")

    def test_classifies_shared_stream_curve(self):
        result = MODULE.classify(metrics(0.96, 0.84))
        self.assertEqual(result["classification"], "shared_memory_service")

    def test_both_or_neither_is_inconclusive(self):
        both = metrics(0.5, 0.25, 0.5, 0.25)
        self.assertEqual(MODULE.classify(both)["classification"],
                         "mixed_or_inconclusive")
        self.assertEqual(MODULE.classify(metrics(0.70, 0.50))["classification"],
                         "mixed_or_inconclusive")

    def test_rejects_missing_arm(self):
        value = metrics(0.5, 0.25)
        del value["quarter16"]
        with self.assertRaisesRegex(ValueError, "all three arms"):
            MODULE.classify(value)

    def test_rejects_ignored_or_wrong_mask(self):
        value = metrics(0.5, 0.25)
        value["half32"]["iu4"] = 390.0
        with self.assertRaisesRegex(ValueError, "does not validate"):
            MODULE.classify(value)

    def test_ten_percent_boundary_is_inclusive(self):
        value = metrics(0.55, 0.275)
        result = MODULE.classify(value)
        self.assertEqual(result["classification"], "cu_local_service")

    def test_peak_control_contract(self):
        plan = json.loads((HERE / "plan.json").read_text())
        report = {
            "artifact_type": "ninfer_r9700_q4_hardware_peak_probe", "schema_version": 2,
            "status": "passed", "hardware": {"device": MODULE.GPU,
                "architecture": MODULE.ARCH, "wave_size": 32, "l2_bytes": 8 << 20},
            "power_profile": {"required": "auto", "before": "auto", "after": "auto"},
            "iu4_issue": {"opcode": "v_wmma_i32_16x16x32_iu4",
                "independent_accumulator_chains": 8, "samples_ms": [3, 1, 2, 5, 4],
                "median_ms": 3, "median_useful_w4a8_tops": 200},
            "q4g64_stream": {"group_size": 64, "code_bytes_per_group_row": 32,
                "scale_bytes_per_group_row": 2, "code_to_scale_byte_ratio": 16,
                "working_set_over_l2": 32, "samples_ms": [5, 3, 4, 2, 1],
                "median_ms": 3, "median_represented_gbps": 600},
        }
        self.assertEqual(MODULE._validate_peak(report, plan), {"iu4": 200.0, "stream": 600.0})
        report["q4g64_stream"]["median_ms"] = 4
        with self.assertRaisesRegex(ValueError, "medians"):
            MODULE._validate_peak(report, plan)

    def test_trace_binds_mask_launches_and_resources(self):
        plan = json.loads((HERE / "plan.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.db"
            connection = sqlite3.connect(path)
            connection.execute("create table rocpd_info_process_test (id integer, command text, environment text)")
            command = MODULE._expected_command(plan, HERE, "half32")
            connection.execute("insert into rocpd_info_process_test values (1, ?, ?)",
                               (shlex_join(command), json.dumps({"HSA_CU_MASK": "0:0-31"})))
            connection.execute("create table rocpd_info_agent_test (type text, name text, product_name text)")
            connection.execute("insert into rocpd_info_agent_test values ('GPU', ?, ?)",
                               (MODULE.ARCH, MODULE.GPU))
            connection.execute("create table kernels (start integer, name text, duration integer, grid_x integer, grid_y integer, grid_z integer, workgroup_x integer, workgroup_y integer, workgroup_z integer, vgpr_count integer, lds_size integer, scratch_size integer)")
            rows = []
            for index in range(176):
                grid_x = 49152 if index < 48 else 20480
                rows.append((index, MODULE.KERNEL, 1000, grid_x, 32, 1, 512, 1, 1,
                             96, 17152, 0))
            rows.append((176, "unrelated_kernel", 1000, 1, 1, 1, 64, 1, 1,
                         8, 0, 0))
            connection.executemany("insert into kernels values (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            connection.commit(); connection.close()
            result = MODULE._validate_trace(path, command, "0:0-31", plan)
            self.assertEqual(result["dispatches"], 176)
            self.assertEqual(result["summed_duration_ms"], 0.176)
            with self.assertRaisesRegex(ValueError, "HSA_CU_MASK"):
                MODULE._validate_trace(path, command, "0:0-15", plan)


def shlex_join(command):
    import shlex
    return shlex.join(command)


if __name__ == "__main__":
    unittest.main()
