#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.bench import test_validate_profile_pmc as pmc_test
from tools.bench.validate_profile_trace import main, validate


class ValidateProfileTraceTest(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, Path]:
        paths = pmc_test.ValidateProfilePmcTest().fixture(root)
        plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
        plan["profile_kind"] = "trace"
        plan["measured_region"] = "ninfer_bench_measured"
        plan["counters"] = []
        plan["kernel_include_regex"] = None
        plan["required_power_profile"]["value"] = "auto"
        plan["profiler_command"] = [
            "/opt/rocm/bin/rocprofv3", "--selected-regions", "-f", "rocpd", "-d",
            str((root / "rocprof-trace").resolve()), "--marker-trace", "--kernel-trace",
            "--memory-copy-trace", "--stats", "--summary", "--",
            *plan["benchmark_command"],
        ]
        paths["plan"].write_text(json.dumps(plan), encoding="utf-8")
        paths["before"].write_text("auto\n", encoding="utf-8")
        paths["after"].write_text("auto\n", encoding="utf-8")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        report["tests"][0]["prefill_seconds_mean"] = 0.0000003
        paths["report"].write_text(json.dumps(report), encoding="utf-8")

        connection = sqlite3.connect(paths["database"])
        connection.execute("alter table rocpd_info_agent_x add column absolute_index integer")
        connection.execute("update rocpd_info_agent_x set absolute_index=1 where type='GPU'")
        connection.executescript("""
            create table regions(start integer, "end" integer, extdata text);
            create table kernels(
                dispatch_id integer, name text, region text, start integer, "end" integer,
                duration integer, stream_id integer, agent_abs_index integer,
                grid_x integer, grid_y integer, grid_z integer,
                workgroup_x integer, workgroup_y integer, workgroup_z integer,
                sgpr_count integer, vgpr_count integer, accum_vgpr_count integer, lds_size integer,
                scratch_size integer, static_lds_size integer, static_scratch_size integer);
            create table memory_copies(start integer, "end" integer, duration integer,
                name text, region_name text, size integer);
        """)
        connection.executemany("insert into regions values (?,?,?)", [
            (0, 400, json.dumps({"message": "ninfer_bench_measured"})),
            (50, 350, json.dumps({
                "message": "ninfer.prefill.prefill.chunk payload=2048"})),
        ])
        connection.executemany("insert into kernels values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, "kernel_a", "ninfer.attention.prefill.attention", 100, 200, 100, 7, 1,
             64, 32, 1, 256, 1, 1, 128, 24, 8, 4096, 0, 4096, 0),
            (2, "kernel_b", "ninfer.gdn.prefill.gdn", 150, 250, 100, 8, 1,
             48, 4, 1, 256, 1, 1, None, None, None, None, None, None, None),
        ])
        connection.commit()
        connection.close()
        return paths

    def run_validate(self, paths: dict[str, Path]) -> dict:
        return validate(
            paths["plan"], paths["report"], paths["database"], paths["before"],
            paths["after"], paths["terminal"], paths["artifact"], paths["executable"],
            paths["corpus"],
        )

    def test_emits_lossless_dispatches_and_conserved_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_validate(self.fixture(Path(directory)))
            self.assertEqual(result["artifact_type"], "ninfer_r9700_selected_profile_trace")
            self.assertFalse(result["profile_timing_admissible"])
            self.assertEqual(result["aggregates"], {
                "dispatch_count": 2,
                "independent_device_service_time_ns": 200,
                "device_wall_union_ns": 150,
            })
            first, second = result["dispatches"]
            self.assertEqual(first["dispatch_id"], "trace:1")
            self.assertEqual(first["grid"], {"x": 64, "y": 32, "z": 1})
            self.assertEqual(first["resources"]["sgpr_count"], 128)
            self.assertEqual(first["resources"]["vgpr_count"], 24)
            self.assertIsNone(second["resources"]["scratch_bytes"])
            self.assertEqual(result["power_profile"]["before"], "auto")
            self.assertEqual(result["selected_route"]["artifact"]["weights_id"], "selected")
            self.assertEqual(result["selected_route"]["unprofiled_p2048"]["observed_tok_s"],
                             1900.0)
            self.assertEqual(result["analyzer"]["operator_stage_attribution"]["complete"], True)

    def test_rejects_wrong_power_profiler_contract_and_duplicate_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            paths["after"].write_text("profile_standard\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "auto before/after"):
                self.run_validate(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
            plan["profiler_command"].remove("--marker-trace")
            paths["plan"].write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker/kernel trace contract"):
                self.run_validate(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            connection = sqlite3.connect(paths["database"])
            connection.execute("update kernels set dispatch_id=1 where dispatch_id=2")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "IDs must be present and unique"):
                self.run_validate(paths)

    def test_rejects_authority_drift_and_report_command_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            paths["terminal"].write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "terminal selection authority"):
                self.run_validate(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["command"] += " --forged"
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "planned selected P2048 trace"):
                self.run_validate(paths)

    def test_cli_refuses_to_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            output = Path(directory) / "trace-evidence.json"
            output.write_text("retained", encoding="utf-8")
            arguments = [
                "--plan", str(paths["plan"]), "--benchmark-report", str(paths["report"]),
                "--database", str(paths["database"]), "--power-before", str(paths["before"]),
                "--power-after", str(paths["after"]), "--terminal-selection",
                str(paths["terminal"]), "--artifact", str(paths["artifact"]),
                "--executable", str(paths["executable"]), "--corpus", str(paths["corpus"]),
                "--out", str(output),
            ]
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                main(arguments)
            self.assertEqual(output.read_text(encoding="utf-8"), "retained")


if __name__ == "__main__":
    unittest.main()
