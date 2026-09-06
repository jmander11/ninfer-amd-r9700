#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.bench.analyze_retained_prefill_trace import (
    _load,
    _validate_database_contract,
)


class AnalyzeRetainedPrefillTraceTest(unittest.TestCase):
    def test_direct_script_invocation_bootstraps_repository_package(self) -> None:
        script = Path(__file__).with_name("analyze_retained_prefill_trace.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"], cwd=directory,
                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def fixture(self, root: Path) -> tuple[Path, list[str]]:
        raw = root / "raw"
        raw.mkdir()
        database = raw / "retained-production-p2048_results.db"
        command = ["/bench", "-p", "2048"]
        connection = sqlite3.connect(database)
        connection.executescript("""
            create table rocpd_info_process_x(
                id integer, nid integer, pid integer, command text,
                environment text, extdata text);
            create table rocpd_info_agent_x(
                id integer, nid integer, pid integer, type text, absolute_index integer,
                logical_index integer, type_index integer, name text, product_name text,
                extdata text);
            create table regions(
                nid integer, pid integer, category text, name text, start integer, "end" integer,
                duration integer, extdata text);
            create table kernels(
                nid integer, pid integer, dispatch_id integer, name text, region text,
                start integer, "end" integer, duration integer, agent_abs_index integer,
                agent_log_index integer, agent_type_index integer, agent_type text);
            create table memory_copies(
                nid integer, pid integer, start integer, "end" integer, duration integer,
                region_name text);
        """)
        environment = {
            "ROCPROF_OUTPUT_FILE_NAME": "retained-production-p2048",
            "ROCPROF_OUTPUT_PATH": str(raw), "ROCPROF_OUTPUT_FORMAT": "rocpd",
            "ROCPROF_MARKER_API_TRACE": "1", "ROCPROF_KERNEL_TRACE": "1",
            "ROCPROF_MEMORY_COPY_TRACE": "1", "ROCPROF_SELECTED_REGIONS": "1",
            "ROCPROFILER_REGISTER_LIBRARY":
                "/opt/rocm/core-10.0/lib/librocprofiler-sdk.so.1.3.5",
        }
        extdata = {
            "output_path": str(raw), "output_file": "retained-production-p2048",
            "raw_output_path": str(raw), "raw_output_file": "retained-production-p2048",
            "rocpd_output": True, "kernel_rename": False,
        }
        connection.execute("insert into rocpd_info_process_x values(?,?,?,?,?,?)", (
            7, 8, 9, " ".join(command), json.dumps(environment), json.dumps(extdata)))
        connection.execute("insert into rocpd_info_agent_x values(?,?,?,?,?,?,?,?,?,?)", (
            1, 8, 9, "GPU", 1, 1, 0, "gfx1201", "AMD Radeon AI PRO R9700",
            json.dumps({"cu_count": 64, "simd_count": 128, "wave_front_size": 32})))
        connection.execute("insert into rocpd_info_agent_x values(?,?,?,?,?,?,?,?,?,?)", (
            2, 8, 9, "GPU", 2, 2, 1, "gfx1036", "AMD Radeon Graphics",
            json.dumps({"cu_count": 2, "simd_count": 4, "wave_front_size": 32})))
        connection.execute("insert into regions values(?,?,?,?,?,?,?,?)", (
            8, 9, "MARKER_CONTROL_API", "roctxProfilerResume", 0, 10, 10, "{}"))
        def region(begin: int, end: int, message: str) -> None:
            connection.execute("insert into regions values(?,?,?,?,?,?,?,?)", (
                8, 9, "MARKER_CORE_RANGE_API", "roctxThreadRangeA", begin, end,
                end - begin, json.dumps({"message": message})))
        region(20, 1_000_000_000, "ninfer_bench_measured")
        region(30, 900_000_000, "ninfer.prefill.prefill.chunk payload=2048")
        cursor = 100
        for layer in range(64):
            outer = (f"ninfer.attention.prefill.layer.full payload={layer}"
                     if layer % 4 == 3 else f"ninfer.gdn.prefill.layer.gdn payload={layer}")
            leaf = (f"ninfer.attention.prefill.attention payload={layer}"
                    if layer % 4 == 3 else f"ninfer.gdn.prefill.gdn payload={layer}")
            region(cursor, cursor + 90, outer)
            region(cursor + 1, cursor + 40, leaf)
            region(cursor + 50, cursor + 89,
                   f"ninfer.post-mixer.prefill.post_mixer payload={layer}")
            cursor += 100
        connection.execute("insert into regions values(?,?,?,?,?,?,?,?)", (
            8, 9, "MARKER_CONTROL_API", "roctxProfilerPause", 1_000_000_010,
            1_000_000_020, 10, "{}"))
        connection.execute("insert into kernels values(?,?,?,?,?,?,?,?,?,?,?,?)", (
            8, 9, 1, "kernel", "ninfer.gdn.prefill.gdn payload=0", 200, 300, 100,
            1, 1, 0, "GPU"))
        connection.commit()
        connection.close()
        return database, command

    def test_accepts_exact_topology_and_dual_gpu_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, command = self.fixture(root)
            result = _validate_database_contract(database, command, root)
            self.assertEqual(result["selected_kernel_owner"]["cu_count"], 64)
            self.assertEqual(len(result["marker_inventory"]), 7)
            self.assertEqual(result["gpu_agents"][1]["architecture"], "gfx1036")

    def test_rejects_process_agent_and_layer_identity_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, command = self.fixture(root)
            connection = sqlite3.connect(database)
            connection.execute("update kernels set agent_log_index=2")
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "did not execute on the R9700"):
                _validate_database_contract(database, command, root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, command = self.fixture(root)
            connection = sqlite3.connect(database)
            row = connection.execute(
                "select rowid,extdata from regions where extdata like '%layer.full payload=3%'"
            ).fetchone()
            connection.execute("update regions set extdata=? where rowid=?", (
                row[1].replace("payload=3", "payload=4"), row[0]))
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "payload sets"):
                _validate_database_contract(database, command, root)

    def test_rejects_bad_nested_marker_and_oversized_unmarked_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, command = self.fixture(root)
            connection = sqlite3.connect(database)
            connection.execute(
                "update regions set start=450, \"end\"=510, duration=60 "
                "where extdata like '%attention payload=3%'")
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "order or nesting"):
                _validate_database_contract(database, command, root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, command = self.fixture(root)
            connection = sqlite3.connect(database)
            connection.execute("insert into kernels values(?,?,?,?,?,?,?,?,?,?,?,?)", (
                8, 9, 2, "__amd_rocclr_fillBufferUnAligned", "", 1000, 22001, 21001,
                1, 1, 0, "GPU"))
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "infrastructure allowance"):
                _validate_database_contract(database, command, root)

    def test_rejects_unknown_association_copy_and_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, command = self.fixture(root)
            connection = sqlite3.connect(database)
            connection.execute("update kernels set region='ninfer.gdn.prefill.fake payload=0'")
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "unexpected ROCTX"):
                _validate_database_contract(database, command, root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, command = self.fixture(root)
            connection = sqlite3.connect(database)
            connection.execute(
                "update kernels set region='ninfer.gdn.prefill.gdn payload=1', "
                "start=150, \"end\"=250, duration=100")
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "starts before"):
                _validate_database_contract(database, command, root)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text('{"timing": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                _load(path, "report")


if __name__ == "__main__":
    unittest.main()
