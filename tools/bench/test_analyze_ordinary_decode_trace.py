#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.bench.analyze_ordinary_decode_trace import (
    ORDINARY,
    _dispatch_timeline,
    analyze,
    main,
)


class AnalyzeOrdinaryDecodeTraceTest(unittest.TestCase):
    def test_dispatch_interval_concurrency_and_gap_math(self) -> None:
        timeline = _dispatch_timeline([(0, 2), (1, 3), (5, 7)], 6)
        self.assertEqual(timeline["dispatch_interval_union_ns"], 5)
        self.assertEqual(timeline["first_to_last_dispatch_span_ns"], 7)
        self.assertEqual(timeline["no_dispatch_interval_gap_ns"], 2)
        self.assertEqual(timeline["connected_dispatch_interval_segments"], 2)
        self.assertEqual(timeline["maximum_concurrent_dispatch_intervals"], 2)
        self.assertAlmostEqual(
            timeline["mean_concurrent_dispatch_intervals_while_present"], 6 / 5)
        self.assertAlmostEqual(
            timeline["mean_concurrent_dispatch_intervals_over_span"], 6 / 7)

    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        report = root / "report.json"
        report.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": 20,
            "config": {
                "concurrency": 1, "spec": "none", "draft_tokens": 0,
                "speculative_execution": False, "dflash_verify_width": 0,
                "use_device_graph": True, "decode_path": "device_graph",
                "repetitions": 1,
            },
            "tests": [{
                "kind": "whole", "n_prompt": 8192, "n_gen": 1,
                "speculative": {"enabled": False},
                "reps": [{"decode_output_tokens": 1, "decode_engine_tokens": 1}],
            }],
        }), encoding="utf-8")
        trace = root / "rocpd"
        trace.mkdir()
        database = trace / "ordinary_results.db"
        connection = sqlite3.connect(database)
        connection.executescript("""
            create table regions(
                start integer, "end" integer, duration integer, name text,
                category text, extdata text);
            create table kernels(
                id integer, dispatch_id integer, start integer, "end" integer,
                duration integer, name text, region text, graph_exec_id integer);
            create table memory_copies(
                id integer, start integer, "end" integer, duration integer,
                name text, region_name text, size integer, graph_exec_id integer);
            create table graph_launches(
                start integer, "end" integer, duration integer,
                graph_exec_id integer, kernel_dispatch_count integer);
        """)
        connection.executemany("insert into regions values (?,?,?,?,?,?)", [
            (0, 1_000, 1_000, "roctxThreadRangeA", "MARKER_CORE_RANGE_API",
             json.dumps({"message": "ninfer_bench_measured"})),
            (100, 900, 800, "roctxThreadRangeA", "MARKER_CORE_RANGE_API",
             json.dumps({"message": ORDINARY})),
            (110, 140, 30, "hipGraphLaunch", "HIP_RUNTIME_API_EXT", "{}"),
            (800, 850, 50, "hipEventSynchronize", "HIP_RUNTIME_API_EXT", "{}"),
        ])
        connection.executemany("insert into kernels values (?,?,?,?,?,?,?,?)", [
            (1, 10, 200, 500, 300, "slow", ORDINARY, 7),
            (2, 11, 300, 600, 300, "slow", ORDINARY, 7),
            (3, 12, 650, 700, 50, "fast", ORDINARY, 0),
            (5, 14, 150, 650, 500, "wide", ORDINARY, 0),
            # Temporal overlap alone is deliberately not an association.
            (4, 13, 120, 130, 10, "prior-work", "", 0),
        ])
        connection.executemany("insert into memory_copies values (?,?,?,?,?,?,?,?)", [
            (1, 710, 730, 20, "D2D", ORDINARY, 0, 0),
            (2, 720, 750, 30, "D2D", ORDINARY, 64, 0),
        ])
        connection.execute("insert into graph_launches values (?,?,?,?,?)",
                           (115, 145, 30, 7, 2))
        connection.commit()
        connection.close()
        return report, trace, database

    def test_exact_association_and_interval_union_math(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, trace, _ = self.fixture(Path(directory))
            result = analyze(report, trace)
            self.assertEqual(result["kernels"]["dispatch_count"], 4)
            self.assertEqual(result["kernels"]["graph_dispatch_count"], 2)
            self.assertEqual(result["kernels"]["summed_device_duration_ns"], 1_150)
            timeline = result["kernels"]["timeline"]
            self.assertEqual(timeline["dispatch_interval_union_ns"], 550)
            self.assertEqual(timeline["first_to_last_dispatch_span_ns"], 550)
            self.assertEqual(timeline["no_dispatch_interval_gap_ns"], 0)
            self.assertEqual(timeline["connected_dispatch_interval_segments"], 1)
            self.assertEqual(timeline["maximum_concurrent_dispatch_intervals"], 3)
            self.assertAlmostEqual(
                timeline["mean_concurrent_dispatch_intervals_while_present"],
                1_150 / 550)
            self.assertEqual(
                [row["name"] for row in result["kernels"][
                    "contributors_by_summed_device_duration"]],
                ["slow", "wide", "fast"])
            self.assertEqual(
                [row["name"] for row in result["kernels"][
                    "contributors_by_dispatch_interval_union"]],
                ["wide", "slow", "fast"])
            slow = result["kernels"]["contributors_by_summed_device_duration"][0]
            self.assertEqual(slow["summed_device_duration_ns"], 600)
            self.assertEqual(slow["interval_union_ns"], 400)
            self.assertNotIn("prior-work", json.dumps(result))
            self.assertEqual(result["memory_copies"]["bytes"], 64)
            self.assertEqual(result["memory_copies"]["contributors"][0]["record_count"], 2)
            self.assertEqual(result["runtime_api"]["record_count"], 2)
            self.assertEqual(result["graph_launch"]["reported_kernel_dispatch_count"], 2)

    def test_rejects_schema_and_marker_contract_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, trace, database = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute("alter table kernels rename to kernels_old")
            connection.execute("create table kernels(start integer)")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "lacks required columns"):
                analyze(report, trace)
        for message, expected in (
            ("ninfer.decode.decode.mtp_round payload=8192", "MTP or DFlash"),
            ("ninfer.dflash.decode.dflash_round payload=8192", "MTP or DFlash"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                report, trace, database = self.fixture(Path(directory))
                connection = sqlite3.connect(database)
                connection.execute("insert into regions values (?,?,?,?,?,?)", (
                    150, 160, 10, "roctxThreadRangeA", "MARKER_CORE_RANGE_API",
                    json.dumps({"message": message})))
                connection.commit()
                connection.close()
                with self.assertRaisesRegex(ValueError, expected):
                    analyze(report, trace)

    def test_rejects_duplicate_or_non_nested_ordinary_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, trace, database = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute("insert into regions values (?,?,?,?,?,?)", (
                200, 300, 100, "roctxThreadRangeA", "MARKER_CORE_RANGE_API",
                json.dumps({"message": ORDINARY})))
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "exactly one.*ordinary_round"):
                analyze(report, trace)

    def test_rejects_graph_dispatch_or_temporal_association_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, trace, database = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute("update graph_launches set kernel_dispatch_count=3")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "graph dispatch count"):
                analyze(report, trace)
        with tempfile.TemporaryDirectory() as directory:
            report, trace, database = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute("update kernels set region='' where dispatch_id=10")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "unmarked"):
                analyze(report, trace)
        with tempfile.TemporaryDirectory() as directory:
            report, trace, database = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute(
                "insert into regions values (?,?,?,?,?,?)",
                (90, 120, 30, "hipMemcpy", "HIP_RUNTIME_API", "{}"))
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "ambiguously crosses"):
                analyze(report, trace)

    def test_stdout_is_deterministic_and_out_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, trace, _ = self.fixture(Path(directory))
            first, second = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(first):
                self.assertEqual(main([
                    "--benchmark-report", str(report), "--rocpd", str(trace)]), 0)
            with contextlib.redirect_stdout(second):
                self.assertEqual(main([
                    "--benchmark-report", str(report), "--rocpd", str(trace)]), 0)
            self.assertEqual(first.getvalue(), second.getvalue())
            self.assertEqual(json.loads(first.getvalue())["schema_version"], 1)
            output = Path(directory) / "nested" / "analysis.json"
            self.assertEqual(main([
                "--benchmark-report", str(report), "--rocpd", str(trace),
                "--out", str(output)]), 0)
            with self.assertRaisesRegex(SystemExit, "File exists"):
                main(["--benchmark-report", str(report), "--rocpd", str(trace),
                      "--out", str(output)])


if __name__ == "__main__":
    unittest.main()
