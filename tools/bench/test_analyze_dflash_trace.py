from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from tools.bench.analyze_dflash_trace import (
    EXPECTED_CLAIMS,
    EXPECTED_COMMON_WORKLOAD,
    EXPECTED_LIMITATIONS,
    EXPECTED_POWER_PROFILE,
    EXPECTED_STATUS,
    FULL_LAYERS,
    GDN_LAYERS,
    MEASURED,
    analyze,
    _validate_kernel_intervals,
    _validate_report,
    _validate_verify_ranges,
)


def fixture(rounds: int = 1):
    rows = [(0, 1_000_000, MEASURED)]
    clock = 100
    for _ in range(rounds):
        for layer in range(64):
            outer_family = (
                "ninfer.attention.verify.layer.full"
                if layer in FULL_LAYERS else "ninfer.gdn.verify.layer.gdn"
            )
            leaf_family = (
                "ninfer.attention.verify.attention"
                if layer in FULL_LAYERS else "ninfer.gdn.verify.gdn"
            )
            rows.append((clock, clock + 80, f"{outer_family} payload={layer}"))
            rows.append((clock + 5, clock + 35, f"{leaf_family} payload={layer}"))
            rows.append((clock + 40, clock + 70,
                         f"ninfer.post-mixer.verify.post_mixer payload={layer}"))
            clock += 100
    return sorted(rows)


class DFlashTraceMarkerTest(unittest.TestCase):
    def test_valid_report_reaches_success_path(self):
        cell = {"draft_tokens": 4, "verify_width": 5}
        config = {
            "concurrency": 1, "spec": "dflash", "draft_tokens": 4,
            "dflash_verify_width_requested": 5, "dflash_verify_width": 5,
            "proposal_head": "optimized", "use_device_graph": False,
            "speculative_execution": True, "decode_path": "dflash_eager",
            "prefill_chunk": 4096,
            "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 32,
            "repetitions": 1, "warmup": 1, "max_context": 256,
        }
        speculative = {"enabled": True, "draft_window": 4, "rounds": 2,
                       "drafted_tokens": 8, "accepted_tokens": 3,
                       "fallback_steps": 0}
        test = {"kind": "whole", "n_prompt": 128, "n_gen": 64,
                "requested_output_tokens": 65,
                "decode_seconds_mean": 1.0, "decode_output_tok_s_mean": 2.0,
                "decode_engine_tok_s_mean": 3.0, "speculative": speculative,
                "reps": [{"generated_output_tokens": 65, "decode_output_tokens": 64,
                    "decode_engine_tokens": 5, "speculative": speculative,
                    "timings": {"prepare_seconds": 0.1, "vision_seconds": 0.0,
                        "prefill_seconds": 0.2, "decode_seconds": 1.0,
                        "total_seconds": 1.3}}]}
        actual_test, actual_speculative = _validate_report(
            {"artifact_type": "ninfer_bench_report", "schema_version": 20,
             "config": config, "tests": [test]}, cell
        )
        self.assertIs(actual_test, test)
        self.assertIs(actual_speculative, speculative)

    def test_analyze_success_path_uses_valid_speculative_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell_root = root / "k4w5"
            raw = cell_root / "raw"
            raw.mkdir(parents=True)
            identities = {}
            for label in ("bench", "artifact", "corpus", "profiler", "inventory",
                          "cmake", "compile", "receipt"):
                path = root / label
                path.write_bytes(label.encode())
                identities[label] = {
                    "path": str(path), "file_size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            analyzer_path = Path(__file__).with_name("analyze_dflash_trace.py")
            analyzer_identity = {
                "path": str(analyzer_path), "file_size_bytes": analyzer_path.stat().st_size,
                "sha256": hashlib.sha256(analyzer_path.read_bytes()).hexdigest(),
            }
            command = "bench exact"
            cell = {"draft_tokens": 4, "verify_width": 5,
                    "benchmark_command_string": command}
            plan = {
                "artifact_type": "ninfer_r9700_dflash_owner_trace_plan",
                "schema_version": 1, "status": EXPECTED_STATUS, "output_root": str(root),
                "common_workload": EXPECTED_COMMON_WORKLOAD,
                "power_profile": EXPECTED_POWER_PROFILE,
                "cells": {"k4w5": cell, "k5w6": {"draft_tokens": 5,
                    "verify_width": 6, "benchmark_command_string": "other"}},
                "benchmark_executable": identities["bench"],
                "artifact": {**identities["artifact"], "weights_id": "fixture-dflash",
                             "conversion_report": identities["receipt"]},
                "corpus": identities["corpus"], "profiler": identities["profiler"],
                "inventory": identities["inventory"], "analyzer": analyzer_identity,
                "build_configuration": {"cmake_cache": identities["cmake"],
                                        "compile_commands": identities["compile"]},
                "claims": EXPECTED_CLAIMS, "limitations": EXPECTED_LIMITATIONS,
            }
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (cell_root / "power-before.txt").write_text("auto\n")
            (cell_root / "power-after.txt").write_text("auto\n")
            config = {
                "concurrency": 1, "spec": "dflash", "draft_tokens": 4,
                "dflash_verify_width_requested": 5, "dflash_verify_width": 5,
                "proposal_head": "optimized", "use_device_graph": False,
                "speculative_execution": True, "decode_path": "dflash_eager",
                "prefill_chunk": 4096,
                "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 32,
                "repetitions": 1, "warmup": 1, "max_context": 256,
            }
            speculative = {"enabled": True, "draft_window": 4, "rounds": 1,
                           "drafted_tokens": 4, "accepted_tokens": 2,
                           "fallback_steps": 0}
            report = {
                "artifact_type": "ninfer_bench_report", "schema_version": 20,
                "command": command,
                "environment": {"gpu_name": "AMD Radeon AI PRO R9700",
                    "architecture_name": "gfx1201", "hip_runtime_version": "fixture",
                    "hip_driver_version": "fixture", "device_id": 0},
                "artifact": {"path": identities["artifact"]["path"],
                             "file_size_bytes": identities["artifact"]["file_size_bytes"]},
                "load": {"target": "qwen3_8_27b_r9700", "weights_id": "fixture-dflash"},
                "config": config,
                "tests": [{"kind": "whole", "n_prompt": 128, "n_gen": 64,
                    "requested_output_tokens": 65,
                    "decode_seconds_mean": 1.0, "decode_output_tok_s_mean": 2.0,
                    "decode_engine_tok_s_mean": 3.0, "speculative": speculative,
                    "reps": [{"generated_output_tokens": 65, "decode_output_tokens": 64,
                        "decode_engine_tokens": 3, "speculative": speculative,
                        "timings": {"prepare_seconds": 0.1, "vision_seconds": 0.0,
                            "prefill_seconds": 0.2, "decode_seconds": 1.0,
                            "total_seconds": 1.3}}]}],
            }
            (cell_root / "benchmark-report.json").write_text(json.dumps(report))
            database = raw / "k4w5_results.db"
            connection = sqlite3.connect(database)
            connection.executescript("""
                create table rocpd_info_process(nid integer,pid integer,command text,environment text,extdata text);
                create table rocpd_info_agent(nid integer,pid integer,type text,absolute_index integer,logical_index integer,type_index integer,name text,product_name text,extdata text);
                create table regions(nid integer,pid integer,category text,name text,start integer,end integer,duration integer,extdata text);
                create table kernels(nid integer,pid integer,agent_abs_index integer,agent_log_index integer,agent_type_index integer,agent_type text,dispatch_id integer,start integer,end integer,duration integer,name text,region text);
                create table memory_copies(nid integer,pid integer,start integer,end integer,duration integer,size integer,name text,region_name text);
            """)
            environment = {"ROCPROF_OUTPUT_FORMAT": "rocpd", "ROCPROF_MARKER_API_TRACE": "1",
                "ROCPROF_KERNEL_TRACE": "1", "ROCPROF_MEMORY_COPY_TRACE": "1",
                "ROCPROF_SELECTED_REGIONS": "1"}
            process_extdata = {"output_path": str(raw), "output_file": "k4w5",
                "raw_output_path": str(raw), "raw_output_file": "k4w5",
                "rocpd_output": True, "kernel_rename": False}
            connection.execute("insert into rocpd_info_process values(?,?,?,?,?)",
                               (1, 2, command, json.dumps(environment),
                                json.dumps(process_extdata)))
            topology = {"cu_count": 64, "simd_count": 128, "wave_front_size": 32}
            connection.execute("insert into rocpd_info_agent values(?,?,?,?,?,?,?,?,?)",
                (1, 2, "GPU", 1, 1, 0, "gfx1201", "AMD Radeon AI PRO R9700",
                 json.dumps(topology)))
            region_rows = [(10, 999_990, MEASURED)]
            clock = 1_000
            for layer in range(64):
                outer = ("ninfer.attention.prefill.layer.full" if layer in FULL_LAYERS
                         else "ninfer.gdn.prefill.layer.gdn")
                leaf = ("ninfer.attention.prefill.attention" if layer in FULL_LAYERS
                        else "ninfer.gdn.prefill.gdn")
                region_rows.extend(((clock, clock+80, f"{outer} payload={layer}"),
                                    (clock+5, clock+35, f"{leaf} payload={layer}"),
                                    (clock+40, clock+70,
                                     f"ninfer.post-mixer.prefill.post_mixer payload={layer}")))
                clock += 100
            region_rows.append((900, clock, "ninfer.prefill.prefill.chunk payload=128"))
            for begin, end, message in fixture()[1:]:
                region_rows.append((begin + 20_000, end + 20_000, message))
            connection.execute("insert into regions values(?,?,?,?,?,?,?,?)",
                (1, 2, "MARKER_CONTROL_API", "roctxProfilerResume", 1, 2, 1, "{}"))
            for begin, end, message in sorted(region_rows):
                connection.execute("insert into regions values(?,?,?,?,?,?,?,?)",
                    (1, 2, "MARKER_CORE_RANGE_API", "roctxThreadRangeA", begin, end,
                     end-begin, json.dumps({"message": message})))
            connection.execute("insert into regions values(?,?,?,?,?,?,?,?)",
                (1, 2, "MARKER_CONTROL_API", "roctxProfilerPause", 999_998, 999_999, 1, "{}"))
            connection.execute("insert into kernels values(?,?,?,?,?,?,?,?,?,?,?,?)",
                (1, 2, 1, 1, 0, "GPU", 1, 20_110, 20_120, 10, "kernel",
                 "ninfer.gdn.verify.gdn payload=0"))
            connection.commit(); connection.close()
            result = analyze(root / "plan.json", "k4w5")
            self.assertEqual(result["target_verification_rounds"], 1)
            self.assertEqual(result["benchmark"]["decode_engine_tok_s_profiled"], 3.0)
            connection = sqlite3.connect(database)
            connection.execute("update kernels set region='unknown' where dispatch_id=1")
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "unknown region"):
                analyze(root / "plan.json", "k4w5")

    def test_exact_two_round_topology(self):
        rounds, families = _validate_verify_ranges(fixture(2), (0, 1_000_000))
        self.assertEqual(rounds, 2)
        self.assertEqual(len(families["ninfer.attention.verify.layer.full"]), 32)
        self.assertEqual(len(families["ninfer.gdn.verify.layer.gdn"]), 96)

    def test_missing_layer_fails(self):
        rows = fixture()
        rows = [row for row in rows if row[2] != "ninfer.gdn.verify.gdn payload=0"]
        with self.assertRaisesRegex(ValueError, "complete 64-layer"):
            _validate_verify_ranges(rows, (0, 1_000_000))

    def test_wrong_partition_fails(self):
        rows = fixture()
        rows.append((900_000, 900_010, "ninfer.attention.verify.attention payload=0"))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            _validate_verify_ranges(sorted(rows), (0, 1_000_000))

    def test_outside_measured_fails(self):
        rows = fixture()
        begin, end, message = rows[-1]
        rows[-1] = (begin, 1_000_001, message)
        with self.assertRaisesRegex(ValueError, "outside measured"):
            _validate_verify_ranges(rows, (0, 1_000_000))

    def test_overlapping_target_layer_outer_fails(self):
        rows = fixture()
        message = "ninfer.gdn.verify.layer.gdn payload=1"
        rows = [((170, 280, value) if value == message else (begin, end, value))
                for begin, end, value in rows]
        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_verify_ranges(sorted(rows), (0, 1_000_000))

    def test_bounded_terminal_async_drain_is_retained(self):
        rows = [
            {"dispatch_id": 1, "start": 110, "end": 120, "duration": 10},
            {"dispatch_id": 2, "start": 210, "end": 220, "duration": 10},
            {"dispatch_id": 3, "start": 150, "end": 160, "duration": 10},
        ]
        actual = _validate_kernel_intervals(rows, (100, 200))
        self.assertEqual(actual["calls"], 1)
        self.assertEqual(actual["dispatch_id_span_to_capture_end"], 1)
        self.assertEqual(actual["wholly_post_marker_calls"], 1)

    def test_nonterminal_post_measured_timestamp_fails(self):
        rows = [
            {"dispatch_id": 1, "start": 210, "end": 220, "duration": 10},
            {"dispatch_id": 200, "start": 110, "end": 120, "duration": 10},
        ]
        with self.assertRaisesRegex(ValueError, "bounded allowance"):
            _validate_kernel_intervals(rows, (100, 200))

    def test_bounded_crossing_async_drain_is_retained_and_prestart_fails(self):
        crossing = [{"dispatch_id": 1, "start": 190, "end": 210, "duration": 20}]
        actual = _validate_kernel_intervals(crossing, (100, 200))
        self.assertEqual(actual["crossing_calls"], 1)
        before = [{"dispatch_id": 1, "start": 90, "end": 99, "duration": 9}]
        with self.assertRaisesRegex(ValueError, "precedes"):
            _validate_kernel_intervals(before, (100, 200))

    def test_terminal_async_drain_call_cap_fails(self):
        rows = []
        for index in range(33):
            rows.extend((
                {"dispatch_id": 2 * index, "start": 210 + index,
                 "end": 211 + index, "duration": 1},
                {"dispatch_id": 2 * index + 1, "start": 110 + index,
                 "end": 111 + index, "duration": 1},
            ))
        with self.assertRaisesRegex(ValueError, "bounded allowance"):
            _validate_kernel_intervals(rows, (100, 200))


if __name__ == "__main__":
    unittest.main()
