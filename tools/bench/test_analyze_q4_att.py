#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bench.check_q4_att_parser import validate_parser
from tools.bench.analyze_q4_att import (
    EXPECTED_GRID,
    EXPECTED_Q4_DISPATCHES,
    EXPECTED_WORKGROUP,
    KERNEL,
    MATCH_ITERATION,
    ROLE_REGION,
    _read_att,
    _read_static,
    _resolve_att_outputs,
    _selected_dispatch,
    _validate_capture_log,
    _validate_closure,
)


def dispatch(index: int) -> dict:
    return {
        "dispatch_id": 100 + index,
        "symbol": f"namespace::{KERNEL}(args)",
        "region": ROLE_REGION if index == MATCH_ITERATION - 1 else "ninfer.gdn.prefill.gdn payload=0",
        "grid": EXPECTED_GRID if index == MATCH_ITERATION - 1 else (49152, 32, 1),
        "workgroup": EXPECTED_WORKGROUP,
        "private_bytes": 0,
    }


class SelectionTest(unittest.TestCase):
    def test_installed_parser_accepts_exact_nonconflicting_att_options(self):
        validate_parser(Path("/opt/rocm/core-10.0/bin/rocprofv3"), [
            "--selected-regions", "--att", "--att-target-cu", "1",
            "--att-simd-select", "0", "--att-buffer-size", "1610612736",
            "--att-shader-engine-mask", "0x1", "--att-gpu-index", "0",
            "--kernel-include-regex", "a8q4g64_linear_prefill_cta_kernel",
            "--kernel-iteration-range", "[3]", "--marker-trace", "--kernel-trace",
            "-f", "csv", "rocpd", "-d", "/tmp/ninfer-att-parser-only",
            "-o", "production-q4-p2048-att", "--", "/usr/bin/true",
        ])

    def test_capture_buffer_overflow_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "capture.log"
            for message in ("SQTT data buffer full, SE(0)",
                            "Thread trace buffer full!",
                            "Data Lost: 4096 bytes",
                            "Invalid buffer size: 1610612736",
                            "SQTT Buffer size too low",
                            "SQTT Buffer size too high",
                            "Failed to allocate tracing buffers",
                            "Unable to start thread trace worker thread"):
                log.write_text(message, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "incomplete or lost"):
                    _validate_capture_log(log)

    def test_checksum_closure_requires_exact_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            closure = root / "prepared.sha256"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            expected = {
                "first": {"path": str(first), "sha256": "1" * 64},
                "second": {"path": str(second), "sha256": "2" * 64},
            }
            closure.write_text(
                f"{'1' * 64}  {first}\n{'2' * 64}  {second}\n", encoding="utf-8")
            _validate_closure(closure, expected)
            closure.write_text(f"{'1' * 64}  {first}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                _validate_closure(closure, expected)

    def test_exact_third_match_is_selected(self):
        rows = [dispatch(index) for index in range(EXPECTED_Q4_DISPATCHES)]
        self.assertEqual(_selected_dispatch(rows, require_full_inventory=True)["dispatch_id"], 102)

    def test_wrong_role_fails_closed(self):
        rows = [dispatch(index) for index in range(EXPECTED_Q4_DISPATCHES)]
        rows[MATCH_ITERATION - 1]["region"] = "ninfer.attention.prefill.attention payload=0"
        with self.assertRaisesRegex(ValueError, "not exact post-mixer"):
            _selected_dispatch(rows, require_full_inventory=True)

    def test_incomplete_source_inventory_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exact 176"):
            _selected_dispatch([dispatch(index) for index in range(3)],
                               require_full_inventory=True)

    def test_current_emitted_production_kernel_is_exact(self):
        root = Path(__file__).resolve().parents[2]
        static = _read_static(root / "tools/r9700/build/q4g64_linear.s")
        self.assertEqual(static["vgpr"], 88)
        self.assertEqual(static["occupancy"], 16)
        self.assertEqual(static["iu4_wmma_pc_count"], 8)

    def test_att_outputs_are_derived_without_glob_selection(self):
        rows = [dispatch(index) for index in range(EXPECTED_Q4_DISPATCHES)]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            stem = "ui_output_agent_1390_dispatch_1"
            (raw / "production-q4-p2048-att_1390_shader_engine_0_1.att").touch()
            (raw / stem).mkdir()
            (raw / f"stats_{stem}.csv").touch()
            with mock.patch("tools.bench.analyze_q4_att._q4_dispatches",
                            return_value=(rows, {"agent_id": 1, "absolute_index": 1,
                                                "type_index": 0, "gpu_index": 0,
                                                "agent_handle": 1390})):
                stats, ui = _resolve_att_outputs(raw, raw / "results.db")
                self.assertEqual(stats.name, f"stats_{stem}.csv")
                self.assertEqual(ui.name, stem)
                (raw / "ui_output_agent_1390_dispatch_2").mkdir()
                with self.assertRaisesRegex(ValueError, "extra or mismatched"):
                    _resolve_att_outputs(raw, raw / "results.db")

    def test_att_output_from_wrong_gpu_handle_fails_closed(self):
        rows = [dispatch(index) for index in range(EXPECTED_Q4_DISPATCHES)]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "production-q4-p2048-att_7433_shader_engine_0_1.att").touch()
            (raw / "ui_output_agent_7433_dispatch_1").mkdir()
            (raw / "stats_ui_output_agent_7433_dispatch_1.csv").touch()
            r9700 = {"agent_id": 1, "absolute_index": 1, "type_index": 0,
                     "gpu_index": 0, "agent_handle": 7432}
            with mock.patch("tools.bench.analyze_q4_att._q4_dispatches",
                            return_value=(rows, r9700)):
                with self.assertRaisesRegex(ValueError, "not the R9700"):
                    _resolve_att_outputs(raw, raw / "results.db")


class StatsTest(unittest.TestCase):
    def write_stats(self, path: Path, *, duplicate=False, wmma_count=8):
        fields = ["CodeObj", "Vaddr", "Instruction", "Hitcount", "Latency", "Stall",
                  "Idle", "Source"]
        rows = [
            [7, 0x100, "global_load_b32 v1, v[2:3], off", 10, 20, 3, 1, "source:1"],
            [7, 0x104, "s_wait_loadcnt 0x0", 10, 30, 12, 2, "source:2"],
            [7, 0x108, "ds_load_b32 v1, v2", 10, 20, 2, 1, "source:3"],
            [7, 0x10C, "s_barrier_wait -1", 10, 20, 4, 1, "source:4"],
            [7, 0x110, "v_lshl_add_u32 v1, v2, 4, v3", 10, 10, 0, 0, "source:5"],
        ]
        rows.extend([
            [7, 0x200 + 4 * index,
             "v_wmma_i32_16x16x32_iu4 v[1:8], v[9:10], v[11:12], v[1:8] "
             + ("neg_lo:[0,1,0]" if index < 4 else "neg_lo:[1,1,0]"),
             10, 40, index, 0, f"source:{10 + index}"]
            for index in range(wmma_count)
        ])
        if duplicate:
            rows.append(rows[0])
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_exact_stats_are_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            self.write_stats(path)
            rows, summary = _read_att(path)
            self.assertEqual(len(rows), 13)
            self.assertEqual(summary["code_object_id"], 7)
            self.assertEqual(summary["regions"]["iu4_wmma"]["pc_count"], 8)
            self.assertEqual(summary["regions"]["global_load_and_wait"]["stall"], 15)

    def test_missing_wmma_or_duplicate_pc_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            self.write_stats(path, wmma_count=7)
            with self.assertRaisesRegex(ValueError, "eight signed IU4"):
                _read_att(path)
            self.write_stats(path, duplicate=True)
            with self.assertRaisesRegex(ValueError, "repeats"):
                _read_att(path)


if __name__ == "__main__":
    unittest.main()
