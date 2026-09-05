#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.prepare_selected_mtp_shortlist_head import REPO, prepare


class SelectedMtpShortlistHeadPrepareTest(unittest.TestCase):
    def test_rejects_dangling_output_namespace_before_route_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "profiles/rocprof") as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text("{}", encoding="utf-8")
            output = root / "out"
            output.symlink_to(root / "missing-target", target_is_directory=True)
            with patch(
                "tools.bench.prepare_selected_mtp_shortlist_head.validate_terminal_production_authority"
            ) as validate, self.assertRaisesRegex(ValueError, "output directory already exists"):
                prepare(selection, output)
            validate.assert_not_called()
            self.assertTrue(os.path.lexists(output))
            self.assertFalse((root / "missing-target").exists())

    def test_binds_one_c1_8k_winner_trace_and_fixed_database(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "profiles/rocprof") as directory:
            root = Path(directory)
            selection = root / "selection.json"
            matrix = root / "matrix" / "manifest.json"
            report = root / "matrix" / "mtp.json"
            artifact = root / "selected.ninfer"
            bench = root / "build" / "bench" / "ninfer_bench"
            planner = root / "build" / "tools" / "planner"
            for path in (selection, matrix, report, artifact, bench, planner):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")
            selection.write_text("{}", encoding="utf-8")
            route = {
                "source_matrices": {"pareto-whole": {
                    "path": str(matrix), "sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
                }},
                "artifact": {"path": str(artifact), "sha256": "artifact", "weights_id": "weights"},
                "benchmark": {"path": str(bench), "sha256": "bench"},
                "cache_profile": {"value_group": 32},
                "execution_profile": {"xattention_profile": "b128-s16-tau900"},
                "selected_prefill_chunk": 2048,
                "hybrid_width_tool": {"path": str(planner)},
            }
            output = root / "out"

            def fake_profile(matrix_dir, out, **kwargs):
                self.assertEqual(matrix_dir, matrix.parent)
                self.assertEqual(kwargs["concurrency"], 1)
                self.assertEqual(kwargs["prompt_tokens"], 8192)
                self.assertEqual(kwargs["generated_tokens"], 256)
                out.mkdir()
                command = [str(bench), "--weights", str(artifact), "--spec", "mtp",
                           "--draft-tokens", "3", "--lm-head-draft", "--whole-pg", "8192,256",
                           "--concurrency", "1", "--prefill-chunk", "2048", "-r", "1",
                           "--output-file", str(out / "benchmark-report.json"), "--profile-measured"]
                profiler = ["/opt/rocm/bin/rocprofv3", "--selected-regions", "-f", "rocpd",
                            "-d", str(out / "rocprof-trace"), "--marker-trace", "--", *command]
                value = {
                    "artifact_type": "ninfer_whole_profile_plan", "schema_version": 2,
                    "status": "command_only_not_executed", "profile_kind": "trace",
                    "measured_region": "ninfer_bench_measured", "kernel_include_regex": None,
                    "counters": [], "artifact": route["artifact"],
                    "benchmark_executable": route["benchmark"],
                    "source_matrix": {"path": str(matrix), "sha256": "matrix",
                                      "preset": "pareto-whole",
                                      "report": {"path": str(report), "sha256": "report"}},
                    "workload": {"concurrency": 1, "prompt_tokens": 8192,
                                 "generated_tokens": 256, "spec": "mtp", "draft_tokens": 3,
                                 "dflash_verify_width": 0, "kv_value_group": 32,
                                 "xattention_profile": "b128-s16-tau900", "prefill_chunk": 2048},
                    "benchmark_command": command, "profiler_command": profiler,
                    "required_power_profile": {"value": "auto", "before_evidence": str(out / "before"),
                                               "after_evidence": str(out / "after")},
                }
                (out / "plan.json").write_text(json.dumps(value))
                (out / "commands.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + " ".join(profiler) + "\n")
                return value

            terminal = {
                "production_status": "selected_route_pending_shortlist_head_trace_and_niah",
                "shortlist_head_precision_status": "q4_round_gate_pass_pending_trace",
            }
            with patch("tools.bench.prepare_selected_mtp_shortlist_head.validate_terminal_production_authority",
                       return_value=(terminal, {})), patch(
                "tools.bench.prepare_selected_mtp_shortlist_head.resolve_route",
                return_value=route), patch(
                "tools.bench.prepare_selected_mtp_shortlist_head.prepare_whole_profile",
                side_effect=fake_profile,
            ):
                value = prepare(selection, output)
            self.assertEqual(value["workload"]["concurrency"], 1)
            self.assertEqual(value["expected_trace_database"],
                             str(output / "rocprof-trace/selected-mtp-shortlist-head_results.db"))
            self.assertIn("-o", value["profiler_command"])
            self.assertEqual(value["head_quantization"]["activation"]["codec"], "signed A8G64")
            self.assertTrue((output / "prepared.sha256").is_file())
            self.assertIn(
                "python3 -m tools.bench.finalize_selected_mtp_shortlist_head",
                (output / "postprocess.sh").read_text(encoding="utf-8"),
            )

    def test_module_entry_points_import_from_repo(self) -> None:
        static_prepare = (
            REPO / "profiles/rocprof/selected-mtp-shortlist-head-prepare-20260905/prepare.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m tools.bench.prepare_selected_mtp_shortlist_head", static_prepare)
        for module in ("tools.bench.prepare_selected_mtp_shortlist_head",
                       "tools.bench.finalize_selected_mtp_shortlist_head"):
            result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO,
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, (module, result.stderr))


if __name__ == "__main__":
    unittest.main()
