#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bench.analyze_fp8_gate_up_pmc import validate_benchmark_report, validate_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "profiles/rocprof/r9700-selected-fp8-gate-up-p2048-pmc-20260905/plan.json"


class Fp8GateUpPmcTest(unittest.TestCase):
    def test_prepared_postprocessor_uses_repo_module_invocation(self):
        script = (PLAN.parent / "postprocess.sh").read_text(encoding="utf-8")
        self.assertIn("python3 -m tools.bench.analyze_fp8_gate_up_pmc", script)
        result = subprocess.run(
            [sys.executable, "-m", "tools.bench.analyze_fp8_gate_up_pmc", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_current_plan_reopens_all_bound_identity_without_hashing_large_artifact(self):
        value = json.loads(PLAN.read_text())
        expected = {Path(item["path"]).resolve(): item["sha256"]
                    for item in value["inputs"].values()}
        expected[PLAN.resolve()] = "unused"
        with mock.patch("tools.bench.analyze_fp8_gate_up_pmc.sha",
                        side_effect=lambda path: expected[Path(path).resolve()]):
            loaded, kernel = validate_plan(PLAN)
        self.assertEqual(loaded["selected_algorithm"]["solution_index"], 123104)
        self.assertEqual(
            loaded["resource_contract"]["profiler_dispatch_metadata"]["vgpr_count"], 188)
        self.assertEqual(
            loaded["resource_contract"]["captured_elf_resources"]["architectural_vgpr_count"],
            192)
        self.assertEqual(
            loaded["resource_contract"]["captured_elf_resources"]["accumulator_vgpr_count"],
            0)
        self.assertIn("MT128x128x32", kernel)
        self.assertFalse(kernel.endswith(".kd"))

    def test_benchmark_report_is_exact_and_rejects_concurrency_drift(self):
        plan = json.loads(PLAN.read_text())
        command = plan["benchmark_commands"]["cache_wait"]
        report = {"schema_version": 20, "artifact_type": "ninfer_bench_report",
            "tool": "ninfer_bench", "command": command,
            "environment": {"gpu_name": "AMD Radeon AI PRO R9700", "architecture_name": "gfx1201"},
            "artifact": {"path": plan["inputs"]["artifact"]["path"]},
            "load": {"target": "qwen3_8_27b_r9700",
                     "weights_id": "r9700-q4g64-f8e4m3-four-role-n16k16-eval"},
            "config": {"max_context": 2048, "prefill_chunk": 4096, "kv_value_group": 16,
                "concurrency": 1, "q4_activation_bits": 8, "w8_activation_bits": 8,
                "q4_prefill_cta_profile": "m64n128-pingpong-production",
                "fp8_qk_wmma_enabled": True,
                "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                "fp8_qk_wmma_t1_min_context": 64, "fp8_qk_wmma_t2_min_context": 320,
                "xattention_qualification": False, "spec": "none", "draft_tokens": 0,
                "speculative_execution": False, "proposal_head": "full", "repetitions": 1,
                "warmup": 1, "corpus_path": plan["inputs"]["corpus"]["path"]},
            "tests": [{"label": "pp2048", "kind": "pp", "n_prompt": 2048,
                       "n_gen": 0, "prefill_seconds_mean": 1.0}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(validate_benchmark_report(path, plan, "cache_wait")["prefill_seconds"], 1.0)
            report["config"]["concurrency"] = 2
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "exact workload"):
                validate_benchmark_report(path, plan, "cache_wait")

if __name__ == "__main__": unittest.main()
