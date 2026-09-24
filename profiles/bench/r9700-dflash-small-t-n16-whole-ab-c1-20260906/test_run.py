#!/usr/bin/env python3
"""Focused CPU tests for the whole-DFlash A/B report decision boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest

import run


PROFILE = {
    "kv_cache_format": "fp8-k-int4-v",
    "kv_value_group": 16,
    "q4_activation_bits": 8,
    "q4_prefill_cta_profile": "m64n128-pingpong-n16-k16-scalar-base-production",
    "w8_activation_bits": 8,
    "fp8_qk_wmma_enabled": True,
    "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
    "fp8_qk_wmma_t1_min_context": 64,
    "fp8_qk_wmma_t2_min_context": 320,
    "xattention_qualification": False,
}


def spec(draft: int, repeats: int = 1) -> dict:
    if draft == 0:
        return {"enabled": False, "draft_window": 0, "rounds": 0,
                "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
                "acceptance_rate": None, "acceptance_length": None,
                "accepted_per_position": []}
    rounds = 16 * repeats
    accepted = 32 * repeats
    drafted = rounds * draft
    return {"enabled": True, "draft_window": draft, "rounds": rounds,
            "drafted_tokens": drafted, "accepted_tokens": accepted,
            "fallback_steps": 0, "acceptance_rate": accepted / drafted,
            "acceptance_length": 1.0 + accepted / rounds,
            "accepted_per_position": [8 * repeats] * 4 + [0] * (draft - 4)}


def report(command: list[str], artifact: dict, role: str, draft: int, width: int) -> dict:
    ordinary = draft == 0
    tests = []
    for label, kind in (("pp128+tg64", "pp+tg"),
                        ("whole-pp128+tg64", "whole")):
        reps = []
        for sample in (1.0, 1.01, 0.99):
            rep_spec = spec(draft)
            engine_tokens = (run.DECODE_STEPS if draft == 0 else
                             rep_spec["rounds"] + rep_spec["accepted_tokens"] +
                             rep_spec["fallback_steps"])
            reps.append({"generated_output_tokens": run.REQUESTED_OUTPUT_TOKENS,
                         "decode_output_tokens": run.DECODE_STEPS,
                         "decode_engine_tokens": engine_tokens,
                         "generated_token_ids_by_lane": [list(range(run.REQUESTED_OUTPUT_TOKENS))],
                         "timings": {"prepare_seconds": 0.01, "vision_seconds": 0.0,
                                     "prefill_seconds": 0.5, "decode_seconds": sample,
                                     "total_seconds": sample + 0.5},
                         "speculative": spec(draft)})
        tests.append({"label": label, "kind": kind, "n_prompt": run.PROMPT_TOKENS,
                      "n_gen": run.DECODE_STEPS,
                      "requested_output_tokens": run.REQUESTED_OUTPUT_TOKENS,
                      "speculative": spec(draft, 3),
                      "reps": reps})
    return {"schema_version": 20, "artifact_type": "ninfer_bench_report",
            "tool": "ninfer_bench", "command": shlex.join(command),
            "environment": {"gpu_name": "AMD Radeon AI PRO R9700",
                            "architecture_name": "gfx1201", "device_id": 0},
            "artifact": {"path": artifact["path"],
                         "file_size_bytes": artifact["bytes"]},
            "load": {"target": "qwen3_8_27b_r9700", "weights_id": artifact["weights_id"]},
            "config": {**PROFILE, "dflash_small_t_candidate": role == "candidate",
                       "kv_plane_layouts": {"key": "token-fastest-head-major",
                                            "value": "feature-fastest-page-major",
                                            "value_scale": "feature-fastest-page-major"},
                       "concurrency": 1, "prefill_chunk": 4096,
                       "spec": "none" if ordinary else "dflash", "draft_tokens": draft,
                       "speculative_execution": not ordinary,
                       "dflash_verify_width_requested": 0 if ordinary else width,
                       "dflash_verify_width": 0 if ordinary else width,
                       "proposal_head": "full" if ordinary else "optimized",
                       "use_device_graph": True, "retain_token_ids": True,
                       "isolate_prompt_decode": True,
                       "decode_path": "device_graph" if ordinary else "dflash_device_graph",
                       "repetitions": 3, "warmup": 1},
            "tests": tests}


class ReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifact = {"path": "/model.ninfer", "bytes": 100,
                         "weights_id": "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval"}
        self.command = ["/bench", "--output-file", str(self.root / "report.json")]
        self.payload = report(self.command, self.artifact, "candidate", 4, 5)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, payload: dict) -> Path:
        path = self.root / "report.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_report_recomputes_output_and_spec_accounting(self) -> None:
        result = run.validate_report(self.write(self.payload), self.command, self.artifact,
                                     "candidate", 4, 5, PROFILE)
        self.assertEqual(len(result["decode"]["samples_seconds"]), 3)
        self.assertEqual(len(result["whole"]["tokens"]), run.REQUESTED_OUTPUT_TOKENS)

    def test_rejects_nonfinite_timing(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["tests"][0]["reps"][0]["timings"]["decode_seconds"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            run.validate_report(self.write(changed), self.command, self.artifact,
                                "candidate", 4, 5, PROFILE)

    def test_rejects_mutated_spec_accounting(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["tests"][1]["reps"][1]["speculative"]["accepted_tokens"] += 1
        with self.assertRaisesRegex(RuntimeError, "accounting"):
            run.validate_report(self.write(changed), self.command, self.artifact,
                                "candidate", 4, 5, PROFILE)

    def test_rejects_speculative_engine_work_disagreement(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["tests"][0]["reps"][0]["decode_engine_tokens"] += 1
        with self.assertRaisesRegex(RuntimeError, "decode_engine_tokens"):
            run.validate_report(self.write(changed), self.command, self.artifact,
                                "candidate", 4, 5, PROFILE)

    def test_rejects_ordinary_engine_work_disagreement(self) -> None:
        ordinary = report(self.command, self.artifact, "control", 0, 0)
        ordinary["tests"][1]["reps"][2]["decode_engine_tokens"] = run.DECODE_STEPS + 1
        with self.assertRaisesRegex(RuntimeError, "decode_engine_tokens"):
            run.validate_report(self.write(ordinary), self.command, self.artifact,
                                "control", 0, 0, PROFILE)

    def test_screen_requires_both_orders_and_conservative_upper(self) -> None:
        records = []
        for role, order, seconds in (("control", "control_first", 1.0),
                                     ("candidate", "control_first", 0.9),
                                     ("candidate", "candidate_first", 0.91),
                                     ("control", "candidate_first", 1.0)):
            records.append({"role": role, "order": order,
                            "validated": {"decode": {"samples_seconds": [seconds] * 3}}})
        self.assertTrue(run.cell_summary(records, "decode")["passed_initial_screen"])
        records[2]["validated"]["decode"]["samples_seconds"] = [1.01] * 3
        self.assertFalse(run.cell_summary(records, "decode")["passed_initial_screen"])


if __name__ == "__main__":
    unittest.main()
