#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

from tools.bench.run_a8q4_t1_dot8_whole_ab import (
    ARTIFACT,
    ARTIFACT_SIZE,
    CORPUS,
    decision,
    extract,
    validate_operator_report,
)


def runs(decode: float, prefill: float = 30.0) -> list[dict]:
    return [{"decode_seconds": decode, "prefill_seconds": prefill,
             "token_ids": [[1, 2, 3]], "environment": {"gpu": "R9700"},
             "artifact": {"path": "/artifact", "file_size_bytes": 1},
             "workspace": {"capacity_bytes": 2, "peak_used_bytes": 2},
             "config": {"max_context": 8448, "prefill_chunk": 4096,
                        "kv_cache_format": "fp8-k-int4-v", "concurrency": 1,
                        "spec": "none", "draft_tokens": 0,
                        "use_device_graph": True, "decode_path": "device_graph",
                        "corpus_path": "/corpus", "corpus_tokens": 65536}}
            for _ in range(8)]


def benchmark_report(decode_tokens: int = 32) -> dict:
    return {
        "schema_version": 20,
        "artifact_type": "ninfer_bench_report",
        "environment": {
            "gpu_name": "AMD Radeon AI PRO R9700",
            "architecture_name": "gfx1201",
            "hip_runtime_version": "7.15.26333",
            "hip_driver_version": "7.15.26333",
            "device_id": 0,
        },
        "artifact": {"path": str(ARTIFACT.resolve()), "file_size_bytes": ARTIFACT_SIZE},
        "memory": {"kv_capacity_mode": "explicit", "workspace": {"capacity_bytes": 2}},
        "config": {
            "concurrency": 1, "spec": "none", "draft_tokens": 0,
            "speculative_execution": False, "dflash_verify_width_requested": 0,
            "dflash_verify_width": 0, "proposal_head": "full",
            "use_device_graph": True, "retain_token_ids": True,
            "prefill_chunk": 4096, "max_context": 8192 + decode_tokens,
            "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 16,
            "kv_plane_layouts": {"key": "token-fastest-head-major",
                                 "value": "feature-fastest-page-major",
                                 "value_scale": "feature-fastest-page-major"},
            "q4_activation_bits": 8,
            "q4_prefill_cta_profile":
                "m64n128-pingpong-n16-k16-scalar-base-production",
            "w8_activation_bits": 8, "xattention_qualification": False,
            "fp8_qk_wmma_enabled": True,
            "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
            "fp8_qk_wmma_t1_min_context": 64,
            "fp8_qk_wmma_t2_min_context": 320,
            "decode_path": "device_graph", "repetitions": 1, "warmup": 1,
            "corpus_path": str(CORPUS.resolve()), "corpus_tokens": 65536,
        },
        "tests": [{
            "kind": "whole", "n_prompt": 8192, "n_gen": decode_tokens,
            "prefill_seconds_mean": 4.0, "decode_seconds_mean": 2.0,
            "reps": [{"generated_output_tokens": decode_tokens + 1,
                      "decode_output_tokens": decode_tokens,
                      "generated_token_ids_by_lane": [list(range(decode_tokens + 1))]}],
        }],
    }


def operator_report() -> dict:
    shapes = [(4096, 5120, 48), (5120, 6144, 64), (5120, 17408, 64),
              (7168, 5120, 32), (12288, 5120, 48), (34816, 5120, 64),
              (248320, 5120, 1)]
    return {
        "status": "passed",
        "correctness": {"passed": True},
        "static_and_resources": {"runtime_resources_passed": True,
                                 "required_opcode": "v_dot8_i32_iu4",
                                 "required_opcode_count": 16,
                                 "wmma_forbidden": True},
        "performance_decision": {"accepted": True, "every_cell_passed": True,
                                 "aggregate_passed": True},
        "shapes": [{"rows": n, "columns": k, "exact_decode_calls": calls}
                   for n, k, calls in shapes],
    }


class Dot8WholeAbTest(unittest.TestCase):
    def test_accepts_stable_material_decode_saving_and_unchanged_prefill(self) -> None:
        result = decision(runs(16.0), runs(14.0))
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["decode_saving_lower_ms_per_token"], 7.8125)
        self.assertEqual(result["candidate_over_control_prefill_robust_upper"], 1.0)

    def test_rejects_decode_regression_or_prefill_change(self) -> None:
        self.assertFalse(decision(runs(16.0), runs(16.1))["accepted"])
        self.assertFalse(decision(runs(16.0), runs(14.0, 31.0))["accepted"])

    def test_one_pair_short_decode_is_screen_only(self) -> None:
        result = decision(runs(16.0)[:2], runs(14.0)[:2], pairs=1, decode_tokens=32)
        self.assertTrue(result["performance_passed"])
        self.assertFalse(result["promotion_eligible_full_gate"])
        self.assertFalse(result["accepted"])

    def test_rejects_token_or_workspace_mismatch(self) -> None:
        candidate = runs(14.0)
        candidate[7] = copy.deepcopy(candidate[7])
        candidate[7]["token_ids"] = [[9]]
        with self.assertRaisesRegex(ValueError, "parity failed"):
            decision(runs(16.0), candidate)
        candidate = runs(14.0)
        candidate[0] = copy.deepcopy(candidate[0])
        candidate[0]["workspace"]["capacity_bytes"] = 3
        with self.assertRaisesRegex(ValueError, "parity failed"):
            decision(runs(16.0), candidate)

    def test_extract_accepts_only_complete_fixed_report(self) -> None:
        self.assertEqual(extract(benchmark_report(), 32)["generated_token_count"], 33)
        for path, wrong in [
                (("config", "w8_activation_bits"), 4),
                (("config", "kv_plane_layouts"), {}),
                (("config", "corpus_tokens"), 1),
                (("artifact", "file_size_bytes"), 1)]:
            report = benchmark_report()
            report[path[0]][path[1]] = wrong
            with self.assertRaisesRegex(ValueError, "fixed C1 ordinary graph case"):
                extract(report, 32)

    def test_extract_rejects_wrong_token_count(self) -> None:
        report = benchmark_report()
        report["tests"][0]["reps"][0]["generated_token_ids_by_lane"][0].pop()
        with self.assertRaisesRegex(ValueError, "retained generated-token lane"):
            extract(report, 32)

    def test_operator_authority_validation_is_fail_closed(self) -> None:
        validate_operator_report(operator_report())
        report = operator_report()
        report["shapes"][6]["exact_decode_calls"] = 2
        with self.assertRaisesRegex(ValueError, "exact accepted seven-cell"):
            validate_operator_report(report)


if __name__ == "__main__":
    unittest.main()
