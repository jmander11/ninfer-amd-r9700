#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

from tools.bench.validate_fp8_gate_up_comparison import OPERATIONS, SCHEMA, SHAPE, validate


def report(fp8: list[float], q4: list[float]) -> dict[str, object]:
    fp8_ordered, q4_ordered = sorted(fp8), sorted(q4)
    fp8_median = 0.5 * (fp8_ordered[6] + fp8_ordered[7])
    q4_median = 0.5 * (q4_ordered[6] + q4_ordered[7])
    ratio = q4_median / fp8_median
    return {
        "schema": SCHEMA,
        "artifact_type": "ninfer_r9700_fp8_gate_up_comparison",
        "schema_version": 2,
        "provenance": {
            "source_root": "/source", "source_sha256": "a" * 64,
            "executable_path": "/build/qual", "executable_sha256": "b" * 64,
        },
        "hardware": {
            "ordinal": 0,
            "device": "AMD Radeon AI PRO R9700",
            "pci_bus_id": "0000:03:00.0",
            "vendor_id": "0x1002",
            "device_id": "0x7551",
            "architecture": "gfx1201",
            "wave_size": 32,
            "compute_units": 64,
            "l2_bytes": 8388608,
            "vram_bytes": 34359738368,
        },
        "power_profile": {
            "path": "/sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level",
            "required": "auto",
            "before": "auto",
            "after": "auto",
        },
        "toolchain": {
            "hip_runtime_version": 70000000,
            "hip_driver_version": 70000000,
            "hipblaslt_version": 10401,
        },
        "shape": SHAPE,
        "fp8_profile": "E4M3-outer-vec32f-hipBLASLt-top-supported",
        "q4_control": "A8Q4G64-m64n128-pingpong-production",
        "selected_workspace_bytes": 1 << 20,
        "selected_waves_count": 1.0,
        "algorithm": {
            "heuristic_requested": 8,
            "heuristic_returned": 8,
            "selected_rank": 0,
            "opaque_data_hex": "0123456789abcdeffedcba9876543210",
            "selected_workspace_bytes": 1 << 20,
            "algorithm_max_workspace_bytes": 512 << 20,
            "waves_count": 1.0,
            "compute": "HIPBLAS_COMPUTE_32F",
            "scale_type": "HIP_R_32F",
            "a": "HIP_R_8F_E4M3-col[K,N]-ldK-opT-outer_vec32f",
            "b": "HIP_R_8F_E4M3-col[K,T]-ldK-outer_vec32f",
            "d": "HIP_R_16BF-col[N,T]-ldN",
        },
        "fp8_complete_median_ms": fp8_median,
        "q4_complete_median_ms": q4_median,
        "fp8_complete_ms_samples": fp8,
        "q4_complete_ms_samples": q4,
        "trial_order": "7 paired repetitions, each fp8,q4 then q4,fp8",
        "fp8_complete_tflops": OPERATIONS / (fp8_median * 1.0e9),
        "q4_complete_tflops_equivalent": OPERATIONS / (q4_median * 1.0e9),
        "fp8_over_q4_speedup": ratio,
        "current_q4_implementation_underperforms_fp8": ratio > 1.0,
        "current_q4_implementation_poor_vs_fp8": ratio > 1.0,
        "direct_weight_binding": True,
        "represented_format_axis_oracle": {
            "probe_count": 9, "tokens": [0, 17, 2047], "rows": [0, 12345, 34815],
            "fp8_max_bf16_steps": 1, "q4_max_bf16_steps": 2,
        },
        "outer_vector_scales": True,
        "nonfinite_status_poisoning": True,
        "q4_nonfinite_status_poisoning": True,
        "balanced_interleaved_pairs": 7,
        "no_clobber_rejection": True,
        "pass": True,
    }


class Fp8GateUpComparisonReportTest(unittest.TestCase):
    def test_classifies_both_matched_outcomes(self) -> None:
        slower_q4 = validate(report([10.0, 11.0] * 7, [20.0, 22.0] * 7))
        self.assertTrue(slower_q4["current_q4_implementation_underperforms_fp8"])
        self.assertEqual(slower_q4["fp8_over_q4_speedup"], 2.0)
        faster_q4 = validate(report([20.0, 22.0] * 7, [10.0, 11.0] * 7))
        self.assertFalse(faster_q4["current_q4_implementation_underperforms_fp8"])

    def test_rejects_inconsistent_ratio_and_failed_contract_check(self) -> None:
        value = report([10.0, 11.0] * 7, [20.0, 22.0] * 7)
        inconsistent = copy.deepcopy(value)
        inconsistent["fp8_over_q4_speedup"] = 1.0
        with self.assertRaisesRegex(ValueError, "timing samples"):
            validate(inconsistent)
        failed = copy.deepcopy(value)
        failed["no_clobber_rejection"] = False
        with self.assertRaisesRegex(ValueError, "admitted matched"):
            validate(failed)

    def test_rejects_wrong_device_power_and_algorithm_provenance(self) -> None:
        value = report([10.0, 11.0] * 7, [20.0, 22.0] * 7)
        wrong_device = copy.deepcopy(value)
        wrong_device["hardware"]["ordinal"] = 1
        with self.assertRaisesRegex(ValueError, "admitted matched"):
            validate(wrong_device)
        wrong_power = copy.deepcopy(value)
        wrong_power["power_profile"]["after"] = "profile_standard"
        with self.assertRaisesRegex(ValueError, "admitted matched"):
            validate(wrong_power)
        wrong_algorithm = copy.deepcopy(value)
        wrong_algorithm["algorithm"]["opaque_data_hex"] = "unknown"
        with self.assertRaisesRegex(ValueError, "admitted matched"):
            validate(wrong_algorithm)


if __name__ == "__main__":
    unittest.main()
