#!/usr/bin/env python3
"""CPU contract test for the standalone real-weight FP8 prefix discriminator."""

import json
import subprocess
import sys
import unittest
from pathlib import Path


class Fp8LinearPrefixDiscriminatorTest(unittest.TestCase):
    def test_functional_output_is_prepoisoned_before_every_launch(self) -> None:
        source = (Path(__file__).parents[2] / "tools/r9700/fp8_linear_prefix_discriminator.hip").read_text()
        start = source.index("Run run(ninfer::ops::LinearExecution&")
        end = source.index("std::uint32_t steps", start)
        body = source[start:end]
        poison = body.index("hipMemset(output.value, 0xff, output.count * sizeof(hip_bfloat16))")
        launch = body.index("execution.run_qualified_algorithm")
        self.assertLess(poison, launch)
        self.assertIn("std::none_of(result.output.begin(), result.output.end()", body)

    def test_describe_is_dependency_light_and_exact(self) -> None:
        if len(sys.argv) < 2:
            self.skipTest("discriminator executable was not supplied")
        executable = Path(sys.argv.pop())
        value = json.loads(subprocess.check_output(
            [executable, "--describe"], text=True, stderr=subprocess.DEVNULL))
        selector = value.pop("common_algorithm_candidate_selector")
        self.assertIn(selector, (0, 1))
        self.assertEqual(value.pop("qualified_common_algorithm_fingerprint"),
                         "e2e00100000000000000000000000000")
        self.assertEqual(value.pop("qualified_selection"),
                         "preserve_t128_default_select_t129_viable")
        self.assertEqual(value.pop("oracle_rounding"), "direct_fp64_to_bf16_ties_even")
        self.assertEqual(value.pop("functional_output_prepoison"), "bf16_nan_full_scan")
        self.assertEqual(value, {
            "artifact_type": "ninfer_r9700_fp8_linear_prefix_discriminator",
            "schema_version": 2,
            "device": "gfx1201",
            "tokens": [128, 129],
            "shapes": [[34816, 5120], [4096, 5120]],
            "requires_real_production_fp8_artifact": True,
        })
        formatted = json.loads(subprocess.check_output(
            [executable, "--format-self-test"], text=True, stderr=subprocess.DEVNULL))
        cell = formatted["cell"]
        self.assertEqual(cell["weight_name"], "text/layers/0/mlp/gate_up")
        self.assertEqual([item["tokens"] for item in cell["profiles"]], [128, 129])
        self.assertTrue(cell["activation_code_prefix_exact"])
        self.assertIsNone(cell["first_output_mismatch"])
        self.assertEqual([item["tokens"] for item in cell["balanced_timings"]], [128, 129])
        self.assertTrue(all("viable_algorithms" in item for item in cell["profiles"]))
        self.assertEqual(cell["common_algorithm_trials"], [])
        rounding = json.loads(subprocess.check_output(
            [executable, "--rounding-self-test"], text=True, stderr=subprocess.DEVNULL))
        self.assertEqual(rounding, {
            "direct_midpoint_bits": 0x3F80,
            "direct_above_midpoint_bits": 0x3F81,
            "legacy_double_rounded_bits": 0x3F80,
        })


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
