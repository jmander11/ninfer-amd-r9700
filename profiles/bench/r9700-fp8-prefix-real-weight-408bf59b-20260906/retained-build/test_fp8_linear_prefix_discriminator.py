#!/usr/bin/env python3
"""CPU contract test for the standalone real-weight FP8 prefix discriminator."""

import json
import subprocess
import sys
import unittest
from pathlib import Path


class Fp8LinearPrefixDiscriminatorTest(unittest.TestCase):
    def test_describe_is_dependency_light_and_exact(self) -> None:
        if len(sys.argv) < 2:
            self.skipTest("discriminator executable was not supplied")
        executable = Path(sys.argv.pop())
        value = json.loads(subprocess.check_output(
            [executable, "--describe"], text=True, stderr=subprocess.DEVNULL))
        self.assertEqual(value, {
            "artifact_type": "ninfer_r9700_fp8_linear_prefix_discriminator",
            "schema_version": 1,
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


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
