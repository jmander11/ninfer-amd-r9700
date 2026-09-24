#!/usr/bin/env python3
import math
import struct
import unittest
from unittest import mock

import analyze


class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = analyze.RESULTS.joinpath("fresh.state.bin").read_bytes()

    def test_complete_capture_passes(self):
        self.assertEqual(analyze.analyze()["classification"],
                         "layer0_recurrent_prefix_state_exact")

    def test_state_mismatch_rejected(self):
        changed = bytearray(self.state)
        changed[4] ^= 1
        with self.assertRaisesRegex(RuntimeError, "differs at 1"):
            analyze.compare_state(self.state, bytes(changed))

    def test_nonfinite_state_rejected(self):
        changed = bytearray(self.state)
        changed[:4] = struct.pack("<f", math.nan)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            analyze.compare_state(bytes(changed), self.state)

    def test_manifest_layer_mutation_rejected(self):
        original = analyze.load
        def changed(path):
            value = original(path)
            if path.name == "fresh.state.json":
                value["text_layer"] = 1
            return value
        with mock.patch.object(analyze, "load", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "manifest differs"):
                analyze.validate_manifest("fresh")


if __name__ == "__main__":
    unittest.main()
