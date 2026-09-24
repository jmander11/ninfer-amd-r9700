"""Dependency-light tests for the sampled full-span GDN probe."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from . import gdn_full_span_probe as probe


class GdnFullSpanProbeTest(unittest.TestCase):
    def test_geometry_is_the_two_production_prefill_spans(self) -> None:
        self.assertEqual(probe.ROW_EXTENTS, (4095, 4096))
        self.assertEqual(probe.sampled_rows(4095), (0, 1, 2047, 4093, 4094))
        self.assertEqual(probe.sampled_rows(4096), (0, 1, 2048, 4094, 4095))
        self.assertEqual(len(set(probe.SAMPLE_COLUMNS)), len(probe.SAMPLE_COLUMNS))
        self.assertTrue(all(0 <= head < 48 and 0 <= feature < 128
                            for head, feature in probe.SAMPLE_COLUMNS))

    def test_fp64_column_oracle_applies_decay_prediction_update_and_output(self) -> None:
        outputs, final = probe.sampled_column_fp64(
            q_rows=((0.0, 1.0), (1.0, 1.0)),
            k_rows=((1.0, 0.0), (0.0, 1.0)),
            values=(3.0, 1.0),
            decays=(math.log(0.5), 0.0),
            betas=(0.5, 1.0),
            initial_column=(1.0, 2.0),
            output_rows=(0, 1),
            scale=2.0,
        )
        self.assertEqual(outputs, {0: 2.0, 1: 5.5})
        self.assertEqual(final, (1.75, 1.0))

    def test_fp64_prediction_and_output_reduce_all_128_key_features(self) -> None:
        ones = tuple(1.0 for _ in range(probe.KEY_DIM))
        outputs, final = probe.sampled_column_fp64(
            q_rows=(ones,), k_rows=(ones,), values=(0.0,), decays=(0.0,),
            betas=(0.0,), initial_column=ones, output_rows=(0,), scale=1.0,
        )
        self.assertEqual(outputs[0], 128.0)
        self.assertEqual(final, ones)

    def test_oracle_rejects_malformed_operands_and_rows(self) -> None:
        with self.assertRaises(ValueError):
            probe.sampled_column_fp64(
                ((1.0,),), (), (1.0,), (0.0,), (1.0,), (0.0,), (0,), scale=1.0
            )
        with self.assertRaises(ValueError):
            probe.sampled_column_fp64(
                ((1.0,),), ((1.0,),), (1.0,), (0.0,), (1.0,), (0.0,),
                (1,), scale=1.0,
            )

    def test_atomic_report_and_cli_contract(self) -> None:
        options = probe.build_parser().parse_args(["--device", "2", "--out-json", "x.json"])
        self.assertEqual(options.device, 2)
        self.assertEqual(options.out_json, Path("x.json"))
        payload = {
            "artifact_type": probe.SCHEMA,
            "schema_version": probe.SCHEMA_VERSION,
            "quality_evidence": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "report.json"
            probe.write_report(path, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(path.parent.glob("*.tmp-*")), [])
        with self.assertRaises(ValueError):
            probe.write_report(Path("ignored.json"), {"artifact_type": "wrong"})


if __name__ == "__main__":
    unittest.main()
