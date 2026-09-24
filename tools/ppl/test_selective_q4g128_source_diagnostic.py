from argparse import Namespace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.convert.qwen3_8_27b_r9700.screen_selective_q4g128_mse_quality import (
    _mse_q4g128_decode,
)
from tools.ppl import selective_q4g128_source_diagnostic as diagnostic
from tools.ppl.q4_group_source_diagnostic import quantize_decode_q4


class SelectiveQ4G128SourceDiagnosticTest(unittest.TestCase):
    def test_exact_logical_and_raw_scope(self) -> None:
        self.assertEqual(len(diagnostic.SELECTED_SPECS), 160)
        self.assertEqual(len(diagnostic.SELECTED_SOURCE_ROWS), 208)
        self.assertEqual(
            sum(end - begin for spans in diagnostic.SELECTED_SOURCE_ROWS.values()
                for begin, end in spans),
            1_163_264,
        )
        self.assertEqual(
            diagnostic._scope()["selected_roles"],
            {"gdn_output": 48, "gdn_value_z": 48, "mlp_down": 64},
        )

    def test_selected_rows_use_selected_codec_and_remaining_rows_stay_bf16(self) -> None:
        torch.manual_seed(19)
        source = (torch.randn(5, 256) * 0.2).to(torch.bfloat16)
        checkpoint = diagnostic.SelectiveQ4Checkpoint(None, "q4g128-mse")
        with patch.object(diagnostic, "SELECTED_SOURCE_ROWS", {"matrix": ((1, 4),)}):
            actual = checkpoint._represented("matrix", source)
        expected = source.clone()
        expected[1:4] = _mse_q4g128_decode(source[1:4]).to(torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[[0, 4]], source[[0, 4]]))

        control = diagnostic.SelectiveQ4Checkpoint(None, "q4g64-absmax")
        with patch.object(diagnostic, "SELECTED_SOURCE_ROWS", {"matrix": ((1, 4),)}):
            actual_control = control._represented("matrix", source)
        expected_control = source.clone()
        expected_control[1:4] = quantize_decode_q4(source[1:4], 64)
        self.assertTrue(torch.equal(actual_control, expected_control))

    def test_nonselected_matrix_and_nonmatrix_boundaries(self) -> None:
        matrix = torch.arange(256, dtype=torch.float32).reshape(1, 256).to(torch.bfloat16)
        vector = torch.arange(8, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(
            diagnostic.SelectiveQ4Checkpoint(None, "q4g128-mse")._represented("other", matrix),
            matrix,
        ))
        self.assertIs(
            diagnostic.SelectiveQ4Checkpoint(None, "q4g128-mse")._represented("other", vector), vector
        )

    def test_mse_g128_row_chunk_is_exactly_row_independent(self) -> None:
        torch.manual_seed(23)
        source = (torch.randn(9, 256) * 0.1).to(torch.bfloat16)
        whole = _mse_q4g128_decode(source).to(torch.bfloat16)
        self.assertTrue(torch.equal(
            diagnostic._mse_q4g128_bf16(source, row_chunk=2), whole
        ))
        with self.assertRaises(ValueError):
            diagnostic._mse_q4g128_bf16(source, row_chunk=0)

    def test_gate_is_direct_capacity_speed_against_bf16(self) -> None:
        bf16 = [1.0, 1.0, 11.0, 1.0]
        control = [1.01, 1.01, 11.0, 1.01]
        candidate = [1.02, 1.02, 9.0, 1.02]
        gate = diagnostic._gate(candidate, [1, 2, 3, 4], bf16, [1, 2, 3, 4])
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["quality_tier"], "capacity-speed")
        self.assertAlmostEqual(gate["maximum_mean_nll_delta"], math.log(1.05))
        self.assertIn("against_bf16", gate)

    def test_current_retained_bf16_authority_reopens(self) -> None:
        bf16 = Path(
            "profiles/ppl/bf16-reference-deterministic-pv-gdn-v3-a-20260904/"
            "8192.prefill.bf16-reference.json"
        )
        report = json.loads(bf16.read_text())
        source = {
            "config_sha256": report["source_config_sha256"],
            "index_sha256": report["source_index_sha256"],
            "shards_sha256": report["source_shards_sha256"],
            "corpus_ids_sha256": report["corpus_ids_sha256"],
        }
        args = Namespace(bf16=bf16, device=0)
        values = diagnostic._load_bf16_authority(args, source)
        self.assertEqual(len(values[1]), 4095)


if __name__ == "__main__":
    unittest.main()
