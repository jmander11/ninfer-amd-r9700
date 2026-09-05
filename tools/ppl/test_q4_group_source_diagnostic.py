import json
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import torch

from tools.ppl.q4_group_source_diagnostic import (
    GROUPS, _atomic_new, preflight_payload, quantize_decode_q4,
)
from tools.ppl.compare_q4_group_source import _against, _load_bf16, _source_key
from tools.convert.qwen3_8_27b_r9700.q4_row_scaled import quantize_dequantize


class Q4GroupSourceDiagnosticTest(unittest.TestCase):
    def test_q4_decode_matches_existing_cpu_codec_math(self) -> None:
        torch.manual_seed(17)
        source = (torch.randn(7, 256) * 0.03).to(torch.bfloat16)
        for group in GROUPS:
            expected = quantize_dequantize(source, group_size=group).to(torch.bfloat16)
            actual = quantize_decode_q4(source, group, row_chunk=3)
            self.assertTrue(torch.equal(actual, expected))

    def test_zero_underflow_and_chunking_are_stable(self) -> None:
        source = torch.zeros(5, 128, dtype=torch.bfloat16)
        source[1, 0] = torch.tensor(2.0**-126, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(
            quantize_decode_q4(source, 128, row_chunk=1),
            quantize_decode_q4(source, 128, row_chunk=5),
        ))

    def test_invalid_geometry_and_bool_group_fail(self) -> None:
        source = torch.zeros(1, 128, dtype=torch.bfloat16)
        for group in (True, 0, 256):
            with self.assertRaises(ValueError):
                quantize_decode_q4(source, group)
        with self.assertRaises(ValueError):
            quantize_decode_q4(source[:, :64], 64)

    def test_no_clobber_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            _atomic_new(path, b"first")
            with self.assertRaises(FileExistsError):
                _atomic_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_comparison_derives_quality_and_exact_token_diagnostics(self) -> None:
        result = _against([1.0, 11.0], [1.01, 9.0], [3, 4], [3, 5])
        self.assertTrue(result["mean_nll_delta_pass"])
        self.assertEqual(result["repaired_severe_positions"], 1)
        self.assertEqual(result["new_severe_positions"], 0)
        self.assertEqual(result["argmax_mismatches"], 1)
        self.assertFalse(result["argmax_identity_is_gate"])

    def test_bf16_raw_authority_uses_enclosing_campaign_sidecar_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "8192.prefill.bf16-reference.json"
            nll = struct.pack("<ff", 1.0, 2.0)
            argmax = struct.pack("<ii", 3, 4)
            path.with_suffix(".nllf32").write_bytes(nll)
            path.with_suffix(".argmaxi32").write_bytes(argmax)
            common = {
                "weights_id": "bf16-source",
                "formula_profile": "checkpoint-direct-qwen3.8-source-bf16",
                "schedule": "prefill", "prompt_tokens": 4, "skip_tokens": 2,
                "tokens_scored": 2, "argmax_tokens": 2,
                "source_config_sha256": "a", "source_index_sha256": "b",
                "source_shards_sha256": {"shard": "c"}, "corpus_ids_sha256": "d",
            }
            path.write_text(json.dumps(common))
            cell = dict(common)
            cell.update({
                "scheme": "bf16-reference",
                "nll_sha256": hashlib.sha256(nll).hexdigest(),
                "argmax_sha256": hashlib.sha256(argmax).hexdigest(),
            })
            (root / "results.json").write_text(json.dumps({"cells": [cell]}))
            report, nlls, argmax_values = _load_bf16(path)
            self.assertEqual(report, common)
            self.assertEqual(nlls, [1.0, 2.0])
            self.assertEqual(argmax_values, [3, 4])

    def test_source_key_normalizes_shard_list_and_mapping(self) -> None:
        common = {"config_sha256": "a", "index_sha256": "b",
                  "corpus_ids_sha256": "c"}
        mapping = {"source": {**common, "shards_sha256": {"s1": "h1", "s2": "h2"}}}
        pairs = {"source": {**common, "shards_sha256": [["s2", "h2"], ["s1", "h1"]]}}
        self.assertEqual(_source_key(mapping), _source_key(pairs))

    def test_preflight_models_all_raw_text_matrices(self) -> None:
        class Args:
            group = 128
            weights = Path("/source")
            device = 0

        # Avoid source hashing here; the real-source preflight exercises it.
        from unittest.mock import patch
        identity = {"shard_count": 18}
        with patch("tools.ppl.q4_group_source_diagnostic._source_identity",
                   return_value=identity), patch(
                       "tools.ppl.q4_group_source_diagnostic.validate_source_metadata"
                   ):
            result = preflight_payload(Args(), {}, [0] * 8192)
        self.assertEqual(result["matrix_scope"]["raw_text_matrix_count"], 498)
        self.assertEqual(result["matrix_scope"]["logical_product_q4_matrix_count"], 322)
        self.assertGreater(result["matrix_scope"]["represented_scale_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
