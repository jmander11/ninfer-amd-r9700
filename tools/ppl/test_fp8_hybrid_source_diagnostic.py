from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tools.convert.qwen3_8_27b_r9700 import fp8_hybrid_decision
from tools.convert.qwen3_8_27b_r9700 import fp8_hybrid_inventory
from tools.convert.qwen3_8_27b_r9700 import source_recipe
from tools.convert.qwen3.common.recipe import expression_sources
from tools.convert.qwen3_8_27b_r9700.e4m3_rowwise import (
    decode_e4m3_rowwise, encode_e4m3_rowwise,
)
from tools.ppl.compare_fp8_hybrid_source import compare
from tools.ppl.fp8_hybrid_source_diagnostic import (
    Fp8HybridCheckpoint, Q4_SOURCE_ROWS, SELECTED_SOURCE_ROWS, TEXT_SPECS, _matrix_scope,
    quantize_decode_e4m3,
)
from tools.ppl.q4_group_source_diagnostic import quantize_decode_q4


class Fp8HybridSourceDiagnosticTest(unittest.TestCase):
    def test_authority_derives_exact_fused_source_row_ownership(self) -> None:
        self.assertEqual(len(fp8_hybrid_decision.DECISION.matrix_names), 144)
        scope = _matrix_scope()
        self.assertEqual(scope["logical_e4m3_matrix_count"], 144)
        self.assertEqual(scope["logical_q4g64_matrix_count"], 178)
        self.assertEqual(scope["logical_direct_bf16_matrix_count"], 144)
        self.assertEqual(scope["selected_raw_source_tensor_count"], 224)
        self.assertEqual(fp8_hybrid_inventory.FORMAT_COUNTS["F8E4M3_ROW_F32S"], 144)
        self.assertEqual(fp8_hybrid_inventory.FORMAT_COUNTS["Q4G64_F16S"], 295)

        prefix = "model.language_model.layers.3.self_attn."
        self.assertEqual(SELECTED_SOURCE_ROWS[prefix + "q_proj.weight"], ((0, 12288),))
        self.assertEqual(SELECTED_SOURCE_ROWS[prefix + "k_proj.weight"], ((0, 1024),))
        self.assertEqual(SELECTED_SOURCE_ROWS[prefix + "v_proj.weight"], ((0, 1024),))
        gdn = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        self.assertEqual(SELECTED_SOURCE_ROWS[gdn], ((0, 4096),))
        self.assertNotIn(
            "model.language_model.layers.0.mlp.down_proj.weight", SELECTED_SOURCE_ROWS,
        )

    def test_e4m3_decode_matches_canonical_converter_codec(self) -> None:
        torch.manual_seed(1201)
        source = (torch.randn(7, 256) * 0.025).to(torch.bfloat16)
        expected = decode_e4m3_rowwise(
            encode_e4m3_rowwise(source), *source.shape,
        ).to(torch.bfloat16)
        actual = quantize_decode_e4m3(source, row_chunk=3)
        self.assertTrue(torch.equal(actual, expected))

    def test_partial_gdn_source_uses_e4m3_then_q4(self) -> None:
        torch.manual_seed(64)
        name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        # Temporarily use a small interval to exercise the same mixed-row path without
        # allocating the real 10,240 x 5,120 source matrix.
        source = (torch.randn(6, 128) * 0.02).to(torch.bfloat16)
        with patch.dict(SELECTED_SOURCE_ROWS, {name: ((0, 2),)}), patch.dict(
            Q4_SOURCE_ROWS, {name: ((2, 6),)}
        ):
            actual = Fp8HybridCheckpoint._represented(name, source)
        expected = torch.empty_like(source)
        expected[:2] = quantize_decode_e4m3(source[:2])
        expected[2:] = quantize_decode_q4(source[2:], 64)
        self.assertTrue(torch.equal(actual, expected))

    def test_all_retained_rank_two_bf16_roles_remain_exact(self) -> None:
        direct_specs = tuple(
            spec for spec in TEXT_SPECS
            if len(spec.shape) == 2 and spec.format == "BF16"
        )
        self.assertEqual(len(direct_specs), 144)
        raw_names = {
            source.name
            for spec in direct_specs
            for source in expression_sources(source_recipe.RECIPES_BY_NAME[spec.name].expression)
        }
        self.assertEqual(len(raw_names), 144)
        self.assertFalse(raw_names & SELECTED_SOURCE_ROWS.keys())
        self.assertFalse(raw_names & Q4_SOURCE_ROWS.keys())
        for name in (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        ):
            source = torch.randn(3, 128).to(torch.bfloat16)
            self.assertIs(Fp8HybridCheckpoint._represented(name, source), source)
        convolution_name = "model.language_model.layers.0.linear_attn.conv1d.weight"
        convolution = torch.randn(10240, 1, 4).to(torch.bfloat16)
        self.assertIs(
            Fp8HybridCheckpoint._represented(convolution_name, convolution), convolution,
        )

    def test_comparison_binds_identity_source_and_capacity_speed_gate(self) -> None:
        decision = fp8_hybrid_decision.DECISION
        source = {
            "config_sha256": "a", "index_sha256": "b",
            "shards_sha256": {"s": "c"}, "corpus_ids_sha256": "d",
        }
        hybrid = {
            "identity": {
                "weights_id": decision.weights_id, "recipe_id": decision.recipe_id,
                "selection_sha256": decision.selection_sha256,
            },
            "source": source,
            "workload": {"tokens": 8192, "skip": "half", "prefill_chunk": 4096,
                         "schedule": "prefill", "device": 0},
        }
        bf16 = {
            "source_config_sha256": "a", "source_index_sha256": "b",
            "source_shards_sha256": {"s": "c"}, "corpus_ids_sha256": "d",
            "prompt_tokens": 8192, "skip_tokens": 4096, "schedule": "prefill",
        }
        with patch(
            "tools.ppl.compare_fp8_hybrid_source._load_hybrid",
            return_value=(hybrid, [1.01, 2.0], [3, 5]),
        ), patch(
            "tools.ppl.compare_fp8_hybrid_source._load_bf16",
            return_value=(bf16, [1.0, 2.0], [3, 4]),
        ), patch(
            "tools.ppl.compare_fp8_hybrid_source.sha256_file", return_value="0" * 64,
        ):
            report = compare(Path("hybrid.json"), Path("bf16.json"))
        self.assertEqual(report["identity"]["selection_sha256"], decision.selection_sha256)
        self.assertTrue(report["pass"])
        self.assertEqual(report["against_bf16"]["argmax_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
