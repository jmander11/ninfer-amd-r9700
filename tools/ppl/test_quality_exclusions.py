"""Measured numerical exclusions are distinct from corrupt or incomplete evidence."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from tools.ppl import assemble_pareto as assembly, pareto, run
from tools.ppl.test_pareto import migration_receipt


WEIGHTS = "r9700-q4-w8-mse-n16k16-eval"


def campaign_fixture(root: Path, weights=WEIGHTS, mean_gate=None):
    tier_name = "accuracy" if weights == WEIGHTS else "capacity-speed"
    tier = run.QUALITY_TIERS[tier_name]
    gates = {f"r9700-g{g}": tier["maximum_mean_nll_delta"] if mean_gate is None else mean_gate
             for g in (16, 32)}
    reference_path = root / "reference.json"
    reference_path.write_text("{}")
    repeat = {"path": str(root / "repeat.json"), "sha256": "a" * 64}
    reference = {"model_id": "qwen3.8-27b", "reference_weights_id": "bf16",
        "reference_source": {}, "reference_execution": {},
        "corpus": {"path": str(root / "corpus.ids")}, "lengths": [8192, 32768],
        "skip": "half", "prefill_chunk": 2048, "schedules": ["prefill"],
        "spec": "none", "draft_tokens": 0, "terrible_nll": 10.,
        "scorers": {run.BASELINE: {"path": "/bf16-scorer"}}}
    campaign = {**reference, "artifact_type": run.CAMPAIGN_ARTIFACT_TYPE,
        "schema_version": run.CAMPAIGN_SCHEMA_VERSION, "q4_activation_bits": 8,
        "w8_activation_bits": 8, "decode_attention_profile": run.DECODE_ATTENTION_PROFILE,
        "xattention_profile": "b128-s16-tau900",
        "candidate_kv_plane_layouts": run.R9700_KV_PLANE_LAYOUTS,
        "candidate_artifact": {"weights_id": weights, "sha256": "b" * 64,
            "bytes": 123, "conversion_receipt": migration_receipt(weights)},
        "reused_bf16_campaign": {"path": str(reference_path), "sha256": run.file_sha256(reference_path)},
        "bf16_repeat_comparison": repeat, "quality_tier": tier_name, "gates": gates,
        "quality_gate_contract": {"complete_finite_aligned_sidecars": True,
            "maximum_mean_nll_delta_by_profile": gates, "severe_nll_threshold": 10.,
            "tier": tier_name, "tier_maximum_mean_nll_delta": tier["maximum_mean_nll_delta"],
            "maximum_new_severe_rate": tier["maximum_new_severe_rate"],
            "minimum_new_severe_budget": tier["minimum_new_severe_budget"]}, "cells": [], "pass": False}
    refs = {}
    for tokens in (8192, 32768):
        n = tokens // 2 - 1
        base = {key: None for key in run.BF16_SCORER_REPORT_FIELDS}
        base.update({"scheme": run.BASELINE, "prompt_tokens": tokens, "skip_tokens": tokens // 2,
            "tokens_scored": n, "argmax_tokens": n, "non_finite": 0, "terrible_tokens": 0,
            "terrible_nll": 10., "sum_nll": float(n), "mean_nll": 1., "max_nll": 1.,
            "ppl": math.e, "pass": True})
        path = root / f"bf16-{tokens}.json"
        path.write_text(json.dumps(base))
        path.with_suffix(".nllf32").write_bytes(struct.pack(f"<{n}f", *([1.] * n)))
        path.with_suffix(".argmaxi32").write_bytes(struct.pack(f"<{n}i", *([0] * n)))
        refs[tokens] = (base, path)
        campaign["cells"].append(copy.deepcopy(base))
        for group in (16, 32):
            severe = (19 if group == 16 else 18) if tokens == 32768 else 3
            nll = [10.] * severe + [1.] * (n - severe)
            cell = {"scheme": f"r9700-g{group}", "schedule": "prefill",
                "model_id": "qwen3.8-27b", "weights_id": weights,
                "kv_format": "fp8-k-int4-v", "kv_value_group": group,
                "kv_plane_layouts": run.R9700_KV_PLANE_LAYOUTS,
                "q4_activation_bits": 8, "w8_activation_bits": 8,
                "split512_enabled": True, "decode_attention_profile": run.DECODE_ATTENTION_PROFILE,
                "xattention_qualification": True, "xattention_profile": "b128-s16-tau900",
                "xattention_find_block": 128, "xattention_stride": 16, "xattention_tau_permille": 900,
                "prefill_chunk": 2048, "prompt_tokens": tokens, "skip_tokens": tokens // 2,
                "tokens_scored": n, "argmax_tokens": n, "non_finite": 0,
                "terrible_tokens": severe, "terrible_nll": 10., "sum_nll": math.fsum(nll),
                "mean_nll": math.fsum(nll) / n, "max_nll": 10., "ppl": math.exp(math.fsum(nll) / n)}
            path = root / f"{tokens}-g{group}.json"
            path.write_text(json.dumps(cell))
            path.with_suffix(".nllf32").write_bytes(struct.pack(f"<{n}f", *nll))
            path.with_suffix(".argmaxi32").write_bytes(struct.pack(f"<{n}i", *([0] * n)))
            cell.update({"command": ["ppl", "--ids", reference["corpus"]["path"],
                "--tokens", str(tokens), "--skip", "half", "--prefill-chunk", "2048",
                "--out-json", str(path)], "nll_sha256": run.file_sha256(path.with_suffix(".nllf32")),
                "argmax_sha256": run.file_sha256(path.with_suffix(".argmaxi32")), "quality_tier": tier_name})
            run.apply_baseline(cell, nll, 1., gates, cell["scheme"], [1.] * n, [0] * n, [0] * n,
                0, tier["maximum_new_severe_rate"], tier["minimum_new_severe_budget"])
            campaign["cells"].append(cell)
    campaign["pass"] = all(cell["pass"] for cell in campaign["cells"])
    return campaign, reference, refs, repeat


class QualityExclusionTest(unittest.TestCase):
    def test_decimal_capacity_gate_remains_stricter_than_tier_ceiling(self):
        weights = "r9700-q4g64-n16k16-eval"
        gate = 0.048790164169432
        with tempfile.TemporaryDirectory() as directory:
            campaign, ref, refs, repeat = campaign_fixture(Path(directory), weights, gate)
            with patch("tools.ppl.compare_bf16_repeats.load_campaign", return_value=(ref, refs)), \
                 patch.object(run, "validate_bf16_repeat_comparison", return_value=repeat):
                cells, _ = assembly._campaign_quality_candidate(campaign, weights, 16, 2048)
                self.assertEqual(cells["32k"]["maximum_mean_nll_delta"], gate)
                self.assertTrue(assembly._quality_cells_eligible(cells))
                campaign["gates"]["r9700-g16"] = .05
                with self.assertRaisesRegex(ValueError, "loosens"):
                    assembly._campaign_quality_candidate(campaign, weights, 16, 2048)
        boundary = {"eligible": False, "tier": "capacity-speed", "mean_nll_delta": math.log(1.05),
            "maximum_mean_nll_delta": gate, "complete_finite_aligned": True,
            "scored_positions": 16383, "new_severe_positions": 0}
        self.assertEqual(pareto._quality_objective(boundary, "32k")[1],
                         ["quality_guardrails_not_met:32k"])

    def test_failed_severe_gate_is_retained_and_cannot_be_waived(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign, ref, refs, repeat = campaign_fixture(Path(directory))
            with patch("tools.ppl.compare_bf16_repeats.load_campaign", return_value=(ref, refs)), \
                 patch.object(run, "validate_bf16_repeat_comparison", return_value=repeat):
                for group, severe in ((16, 19), (32, 18)):
                    cells, source = assembly._campaign_quality_candidate(campaign, WEIGHTS, group, 2048)
                    self.assertTrue(cells["8k"]["eligible"])
                    self.assertFalse(cells["32k"]["eligible"])
                    self.assertEqual(cells["32k"]["new_severe_positions"], severe)
                    self.assertLess(cells["32k"]["mean_nll_delta"], .02)
                    self.assertIn("32k", source["cells"])
                path = Path(directory) / "campaign.json"
                path.write_text(json.dumps(campaign))
                row = {"weight_recipe": {"weights_id": WEIGHTS},
                    "cache_profile": {"value_group": 32}, "prefill_chunk": 2048,
                    "quality_cells": cells}
                bound = {"quality": {"path": str(path), "sha256": run.file_sha256(path),
                    "measurements": cells, "artifact": {key: source[key]
                        for key in ("weights_id", "sha256", "file_size_bytes")},
                    **{key: source[key] for key in ("cells", "representation", "campaign_identity")}}}
                pareto._revalidate_numerical_quality(row, bound)
                changed_row = copy.deepcopy(row)
                changed_row["quality_cells"]["32k"]["new_severe_positions"] = 17
                with self.assertRaisesRegex(ValueError, "differ from bound sidecars"):
                    pareto._revalidate_numerical_quality(changed_row, bound)
                for mutate in (
                    lambda c: c.update({"pass": True}),
                    lambda c: c.update({"quality_tier": "capacity-speed"}),
                    lambda c: c["gates"].update({"r9700-g16": .03}),
                    lambda c: c["cells"][-1].update({"new_severe_positions": 17}),
                    lambda c: c["cells"].pop(),
                ):
                    changed = copy.deepcopy(campaign)
                    mutate(changed)
                    with self.assertRaises(ValueError):
                        assembly._campaign_quality_candidate(changed, WEIGHTS, 16, 2048)

    def test_malformed_sidecars_are_not_quality_exclusions(self):
        for payload in (b"bad", struct.pack("<f", float("nan"))):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                campaign, ref, refs, repeat = campaign_fixture(Path(directory))
                cell = campaign["cells"][-1]
                path = Path(cell["command"][-1]).with_suffix(".nllf32")
                path.write_bytes(payload)
                cell["nll_sha256"] = run.file_sha256(path)
                with patch("tools.ppl.compare_bf16_repeats.load_campaign", return_value=(ref, refs)), \
                     patch.object(run, "validate_bf16_repeat_comparison", return_value=repeat), \
                     self.assertRaises(ValueError):
                    assembly._campaign_quality_candidate(campaign, WEIGHTS, 16, 2048)


if __name__ == "__main__":
    unittest.main()
