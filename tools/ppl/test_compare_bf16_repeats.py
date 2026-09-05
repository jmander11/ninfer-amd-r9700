#!/usr/bin/env python3
"""Focused tests for deterministic BF16 campaign repeat comparison."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ppl import compare_bf16_repeats as compare
from tools.ppl import run


class CompareBf16RepeatsTest(unittest.TestCase):
    def payload(self) -> dict:
        return {
            "model_id": run.MODEL_ID,
            "reference_weights_id": run.BF16_WEIGHTS_ID,
            "reference_source": {"hash": "source"},
            "reference_execution": {"profile": "deterministic"},
            "corpus": {"hash": "corpus"},
            "lengths": [run.DEFAULT_TOKENS, run.LONG_TOKENS],
            "skip": "half", "prefill_chunk": 4096, "schedules": ["prefill"],
            "spec": "none", "draft_tokens": 0, "terrible_nll": run.TERRIBLE_NLL,
            "scorers": {run.BASELINE: {"sha256": "a" * 64}},
        }

    def cells(self, root: Path) -> dict[int, tuple[dict, Path]]:
        result = {}
        for tokens in (run.DEFAULT_TOKENS, run.LONG_TOKENS):
            path = root / f"{tokens}.json"
            path.write_text("{}", encoding="utf-8")
            cell = {field: field for field in run.BF16_SCORER_REPORT_FIELDS}
            cell["prompt_tokens"] = tokens
            cell["skip_tokens"] = tokens // 2
            count = tokens - tokens // 2 - 1
            cell["tokens_scored"] = count
            cell["argmax_tokens"] = count
            cell["non_finite"] = 0
            cell["terrible_nll"] = run.TERRIBLE_NLL
            cell["terrible_tokens"] = 0
            cell["sum_nll"] = 0.0
            cell["mean_nll"] = 0.0
            cell["max_nll"] = 0.0
            cell["ppl"] = 1.0
            path.with_suffix(".nllf32").write_bytes(b"\0\0\0\0" * count)
            path.with_suffix(".argmaxi32").write_bytes(b"\1\0\0\0" * count)
            result[tokens] = cell, path
        return result

    def test_exact_and_sidecar_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            first = self.cells(root / "first")
            second = self.cells(root / "second")
            result = compare.compare(self.payload(), first, self.payload(), second)
            self.assertTrue(result["exact"])
            changed_sidecar = second[run.LONG_TOKENS][1].with_suffix(".argmaxi32")
            changed = bytearray(changed_sidecar.read_bytes())
            changed[0] = 2
            changed_sidecar.write_bytes(changed)
            result = compare.compare(self.payload(), first, self.payload(), second)
            self.assertFalse(result["exact"])
            self.assertFalse(result["rows"][1]["exact"])

    def test_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            first = self.cells(root / "first")
            second = self.cells(root / "second")
            changed = copy.deepcopy(self.payload())
            changed["reference_execution"] = {"profile": "different"}
            with self.assertRaisesRegex(ValueError, "reference_execution"):
                compare.compare(self.payload(), first, changed, second)

    def test_runtime_source_and_corpus_identity_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            first = self.cells(root / "first")
            second = self.cells(root / "second")
            for field in ("reference_execution", "reference_source", "corpus", "prefill_chunk"):
                changed = copy.deepcopy(self.payload())
                changed[field] = {"changed": True} if isinstance(changed[field], dict) else 2048
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    compare.compare(self.payload(), first, changed, second)

    def test_all_scorer_semantics_including_pv_and_gdn_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            first = self.cells(root / "first")
            second = self.cells(root / "second")
            for field in (
                "formula_profile", "corpus_ids_sha256", "cache_append_boundary",
            ):
                changed = copy.deepcopy(second)
                changed[run.DEFAULT_TOKENS][0][field] = "changed"
                with self.subTest(field=field):
                    result = compare.compare(self.payload(), first, self.payload(), changed)
                    self.assertFalse(result["exact"])
                    self.assertFalse(result["rows"][0]["semantic_fields_exact"])
            for field in ("terrible_tokens", "sum_nll", "mean_nll", "max_nll", "ppl"):
                changed = copy.deepcopy(second)
                changed[run.DEFAULT_TOKENS][0][field] = 2
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "aggregates differ"
                ):
                    compare.compare(self.payload(), first, self.payload(), changed)
            for identity in ("attention_pv", "gdn_recurrence"):
                changed = copy.deepcopy(second)
                changed[run.DEFAULT_TOKENS][0]["execution_provenance"] = {
                    identity: {"changed": True}
                }
                with self.subTest(identity=identity):
                    result = compare.compare(self.payload(), first, self.payload(), changed)
                    self.assertFalse(result["exact"])

    def test_equal_truncated_sidecars_are_invalid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            first = self.cells(root / "first")
            second = self.cells(root / "second")
            for cells in (first, second):
                cells[run.DEFAULT_TOKENS][1].with_suffix(".nllf32").write_bytes(b"\0\0\0\0")
            with self.assertRaisesRegex(ValueError, "invalid length"):
                compare.compare(self.payload(), first, self.payload(), second)

    def test_equal_nonfinite_or_out_of_domain_sidecars_are_invalid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            first = self.cells(root / "first")
            second = self.cells(root / "second")
            for cells in (first, second):
                cells[run.DEFAULT_TOKENS][1].with_suffix(".nllf32").write_bytes(
                    b"\0\0\xc0\x7f" * cells[run.DEFAULT_TOKENS][0]["tokens_scored"]
                )
            with self.assertRaisesRegex(ValueError, "invalid content"):
                compare.compare(self.payload(), first, self.payload(), second)

    def test_load_campaign_recomputes_corpus_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scorer = root / "ppl.py"
            scorer.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            campaign = root / "results.json"
            command = [
                str(scorer), "--weights", str(root / "weights"), "--ids",
                str(root / "corpus.ids"), "--device", "0",
            ]
            value = {
                "artifact_type": run.CAMPAIGN_ARTIFACT_TYPE,
                "schema_version": run.CAMPAIGN_SCHEMA_VERSION,
                "pass": True,
                "candidate_artifact": None,
                "weights_inputs": {run.BASELINE: str(root / "weights")},
                "scorers": {run.BASELINE: {
                    "path": str(scorer.resolve()), "bytes": scorer.stat().st_size,
                    "sha256": run.file_sha256(scorer),
                }},
                "schedules": ["prefill"], "spec": "none", "draft_tokens": 0,
                "lengths": [run.DEFAULT_TOKENS, run.LONG_TOKENS],
                "corpus": {"forged": True},
                "cells": [
                    {"scheme": run.BASELINE, "command": command},
                    {"scheme": run.BASELINE, "command": command},
                ],
            }
            campaign.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(run, "validate_corpus", return_value={"actual": True}), \
                    self.assertRaisesRegex(ValueError, "corpus provenance differs"):
                compare.load_campaign(campaign)

    def test_same_campaign_path_is_not_a_fresh_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "results.json"
            campaign.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct fresh campaign"):
                compare._require_distinct_campaigns(campaign, campaign)
            symlink = root / "symlink.json"
            symlink.symlink_to(campaign)
            with self.assertRaisesRegex(ValueError, "distinct fresh campaign"):
                compare._require_distinct_campaigns(campaign, symlink)
            hardlink = root / "hardlink.json"
            os.link(campaign, hardlink)
            with self.assertRaisesRegex(ValueError, "distinct fresh campaign"):
                compare._require_distinct_campaigns(campaign, hardlink)

    def test_output_must_not_alias_either_input_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text("first\n", encoding="utf-8")
            second.write_text("second\n", encoding="utf-8")
            for output in (first, second):
                with self.subTest(output=output), self.assertRaisesRegex(
                    ValueError, "must not alias"
                ):
                    compare._require_distinct_campaigns(first, second, output)
            hardlink = root / "output.json"
            os.link(first, hardlink)
            with self.assertRaisesRegex(ValueError, "must not alias"):
                compare._require_distinct_campaigns(first, second, hardlink)

    def test_reuse_binding_requires_exact_pair_and_selected_authority_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_campaign = root / "first.json"
            second_campaign = root / "second.json"
            first_campaign.write_text('{"campaign":"first"}\n', encoding="utf-8")
            second_campaign.write_text('{"campaign":"second"}\n', encoding="utf-8")
            (root / "first").mkdir()
            (root / "second").mkdir()
            result = compare.compare(
                self.payload(), self.cells(root / "first"),
                self.payload(), self.cells(root / "second"),
            )
            result.update({
                "artifact_type": run.BF16_REPEAT_ARTIFACT_TYPE,
                "schema_version": run.BF16_REPEAT_SCHEMA_VERSION,
                "quality_evidence": False,
                "inputs": {
                    "first": {"path": str(first_campaign),
                              "sha256": run.file_sha256(first_campaign)},
                    "second": {"path": str(second_campaign),
                               "sha256": run.file_sha256(second_campaign)},
                },
            })
            comparison = root / "comparison.json"
            comparison.write_text(json.dumps(result) + "\n", encoding="utf-8")
            with patch.object(compare, "load_campaign") as load_campaign, patch.object(
                compare, "compare", return_value=result
            ):
                load_campaign.side_effect = [
                    (self.payload(), self.cells(root / "first")),
                    (self.payload(), self.cells(root / "second")),
                ]
                binding = run.validate_bf16_repeat_comparison(comparison, first_campaign)
            self.assertEqual(binding["authority_input"], "first")
            self.assertEqual(
                binding["authority_campaign_sha256"], run.file_sha256(first_campaign)
            )

            changed = copy.deepcopy(result)
            changed["rows"][1]["exact"] = False
            comparison.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "non-exact"):
                run.validate_bf16_repeat_comparison(comparison, first_campaign)

    def test_reuse_binding_rejects_unrelated_or_changed_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            unrelated = root / "unrelated.json"
            for path in (first, second, unrelated):
                path.write_text(path.name + "\n", encoding="utf-8")
            row = {
                "prompt_tokens": run.DEFAULT_TOKENS, "exact": True,
                "semantic_fields_exact": True, "semantic_fields": {},
                "nll_sha256": {"first": "1" * 64, "second": "1" * 64},
                "argmax_sha256": {"first": "2" * 64, "second": "2" * 64},
            }
            other_row = {**row, "prompt_tokens": run.LONG_TOKENS}
            payload = {
                "artifact_type": run.BF16_REPEAT_ARTIFACT_TYPE,
                "schema_version": run.BF16_REPEAT_SCHEMA_VERSION,
                "quality_evidence": False, "exact": True,
                "inputs": {
                    "first": {"path": str(first), "sha256": run.file_sha256(first)},
                    "second": {"path": str(second), "sha256": run.file_sha256(second)},
                },
                "rows": [row, other_row],
            }
            comparison = root / "comparison.json"
            comparison.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not an input"):
                run.validate_bf16_repeat_comparison(comparison, unrelated)
            first.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "input bytes changed"):
                run.validate_bf16_repeat_comparison(comparison, first)


if __name__ == "__main__":
    unittest.main()
