#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.assemble_dflash_selection import (
    _auxiliary,
    _matched_ordinary_speed_gate,
    _matrix,
    _records,
    _same_campaign,
    _selected_base,
    assemble,
    main,
)
from tools.bench.run_ninfer_bench_matrix import (
    BenchCase,
    MATRIX_SCHEMA_VERSION,
    R9700_POWER_PROFILE,
)
from tools.ppl.pareto import _file_sha256, classify
from tools.bench import assemble_dflash_selection as selection

class DFlashSelectionTest(unittest.TestCase):
    def test_recipe_evidence_accepts_same_base_binary_and_keeps_evidence_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversion = root / "conversion.json"
            conversion.write_text("{}")
            artifact = {**self.artifact, "dflash_base_artifact": {"weights_id": "r9700-q4g64-n16k16-eval"},
                        "dflash_conversion_report": selection._identity(conversion)}
            route = {"base_artifact": artifact["dflash_base_artifact"], "base_benchmark": self.bench,
                     "cache_group": 16, "text_prefill_attention_profile": "dense",
                     "selected_prefill_chunk": 2048, "hybrid_base_authority": None}
            manifest = {"artifact_type": "ninfer_bench_matrix_run", "schema_version": MATRIX_SCHEMA_VERSION,
                        "preset": "dflash-shortlist", "dry_run": False, "failures": [],
                        "selected_prefill_chunk": 2048, "prefill_chunks": [2048],
                        "artifact": artifact, "bench": self.bench, "expected_kv_value_group": 16,
                        "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
                        "expected_split512_enabled": True, "expected_xattention_profile": "dense",
                        "power_profile": self.auto_power,
                        "commands": [{"command": [self.bench["path"], "--prefill-chunk", "2048"]}]}
            shortlist = {"artifact_type": "ninfer_dflash_shortlist",
                         "schema_version": selection.DFLASH_SHORTLIST_SCHEMA_VERSION,
                         "artifact": artifact, "benchmark_executable": self.bench, "pass": True,
                         "candidates": [{"valid_for_ranking": True, "profile": {
                             "draft_tokens_requested": k, "verify_width_resolved": w}}
                             for k, w in selection.DFLASH_PRODUCTION_PROFILES]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "dflash-shortlist.json").write_text(json.dumps(shortlist))
            with patch.object(selection, "inspect_artifact", return_value=artifact), \
                 patch.object(selection, "require_dflash_companion", return_value=artifact), \
                 patch.object(selection, "benchmark_profile", return_value={"benchmark": self.bench}), \
                 patch.object(selection, "_records") as records, \
                 patch.object(selection, "_validate_shortlist_output") as validate_shortlist:
                result = selection.recipe_evidence(route, selection.RECIPES[0], conversion, root)
                self.assertEqual(result["benchmark"], route["base_benchmark"])
                records.assert_called_once_with(root, manifest, "dflash-shortlist", 4, 5, 2048)
                validate_shortlist.assert_called_once_with(root, manifest, shortlist,
                                                           artifact, self.bench, 2048)
                validate_shortlist.side_effect = ValueError("raw parity failed")
                with self.assertRaisesRegex(ValueError, "raw parity failed"):
                    selection.recipe_evidence(route, selection.RECIPES[0], conversion, root)
                with self.assertRaisesRegex(ValueError, "recipe, base, or conversion receipt"):
                    selection.recipe_evidence(route, selection.RECIPES[1], conversion, root)

    artifact = {
        "path": "/dflash.ninfer", "file_size_bytes": 200, "sha256": "d" * 64,
        "model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-n16k16-dflash2-q4-eval",
    }
    bench = {"path": "/build/bench/ninfer_bench", "file_size_bytes": 300, "sha256": "e" * 64}
    auto_power = {
        "required": "auto",
        "sysfs_path": str(R9700_POWER_PROFILE),
        "observed": "auto",
        "rechecked_after": "auto",
    }

    @staticmethod
    def write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def campaign(self, *, missing=None, c1_win=True):
        recipes = [(recipe, Path("receipt"), Path(recipe)) for recipe in selection.RECIPES]
        capacity = [(recipe, k, w, Path(f"{recipe}/capacity"))
                    for recipe in selection.RECIPES for k, w in selection.DFLASH_PRODUCTION_PROFILES]
        c1 = [(recipe, k, w, Path(f"{recipe}/c1")) for recipe, k, w, _ in capacity]
        pareto = [(recipe, k, w, Path(f"{recipe}/pareto")) for recipe, k, w, _ in capacity]
        if missing == "c1": c1.pop()
        if missing == "pareto": pareto.pop()
        if not c1_win: pareto = []
        def evidence(route, recipe, *_):
            return {"recipe": recipe, "benchmark": self.bench}
        def cells(*_):
            return {"eligible_concurrency": [1, 2], "cells": {
                "1": {"eligible": True, "tokens": 40000},
                "2": {"eligible": True, "tokens": 34000},
                "3": {"eligible": False, "exclusion": {"status": "capacity_failure", "reason": "arena exhausted"}},
                "4": {"eligible": False, "exclusion": {"status": "insufficient_capacity"}}}}
        def performance(route, evidence, k, w, root, declared):
            # Canonical K4 is C1 winner but slower at C2, without erasing its C1 result.
            fastest = evidence["recipe"] == selection.RECIPES[0] and k == 4
            return {"declared_concurrency": declared, "corpus": {"sha256": "same"},
                "matched_speed_by_concurrency": {str(c): {"pass": c1_win and not (fastest and c == 2)} for c in declared},
                "objectives": {str(c): {"whole": {"8K": 150 if fastest else 120, "32K": 150 if fastest else 120},
                            "acceptance": {"8K": 3., "32K": 3.}} for c in declared}}
        with patch.object(selection, "selected_base_route", return_value={}), \
             patch.object(selection, "recipe_evidence", side_effect=evidence), \
             patch.object(selection, "capacity_cells", side_effect=cells), \
             patch.object(selection, "performance_cells", side_effect=performance):
            return assemble(Path("base"), recipes, capacity, c1, pareto)

    def test_primary_c1_survives_c2_slow_and_capacity_exclusions(self):
        result = self.campaign()
        winner = f"{selection.RECIPES[0]}/k4-w5"
        self.assertEqual(result["winner"], winner)
        self.assertEqual(result["winner_qualified_concurrency"], [1])
        self.assertNotEqual(result["per_concurrency"]["2"]["winner"], winner)
        self.assertEqual(result["winner_exclusions"]["3"]["reason"], "arena exhausted")
        self.assertFalse(result["production_selected"])

    def test_missing_declared_survivor_is_failure_not_exclusion(self):
        for missing in ("c1", "pareto"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "lacks"):
                self.campaign(missing=missing)

    def test_no_material_c1_win_does_not_publish_production_choice(self):
        result = self.campaign(c1_win=False)
        self.assertIsNone(result["winner"])
        self.assertEqual(result["status"], "no_qualified_C1_winner")

    def test_missing_declared_report_does_not_shrink_subset(self):
        case = BenchCase("suite", "case", (), 1, 0, "fixture")
        manifest = {"concurrency": [1, 2], "expected_kv_value_group": 16,
            "expected_xattention_profile": "dense", "artifact": self.artifact,
            "bench": self.bench, "commands": [{"suite": "suite", "case": "case",
                "concurrency": 1, "report": "/report", "command": ["bench"]}]}
        with patch.object(selection, "build_cases", return_value=[case]), \
             patch.object(selection, "load_bench_report", return_value={}):
            with self.assertRaisesRegex(ValueError, "exact dflash-pareto point set"):
                _records(Path("matrix"), manifest, "dflash-pareto", 4, 5, 2048,
                         required_concurrency=[1, 2])

    def test_capacity_retains_each_bound_failure_not_entire_profile(self):
        route = {"selected_prefill_chunk": 2048, "cache_group": 16,
                 "text_prefill_attention_profile": "dense", "hybrid_base_authority": None}
        with patch.object(selection, "_matrix", return_value={}), \
             patch.object(selection, "_same_campaign"), \
             patch.object(selection, "_records", return_value={1: [{}], 2: [{}]}), \
             patch.object(selection, "_missing_capacity_provenance", return_value=[
                 {"concurrency": 3, "reason": "arena"}, {"concurrency": 4, "reason": "arena"}]), \
             patch.object(selection, "validate_automatic_feasibility", side_effect=[
                 {"measurement_kind": "resolved_effective_maximum", "resolved_effective_maximum_tokens": 40000},
                 {"measurement_kind": "resolved_effective_maximum", "resolved_effective_maximum_tokens": 32000}]), \
             patch.object(selection, "_identity", return_value={}):
            result = selection.capacity_cells(route, {"artifact": {}, "benchmark": {}}, 4, 5, Path("cap"))
        self.assertEqual(result["eligible_concurrency"], [1])
        self.assertEqual(result["cells"]["2"]["exclusion"]["status"], "insufficient_32k_generation_capacity")
        self.assertEqual(result["cells"]["3"]["exclusion"]["reason"], "arena")

    def test_create_only_selection_revalidates_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selection.json"
            argv = ["--base-selection", "base", "--recipe", selection.RECIPES[0], "receipt", "shortlist",
                    "--out", str(output)]
            with patch.object(selection, "assemble", side_effect=[{"winner": "a"}, {"winner": "b"}]):
                with self.assertRaisesRegex(ValueError, "changed"):
                    main(argv)
            self.assertFalse(output.exists())
            with patch.object(selection, "assemble", return_value={"winner": "a"}):
                main(argv)
                with self.assertRaises(SystemExit): main(argv)

    def test_matched_ordinary_speed_gate_is_strict_and_fail_closed(self) -> None:
        rows = []
        for concurrency in range(1, 5):
            for prompt in (8192, 32768):
                common = {
                    "n_prompt": prompt, "n_gen": 256,
                    "requested_output_tokens": 257, "concurrency": concurrency,
                }
                rows.extend((
                    {**common, "suite": "dflash_pareto_whole_inference",
                     "whole_output_tok_s_mean": 110.0, "whole_output_tok_s_stddev": 1.0},
                    {**common, "suite": "dflash_pareto_whole_control",
                     "whole_output_tok_s_mean": 100.0, "whole_output_tok_s_stddev": 1.0},
                    {**common, "suite": "dflash_pareto_decode",
                     "decode_output_tok_s_mean": 110.0, "decode_output_tok_s_stddev": 1.0},
                    {**common, "suite": "dflash_pareto_control",
                     "decode_output_tok_s_mean": 100.0, "decode_output_tok_s_stddev": 1.0},
                ))
        gate = _matched_ordinary_speed_gate(rows)
        self.assertTrue(gate["pass"])
        self.assertGreater(gate["minimum_conservative_speedup"]["whole"], 1.0)
        self.assertGreater(gate["minimum_conservative_speedup"]["decode"], 1.0)

        changed = json.loads(json.dumps(rows))
        changed[2]["decode_output_tok_s_mean"] = 101.0
        gate = _matched_ordinary_speed_gate(changed)
        self.assertFalse(gate["pass"])
        self.assertLess(gate["minimum_raw_mean_speedup"]["decode"], 1.02)

        missing = rows[:-1]
        with self.assertRaisesRegex(ValueError, "lacks exact"):
            _matched_ordinary_speed_gate(missing)
        malformed = json.loads(json.dumps(rows))
        malformed[0]["whole_output_tok_s_mean"] = float("inf")
        with self.assertRaisesRegex(ValueError, "malformed"):
            _matched_ordinary_speed_gate(malformed)

    def test_timing_campaign_requires_exact_auto_power_before_and_after(self) -> None:
        manifest = {
            "preset": "dflash-pareto",
            "artifact": self.artifact, "bench": self.bench,
            "expected_kv_value_group": 16,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_split512_enabled": True,
            "expected_xattention_profile": "dense",
            "power_profile": self.auto_power,
        }
        _same_campaign(
            manifest, self.artifact, self.bench, 16, "dense", 4096, None
        )
        for field, value in (
            ("observed", "profile_standard"),
            ("rechecked_after", "profile_standard"),
            ("sysfs_path", "/different/device"),
        ):
            changed = json.loads(json.dumps(manifest))
            changed["power_profile"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "stable auto power"
            ):
                _same_campaign(
                    changed, self.artifact, self.bench, 16, "dense", 4096, None
                )
        missing = json.loads(json.dumps(manifest))
        del missing["power_profile"]
        with self.assertRaisesRegex(ValueError, "stable auto power"):
            _same_campaign(
                missing, self.artifact, self.bench, 16, "dense", 4096, None
            )

    def test_capacity_campaign_does_not_claim_timing_power_admission(self) -> None:
        manifest = {
            "preset": "dflash-capacity",
            "artifact": self.artifact, "bench": self.bench,
            "expected_kv_value_group": 16,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_split512_enabled": True,
            "expected_xattention_profile": "dense",
        }
        _same_campaign(
            manifest, self.artifact, self.bench, 16, "dense", 4096, None
        )

    def test_campaign_accepts_fresh_semantically_equal_hybrid_planner(self) -> None:
        authority = {
            "tool": {"path": "/planner", "file_size_bytes": 10, "sha256": "a" * 64},
            "build": {"root": "/build",
                      "cmake_cache": {"path": "/build/CMakeCache.txt", "file_size_bytes": 10,
                                      "sha256": "c" * 64},
                      "compile_commands": {"path": "/build/compile_commands.json",
                                           "file_size_bytes": 10, "sha256": "d" * 64}},
            "maximum_concurrency": 4, "prefill_chunks": [4096],
            "inventories_by_prefill_chunk": {
                "4096": {
                    "ordinary": [1, 2, 3, 4, 4096],
                    "mtp3": [1, 2, 3, 4, 8, 12, 16, 4096],
                }
            },
        }
        manifest = {
            "artifact": self.artifact, "bench": self.bench,
            "expected_kv_value_group": 16,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_split512_enabled": True,
            "expected_xattention_profile": "dense",
            "required_candidate_identity": "fp8-hybrid-selection-authority",
            "hybrid_shared_workspace_authority": {
                **authority, "tool": {**authority["tool"], "sha256": "b" * 64},
            },
        }
        _same_campaign(manifest, self.artifact, self.bench, 16, "dense", 4096, authority)

    def test_hybrid_dflash_campaign_binds_its_real_verify_width(self) -> None:
        authority = {
            "tool": {"path": "/planner", "file_size_bytes": 10, "sha256": "a" * 64},
            "build": {"root": "/build",
                      "cmake_cache": {"path": "/build/CMakeCache.txt", "file_size_bytes": 10,
                                      "sha256": "c" * 64},
                      "compile_commands": {"path": "/build/compile_commands.json",
                                           "file_size_bytes": 10, "sha256": "d" * 64}},
            "maximum_concurrency": 4, "prefill_chunks": [4096],
            "inventories_by_prefill_chunk": {
                "4096": {
                    "ordinary": [1, 2, 3, 4, 4096],
                    "mtp3": [1, 2, 3, 4, 8, 12, 16, 4096],
                }
            },
        }
        extended = {
            **authority,
            "inventories_by_prefill_chunk": {
                "4096": {
                    **authority["inventories_by_prefill_chunk"]["4096"],
                    "dflash-w12": [1, 2, 3, 4, 12, 24, 36, 48, 4096],
                }
            },
        }
        manifest = {
            "preset": "dflash-capacity",
            "artifact": self.artifact, "bench": self.bench,
            "expected_kv_value_group": 16,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_split512_enabled": True,
            "expected_xattention_profile": "dense",
            "required_candidate_identity": "fp8-hybrid-selection-authority",
            "hybrid_shared_workspace_authority": extended,
        }
        _same_campaign(
            manifest, self.artifact, self.bench, 16, "dense", 4096, authority, [12]
        )
        manifest["hybrid_shared_workspace_authority"] = authority
        with self.assertRaisesRegex(ValueError, "width authority differs"):
            _same_campaign(
                manifest, self.artifact, self.bench, 16, "dense", 4096, authority, [12]
            )

    def test_nonhybrid_campaign_rejects_hybrid_authority(self) -> None:
        manifest = {
            "artifact": self.artifact, "bench": self.bench,
            "expected_kv_value_group": 16,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_split512_enabled": True,
            "expected_xattention_profile": "dense",
            "required_candidate_identity": "fp8-hybrid-selection-authority",
        }
        with self.assertRaisesRegex(ValueError, "non-hybrid"):
            _same_campaign(
                manifest, self.artifact, self.bench, 16, "dense", 4096, None
            )

    def test_sparse_dflash_reports_are_validated_as_sparse_prefill_builds(self) -> None:
        case = BenchCase("suite", "case", (), 1, 0, "fixture")
        manifest = {
            "concurrency": [1],
            "expected_kv_value_group": 16,
            "expected_xattention_profile": "b128-s16-tau900",
            "artifact": self.artifact,
            "bench": self.bench,
            "commands": [{
                "suite": "suite", "case": "case", "concurrency": 1,
                "report": "/report.json", "command": ["ninfer_bench"],
            }],
        }
        with patch(
            "tools.bench.assemble_dflash_selection.build_cases", return_value=[case]
        ), patch(
            "tools.bench.assemble_dflash_selection.load_bench_report", return_value={}
        ) as load:
            _records(Path("/matrix"), manifest, "dflash-shortlist", 1, 2, 4096)
        self.assertEqual(load.call_args.args[-1], "b128-s16-tau900")

    def test_auxiliary_gate_requires_exact_complete_bound_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parity_path = root / "greedy-token-parity.json"
            determinism_path = root / "dflash-proposal-determinism.json"
            self.write(parity_path, {
                "artifact_type": "ninfer_dflash_ordinary_greedy_parity", "schema_version": 2,
                "artifact": self.artifact, "benchmark_executable": self.bench, "pass": True,
                "comparisons": [{"phase": phase, "concurrency": c, "draft_tokens": 7,
                                 "dflash_verify_width": 12, "includes_seed": phase == "whole",
                                 "exact": True}
                                for phase in ("decode", "whole") for c in range(1, 5)],
            })
            self.write(determinism_path, {
                "artifact_type": "ninfer_dflash_proposal_determinism", "schema_version": 1,
                "artifact": self.artifact, "benchmark_executable": self.bench, "pass": True,
                "comparisons": [{"exact": True}],
            })
            self.write(root / "dflash-generated-quality.json", {
                "artifact_type": "ninfer_dflash_generated_quality_evidence", "schema_version": 1,
                "artifact": self.artifact, "benchmark_executable": self.bench, "pass": True,
                "target_output_gate": {"pass": True, "concurrency": [1, 2, 3, 4], "evidence": {
                    "path": str(parity_path), "sha256": __import__("hashlib").sha256(
                        parity_path.read_bytes()).hexdigest()}},
                "proposal_gate": {"pass": True, "determinism_evidence": {
                    "path": str(determinism_path), "sha256": __import__("hashlib").sha256(
                        determinism_path.read_bytes()).hexdigest()}},
            })
            with patch(
                "tools.bench.assemble_dflash_selection.write_dflash_greedy_parity",
                return_value=(json.loads(parity_path.read_text()), []),
            ), patch(
                "tools.bench.assemble_dflash_selection.write_dflash_determinism",
                return_value=(json.loads(determinism_path.read_text()), []),
            ), patch(
                "tools.bench.assemble_dflash_selection.write_dflash_quality_evidence",
                return_value=(json.loads((root / "dflash-generated-quality.json").read_text()), []),
            ):
                self.assertIn(
                    "generated_quality",
                    _auxiliary(root, {"commands": []}, self.artifact, self.bench, 7, 12),
                )
            changed = json.loads(parity_path.read_text())
            changed["comparisons"][0]["exact"] = False
            self.write(parity_path, changed)
            with self.assertRaisesRegex(ValueError, "parity is incomplete"):
                _auxiliary(root, {"commands": []}, self.artifact, self.bench, 7, 12)



if __name__ == "__main__":
    unittest.main()
