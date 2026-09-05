#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.assemble_dflash_selection import (
    _auxiliary,
    _matrix,
    _records,
    _same_campaign,
    _selected_base,
    assemble,
    main,
)
from tools.bench.run_ninfer_bench_matrix import BenchCase, MATRIX_SCHEMA_VERSION
from tools.ppl.pareto import _file_sha256, classify

CHUNK_SELECTION = {
    "path": "/chunk-selection.json", "sha256": "9" * 64,
    "selection_rule": "global_maximin_normalized_prefill_then_workspace_then_smaller_chunk_v2",
    "selected_prefill_chunk": 4096,
}


def shortlist_head_gate() -> dict:
    cells = {}
    for prompt in (8192, 32768):
        for concurrency in range(1, 5):
            expected = 64 * concurrency
            cells[f"whole-pp{prompt}+tg256_c{concurrency}"] = {
                "generated_tokens_per_lane": 256, "draft_window": 3,
                "minimum_rounds_per_lane": 64, "concurrency": concurrency,
                "expected_rounds_per_repetition": expected,
                "expected_rounds_all_repetitions": expected * 3,
                "observed_rounds_all_repetitions": expected * 3,
                "all_repetitions_minimum": True,
                "repetitions": [{
                    "repetition": repetition,
                    "observed_rounds": expected,
                    "expected_minimum_rounds": expected,
                    "minimum_met": True,
                } for repetition in range(3)],
            }
    return {
        "status": "q4_round_gate_pass_pending_trace",
        "head_format": "Q4G64_F16S", "activation_profile": "A8G64",
        "all_repetitions_minimum": True,
        "counter_semantics": (
            "per-repetition counters sum request-lane rounds; every whole g256 MTP3 "
            "cell requires concurrency * 64 rounds"
        ),
        "cells": cells,
    }


class DFlashSelectionTest(unittest.TestCase):
    artifact = {
        "path": "/dflash.ninfer", "file_size_bytes": 200, "sha256": "d" * 64,
        "model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-n16k16-dflash2-q4-eval",
    }
    bench = {"path": "/ninfer_bench", "file_size_bytes": 300, "sha256": "e" * 64}

    @staticmethod
    def write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self, root: Path) -> tuple[Path, Path, Path, list, list]:
        base = root / "base.json"
        candidates, provenance = [], []
        for recipe, digest, prefix, best_speed in (
            ("r9700-q4g64-n16k16-eval", "b" * 64, "all-q4-", 120.0),
            ("r9700-q4-w8-mse-n16k16-eval", "c" * 64, "mixed-", 110.0),
            ("r9700-q4g64-f8e4m3-four-role-n16k16-eval", "f" * 64, "hybrid-", 105.0),
        ):
            for group, profile in (
                (16, "dense"), (32, "dense"),
                (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
            ):
                name = f"{prefix}g{group}-{profile}"
                selected = group == 16 and profile == "dense"
                candidates.append({
                    "name": name,
                    "prefill_chunk": 4096,
                    "shortlist_head_precision_gate": shortlist_head_gate(),
                    "cache_profile": {"value_group": group, "plane_layouts": {
                        "key": "k", "value": "v", "value_scale": "s",
                    }},
                    "execution_profile": {
                        "q4_activation_bits": 8, "w8_activation_bits": 8,
                        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                        "xattention_profile": profile,
                    },
                    "quality": {
                        "eligible": True, "tier": "accuracy", "mean_nll_delta": 0.001,
                        "complete_finite_aligned": True, "scored_positions": 10000,
                        "new_severe_positions": 0,
                    },
                    "whole_inference_tokens_per_second": {
                        "whole_8k_c1": best_speed if selected else 90.0,
                    },
                    "capacity": {
                        "measurement_kind": "resolved_effective_maximum",
                        "binding_constraint": "device_memory", "tokens": 1000,
                    },
                })
                provenance.append({
                    "candidate": name, "artifact": {
                        "weights_id": recipe, "sha256": digest,
                    },
                })
        source = {
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": ["whole_8k_c1"],
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
            "candidates": candidates, "source_provenance": provenance,
        }
        self.base_input_path = root / "base-input.json"
        self.write(self.base_input_path, source)
        result = classify(source)
        result["pareto_input"] = {
            "path": str(self.base_input_path), "sha256": _file_sha256(self.base_input_path),
        }
        self.write(base, result)
        self.base_candidate_inputs = candidates
        self.base_candidate_provenance = provenance
        conversion = root / "conversion.json"
        self.write(conversion, {
            "target_key": "qwen3_8_27b_r9700",
            "recipe_id": "r9700-dflash2-all-q4g64-n16k16-bf16-codebook-eval-v1",
            "status": "registered-evaluation-only", "weight_recipe_selected": False,
            "identity": {"model_id": "qwen3.8-27b",
                         "weights_id": self.artifact["weights_id"]},
            "base": {"identity": {"weights_id": "r9700-q4g64-n16k16-eval"},
                     "sha256": "b" * 64, "payload_copy": "byte_exact"},
            "dflash_recipe": {
                "matrix_format": "Q4G64_F16S", "selector_codebook_format": "BF16",
                "objects": 66, "source_tensors": 81, "runtime_repack": False,
                "format_counts": {"BF16": 34, "Q4G64_F16S": 32},
                "format_encoded_bytes": {
                    "BF16": 254_814_720, "Q4G64_F16S": 954_654_720,
                },
                "tensor_encoded_bytes": 1_209_469_440,
            },
            "artifact": {"path": self.artifact["path"], "bytes": 200,
                         "sha256": "d" * 64},
        })
        shortlist = root / "shortlist"
        self.write(shortlist / "manifest.json", {
            "artifact_type": "ninfer_bench_matrix_run", "schema_version": MATRIX_SCHEMA_VERSION,
            "preset": "dflash-shortlist", "dry_run": False,
            "artifact": self.artifact, "bench": self.bench,
            "expected_kv_value_group": 16, "expected_q4_activation_bits": 8,
            "expected_w8_activation_bits": 8, "expected_fp8_qk_wmma_enabled": True,
            "expected_xattention_profile": "dense",
            "selected_prefill_chunk": 4096,
            "commands": [{"command": ["bench", "--prefill-chunk", "4096"]}],
        })
        self.write(shortlist / "dflash-shortlist.json", {
            "artifact_type": "ninfer_dflash_shortlist", "schema_version": 1, "pass": True,
            "artifact": self.artifact, "benchmark_executable": self.bench,
            "non_dominated_candidates": [
                {"draft_tokens": 1, "verify_width_resolved": 2},
                {"draft_tokens": 2, "verify_width_resolved": 3},
                {"draft_tokens": 3, "verify_width_resolved": 4},
            ],
        })
        capacities, paretos = [], []
        for k, w, eligible in ((1, 2, True), (2, 3, True), (3, 4, False)):
            capacity = root / f"capacity-k{k}"
            self.write(capacity / "manifest.json", {
                "artifact_type": "ninfer_bench_matrix_run", "schema_version": MATRIX_SCHEMA_VERSION,
                "preset": "dflash-capacity", "dry_run": False,
                "artifact": self.artifact, "bench": self.bench,
                "expected_kv_value_group": 16, "expected_q4_activation_bits": 8,
                "expected_w8_activation_bits": 8, "expected_fp8_qk_wmma_enabled": True,
                "expected_xattention_profile": "dense",
                "dflash_draft_tokens": k, "dflash_verify_width": w,
                "selected_prefill_chunk": 4096,
                "commands": [{"command": ["bench", "--prefill-chunk", "4096"]}],
            })
            capacities.append((k, w, capacity))
            if eligible:
                pareto = root / f"pareto-k{k}"
                commands = []
                for concurrency in range(1, 5):
                    commands.extend((
                        {"suite": "dflash_pareto_whole_inference",
                         "case": "whole_p8192_p32768_g256_dflash_graph",
                         "concurrency": concurrency, "report": str(pareto / f"whole-c{concurrency}"),
                         "command": ["whole", str(k), "--prefill-chunk", "4096"]},
                        {"suite": "dflash_pareto_decode",
                         "case": "context_p8192_p32768_g256_dflash_graph",
                         "concurrency": concurrency, "report": str(pareto / f"decode-c{concurrency}"),
                         "command": ["decode", str(k), "--prefill-chunk", "4096"]},
                    ))
                self.write(pareto / "manifest.json", {
                    "artifact_type": "ninfer_bench_matrix_run", "schema_version": MATRIX_SCHEMA_VERSION,
                    "preset": "dflash-pareto", "dry_run": False,
                    "artifact": self.artifact, "bench": self.bench,
                    "expected_kv_value_group": 16, "expected_q4_activation_bits": 8,
                    "expected_w8_activation_bits": 8, "expected_fp8_qk_wmma_enabled": True,
                    "expected_xattention_profile": "dense",
                    "dflash_draft_tokens": k, "dflash_verify_width": w, "commands": commands,
                    "selected_prefill_chunk": 4096,
                })
                paretos.append((k, w, pareto))
        return base, conversion, shortlist, capacities, paretos

    def select_base_winner(self, name: str) -> dict:
        candidates = json.loads(json.dumps(self.base_candidate_inputs))
        for row in candidates:
            row["whole_inference_tokens_per_second"]["whole_8k_c1"] = (
                200.0 if row["name"] == name else 90.0
            )
        source = {
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": ["whole_8k_c1"],
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
            "candidates": candidates, "source_provenance": self.base_candidate_provenance,
        }
        self.write(self.base_input_path, source)
        result = classify(source)
        result["pareto_input"] = {
            "path": str(self.base_input_path), "sha256": _file_sha256(self.base_input_path),
        }
        return result

    def test_selection_binds_exclusion_and_uses_maximin_whole_speed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, conversion, shortlist, capacities, paretos = self.fixture(Path(directory))

            def records(root: Path, _manifest: dict, preset: str, k: int, _w: int,
                        _prefill_chunk: int, **_options: object) -> dict:
                if preset == "dflash-capacity":
                    end = 3 if k == 3 else 4
                    return {c: [{"k": k, "tokens": (100 if k == 1 else 120)}]
                            for c in range(1, end + 1)}
                return {c: [{}] for c in range(1, 5)}

            def rows(path: Path, *_args: object, **_kwargs: object) -> list[dict]:
                k = int(_args[-2][1])
                concurrency = int(path.name.split("c")[-1])
                speed = 100.0 if k == 1 else 80.0
                acceptance = 1.0 if k == 1 else 2.0
                if "whole" in path.name:
                    return [{"suite": "dflash_pareto_whole_inference",
                             "label": f"whole-{tokens}", "concurrency": concurrency,
                             "whole_output_tok_s_mean": speed} for tokens in (8192, 32768)]
                return [{"suite": "dflash_pareto_decode", "label": f"decode-{tokens}",
                         "concurrency": concurrency, "spec_acceptance_length": acceptance}
                        for tokens in (8192, 32768)]

            with patch("tools.bench.assemble_dflash_selection._records", side_effect=records), patch(
                "tools.bench.assemble_dflash_selection._validate_shortlist_output",
            ), patch(
                "tools.bench.assemble_dflash_selection._missing_capacity_provenance",
                side_effect=lambda root, _manifest: ([{"failure": "C4"}]
                                                     if "capacity-k3" in str(root) else []),
            ), patch(
                "tools.bench.assemble_dflash_selection.validate_automatic_feasibility",
                side_effect=lambda report: {"measurement_kind": "resolved_effective_maximum",
                                            "resolved_effective_maximum_tokens": report["tokens"]},
            ), patch(
                "tools.bench.assemble_dflash_selection._auxiliary",
                return_value={"quality": "bound"},
            ), patch(
                "tools.bench.assemble_dflash_selection.report_rows", side_effect=rows,
            ):
                result = assemble(base, conversion, shortlist, capacities, paretos)
            self.assertEqual(result["eligible_frontier"], ["k1-w2", "k2-w3"])
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["winner"], "k1-w2")
            self.assertEqual(result["selected_prefill_chunk"], 4096)
            self.assertEqual(result["selected_base"]["prefill_chunk"], 4096)
            self.assertEqual(result["decisive_stage"], "maximin_whole_inference_throughput")
            excluded = next(row for row in result["candidates"] if row["draft_tokens"] == 3)
            self.assertFalse(excluded["capacity_eligible"])
            self.assertEqual(excluded["capacity_failures"], [{"failure": "C4"}])

    def test_matrix_requires_selected_base_prefill_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _base, _conversion, shortlist, _capacities, _paretos = self.fixture(
                Path(directory)
            )
            manifest_path = shortlist / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["selected_prefill_chunk"] = 2048
            self.write(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "does not bind selected prefill chunk"):
                _matrix(shortlist, "dflash-shortlist", prefill_chunk=2048)
            manifest["commands"][0]["command"][-1] = "2048"
            self.write(manifest_path, manifest)
            self.assertEqual(
                _matrix(shortlist, "dflash-shortlist", prefill_chunk=2048)[
                    "selected_prefill_chunk"
                ],
                2048,
            )

    def test_rejects_pareto_input_for_capacity_excluded_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, conversion, shortlist, capacities, paretos = self.fixture(Path(directory))
            paretos.append((3, 4, Path(directory) / "unused"))
            with patch("tools.bench.assemble_dflash_selection._records", return_value={}), patch(
                "tools.bench.assemble_dflash_selection._validate_shortlist_output",
            ), patch(
                "tools.bench.assemble_dflash_selection._missing_capacity_provenance",
                return_value=[{"failure": "C1"}],
            ):
                with self.assertRaisesRegex(ValueError, "exactly cover capacity-eligible"):
                    assemble(base, conversion, shortlist, capacities, paretos)

    def test_rejects_unretained_pareto_input_instead_of_ignoring_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, conversion, shortlist, capacities, paretos = self.fixture(Path(directory))
            paretos.append((99, 100, Path(directory) / "unretained"))
            with patch("tools.bench.assemble_dflash_selection._records", return_value={}), patch(
                "tools.bench.assemble_dflash_selection._validate_shortlist_output",
            ):
                with self.assertRaisesRegex(ValueError, "unique retained-shortlist"):
                    assemble(base, conversion, shortlist, capacities, paretos)

    def test_rejects_malformed_selected_base_recipe_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, conversion, shortlist, capacities, paretos = self.fixture(Path(directory))
            payload = json.loads(base.read_text(encoding="utf-8"))
            payload["terminal_production_selection"]["winner_artifact"][
                "sha256"
            ] = "not-a-sha256"
            self.write(base, payload)
            with self.assertRaisesRegex(ValueError, "does not recompute exactly"):
                assemble(base, conversion, shortlist, capacities, paretos)

    def test_dflash_campaign_matches_selected_text_prefill_but_keeps_verify_dense(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, _conversion, shortlist, _capacities, _paretos = self.fixture(Path(directory))
            payload = self.select_base_winner("all-q4-g16-b128-s16-tau900")
            self.write(base, payload)
            group, profile, _recipe, _value = _selected_base(base)
            self.assertEqual((group, profile), (16, "b128-s16-tau900"))

            manifest = json.loads((shortlist / "manifest.json").read_text())
            with self.assertRaisesRegex(ValueError, "selected Text-prefill"):
                _same_campaign(
                    manifest, self.artifact, self.bench, group, profile, 4096, None
                )
            manifest["expected_xattention_profile"] = profile
            _same_campaign(
                manifest, self.artifact, self.bench, group, profile, 4096, None
            )

    def test_campaign_rejects_rebound_hybrid_planner(self) -> None:
        authority = {
            "tool": {"path": "/planner", "file_size_bytes": 10, "sha256": "a" * 64},
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
            "expected_fp8_qk_wmma_enabled": True,
            "expected_xattention_profile": "dense",
            "required_candidate_identity": "fp8-hybrid-selection-authority",
            "hybrid_shared_workspace_authority": {
                **authority, "tool": {**authority["tool"], "sha256": "b" * 64},
            },
        }
        with self.assertRaisesRegex(ValueError, "planner differs"):
            _same_campaign(
                manifest, self.artifact, self.bench, 16, "dense", 4096, authority
            )

    def test_nonhybrid_campaign_rejects_hybrid_authority(self) -> None:
        manifest = {
            "artifact": self.artifact, "bench": self.bench,
            "expected_kv_value_group": 16,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_fp8_qk_wmma_enabled": True,
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
                "artifact_type": "ninfer_dflash_ordinary_greedy_parity", "schema_version": 1,
                "artifact": self.artifact, "benchmark_executable": self.bench, "pass": True,
                "comparisons": [{"concurrency": c, "draft_tokens": 7,
                                 "dflash_verify_width": 12, "exact": True}
                                for c in range(1, 5)],
            })
            self.write(determinism_path, {
                "artifact_type": "ninfer_dflash_proposal_determinism", "schema_version": 1,
                "artifact": self.artifact, "benchmark_executable": self.bench, "pass": True,
                "comparisons": [{"exact": True}],
            })
            self.write(root / "dflash-generated-quality.json", {
                "artifact_type": "ninfer_dflash_generated_quality_evidence", "schema_version": 1,
                "artifact": self.artifact, "benchmark_executable": self.bench, "pass": True,
                "target_output_gate": {"pass": True, "evidence": {
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

    def test_main_publishes_durably_and_rolls_back_owned_inode_on_revalidation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "selection.json"
            argv = ["--base-selection", str(root / "base.json"),
                    "--conversion-report", str(root / "conversion.json"),
                    "--shortlist-dir", str(root / "shortlist"),
                    "--capacity", "7", "12", str(root / "capacity"),
                    "--out", str(output)]
            result = {"artifact_type": "fixture", "status": "passed"}
            with patch("tools.bench.assemble_dflash_selection.assemble",
                       side_effect=(result, result)):
                main(argv)
            self.assertEqual(json.loads(output.read_text()), result)

            output.unlink()
            with patch("tools.bench.assemble_dflash_selection.assemble",
                       side_effect=(result, {**result, "status": "changed"})):
                with self.assertRaisesRegex(ValueError, "does not revalidate"):
                    main(argv)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
