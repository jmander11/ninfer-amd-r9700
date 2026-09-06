#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ppl.assemble_pareto import (
    _quality_candidate,
    _reports,
    _manifest_prefill_chunk,
    _validate_mtp_target_parity,
    assemble_candidate,
    validate_chunk_candidate_bindings,
    validate_xattention_dense_controls,
)
from tools.bench.run_ninfer_bench_matrix import BenchCase, file_sha256
from tools.bench.run_ninfer_bench_matrix import MATRIX_SCHEMA_VERSION, R9700_KV_PLANE_LAYOUTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto import _valid_capacity_failure, classify


class AssembleParetoTest(unittest.TestCase):
    def test_mtp_whole_requires_exact_target_output_parity(self) -> None:
        def report(drafts: int, token: int) -> dict:
            speculative = {
                "rounds": 64, "drafted_tokens": 192,
                "accepted_tokens": 192, "fallback_steps": 0,
            }
            return {
                "config": {"draft_tokens": drafts},
                "tests": [{
                    "label": f"whole-pp{tokens}+tg256",
                    "n_gen": 256,
                    "speculative": dict(speculative),
                    "reps": [{
                        "generated_token_ids_by_lane": [[token, tokens]],
                        "speculative": dict(speculative),
                    }],
                } for tokens in (8192, 32768)],
            }

        mtp = report(3, 7)
        ordinary = report(0, 7)
        selected, parity = _validate_mtp_target_parity([mtp, ordinary], 1)
        self.assertIs(selected, mtp)
        self.assertTrue(parity["pass"])
        self.assertEqual(parity["compared_tokens"], 4)
        self.assertEqual(
            parity["round_accounting"]["cells"]["whole-pp8192+tg256"]
            ["theoretical_minimum_rounds_per_repetition"],
            64,
        )
        with self.assertRaisesRegex(ValueError, "requires one MTP3 row"):
            _validate_mtp_target_parity([mtp], 1)
        changed = copy.deepcopy(ordinary)
        changed["tests"][1]["reps"][0]["generated_token_ids_by_lane"][0][0] = 8
        with self.assertRaisesRegex(ValueError, "target-output parity failed"):
            _validate_mtp_target_parity([mtp, changed], 1)
        extra_round = copy.deepcopy(mtp)
        for test in extra_round["tests"]:
            test["speculative"]["rounds"] = 65
            test["reps"][0]["speculative"]["rounds"] = 65
        _, accounting = _validate_mtp_target_parity([extra_round, ordinary], 1)
        self.assertEqual(
            accounting["round_accounting"]["cells"]["whole-pp8192+tg256"]
            ["repetitions"][0],
            {
                "repetition": 0, "observed_rounds": 65,
                "theoretical_minimum_rounds": 64,
            },
        )
        inconsistent = copy.deepcopy(mtp)
        inconsistent["tests"][0]["speculative"]["rounds"] = 65
        with self.assertRaisesRegex(ValueError, "aggregate counters do not equal repetitions"):
            _validate_mtp_target_parity([inconsistent, ordinary], 1)

    def test_chunk_selection_binds_candidate_artifact_bench_group_and_profile(self) -> None:
        artifact = {"weights_id": "r9700-q4g64-n16k16-eval", "sha256": "a" * 64, "file_size_bytes": 1}
        bench = {"sha256": "b" * 64, "file_size_bytes": 2}
        selection = {"sources": [{
            "weights_id": artifact["weights_id"], "kv_value_group": 16,
            "xattention_profile": "dense", "artifact": artifact,
            "benchmark_executable": bench,
        }]}
        provenance = [{
            "artifact": artifact, "benchmark_executable": bench, "cache_value_group": 16,
            "quality": {"representation": {"xattention_profile": "dense"}},
        }]
        validate_chunk_candidate_bindings(selection, provenance)
        changed = copy.deepcopy(provenance)
        changed[0]["benchmark_executable"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_chunk_candidate_bindings(selection, changed)

    def test_manifest_requires_one_explicit_selected_prefill_chunk(self) -> None:
        manifest = {
            "selected_prefill_chunk": 2048,
            "commands": [
                {"command": ["bench", "--prefill-chunk", "2048"]},
                {"command": ["bench", "--prefill-chunk", "2048"]},
            ],
        }
        self.assertEqual(_manifest_prefill_chunk(manifest, "pareto-whole"), 2048)
        manifest["commands"][1]["command"][-1] = "4096"
        with self.assertRaisesRegex(ValueError, "do not bind one selected prefill chunk"):
            _manifest_prefill_chunk(manifest, "pareto-whole")
        manifest["commands"][1]["command"] = ["bench"]
        with self.assertRaisesRegex(ValueError, "lacks one explicit"):
            _manifest_prefill_chunk(manifest, "pareto-whole")
        manifest["commands"][1]["command"] = ["bench", "--prefill-chunk", "2048"]
        manifest["selected_prefill_chunk"] = 4096
        with self.assertRaisesRegex(ValueError, "differs from its commands"):
            _manifest_prefill_chunk(manifest, "pareto-whole")
        manifest["selected_prefill_chunk"] = True
        with self.assertRaisesRegex(ValueError, "lacks an exact"):
            _manifest_prefill_chunk(manifest, "pareto-whole")

    @patch("tools.ppl.assemble_pareto.ppl_run.validate_bf16_repeat_comparison")
    def test_xattention_admission_requires_complete_controls_per_artifact(
        self, validate_repeat,
    ) -> None:
        candidates = [
            {
                "name": f"g{group}-{profile}",
                "prefill_chunk": 4096,
                "cache_profile": {"value_group": group},
                "execution_profile": {"xattention_profile": profile},
            }
            for group, profile in (
                (16, "dense"), (32, "dense"),
                (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
            )
        ]
        reused_bf16 = {"path": "/authority.json", "sha256": "4" * 64}
        repeat_comparison = {
            "path": "/comparison.json", "sha256": "5" * 64,
            "authority_input": "first", "authority_campaign_sha256": "4" * 64,
        }
        validate_repeat.return_value = repeat_comparison
        campaign_identity = {
            "artifact_type": "ninfer_r9700_ppl_campaign",
            "schema_version": 6,
            "prefill_chunk": 4096,
            "corpus": {
                "ids_sha256": "c" * 64,
                "manifest_sha256": "d" * 64,
            },
            "reference_weights_id": "bf16-source",
            "reference_source": {
                "config_sha256": "e" * 64,
                "index_sha256": "f" * 64,
            },
            "reference_execution": {"profile": "deterministic-fixture"},
            "bf16_scorer": {"sha256": "1" * 64, "bytes": 123},
            "reused_bf16_campaign": reused_bf16,
            "bf16_repeat_comparison": repeat_comparison,
        }
        provenance = []
        for candidate in candidates:
            profile = candidate["execution_profile"]["xattention_profile"]
            provenance.append({
                "candidate": candidate["name"],
                "artifact": {
                    "weights_id": "r9700-q4g64-n16k16-eval",
                    "sha256": "a" * 64,
                    "file_size_bytes": 123,
                },
                "quality": {
                    "path": f"/{profile}.json",
                    "sha256": ("2" if profile == "dense" else "3") * 64,
                    "representation": {"xattention_profile": profile},
                    "campaign_identity": campaign_identity,
                },
                "matrices": {"pareto-capacity": {}, "pareto-whole": {}},
                "capacity_failures": [],
            })
        with self.assertRaisesRegex(ValueError, "all-Q4, mixed Q4/W8"):
            validate_xattention_dense_controls(candidates, provenance)
        with self.assertRaisesRegex(ValueError, "every candidate artifact"):
            validate_xattention_dense_controls(candidates[2:], provenance[2:])
        second_candidates = [
            {
                **candidate,
                "name": f"mixed-{candidate['name']}",
            }
            for candidate in candidates
        ]
        second_provenance = [
            {
                **source,
                "candidate": f"mixed-{source['candidate']}",
                "artifact": {
                    "weights_id": "r9700-q4-w8-mse-n16k16-eval",
                    "sha256": "b" * 64,
                    "file_size_bytes": 456,
                },
                "quality": {
                    **source["quality"],
                    "path": f"/mixed-{source['quality']['path'].lstrip('/')}",
                },
            }
            for source in provenance
        ]
        third_candidates = [
            {**candidate, "name": f"hybrid-{candidate['name']}"}
            for candidate in candidates
        ]
        third_provenance = [
            {
                **source,
                "candidate": f"hybrid-{source['candidate']}",
                "artifact": {
                    "weights_id": "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
                    "sha256": "7" * 64,
                    "file_size_bytes": 789,
                },
                "quality": {
                    **source["quality"],
                    "path": f"/hybrid-{source['quality']['path'].lstrip('/')}",
                },
            }
            for source in provenance
        ]
        validate_xattention_dense_controls(
            [*candidates, *second_candidates, *third_candidates],
            [*provenance, *second_provenance, *third_provenance],
        )
        all_candidates = [*candidates, *second_candidates, *third_candidates]
        failed_pair = copy.deepcopy([
            *provenance, *second_provenance, *third_provenance,
        ])
        for index in (8, 10):
            failed_pair[index]["matrices"] = {"pareto-capacity": {}}
            failed_pair[index]["capacity_failures"] = [{
                "status": "memory_admission_ineligible",
                "concurrency": 4,
            }]
        validate_xattention_dense_controls(all_candidates, failed_pair)
        unmatched = copy.deepcopy([
            *provenance, *second_provenance, *third_provenance,
        ])
        unmatched[8]["matrices"] = {"pareto-capacity": {}}
        unmatched[8]["capacity_failures"] = [{
            "status": "memory_admission_ineligible",
            "concurrency": 4,
        }]
        with self.assertRaisesRegex(ValueError, "matched dense/sparse capacity eligibility"):
            validate_xattention_dense_controls(all_candidates, unmatched)
        failed_with_whole = copy.deepcopy(failed_pair)
        failed_with_whole[8]["matrices"]["pareto-whole"] = {}
        with self.assertRaisesRegex(ValueError, "exactly for capacity-eligible profiles"):
            validate_xattention_dense_controls(all_candidates, failed_with_whole)
        changed_repeat = copy.deepcopy(second_provenance)
        changed_repeat[-1]["quality"]["campaign_identity"] = copy.deepcopy(campaign_identity)
        changed_repeat[-1]["quality"]["campaign_identity"]["bf16_repeat_comparison"] = {
            **repeat_comparison, "sha256": "6" * 64,
        }
        with self.assertRaisesRegex(ValueError, "repeat authority binding differs"):
            validate_xattention_dense_controls(
                [*candidates, *second_candidates, *third_candidates],
                [*provenance, *changed_repeat, *third_provenance],
            )
        provenance[-1] = {
            **provenance[-1],
            "artifact": {
                "weights_id": "r9700-q4g64-n16k16-eval", "sha256": "b" * 64,
                "file_size_bytes": 123,
            },
        }
        with self.assertRaisesRegex(ValueError, "every candidate artifact"):
            validate_xattention_dense_controls(candidates, provenance)

    def test_xattention_admission_rejects_legacy_quality(self) -> None:
        candidates = [
            {
                "name": f"g{group}-{profile}",
                "cache_profile": {"value_group": group},
                "execution_profile": {"xattention_profile": profile},
            }
            for group, profile in (
                (16, "dense"), (32, "dense"),
                (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
            )
        ]
        provenance = [{
            "candidate": candidate["name"],
            "artifact": {
                "weights_id": "other", "sha256": "a" * 64,
                "file_size_bytes": 123,
            },
            "quality": {
                "representation": {
                    "xattention_profile": candidate["execution_profile"]["xattention_profile"]
                },
            },
            "matrices": {"pareto-capacity": {}, "pareto-whole": {}},
            "capacity_failures": [],
        } for candidate in candidates]
        with self.assertRaisesRegex(ValueError, "native schema-v6 PPL campaigns"):
            validate_xattention_dense_controls(candidates, provenance)

    def test_xattention_admission_requires_capacity_and_whole_matrices(self) -> None:
        candidates = [
            {
                "name": f"g{group}-{profile}",
                "cache_profile": {"value_group": group},
                "execution_profile": {"xattention_profile": profile},
            }
            for group, profile in (
                (16, "dense"), (32, "dense"),
                (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
            )
        ]
        identity = {
            "artifact_type": "ninfer_r9700_ppl_campaign", "schema_version": 6,
            "corpus": {"ids_sha256": "c" * 64, "manifest_sha256": "d" * 64},
            "reference_weights_id": "bf16-source",
            "reference_source": {
                "config_sha256": "e" * 64, "index_sha256": "f" * 64,
            },
            "reference_execution": {"profile": "deterministic-fixture"},
            "bf16_scorer": {"sha256": "1" * 64, "bytes": 123},
        }
        provenance = [{
            "candidate": candidate["name"],
            "artifact": {
                "weights_id": "r9700-q4g64-n16k16-eval", "sha256": "a" * 64,
                "file_size_bytes": 123,
            },
            "quality": {
                "path": f"/{candidate['execution_profile']['xattention_profile']}.json",
                "sha256": (
                    "2" if candidate["execution_profile"]["xattention_profile"] == "dense"
                    else "3"
                ) * 64,
                "representation": candidate["execution_profile"],
                "campaign_identity": identity,
            },
            "matrices": {"pareto-capacity": {}},
            "capacity_failures": [],
        } for candidate in candidates]
        with self.assertRaisesRegex(ValueError, "capacity and whole matrices"):
            validate_xattention_dense_controls(candidates, provenance)

    def test_native_ppl_campaign_is_direct_quality_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = []
            for tokens in (8192, 32768):
                path = root / f"{tokens}.prefill.r9700-g16.json"
                path.write_text("{}", encoding="utf-8")
                path.with_suffix(".nllf32").write_bytes(b"nll")
                path.with_suffix(".argmaxi32").write_bytes(b"argmax")
                cells.append({
                    "scheme": "r9700-g16", "schedule": "prefill",
                    "model_id": "qwen3.8-27b", "weights_id": "weights",
                    "kv_format": "fp8-k-int4-v", "kv_value_group": 16,
                    "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                    "q4_activation_bits": 8, "w8_activation_bits": 8,
                    "fp8_qk_wmma_enabled": True,
                    "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                    "xattention_qualification": True,
                    "xattention_profile": "b128-s16-tau900",
                    "xattention_find_block": 128, "xattention_stride": 16,
                    "xattention_tau_permille": 900,
                    "prefill_chunk": 4096,
                    "prompt_tokens": tokens, "tokens_scored": tokens - 1,
                    "quality_tier": "capacity-speed", "pass": True,
                    "quality_eligible": True, "complete_finite_aligned": True,
                    "delta_mean_nll": 0.01, "new_severe_positions": 1,
                    "command": ["ppl", "--out-json", str(path)],
                    "nll_sha256": file_sha256(path.with_suffix(".nllf32")),
                    "argmax_sha256": file_sha256(path.with_suffix(".argmaxi32")),
                })
                path.write_text(json.dumps({
                    key: value for key, value in cells[-1].items()
                    if key not in {
                        "command", "nll_sha256", "argmax_sha256", "quality_tier",
                        "pass", "quality_eligible", "complete_finite_aligned",
                        "delta_mean_nll", "new_severe_positions",
                    }
                }), encoding="utf-8")
            quality, source = _quality_candidate({
                "artifact_type": "ninfer_r9700_ppl_campaign", "schema_version": 6,
                "pass": True, "model_id": "qwen3.8-27b", "schedules": ["prefill"],
                "lengths": [8192, 32768], "prefill_chunk": 4096,
                "q4_activation_bits": 8,
                "w8_activation_bits": 8,
                "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                "xattention_profile": "b128-s16-tau900",
                "candidate_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                "candidate_artifact": {
                    "weights_id": "weights", "sha256": "a" * 64,
                    "bytes": 123,
                },
                "cells": cells,
            }, "weights", 16, 4096)
            self.assertEqual(set(quality), {"8k", "32k"})
            self.assertEqual(
                source["representation"]["xattention_profile"], "b128-s16-tau900"
            )
            self.assertEqual(source["file_size_bytes"], 123)

            with self.assertRaisesRegex(ValueError, "unsupported identity or execution profile"):
                _quality_candidate({
                    "artifact_type": "ninfer_r9700_ppl_campaign", "schema_version": 6,
                    "pass": True, "model_id": "qwen3.8-27b", "schedules": ["prefill"],
                    "lengths": [8192, 32768], "prefill_chunk": 4096,
                    "q4_activation_bits": 8, "w8_activation_bits": 8,
                    "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                    "xattention_profile": "b128-s16-tau900",
                    "candidate_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                    "candidate_artifact": {"weights_id": "weights", "sha256": "a" * 64, "bytes": 123},
                    "cells": cells,
                }, "weights", 16, 2048)

    def test_native_ppl_campaign_rejects_changed_raw_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "8192.prefill.r9700-g16.json"
            path.write_text("{}", encoding="utf-8")
            path.with_suffix(".nllf32").write_bytes(b"nll")
            path.with_suffix(".argmaxi32").write_bytes(b"argmax")
            cell = {
                "scheme": "r9700-g16", "schedule": "prefill",
                "model_id": "qwen3.8-27b", "weights_id": "weights",
                "kv_format": "fp8-k-int4-v", "kv_value_group": 16,
                "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                "q4_activation_bits": 8, "w8_activation_bits": 8,
                "fp8_qk_wmma_enabled": True,
                "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                "xattention_qualification": False,
                "prompt_tokens": 8192, "tokens_scored": 8191,
                "quality_tier": "capacity-speed", "pass": True,
                "quality_eligible": True, "complete_finite_aligned": True,
                "delta_mean_nll": 0.01, "new_severe_positions": 1,
                "command": ["ppl", "--out-json", str(path)],
                "nll_sha256": file_sha256(path.with_suffix(".nllf32")),
                "argmax_sha256": file_sha256(path.with_suffix(".argmaxi32")),
            }
            campaign = {
                "artifact_type": "ninfer_r9700_ppl_campaign", "schema_version": 6,
                "pass": True, "model_id": "qwen3.8-27b", "schedules": ["prefill"],
                "lengths": [8192, 32768], "q4_activation_bits": 8,
                "w8_activation_bits": 8,
                "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                "xattention_profile": "dense",
                "candidate_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                "candidate_artifact": {
                    "weights_id": "weights", "sha256": "a" * 64, "bytes": 123,
                },
                "cells": [cell],
            }
            with self.assertRaisesRegex(ValueError, "json differs from its campaign"):
                _quality_candidate(campaign, "weights", 16)

    def test_report_loader_rejects_duplicate_point_covering_a_missing_concurrency(self) -> None:
        case = BenchCase("suite", "case", ("-p", "128"), 1, 0)
        manifest = {
            "concurrency": [1, 2],
            "commands": [
                {"suite": "suite", "case": "case", "concurrency": 1},
                {"suite": "suite", "case": "case", "concurrency": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.ppl.assemble_pareto.build_cases", return_value=[case]
        ):
            with self.assertRaisesRegex(ValueError, "exact pareto matrix point set"):
                _reports(Path(directory), manifest, "pareto", 4096)

    def test_report_loader_rejects_report_outside_campaign(self) -> None:
        case = BenchCase("suite", "case", ("-p", "128"), 1, 0)
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            external = Path(outside) / "stale.json"
            external.write_text("{}", encoding="utf-8")
            manifest = {
                "concurrency": [1],
                "commands": [{
                    "suite": "suite", "case": "case", "concurrency": 1,
                    "command": ["bench"], "report": str(external),
                }],
            }
            with patch("tools.ppl.assemble_pareto.build_cases", return_value=[case]):
                with self.assertRaisesRegex(ValueError, "resolves outside"):
                    _reports(root, manifest, "pareto", 4096)

    def test_assembly_binds_artifact_bench_and_all_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quality_path = root / "quality.json"
            artifact = {
                "path": "/model.ninfer", "weights_id": "weights", "model_id": "qwen3.8-27b",
                "sha256": "a" * 64, "file_size_bytes": 123,
            }
            quality_results = []
            for tokens in (8192, 32768):
                cell = root / f"q{tokens}.json"
                cell.write_text("{}", encoding="utf-8")
                cell.with_suffix(".nllf32").write_bytes(b"nll")
                cell.with_suffix(".argmaxi32").write_bytes(b"argmax")
                quality_results.append({
                    "artifact_weights_id": "weights", "kv_value_group": 16,
                    "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                    "prompt_tokens": tokens, "scored_positions": tokens - 1,
                    "complete_finite_aligned": True, "cell": str(cell),
                    "paired_vs_bf16": {"mean_nll_delta": 0.01,
                                       "new_severe_positions": 1},
                    "gates": {"accuracy": {"pass": True}},
                    "sha256": {
                        "json": file_sha256(cell),
                        "nllf32": file_sha256(cell.with_suffix(".nllf32")),
                        "argmaxi32": file_sha256(cell.with_suffix(".argmaxi32")),
                    },
                })
            quality_path.write_text(json.dumps({
                "schema": "ninfer-r9700-q4-a8-final-quality-v2",
                "representation": {
                    "q4_activation_bits": 8, "w8_activation_bits": 8,
                    "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                    "xattention_profile": "dense",
                    "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                },
                "artifacts": [{"weights_id": "weights", "sha256": "a" * 64, "bytes": 123}],
                "results": quality_results,
            }), encoding="utf-8")
            bench = {"path": "/bench", "sha256": "b" * 64, "file_size_bytes": 456}
            chunk_authority = {
                "path": str(root / "selected-prefill-chunk.json"),
                "sha256": "9" * 64,
                "artifact_type": "ninfer_r9700_prefill_chunk_selection",
                "schema_version": 2,
                "selected_prefill_chunk": 4096,
            }
            roots = {}
            for preset in ("pareto-capacity", "pareto-whole"):
                item = root / preset
                item.mkdir()
                report_records = []
                for concurrency in range(1, 5):
                    report = item / f"c{concurrency}.json"
                    report.write_text("{}", encoding="utf-8")
                    report_records.append({
                        "suite": "pareto_effective_capacity",
                        "case": "effective_capacity_mtp3",
                        "concurrency": concurrency,
                        "command": [
                            "bench", "--weights", "/model.ninfer",
                            "--max-ctx", "262144", "--concurrency", str(concurrency),
                            "--kv-capacity", "auto", "--spec", "mtp",
                            "--draft-tokens", "3", "--lm-head-draft",
                            "--prefill-chunk", "4096", "--output-file", str(report),
                        ],
                        "report": str(report),
                    })
                (item / "manifest.json").write_text(json.dumps({
                    "artifact_type": "ninfer_bench_matrix_run",
                    "schema_version": MATRIX_SCHEMA_VERSION,
                    "preset": preset, "dry_run": False, "artifact": artifact, "bench": bench,
                    "selected_prefill_chunk": 4096,
                    "prefill_chunk_authority": chunk_authority,
                    **({"post_chunk_capacity_gate": True}
                       if preset == "pareto-capacity" else {}),
                    "expected_kv_value_group": 16, "expected_q4_activation_bits": 8,
                    "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                    "expected_w8_activation_bits": 8,
                    "expected_fp8_qk_wmma_enabled": True,
                    "expected_fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                    "expected_xattention_profile": "dense",
                    "concurrency": list(range(1, 5)),
                    "power_profile": ({
                        "required": "auto",
                        "sysfs_path": (
                            "/sys/class/drm/card2/device/"
                            "power_dpm_force_performance_level"
                        ),
                        "observed": "auto",
                        "rechecked_after": "auto",
                    } if preset == "pareto-whole" else None),
                    "commands": report_records,
                }), encoding="utf-8")
                roots[preset] = item

            def reports(
                _root: Path, _manifest: dict, preset: str, _prefill_chunk: int,
                **_options: object,
            ) -> dict:
                output = {}
                for concurrency in range(1, 5):
                    if preset == "pareto-capacity":
                        if (_root / f"c{concurrency}.json").is_file():
                            output[concurrency] = [{"capacity": concurrency}]
                    else:
                        rows = [{
                            "label": f"whole-pp{tokens}+tg256",
                            "n_gen": 256,
                            "prefill_tok_s_mean": 1.0,
                            "decode_output_tok_s_mean": 2.0,
                            "whole_output_tok_s_mean": 3.0,
                            "speculative": {
                                "rounds": 192 * concurrency,
                                "drafted_tokens": 576 * concurrency,
                                "accepted_tokens": 576 * concurrency,
                                "fallback_steps": 0,
                            },
                            "reps": [{
                                "generated_token_ids_by_lane": [
                                    [tokens, repetition] for _ in range(concurrency)
                                ],
                                "speculative": {
                                    "rounds": 64 * concurrency,
                                    "drafted_tokens": 192 * concurrency,
                                    "accepted_tokens": 192 * concurrency,
                                    "fallback_steps": 0,
                                },
                            } for repetition in range(3)],
                        } for tokens in (8192, 32768)]
                        output[concurrency] = [
                            {"config": {"draft_tokens": 3}, "tests": copy.deepcopy(rows)},
                            {"config": {"draft_tokens": 0}, "tests": copy.deepcopy(rows)},
                        ]
                return output

            with patch("tools.ppl.assemble_pareto._reports", side_effect=reports), patch(
                "tools.ppl.assemble_pareto.validate_automatic_feasibility",
                side_effect=lambda report: {
                    "measurement_kind": "resolved_effective_maximum",
                    "binding_constraint": "model_context" if report["capacity"] < 3 else "device_memory",
                    "resolved_effective_maximum_tokens": report["capacity"] * 1000,
                },
            ):
                candidate, provenance = assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            self.assertEqual(len(candidate["capacity_by_cell"]), 4)
            self.assertEqual(len(candidate["whole_inference_tokens_per_second"]), 24)
            self.assertEqual(set(candidate["mtp_target_token_parity"]), {
                "c1", "c2", "c3", "c4",
            })
            self.assertNotIn("shortlist_head_precision_gate", candidate)
            self.assertEqual(
                candidate["mtp_target_token_parity"]["c4"]["round_accounting"]["cells"]
                ["whole-pp32768+tg256"]["theoretical_minimum_rounds_per_repetition"],
                256,
            )
            self.assertEqual(
                candidate["whole_inference_tokens_per_second"]["prefill_8192_c1"], 1.0
            )
            self.assertEqual(
                candidate["whole_inference_tokens_per_second"]["decode_32768_c4"], 2.0
            )
            self.assertEqual(
                candidate["whole_inference_tokens_per_second"]["whole_32768_c4"], 3.0
            )
            self.assertEqual(
                set(provenance["matrices"]), {"pareto-capacity", "pareto-whole"}
            )
            self.assertEqual(provenance["artifact"]["sha256"], "a" * 64)
            self.assertEqual(candidate["prefill_chunk"], 4096)
            self.assertEqual(provenance["selected_prefill_chunk"], 4096)
            self.assertEqual(provenance["quality"]["sha256"], file_sha256(quality_path))
            self.assertEqual(provenance["quality"]["artifact"]["sha256"], "a" * 64)
            classified = classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_quality_cells": ["8k", "32k"],
                "required_capacity_cells": [f"c{i}" for i in range(1, 5)],
                "required_speed_workloads": sorted(
                    candidate["whole_inference_tokens_per_second"]
                ),
                "candidates": [candidate],
                "source_provenance": [provenance],
            })
            self.assertEqual(classified["schema_version"], 7)
            self.assertEqual(
                classified["candidates"][0]["execution_profile"]["xattention_profile"],
                "dense",
            )
            self.assertEqual(classified["frontier"], ["candidate"])
            self.assertEqual(
                classified["candidates"][0]["weight_recipe"],
                {"kind": "artifact", "weights_id": "weights", "sha256": "a" * 64},
            )

            capacity_manifest = roots["pareto-capacity"] / "manifest.json"
            invalid_gate = json.loads(capacity_manifest.read_text(encoding="utf-8"))
            invalid_gate["post_chunk_capacity_gate"] = False
            capacity_manifest.write_text(json.dumps(invalid_gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a post-chunk capacity gate"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            invalid_gate["post_chunk_capacity_gate"] = True
            invalid_gate["prefill_chunk_authority"] = {
                **chunk_authority, "sha256": "8" * 64,
            }
            capacity_manifest.write_text(json.dumps(invalid_gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind the selected"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            invalid_gate["prefill_chunk_authority"] = chunk_authority
            capacity_manifest.write_text(json.dumps(invalid_gate), encoding="utf-8")

            whole_manifest = roots["pareto-whole"] / "manifest.json"
            mismatched_chunk = json.loads(whole_manifest.read_text(encoding="utf-8"))
            mismatched_chunk["prefill_chunk_authority"] = {
                **chunk_authority, "path": "/different-selection.json",
            }
            whole_manifest.write_text(json.dumps(mismatched_chunk), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind the selected"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            mismatched_chunk["prefill_chunk_authority"] = chunk_authority
            chunk_index = mismatched_chunk["commands"][0]["command"].index("--prefill-chunk")
            mismatched_chunk["commands"][0]["command"][chunk_index + 1] = "2048"
            whole_manifest.write_text(json.dumps(mismatched_chunk), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not bind one selected prefill chunk"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            mismatched_chunk["commands"][0]["command"][chunk_index + 1] = "4096"
            whole_manifest.write_text(json.dumps(mismatched_chunk), encoding="utf-8")

            superseded = json.loads(capacity_manifest.read_text(encoding="utf-8"))
            superseded["concurrency"].append(5)
            capacity_manifest.write_text(json.dumps(superseded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required ordered C=1..4"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            superseded["concurrency"] = list(range(1, 5))
            capacity_manifest.write_text(json.dumps(superseded), encoding="utf-8")

            changed_layout = json.loads(whole_manifest.read_text(encoding="utf-8"))
            changed_layout["expected_xattention_profile"] = "b128-s16-tau900"
            whole_manifest.write_text(json.dumps(changed_layout), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "execution profile"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            changed_layout["expected_xattention_profile"] = "dense"
            changed_layout["expected_kv_plane_layouts"] = {
                **R9700_KV_PLANE_LAYOUTS,
                "key": "feature-fastest-page-major",
            }
            whole_manifest.write_text(json.dumps(changed_layout), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cache plane layouts"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            changed_layout["expected_kv_plane_layouts"] = R9700_KV_PLANE_LAYOUTS
            whole_manifest.write_text(json.dumps(changed_layout), encoding="utf-8")

            (roots["pareto-whole"] / "failures.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a valid physical pareto-whole"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            (roots["pareto-whole"] / "failures.json").unlink()

            changed_power = json.loads(whole_manifest.read_text(encoding="utf-8"))
            changed_power["power_profile"]["rechecked_after"] = "profile_standard"
            whole_manifest.write_text(json.dumps(changed_power), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stable auto power"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )
            changed_power["power_profile"]["rechecked_after"] = "auto"
            whole_manifest.write_text(json.dumps(changed_power), encoding="utf-8")

            missing = roots["pareto-capacity"] / "c4.json"
            missing.unlink()
            logs = roots["pareto-capacity"] / "logs"
            logs.mkdir()
            (logs / "pareto_effective_capacity.effective_capacity_mtp3.c4.stderr.txt").write_text(
                "[ninfer_bench] loading /model.ninfer (max_context=262144, concurrency=4, "
                "kv_format=fp8-k-int4-v)\n"
                "ninfer_bench: minimum Engine runtime reservation requires 10663212291 bytes "
                "in addition to 1073741824 bytes of automatic headroom, but only "
                "11477728256 bytes are available after weights\n",
                encoding="utf-8",
            )
            (logs / "pareto_effective_capacity.effective_capacity_mtp3.c4.stdout.txt").write_text(
                "", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "no failures.json"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], None, 4096, chunk_authority,
                )
            missing_record = json.loads(
                (roots["pareto-capacity"] / "manifest.json").read_text(encoding="utf-8")
            )["commands"][-1]
            (roots["pareto-capacity"] / "failures.json").write_text(json.dumps([{
                "suite": missing_record["suite"],
                "case": missing_record["case"],
                "concurrency": 4,
                "returncode": 1,
                "stdout": str(
                    logs / "pareto_effective_capacity.effective_capacity_mtp3.c4.stdout.txt"
                ),
                "stderr": str(
                    logs / "pareto_effective_capacity.effective_capacity_mtp3.c4.stderr.txt"
                ),
                "command": missing_record["command"],
            }]), encoding="utf-8")
            with patch("tools.ppl.assemble_pareto._reports", side_effect=reports), patch(
                "tools.ppl.assemble_pareto.validate_automatic_feasibility",
                side_effect=lambda report: {
                    "measurement_kind": "resolved_effective_maximum",
                    "binding_constraint": "device_memory",
                    "resolved_effective_maximum_tokens": report["capacity"] * 1000,
                },
            ):
                incomplete, failed_source = assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], None, 4096, chunk_authority,
                )
            self.assertNotIn("c4", incomplete["capacity_by_cell"])
            self.assertFalse(incomplete["whole_inference_tokens_per_second"])
            self.assertEqual(failed_source["capacity_failures"][0]["concurrency"], 4)
            self.assertEqual(
                failed_source["capacity_failures"][0]["status"],
                "memory_admission_ineligible",
            )
            self.assertEqual(
                failed_source["capacity_failures"][0]["memory_admission"],
                {
                    "kind": "minimum_runtime_reservation",
                    "minimum_runtime_reservation_bytes": 10663212291,
                    "automatic_headroom_bytes": 1073741824,
                    "available_after_weights_bytes": 11477728256,
                    "shortfall_bytes": 259225859,
                },
            )
            self.assertTrue(_valid_capacity_failure(failed_source["capacity_failures"][0]))
            malformed_failure = copy.deepcopy(failed_source["capacity_failures"][0])
            malformed_failure["campaign_failure"]["returncode"] = 139
            self.assertFalse(_valid_capacity_failure(malformed_failure))
            stderr_path = (
                logs / "pareto_effective_capacity.effective_capacity_mtp3.c4.stderr.txt"
            )
            valid_stderr = stderr_path.read_text(encoding="utf-8")
            stderr_path.write_text("ninfer_bench: segmentation fault\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact startup admission failure"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], None, 4096, chunk_authority,
                )
            stderr_path.write_text(valid_stderr, encoding="utf-8")
            failures_path = roots["pareto-capacity"] / "failures.json"
            generic = json.loads(failures_path.read_text(encoding="utf-8"))
            generic[0]["returncode"] = 139
            failures_path.write_text(json.dumps(generic), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact process failure record"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], None, 4096, chunk_authority,
                )
            generic[0]["returncode"] = 1
            failures_path.write_text(json.dumps(generic), encoding="utf-8")
            excluded = classify({
                "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
                "required_quality_cells": ["8k", "32k"],
                "required_capacity_cells": [f"c{i}" for i in range(1, 5)],
                "required_speed_workloads": sorted(
                    candidate["whole_inference_tokens_per_second"]
                ),
                "candidates": [incomplete],
            })
            self.assertFalse(excluded["candidates"][0]["comparable"])
            self.assertIn(
                "missing_resolved_effective_maximum_capacity:c4",
                excluded["candidates"][0]["reasons"],
            )

            whole_manifest = roots["pareto-whole"] / "manifest.json"
            changed = json.loads(whole_manifest.read_text(encoding="utf-8"))
            changed["bench"]["sha256"] = "c" * 64
            whole_manifest.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different benchmark bytes"):
                assemble_candidate(
                    "candidate", "weights", 16, quality_path,
                    roots["pareto-capacity"], roots["pareto-whole"], 4096,
                    chunk_authority,
                )


if __name__ == "__main__":
    unittest.main()
