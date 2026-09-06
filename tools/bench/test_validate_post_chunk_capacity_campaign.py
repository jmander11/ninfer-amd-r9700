#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from tools.bench.select_prefill_chunk import REQUIRED_GROUPS, REQUIRED_PROFILES, REQUIRED_RECIPES
from tools.bench.matrix_contract import R9700_KV_PLANE_LAYOUTS
from tools.bench.run_ninfer_bench_matrix import (
    FP8_QK_WMMA_PROFILE,
    FP8_QK_WMMA_T1_MIN_CONTEXT,
    FP8_QK_WMMA_T2_MIN_CONTEXT,
)
from tools.bench.validate_post_chunk_capacity_campaign import (
    EXPECTED_IDENTITIES,
    validate_campaign,
    validate_matrix,
)

STATIC_CONTRACT = {
    "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
    "expected_q4_activation_bits": 8,
    "expected_w8_activation_bits": 8,
    "expected_fp8_qk_wmma_enabled": True,
    "expected_fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
    "expected_fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
    "expected_fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
}


class PostChunkCapacityCampaignTest(unittest.TestCase):
    @staticmethod
    def selection_record() -> dict:
        return {
            "selected_prefill_chunk": 2048,
            "sources": [
                {
                    "screen_manifest": {
                        "path": f"/screen-{index}-receipt-bound-n16k16-20260905/manifest.json"
                    },
                    "finalist_manifest": {
                        "path": f"/final-{index}-receipt-bound-n16k16-20260905/manifest.json"
                    },
                }
                for index in range(12)
            ],
        }

    def test_gpu_script_is_capacity_first_and_uses_validated_pair_eligibility(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "profiles/bench/post-chunk-twelve-candidate-20260905/run.sh"
        ).read_text(encoding="utf-8")
        last_capacity = script.rindex("run_capacity 32 b128-s16-tau900")
        validation = script.index("--executed --out \"$capacity_validation\"")
        first_whole = script.index("run_whole 16 dense dense all-q4")
        self.assertLess(last_capacity, validation)
        self.assertLess(validation, first_whole)
        self.assertIn('result["whole_eligible_identities"]', script)

    def test_terminal_selection_consumes_receipt_bound_conditional_campaign(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "profiles/bench/terminal-static-selection-20260905/assemble-and-select.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "prefill-chunk-selection-receipt-bound-n16k16-20260905.json", script
        )
        self.assertIn("--post-chunk-capacity-validation", script)
        self.assertEqual(script.count("pareto-capacity-post-promotion-"), 12)
        self.assertEqual(script.count("$(whole_arg "), 12)
        self.assertNotIn("-g16-n16k16-20260905", script)
        self.assertNotIn("-g32-n16k16-20260905", script)

    def test_campaign_requires_exact_cartesian_set_and_twelve_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(f"/matrix-{index}") for index in range(12)]
            authority = Path(directory) / (
                "prefill-chunk-selection-receipt-bound-n16k16-20260905.json"
            )
            authority.write_text(json.dumps(self.selection_record()), encoding="utf-8")
            bound = {"selected_prefill_chunk": 2048}
            with mock.patch(
                "tools.bench.validate_post_chunk_capacity_campaign.validate_prefill_chunk_authority",
                return_value=(bound, self.selection_record()),
            ), mock.patch(
                "tools.bench.validate_post_chunk_capacity_campaign.validate_matrix",
                side_effect=[(identity, 0) for identity in EXPECTED_IDENTITIES],
            ):
                result = validate_campaign(authority, roots, executed=False)
            self.assertEqual((result["matrix_count"], result["point_count"]), (12, 48))
            self.assertEqual(result["validation_scope"], "prepared")
            self.assertNotIn("successful_point_count", result)
            self.assertNotIn("capacity_eligible_matrix_count", result)
            self.assertNotIn("whole_eligible_identities", result)

            with self.assertRaisesRegex(ValueError, "exactly twelve"):
                validate_campaign(authority, roots[:-1], executed=False)

    def test_duplicate_candidate_identity_fails_closed(self) -> None:
        roots = [Path(f"/matrix-{index}") for index in range(12)]
        duplicate = (REQUIRED_RECIPES[0], REQUIRED_GROUPS[0], REQUIRED_PROFILES[0])
        with mock.patch.object(
            Path,
            "resolve",
            return_value=Path(
                "/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"
            ),
        ), mock.patch.object(
            Path, "read_text", return_value=json.dumps(self.selection_record())
        ), mock.patch(
            "tools.bench.validate_post_chunk_capacity_campaign.validate_prefill_chunk_authority",
            return_value=({"selected_prefill_chunk": 2048}, self.selection_record()),
        ), mock.patch(
            "tools.bench.validate_post_chunk_capacity_campaign.validate_matrix",
            return_value=(duplicate, 0),
        ):
            with self.assertRaisesRegex(ValueError, "exact Cartesian"):
                validate_campaign(
                    Path("/prefill-chunk-selection-receipt-bound-n16k16-20260905.json"),
                    roots, executed=False,
                )

    def test_asymmetric_attention_capacity_eligibility_fails_closed(self) -> None:
        roots = [Path(f"/matrix-{index}") for index in range(12)]
        authority = Path("/selection.json")
        ordered = sorted(EXPECTED_IDENTITIES)
        failed_identity = ordered[0]
        paired_identity = (
            failed_identity[0], failed_identity[1],
            next(profile for profile in REQUIRED_PROFILES if profile != failed_identity[2]),
        )
        outcomes = [
            (identity, 1 if identity == failed_identity else 0) for identity in ordered
        ]
        with mock.patch(
            "tools.bench.validate_post_chunk_capacity_campaign.validate_prefill_chunk_authority",
            return_value=(
                {"selected_prefill_chunk": 2048}, self.selection_record(),
            ),
        ), mock.patch(
            "tools.bench.validate_post_chunk_capacity_campaign.validate_matrix",
            side_effect=outcomes,
        ):
            with self.assertRaisesRegex(ValueError, "differs across matched"):
                validate_campaign(authority, roots, executed=True)

        outcomes = [
            (identity, 1 if identity in (failed_identity, paired_identity) else 0)
            for identity in ordered
        ]
        with mock.patch(
            "tools.bench.validate_post_chunk_capacity_campaign.validate_prefill_chunk_authority",
            return_value=(
                {"selected_prefill_chunk": 2048}, self.selection_record(),
            ),
        ), mock.patch(
            "tools.bench.validate_post_chunk_capacity_campaign.validate_matrix",
            side_effect=outcomes,
        ):
            result = validate_campaign(authority, roots, executed=True)
        self.assertEqual(result["capacity_eligible_matrix_count"], 10)
        self.assertEqual(result["whole_eligible_matrix_count"], 10)
        self.assertNotIn(failed_identity, result["whole_eligible_identities"])
        self.assertNotIn(paired_identity, result["whole_eligible_identities"])

    def test_matrix_requires_mtp3_optimized_head_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            authority = {
                "path": "/selection.json",
                "sha256": "a" * 64,
                "artifact_type": "ninfer_r9700_prefill_chunk_selection",
                "schema_version": 2,
                "selected_prefill_chunk": 2048,
            }
            common = [
                "bench", "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
                "--kv-capacity", "auto", "--max-ctx", "262144", "--prefill-chunk", "2048",
            ]
            manifest = {
                "artifact_type": "ninfer_bench_matrix_run", "schema_version": 14,
                "preset": "pareto-capacity", "dry_run": False, "prepare_only": True,
                "post_chunk_capacity_gate": True, "prefill_chunk_authority": authority,
                "selected_prefill_chunk": 2048, "concurrency": [1, 2, 3, 4],
                "case_count": 1, "point_count": 4,
                "artifact": {"weights_id": REQUIRED_RECIPES[0]},
                "expected_kv_value_group": 16, "expected_xattention_profile": "dense",
                **STATIC_CONTRACT,
                "commands": [
                    {"suite": "pareto_effective_capacity", "case": "effective_capacity_mtp3",
                     "concurrency": concurrency,
                     "report": str(root / f"report-c{concurrency}.json"),
                     "command": [*common, "--concurrency", str(concurrency), "--output-file",
                                 str(root / f"report-c{concurrency}.json")]}
                    for concurrency in (1, 2, 3, 4)
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "tools.bench.validate_post_chunk_capacity_campaign.validate_n16_receipt_summary"
            ):
                self.assertEqual(
                    validate_matrix(root, authority, executed=False),
                    ((REQUIRED_RECIPES[0], 16, "dense"), 0),
                )
                manifest["commands"][0]["command"].append("--no-device-graph")
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Device Graph"):
                    validate_matrix(root, authority, executed=False)

    def test_prepared_matrix_is_not_executed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = {"selected_prefill_chunk": 2048}
            command = [
                "bench", "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
                "--kv-capacity", "auto", "--max-ctx", "262144", "--prefill-chunk", "2048",
                "--concurrency", "1", "--output-file", str(root / "report.json"),
            ]
            manifest = {
                "artifact_type": "ninfer_bench_matrix_run", "schema_version": 14,
                "preset": "pareto-capacity", "dry_run": False, "prepare_only": True,
                "post_chunk_capacity_gate": True, "prefill_chunk_authority": authority,
                "selected_prefill_chunk": 2048, "concurrency": [1, 2, 3, 4],
                "case_count": 1, "point_count": 4,
                "artifact": {"weights_id": REQUIRED_RECIPES[0]},
                "expected_kv_value_group": 16, "expected_xattention_profile": "dense",
                **STATIC_CONTRACT,
                "commands": [
                    {
                        "suite": "pareto_effective_capacity",
                        "case": "effective_capacity_mtp3", "concurrency": concurrency,
                        "report": str(root / f"report-c{concurrency}.json"),
                        "command": [
                            *(command[:-4]), "--concurrency", str(concurrency),
                            "--output-file", str(root / f"report-c{concurrency}.json"),
                        ],
                    }
                    for concurrency in (1, 2, 3, 4)
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "tools.bench.validate_post_chunk_capacity_campaign.validate_n16_receipt_summary"
            ):
                validate_matrix(root, authority, executed=False)
                with self.assertRaisesRegex(ValueError, "exact schema-v14"):
                    validate_matrix(root, authority, executed=True)

    def test_record_concurrency_must_match_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = {
                "path": "/selection.json", "sha256": "a" * 64,
                "artifact_type": "ninfer_r9700_prefill_chunk_selection",
                "schema_version": 2, "selected_prefill_chunk": 2048,
            }
            common = [
                "bench", "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
                "--kv-capacity", "auto", "--max-ctx", "262144", "--prefill-chunk", "2048",
            ]
            records = []
            for concurrency in (1, 2, 3, 4):
                report = root / f"report-c{concurrency}.json"
                records.append({
                    "suite": "pareto_effective_capacity", "case": "effective_capacity_mtp3",
                    "concurrency": concurrency, "report": str(report),
                    "command": [*common, "--concurrency", "4", "--output-file", str(report)],
                })
            manifest = {
                "artifact_type": "ninfer_bench_matrix_run", "schema_version": 14,
                "preset": "pareto-capacity", "dry_run": False, "prepare_only": True,
                "post_chunk_capacity_gate": True, "prefill_chunk_authority": authority,
                "selected_prefill_chunk": 2048, "concurrency": [1, 2, 3, 4],
                "case_count": 1, "point_count": 4,
                "artifact": {"weights_id": REQUIRED_RECIPES[0]},
                "expected_kv_value_group": 16, "expected_xattention_profile": "dense",
                **STATIC_CONTRACT,
                "commands": records,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "tools.bench.validate_post_chunk_capacity_campaign.validate_n16_receipt_summary"
            ), self.assertRaisesRegex(ValueError, "--concurrency 1"):
                validate_matrix(root, authority, executed=False)

    def test_executed_matrix_requires_exact_retained_capacity_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            authority = {"selected_prefill_chunk": 2048}
            common = [
                "bench", "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
                "--kv-capacity", "auto", "--max-ctx", "262144", "--prefill-chunk", "2048",
            ]
            records, failures = [], []
            for concurrency in (1, 2, 3, 4):
                stem = f"pareto_effective_capacity.effective_capacity_mtp3.c{concurrency}"
                report = root / f"report-c{concurrency}.json"
                stdout = root / "logs" / f"{stem}.stdout.txt"
                stderr = root / "logs" / f"{stem}.stderr.txt"
                command = [
                    *common, "--weights", "model.ninfer", "--concurrency", str(concurrency),
                    "--output-file", str(report),
                ]
                stderr.write_text(
                    f"[ninfer_bench] loading model.ninfer (max_context=262144, "
                    f"concurrency={concurrency}, kv_format=fp8-k-int4-v)\n"
                    "ninfer_bench: automatic KV headroom requires 1073741824 bytes, but only "
                    "10 bytes are available after weights\n",
                    encoding="utf-8",
                )
                stdout.write_text("", encoding="utf-8")
                records.append({
                    "suite": "pareto_effective_capacity", "case": "effective_capacity_mtp3",
                    "concurrency": concurrency, "report": str(report), "command": command,
                })
                failures.append({
                    "suite": "pareto_effective_capacity", "case": "effective_capacity_mtp3",
                    "concurrency": concurrency, "returncode": 1,
                    "stdout": str(stdout), "stderr": str(stderr), "command": command,
                })
            manifest = {
                "artifact_type": "ninfer_bench_matrix_run", "schema_version": 14,
                "preset": "pareto-capacity", "dry_run": False, "prepare_only": False,
                "post_chunk_capacity_gate": True, "prefill_chunk_authority": authority,
                "selected_prefill_chunk": 2048, "concurrency": [1, 2, 3, 4],
                "case_count": 1, "point_count": 4,
                "artifact": {"weights_id": REQUIRED_RECIPES[0]},
                "expected_kv_value_group": 16,
                "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
                "expected_fp8_qk_wmma": 1, "expected_xattention_profile": "dense",
                **STATIC_CONTRACT,
                "commands": records,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "tools.bench.validate_post_chunk_capacity_campaign.validate_n16_receipt_summary"
            ), mock.patch(
                "tools.bench.validate_post_chunk_capacity_campaign.build_cases",
                return_value=[object()],
            ):
                with self.assertRaisesRegex(ValueError, "no failures.json"):
                    validate_matrix(root, authority, executed=True)
                (root / "failures.json").write_text(json.dumps(failures), encoding="utf-8")
                first_stderr = Path(failures[0]["stderr"])
                structured_stderr = first_stderr.read_text(encoding="utf-8")
                first_stderr.write_text("capacity unavailable\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "startup admission failure"):
                    validate_matrix(root, authority, executed=True)
                first_stderr.write_text(structured_stderr, encoding="utf-8")
                identity, failed = validate_matrix(root, authority, executed=True)
                self.assertEqual(identity, (REQUIRED_RECIPES[0], 16, "dense"))
                self.assertEqual(failed, 4)


if __name__ == "__main__":
    unittest.main()
