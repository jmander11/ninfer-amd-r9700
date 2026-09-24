#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
import json
import statistics
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tools.bench.select_prefill_chunk import (
    MATRIX_SCHEMA_VERSION,
    MEASUREMENT_SEMANTICS,
    REQUIRED_GROUPS,
    REQUIRED_PROFILES,
    REQUIRED_RECIPES,
    _manifest,
    _owned_path,
    _rank,
    _stable_manifest,
    build_cases,
    _validated_prefill_measurement,
    build_screening,
    build_selection,
    main,
    validate_screening_record,
    validate_selection_record,
)


def migration_receipt(weights_id: str) -> dict:
    recipe_id = {
        REQUIRED_RECIPES[0]: "r9700-all-q4g64-n16k16-eval-v1",
        REQUIRED_RECIPES[1]: "r9700-source-q4-n16k16-promoted-w8-source-mse8-eval-v1",
        REQUIRED_RECIPES[2]: "r9700-q4g64-f8e4m3-four-role-n16k16-eval-v1",
    }[weights_id]
    value = {"path": "/receipt.json", "sha256": "f" * 64, "recipe_id": recipe_id,
             "object_plan_sha256": "e" * 64, "source_artifact_sha256": "d" * 64,
             "source_receipt_sha256": "c" * 64, "transcoder_sha256": "b" * 64}
    if weights_id == REQUIRED_RECIPES[2]:
        value.update({"selection_sha256": "b2ceeb63c581c0f26aab5a4d8c0958da34d836fcc5c47d377bce709eaf37e3e8", "source_index_sha256": "9" * 64,
                      "source_ranking_sha256": "8" * 64})
    else:
        value["receipt_producer_sha256"] = "7" * 64
    return value


class PrefillChunkSelectionTest(unittest.TestCase):
    def test_selection_point_estimate_is_derived_from_retained_repetitions(self) -> None:
        seconds = (4.0, 5.0, 6.0)
        speeds = tuple(8192 / value for value in seconds)
        test = {
            "workspace_peak_bytes": 123,
            "prefill_seconds_mean": 5.0,
            "prefill_seconds_stddev": 1.0,
            "prefill_tok_s_mean": sum(speeds) / len(speeds),
            "prefill_tok_s_stddev": statistics.stdev(speeds),
            "reps": [
                {"timings": {"prefill_seconds": value}} for value in seconds
            ],
            "speculative": {
                "enabled": False, "draft_window": 0, "rounds": 0,
                "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
                "acceptance_rate": None, "acceptance_length": None,
                "accepted_per_position": [],
            },
        }
        throughput, workspace = _validated_prefill_measurement(
            test, 8192, Path("report.json")
        )
        self.assertAlmostEqual(throughput, sum(speeds) / len(speeds))
        self.assertEqual(workspace, 123)

        wrong_protocol = json.loads(json.dumps(test))
        wrong_protocol["speculative"]["rounds"] = 1
        with self.assertRaisesRegex(ValueError, "spec-none ordinary"):
            _validated_prefill_measurement(
                wrong_protocol, 8192, Path("report.json")
            )

        test["prefill_tok_s_mean"] *= 1.01
        with self.assertRaisesRegex(ValueError, "not derived from its retained repetitions"):
            _validated_prefill_measurement(test, 8192, Path("report.json"))

    def test_manifest_requires_successful_post_run_auto_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "artifact_type": "ninfer_bench_matrix_run",
                "schema_version": MATRIX_SCHEMA_VERSION,
                "preset": "prefill-chunk",
                "base_chunk_profile": "spec-none-ordinary",
                "dry_run": False,
                "failures": [],
                "concurrency": [1],
                "expected_kv_plane_layouts": {
                    "key": "token-fastest-head-major",
                    "value": "feature-fastest-page-major",
                    "value_scale": "feature-fastest-page-major",
                },
                "expected_q4_activation_bits": 8,
                "expected_w8_activation_bits": 8,
                "expected_fp8_qk_wmma_enabled": True,
                "bench": {"sha256": "b" * 64, "file_size_bytes": 1},
                "artifact": {
                    "model_id": "qwen3.8-27b",
                    "weights_id": REQUIRED_RECIPES[0],
                    "sha256": "a" * 64,
                    "file_size_bytes": 1,
                    "conversion_receipt": migration_receipt(REQUIRED_RECIPES[0]),
                },
                "expected_kv_value_group": REQUIRED_GROUPS[0],
                "expected_xattention_profile": REQUIRED_PROFILES[0],
                "corpus_sha256": "c" * 64,
                "power_profile": {"required": "auto", "observed": "auto"},
                "commands": [],
            }
            for post_state in (None, "profile_standard"):
                with self.subTest(post_state=post_state):
                    manifest = json.loads(json.dumps(base))
                    if post_state is not None:
                        manifest["power_profile"]["rechecked_after"] = post_state
                    (root / "manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        ValueError, "not a valid physical production prefill-chunk matrix"
                    ):
                        _manifest(root, 8192, (1024, 2048, 4096, 8192))

            base["power_profile"]["rechecked_after"] = "auto"
            (root / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not retain the exact"):
                _manifest(root, 8192, (1024, 2048, 4096, 8192))

    def test_manifest_rejects_report_outside_its_campaign_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "moved-diagnostic"
            official = root / "recreated-official"
            campaign.mkdir()
            official.mkdir()
            report = official / "prefill.json"
            report.write_text("{}", encoding="utf-8")
            cases = build_cases(
                "prefill-chunk", prefill_chunks=(1024, 2048, 4096, 8192),
                prefill_prompt=8192,
            )
            manifest = {
                "artifact_type": "ninfer_bench_matrix_run",
                "schema_version": MATRIX_SCHEMA_VERSION,
                "preset": "prefill-chunk", "dry_run": False, "concurrency": [1],
                "base_chunk_profile": "spec-none-ordinary",
                "expected_kv_plane_layouts": {
                    "key": "token-fastest-head-major",
                    "value": "feature-fastest-page-major",
                    "value_scale": "feature-fastest-page-major",
                },
                "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
                "expected_fp8_qk_wmma_enabled": True,
                "bench": {"sha256": "b" * 64, "file_size_bytes": 1},
                "artifact": {
                    "model_id": "qwen3.8-27b", "weights_id": REQUIRED_RECIPES[0],
                    "sha256": "a" * 64, "file_size_bytes": 1,
                },
                "expected_kv_value_group": REQUIRED_GROUPS[0],
                "expected_xattention_profile": REQUIRED_PROFILES[0],
                "corpus_sha256": "c" * 64,
                "power_profile": {
                    "required": "auto", "observed": "auto", "rechecked_after": "auto",
                },
                "commands": [
                    {
                        "suite": case.suite, "case": case.name, "concurrency": 1,
                        "report": str(report), "command": [],
                    }
                    for case in cases
                ],
            }
            (campaign / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside its output directory"):
                _manifest(campaign, 8192, (1024, 2048, 4096, 8192))

    def test_manifest_rejects_report_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            outside = root / "outside"
            campaign.mkdir()
            outside.mkdir()
            (campaign / "json").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside its output directory"):
                _owned_path(campaign, "json/report.json", "report")

    def test_stable_manifest_rejects_failure_marker_before_loading_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "failures.json").write_text("[]", encoding="utf-8")
            with patch("tools.bench.select_prefill_chunk._manifest") as load:
                with self.assertRaisesRegex(ValueError, "retains a failed matrix marker"):
                    _stable_manifest(root, 8192, (1024, 2048, 4096, 8192))
            load.assert_not_called()

    def test_stable_manifest_rejects_dangling_failure_marker_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "failures.json").symlink_to(root / "missing-outside-campaign")
            with patch("tools.bench.select_prefill_chunk._manifest") as load:
                with self.assertRaisesRegex(ValueError, "retains a failed matrix marker"):
                    _stable_manifest(root, 8192, (1024, 2048, 4096, 8192))
            load.assert_not_called()

    def test_global_maximin_uses_all_candidates_and_both_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = []
            identities = [
                (recipe, group, profile)
                for recipe in REQUIRED_RECIPES
                for group in REQUIRED_GROUPS
                for profile in REQUIRED_PROFILES
            ]
            by_root = {}
            for index, identity in enumerate(identities):
                screen = root / f"candidate-{index}-8k"
                final = root / f"candidate-{index}-32k"
                screen.mkdir()
                final.mkdir()
                (screen / "manifest.json").write_text("{}", encoding="utf-8")
                (final / "manifest.json").write_text("{}", encoding="utf-8")
                roots.append((screen, final))
                artifact = {
                    "model_id": "qwen3.8-27b", "weights_id": identity[0],
                    "sha256": chr(ord("a") + REQUIRED_RECIPES.index(identity[0])) * 64,
                    "file_size_bytes": 100 * (1 + REQUIRED_RECIPES.index(identity[0])),
                }
                hybrid = identity[0] == REQUIRED_RECIPES[2]
                artifact["conversion_receipt"] = migration_receipt(identity[0])
                common = {
                    "artifact": artifact, "bench": {"sha256": str(index) * 64},
                    "corpus_sha256": "c" * 64,
                    "expected_kv_value_group": identity[1],
                    "expected_kv_plane_layouts": {
                        "key": "token-fastest-head-major",
                        "value": "feature-fastest-page-major",
                        "value_scale": "feature-fastest-page-major",
                    },
                    "expected_q4_activation_bits": 8,
                    "expected_w8_activation_bits": 8,
                    "expected_fp8_qk_wmma_enabled": True,
                    "expected_fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                    "expected_xattention_profile": identity[2],
                    "required_candidate_identity": (
                        "fp8-hybrid-selection-authority" if hybrid else None
                    ),
                    "hybrid_shared_workspace_authority": (
                        {"tool": {"path": "/planner", "file_size_bytes": 1,
                                  "sha256": "e" * 64}}
                        if hybrid else None
                    ),
                }
                by_root[screen] = (common, {
                    1024: self.row(50, 900), 2048: self.row(90, 800),
                    4096: self.row(100, 1000), 8192: self.row(80, 1100),
                })
                by_root[final] = (common, {
                    4096: self.row(80, 1000), 2048: self.row(100, 800),
                })

            with patch(
                "tools.bench.select_prefill_chunk._manifest",
                side_effect=lambda candidate_root, _prompt, _chunks: by_root[candidate_root],
            ):
                result = build_selection(roots)
            self.assertEqual(result["candidate_count"], 12)
            self.assertEqual(result["finalist_chunks"], [4096, 2048])
            self.assertEqual(result["selected_prefill_chunk"], 2048)
            self.assertEqual(result["measurement_semantics"], MEASUREMENT_SEMANTICS)
            self.assertEqual(result["base_chunk_profile"], "spec-none-ordinary")
            self.assertEqual(
                result["measurement_semantics"]["configured_speculative_backend"], "none"
            )
            self.assertFalse(result["measurement_semantics"]["proposal_head_executed"])
            self.assertEqual(result["measurement_semantics"]["decode_rounds"], 0)
            self.assertEqual(len(result["final_ranking"][0]["normalized_throughput"]), 24)
            self.assertEqual(len(result["sources"]), 12)
            self.assertTrue(all(len(source["screen_reports"]) == 4 for source in result["sources"]))
            self.assertTrue(all(len(source["finalist_reports"]) == 2 for source in result["sources"]))

    def test_requires_exact_cartesian_candidate_set(self) -> None:
        roots = [(Path(f"/screen-{index}"), Path(f"/final-{index}")) for index in range(12)]
        identity = (REQUIRED_RECIPES[0], 16, "dense")
        manifest = {
            "artifact": {"model_id": "qwen3.8-27b", "weights_id": identity[0],
                         "sha256": "a" * 64, "file_size_bytes": 1,
                         "conversion_receipt": migration_receipt(identity[0])},
            "bench": {"sha256": "b" * 64, "file_size_bytes": 2},
            "corpus_sha256": "c" * 64,
            "expected_kv_value_group": identity[1], "expected_xattention_profile": identity[2],
        }
        with patch(
            "tools.bench.select_prefill_chunk._manifest",
            return_value=(manifest, {chunk: self.row(1, 1) for chunk in (1024, 2048, 4096, 8192)}),
        ), patch("tools.bench.select_prefill_chunk.file_sha256", return_value="f" * 64):
            with self.assertRaisesRegex(ValueError, "duplicate prefill-chunk candidate"):
                build_selection(roots)

    def test_workspace_then_smaller_chunk_break_exact_throughput_tie(self) -> None:
        rows = {
            ("recipe", 16, "dense"): {
                1024: self.row(10, 20), 2048: self.row(10, 10), 4096: self.row(10, 10),
            }
        }
        self.assertEqual([row["chunk"] for row in _rank((1024, 2048, 4096), rows)], [2048, 4096, 1024])

    def test_final_rank_preserves_full_screen_normalizer(self) -> None:
        rows = {
            ("candidate-a", 8192): {
                1024: self.row(90, 10), 2048: self.row(80, 10),
            },
            ("candidate-a", 32768): {
                1024: self.row(80, 10), 2048: self.row(100, 10),
            },
        }
        # The eliminated 8K leader reached 100. Retaining that denominator makes both finalists'
        # maximin ratio 0.8, so the smaller-chunk tie-break selects 1024. Re-normalizing against
        # only the two finalists would incorrectly select 2048.
        normalizers = {
            ("candidate-a", 8192): 100.0,
            ("candidate-a", 32768): 100.0,
        }
        self.assertEqual(_rank((1024, 2048), rows, normalizers)[0]["chunk"], 1024)

    def test_screening_rebuilds_exact_twelve_candidate_authority(self) -> None:
        roots = [Path(f"/screen-{index}") for index in range(12)]
        identities = [
            (recipe, group, profile)
            for recipe in REQUIRED_RECIPES
            for group in REQUIRED_GROUPS
            for profile in REQUIRED_PROFILES
        ]
        loaded = []
        for index, identity in enumerate(identities):
            manifest = {
                "artifact": {
                    "model_id": "qwen3.8-27b", "weights_id": identity[0],
                    "sha256": chr(ord("a") + REQUIRED_RECIPES.index(identity[0])) * 64,
                    "file_size_bytes": 1,
                    "conversion_receipt": migration_receipt(identity[0]),
                },
                "corpus_sha256": "c" * 64,
                "expected_kv_value_group": identity[1],
                "expected_xattention_profile": identity[2],
                "required_candidate_identity": (
                    "fp8-hybrid-selection-authority"
                    if identity[0] == REQUIRED_RECIPES[2] else None
                ),
            }
            reports = {
                1024: self.row(50, 900), 2048: self.row(90, 800),
                4096: self.row(100, 1000), 8192: self.row(80, 1100),
            }
            loaded.append((manifest, reports, {
                "path": f"/screen-{index}/manifest.json", "sha256": str(index) * 64,
            }))
        with patch(
            "tools.bench.select_prefill_chunk._stable_manifest", side_effect=loaded
        ):
            result = build_screening(roots)
        self.assertEqual(result["candidate_count"], 12)
        self.assertEqual(result["finalist_chunks"], [4096, 2048])
        self.assertEqual(len(result["sources"]), 12)
        self.assertTrue(all(len(source["screen_reports"]) == 4 for source in result["sources"]))

    def test_screening_record_is_recomputed_before_emitting_finalists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "screening.json"
            payload = {"finalist_chunks": [4096, 2048]}
            record.write_text(json.dumps(payload), encoding="utf-8")
            roots = [Path(f"/screen-{index}") for index in range(12)]
            with patch(
                "tools.bench.select_prefill_chunk.build_screening", return_value=payload
            ) as rebuild, patch(
                "tools.bench.select_prefill_chunk.file_sha256", return_value="a" * 64
            ):
                self.assertEqual(validate_screening_record(record, roots), payload)
                rebuild.assert_called_once_with(roots)

            changed = {"finalist_chunks": [2048, 4096]}
            record.write_text(json.dumps(changed), encoding="utf-8")
            with patch(
                "tools.bench.select_prefill_chunk.build_screening", return_value=payload
            ), patch(
                "tools.bench.select_prefill_chunk.file_sha256", return_value="a" * 64
            ), self.assertRaisesRegex(ValueError, "does not match its source evidence"):
                validate_screening_record(record, roots)

    def test_verify_screening_cli_prints_only_validated_finalists(self) -> None:
        arguments = [
            part
            for index in range(12)
            for part in ("--screen", f"/screen-{index}")
        ] + ["--verify-screening", "/screening.json"]
        output = StringIO()
        with patch(
            "tools.bench.select_prefill_chunk.validate_screening_record",
            return_value={"finalist_chunks": [4096, 2048]},
        ) as validate, redirect_stdout(output):
            self.assertEqual(main(arguments), 0)
        self.assertEqual(output.getvalue(), "4096\n2048\n")
        validate.assert_called_once_with(
            Path("/screening.json"), [Path(f"/screen-{index}") for index in range(12)]
        )

    def test_record_reopens_bound_manifest_bytes_and_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            roots = []
            for index in range(12):
                screen = root / f"screen-{index}" / "manifest.json"
                final = root / f"final-{index}" / "manifest.json"
                screen.parent.mkdir()
                final.parent.mkdir()
                screen.write_text(f"screen-{index}", encoding="utf-8")
                final.write_text(f"final-{index}", encoding="utf-8")
                screen_reports = []
                final_reports = []
                for chunk in (1024, 2048, 4096, 8192):
                    report = screen.parent / f"screen-{chunk}.json"
                    report.write_text(f"screen-{index}-{chunk}", encoding="utf-8")
                    screen_reports.append({
                        "chunk": chunk, "path": str(report), "sha256": self.digest(report),
                    })
                for chunk in (2048, 4096):
                    report = final.parent / f"final-{chunk}.json"
                    report.write_text(f"final-{index}-{chunk}", encoding="utf-8")
                    final_reports.append({
                        "chunk": chunk, "path": str(report), "sha256": self.digest(report),
                    })
                roots.append((screen.parent, final.parent))
                sources.append({
                    "screen_manifest": {"path": str(screen), "sha256": self.digest(screen)},
                    "finalist_manifest": {"path": str(final), "sha256": self.digest(final)},
                    "screen_reports": screen_reports,
                    "finalist_reports": final_reports,
                })
            payload = {
                "artifact_type": "ninfer_r9700_prefill_chunk_selection",
                "schema_version": 2,
                "selected_prefill_chunk": 2048,
                "sources": sources,
            }
            record = root / "selection.json"
            record.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "tools.bench.select_prefill_chunk.build_selection", return_value=payload
            ) as rebuild:
                self.assertEqual(validate_selection_record(record), payload)
                rebuild.assert_called_once_with(roots)
            screen = Path(sources[0]["screen_manifest"]["path"])
            screen.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source bytes changed"):
                validate_selection_record(record)

    def test_record_rejects_changed_bound_raw_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            report = root / "report.json"
            manifest.write_text("manifest", encoding="utf-8")
            report.write_text("report", encoding="utf-8")
            source = {
                "screen_manifest": {"path": str(manifest), "sha256": self.digest(manifest)},
                "finalist_manifest": {"path": str(manifest), "sha256": self.digest(manifest)},
                "screen_reports": [
                    {"chunk": chunk, "path": str(report), "sha256": self.digest(report)}
                    for chunk in (1024, 2048, 4096, 8192)
                ],
                "finalist_reports": [
                    {"chunk": chunk, "path": str(report), "sha256": self.digest(report)}
                    for chunk in (2048, 4096)
                ],
            }
            record = root / "selection.json"
            record.write_text(json.dumps({
                "artifact_type": "ninfer_r9700_prefill_chunk_selection",
                "schema_version": 2,
                "sources": [source] * 12,
            }), encoding="utf-8")
            report.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source bytes changed"):
                validate_selection_record(record)

    @staticmethod
    def row(speed: float, workspace: int) -> dict:
        return {
            "prefill_tok_s_mean": float(speed),
            "workspace_peak_bytes": workspace,
            "report": {"path": "/report", "sha256": "d" * 64},
        }

    @staticmethod
    def digest(path: Path) -> str:
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
