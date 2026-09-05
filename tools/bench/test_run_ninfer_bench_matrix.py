#!/usr/bin/env python3
"""Dependency-light report identity tests for the native benchmark matrix."""

from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from tools.bench.run_ninfer_bench_matrix import (
    _validate_speculative,
    NINFER_PREFIX,
    MATRIX_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
    BenchCase,
    add_repetition_args,
    bind_dflash_diagnostic,
    bind_n16_migration_receipt,
    build_hybrid_shared_workspace_authority,
    build_cases,
    dflash_shortlist_frontier,
    dflash_shortlist_profiles,
    durable_replace_text,
    file_sha256,
    inspect_artifact,
    load_bench_report,
    main,
    manifest_owned_path,
    require_auto_power_profile,
    require_fp8_hybrid_artifact,
    validate_fp8_hybrid_performance_contract,
    validate_hybrid_shared_workspace_authority,
    validate_automatic_feasibility,
    validate_dflash_campaign_artifact,
    validate_dflash_diagnostic_raw,
    validate_case_profile,
    validate_report_tests,
    validate_whole_mtp_command,
    write_dflash_determinism,
    write_dflash_greedy_parity,
    write_dflash_quality_evidence,
    write_dflash_shortlist,
)
from tools.bench.run_serve_concurrency import main as serve_concurrency_main
from tools.bench.run_serve_corpus import CampaignError
from tools.ppl.run import validate_n16_receipt_summary


class CompiledKvGroupTest(unittest.TestCase):
    def test_n16_receipt_summary_rejects_minimal_dict(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_n16_receipt_summary(
                {"sha256": "a" * 64}, "r9700-q4g64-n16k16-eval")

    def test_n16_receipt_summary_rejects_cross_profile_recipe(self) -> None:
        value = {
            "path": "/receipt", "sha256": "1" * 64,
            "recipe_id": "r9700-source-q4-n16k16-promoted-w8-source-mse8-eval-v1",
            "object_plan_sha256": "2" * 64,
            "source_artifact_sha256": "3" * 64,
            "source_receipt_sha256": "4" * 64,
            "transcoder_sha256": "5" * 64,
            "receipt_producer_sha256": "6" * 64,
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_n16_receipt_summary(value, "r9700-q4g64-n16k16-eval")

    def test_all_n16_recipes_attach_exact_migration_receipt(self) -> None:
        for weights_id in (
            "r9700-q4g64-n16k16-eval",
            "r9700-q4-w8-mse-n16k16-eval",
            "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
        ):
            with self.subTest(weights_id=weights_id), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.ninfer"; path.write_bytes(b"artifact")
                artifact = {"path": str(path.resolve()), "file_size_bytes": 8,
                            "sha256": "a" * 64, "model_id": "qwen3.8-27b",
                            "weights_id": weights_id}
                inspected = {"path": artifact["path"], "bytes": 8,
                             "sha256": "a" * 64, "model_id": "qwen3.8-27b",
                             "weights_id": weights_id,
                             "conversion_receipt": {"sha256": "b" * 64}}
                with mock.patch(
                    "tools.bench.run_ninfer_bench_matrix.ppl_run.inspect_candidate_artifact",
                    return_value=inspected,
                ):
                    self.assertEqual(
                        bind_n16_migration_receipt(path, artifact)["conversion_receipt"],
                        inspected["conversion_receipt"],
                    )

    def write_artifact(self, path: Path, weights_id: str, object_count: int = 0) -> None:
        directory = json.dumps(
            {"identity": {"model_id": "qwen3.8-27b", "weights_id": weights_id},
             "objects": [{} for _ in range(object_count)]},
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(NINFER_PREFIX.pack(b"NINFER\x00\x02", len(directory)) + directory)

    def write_report(
        self,
        path: Path,
        group: int,
        concurrency: int = 1,
        artifact: str = "model.ninfer",
        command: str = "ninfer_bench --weights model.ninfer",
        w8_activation_bits: int = 16,
        fp8_qk_wmma: bool = True,
        xattention_profile: str = "dense",
    ) -> None:
        xattention = (
            {"xattention_qualification": False}
            if xattention_profile == "dense"
            else {
                "xattention_qualification": True,
                "xattention_profile": "b128-s16-tau900",
                "xattention_find_block": 128,
                "xattention_stride": 16,
                "xattention_tau_permille": 900,
            }
        )

        path.write_text(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "artifact_type": "ninfer_bench_report",
                    "tool": "ninfer_bench",
                    "command": command,
                    "artifact": {"path": artifact},
                    "memory": {"kv_capacity_mode": "explicit"},
                    "config": {
                        "kv_value_group": group,
                        "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                        "q4_activation_bits": 8,
                        "q4_prefill_cta_profile": "m64n128-pingpong-production",
                        "w8_activation_bits": w8_activation_bits,
                        "fp8_qk_wmma_enabled": fp8_qk_wmma,
                        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                        "fp8_qk_wmma_t1_min_context": 64,
                        "fp8_qk_wmma_t2_min_context": 320,
                        **xattention,
                        "concurrency": concurrency,
                        "pending_timeout_ms": 0xFFFFFFFF,
                        "pending_deadline": "unbounded",
                        "spec": "none",
                        "draft_tokens": 0,
                        "speculative_execution": False,
                        "dflash_verify_width_requested": 0,
                        "dflash_verify_width": 0,
                        "proposal_head": "full",
                        "use_device_graph": True,
                        "retain_token_ids": False,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_hybrid_performance_contract_is_exact_c1_to_c4(self) -> None:
        args = SimpleNamespace(
            require_fp8_hybrid=True,
            preset="pareto-whole",
            concurrency=[1, 2, 3, 4],
            prefill_chunk=[4096],
            expected_kv_value_group=16,
            expected_xattention_profile="dense",
            suite=[], limit=None, repetitions=None, warmup=None,
            hybrid_width_tool=Path("planner"),
        )
        validate_fp8_hybrid_performance_contract(args)
        args.expected_kv_value_group = 32
        args.expected_xattention_profile = "b128-s16-tau900"
        validate_fp8_hybrid_performance_contract(args)
        args.concurrency = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(SystemExit, "exactly C=1,2,3,4"):
            validate_fp8_hybrid_performance_contract(args)
        args.concurrency = [1, 2, 3, 4]
        args.prefill_chunk = [8192]
        validate_fp8_hybrid_performance_contract(args)
        args.preset = "prefill-chunk"
        args.concurrency = [1]
        args.prefill_chunk = [1024, 2048, 4096, 8192]
        validate_fp8_hybrid_performance_contract(args)
        args.preset = "low-context-prefill"
        args.prefill_chunk = [8192]
        validate_fp8_hybrid_performance_contract(args)
        args.preset = "dflash-shortlist"
        args.prefill_chunk = [2048]
        validate_fp8_hybrid_performance_contract(args)
        args.preset = "dflash-pareto"
        args.concurrency = [1, 2, 3, 4]
        validate_fp8_hybrid_performance_contract(args)
        args.preset = "dflash-capacity"
        validate_fp8_hybrid_performance_contract(args)
        args.preset = "dflash-feasibility"
        with self.assertRaisesRegex(SystemExit, "supports only"):
            validate_fp8_hybrid_performance_contract(args)

    def test_durable_replace_is_atomic_and_preserves_a_raced_foreign_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            durable_replace_text(path, "first\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "first\n")
            durable_replace_text(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "second\n")
            real_exchange = __import__(
                "tools.bench.run_ninfer_bench_matrix", fromlist=["_rename_exchange"]
            )._rename_exchange
            raced = False

            def replace_then_exchange(left: Path, right: Path) -> None:
                nonlocal raced
                if not raced:
                    raced = True
                    replacement = right.with_name("foreign-replacement")
                    replacement.write_text("foreign\n", encoding="utf-8")
                    os.replace(replacement, right)
                real_exchange(left, right)

            with mock.patch(
                "tools.bench.run_ninfer_bench_matrix._rename_exchange",
                side_effect=replace_then_exchange,
            ), self.assertRaisesRegex(ValueError, "changed during publication"):
                durable_replace_text(path, "third\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "foreign\n")
            self.assertEqual(list(Path(directory).glob(".*.pending-*")), [])

            target = Path(directory) / "target"
            target.write_text("outside\n", encoding="utf-8")
            alias = Path(directory) / "alias"
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "cannot be opened safely|not a regular file"):
                durable_replace_text(alias, "clobber\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "outside\n")

    def test_durable_replace_rolls_back_when_directory_fsync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "manifest.json"
            durable_replace_text(existing, "old\n")
            with mock.patch(
                "tools.bench.run_ninfer_bench_matrix._fsync_directory",
                side_effect=OSError("injected fsync failure"),
            ), self.assertRaisesRegex(OSError, "injected"):
                durable_replace_text(existing, "new\n")
            self.assertEqual(existing.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(list(root.glob(".*.pending-*")), [])

            created = root / "summary.json"
            with mock.patch(
                "tools.bench.run_ninfer_bench_matrix._fsync_directory",
                side_effect=OSError("injected fsync failure"),
            ), self.assertRaisesRegex(OSError, "injected"):
                durable_replace_text(created, "new\n")
            self.assertFalse(os.path.lexists(created))
            self.assertEqual(list(root.glob(".*.pending-*")), [])

    def test_hybrid_width_authority_covers_every_requested_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool = Path(directory) / "planner"
            tool.write_bytes(b"planner")

            def widths(_tool: Path, chunk: int, concurrency: int, drafts: int):
                self.assertEqual(concurrency, 4)
                return (
                    [1, 2, 3, 4, chunk]
                    if drafts == 0 else [1, 2, 3, 4, 8, 12, 16, chunk]
                )

            with mock.patch(
                "tools.bench.run_ninfer_bench_matrix.query_widths", side_effect=widths
            ):
                authority = build_hybrid_shared_workspace_authority(
                    tool, [8192, 1024, 4096, 2048]
                )
            self.assertEqual(authority["prefill_chunks"], [1024, 2048, 4096, 8192])
            self.assertEqual(
                set(authority["inventories_by_prefill_chunk"]),
                {"1024", "2048", "4096", "8192"},
            )
            validate_hybrid_shared_workspace_authority(
                authority, [1024, 2048, 4096, 8192]
            )
            with self.assertRaisesRegex(ValueError, "width authority differs"):
                validate_hybrid_shared_workspace_authority(authority, [2048, 4096])

    def test_expected_compiled_group_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16)
            self.assertEqual(
                load_bench_report(report, 16, 8, 16, True)["config"]["kv_value_group"], 16
            )
            with self.assertRaisesRegex(ValueError, "expected compiled G32"):
                load_bench_report(report, 32, 8, 16, True)

    def test_dflash_campaign_accepts_each_terminal_recipe_companion(self) -> None:
        mixed = {
            "model_id": "qwen3.8-27b",
            "weights_id": "r9700-q4-w8-mse-n16k16-dflash2-q4-eval",
        }
        validate_dflash_campaign_artifact("dflash-pareto", mixed, dry_run=False)
        validate_dflash_campaign_artifact("pareto", mixed, dry_run=False)
        validate_dflash_campaign_artifact("dflash-pareto", mixed, dry_run=True)
        validate_dflash_campaign_artifact(
            "dflash-pareto",
            {"model_id": "qwen3.8-27b", "weights_id": "r9700-q4g64-n16k16-dflash2-q4-eval"},
            dry_run=False,
        )
        validate_dflash_campaign_artifact(
            "dflash-pareto",
            {"model_id": "qwen3.8-27b", "weights_id":
             "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval"},
            dry_run=False,
        )
        with self.assertRaisesRegex(SystemExit, "registered DFlash companion"):
            validate_dflash_campaign_artifact(
                "dflash-pareto",
                {"model_id": "qwen3.8-27b", "weights_id": "unregistered"},
                dry_run=False,
            )

    def test_hybrid_dflash_companion_binds_exact_base_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.ninfer"
            companion = root / "companion.ninfer"
            self.write_artifact(base, "r9700-q4g64-f8e4m3-four-role-n16k16-eval")
            self.write_artifact(
                companion, "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval", 1190
            )
            base_artifact = inspect_artifact(base)
            companion_artifact = inspect_artifact(companion)
            receipt = {
                "path": str(root / "base.conversion.json"),
                "sha256": "1" * 64,
                "recipe_id": "recipe",
                "selection_sha256": "2" * 64,
                "object_plan_sha256": "3" * 64,
                "source_index_sha256": "4" * 64,
                "source_ranking_sha256": "5" * 64,
            }
            authority = {
                "receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
                **{key: receipt[key] for key in (
                    "recipe_id", "selection_sha256", "object_plan_sha256",
                    "source_index_sha256", "source_ranking_sha256",
                )},
            }
            report = {
                "target_key": "qwen3_8_27b_r9700",
                "recipe_id": "r9700-dflash2-all-q4g64-n16k16-bf16-codebook-eval-v1",
                "status": "registered-evaluation-only",
                "weight_recipe_selected": False,
                "identity": {
                    "model_id": "qwen3.8-27b",
                    "weights_id": companion_artifact["weights_id"],
                },
                "base": {
                    "path": str(base),
                    "identity": {
                        "model_id": "qwen3.8-27b",
                        "weights_id": base_artifact["weights_id"],
                    },
                    "bytes": base_artifact["file_size_bytes"],
                    "sha256": base_artifact["sha256"],
                    "payload_copy": "byte_exact",
                    "authority": authority,
                },
                "artifact": {
                    "path": str(companion),
                    "bytes": companion_artifact["file_size_bytes"],
                    "sha256": companion_artifact["sha256"],
                    "projected_bytes": companion_artifact["file_size_bytes"],
                    "projected_device_arena_bytes": 1,
                },
                "dflash_recipe": {
                    "matrix_format": "Q4G64_F16S",
                    "activation_profile": "compile_selected_adaptive_A8G64",
                    "selector_codebook_format": "BF16",
                    "objects": 66,
                    "source_tensors": 81,
                    "format_counts": {"BF16": 34, "Q4G64_F16S": 32},
                    "format_encoded_bytes": {
                        "BF16": 254_814_720, "Q4G64_F16S": 954_654_720,
                    },
                    "tensor_encoded_bytes": 1_209_469_440,
                    "runtime_repack": False,
                },
                "dflash_source": {
                    "config_sha256": "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980",
                    "readme_sha256": "0c06405ffff835f4da26115114a6dd7bb4a8b8a6881c17edd3a1086a99281269",
                    "tensor_count": 81,
                    "safetensors_bytes": 3_848_817_896,
                    "safetensors_sha256": "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
                },
            }
            report_path = Path(str(companion) + ".conversion.json")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            hybrid = {
                "path": str(base.resolve()), "bytes": base_artifact["file_size_bytes"],
                "sha256": base_artifact["sha256"], "model_id": "qwen3.8-27b",
                "weights_id": base_artifact["weights_id"], "conversion_receipt": receipt,
            }
            with mock.patch(
                "tools.bench.run_ninfer_bench_matrix.ppl_run.inspect_candidate_artifact",
                return_value=hybrid,
            ), mock.patch(
                "tools.bench.run_ninfer_bench_matrix.ppl_run.require_fp8_hybrid_candidate"
            ):
                bound = require_fp8_hybrid_artifact(
                    companion, companion_artifact, "dflash-pareto"
                )
                self.assertEqual(bound["hybrid_base_artifact"]["conversion_receipt"], receipt)
                report["base"]["authority"]["selection_sha256"] = "f" * 64
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "does not bind"):
                    require_fp8_hybrid_artifact(
                        companion, companion_artifact, "dflash-pareto"
                    )

    def test_unrecognized_compiled_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 64)
            with self.assertRaisesRegex(ValueError, "invalid kv_value_group"):
                load_bench_report(report)

    def test_missing_or_mislabeled_plane_layout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16)
            payload = json.loads(report.read_text(encoding="utf-8"))
            del payload["config"]["kv_plane_layouts"]
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "kv_plane_layouts"):
                load_bench_report(report)
            payload["config"]["kv_plane_layouts"] = {
                **R9700_KV_PLANE_LAYOUTS,
                "key": "feature-fastest-page-major",
            }
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "kv_plane_layouts"):
                load_bench_report(report)

    def test_expected_compiled_q4_activation_width_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16)
            self.assertEqual(
                load_bench_report(report, 16, 8, 16, True)["config"]["q4_activation_bits"], 8
            )
            with self.assertRaisesRegex(ValueError, "expected compiled A4"):
                load_bench_report(report, 16, 4, 16, True)

    def test_q4_prefill_cta_build_identity_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16)
            self.assertEqual(
                load_bench_report(report)["config"]["q4_prefill_cta_profile"],
                "m64n128-pingpong-production",
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["config"]["q4_prefill_cta_profile"] = "m64n128-production"
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "q4_prefill_cta_profile"):
                load_bench_report(report)

    def test_expected_w8_and_fp8_qk_profiles_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16, w8_activation_bits=8, fp8_qk_wmma=False)
            with self.assertRaisesRegex(ValueError, "expected compiled A16"):
                load_bench_report(report, 16, 8, 16, False)
            with self.assertRaisesRegex(ValueError, "expected True"):
                load_bench_report(report, 16, 8, 8, True)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["config"]["fp8_qk_wmma_t2_min_context"] = 1
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "t2_min_context=1"):
                load_bench_report(report)

    def test_compile_bound_xattention_profile_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16, xattention_profile="b128-s16-tau900")
            loaded = load_bench_report(
                report, expected_xattention_profile="b128-s16-tau900"
            )
            self.assertEqual(
                loaded["config"]["xattention_profile"], "b128-s16-tau900"
            )
            with self.assertRaisesRegex(ValueError, "xattention_qualification=True"):
                load_bench_report(report)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["config"]["xattention_tau_permille"] = 901
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "xattention_tau_permille=901"):
                load_bench_report(
                    report, expected_xattention_profile="b128-s16-tau900"
                )

    def test_empty_report_cannot_satisfy_a_matrix_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["config"].update({"repetitions": 1, "warmup": 0})
            payload["tests"] = []
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no test rows"):
                load_bench_report(
                    report, expected_case=BenchCase(
                        "pure_decode", "tg8", ("-n", "8"), repetitions=1, warmup=0
            )
        )

    def test_artifact_inspection_rejects_retired_q4_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.ninfer"
            payload = json.dumps({
                "identity": {"model_id": "qwen3.8-27b",
                             "weights_id": "r9700-q4g64-f8e4m3-four-role-eval"},
                "objects": [{"kind": "tensor", "name": "weight", "shape": [16, 128],
                             "format": "Q4G64_F16S", "layout": "row-split-k128-v1",
                             "offset": 0, "bytes": 1088}],
            }, separators=(",", ":")).encode()
            path.write_bytes(NINFER_PREFIX.pack(b"NINFER\x00\x02", len(payload)) + payload)
            with self.assertRaisesRegex(SystemExit, "retired non-N16/K16 Q4 storage"):
                inspect_artifact(path)

    def test_prefill_accepts_enabled_profile_without_decode_acceptance_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.write_report(report, 16)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["config"].update({
                "repetitions": 1, "warmup": 0, "draft_tokens": 3, "spec": "mtp",
                "speculative_execution": True,
            })
            payload["tests"] = [
                {
                    "label": "pp8192",
                    "kind": "pp",
                    "n_prompt": 8192,
                    "n_gen": 0,
                    "requested_output_tokens": 1,
                    "workspace_peak_bytes": 1,
                    "workspace_allocator_peak_bytes": 1,
                    "prefill_seconds_mean": 1.0,
                    "prefill_tok_s_mean": 8192.0,
                    "total_seconds_mean": 1.1,
                    "speculative": {
                        "enabled": True,
                        "draft_window": 3,
                        "rounds": 0,
                        "drafted_tokens": 0,
                        "accepted_tokens": 0,
                        "fallback_steps": 0,
                        "acceptance_rate": None,
                        "acceptance_length": None,
                        "accepted_per_position": [0, 0, 0],
                    },
                    "reps": [{}],
                }
            ]
            report.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_bench_report(
                report,
                expected_case=BenchCase(
                    "pareto_prefill",
                    "pareto_prefill_mtp3",
                    ("-p", "8192", "--draft-tokens", "3"),
                    repetitions=1,
                    warmup=0,
                ),
            )
            self.assertTrue(loaded["tests"][0]["speculative"]["enabled"])
            payload["config"]["warmup"] = 1
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "warmup=1; expected 0"):
                load_bench_report(
                    report,
                    expected_case=BenchCase(
                        "pareto_prefill", "pareto_prefill_mtp3",
                        ("-p", "8192", "--draft-tokens", "3"),
                        repetitions=1, warmup=0,
                    ),
                )

    def test_expected_concurrency_and_artifact_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "selected.ninfer"
            report = root / "report.json"
            self.write_report(report, 16, 4, str(artifact))
            artifact_provenance = {
                "path": str(artifact),
                "file_size_bytes": 0,
                "sha256": "0" * 64,
                "model_id": "qwen3.8-27b",
                "weights_id": "candidate",
            }
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["artifact"]["file_size_bytes"] = 0
            payload["load"] = {"target": "qwen3_8_27b_r9700", "weights_id": "candidate"}
            report.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_bench_report(
                    report,
                    16,
                    8,
                    16,
                    True,
                    4,
                    artifact_provenance,
                    ("ninfer_bench", "--weights", "model.ninfer"),
                )["config"]["concurrency"],
                4,
            )
            with self.assertRaisesRegex(ValueError, "expected C=2"):
                load_bench_report(report, 16, 8, 16, True, 2, artifact_provenance)
            with self.assertRaisesRegex(ValueError, "expected .*other.ninfer"):
                load_bench_report(
                    report, 16, 8, 16, True, 4,
                    {**artifact_provenance, "path": str(root / "other.ninfer")},
                )
            with self.assertRaisesRegex(ValueError, "command does not match"):
                load_bench_report(
                    report,
                    16,
                    8,
                    16,
                    True,
                    4,
                    artifact_provenance,
                    ("ninfer_bench", "--weights", "other.ninfer"),
                )

    def test_concurrency_dry_run_accepts_future_artifact_and_writes_all_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            artifact = root / "eventual-selected.ninfer"
            argv = [
                "--preset",
                "concurrency",
                "--weights",
                str(artifact),
                "--output-dir",
                str(output),
                "--dry-run",
            ]
            for concurrency in range(1, 5):
                argv.extend(("--concurrency", str(concurrency)))
            self.assertEqual(main(argv), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], MATRIX_SCHEMA_VERSION)
            self.assertEqual(manifest["expected_q4_activation_bits"], 8)
            self.assertEqual(manifest["expected_w8_activation_bits"], 8)
            self.assertTrue(manifest["expected_fp8_qk_wmma_enabled"])
            self.assertEqual(manifest["expected_fp8_qk_wmma_t1_min_context"], 64)
            self.assertEqual(manifest["expected_fp8_qk_wmma_t2_min_context"], 320)
            self.assertEqual(manifest["expected_xattention_profile"], "dense")
            self.assertEqual(manifest["case_count"], 3)
            self.assertEqual(manifest["point_count"], 12)
            self.assertEqual(manifest["concurrency"], list(range(1, 5)))
            self.assertEqual(
                {record["concurrency"] for record in manifest["commands"]}, set(range(1, 5))
            )
            self.assertTrue(
                all("--concurrency" in record["command"] for record in manifest["commands"])
            )

    def test_product_matrix_rejects_concurrency_above_four(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(SystemExit, r"\[1, 4\]"):
                main([
                    "--preset", "pareto-capacity",
                    "--weights", str(root / "eventual-selected.ninfer"),
                    "--output-dir", str(root / "matrix"),
                    "--concurrency", "5", "--dry-run",
                ])

    def test_prefill_chunk_sweep_is_fixed_c1_and_supports_32k_finalists(self) -> None:
        cases = build_cases("prefill-chunk")
        self.assertEqual(len(cases), 4)
        self.assertEqual(
            [case.args[case.args.index("--prefill-chunk") + 1] for case in cases],
            ["1024", "2048", "4096", "8192"],
        )
        self.assertTrue(all(case.concurrency_one_only for case in cases))
        self.assertTrue(all((case.repetitions, case.warmup) == (3, 1) for case in cases))
        self.assertTrue(all("--draft-tokens" in case.args and "3" in case.args
                            for case in cases))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            self.assertEqual(main([
                "--preset", "prefill-chunk",
                "--weights", str(root / "eventual-selected.ninfer"),
                "--output-dir", str(output),
                "--prefill-prompt", "32768",
                "--prefill-chunk", "2048",
                "--prefill-chunk", "8192",
                "--expected-xattention-profile", "b128-s16-tau900",
                "--dry-run",
            ]), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["case_count"], manifest["point_count"]), (2, 2))
            self.assertEqual(
                manifest["power_profile"],
                {
                    "required": "auto",
                    "sysfs_path": (
                        "/sys/class/drm/card2/device/power_dpm_force_performance_level"
                    ),
                    "observed": "not-checked-dry-run",
                    "rechecked_after": None,
                },
            )
            self.assertEqual(manifest["corpus_sha256"], file_sha256(Path(
                "bench/fixtures/bench_corpus.ids"
            )))
            power_guard = (
                'test "$(cat /sys/class/drm/card2/device/'
                'power_dpm_force_performance_level)" = auto'
            )
            commands_script = (output / "commands.sh").read_text(encoding="utf-8")
            self.assertEqual(commands_script.count(power_guard), 2)
            self.assertTrue(commands_script.rstrip().endswith(power_guard))
            commands = [record["command"] for record in manifest["commands"]]
            self.assertTrue(all("32768" in command for command in commands))
            self.assertEqual(
                {command[command.index("--prefill-chunk") + 1] for command in commands},
                {"2048", "8192"},
            )

        for preset in ("pareto", "pareto-whole", "pareto-feasibility", "pareto-capacity"):
            cases = build_cases(preset, production_prefill_chunk=8192)
            self.assertTrue(cases)
            self.assertTrue(all(
                case.args[case.args.index("--prefill-chunk") + 1] == "8192"
                for case in cases
            ))

    def test_prefill_chunk_sweep_rejects_non_c1_and_duplicate_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = [
                "--preset", "prefill-chunk",
                "--weights", str(root / "eventual-selected.ninfer"),
                "--dry-run",
            ]
            with self.assertRaisesRegex(SystemExit, "fixed at C=1"):
                main([*common, "--concurrency", "2"])
            with self.assertRaisesRegex(SystemExit, "duplicate --prefill-chunk"):
                main([*common, "--prefill-chunk", "2048", "--prefill-chunk", "2048"])
            with self.assertRaisesRegex(SystemExit, "8K prefill-chunk screening requires exactly"):
                main([*common, "--prefill-chunk", "2048", "--prefill-chunk", "4096"])
            with self.assertRaisesRegex(SystemExit, "requires exactly two finalists"):
                main([
                    *common, "--prefill-prompt", "32768", "--prefill-chunk", "4096",
                ])

    def test_low_context_prefill_is_strict_dense_c1_prefill_only_ladder(self) -> None:
        cases = build_cases("low-context-prefill", production_prefill_chunk=2048)
        self.assertEqual(
            [int(case.args[case.args.index("-p") + 1]) for case in cases],
            [128, 512, 1024, 2048, 4096],
        )
        self.assertTrue(all((case.repetitions, case.warmup) == (3, 1) for case in cases))
        self.assertTrue(all("-pg" not in case.args and "--whole-pg" not in case.args
                            for case in cases))
        self.assertTrue(all(
            case.args[case.args.index("--draft-tokens") + 1] == "0" for case in cases
        ))
        self.assertTrue(all(
            case.args[case.args.index("--prefill-chunk") + 1] == "2048"
            for case in cases
        ))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "ladder"
            self.assertEqual(main([
                "--preset", "low-context-prefill",
                "--weights", str(root / "selected.ninfer"),
                "--bench", str(root / "selected-bench"),
                "--prefill-chunk", "2048",
                "--expected-kv-value-group", "16",
                "--expected-xattention-profile", "dense",
                "--output-dir", str(output),
                "--dry-run",
            ]), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["case_count"], manifest["point_count"]), (5, 5))
            self.assertEqual(manifest["selected_prefill_chunk"], 2048)
            self.assertEqual(manifest["concurrency"], [1])
            self.assertEqual(manifest["expected_xattention_profile"], "dense")
            self.assertEqual(len({record["report"] for record in manifest["commands"]}), 5)
            power_guard = (
                'test "$(cat /sys/class/drm/card2/device/'
                'power_dpm_force_performance_level)" = auto'
            )
            script = (output / "commands.sh").read_text(encoding="utf-8")
            self.assertEqual(script.count(power_guard), 2)
            self.assertTrue(script.rstrip().endswith(power_guard))

            common = [
                "--preset", "low-context-prefill",
                "--weights", str(root / "selected.ninfer"),
                "--prefill-chunk", "2048", "--dry-run",
            ]
            with self.assertRaisesRegex(SystemExit, "requires --expected-kv-value-group"):
                main(common)
            with self.assertRaisesRegex(SystemExit, "dense attention profile"):
                main([*common, "--expected-kv-value-group", "16",
                      "--expected-xattention-profile", "b128-s16-tau900"])
            with self.assertRaisesRegex(SystemExit, "fixed at C=1"):
                main([*common, "--expected-kv-value-group", "16", "--concurrency", "2"])

    def test_pareto_whole_binds_power_profile_and_selected_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "whole"
            self.assertEqual(main([
                "--preset", "pareto-whole",
                "--weights", str(root / "eventual-selected.ninfer"),
                "--bench", str(root / "eventual-bench"),
                "--concurrency", "1", "--concurrency", "2",
                "--concurrency", "3", "--concurrency", "4",
                "--prefill-chunk", "2048",
                "--output-dir", str(output), "--dry-run",
            ]), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["case_count"], manifest["point_count"]), (2, 8))
            self.assertEqual(manifest["selected_prefill_chunk"], 2048)
            self.assertEqual(manifest["power_profile"], {
                "required": "auto",
                "sysfs_path": "/sys/class/drm/card2/device/power_dpm_force_performance_level",
                "observed": "not-checked-dry-run", "rechecked_after": None,
            })
            self.assertTrue(all(
                record["command"].count("--prefill-chunk") == 1
                and record["command"][record["command"].index("--prefill-chunk") + 1]
                == "2048"
                for record in manifest["commands"]
            ))
            mtp_records = [
                record for record in manifest["commands"]
                if record["suite"] == "pareto_whole_inference"
            ]
            ordinary_records = [
                record for record in manifest["commands"]
                if record["suite"] == "pareto_whole_control"
            ]
            self.assertEqual((len(mtp_records), len(ordinary_records)), (4, 4))
            self.assertTrue(all(
                record["command"].count("--lm-head-draft") == 1
                and record["command"][record["command"].index("--spec") + 1] == "mtp"
                and record["command"][record["command"].index("--draft-tokens") + 1] == "3"
                and record["command"].count("--retain-token-ids") == 1
                for record in mtp_records
            ))
            self.assertTrue(all(
                record["command"][record["command"].index("--draft-tokens") + 1] == "0"
                and "--lm-head-draft" not in record["command"]
                and record["command"].count("--retain-token-ids") == 1
                for record in ordinary_records
            ))
            self.assertIn(
                'test "$(cat /sys/class/drm/card2/device/'
                'power_dpm_force_performance_level)" = auto',
                (output / "commands.sh").read_text(encoding="utf-8"),
            )

    def test_pareto_whole_requires_executed_optimized_mtp_head(self) -> None:
        case = build_cases("pareto-whole")[0]
        command = ["ninfer_bench", *case.args]
        validate_whole_mtp_command(case, command)
        with self.assertRaisesRegex(ValueError, "exactly one --lm-head-draft"):
            validate_whole_mtp_command(
                case, [part for part in command if part != "--lm-head-draft"]
            )

        config = {
            "spec": "mtp", "draft_tokens": 3, "speculative_execution": True,
            "dflash_verify_width_requested": 0, "dflash_verify_width": 0,
            "proposal_head": "optimized", "use_device_graph": True,
            "retain_token_ids": True, "repetitions": 3, "warmup": 1,
            "prefill_chunk": 4096, "concurrency": 1,
        }
        validate_case_profile(config, case)
        with self.assertRaisesRegex(ValueError, "proposal_head='full'"):
            validate_case_profile({**config, "proposal_head": "full"}, case)

        def whole_row(tokens: int) -> dict[str, object]:
            return {
                "label": f"whole-pp{tokens}+tg256", "kind": "whole",
                "n_prompt": tokens, "n_gen": 256, "requested_output_tokens": 257,
                "workspace_peak_bytes": 1, "workspace_allocator_peak_bytes": 1,
                "prefill_seconds_mean": 1.0, "prefill_tok_s_mean": float(tokens),
                "decode_seconds_mean": 1.0, "decode_output_tok_s_mean": 256.0,
                "decode_engine_tok_s_mean": 512.0, "whole_output_tok_s_mean": 128.0,
                "total_seconds_mean": 2.0,
                "speculative": {
                    "enabled": True, "draft_window": 3, "rounds": 0,
                    "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
                    "acceptance_rate": None, "acceptance_length": None,
                    "accepted_per_position": [0, 0, 0],
                },
                "reps": [{"generated_token_ids_by_lane": [[1] * 257]} for _ in range(3)],
            }

        report = {"config": config, "tests": [whole_row(8192), whole_row(32768)]}
        with self.assertRaisesRegex(ValueError, "no speculative acceptance sample"):
            validate_report_tests(report, case)

        for row in report["tests"]:
            row["speculative"].update({
                "rounds": 192,
                "drafted_tokens": 576,
                "acceptance_rate": 0.0,
                "acceptance_length": 1.0,
            })
        with self.assertRaisesRegex(ValueError, "no accepted speculative tokens"):
            validate_report_tests(report, case)

    def test_generic_speculative_diagnostic_allows_zero_acceptance(self) -> None:
        _validate_speculative(
            {
                "enabled": True, "draft_window": 3, "rounds": 192,
                "drafted_tokens": 576, "accepted_tokens": 0, "fallback_steps": 192,
                "acceptance_rate": 0.0, "acceptance_length": 1.0,
                "accepted_per_position": [0, 0, 0],
            },
            enabled=True,
            draft_window=3,
            require_sample=True,
            require_accepted=False,
            label="diagnostic",
        )

    def test_ordinary_diagnostic_is_exact_c1_8k_non_speculative(self) -> None:
        cases = build_cases("ordinary-diagnostic")
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual((case.repetitions, case.warmup), (3, 1))
        self.assertTrue(case.concurrency_one_only)
        self.assertEqual(case.args, (
            "--whole-pg", "8192,256", "--prefill-chunk", "4096",
            "--spec", "mtp", "--draft-tokens", "0",
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "ordinary"
            self.assertEqual(main([
                "--preset", "ordinary-diagnostic",
                "--weights", str(root / "eventual-all-q4.ninfer"),
                "--bench", str(root / "eventual-g16-xattention-bench"),
                "--concurrency", "1",
                "--expected-kv-value-group", "16",
                "--expected-xattention-profile", "b128-s16-tau900",
                "--output-dir", str(output), "--dry-run",
            ]), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["power_profile"], {
                "required": "auto",
                "sysfs_path": "/sys/class/drm/card2/device/power_dpm_force_performance_level",
                "observed": "not-checked-dry-run", "rechecked_after": None,
            })
            command = manifest["commands"][0]["command"]
            self.assertEqual(command[command.index("--draft-tokens") + 1], "0")
            self.assertEqual(command[command.index("--concurrency") + 1], "1")
            self.assertIn("--whole-pg", command)
            self.assertIn(
                'test "$(cat /sys/class/drm/card2/device/'
                'power_dpm_force_performance_level)" = auto',
                (output / "commands.sh").read_text(encoding="utf-8"),
            )

    def test_ordinary_diagnostic_rejects_non_c1_and_resume(self) -> None:
        common = ["--preset", "ordinary-diagnostic", "--weights", "/eventual.ninfer",
                  "--dry-run"]
        with self.assertRaisesRegex(SystemExit, "fixed at C=1"):
            main([*common, "--concurrency", "2"])
        with self.assertRaisesRegex(SystemExit, "fresh complete run"):
            main([*common, "--resume"])
            with self.assertRaisesRegex(SystemExit, "accepts exactly one"):
                main([
                    "--preset", "pareto-whole",
                    "--weights", str(root / "eventual-selected.ninfer"),
                    "--prefill-chunk", "2048", "--prefill-chunk", "4096",
                    "--dry-run",
                ])

    def test_prefill_chunk_power_profile_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "power_dpm_force_performance_level"
            profile.write_text("auto\n", encoding="utf-8")
            self.assertEqual(require_auto_power_profile(profile), "auto")

            profile.write_text("profile_standard\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires stable R9700 power profile auto"):
                require_auto_power_profile(profile)

            profile.unlink()
            with self.assertRaisesRegex(ValueError, "cannot read R9700 power profile"):
                require_auto_power_profile(profile)

    def test_dry_run_manifest_binds_xattention_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            self.assertEqual(main([
                "--preset", "pareto-capacity",
                "--weights", str(root / "eventual-selected.ninfer"),
                "--output-dir", str(output),
                "--prefill-chunk", "4096",
                "--expected-xattention-profile", "b128-s16-tau900",
                "--dry-run",
            ]), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["expected_xattention_profile"], "b128-s16-tau900"
            )

    def test_fresh_campaign_rejects_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "already exists"):
                main([
                    "--preset", "pareto-capacity",
                    "--weights", str(root / "eventual-selected.ninfer"),
                    "--output-dir", str(output),
                    "--prefill-chunk", "4096",
                    "--expected-xattention-profile", "b128-s16-tau900",
                    "--dry-run",
                ])
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
            self.assertEqual(list(output.iterdir()), [marker])

    def test_pareto_matrix_retains_8k_32k_prefill_decode_at_c1_through_c4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            argv = [
                "--preset", "pareto",
                "--weights", str(root / "eventual-selected.ninfer"),
                "--prefill-chunk", "4096",
                "--output-dir", str(output), "--dry-run",
            ]
            for concurrency in range(1, 5):
                argv.extend(("--concurrency", str(concurrency)))
            self.assertEqual(main(argv), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            commands = manifest["commands"]
            self.assertEqual(manifest["case_count"], 2)
            self.assertEqual(manifest["point_count"], 8)
            self.assertEqual({record["concurrency"] for record in commands}, set(range(1, 5)))
            for concurrency in range(1, 5):
                selected = [record["command"] for record in commands
                            if record["concurrency"] == concurrency]
                self.assertTrue(any("8192,32768" in " ".join(command)
                    for command in selected))
                self.assertTrue(any("8192,256;32768,256" in " ".join(command)
                                    and "--draft-tokens 3" in " ".join(command)
                                    for command in selected))
                self.assertTrue(all("--prefill-chunk 4096" in " ".join(command)
                                    for command in selected))

    def test_pareto_supplements_retain_whole_inference_and_workload_feasibility(self) -> None:
        whole = build_cases("pareto-whole")
        self.assertEqual(len(whole), 2)
        self.assertEqual([case.parity_role for case in whole], ["mtp", "ordinary"])
        self.assertTrue(all(case.retain_token_ids for case in whole))
        self.assertIn("--whole-pg", whole[0].args)
        self.assertEqual(
            whole[0].args[whole[0].args.index("--whole-pg") + 1],
            "8192,256;32768,256",
        )
        self.assertEqual(
            whole[0].args[whole[0].args.index("--prefill-chunk") + 1], "4096"
        )
        self.assertEqual((whole[0].repetitions, whole[0].warmup), (3, 1))
        capacity = build_cases("pareto-feasibility")
        self.assertEqual(len(capacity), 1)
        self.assertEqual(
            capacity[0].args[capacity[0].args.index("--kv-capacity") + 1], "auto"
        )
        self.assertEqual(capacity[0].args[capacity[0].args.index("--max-ctx") + 1], "33030")
        effective_capacity = build_cases("pareto-capacity")
        self.assertEqual(
            effective_capacity[0].args[effective_capacity[0].args.index("--max-ctx") + 1],
            "262144",
        )
        self.assertEqual(
            effective_capacity[0].args[
                effective_capacity[0].args.index("--prefill-chunk") + 1
            ],
            "4096",
        )
        dflash_capacity = build_cases("dflash-feasibility", 7, 12)
        self.assertEqual(len(dflash_capacity), 1)
        self.assertIn("--kv-capacity", dflash_capacity[0].args)
        self.assertIn("dflash", dflash_capacity[0].args)
        self.assertEqual(
            dflash_capacity[0].args[dflash_capacity[0].args.index("--max-ctx") + 1],
            "33048",
        )
        dflash_effective_capacity = build_cases("dflash-capacity", 7, 12)
        self.assertEqual(
            dflash_effective_capacity[0].args[
                dflash_effective_capacity[0].args.index("--max-ctx") + 1
            ],
            "262144",
        )
        for preset in (
            "dflash-shortlist", "dflash-pareto", "dflash-feasibility", "dflash-capacity",
        ):
            cases = build_cases(
                preset,
                7 if preset != "dflash-shortlist" else None,
                12 if preset != "dflash-shortlist" else 0,
                production_prefill_chunk=8192,
            )
            self.assertTrue(all(
                case.args[case.args.index("--prefill-chunk") + 1] == "8192"
                for case in cases
            ))

    def test_automatic_feasibility_distinguishes_memory_from_logical_ceiling(self) -> None:
        report = {
            "config": {
                "max_context": 33030,
                "concurrency": 4,
                "kv_cache_format": "fp8-k-int4-v",
            },
            "memory": {
                "max_context": 33030,
                "kv_cache_format": "fp8-k-int4-v",
                "kv_capacity": 2000 * 64,
                "kv_capacity_page_groups": 2000,
                "kv_capacity_max_page_groups": 2068,
                "minimum_runtime_reservation_bytes": 1000,
                "kv_capacity_increment_bytes": 10,
                "runtime_reservation_bytes": 1000 + (2000 - 517) * 10,
                "available_after_weights_bytes": 1000 + (2000 - 517) * 10
                    + 1024 * 1024 * 1024 + 9,
                "kv_capacity_headroom_bytes": 1024 * 1024 * 1024,
                "planned_slack_bytes": 1024 * 1024 * 1024 + 9,
                "device_graph_allowance_bytes": 4096,
                "device_graph_observed_bytes": 3072,
            },
        }
        classified = validate_automatic_feasibility(report)
        self.assertEqual(classified["binding_constraint"], "device_memory")
        self.assertEqual(classified["memory_limited_capacity_tokens"], 2000 * 64)

        report["memory"]["kv_capacity_page_groups"] = 1999
        with self.assertRaisesRegex(ValueError, "token/page resolution"):
            validate_automatic_feasibility(report)

        report["memory"]["kv_capacity_page_groups"] = 2000
        report["memory"]["planned_slack_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "headroom/slack accounting"):
            validate_automatic_feasibility(report)

        report["memory"]["planned_slack_bytes"] -= 1
        report["memory"]["available_after_weights_bytes"] = (
            report["memory"]["runtime_reservation_bytes"]
            + report["memory"]["planned_slack_bytes"]
        )
        report["memory"]["kv_capacity"] = 2068 * 64
        report["memory"]["kv_capacity_page_groups"] = 2068
        report["memory"]["runtime_reservation_bytes"] = 1000 + (2068 - 517) * 10
        report["memory"]["available_after_weights_bytes"] = (
            report["memory"]["runtime_reservation_bytes"]
            + 1024 * 1024 * 1024 + 100
        )
        report["memory"]["planned_slack_bytes"] = 1024 * 1024 * 1024 + 100
        classified = validate_automatic_feasibility(report)
        self.assertEqual(classified["binding_constraint"], "logical_context_ceiling")
        self.assertIsNone(classified["memory_limited_capacity_tokens"])

        report["config"]["max_context"] = 262144
        report["memory"]["max_context"] = 262144
        report["memory"]["kv_capacity_max_page_groups"] = 4 * 4096
        report["memory"]["kv_capacity"] = 5000 * 64
        report["memory"]["kv_capacity_page_groups"] = 5000
        report["memory"]["minimum_runtime_reservation_bytes"] = 1000
        report["memory"]["runtime_reservation_bytes"] = 1000 + (5000 - 4096) * 10
        report["memory"]["available_after_weights_bytes"] = (
            report["memory"]["runtime_reservation_bytes"] + 1024 * 1024 * 1024 + 9
        )
        report["memory"]["planned_slack_bytes"] = 1024 * 1024 * 1024 + 9
        classified = validate_automatic_feasibility(report)
        self.assertEqual(classified["measurement_kind"], "resolved_effective_maximum")
        self.assertEqual(classified["binding_constraint"], "device_memory")

        report["config"]["concurrency"] = 1
        report["memory"]["max_context"] = 262144
        report["memory"]["kv_capacity"] = 4096 * 64
        report["memory"]["kv_capacity_page_groups"] = 4096
        report["memory"]["kv_capacity_max_page_groups"] = 4096
        report["memory"]["minimum_runtime_reservation_bytes"] = 1000
        report["memory"]["kv_capacity_increment_bytes"] = 0
        report["memory"]["runtime_reservation_bytes"] = 1000
        report["memory"]["available_after_weights_bytes"] = 1000 + 2 * 1024 * 1024 * 1024
        report["memory"]["planned_slack_bytes"] = 2 * 1024 * 1024 * 1024
        classified = validate_automatic_feasibility(report)
        self.assertEqual(classified["measurement_kind"], "resolved_effective_maximum")
        self.assertEqual(classified["binding_constraint"], "model_context")
        self.assertTrue(classified["uncensored"])

        report["memory"]["device_graph_observed_bytes"] = 4097
        with self.assertRaisesRegex(ValueError, "Device Graph observed allocation"):
            validate_automatic_feasibility(report)
        report["memory"]["device_graph_observed_bytes"] = 3072
        report["memory"]["device_graph_allowance_bytes"] = True
        with self.assertRaisesRegex(ValueError, "Device Graph allowance"):
            validate_automatic_feasibility(report)
        report["memory"]["device_graph_allowance_bytes"] = 4096
        del report["memory"]["device_graph_observed_bytes"]
        with self.assertRaisesRegex(ValueError, "Device Graph observed allocation"):
            validate_automatic_feasibility(report)

    def test_pareto_campaigns_require_and_bind_one_selected_prefill_chunk(self) -> None:
        for preset in ("pareto", "pareto-whole", "pareto-feasibility", "pareto-capacity"):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                common = [
                    "--preset", preset,
                    "--weights", str(root / "eventual-selected.ninfer"),
                    "--dry-run",
                ]
                with self.assertRaisesRegex(SystemExit, "requires exactly one --prefill-chunk"):
                    main(common)
                output = root / "matrix"
                self.assertEqual(main([
                    *common, "--prefill-chunk", "2048", "--output-dir", str(output),
                ]), 0)
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["selected_prefill_chunk"], 2048)
                self.assertTrue(all(
                    record["command"][record["command"].index("--prefill-chunk") + 1]
                    == "2048"
                    for record in manifest["commands"]
                ))

    def test_resume_rejects_same_path_replacement_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "selected.ninfer"
            output = root / "matrix"
            output.mkdir()
            self.write_artifact(artifact, "candidate-a")
            original = inspect_artifact(artifact)
            (output / "manifest.json").write_text(
                json.dumps(
                    {"artifact_type": "ninfer_bench_matrix_run",
                     "schema_version": MATRIX_SCHEMA_VERSION,
                     "artifact": original}
                ),
                encoding="utf-8",
            )
            self.write_artifact(artifact, "candidate-b")
            self.assertEqual(artifact.stat().st_size, original["file_size_bytes"])
            with self.assertRaisesRegex(SystemExit, "artifact bytes or identity differ"):
                main(
                    ["--preset", "smoke", "--weights", str(artifact),
                     "--output-dir", str(output), "--resume", "--no-build"]
                )

    def test_resume_rejects_report_path_outside_campaign_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            artifact = root / "eventual-selected.ninfer"
            args = [
                "--preset", "smoke", "--weights", str(artifact),
                "--output-dir", str(output), "--dry-run",
            ]
            self.assertEqual(main(args), 0)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["commands"][0]["report"] = str(root / "recreated-official.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "outside its output directory"):
                main([*args, "--resume"])

    def test_manifest_owned_path_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            outside = root / "official"
            campaign.mkdir()
            outside.mkdir()
            (campaign / "json").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside its output directory"):
                manifest_owned_path(campaign, "json/report.json", "report")

    def test_dflash_pareto_retains_matched_cells_parity_and_isolated_diagnostic(self) -> None:
        cases = build_cases("dflash-pareto", 7, 12)
        expected_repetitions = {
            "prefill_p8192_p32768_dflash": (3, 1),
            "context_p8192_p32768_g256_dflash_graph": (3, 1),
            "context_p8192_p32768_g256_ordinary_graph": (3, 1),
            "whole_p8192_p32768_g256_dflash_graph": (3, 1),
            "context_p8192_g256_dflash_eager_diagnostic_a": (1, 0),
            "context_p8192_g256_dflash_eager_diagnostic_b": (1, 0),
        }
        self.assertEqual({case.name for case in cases}, set(expected_repetitions))
        for case in cases:
            self.assertIs(type(case.repetitions), int)
            self.assertIs(type(case.warmup), int)
            self.assertEqual(
                (case.repetitions, case.warmup), expected_repetitions[case.name]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            argv = [
                "--preset", "dflash-pareto",
                "--prefill-chunk", "2048",
                "--dflash-draft-tokens", "7",
                "--dflash-verify-width", "12",
                "--weights", str(root / "eventual-dflash.ninfer"),
                "--output-dir", str(output),
                "--dry-run",
            ]
            for concurrency in range(1, 5):
                argv.extend(("--concurrency", str(concurrency)))
            self.assertEqual(main(argv), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], MATRIX_SCHEMA_VERSION)
            self.assertEqual(manifest["dflash_draft_tokens"], 7)
            self.assertEqual(manifest["dflash_verify_width"], 12)
            self.assertEqual(manifest["selected_prefill_chunk"], 2048)
            self.assertEqual(manifest["case_count"], 6)
            self.assertEqual(manifest["point_count"], 18)
            records = manifest["commands"]
            for record in records:
                command = record["command"]
                self.assertEqual(
                    command[command.index("--prefill-chunk") + 1], "2048"
                )
                self.assertTrue(all(isinstance(part, str) for part in command))
                self.assertEqual(command.count("-r"), 1)
                self.assertEqual(command.count("--warmup"), 1)
                repetitions = int(command[command.index("-r") + 1])
                warmup = int(command[command.index("--warmup") + 1])
                self.assertEqual(
                    (repetitions, warmup), expected_repetitions[record["case"]]
                )
            diagnostics = [record for record in records if not record["performance_eligible"]]
            self.assertEqual(len(diagnostics), 2)
            self.assertTrue(all(record["concurrency"] == 1 for record in diagnostics))
            self.assertEqual(
                diagnostics[0]["environment"]["NINFER_DFLASH_CANDIDATE_STATS"], "1"
            )
            dflash = [record for record in records if record["parity_role"] == "dflash"]
            ordinary = [record for record in records if record["parity_role"] == "ordinary"]
            self.assertEqual({record["concurrency"] for record in dflash}, set(range(1, 5)))
            self.assertEqual({record["concurrency"] for record in ordinary}, set(range(1, 5)))
            self.assertTrue(all("--retain-token-ids" in record["command"] for record in dflash + ordinary))
            self.assertTrue(
                all("--spec" in record["command"] and "dflash" in record["command"] for record in dflash)
            )

    def test_case_command_rejects_malformed_repetition_fields(self) -> None:
        malformed_repetitions = BenchCase(
            "dflash", "bad_repetitions", (), ("--spec", "dflash"), 1
        )
        with self.assertRaisesRegex(ValueError, "invalid repetitions"):
            add_repetition_args([], malformed_repetitions, None, None)

        malformed_warmup = BenchCase("dflash", "bad_warmup", (), 3, (1,))
        with self.assertRaisesRegex(ValueError, "invalid warmup"):
            add_repetition_args([], malformed_warmup, None, None)

    def test_dflash_pareto_requires_explicit_draft_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(SystemExit, "requires --dflash-draft-tokens"):
                main(
                    ["--preset", "dflash-pareto", "--weights", str(root / "model.ninfer"),
                     "--output-dir", str(root / "matrix"), "--dry-run"]
                )

    def test_dflash_campaign_requires_one_explicit_selected_prefill_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(SystemExit, "requires exactly one --prefill-chunk"):
                main([
                    "--preset", "dflash-shortlist",
                    "--weights", str(root / "model.ninfer"), "--dry-run",
                ])
            with self.assertRaisesRegex(SystemExit, "requires exactly one --prefill-chunk"):
                main([
                    "--preset", "dflash-shortlist",
                    "--weights", str(root / "model.ninfer"), "--dry-run",
                    "--prefill-chunk", "2048", "--prefill-chunk", "4096",
                ])

    def test_dflash_shortlist_is_complete_explicit_and_compact(self) -> None:
        cases = build_cases("dflash-shortlist")
        self.assertEqual(len(cases), 23)
        self.assertEqual(sum(case.parity_role == "ordinary" for case in cases), 1)
        candidates = [case for case in cases if case.parity_role == "dflash"]
        diagnostics = [case for case in cases if case.diagnostic]
        self.assertEqual(len(candidates), 11)
        self.assertEqual(len(diagnostics), 11)
        self.assertTrue(all((case.repetitions, case.warmup) == (2, 1)
                            for case in candidates))
        self.assertTrue(all((case.repetitions, case.warmup) == (1, 0)
                            for case in diagnostics))
        expected_profiles = dflash_shortlist_profiles()
        expected = {
            profile["draft_tokens_requested"]: profile for profile in expected_profiles
        }
        for case in candidates + diagnostics:
            args = list(case.args)
            k = int(args[args.index("--draft-tokens") + 1])
            width = int(args[args.index("--dflash-verify-width") + 1])
            self.assertEqual(width, expected[k]["verify_width_resolved"])
            self.assertIn("--lm-head-draft", args)
            self.assertEqual(args[args.index("-pg") + 1], "8192,256")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            argv = [
                "--preset", "dflash-shortlist",
                "--prefill-chunk", "2048",
                "--weights", str(root / "eventual-dflash.ninfer"),
                "--output-dir", str(output), "--dry-run",
            ]
            self.assertEqual(main(argv), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], MATRIX_SCHEMA_VERSION)
            self.assertEqual(manifest["case_count"], 23)
            self.assertEqual(manifest["point_count"], 23)
            self.assertEqual(manifest["selected_prefill_chunk"], 2048)
            self.assertEqual(manifest["dflash_shortlist_profiles"], expected_profiles)
            self.assertEqual(manifest["commands"][0]["concurrency"], 1)
            records = manifest["commands"]
            self.assertEqual(
                {record["dflash_draft_tokens"] for record in records if record["parity_role"] == "dflash"},
                set(range(1, 12)),
            )
            self.assertTrue(all(record["concurrency"] == 1 for record in records))
            self.assertTrue(all(
                record["command"][record["command"].index("--prefill-chunk") + 1]
                == "2048" for record in records
            ))
            self.assertTrue(all(record["performance_eligible"] is False
                                for record in records if record["bound_diagnostic"] is not None))
            for record in records:
                if record["dflash_draft_tokens"]:
                    self.assertEqual(
                        record["dflash_verify_width_requested"],
                        record["dflash_verify_width_resolved"],
                    )
                    self.assertIsNotNone(record["dflash_topology"])

            for extra in (
                ("--concurrency", "4"),
                ("--limit", "1"),
                ("--repetitions", "1"),
                ("--dflash-draft-tokens", "3"),
            ):
                with self.assertRaises(SystemExit):
                    main([*argv, *extra])

    def test_dflash_shortlist_frontier_preserves_tradeoffs_and_excludes_invalid(self) -> None:
        def candidate(k: int, speed: float, accepted: float, fallback: float,
                      repair: float, valid: bool = True) -> dict:
            return {
                "profile": {"draft_tokens_requested": k},
                "valid_for_ranking": valid,
                "target_equivalent_generated_tok_s": speed,
                "accepted_tokens_per_round": accepted,
                "fallback_rate_per_attempt": fallback,
                "first_reject_rate_per_proposal_hop": repair,
            }

        faster = candidate(2, 120.0, 1.5, 0.1, 0.2)
        accepting = candidate(3, 100.0, 2.0, 0.05, 0.1)
        dominated = candidate(4, 90.0, 1.0, 0.2, 0.3)
        invalid = candidate(5, 1000.0, 10.0, 0.0, 0.0, False)
        frontier = dflash_shortlist_frontier([faster, accepting, dominated, invalid])
        self.assertEqual(
            {entry["profile"]["draft_tokens_requested"] for entry in frontier}, {2, 3}
        )

    def test_dflash_shortlist_report_binds_profiles_metrics_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = {"path": "model.ninfer", "sha256": "a" * 64}
            bench = {"path": "ninfer_bench", "sha256": "b" * 64}
            rows = [{
                "suite": "dflash_shortlist_control", "concurrency": 1,
                "label": "pp8192+tg256", "report": "ordinary.json",
            }]
            records = []
            comparisons = []
            for profile in dflash_shortlist_profiles():
                k = profile["draft_tokens_requested"]
                report_path = root / f"k{k}.json"
                report_path.write_text("{}\n", encoding="utf-8")
                raw_path = root / f"k{k}.raw.json"
                raw = {
                    "artifact_type": "ninfer_dflash_proposal_selector_raw",
                    "schema_version": 2,
                    "diagnostic_only": True,
                    "timing_eligible": False,
                    "logits": {
                        "rows": 1, "drafts": k, "elements": k, "nan": 0, "inf": 0,
                        "finite_min": -1.0, "finite_max": 1.0,
                    },
                    "proposal": {
                        "hops": 10, "hits": 9, "in_tree": 9, "in_top16": 9,
                        "in_top64": 9, "in_top256": 10, "in_draft_head": 10,
                        "absent_from_draft_head": 0,
                    },
                    "reject": {
                        "count": 1, "in_tree": 0, "in_top16": 0, "in_top64": 0,
                        "in_top256": 1, "in_draft_head": 1,
                        "absent_from_draft_head": 0,
                    },
                    "by_depth": {
                        "hops": [10, *([0] * (k - 1))],
                        "hits": [9, *([0] * (k - 1))],
                        "top16": [9, *([0] * (k - 1))],
                        "top256": [10, *([0] * (k - 1))],
                        "rejects": [1, *([0] * (k - 1))],
                    },
                    "trace": [
                        {
                            "round_index": index,
                            "proposal_ids": list(range(profile["verify_width_resolved"])),
                            "parent_index": [-1, *([0] * (profile["verify_width_resolved"] - 1))],
                            "target_licensed_tokens": [0],
                        }
                        for index in range(10)
                    ],
                }
                raw_path.write_text(json.dumps(raw), encoding="utf-8")
                evidence_path = root / f"k{k}.evidence.json"
                evidence_path.write_text(json.dumps({
                    "artifact_type": "ninfer_dflash_proposal_selector_evidence",
                    "schema_version": 2,
                    "diagnostic_only": True,
                    "timing_eligible": False,
                    "artifact": artifact,
                    "benchmark_executable": bench,
                    "profile": {
                        "draft_tokens": k,
                        "dflash_verify_width_requested": profile["verify_width_requested"],
                        "dflash_verify_width": profile["verify_width_resolved"],
                        "proposal_head": "optimized",
                        "use_device_graph": False,
                    },
                    "benchmark_report": {
                        "path": str(report_path), "sha256": file_sha256(report_path)
                    },
                    "raw_diagnostic": {
                        "path": str(raw_path), "sha256": file_sha256(raw_path),
                        "report": raw,
                    },
                }), encoding="utf-8")
                records.append({
                    "suite": "dflash_shortlist_repair_diagnostic",
                    "concurrency": 1, "dflash_draft_tokens": k,
                    "report": str(report_path),
                    "bound_diagnostic": str(evidence_path),
                })
                rows.append({
                    "suite": "dflash_shortlist_decode", "concurrency": 1,
                    "label": "pp8192+tg256", "draft_tokens": k,
                    "dflash_verify_width_requested": profile["verify_width_requested"],
                    "dflash_verify_width": profile["verify_width_resolved"],
                    "dflash_topology": profile["topology"], "proposal_head": "optimized",
                    "decode_path": "dflash_device_graph", "report": str(report_path),
                    "n_prompt": 8192, "n_gen": 256, "repetitions": 2, "warmup": 1,
                    "spec_rounds": 100, "spec_drafted_tokens": 100 * k,
                    "spec_accepted_tokens": 50 + k, "spec_fallback_steps": 1,
                    "spec_acceptance_rate": (50 + k) / (100 * k),
                    "spec_acceptance_length": 1.5 + k / 100,
                    "decode_engine_tok_s_mean": 100.0 + k,
                    "decode_output_tok_s_mean": 90.0 + k,
                    "decode_seconds_mean": 2.0, "total_seconds_mean": 3.0,
                    "kv_capacity": 32768, "kv_payload_bytes": 1,
                    "weights_capacity_bytes": 2, "sequence_capacity_bytes": 3,
                    "workspace_capacity_bytes": 4, "request_transient_capacity_bytes": 5,
                    "device_graph_allowance_bytes": 6, "workspace_peak_bytes": 7,
                    "workspace_allocator_peak_bytes": 8,
                })
                comparisons.append({
                    "concurrency": 1, "draft_tokens": k,
                    "dflash_verify_width": profile["verify_width_resolved"],
                    "exact": True,
                })
            parity_path = root / "greedy-token-parity.json"
            parity = {"pass": True, "comparisons": comparisons}
            parity_path.write_text(json.dumps(parity), encoding="utf-8")
            payload, failures = write_dflash_shortlist(
                root, records, rows, parity, artifact=artifact, bench=bench,
            )
            self.assertFalse(failures)
            self.assertTrue(payload["pass"])
            self.assertEqual(len(payload["speed_ranking_exact_parity_only"]), 11)
            self.assertEqual(
                payload["speed_ranking_exact_parity_only"][0]["draft_tokens"], 11
            )
            self.assertTrue(payload["non_dominated_candidates"])
            self.assertEqual(payload["followup"]["concurrency"], list(range(1, 5)))
            diagnostic = payload["candidates"][0]["first_reject_repair_diagnostic"]
            self.assertFalse(diagnostic["timing_eligible"])
            self.assertNotIn("seconds", diagnostic)

    def test_dflash_diagnostic_validation_and_provenance_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.json"
            report_path = root / "bench.json"
            evidence_path = root / "evidence.json"
            raw = {
                "artifact_type": "ninfer_dflash_proposal_selector_raw",
                "schema_version": 2,
                "diagnostic_only": True,
                "timing_eligible": False,
                "logits": {
                    "rows": 16, "drafts": 7, "elements": 112, "nan": 0, "inf": 0,
                    "finite_min": -2.0, "finite_max": 3.0,
                },
                "proposal": {
                    "hops": 3, "hits": 2, "in_tree": 2, "in_top16": 2,
                    "in_top64": 2, "in_top256": 3, "in_draft_head": 3,
                    "absent_from_draft_head": 0,
                },
                "reject": {
                    "count": 1, "in_tree": 0, "in_top16": 0, "in_top64": 0,
                    "in_top256": 1, "in_draft_head": 1, "absent_from_draft_head": 0,
                },
                "by_depth": {
                    "hops": [2, 1, 0, 0, 0, 0, 0],
                    "hits": [2, 0, 0, 0, 0, 0, 0],
                    "top16": [2, 0, 0, 0, 0, 0, 0],
                    "top256": [2, 1, 0, 0, 0, 0, 0],
                    "rejects": [0, 1, 0, 0, 0, 0, 0],
                },
                "trace": [
                    {
                        "round_index": 0,
                        "proposal_ids": list(range(12)),
                        "parent_index": [-1, *([0] * 11)],
                        "target_licensed_tokens": [0, 1],
                    },
                    {
                        "round_index": 1,
                        "proposal_ids": list(range(12)),
                        "parent_index": [-1, *([0] * 11)],
                        "target_licensed_tokens": [0],
                    },
                ],
            }
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            report_path.write_text(
                json.dumps({"config": {
                    "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 16,
                    "q4_activation_bits": 8,
                    "q4_prefill_cta_profile": "m64n128-pingpong-production",
                    "w8_activation_bits": 8,
                    "fp8_qk_wmma_enabled": True,
                    "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                    "fp8_qk_wmma_t1_min_context": 64,
                    "fp8_qk_wmma_t2_min_context": 320,
                }}),
                encoding="utf-8",
            )
            self.assertEqual(validate_dflash_diagnostic_raw(raw_path, 7)["proposal"]["hops"], 3)
            case = BenchCase(
                "dflash_selector_diagnostic", "diagnostic",
                ("-pg", "8192,256", "--spec", "dflash", "--draft-tokens", "7",
                 "--lm-head-draft", "--dflash-verify-width", "12", "--no-device-graph"),
                1, 0, diagnostic=True, concurrency_one_only=True,
            )
            artifact = {"path": "model.ninfer", "sha256": "a" * 64}
            bench = {"path": "ninfer_bench", "sha256": "b" * 64}
            evidence = bind_dflash_diagnostic(
                raw_path, evidence_path, artifact=artifact, bench=bench,
                report_path=report_path, case=case, concurrency=1,
            )
            self.assertFalse(evidence["timing_eligible"])
            self.assertEqual(evidence["profile"]["dflash_verify_width"], 12)
            raw["proposal"]["hops"] = 0
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no proposal hops"):
                validate_dflash_diagnostic_raw(raw_path, 7)

    def test_dflash_greedy_parity_is_exact_and_reports_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary_path = root / "ordinary.json"
            dflash_path = root / "dflash.json"
            def report(tokens: list[int], *, dflash: bool = False) -> dict:
                return {
                    "config": {
                        "draft_tokens": 7 if dflash else 0,
                        "dflash_verify_width": 12 if dflash else 0,
                    },
                    "tests": [{
                        "label": "pp8192+tg256", "kind": "pp+tg", "n_prompt": 8192,
                        "n_gen": 256,
                        "reps": [{"generated_token_ids_by_lane": [tokens]}],
                    }]
                }
            ordinary_path.write_text(json.dumps(report([1, 2, 3])), encoding="utf-8")
            dflash_path.write_text(json.dumps(report([1, 2, 3], dflash=True)), encoding="utf-8")
            records = [
                {"case": "ordinary", "concurrency": 1, "parity_role": "ordinary",
                 "report": str(ordinary_path)},
                {"case": "dflash-k7", "concurrency": 1, "parity_role": "dflash",
                 "report": str(dflash_path)},
            ]
            result, failures = write_dflash_greedy_parity(
                root, records, artifact={"sha256": "a" * 64}, bench={"sha256": "b" * 64}
            )
            self.assertTrue(result["pass"])
            self.assertFalse(failures)
            dflash_path.write_text(json.dumps(report([1, 9, 3], dflash=True)), encoding="utf-8")
            result, failures = write_dflash_greedy_parity(
                root, records, artifact={"sha256": "a" * 64}, bench={"sha256": "b" * 64}
            )
            self.assertFalse(result["pass"])
            self.assertEqual(result["comparisons"][0]["first_mismatch"]["position"], 1)
            self.assertTrue(failures)

    def test_dflash_repeated_proposal_trace_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = {"path": "model.ninfer", "sha256": "a" * 64}
            bench = {"path": "ninfer_bench", "sha256": "b" * 64}
            raw = {
                "artifact_type": "ninfer_dflash_proposal_selector_raw",
                "schema_version": 2,
                "diagnostic_only": True,
                "timing_eligible": False,
                "logits": {"rows": 2, "drafts": 1, "elements": 2, "nan": 0, "inf": 0,
                           "finite_min": -1.0, "finite_max": 1.0},
                "proposal": {"hops": 1, "hits": 1, "in_tree": 1, "in_top16": 1,
                             "in_top64": 1, "in_top256": 1, "in_draft_head": 1,
                             "absent_from_draft_head": 0},
                "reject": {"count": 0, "in_tree": 0, "in_top16": 0, "in_top64": 0,
                           "in_top256": 0, "in_draft_head": 0,
                           "absent_from_draft_head": 0},
                "by_depth": {"hops": [1], "hits": [1], "top16": [1],
                             "top256": [1], "rejects": [0]},
                "trace": [{"round_index": 0, "proposal_ids": [7, 8],
                           "parent_index": [-1, 0], "target_licensed_tokens": [8]}],
            }
            records = []
            for repeat in ("a", "b"):
                raw_path = root / f"{repeat}.raw.json"
                evidence_path = root / f"{repeat}.evidence.json"
                report_path = root / f"{repeat}.report.json"
                raw_path.write_text(json.dumps(raw), encoding="utf-8")
                report_path.write_text(json.dumps({
                    "tests": [{"reps": [{"generated_token_ids_by_lane": [[8, 9]]}]}]
                }), encoding="utf-8")
                evidence_path.write_text(json.dumps({
                    "artifact_type": "ninfer_dflash_proposal_selector_evidence",
                    "schema_version": 2,
                    "artifact": artifact,
                    "benchmark_executable": bench,
                    "profile": {"draft_tokens": 1, "dflash_verify_width": 2},
                    "benchmark_report": {
                        "path": str(report_path), "sha256": file_sha256(report_path)
                    },
                    "raw_diagnostic": {
                        "path": str(raw_path), "sha256": file_sha256(raw_path), "report": raw
                    },
                }), encoding="utf-8")
                records.append({
                    "suite": "dflash_selector_diagnostic", "case": repeat,
                    "bound_diagnostic": str(evidence_path), "report": str(report_path),
                })
            result, failures = write_dflash_determinism(
                root, records, artifact=artifact, bench=bench
            )
            self.assertTrue(result["pass"])
            self.assertFalse(failures)
            parity = {
                "artifact": artifact,
                "benchmark_executable": bench,
                "comparisons": [
                    {
                        "concurrency": concurrency,
                        "draft_tokens": 1,
                        "dflash_verify_width": 2,
                        "exact": True,
                    }
                    for concurrency in range(1, 5)
                ],
                "pass": True,
            }
            (root / "greedy-token-parity.json").write_text(
                json.dumps(parity), encoding="utf-8"
            )
            quality, quality_failures = write_dflash_quality_evidence(
                root, records, parity, result, artifact=artifact, bench=bench
            )
            self.assertFalse(quality_failures)
            self.assertTrue(quality["pass"])
            self.assertEqual(quality["nll_diagnostic"]["status"], "not_applicable")
            self.assertEqual(
                quality["proposal_gate"]["diagnostics"][0]["finite_logit_elements"], 2
            )
            second_report = Path(records[1]["report"])
            original_report = second_report.read_text(encoding="utf-8")
            second_report.write_text(original_report + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance is invalid"):
                write_dflash_determinism(root, records, artifact=artifact, bench=bench)
            second_report.write_text(original_report, encoding="utf-8")
            changed = json.loads(Path(records[1]["bound_diagnostic"]).read_text(encoding="utf-8"))
            changed["raw_diagnostic"]["report"]["trace"][0]["proposal_ids"][1] = 9
            Path(records[1]["bound_diagnostic"]).write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "embedded trace differs"):
                write_dflash_determinism(root, records, artifact=artifact, bench=bench)

    def test_serve_concurrency_dry_run_accepts_future_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "eventual-selected.ninfer"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    serve_concurrency_main(
                        [
                            "--artifact",
                            str(artifact),
                            "--mode",
                            "mtp3",
                            "--suite",
                            "decode-saturation",
                            "--concurrency",
                            "4",
                            "--prefill-chunk",
                            "2048",
                            "--expected-kv-value-group",
                            "16",
                            "--expected-xattention-profile",
                            "dense",
                            "--output",
                            str(root / "output"),
                            "--dry-run",
                        ]
                    ),
                    0,
                )
            command = output.getvalue()
            self.assertIn(str(artifact), command)
            self.assertIn("--max-concurrency 4", command)

    def test_serve_concurrency_rejects_concurrency_above_four(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(CampaignError, r"\[1, 4\]"):
                serve_concurrency_main(
                    [
                        "--artifact",
                        str(root / "eventual-selected.ninfer"),
                        "--mode",
                        "mtp3",
                        "--suite",
                        "decode-saturation",
                        "--concurrency",
                        "5",
                        "--prefill-chunk",
                        "2048",
                        "--expected-kv-value-group",
                        "16",
                        "--expected-xattention-profile",
                        "dense",
                        "--output",
                        str(root / "output"),
                        "--dry-run",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
