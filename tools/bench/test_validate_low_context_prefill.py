#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.run_ninfer_bench_matrix import (
    FP8_QK_WMMA_PROFILE,
    FP8_QK_WMMA_T1_MIN_CONTEXT,
    FP8_QK_WMMA_T2_MIN_CONTEXT,
    LOW_CONTEXT_PREFILL_PROMPTS,
    MATRIX_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
    add_repetition_args,
    build_cases,
)
from tools.bench.validate_low_context_prefill import (
    main,
    resolve_selected_dense_route,
    validate_ladder,
)
from tools.ppl.pareto import classify


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


class LowContextPrefillValidationTest(unittest.TestCase):
    @staticmethod
    def migration_receipt(weights_id: str) -> dict:
        recipe = {
            "r9700-q4g64-n16k16-eval": "r9700-all-q4g64-n16k16-eval-v1",
            "r9700-q4-w8-mse-n16k16-eval":
                "r9700-source-q4-n16k16-promoted-w8-source-mse8-eval-v1",
            "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval-v1",
        }[weights_id]
        value = {"path": "/receipt", "sha256": "1" * 64, "recipe_id": recipe,
                 "object_plan_sha256": "2" * 64, "source_artifact_sha256": "3" * 64,
                 "source_receipt_sha256": "4" * 64, "transcoder_sha256": "5" * 64}
        if "four-role" in weights_id:
            value.update({"selection_sha256": "b2ceeb63c581c0f26aab5a4d8c0958da34d836fcc5c47d377bce709eaf37e3e8",
                          "source_index_sha256": "6" * 64,
                          "source_ranking_sha256": "7" * 64})
        else:
            value["receipt_producer_sha256"] = "8" * 64
        return value

    @staticmethod
    def make_selection(
        root: Path,
        *,
        chunk: int = 2048,
        winner_recipe: str = "r9700-q4g64-n16k16-eval",
        winner_group: int = 16,
    ) -> Path:
        candidates = []
        provenance = []
        recipes = (
            ("r9700-q4g64-n16k16-eval", "a" * 64, "all-q4"),
            ("r9700-q4-w8-mse-n16k16-eval", "d" * 64, "mixed"),
            (
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
                "e" * 64,
                "four-role",
            ),
        )
        for recipe, digest, prefix in recipes:
            for group, profile in (
                (16, "dense"), (32, "dense"),
                (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
            ):
                name = f"{prefix}-g{group}-{profile}"
                if recipe == winner_recipe and group == winner_group and profile == "dense":
                    speed = 200.0
                elif group == 16 and profile == "dense":
                    speed = 110.0
                else:
                    speed = 90.0
                candidates.append({
                    "name": name,
                    "prefill_chunk": chunk,
                    "shortlist_head_precision_gate": shortlist_head_gate(),
                    "cache_profile": {
                        "value_group": group,
                        "plane_layouts": R9700_KV_PLANE_LAYOUTS,
                    },
                    "execution_profile": {
                        "q4_activation_bits": 8,
                        "w8_activation_bits": 8,
                        "fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
                        "xattention_profile": profile,
                    },
                    "quality": {
                        "eligible": True,
                        "tier": "accuracy",
                        "mean_nll_delta": 0.001,
                        "complete_finite_aligned": True,
                        "scored_positions": 10000,
                        "new_severe_positions": 0,
                    },
                    "whole_inference_tokens_per_second": {"whole_8k_c1": speed},
                    "capacity": {
                        "measurement_kind": "resolved_effective_maximum",
                        "binding_constraint": "device_memory",
                        "tokens": 1000,
                    },
                })
                provenance.append({
                    "candidate": name,
                    "artifact": {"weights_id": recipe, "sha256": digest,
                                 "conversion_receipt":
                                     LowContextPrefillValidationTest.migration_receipt(recipe)},
                })
        source = {
            "artifact_type": "ninfer_r9700_pareto_input",
            "schema_version": 4,
            "required_speed_workloads": ["whole_8k_c1"],
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": chunk,
            "prefill_chunk_selection": {
                "path": "/prefill-selection.json",
                "sha256": "9" * 64,
                "selection_rule": (
                    "global_maximin_normalized_prefill_then_workspace_then_smaller_chunk_v2"
                ),
                "selected_prefill_chunk": chunk,
            },
            "candidates": candidates,
            "source_provenance": provenance,
        }
        input_path = root / "pareto-input.json"
        input_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
        authority = classify(source)
        authority["pareto_input"] = {
            "path": str(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        }
        path = root / "selection.json"
        path.write_text(json.dumps(authority), encoding="utf-8")
        return path

    def make_campaign(self, root: Path, p2048_speed: float = 2100.0) -> Path:
        artifact = {
            "path": str(root / "selected.ninfer"), "model_id": "qwen3.8-27b",
            "weights_id": "r9700-q4g64-n16k16-eval", "sha256": "a" * 64,
            "file_size_bytes": 10,
            "conversion_receipt": self.migration_receipt(
                "r9700-q4g64-n16k16-eval"
            ),
        }
        bench = {"path": str(root / "bench"), "sha256": "b" * 64,
                 "file_size_bytes": 20}
        (root / "selected.ninfer").write_bytes(b"artifact")
        (root / "bench").write_bytes(b"bench")
        (root / "corpus.ids").write_text("1 2 3\n", encoding="utf-8")
        commands = []
        for case in build_cases("low-context-prefill", production_prefill_chunk=2048):
            report = root / "json" / case.suite / "c1" / f"{case.name}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("{}", encoding="utf-8")
            command = add_repetition_args([
                str(root / "bench"), "--weights", str(root / "selected.ninfer"),
                "--corpus", str(root / "corpus.ids"), "--device", "0",
                "--concurrency", "1", *case.args, "--output", "json",
                "--output-file", str(report),
            ], case, None, None)
            commands.append({
                "suite": case.suite, "case": case.name, "concurrency": 1,
                "report": str(report), "command": command,
            })
        manifest = {
            "artifact_type": "ninfer_bench_matrix_run",
            "schema_version": MATRIX_SCHEMA_VERSION,
            "preset": "low-context-prefill", "dry_run": False,
            "case_count": 5, "point_count": 5,
            "concurrency": [1], "selected_prefill_chunk": 2048,
            "expected_xattention_profile": "dense", "expected_kv_value_group": 16,
            "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_fp8_qk_wmma_enabled": True,
            "expected_fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
            "expected_fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
            "expected_fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
            "corpus": str(root / "corpus.ids"), "corpus_tokens": 3,
            "corpus_sha256": "c" * 64,
            "power_profile": {
                "required": "auto",
                "sysfs_path": (
                    "/sys/class/drm/card2/device/power_dpm_force_performance_level"
                ),
                "observed": "auto", "rechecked_after": "auto",
            },
            "artifact": artifact, "bench": bench, "commands": commands,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.make_resolvable_selection(root)
        return manifest_path

    def make_resolvable_selection(self, root: Path) -> tuple[Path, dict, dict]:
        selection = self.make_selection(root)
        input_path = root / "pareto-input.json"
        source = json.loads(input_path.read_text(encoding="utf-8"))
        artifact = {
            "path": str(root / "selected.ninfer"), "model_id": "qwen3.8-27b",
            "weights_id": "r9700-q4g64-n16k16-eval", "sha256": "a" * 64,
            "file_size_bytes": 10,
            "conversion_receipt": self.migration_receipt(
                "r9700-q4g64-n16k16-eval"
            ),
        }
        bench = {
            "path": str(root / "bench"), "sha256": "b" * 64,
            "file_size_bytes": 20,
        }
        (root / "selected.ninfer").write_bytes(b"artifact")
        (root / "bench").write_bytes(b"bench")
        selected_name = "all-q4-g16-dense"
        provenance = next(
            row for row in source["source_provenance"] if row["candidate"] == selected_name
        )
        provenance["artifact"] = artifact
        provenance["benchmark_executable"] = bench
        matrices = {}
        for preset in ("pareto-capacity", "pareto-whole"):
            path = root / preset / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "artifact_type": "ninfer_bench_matrix_run",
                "schema_version": MATRIX_SCHEMA_VERSION,
                "preset": preset,
                "concurrency": [1, 2, 3, 4],
                "selected_prefill_chunk": 2048,
                "expected_kv_value_group": 16,
                "expected_xattention_profile": "dense",
                "artifact": artifact,
                "bench": bench,
            }), encoding="utf-8")
            matrices[preset] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        provenance["matrices"] = matrices
        input_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
        authority = classify(source)
        authority["pareto_input"] = {
            "path": str(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        }
        selection.write_text(json.dumps(authority), encoding="utf-8")
        return selection, artifact, bench

    def test_resolves_exact_dense_control_from_terminal_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, artifact, bench = self.make_resolvable_selection(root)
            with (
                patch("tools.bench.validate_low_context_prefill.inspect_artifact",
                      return_value=artifact),
                patch("tools.bench.validate_low_context_prefill.bind_n16_migration_receipt",
                      return_value=artifact),
                patch("tools.bench.validate_low_context_prefill.inspect_executable",
                      return_value=bench),
            ):
                route = resolve_selected_dense_route(selection)
            self.assertEqual(route["weights_id"], "r9700-q4g64-n16k16-eval")
            self.assertEqual(route["value_group"], 16)
            self.assertEqual(route["selected_prefill_chunk"], 2048)
            self.assertEqual(route["dense_control"], "all-q4-g16-dense")
            self.assertIsNone(route["hybrid_width_tool"])
            self.assertIsNone(route["hybrid_width_tool_identity"])

    def test_dense_route_rejects_changed_bound_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, artifact, bench = self.make_resolvable_selection(root)
            (root / "pareto-capacity/manifest.json").write_text("{}\n", encoding="utf-8")
            with (
                patch("tools.bench.validate_low_context_prefill.inspect_artifact",
                      return_value=artifact),
                patch("tools.bench.validate_low_context_prefill.bind_n16_migration_receipt",
                      return_value=artifact),
                patch("tools.bench.validate_low_context_prefill.inspect_executable",
                      return_value=bench),
                self.assertRaisesRegex(ValueError, "manifest bytes changed"),
            ):
                resolve_selected_dense_route(selection)

    @staticmethod
    def fake_report(_path: Path, *_args, **kwargs):
        case = kwargs.get("expected_case") or _args[7]
        prompt = int(case.args[case.args.index("-p") + 1])
        speed = 2100.0 if prompt == 2048 else 1000.0 + prompt
        seconds = [prompt / speed] * 3
        return {"tests": [{
            "label": f"pp{prompt}",
            "prefill_seconds_mean": seconds[0], "prefill_seconds_stddev": 0.0,
            "prefill_tok_s_mean": speed, "prefill_tok_s_stddev": 0.0,
            "reps": [{"timings": {"prefill_seconds": value}} for value in seconds],
        }]}

    def validate(
        self, manifest: Path, threshold: float = 2000.0, report_loader=None,
        route_mutation=None,
    ):
        root = manifest.parent
        document = json.loads(manifest.read_text(encoding="utf-8"))
        expected_bench = {
            "path": str(root / "bench"), "sha256": "b" * 64,
            "file_size_bytes": 20,
        }
        expected_artifact = {
            "path": str(root / "selected.ninfer"), "model_id": "qwen3.8-27b",
            "weights_id": "r9700-q4g64-n16k16-eval", "sha256": "a" * 64,
            "file_size_bytes": 10,
            "conversion_receipt": self.migration_receipt(
                "r9700-q4g64-n16k16-eval"
            ),
        }
        selection_path = root / "selection.json"
        selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        selected_route = {
            "terminal_selection": {"sha256": selection_sha},
            "winner": "all-q4-g16-dense",
            "dense_control": "all-q4-g16-dense",
            "weights_id": "r9700-q4g64-n16k16-eval",
            "value_group": 16,
            "selected_prefill_chunk": 2048,
            "artifact": expected_artifact,
            "executable": expected_bench,
            "hybrid_width_tool": None,
            "hybrid_width_tool_identity": None,
        }
        if route_mutation is not None:
            route_mutation(selected_route)
        with (
            patch("tools.bench.validate_low_context_prefill.inspect_executable",
                  return_value=expected_bench),
            patch("tools.bench.validate_low_context_prefill.inspect_artifact",
                  return_value=expected_artifact),
            patch("tools.bench.validate_low_context_prefill.bind_n16_migration_receipt",
                  return_value=expected_artifact),
            patch("tools.bench.validate_low_context_prefill.file_sha256",
                  side_effect=lambda path: (
                      document["corpus_sha256"] if Path(path) == root / "corpus.ids"
                      else __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
                  )),
            patch("tools.bench.validate_low_context_prefill.load_bench_report",
                  side_effect=report_loader or self.fake_report),
            patch("tools.bench.validate_low_context_prefill.resolve_selected_dense_route",
                  return_value=selected_route),
        ):
            return validate_ladder(
                manifest, threshold, root / "bench", root / "selected.ninfer",
                root / "selection.json",
                power_reader=lambda _path: "auto",
            )

    def test_validates_exact_ladder_and_applies_explicit_downstream_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_campaign(root)
            result = self.validate(manifest)
            self.assertTrue(result["passes_p2048_gate"])
            self.assertEqual(result["observed_p2048_tok_s"], 2100.0)
            self.assertEqual(
                [row["prompt_tokens"] for row in result["ladder"]],
                list(LOW_CONTEXT_PREFILL_PROMPTS),
            )

            output = root / "failed-gate.json"
            document = json.loads(manifest.read_text(encoding="utf-8"))
            with (
                patch("tools.bench.validate_low_context_prefill.inspect_executable",
                      return_value=document["bench"]),
                patch("tools.bench.validate_low_context_prefill.inspect_artifact",
                      return_value=document["artifact"]),
                patch("tools.bench.validate_low_context_prefill.bind_n16_migration_receipt",
                      return_value=document["artifact"]),
                patch("tools.bench.validate_low_context_prefill.file_sha256",
                      side_effect=lambda path: (
                          document["corpus_sha256"] if Path(path) == root / "corpus.ids"
                          else __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
                      )),
                patch("tools.bench.validate_low_context_prefill.load_bench_report",
                      side_effect=self.fake_report),
                patch("tools.bench.validate_low_context_prefill._read_power",
                      return_value="auto"),
            ):
                self.assertEqual(main([
                    "--manifest", str(manifest),
                    "--executable", str(root / "bench"),
                    "--artifact", str(root / "selected.ninfer"),
                    "--selection", str(root / "selection.json"),
                    "--min-p2048-tok-s", "2200", "--out", str(output),
                ]), 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))[
                "passes_p2048_gate"
            ])

    def test_rejects_non_auto_post_state_and_failure_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_campaign(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["power_profile"]["rechecked_after"] = "profile_standard"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete physical"):
                self.validate(manifest_path)
            manifest["power_profile"]["rechecked_after"] = "auto"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "failures.json").symlink_to(root / "missing-failures")
            with self.assertRaisesRegex(ValueError, "complete physical"):
                self.validate(manifest_path)
            (root / "failures.json").unlink()
            (root / "failures.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete physical"):
                self.validate(manifest_path)

    def test_rejects_forged_command_types_identity_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_campaign(root)
            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutations = (
                ("boolean schema", lambda doc: doc.__setitem__("schema_version", 13.0)),
                ("boolean group", lambda doc: doc.__setitem__("expected_kv_value_group", True)),
                ("command", lambda doc: doc["commands"][0]["command"].append("--forged")),
                ("bench hash", lambda doc: doc["bench"].__setitem__("sha256", "d" * 64)),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    document = json.loads(json.dumps(original))
                    mutate(document)
                    manifest_path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.validate(manifest_path)

            manifest_path.write_text(json.dumps(original), encoding="utf-8")
            forged = self.fake_report(
                None, *([None] * 7),
                build_cases("low-context-prefill", production_prefill_chunk=2048)[0],
            )
            forged["tests"][0]["prefill_tok_s_mean"] += 1.0
            with (
                patch("tools.bench.validate_low_context_prefill.inspect_executable",
                      return_value=original["bench"]),
                patch("tools.bench.validate_low_context_prefill.inspect_artifact",
                      return_value=original["artifact"]),
                patch("tools.bench.validate_low_context_prefill.bind_n16_migration_receipt",
                      return_value=original["artifact"]),
                patch("tools.bench.validate_low_context_prefill.file_sha256",
                      side_effect=lambda path: (
                          original["corpus_sha256"] if Path(path) == root / "corpus.ids"
                          else __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
                      )),
                patch("tools.bench.validate_low_context_prefill.load_bench_report",
                      return_value=forged),
            ):
                with self.assertRaisesRegex(ValueError, "aggregate differs"):
                    validate_ladder(
                        manifest_path, 2000.0, root / "bench", root / "selected.ninfer",
                        root / "selection.json",
                        power_reader=lambda _path: "auto",
                    )

    def test_rejects_terminal_selection_tuple_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_campaign(root)
            for label, kwargs, message in (
                ("artifact", {"winner_recipe": "r9700-q4-w8-mse-n16k16-eval"}, "artifact bytes"),
                ("group", {"winner_group": 32}, "value group"),
                ("chunk", {"chunk": 4096}, "prefill chunk"),
            ):
                with self.subTest(label=label):
                    self.make_selection(root, **kwargs)
                    with self.assertRaisesRegex(ValueError, message):
                        self.validate(manifest_path)

    def test_rejects_ladder_executable_not_owned_by_resolved_dense_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_campaign(root)
            with self.assertRaisesRegex(ValueError, "terminal winner's dense control"):
                self.validate(
                    manifest_path,
                    route_mutation=lambda route: route.__setitem__(
                        "executable", {**route["executable"], "sha256": "f" * 64}
                    ),
                )

    def test_rejects_non_integer_terminal_schema_and_late_report_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_campaign(root)
            selection_path = root / "selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["schema_version"] = 7.0
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema version is not an integer"):
                self.validate(manifest_path)

            self.make_selection(root)
            first_report = (
                root / "json/low_context_prefill/c1/prefill_p128_dense_none.json"
            )
            calls = 0

            def mutate_earlier_report(*args, **kwargs):
                nonlocal calls
                calls += 1
                result = self.fake_report(*args, **kwargs)
                if calls == 5:
                    first_report.write_text('{"changed":true}', encoding="utf-8")
                return result

            with self.assertRaisesRegex(ValueError, "benchmark report changed while validating"):
                self.validate(manifest_path, report_loader=mutate_earlier_report)

    def test_rejects_late_report_symlink_retarget_with_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_campaign(root)
            first_report = (
                root / "json/low_context_prefill/c1/prefill_p128_dense_none.json"
            )
            original_target = root / "json/original-report.json"
            replacement_target = root / "json/replacement-report.json"
            first_report.replace(original_target)
            replacement_target.write_bytes(original_target.read_bytes())
            first_report.symlink_to(original_target)
            calls = 0

            def retarget_earlier_report(*args, **kwargs):
                nonlocal calls
                calls += 1
                result = self.fake_report(*args, **kwargs)
                if calls == 5:
                    first_report.unlink()
                    first_report.symlink_to(replacement_target)
                return result

            with self.assertRaisesRegex(ValueError, "regular non-symlink|benchmark report changed"):
                self.validate(manifest_path, report_loader=retarget_earlier_report)

    def test_rejects_report_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            root = outer / "campaign"
            root.mkdir()
            manifest_path = self.make_campaign(root)
            report = root / "json/low_context_prefill/c1/prefill_p128_dense_none.json"
            outside = outer / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            report.unlink()
            report.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                self.validate(manifest_path)


if __name__ == "__main__":
    unittest.main()
