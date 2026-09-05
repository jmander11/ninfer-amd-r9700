from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bench import run_niah_check as niah
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


class NiahEvidenceTest(unittest.TestCase):
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

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifact = self.root / "model.ninfer"
        self.artifact.write_bytes(b"artifact")
        self.serve = self.root / "ninfer-serve"
        self.serve.write_bytes(b"executable")
        self.selection = self.root / "selection.json"
        cache = {"value_group": 16, "plane_layouts": {
            "key": "token-fastest-head-major", "value": "feature-fastest-page-major",
            "value_scale": "feature-fastest-page-major",
        }}
        execution = {"q4_activation_bits": 8, "w8_activation_bits": 8,
                     "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                     "xattention_profile": "dense"}
        recipe = {
            "kind": "artifact", "weights_id": "r9700-q4g64-n16k16-eval",
            "sha256": niah.file_identity(self.artifact)["sha256"],
        }
        candidates = []
        provenance = []
        for recipe_name, digest, prefix, best_speed in (
            ("r9700-q4g64-n16k16-eval", recipe["sha256"], "", 120.0),
            ("r9700-q4-w8-mse-n16k16-eval", "b" * 64, "mixed-", 110.0),
            ("r9700-q4g64-f8e4m3-four-role-n16k16-eval", "c" * 64, "hybrid-", 105.0),
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
                    "cache_profile": {**cache, "value_group": group},
                    "execution_profile": {**execution, "xattention_profile": profile},
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
                    "candidate": name,
                    "artifact": {"weights_id": recipe_name, "sha256": digest,
                                 "conversion_receipt": self.migration_receipt(recipe_name)},
                })
        source = {
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": ["whole_8k_c1"],
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": 4096,
            "prefill_chunk_selection": CHUNK_SELECTION,
            "candidates": candidates, "source_provenance": provenance,
        }
        self.input = self.root / "pareto-input.json"
        self.input.write_text(json.dumps(source) + "\n", encoding="utf-8")
        authority = classify(source)
        authority["pareto_input"] = {
            "path": str(self.input), "sha256": _file_sha256(self.input),
        }
        self.candidate_inputs = candidates
        self.candidate_provenance = provenance
        self.selection.write_text(json.dumps(authority) + "\n", encoding="utf-8")
        self.log = self.root / "server.jsonl"
        self.start = {
            "artifact_type": niah.SERVER_LOG_ARTIFACT_TYPE,
            "schema_version": niah.SERVER_LOG_SCHEMA_VERSION,
            "event": "server_start",
            "server_instance_id": "instance-1",
            "server": {"public_model_id": "coding"},
            "artifact": {
                "path": str(self.artifact),
                "size_bytes": self.artifact.stat().st_size,
                "target": "qwen3_8_27b_r9700",
                "weights_id": "r9700-q4g64-n16k16-eval",
            },
            "engine": {
                **niah.NIAH_ENGINE_PROFILE,
                "prefill_chunk": 4096,
                "kv_value_group": 16,
                "xattention_qualification": False,
            },
            "argv": [str(self.serve), str(self.artifact)],
        }
        self.log.write_text(json.dumps(self.start) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def select_authority_winner(self, name: str) -> dict:
        candidates = json.loads(json.dumps(self.candidate_inputs))
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
            "candidates": candidates, "source_provenance": self.candidate_provenance,
        }
        input_path = self.root / f"pareto-input-{name}.json"
        input_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
        result = classify(source)
        result["pareto_input"] = {
            "path": str(input_path), "sha256": _file_sha256(input_path),
        }
        return result

    def append_done(self, *, computed: int = 1024, cache_hit: int = 0,
                    restored: int = 0, reuse_source: str = "none") -> None:
        event = {
            "artifact_type": niah.SERVER_LOG_ARTIFACT_TYPE,
            "schema_version": niah.SERVER_LOG_SCHEMA_VERSION,
            "event": "request_done",
            "server_instance_id": "instance-1",
            "request": {
                "request_id": 7,
                "model": "coding",
                "message_count": 2,
                "requested_output_tokens": 64,
                "enable_thinking": False,
            },
            "result": {
                "prompt_tokens": 1024,
                "completion_tokens": 3,
                "computed_prefill_tokens": computed,
                "prefix_cache_hit_tokens": cache_hit,
                "reuse_source": reuse_source,
                "context_checkpoint": {"restored_tokens": restored, "captured_tokens": 0},
            },
        }
        with self.log.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event) + "\n")

    @property
    def expected(self) -> list[dict[str, object]]:
        return [{
            "model": "coding", "message_count": 2, "requested_output_tokens": 64,
            "enable_thinking": False, "prompt_tokens": 1024, "completion_tokens": 3,
        }]

    def test_binds_files_server_start_and_fresh_full_prefill(self) -> None:
        binding, offset = niah.prepare_server_log_binding(
            self.log, self.artifact, self.serve, self.selection)
        self.append_done()
        result = niah.validate_fresh_prefill_log(
            self.log, offset, binding, self.expected)
        self.assertTrue(result["pass"])
        self.assertEqual(result["requests"][0]["computed_prefill_tokens"], 1024)
        self.assertEqual(binding["artifact"]["sha256"], niah.file_identity(self.artifact)["sha256"])

    def test_64k_admission_ladder_expands_to_exact_five_positions(self) -> None:
        cases = niah.matrix_cases(["64k"], list(niah.NIAH_POSITIONS))
        self.assertEqual(cases, [
            ("context_64k_start", "examples/cli/messages/long_niah_64k_start.json"),
            ("context_64k_q25", "examples/cli/messages/long_niah_64k_q25.json"),
            ("context_64k", "examples/cli/messages/long_niah_64k.json"),
            ("context_64k_q75", "examples/cli/messages/long_niah_64k_q75.json"),
            ("context_64k_end", "examples/cli/messages/long_niah_64k_end.json"),
        ])
        self.assertTrue(all(niah.resolve_fixture(ref).is_file() for _, ref in cases))

    def test_rejects_prefix_reuse_or_checkpoint_restore(self) -> None:
        for values in (
            {"computed": 900, "cache_hit": 124, "reuse_source": "vram_resident"},
            {"computed": 900, "restored": 124},
        ):
            self.log.write_text(json.dumps(self.start) + "\n", encoding="utf-8")
            binding, offset = niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection)
            self.append_done(**values)
            with self.assertRaisesRegex(ValueError, "fresh full prefill"):
                niah.validate_fresh_prefill_log(self.log, offset, binding, self.expected)

    def test_rejects_wrong_server_engine_profile(self) -> None:
        for field, value in (
            ("max_context", 65536),
            ("kv_capacity_mode", "auto"),
            ("kv_capacity", 131072),
            ("max_concurrency", 2),
            ("prefill_chunk", 2048),
            ("prefix_reuse", True),
        ):
            start = json.loads(json.dumps(self.start))
            start["engine"][field] = value
            self.log.write_text(json.dumps(start) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, field):
                niah.prepare_server_log_binding(
                    self.log, self.artifact, self.serve, self.selection)

    def test_accepts_selection_bound_nondefault_prefill_chunk(self) -> None:
        candidates = json.loads(json.dumps(self.candidate_inputs))
        for row in candidates:
            row["prefill_chunk"] = 2048
        source = {
            "artifact_type": "ninfer_r9700_pareto_input", "schema_version": 4,
            "required_speed_workloads": ["whole_8k_c1"],
            "require_single_static_profile_selection": True,
            "selected_prefill_chunk": 2048,
            "prefill_chunk_selection": {
                **CHUNK_SELECTION, "selected_prefill_chunk": 2048,
            },
            "candidates": candidates, "source_provenance": self.candidate_provenance,
        }
        self.input.write_text(json.dumps(source) + "\n", encoding="utf-8")
        authority = classify(source)
        authority["pareto_input"] = {
            "path": str(self.input), "sha256": _file_sha256(self.input),
        }
        self.selection.write_text(json.dumps(authority) + "\n", encoding="utf-8")
        self.start["engine"]["prefill_chunk"] = 2048
        self.log.write_text(json.dumps(self.start) + "\n", encoding="utf-8")
        binding, _ = niah.prepare_server_log_binding(
            self.log, self.artifact, self.serve, self.selection
        )
        self.assertEqual(binding["static_profile_selection"]["prefill_chunk"], 2048)

    def test_rejects_server_group_or_artifact_outside_static_selection(self) -> None:
        start = json.loads(json.dumps(self.start))
        start["engine"]["kv_value_group"] = 32
        self.log.write_text(json.dumps(start) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selected G16"):
            niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection)

        for field, value, error in (
            ("weights_id", "other", "does not select the supplied artifact"),
            ("target", "qwen3_8_27b", "canonical artifact identity"),
        ):
            start = json.loads(json.dumps(self.start))
            start["artifact"][field] = value
            self.log.write_text(json.dumps(start) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, error):
                niah.prepare_server_log_binding(
                    self.log, self.artifact, self.serve, self.selection)

    def test_rejects_malformed_or_ambiguous_static_selection(self) -> None:
        artifact = {**niah.file_identity(self.artifact),
                    "weights_id": "r9700-q4g64-n16k16-eval"}
        original = json.loads(self.selection.read_text())
        malformed = (
            [],
            {**original, "same_recipe_static_profile_selection": []},
            {**original, "single_static_profile_selection_required": False},
            {**original, "candidates": {}},
            {**original, "candidates": original["candidates"][:-1]},
            {**original, "frontier": []},
        )
        for value in malformed:
            self.selection.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                niah.load_static_profile_selection(self.selection, artifact)

        substituted = json.loads(json.dumps(original))
        target = next(
            row for row in substituted["candidates"]
            if row["name"] == "g32-b128-s16-tau900"
        )
        all_q4 = next(
            choice for choice in
            substituted["same_recipe_static_profile_selection"]["selections"]
            if choice["weight_recipe"]["weights_id"] == "r9700-q4g64-n16k16-eval"
        )
        all_q4.update({
            "winner": target["name"],
            "winner_cache_profile": target["cache_profile"],
            "winner_execution_profile": target["execution_profile"],
        })
        terminal = substituted["terminal_production_selection"]
        terminal.update({
            "eligible_profile_winners": sorted(
                choice["winner"] for choice in
                substituted["same_recipe_static_profile_selection"]["selections"]
            ),
            "winner": target["name"],
            "winner_cache_profile": target["cache_profile"],
            "winner_execution_profile": target["execution_profile"],
        })
        self.selection.write_text(json.dumps(substituted) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not recompute exactly"):
            niah.load_static_profile_selection(self.selection, artifact)

        unsupported = json.loads(json.dumps(original))
        unsupported["same_recipe_static_profile_selection"]["selections"][0][
            "winner_execution_profile"]["q4_activation_bits"] = 4
        unsupported["terminal_production_selection"][
            "winner_execution_profile"
        ]["q4_activation_bits"] = 4
        unsupported["candidates"][0]["execution_profile"]["q4_activation_bits"] = 4
        self.selection.write_text(json.dumps(unsupported) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not recompute exactly"):
            niah.load_static_profile_selection(self.selection, artifact)

        selection = original
        selection["terminal_production_selection"]["winner_artifact"][
            "sha256"
        ] = "0" * 64
        self.selection.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        self.log.write_text(json.dumps(self.start) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not recompute exactly"):
            niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection)

    def test_rejects_mismatched_artifact_or_executable(self) -> None:
        other = self.root / "other"
        other.write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "artifact"):
            niah.prepare_server_log_binding(self.log, other, self.serve, self.selection)
        with self.assertRaisesRegex(ValueError, "executable"):
            niah.prepare_server_log_binding(self.log, self.artifact, other, self.selection)

    def test_requires_compile_bound_xattention_profile(self) -> None:
        sparse = dict(self.start)
        sparse["engine"] = {
            **self.start["engine"],
            "xattention_qualification": True,
            "xattention_profile": "b128-s16-tau900",
            "xattention_find_block": 128,
            "xattention_stride": 16,
            "xattention_tau_permille": 900,
        }
        sparse_selection = self.select_authority_winner("g16-b128-s16-tau900")
        self.selection.write_text(json.dumps(sparse_selection) + "\n", encoding="utf-8")
        self.log.write_text(json.dumps(sparse) + "\n", encoding="utf-8")
        binding, _ = niah.prepare_server_log_binding(
            self.log, self.artifact, self.serve, self.selection
        )
        self.assertEqual(
            binding["server_start"]["xattention_profile"], "b128-s16-tau900"
        )
        with self.assertRaisesRegex(ValueError, "xattention_qualification=True"):
            dense_selection = self.select_authority_winner("g16-dense")
            self.selection.write_text(json.dumps(dense_selection) + "\n", encoding="utf-8")
            niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection)

        self.log.write_text(json.dumps(self.start) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "xattention_qualification=False"):
            self.selection.write_text(json.dumps(sparse_selection) + "\n", encoding="utf-8")
            niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection
            )

        stale_dense = json.loads(json.dumps(self.start))
        stale_dense["engine"]["xattention_profile"] = "b128-s16-tau900"
        self.log.write_text(json.dumps(stale_dense) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dense server_start carries"):
            self.selection.write_text(json.dumps(dense_selection) + "\n", encoding="utf-8")
            niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection)

        malformed = json.loads(json.dumps(self.start))
        malformed["engine"] = []
        self.log.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "engine provenance must be an object"):
            niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection)

    def test_accepts_each_selected_group_and_attention_server_tuple(self) -> None:
        for group, profile in (
            (16, "dense"), (32, "dense"),
            (16, "b128-s16-tau900"), (32, "b128-s16-tau900"),
        ):
            selection = self.select_authority_winner(f"g{group}-{profile}")
            self.selection.write_text(json.dumps(selection) + "\n", encoding="utf-8")

            start = json.loads(json.dumps(self.start))
            start["engine"]["kv_value_group"] = group
            if profile == "b128-s16-tau900":
                start["engine"].update({
                    "xattention_qualification": True,
                    "xattention_profile": profile,
                    "xattention_find_block": 128,
                    "xattention_stride": 16,
                    "xattention_tau_permille": 900,
                })
            self.log.write_text(json.dumps(start) + "\n", encoding="utf-8")
            binding, _ = niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection)
            self.assertEqual(
                (binding["static_profile_selection"]["kv_value_group"],
                 binding["static_profile_selection"]["xattention_profile"]),
                (group, profile),
            )

    def test_rejects_artifact_changed_during_run(self) -> None:
        binding, offset = niah.prepare_server_log_binding(
            self.log, self.artifact, self.serve, self.selection)
        self.append_done()
        self.artifact.write_bytes(b"changed!")
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            niah.validate_fresh_prefill_log(self.log, offset, binding, self.expected)

    def test_rejects_selection_changed_during_run(self) -> None:
        binding, offset = niah.prepare_server_log_binding(
            self.log, self.artifact, self.serve, self.selection)
        self.append_done()
        self.selection.write_text(self.selection.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selection bytes changed"):
            niah.validate_fresh_prefill_log(self.log, offset, binding, self.expected)

    def test_rejects_server_start_changed_during_run(self) -> None:
        binding, offset = niah.prepare_server_log_binding(
            self.log, self.artifact, self.serve, self.selection)
        initial = self.log.read_text(encoding="utf-8")
        self.log.write_text(initial.replace("instance-1", "instance-2"), encoding="utf-8")
        self.append_done()
        with self.assertRaisesRegex(ValueError, "log prefix changed"):
            niah.validate_fresh_prefill_log(self.log, offset, binding, self.expected)

    def test_rejects_stale_events_before_server_start(self) -> None:
        stale = dict(self.start)
        stale["server_instance_id"] = "stale-instance"
        self.log.write_text(
            json.dumps(stale) + "\n" + json.dumps(self.start) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "exactly one server_start"):
            niah.prepare_server_log_binding(
                self.log, self.artifact, self.serve, self.selection
            )

    def test_explicit_output_path_is_used(self) -> None:
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps([{"role": "user", "content": "needle"}]),
                           encoding="utf-8")
        output = self.root / "evidence.json"
        response = {
            "choices": [{"message": {"content": niah.DEFAULT_NEEDLE}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }
        argv = ["run_niah_check.py", "--key", "test", "--fixture", str(fixture),
                "--out", str(output)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(niah, "post",
                                                                    return_value=response):
            self.assertEqual(niah.main(), 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(record["pass"])
        self.assertEqual(record["evidence_mode"], "recall-only")
        self.assertEqual(record["cases"][0]["fixture_identity"]["sha256"],
                         niah.file_identity(fixture)["sha256"])

    def test_dangling_output_namespace_is_rejected_without_request(self) -> None:
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps([{"role": "user", "content": "needle"}]),
                           encoding="utf-8")
        output = self.root / "durable.json"
        output.symlink_to(self.root / "missing.json")
        argv = ["run_niah_check.py", "--key", "test", "--fixture", str(fixture),
                "--out", str(output)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(niah, "post") as post:
            with self.assertRaises(SystemExit):
                niah.main()
        post.assert_not_called()
        self.assertTrue(output.is_symlink())

    def test_fixture_changed_during_run_invalidates_evidence(self) -> None:
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps([{"role": "user", "content": "needle"}]),
                           encoding="utf-8")
        output = self.root / "evidence.json"
        response = {
            "choices": [{"message": {"content": niah.DEFAULT_NEEDLE}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }

        def post_and_mutate(*_args, **_kwargs):
            fixture.write_text(json.dumps([{"role": "user", "content": "changed"}]),
                               encoding="utf-8")
            return response

        argv = ["run_niah_check.py", "--key", "test", "--fixture", str(fixture),
                "--out", str(output)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                niah, "post", side_effect=post_and_mutate):
            self.assertEqual(niah.main(), 1)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(record["pass"])
        self.assertFalse(record["cases"][0]["fixture_unchanged"])

    def test_main_emits_provenance_bound_fresh_prefill_evidence(self) -> None:
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps([
            {"role": "system", "content": "answer exactly"},
            {"role": "user", "content": "find the needle"},
        ]), encoding="utf-8")
        output = self.root / "durable.json"
        response = {
            "choices": [{"message": {"content": niah.DEFAULT_NEEDLE}}],
            "usage": {"prompt_tokens": 1024, "completion_tokens": 3},
        }

        def post_and_log(*_args, **_kwargs):
            self.append_done()
            return response

        argv = [
            "run_niah_check.py", "--key", "test", "--fixture", str(fixture),
            "--out", str(output), "--server-log", str(self.log),
            "--artifact", str(self.artifact), "--serve-bin", str(self.serve),
            "--selection", str(self.selection), "--exact-answer",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                niah, "post", side_effect=post_and_log):
            self.assertEqual(niah.main(), 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(record["pass"])
        self.assertEqual(record["answer_match"], "exact")
        self.assertEqual(
            record["cases"][0]["requests"][0]["answer_sha256"],
            __import__("hashlib").sha256(niah.DEFAULT_NEEDLE.encode()).hexdigest(),
        )
        self.assertEqual(record["evidence_mode"], "provenance-bound")
        self.assertTrue(record["fresh_full_prefill"]["pass"])
        self.assertEqual(record["provenance"]["artifact"]["weights_id"],
                         "r9700-q4g64-n16k16-eval")

    def test_exact_answer_rejects_explanatory_wrapper(self) -> None:
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps([
            {"role": "user", "content": "find the needle"},
        ]), encoding="utf-8")
        output = self.root / "durable.json"
        response = {
            "choices": [{"message": {"content": f"The answer is {niah.DEFAULT_NEEDLE}"}}],
            "usage": {"prompt_tokens": 1024, "completion_tokens": 7},
        }

        def post_and_log(*_args, **_kwargs):
            self.append_done()
            return response

        argv = [
            "run_niah_check.py", "--key", "test", "--fixture", str(fixture),
            "--out", str(output), "--server-log", str(self.log),
            "--artifact", str(self.artifact), "--serve-bin", str(self.serve),
            "--selection", str(self.selection), "--exact-answer",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                niah, "post", side_effect=post_and_log):
            self.assertEqual(niah.main(), 1)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(record["pass"])
        self.assertEqual(record["answer_match"], "exact")


if __name__ == "__main__":
    unittest.main()
