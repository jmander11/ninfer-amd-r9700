import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from tools.bench import run_scalar_base_whole_ab as gate


def speculative():
    return {
        "enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
        "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
        "acceptance_length": None, "accepted_per_position": [],
    }


def bench_report(role, command, seconds=1.0, token=17):
    profile = gate.CONTROL_PROFILE if role == "A" else gate.CANDIDATE_PROFILE
    config = {
        "max_context": 2048, "prefill_chunk": 4096, "kv_cache_format": "fp8-k-int4-v",
        "kv_value_group": 16, "kv_plane_layouts": gate.KV_PLANES, "q4_activation_bits": 8,
        "q4_prefill_cta_profile": profile, "w8_activation_bits": 8,
        "fp8_qk_wmma_enabled": True,
        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
        "fp8_qk_wmma_t1_min_context": 64, "fp8_qk_wmma_t2_min_context": 320,
        "xattention_qualification": False, "concurrency": 1,
        "pending_timeout_ms": 0xFFFFFFFF, "pending_deadline": "unbounded",
        "spec": "none", "draft_tokens": 0, "speculative_execution": False,
        "dflash_verify_width_requested": 0, "dflash_verify_width": 0,
        "proposal_head": "full", "use_device_graph": True, "retain_token_ids": True,
        "decode_path": "device_graph", "decode_graph_prime": {"primed": False, "output_tokens": 0},
        "repetitions": 1, "warmup": 1,
        "corpus_path": str(gate.CORPUS.relative_to(gate.ROOT)), "corpus_tokens": 4,
    }
    arena = {"capacity_bytes": 8, "used_bytes": 8, "peak_used_bytes": 8}
    memory = {
        "device": 0, "max_context": 2048, "kv_capacity_mode": "explicit", "kv_capacity": 2048,
        "kv_capacity_page_groups": 32, "kv_capacity_max_page_groups": 32,
        "kv_cache_format": "fp8-k-int4-v", "weights": dict(arena), "sequence": dict(arena),
        "workspace": dict(arena), "request_transient": dict(arena),
        "minimum_runtime_reservation_bytes": 1, "kv_capacity_increment_bytes": 1,
        "runtime_reservation_bytes": 1, "available_after_weights_bytes": 1,
        "available_after_startup_bytes": 1, "kv_capacity_headroom_bytes": 0,
        "planned_slack_bytes": 0, "device_graph_allowance_bytes": 0,
        "device_graph_observed_bytes": 0, "kv_payload_bytes": 1,
    }
    spec = speculative()
    timings = {
        "prepare_seconds": 0.001, "vision_seconds": 0, "prefill_seconds": seconds,
        "decode_seconds": 0, "total_seconds": seconds + 0.001,
    }
    test = {
        "label": "pp2048", "kind": "pp", "n_prompt": 2048, "n_gen": 0,
        "requested_output_tokens": 1, "prefill_tok_s_mean": 2048 / seconds,
        "prefill_tok_s_stddev": 0, "decode_output_tok_s_mean": None,
        "decode_output_tok_s_stddev": None, "decode_engine_tok_s_mean": None,
        "decode_engine_tok_s_stddev": None, "whole_output_tok_s_mean": None,
        "whole_output_tok_s_stddev": None, "prepare_seconds_mean": 0.001,
        "prepare_seconds_stddev": 0, "prefill_seconds_mean": seconds,
        "prefill_seconds_stddev": 0, "decode_seconds_mean": None,
        "decode_seconds_stddev": None, "total_seconds_mean": seconds + 0.001,
        "total_seconds_stddev": 0, "workspace_peak_bytes": 8,
        "workspace_allocator_peak_bytes": 8, "speculative": spec,
        "reps": [{"generated_output_tokens": 1, "decode_output_tokens": None,
                  "decode_engine_tokens": None, "generated_token_ids_by_lane": [[token]],
                  "timings": timings, "speculative": spec}],
    }
    return {
        "schema_version": 20, "artifact_type": "ninfer_bench_report", "tool": "ninfer_bench",
        "command": " ".join(command),
        "environment": {"gpu_name": "AMD Radeon AI PRO R9700", "architecture_name": "gfx1201",
                        "hip_runtime_version": "10.0", "hip_driver_version": "10.0", "device_id": 0},
        "artifact": {"path": str(gate.ARTIFACT.relative_to(gate.ROOT)),
                     "file_size_bytes": gate.ARTIFACT_SIZE_BYTES},
        "load": {"target": gate.TARGET_ID, "weights_id": gate.WEIGHTS_ID, "load_seconds": 2.0,
                 "upload_seconds": 1.0, "artifact_bytes_read": 1, "host_to_device_bytes": 1,
                 "peak_staging_bytes": 1, "tensor_count": 1, "resource_count": 1},
        "memory": memory, "config": config, "tests": [test],
    }


def decision_samples(control_ms=1000.0, candidate_ms=980.0,
                     candidate_prefill_ms=None):
    if candidate_prefill_ms is None:
        candidate_prefill_ms = candidate_ms
    return [
        {"role": role,
         "prefill_ms": control_ms if role == "A" else candidate_prefill_ms,
         "total_ms": control_ms if role == "A" else candidate_ms,
         "tokens": [[17]], "workspace_peak_bytes": 8, "workspace_allocator_peak_bytes": 8,
         "environment": {"runtime": "fixed"}}
        for role in gate.ROLE_ORDER
    ]


class WholeAbTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        paths = {
            "ROOT": self.root,
            "CONTROL_BINARY": self.root / "control/ninfer_bench",
            "CANDIDATE_BINARY": self.root / "candidate/ninfer_bench",
            "CONTROL_BUILD_DIRECTORY": self.root / "control",
            "CANDIDATE_BUILD_DIRECTORY": self.root / "candidate",
            "OPERATOR_PRODUCT_EXECUTABLE":
                self.root / "candidate/tests/ninfer_r9700_a8q4_scalar_base_product_qual",
            "OPERATOR_AUTHORITY": self.root / "profiles/operator.json",
            "ARTIFACT": self.root / "out/model.ninfer",
            "CORPUS": self.root / "bench/corpus.ids",
            "REPORT": self.root / "profiles/result.json",
            "RAW_DIRECTORY": self.root / "profiles/result",
        }
        for name, value in paths.items():
            self.stack.enter_context(mock.patch.object(gate, name, value))
        for path in (gate.CONTROL_BINARY, gate.CANDIDATE_BINARY,
                     gate.OPERATOR_PRODUCT_EXECUTABLE, gate.ARTIFACT, gate.CORPUS):
            path.parent.mkdir(parents=True, exist_ok=True)
        gate.CONTROL_BINARY.write_bytes(b"control")
        gate.CANDIDATE_BINARY.write_bytes(b"candidate")
        gate.OPERATOR_PRODUCT_EXECUTABLE.write_bytes(b"operator-product")
        gate.CONTROL_BINARY.chmod(0o755)
        gate.CANDIDATE_BINARY.chmod(0o755)
        gate.OPERATOR_PRODUCT_EXECUTABLE.chmod(0o755)
        linear = self.root / "src/ops/r9700/linear/r9700_linear.hip"
        sources = (
            self.root / "CMakeLists.txt",
            self.root / "src/ops/r9700/linear/r9700_linear.h",
            linear,
            self.root / "src/ops/r9700/linear/r9700_q4_activation_profile.h",
        )
        for source in sources:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"source:{source.name}\n", encoding="utf-8")
        self.stack.enter_context(mock.patch.object(gate, "LINEAR_SOURCE", linear))
        self.stack.enter_context(mock.patch.object(gate, "BUILD_SOURCE_PATHS", sources))
        directory = json.dumps(
            {"identity": {"model_id": gate.MODEL_ID, "weights_id": gate.WEIGHTS_ID}, "objects": []},
            separators=(",", ":"),
        ).encode()
        gate.ARTIFACT.write_bytes(gate.NINFER_PREFIX.pack(gate.NINFER_MAGIC, len(directory)) + directory)
        self.stack.enter_context(mock.patch.object(
            gate, "ARTIFACT_SIZE_BYTES", gate.ARTIFACT.stat().st_size
        ))
        self.stack.enter_context(mock.patch.object(
            gate, "ARTIFACT_SHA256", gate.file_sha256(gate.ARTIFACT)
        ))
        gate.CORPUS.write_text("1 2 3 4\n", encoding="utf-8")
        gate.REPORT.parent.mkdir(parents=True)

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def test_decision_promotes_and_keeps_floor_separate(self):
        result = gate.decide(decision_samples(1100.0, 1080.0, 1100.0))
        self.assertTrue(result["promotion_pass"])
        self.assertEqual(result["robust_whole_saving_lower_ms"], 20.0)
        self.assertFalse(result["floor_diagnostic_pass"])

    def test_decision_requires_one_token_vector_across_all_16_runs(self):
        samples = decision_samples()
        samples[4]["tokens"] = [[18]]
        result = gate.decide(samples)
        self.assertFalse(result["promotion_pass"])
        self.assertFalse(result["all_16_token_vectors_identical"])

    def test_decision_rejects_output_or_workspace_mismatch(self):
        samples = decision_samples(); samples[1]["tokens"] = [[18]]
        result = gate.decide(samples)
        self.assertFalse(result["promotion_pass"])
        self.assertFalse(result["exact_semantic_output_parity"])
        samples = decision_samples(); samples[1]["workspace_peak_bytes"] = 9
        self.assertFalse(gate.decide(samples)["workspace_identity_pass"])

    def test_decision_rejects_large_pair_regression_and_wrong_order(self):
        samples = decision_samples(); samples[1]["total_ms"] = 1051.0
        self.assertFalse(gate.decide(samples)["no_candidate_sample_over_paired_control_1p05"])
        samples = decision_samples(); samples[0], samples[1] = samples[1], samples[0]
        with self.assertRaisesRegex(ValueError, "sample order"):
            gate.decide(samples)

    def test_report_validation_binds_profile_command_and_exact_token(self):
        command = gate.command_for("B")
        report = bench_report("B", command, 0.98)
        self.assertEqual(gate.validate_bench_report(report, "B", command)["tokens"], [[17]])
        report["config"]["q4_prefill_cta_profile"] = gate.CONTROL_PROFILE
        with self.assertRaisesRegex(ValueError, "configuration"):
            gate.validate_bench_report(report, "B", command)

    def test_artifact_inspection_requires_exact_size_and_sha(self):
        self.assertEqual(gate.inspect_artifact(gate.ARTIFACT)["weights_id"], gate.WEIGHTS_ID)
        with mock.patch.object(gate, "ARTIFACT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "artifact bytes"):
                gate.inspect_artifact(gate.ARTIFACT)

    def test_report_validation_rejects_semantic_and_schema_mutations(self):
        mutations = [
            lambda report: report.__setitem__("command", report["command"] + " --unknown"),
            lambda report: report["config"].__setitem__("prefill_chunk", 2048),
            lambda report: report["config"].__setitem__("retain_token_ids", False),
            lambda report: report["tests"][0]["reps"].append(report["tests"][0]["reps"][0]),
            lambda report: report["tests"][0]["reps"][0].pop("generated_token_ids_by_lane"),
            lambda report: report.__setitem__("unknown", 1),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                command = gate.command_for("A")
                report = bench_report("A", command)
                mutation(report)
                with self.assertRaises(ValueError):
                    gate.validate_bench_report(report, "A", command)

    def test_publish_is_create_only_and_preserves_existing_output(self):
        output = self.root / "existing.json"
        output.write_text("foreign\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            gate.publish_json(output, {"ok": True}, lambda value: None)
        self.assertEqual(output.read_text(encoding="utf-8"), "foreign\n")

    def test_publish_validation_failure_leaves_no_output(self):
        output = self.root / "failed.json"

        def reject(_value):
            raise ValueError("invalid authority")

        with self.assertRaisesRegex(ValueError, "invalid authority"):
            gate.publish_json(output, {"ok": False}, reject)
        self.assertFalse(os.path.lexists(output))
        self.assertEqual(list(self.root.glob(".failed.json.*.pending")), [])

    def test_owned_cleanup_preserves_a_replacement_inode(self):
        output = self.root / "owned.json"
        replacement = self.root / "replacement.json"
        output.write_text("owned\n", encoding="utf-8")
        owner = gate._inode(output)
        replacement.write_text("replacement\n", encoding="utf-8")
        replacement.replace(output)
        self.assertFalse(gate._unlink_if_owned(output, owner))
        self.assertEqual(output.read_text(encoding="utf-8"), "replacement\n")

    def test_campaign_rejects_preexisting_raw_namespace_without_execution(self):
        gate.RAW_DIRECTORY.mkdir()
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(gate.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "namespace must be fresh"):
                    gate.run_campaign()
        finally:
            os.chdir(previous)
        run.assert_not_called()

    def test_source_matched_build_receipts_allow_only_qualification_flag(self):
        linear = gate.LINEAR_SOURCE
        base = ["/opt/rocm/llvm/bin/clang++", "-O3", "--offload-arch=gfx1201"]
        for build, qualified in (
            (gate.CONTROL_BUILD_DIRECTORY, False),
            (gate.CANDIDATE_BUILD_DIRECTORY, True),
        ):
            cache_values = {
                **gate.MATCHED_CACHE_OPTIONS,
                "NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION":
                    "ON" if qualified else "OFF",
            }
            (build / "CMakeCache.txt").write_text(
                "".join(f"{key}:STRING={value}\n" for key, value in cache_values.items()),
                encoding="utf-8",
            )
            command = [*base]
            if qualified:
                command.append("-DNINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION=1")
            command.extend(["-o", "linear.o", "-x", "hip", "-c", str(linear)])
            (build / "compile_commands.json").write_text(json.dumps([{
                "directory": str(build), "command": " ".join(command),
                "file": str(linear), "output": "linear.o",
            }]), encoding="utf-8")
        receipt = gate.build_receipts()
        self.assertFalse(receipt["control"]["scalar_base_qualification"])
        self.assertTrue(receipt["candidate"]["scalar_base_qualification"])
        database = gate.CANDIDATE_BUILD_DIRECTORY / "compile_commands.json"
        payload = json.loads(database.read_text(encoding="utf-8"))
        payload[0]["command"] = payload[0]["command"].replace("-O3", "-O2")
        database.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ beyond qualification flag"):
            gate.build_receipts()

    def test_operator_authority_binds_admission_and_candidate_identities(self):
        profile = self.root / "src/ops/r9700/linear/r9700_q4_activation_profile.h"
        builds = {
            "sources": {
                str(gate.LINEAR_SOURCE.relative_to(gate.ROOT)):
                    gate._regular_identity(gate.LINEAR_SOURCE),
                str(profile.relative_to(gate.ROOT)): gate._regular_identity(profile),
            },
            "candidate": {
                "operator_product_executable":
                    gate._regular_identity(gate.OPERATOR_PRODUCT_EXECUTABLE, executable=True),
            },
        }

        def payload():
            return {
                "schema": "ninfer.r9700.a8q4-n16k16-scalar-base-product-gate.v1",
                "decision": {
                    "admission_pass": True,
                    "robust_weighted_saving_lower_ms": 5.25,
                    "required_robust_weighted_saving_lower_ms": 5.0,
                    "maximum_every_cell_robust_ratio_upper": 1.01,
                    "cells": [{"robust_ratio_upper": 1.0}],
                },
                "static": {"fields": {
                    "product_executable_sha256": builds["candidate"]
                        ["operator_product_executable"]["sha256"],
                    "source_sha256": builds["sources"]
                        [str(gate.LINEAR_SOURCE.relative_to(gate.ROOT))]["sha256"],
                    "profile_sha256": builds["sources"]
                        [str(profile.relative_to(gate.ROOT))]["sha256"],
                }},
            }

        value = payload()
        gate.OPERATOR_AUTHORITY.write_text(json.dumps(value), encoding="utf-8")
        with mock.patch.object(
            gate, "OPERATOR_AUTHORITY_SHA256", gate.file_sha256(gate.OPERATOR_AUTHORITY)
        ):
            self.assertTrue(gate.inspect_operator_authority(builds)["admission_pass"])

        mutations = (
            lambda item: item.__setitem__("schema", "wrong"),
            lambda item: item["decision"].__setitem__("admission_pass", False),
            lambda item: item["decision"].__setitem__(
                "robust_weighted_saving_lower_ms", 4.99
            ),
            lambda item: item["decision"]["cells"][0].__setitem__(
                "robust_ratio_upper", 1.011
            ),
            lambda item: item["static"]["fields"].__setitem__(
                "product_executable_sha256", "0" * 64
            ),
            lambda item: item["static"]["fields"].__setitem__(
                "source_sha256", "0" * 64
            ),
            lambda item: item["static"]["fields"].__setitem__(
                "profile_sha256", "0" * 64
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = payload()
                mutation(value)
                gate.OPERATOR_AUTHORITY.write_text(json.dumps(value), encoding="utf-8")
                with mock.patch.object(
                    gate, "OPERATOR_AUTHORITY_SHA256",
                    gate.file_sha256(gate.OPERATOR_AUTHORITY),
                ), self.assertRaises(ValueError):
                    gate.inspect_operator_authority(builds)

    def test_campaign_rejects_successful_process_stderr_and_cleans_namespace(self):
        identities = {"fixed": {"sha256": "a" * 64}}
        power = {
            "command": ["/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname",
                        "--showprofile", "--showperflevel", "--json"],
            "device": {"ordinal": 0, "name": "AMD Radeon AI PRO R9700",
                       "architecture": "gfx1201"},
            "rocm_smi_performance_level": "auto", "sysfs_path": str(gate.POWER),
            "sysfs_value": "auto",
        }
        command = gate.command_for("A")
        process = subprocess.CompletedProcess(
            command, 0, json.dumps(bench_report("A", command)),
            gate.expected_bench_stderr() + "warning\n",
        )
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(gate, "input_identities", return_value=identities), \
                 mock.patch.object(gate, "power_identity", return_value=power), \
                 mock.patch.object(gate.subprocess, "run", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "stderr protocol"):
                    gate.run_campaign()
        finally:
            os.chdir(previous)
        self.assertFalse(os.path.lexists(gate.RAW_DIRECTORY))
        self.assertFalse(os.path.lexists(gate.REPORT))

    def test_benchmark_stderr_protocol_is_exact(self):
        expected = gate.expected_bench_stderr()
        gate.validate_bench_stderr(expected)
        mutations = (
            "",
            expected + "warning\n",
            expected.replace("max_context=2048", "max_context=4096"),
            expected.replace("warmup=1", "warmup=2"),
            expected.replace("test 1/1 pp2048", "test 1/2 pp2048"),
        )
        for value in mutations:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "stderr protocol"
            ):
                gate.validate_bench_stderr(value)

    def test_mid_campaign_cleanup_preserves_foreign_replacement(self):
        identities = {"fixed": {"sha256": "a" * 64}}
        power = {
            "command": ["/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname",
                        "--showprofile", "--showperflevel", "--json"],
            "device": {"ordinal": 0, "name": "AMD Radeon AI PRO R9700",
                       "architecture": "gfx1201"},
            "rocm_smi_performance_level": "auto", "sysfs_path": str(gate.POWER),
            "sysfs_value": "auto",
        }
        calls = 0

        def fake_run(command, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return subprocess.CompletedProcess(
                    command, 0, json.dumps(bench_report("A", list(command))),
                    gate.expected_bench_stderr(),
                )
            raw = gate._raw_path(0, "A")
            replacement = gate.RAW_DIRECTORY / "replacement"
            replacement.write_text("foreign\n", encoding="utf-8")
            replacement.replace(raw)
            return subprocess.CompletedProcess(command, 7, "", "failed\n")

        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(gate, "input_identities", return_value=identities), \
                 mock.patch.object(gate, "power_identity", return_value=power), \
                 mock.patch.object(gate.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "sample 2 failed"):
                    gate.run_campaign()
        finally:
            os.chdir(previous)
        self.assertEqual(gate._raw_path(0, "A").read_text(encoding="utf-8"), "foreign\n")
        self.assertTrue(gate.RAW_DIRECTORY.is_dir())
        self.assertFalse(os.path.lexists(gate.REPORT))

    def test_campaign_has_only_eight_balanced_pairs_and_process_warmups(self):
        identities = {"fixed": {"sha256": "a" * 64}}
        power = {
            "command": ["/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname",
                        "--showprofile", "--showperflevel", "--json"],
            "device": {"ordinal": 0, "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201"},
            "rocm_smi_performance_level": "auto", "sysfs_path": str(gate.POWER), "sysfs_value": "auto",
        }
        calls = []

        def fake_run(command, **kwargs):
            role = "A" if command[0] == str(gate.CONTROL_BINARY.relative_to(gate.ROOT)) else "B"
            calls.append((role, list(command), kwargs))
            seconds = 1.0 if role == "A" else 0.98
            return subprocess.CompletedProcess(
                command, 0, json.dumps(bench_report(role, list(command), seconds)),
                gate.expected_bench_stderr(),
            )

        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(gate, "input_identities", return_value=identities), \
                 mock.patch.object(gate, "power_identity", return_value=power), \
                 mock.patch.object(gate.subprocess, "run", side_effect=fake_run):
                report = gate.run_campaign()
        finally:
            os.chdir(previous)
        self.assertEqual([role for role, _, _ in calls], list(gate.ROLE_ORDER))
        self.assertEqual(len(calls), 16)
        for _, command, kwargs in calls:
            self.assertEqual(command[command.index("--warmup") + 1], "1")
            self.assertEqual(command[command.index("-r") + 1], "1")
            self.assertIn("--retain-token-ids", command)
            self.assertNotIn("--output-file", command)
            self.assertEqual(kwargs["cwd"], self.root)
        self.assertTrue(report["decision"]["promotion_pass"])
        self.assertTrue(gate.REPORT.is_file())
        self.assertEqual(os.stat(gate.REPORT).st_mode & 0o777, 0o444)
        raw_reports = list(gate.RAW_DIRECTORY.glob("*.json"))
        self.assertEqual(len(raw_reports), 16)
        self.assertTrue(all(os.stat(path).st_mode & 0o777 == 0o444
                            for path in raw_reports))
        self.assertTrue(all(sample["raw_report"]["uid"] == os.getuid()
                            and sample["raw_report"]["mode"] == 0o444
                            for sample in report["samples"]))


if __name__ == "__main__":
    unittest.main()
