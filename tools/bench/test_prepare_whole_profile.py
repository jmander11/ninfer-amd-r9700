#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.prepare_whole_profile import main, prepare, prepare_low_context
from tools.bench.run_ninfer_bench_matrix import (
    MATRIX_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
)


class PrepareWholeProfileTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        matrix = root / "matrix"
        matrix.mkdir()
        artifact = root / "model.ninfer"
        bench = root / "ninfer_bench"
        artifact.write_bytes(b"artifact")
        bench.write_bytes(b"bench")
        report = matrix / "whole.json"
        report.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": REPORT_SCHEMA_VERSION,
            "load": {"weights_id": "selected"},
            "config": {"concurrency": 4, "kv_value_group": 16,
                       "prefill_chunk": 4096,
                       "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                       "xattention_qualification": False,
                       "spec": "mtp", "draft_tokens": 3,
                       "dflash_verify_width": 0},
            "tests": [
                {"kind": "whole", "n_prompt": 8192, "n_gen": 256},
                {"kind": "whole", "n_prompt": 32768, "n_gen": 256},
            ],
        }), encoding="utf-8")
        identity = lambda path: {
            "path": str(path), "file_size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        command = [
            str(bench), "--weights", str(artifact), "--concurrency", "4",
            "--whole-pg", "8192,256;32768,256", "--output", "json",
            "--output-file", str(report), "-r", "3", "--warmup", "1",
            "--prefill-chunk", "4096",
        ]
        manifest = {
            "artifact_type": "ninfer_bench_matrix_run",
            "schema_version": MATRIX_SCHEMA_VERSION,
            "preset": "pareto-whole", "concurrency": [1, 4],
            "artifact": {**identity(artifact), "weights_id": "selected"},
            "bench": identity(bench),
            "expected_kv_value_group": 16,
            "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "expected_xattention_profile": "dense",
            "selected_prefill_chunk": 4096,
            "power_profile": {
                "required": "auto", "observed": "auto", "rechecked_after": "auto",
                "sysfs_path": "/sys/class/drm/card2/device/power_dpm_force_performance_level",
            },
            "commands": [{"suite": "pareto_whole_inference", "concurrency": 4,
                          "report": str(report), "command": command}],
        }
        (matrix / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return matrix

    def low_context_fixture(self, root: Path):
        matrix = root / "low-context"
        report = matrix / "json/low_context_prefill/c1/prefill_p2048_dense_none.json"
        report.parent.mkdir(parents=True)
        artifact = root / "selected.ninfer"
        bench = root / "ninfer_bench"
        selection = root / "selection.json"
        evaluation = root / "evaluation.json"
        artifact.write_bytes(b"artifact")
        bench.write_bytes(b"bench")
        selection.write_text("{}", encoding="utf-8")
        (root / "corpus.ids").write_text("1 2 3\n", encoding="utf-8")
        identity = lambda path: {
            "path": str(path), "file_size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        report.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": REPORT_SCHEMA_VERSION,
            "config": {
                "concurrency": 1, "prefill_chunk": 2048, "kv_value_group": 16,
                "spec": "none", "draft_tokens": 0, "xattention_qualification": False,
                "dflash_verify_width_requested": 0, "dflash_verify_width": 0,
                "repetitions": 3, "warmup": 1,
            },
            "tests": [{"kind": "pp", "n_prompt": 2048, "n_gen": 0}],
        }), encoding="utf-8")
        command = [
            str(bench), "--weights", str(artifact), "--corpus", str(root / "corpus.ids"),
            "--device", "0", "--concurrency", "1", "-p", "2048", "--prefill-chunk",
            "2048", "--spec", "mtp", "--draft-tokens", "0", "--output", "json",
            "--output-file", str(report), "-r", "3", "--warmup", "1",
        ]
        manifest_path = matrix / "manifest.json"
        manifest = {
            "artifact": {**identity(artifact), "weights_id": "selected"},
            "bench": identity(bench), "expected_kv_value_group": 16,
            "selected_prefill_chunk": 2048, "expected_xattention_profile": "dense",
            "corpus": str(root / "corpus.ids"), "corpus_tokens": 3,
            "corpus_sha256": hashlib.sha256((root / "corpus.ids").read_bytes()).hexdigest(),
            "power_profile": {
                "required": "auto", "observed": "auto", "rechecked_after": "auto",
                "sysfs_path": "/sys/class/drm/card2/device/power_dpm_force_performance_level",
            },
            "commands": [{
                "suite": "low_context_prefill", "case": "prefill_p2048_dense_none",
                "concurrency": 1, "report": str(report), "command": command,
            }],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        terminal = {
            "path": str(selection.resolve()), "sha256": hashlib.sha256(b"{}").hexdigest(),
            "winner": "selected-g16-dense", "winner_artifact": manifest["artifact"],
            "winner_cache_profile": {"value_group": 16},
            "winner_execution_profile": {"xattention_profile": "dense"},
            "selected_prefill_chunk": 2048,
        }
        result = {
            "artifact_type": "ninfer_r9700_low_context_prefill_evaluation",
            "schema_version": 1,
            "manifest": {
                "path": str(manifest_path),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            "artifact": manifest["artifact"], "bench": manifest["bench"],
            "terminal_selection": terminal, "expected_kv_value_group": 16,
            "selected_prefill_chunk": 2048, "minimum_p2048_tok_s": 2000.0,
            "observed_p2048_tok_s": 1800.0, "passes_p2048_gate": False,
            "ladder": [{
                "prompt_tokens": 2048, "prefill_tok_s_mean": 1800.0,
                "report": {
                    "path": str(report), "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                },
            }],
        }
        evaluation.write_text(json.dumps(result), encoding="utf-8")
        return manifest_path, evaluation, selection, bench, artifact, result

    def test_trace_plan_is_bound_to_one_measured_whole_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = prepare(
                self.fixture(root), root / "plan", concurrency=4, prompt_tokens=32768,
                generated_tokens=256, kind="trace", question="which kernels dominate C4?",
                expected_weights_id="selected", expected_kv_value_group=16,
                expected_xattention_profile="dense", expected_prefill_chunk=4096,
            )
            command = payload["profiler_command"]
            self.assertIn("--selected-regions", command)
            self.assertIn("--marker-trace", command)
            self.assertIn("--kernel-trace", command)
            self.assertIn("--memory-copy-trace", command)
            self.assertEqual(payload["benchmark_command"].count("--profile-measured"), 1)
            whole = payload["benchmark_command"].index("--whole-pg")
            self.assertEqual(payload["benchmark_command"][whole + 1], "32768,256")
            repeat = payload["benchmark_command"].index("-r")
            self.assertEqual(payload["benchmark_command"][repeat + 1], "1")
            self.assertTrue((root / "plan" / "commands.sh").is_file())
            self.assertEqual(payload["workload"]["xattention_profile"], "dense")
            self.assertEqual(payload["workload"]["prefill_chunk"], 4096)
            self.assertEqual(payload["required_power_profile"]["value"], "auto")
            commands = (root / "plan" / "commands.sh").read_text(encoding="utf-8")
            self.assertEqual(commands.count("power_dpm_force_performance_level"), 2)
            self.assertEqual(commands.count('= auto'), 2)
            self.assertIn("power-profile-before.txt", commands)
            self.assertIn("power-profile-after.txt", commands)
            self.assertLess(commands.index("power-profile-before.txt"), commands.index("rocprofv3"))
            self.assertGreater(commands.rindex("power-profile-after.txt"), commands.index("rocprofv3"))
            subprocess.run(
                ["bash", "-n", str(root / "plan" / "commands.sh")], check=True,
                capture_output=True, text=True,
            )
            fake_profile = root / "power-profile"
            fake_profile.write_text("auto\n", encoding="utf-8")
            script = root / "plan" / "commands.sh"
            script.write_text(
                commands.replace(
                    "/sys/class/drm/card2/device/power_dpm_force_performance_level",
                    str(fake_profile),
                ).replace("/opt/rocm/bin/rocprofv3", "/bin/false"),
                encoding="utf-8",
            )
            failed = subprocess.run(["bash", str(script)], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertEqual((root / "plan" / "power-profile-before.txt").read_text(), "auto\n")
            self.assertEqual((root / "plan" / "power-profile-after.txt").read_text(), "auto\n")

    def test_dispatch_pmc_is_focused_and_requires_profile_standard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = prepare(
                self.fixture(root), root / "plan", concurrency=4, prompt_tokens=8192,
                generated_tokens=256, kind="dispatch-pmc", question="cache locality",
                expected_weights_id="selected", expected_kv_value_group=16,
                expected_xattention_profile="dense", expected_prefill_chunk=4096,
                kernel_include_regex="attention.*kernel",
            )
            command = payload["profiler_command"]
            self.assertEqual(payload["required_power_profile"]["value"], "profile_standard")
            self.assertEqual(payload["kernel_include_regex"], "attention.*kernel")
            self.assertIn("--kernel-include-regex", command)
            self.assertIn("--marker-trace", command)
            self.assertIn("--kernel-trace", command)
            self.assertIn("GL2C_HIT", command)
            self.assertIn("TCP_REQ_MISS", command)
            self.assertIn("SQ_WAVES", command)
            commands = (root / "plan" / "commands.sh").read_text(encoding="utf-8")
            self.assertIn('test "$POWER_PROFILE_VALUE" = auto', commands)
            self.assertIn('trap restore_auto EXIT', commands)
            self.assertIn('sudo tee', commands)
            self.assertIn('restore_auto\ntrap - EXIT', commands)
            self.assertEqual(
                payload["required_power_profile"]["before_evidence"],
                str(root / "plan" / "power-profile-before.txt"),
            )

    def test_low_context_trace_rewrites_only_profile_outputs_and_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, evaluation, selection, bench, artifact, validated = (
                self.low_context_fixture(root)
            )
            source_command = json.loads(manifest.read_text(encoding="utf-8"))["commands"][0][
                "command"
            ]
            with patch(
                "tools.bench.prepare_whole_profile.validate_ladder", return_value=validated
            ):
                payload = prepare_low_context(
                    manifest, evaluation, selection, root / "trace", kind="trace",
                    question="why is selected dense P2048 below its gate?",
                    expected_executable=bench, expected_artifact=artifact,
                    expected_weights_id="selected", expected_kv_value_group=16,
                    expected_prefill_chunk=2048,
                )
            command = payload["benchmark_command"]
            expected = list(source_command)
            expected[expected.index("-r") + 1] = "1"
            expected[expected.index("--output-file") + 1] = str(
                (root / "trace" / "benchmark-report.json").resolve()
            )
            expected.append("--profile-measured")
            self.assertEqual(command, expected)
            self.assertEqual(payload["workload"]["spec"], "none")
            self.assertEqual(payload["workload"]["generated_tokens"], 0)
            self.assertFalse(payload["low_context_evaluation"]["passes_p2048_gate"])
            self.assertEqual(payload["required_power_profile"]["value"], "auto")

    def test_low_context_pmc_and_authority_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, evaluation, selection, bench, artifact, validated = (
                self.low_context_fixture(root)
            )
            with patch(
                "tools.bench.prepare_whole_profile.validate_ladder", return_value=validated
            ):
                payload = prepare_low_context(
                    manifest, evaluation, selection, root / "pmc", kind="dispatch-pmc",
                    question="are selected linears cache resident?",
                    expected_executable=bench, expected_artifact=artifact,
                    expected_weights_id="selected", expected_kv_value_group=16,
                    expected_prefill_chunk=2048,
                    kernel_include_regex="a8q4g64_linear_prefill_cta_kernel",
                )
            self.assertEqual(payload["required_power_profile"]["value"], "profile_standard")
            self.assertIn("GL2C_HIT", payload["profiler_command"])

            forged = json.loads(evaluation.read_text(encoding="utf-8"))
            forged["observed_p2048_tok_s"] = 9999.0
            evaluation.write_text(json.dumps(forged), encoding="utf-8")
            with (
                patch("tools.bench.prepare_whole_profile.validate_ladder", return_value=validated),
                self.assertRaisesRegex(ValueError, "differs from a fresh validation"),
            ):
                prepare_low_context(
                    manifest, evaluation, selection, root / "forged", kind="trace",
                    question="forged", expected_executable=bench,
                    expected_artifact=artifact, expected_weights_id="selected",
                    expected_kv_value_group=16, expected_prefill_chunk=2048,
                )
            self.assertFalse((root / "forged").exists())

    def test_low_context_rejects_post_validation_authority_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, evaluation, selection, bench, artifact, validated = (
                self.low_context_fixture(root)
            )

            def drift_after_validation(*_args, **_kwargs):
                value = json.loads(manifest.read_text(encoding="utf-8"))
                value["post_validation_drift"] = True
                manifest.write_text(json.dumps(value), encoding="utf-8")
                return validated

            with (
                patch(
                    "tools.bench.prepare_whole_profile.validate_ladder",
                    side_effect=drift_after_validation,
                ),
                self.assertRaisesRegex(ValueError, "recomputed authority chain"),
            ):
                prepare_low_context(
                    manifest, evaluation, selection, root / "drifted", kind="trace",
                    question="drifted manifest", expected_executable=bench,
                    expected_artifact=artifact, expected_weights_id="selected",
                    expected_kv_value_group=16, expected_prefill_chunk=2048,
                )
            self.assertFalse((root / "drifted").exists())

    def test_low_context_rejects_non_none_report_or_wrong_selected_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, evaluation, selection, bench, artifact, validated = (
                self.low_context_fixture(root)
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
            report = Path(document["commands"][0]["report"])
            report_value = json.loads(report.read_text(encoding="utf-8"))
            report_value["config"]["spec"] = "mtp"
            report.write_text(json.dumps(report_value), encoding="utf-8")
            with (
                patch("tools.bench.prepare_whole_profile.validate_ladder", return_value=validated),
                self.assertRaisesRegex(ValueError, "spec-none P2048 3/1"),
            ):
                prepare_low_context(
                    manifest, evaluation, selection, root / "wrong-spec", kind="trace",
                    question="wrong spec", expected_executable=bench,
                    expected_artifact=artifact, expected_weights_id="selected",
                    expected_kv_value_group=16, expected_prefill_chunk=2048,
                )
            self.assertFalse((root / "wrong-spec").exists())

            report_value["config"]["spec"] = "none"
            report.write_text(json.dumps(report_value), encoding="utf-8")
            with (
                patch("tools.bench.prepare_whole_profile.validate_ladder", return_value=validated),
                self.assertRaisesRegex(ValueError, "explicitly selected dense route"),
            ):
                prepare_low_context(
                    manifest, evaluation, selection, root / "wrong-group", kind="trace",
                    question="wrong group", expected_executable=bench,
                    expected_artifact=artifact, expected_weights_id="selected",
                    expected_kv_value_group=32, expected_prefill_chunk=2048,
                )
            self.assertFalse((root / "wrong-group").exists())

    def test_low_context_documented_cli_shape_creates_one_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, evaluation, selection, bench, artifact, validated = (
                self.low_context_fixture(root)
            )
            output = root / "cli-plan"
            with patch(
                "tools.bench.prepare_whole_profile.validate_ladder", return_value=validated
            ):
                self.assertEqual(main([
                    "--low-context-manifest", str(manifest),
                    "--low-context-evaluation", str(evaluation),
                    "--terminal-selection", str(selection),
                    "--executable", str(bench), "--artifact", str(artifact),
                    "--out", str(output), "--concurrency", "1",
                    "--prompt-tokens", "2048", "--generated-tokens", "0",
                    "--kind", "trace", "--question", "P2048 trace",
                    "--expected-weights-id", "selected",
                    "--expected-kv-value-group", "16",
                    "--expected-xattention-profile", "dense",
                    "--expected-prefill-chunk", "2048",
                ]), 0)
            self.assertTrue((output / "plan.json").is_file())
            self.assertTrue((output / "commands.sh").is_file())

    def test_rejects_failed_or_unmeasured_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.fixture(root)
            (matrix / "failures.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "retained failures"):
                prepare(
                    matrix, root / "plan", concurrency=4, prompt_tokens=32768,
                    generated_tokens=256, kind="dispatch-pmc", question="wave occupancy",
                    expected_weights_id="selected", expected_kv_value_group=16,
                    expected_xattention_profile="dense", expected_prefill_chunk=4096,
                    kernel_include_regex="attention",
                )
            (matrix / "failures.json").unlink()
            with self.assertRaisesRegex(ValueError, "absent from the measured matrix"):
                prepare(
                    matrix, root / "plan", concurrency=4, prompt_tokens=16384,
                    generated_tokens=256, kind="dispatch-pmc", question="wave occupancy",
                    expected_weights_id="selected", expected_kv_value_group=16,
                    expected_xattention_profile="dense", expected_prefill_chunk=4096,
                    kernel_include_regex="attention",
                )

    def test_rejects_non_product_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, r"\[1, 4\]"):
                prepare(
                    self.fixture(root), root / "plan", concurrency=5,
                    prompt_tokens=32768, generated_tokens=256, kind="trace",
                    question="non-product C5", expected_weights_id="selected",
                    expected_kv_value_group=16,
                    expected_xattention_profile="dense", expected_prefill_chunk=4096,
                )

    def test_rejects_wrong_selected_route_or_unfocused_pmc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "selected XAttention"):
                prepare(
                    matrix, root / "profile", concurrency=4, prompt_tokens=8192,
                    generated_tokens=256, kind="trace", question="wrong route",
                    expected_weights_id="selected", expected_kv_value_group=16,
                    expected_xattention_profile="b128-s16-tau900",
                    expected_prefill_chunk=4096,
                )
            with self.assertRaisesRegex(ValueError, "selected prefill chunk"):
                prepare(
                    matrix, root / "chunk", concurrency=4, prompt_tokens=8192,
                    generated_tokens=256, kind="trace", question="wrong chunk",
                    expected_weights_id="selected", expected_kv_value_group=16,
                    expected_xattention_profile="dense", expected_prefill_chunk=2048,
                )
            with self.assertRaisesRegex(ValueError, "kernel include regex"):
                prepare(
                    matrix, root / "pmc", concurrency=4, prompt_tokens=8192,
                    generated_tokens=256, kind="dispatch-pmc", question="unfocused counters",
                    expected_weights_id="selected", expected_kv_value_group=16,
                    expected_xattention_profile="dense", expected_prefill_chunk=4096,
                )


if __name__ == "__main__":
    unittest.main()
