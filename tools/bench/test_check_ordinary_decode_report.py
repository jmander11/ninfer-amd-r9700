from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.check_ordinary_decode_report import validate_matrix
from tools.bench.run_ninfer_bench_matrix import (
    FP8_QK_WMMA_PROFILE,
    FP8_QK_WMMA_T1_MIN_CONTEXT,
    FP8_QK_WMMA_T2_MIN_CONTEXT,
    MATRIX_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
    R9700_POWER_PROFILE,
    add_repetition_args,
    build_cases,
)


class OrdinaryDecodeReportTest(unittest.TestCase):
    def test_documented_direct_script_imports_from_repo_root(self) -> None:
        root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "tools/bench/check_ordinary_decode_report.py", "--help"],
            cwd=root, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--executable", completed.stdout)

    def fixture(self, root: Path) -> tuple[Path, Path, Path, dict]:
        executable = root / "ninfer_bench"
        artifact = root / "model.ninfer"
        corpus = root / "corpus.ids"
        executable.write_bytes(b"exe")
        artifact.write_bytes(b"artifact")
        corpus.write_bytes(b"tokens")
        matrix = root / "matrix"
        report_path = matrix / "json/ordinary_decode/c1/whole_p8192_g256_none_graph.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text("{}", encoding="utf-8")
        case = build_cases("ordinary-diagnostic")[0]
        command = add_repetition_args([
            str(executable), "--weights", str(artifact), "--corpus", str(corpus),
            "--device", "0", "--concurrency", "1", *case.args,
            "--output", "json", "--output-file", str(report_path),
        ], case, None, None)
        manifest = {
            "artifact_type": "ninfer_bench_matrix_run", "schema_version": MATRIX_SCHEMA_VERSION,
            "preset": "ordinary-diagnostic", "dry_run": False,
            "case_count": 1, "point_count": 1, "concurrency": [1],
            "expected_kv_value_group": 16, "expected_q4_activation_bits": 8,
            "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "expected_w8_activation_bits": 8, "expected_fp8_qk_wmma_enabled": True,
            "expected_fp8_qk_wmma_profile": FP8_QK_WMMA_PROFILE,
            "expected_fp8_qk_wmma_t1_min_context": FP8_QK_WMMA_T1_MIN_CONTEXT,
            "expected_fp8_qk_wmma_t2_min_context": FP8_QK_WMMA_T2_MIN_CONTEXT,
            "expected_xattention_profile": "b128-s16-tau900",
            "power_profile": {"required": "auto", "sysfs_path": str(R9700_POWER_PROFILE),
                              "observed": "auto", "rechecked_after": "auto"},
            "bench": {"path": str(executable), "sha256": "a" * 64},
            "artifact": {"path": str(artifact), "sha256": "b" * 64},
            "corpus": str(corpus),
            "corpus_sha256": __import__("hashlib").sha256(corpus.read_bytes()).hexdigest(),
            "commands": [{"suite": "ordinary_decode",
                          "case": "whole_p8192_g256_none_graph", "concurrency": 1,
                          "report": str(report_path), "command": command}],
        }
        (matrix / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        disabled = {
            "enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
            "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
            "acceptance_length": None, "accepted_per_position": [],
        }
        report = {
            "command": " ".join(command),
            "config": {"spec": "none", "draft_tokens": 0,
                       "speculative_execution": False, "use_device_graph": True,
                       "decode_path": "device_graph", "concurrency": 1,
                       "repetitions": 3, "warmup": 1, "prefill_chunk": 4096},
            "tests": [{"kind": "whole", "n_prompt": 8192, "n_gen": 256,
                       "prepare_seconds_mean": 0.1, "prepare_seconds_stddev": 0.0,
                       "prefill_seconds_mean": 80.0, "prefill_seconds_stddev": 0.0,
                       "decode_seconds_mean": 12.0, "decode_seconds_stddev": 0.0,
                       "total_seconds_mean": 92.0, "total_seconds_stddev": 0.0,
                       "prefill_tok_s_mean": 102.4, "prefill_tok_s_stddev": 0.0,
                       "decode_output_tok_s_mean": 256.0 / 12.0,
                       "decode_output_tok_s_stddev": 0.0,
                       "decode_engine_tok_s_mean": 256.0 / 12.0,
                       "decode_engine_tok_s_stddev": 0.0,
                       "whole_output_tok_s_mean": 257.0 / 92.0,
                       "whole_output_tok_s_stddev": 0.0,
                       "speculative": disabled,
                       "reps": [{"generated_output_tokens": 257,
                                 "decode_output_tokens": 256,
                                 "decode_engine_tokens": 256,
                                 "timings": {"prepare_seconds": 0.1,
                                             "vision_seconds": 0.0,
                                             "prefill_seconds": 80.0,
                                             "decode_seconds": 12.0,
                                             "total_seconds": 92.0},
                                 "speculative": disabled} for _ in range(3)]}],
        }
        return matrix, executable, artifact, report

    def validate(self, matrix: Path, executable: Path, artifact: Path, report: dict,
                 power: str = "auto") -> dict:
        manifest = json.loads((matrix / "manifest.json").read_text(encoding="utf-8"))
        with patch("tools.bench.check_ordinary_decode_report.inspect_executable",
                   return_value=manifest["bench"]), \
             patch("tools.bench.check_ordinary_decode_report.inspect_artifact",
                   return_value=manifest["artifact"]), \
             patch("tools.bench.check_ordinary_decode_report.load_bench_report",
                   return_value=report):
            return validate_matrix(matrix, executable, artifact, lambda _path: power)

    def test_accepts_exact_non_speculative_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            self.assertIs(self.validate(matrix, executable, artifact, report), report)

    def test_rejects_live_non_auto(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "live power"):
                self.validate(matrix, executable, artifact, report, "profile_standard")

    def test_rejects_speculative_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            report["config"]["spec"] = "mtp"
            report["config"]["draft_tokens"] = 3
            report["config"]["speculative_execution"] = True
            with self.assertRaisesRegex(ValueError, "non-speculative"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_missing_phase_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            report["tests"][0]["decode_output_tok_s_mean"] = None
            with self.assertRaisesRegex(ValueError, "separate prefill/decode"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_forged_manifest_and_report_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            manifest_path = matrix / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            forged = [*manifest["commands"][0]["command"]]
            forged[forged.index("--draft-tokens") + 1] = "3"
            forged.extend(["--lm-head-draft"])
            manifest["commands"][0]["command"] = forged
            report["command"] = " ".join(forged)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed protocol"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_nonzero_repetition_speculative_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            report["tests"][0]["reps"][0]["speculative"] = {
                **report["tests"][0]["speculative"], "rounds": 1,
            }
            with self.assertRaisesRegex(ValueError, "separate prefill/decode"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_missing_kv_plane_layout_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            manifest_path = matrix / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["expected_kv_plane_layouts"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest contract"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_forged_fp8_qk_route_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            manifest_path = matrix / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["expected_fp8_qk_wmma_profile"] = "forged"
            manifest["expected_fp8_qk_wmma_t1_min_context"] = 1
            manifest["expected_fp8_qk_wmma_t2_min_context"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest contract"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_boolean_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            report["config"]["warmup"] = True
            with self.assertRaisesRegex(ValueError, "schedule fields"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_boolean_manifest_counts_and_concurrency(self) -> None:
        mutations = [
            lambda manifest: manifest.__setitem__("case_count", True),
            lambda manifest: manifest.__setitem__("point_count", True),
            lambda manifest: manifest.__setitem__("concurrency", [True]),
            lambda manifest: manifest["commands"][0].__setitem__("concurrency", True),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                matrix, executable, artifact, report = self.fixture(Path(directory))
                manifest_path = matrix / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest contract|command geometry"):
                    self.validate(matrix, executable, artifact, report)

    def test_rejects_forged_repetition_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            report["tests"][0]["reps"][0]["decode_engine_tokens"] = 0
            with self.assertRaisesRegex(ValueError, "token counts"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_forged_repetition_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            report["tests"][0]["reps"][0]["timings"]["decode_seconds"] = -1.0
            with self.assertRaisesRegex(ValueError, "timings"):
                self.validate(matrix, executable, artifact, report)

    def test_rejects_aggregate_not_derived_from_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, executable, artifact, report = self.fixture(Path(directory))
            report["tests"][0]["decode_output_tok_s_mean"] = 1000.0
            with self.assertRaisesRegex(ValueError, "aggregate differs"):
                self.validate(matrix, executable, artifact, report)


if __name__ == "__main__":
    unittest.main()
