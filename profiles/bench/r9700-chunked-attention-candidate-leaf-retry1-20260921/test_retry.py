import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("leaf_retry", Path(__file__).with_name("retry.py"))
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


class RetryTest(unittest.TestCase):
    def test_initial_idle_requires_zero_busy_sample(self):
        with patch.object(retry.base, "require_auto_power_profile"), patch.object(
            Path, "read_text", side_effect=["59912192", "12", "59912192", "0"]
        ), patch.object(retry.time, "sleep") as sleep:
            retry.wait_for_idle()
        sleep.assert_called_once()

    def test_initial_idle_does_not_admit_sustained_busy_gpu(self):
        with patch.object(retry.base, "require_auto_power_profile"), patch.object(
            Path, "read_text", side_effect=["59912192", "12"]
        ), patch.object(retry.time, "monotonic", side_effect=[0, 10]):
            with self.assertRaisesRegex(ValueError, "idle wait timed out"):
                retry.wait_for_idle()

    def test_owned_release_waits_for_deferred_teardown(self):
        with patch.object(retry.base, "require_auto_power_profile"), patch.object(
            Path, "read_text", side_effect=[str(14 * 1024 ** 3), "1024"]
        ), patch.object(retry.time, "sleep") as sleep:
            retry.wait_for_release()
        sleep.assert_called_once()
        self.assertLessEqual(sleep.call_args.args[0], 0.1)

    def test_owned_release_timeout_is_bounded(self):
        with patch.object(retry.base, "require_auto_power_profile"), patch.object(
            Path, "read_text", return_value=str(14 * 1024 ** 3)
        ), patch.object(retry.time, "monotonic", side_effect=[0, 10]), patch.object(
            retry.time, "sleep", side_effect=AssertionError("deadline passed")
        ):
            with self.assertRaisesRegex(ValueError, "10 seconds"):
                retry.wait_for_release()

    def test_completion_launches_only_two_missing_p2048_reports(self):
        inputs = {"artifact": {}, "baseline": "baseline", "candidate": "candidate"}
        launched = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole = root / "whole"
            whole.mkdir()
            (whole / "inputs.json").write_text(json.dumps(inputs))
            (whole / "candidate-p8192.json").write_text("retained8k")
            (root / "qualification").mkdir()
            (root / "qualification/result.json").write_text(json.dumps({
                "inputs": inputs, "pass": True, "static_complete": True, "numerical_complete": True,
            }))

            def identity(path):
                return "baseline" if path == retry.base.BASELINE else (
                    "candidate" if path == retry.base.CANDIDATE else str(path))

            def report(path, _artifact, _command, prompt, _chunk):
                seconds = 45 if path == retry.base.RETAINED else (7 if prompt == 8192 else 1)
                return {"tests": [{"reps": [{"timings": {"prefill_seconds": seconds}}] * 3}]}

            def run(command, _directory, name, **_kwargs):
                launched.append((name, command[command.index("-p") + 1]))
                Path(command[command.index("--output-file") + 1]).write_text("fresh2k")

            with patch.object(retry, "PACKAGE", root), patch.object(retry, "preflight", return_value=inputs), patch.object(
                retry.base, "identity", side_effect=identity
            ), patch.object(retry.base, "report", side_effect=report), patch.object(
                retry.base, "idle"
            ), patch.object(retry.base, "require_hip_pci_device"), patch.object(
                retry.base, "run_logged", side_effect=run
            ):
                retry.finish_whole(inputs)
            self.assertEqual(launched, [("baseline-p2048", "2048"), ("candidate-p2048", "2048")])
            self.assertEqual((whole / "candidate-p8192.json").read_text(), "retained8k")
            self.assertEqual(json.loads((whole / "result.json").read_text())["status"], "admitted")

    def inputs(self):
        inputs = json.loads((retry.RETAINED / "inputs.json").read_text())
        inputs["public_leaf"]["sha256"] = "corrected-leaf"
        inputs["sources"][retry.LEAF_SOURCE]["sha256"] = "corrected-fixture"
        return inputs

    def test_complete_retained_evidence_accepts_only_leaf_fixture_changes(self):
        result = retry.validate_retained(self.inputs())
        self.assertEqual(len(result), 6)
        self.assertIn("static.json", result)

    def test_changed_raw_or_bench_or_production_source_rejects(self):
        for key in ("qualifier", "candidate", "assembly", "planner"):
            with self.subTest(key=key):
                inputs = self.inputs()
                inputs[key]["sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "beyond the public-leaf fixture"):
                    retry.validate_retained(inputs)
        inputs = self.inputs()
        inputs["sources"]["src/ops/r9700/kv/fp8_int4_kv_attention.hip"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "beyond the public-leaf fixture"):
            retry.validate_retained(inputs)

    def test_missing_case_cannot_be_replaced_by_final_pass_line(self):
        inputs = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            retained = Path(directory) / "retained"
            shutil.copytree(retry.RETAINED, retained)
            log = retained / "dense-prefill-attention.stdout.txt"
            log.write_text("\n".join(log.read_text().splitlines()[1:]) + "\n")
            with patch.object(retry, "RETAINED", retained):
                with self.assertRaisesRegex(ValueError, "complete PASS"):
                    retry.validate_retained(inputs)

    def test_retry_launches_only_public_leaf(self):
        inputs = self.inputs()
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run(command, output, name, **_kwargs):
                commands.append(command)
                (output / f"{name}.stdout.txt").write_text("r9700_full_attention_leaf: PASS\n")

            inputs["retained_qualification"] = {}
            with patch.object(retry, "PACKAGE", root), patch.object(retry.base, "idle"), patch.object(
                retry.base, "require_hip_pci_device"
            ), patch.object(retry.base, "run_logged", side_effect=run), patch.object(
                retry, "preflight", return_value=inputs
            ):
                retry.qualification(inputs)
            self.assertEqual(commands, [[str(retry.base.LEAF)]])
            self.assertTrue(json.loads((root / "qualification/result.json").read_text())["pass"])


if __name__ == "__main__":
    unittest.main()
