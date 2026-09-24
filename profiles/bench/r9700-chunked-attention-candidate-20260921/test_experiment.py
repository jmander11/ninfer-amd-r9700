import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("attention_experiment", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def measurements(candidate8=(3, 4, 5), candidate2=(1, 1, 1)):
    return {name: {"reps": [{"timings": {"prefill_seconds": value}} for value in values]}
            for name, values in {
                "candidate-p8192": candidate8, "retained-baseline-p8192": (44, 45, 46),
                "candidate-p2048": candidate2, "baseline-p2048": (0.9, 1, 1.1),
            }.items()}


class ExperimentTest(unittest.TestCase):
    def test_owned_completed_job_does_not_recheck_stale_busy_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mem_info_vram_used").write_text("1024")
            (root / "gpu_busy_percent").write_text("100")
            with patch.object(experiment, "R9700_POWER_PROFILE", root / "power"), patch.object(
                experiment, "require_auto_power_profile"
            ):
                experiment.previous_job_released()
                with self.assertRaisesRegex(ValueError, "must be idle"):
                    experiment.idle()

    def test_command_is_matched_unprofiled_three_repetition_prefill(self):
        command = experiment.bench_command(Path("candidate"), 8192, 1024, Path("result.json"))
        for flag, value in (("-p", "8192"), ("--prefill-chunk", "1024"),
                            ("--draft-tokens", "0"), ("--concurrency", "1"),
                            ("-r", "3"), ("--warmup", "1")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--profile-measured", command)

    def test_retained_baseline_passes_real_report_contract(self):
        frozen = json.loads(experiment.FROZEN.read_text())
        artifact = next(item for item in frozen["artifacts"] if item["path"] == str(experiment.WEIGHTS))
        report = experiment.report(experiment.RETAINED, artifact, experiment.bench_command(
            experiment.BASELINE, 8192, 1024, experiment.RETAINED), 8192, 1024)
        self.assertEqual(report["tests"][0]["prefill_tok_s_mean"], 184.8640144)
        self.assertEqual(len(report["tests"][0]["reps"]), 3)

    def test_admission_uses_medians_and_both_thresholds(self):
        self.assertEqual(experiment.admission(measurements())["status"], "admitted")
        self.assertEqual(experiment.admission(measurements(candidate8=(40, 41, 42)))["status"], "not_admitted")
        self.assertEqual(experiment.admission(measurements(candidate2=(1.02, 1.03, 1.04)))["status"], "not_admitted")
        self.assertEqual(experiment.admission(measurements(candidate8=(1, 4, 100)))["median_prefill_seconds"]["candidate-p8192"], 4)

    def test_bad_measurement_cannot_publish_admission(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            experiment.admission(measurements(candidate8=(1, float("nan"), 2)))

    def test_changed_inputs_stop_whole_before_gpu_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qualification").mkdir()
            (root / "qualification/result.json").write_text(json.dumps({
                "pass": True, "static_complete": True, "numerical_complete": True,
                "inputs": {"candidate": "old"},
            }))
            with patch.object(experiment, "PACKAGE", root), patch.object(
                experiment, "idle", side_effect=AssertionError("must not reach GPU")
            ):
                with self.assertRaisesRegex(ValueError, "exact inputs"):
                    experiment.whole({"candidate": "new"})
            self.assertFalse((root / "whole").exists())

    def test_static_failure_stops_before_qualification_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(experiment, "PACKAGE", root), patch.object(
                experiment, "static_checks", side_effect=ValueError("bad ISA")
            ), patch.object(experiment, "idle", side_effect=AssertionError("must not reach GPU")):
                with self.assertRaisesRegex(ValueError, "bad ISA"):
                    experiment.qualification({})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
