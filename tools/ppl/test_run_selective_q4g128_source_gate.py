from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.ppl import run_selective_q4g128_source_gate as runner


class RunSelectiveQ4G128SourceGateTest(unittest.TestCase):
    def _paths(self, root: Path):
        output = root / "result"
        arms = {
            "q4g64-absmax": output / "q4g64-control.json",
            "q4g128-mse": output / "q4g128-candidate.json",
        }
        return output, arms, output / "comparison.json"

    @staticmethod
    def _write_triplet(path: Path) -> None:
        path.write_text("{}")
        path.with_suffix(".nllf32").write_bytes(b"nll")
        path.with_suffix(".argmaxi32").write_bytes(b"arg")

    def test_partial_triplet_and_foreign_entry_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, arms, comparison = self._paths(Path(temporary))
            output.mkdir()
            arms["q4g64-absmax"].write_text("partial")
            with patch.object(runner, "OUTPUT", output), patch.object(
                runner, "ARMS", arms), patch.object(runner, "COMPARISON", comparison
            ):
                with self.assertRaisesRegex(ValueError, "partial"):
                    runner.run()
            arms["q4g64-absmax"].unlink()
            (output / "foreign").write_text("x")
            with patch.object(runner, "OUTPUT", output), patch.object(
                runner, "ARMS", arms), patch.object(runner, "COMPARISON", comparison
            ):
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    runner.run()

    def test_resume_revalidates_complete_arm_and_runs_only_missing_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, arms, comparison = self._paths(Path(temporary))
            output.mkdir()
            self._write_triplet(arms["q4g64-absmax"])
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                if any(item.endswith("selective_q4g128_source_diagnostic") for item in command):
                    self._write_triplet(arms["q4g128-mse"])
                    return subprocess.CompletedProcess(command, 0)
                comparison.write_text("{}")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(runner, "OUTPUT", output), patch.object(
                runner, "ARMS", arms), patch.object(runner, "COMPARISON", comparison
            ), patch.object(runner, "_load_score"), patch.object(
                runner, "validate_comparison", return_value={"pass": True}
            ), patch.object(runner.subprocess, "run", side_effect=fake_run):
                self.assertEqual(runner.run(), 0)
            self.assertEqual(len(commands), 2)
            self.assertIn("q4g128-mse", commands[0])
            self.assertNotIn("q4g64-absmax", commands[0])
            self.assertIn("tools.ppl.compare_selective_q4g128_source", commands[1])

    def test_existing_comparison_is_revalidated_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, arms, comparison = self._paths(Path(temporary))
            output.mkdir()
            for path in arms.values():
                self._write_triplet(path)
            comparison.write_text("{}")
            with patch.object(runner, "OUTPUT", output), patch.object(
                runner, "ARMS", arms), patch.object(runner, "COMPARISON", comparison
            ), patch.object(runner, "_load_score"), patch.object(
                runner, "validate_comparison", return_value={"pass": False}
            ), patch.object(runner.subprocess, "run") as execute:
                self.assertEqual(runner.run(), 1)
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
