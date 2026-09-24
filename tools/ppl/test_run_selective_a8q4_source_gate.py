from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.ppl import run_selective_a8q4_source_gate as runner


class RunSelectiveA8Q4SourceGateTest(unittest.TestCase):
    def _paths(self, root: Path):
        output = root / "result"
        arms = {
            "a8g64-q4g64-control": output / "control.json",
            "a8g128-q4g128-mse": output / "candidate.json",
        }
        return output, arms, output / "comparison.json"

    @staticmethod
    def _triplet(path: Path) -> None:
        path.write_text("{}")
        path.with_suffix(".nllf32").write_bytes(b"n")
        path.with_suffix(".argmaxi32").write_bytes(b"a")

    def test_partial_and_foreign_evidence_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, arms, comparison = self._paths(Path(temporary))
            output.mkdir()
            arms["a8g64-q4g64-control"].write_text("partial")
            with patch.object(runner, "OUTPUT", output), patch.object(
                runner, "ARMS", arms), patch.object(runner, "COMPARISON", comparison
            ):
                with self.assertRaisesRegex(ValueError, "partial"):
                    runner.run()
            arms["a8g64-q4g64-control"].unlink()
            (output / "foreign").write_text("x")
            with patch.object(runner, "OUTPUT", output), patch.object(
                runner, "ARMS", arms), patch.object(runner, "COMPARISON", comparison
            ):
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    runner.run()

    def test_resume_runs_only_missing_candidate_then_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, arms, comparison = self._paths(Path(temporary))
            output.mkdir()
            self._triplet(arms["a8g64-q4g64-control"])
            commands = []

            def execute(command, **kwargs):
                commands.append(command)
                if any(item.endswith("selective_a8q4_source_diagnostic") for item in command):
                    self._triplet(arms["a8g128-q4g128-mse"])
                else:
                    comparison.write_text("{}")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(runner, "OUTPUT", output), patch.object(
                runner, "ARMS", arms), patch.object(runner, "COMPARISON", comparison
            ), patch.object(runner, "_load_score"), patch.object(
                runner, "validate_comparison", return_value={"pass": True}
            ), patch.object(runner.subprocess, "run", side_effect=execute):
                self.assertEqual(runner.run(), 0)
            self.assertEqual(len(commands), 2)
            self.assertIn("a8g128-q4g128-mse", commands[0])
            self.assertNotIn("a8g64-q4g64-control", commands[0])


if __name__ == "__main__":
    unittest.main()
