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
