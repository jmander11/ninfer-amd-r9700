"""Focused campaign dependency and invocation regression checks; no GPU work."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("quality_campaign", Path(__file__).with_name("campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTest(unittest.TestCase):
    def test_changed_frozen_inputs_reject_before_scoring(self):
        for change in ("scorer", "artifact", "selection-artifact", "selection-benchmark"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scorer = root / "build/apps/ninfer-ppl"
                scorer.parent.mkdir(parents=True)
                scorer.write_bytes(b"original scorer")
                scorer.chmod(0o755)
                weights = root / "out/qwen3.8-27b-recipe.ninfer"
                weights.parent.mkdir()
                weights.write_bytes(b"original weights")
                artifact = {"path": str(weights), "weights_id": "recipe", "bytes": weights.stat().st_size,
                            "sha256": campaign.run.file_sha256(weights)}
                selected_artifact = dict(artifact)
                selected_artifact["file_size_bytes"] = selected_artifact.pop("bytes")
                bench_relative = "build-r9700-selection-panel-dense-g16-20260921/bench/ninfer_bench"
                record = {"sources": [{"weights_id": "recipe", "kv_value_group": 16,
                          "xattention_profile": "dense", "artifact": selected_artifact,
                          "benchmark_executable": {"path": str(root / bench_relative), "sha256": "bench"}}]}
                receipt = root / "inputs.json"
                receipt.write_text(json.dumps({
                    "artifact_type": "ninfer_r9700_chunk_campaign_inputs", "schema_version": 1,
                    "artifacts": [artifact], "files": {
                        str(scorer.relative_to(root)): campaign.run.file_sha256(scorer),
                        bench_relative: "bench"},
                }))

                def inspect(_path):
                    return {**artifact, "bytes": weights.stat().st_size,
                            "sha256": campaign.run.file_sha256(weights)}

                with (
                    patch.object(campaign, "REPO", root),
                    patch.object(campaign, "FROZEN_INPUTS", receipt),
                    patch.object(campaign, "EXPECTED_AUTHORITIES", {"TEST": ("recipe", "dense")}),
                    patch.object(campaign, "campaigns", return_value=[
                        ("TEST", "recipe", "dense", {16: scorer}, root / "output")]),
                    patch.object(campaign.run, "inspect_candidate_artifact", side_effect=inspect),
                    patch.object(campaign, "validate_prefill_chunk_authority", return_value=(
                        {"selected_prefill_chunk": 4096}, record)),
                    patch.object(campaign.subprocess, "run", side_effect=AssertionError("must not score")),
                ):
                    campaign.verify_frozen_inputs(record)
                    if change == "scorer":
                        scorer.write_bytes(b"replaced scorer")
                    elif change == "artifact":
                        weights.write_bytes(b"replaced weights")
                    elif change == "selection-artifact":
                        record["sources"][0]["artifact"]["sha256"] = "different"
                    else:
                        record["sources"][0]["benchmark_executable"]["sha256"] = "different"
                    with self.assertRaisesRegex(ValueError, "differs from frozen"):
                        campaign.preflight()
                    self.assertFalse((root / "output").exists())

    def test_missing_selection_preflight_creates_nothing_and_launches_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(campaign, "SELECTION", root / "missing.json"), patch.object(
                campaign.subprocess, "run", side_effect=AssertionError("must not launch")
            ):
                with self.assertRaisesRegex(ValueError, "cannot be opened"):
                    campaign.preflight()
            self.assertEqual(list(root.iterdir()), [])

    def test_six_campaigns_cover_recipes_routes_and_both_groups(self):
        cases = list(campaign.campaigns())
        self.assertEqual(len(cases), 6)
        self.assertEqual(len({output for *_, output in cases}), 6)
        for _, weights_id, profile, binaries, output in cases:
            command = campaign.quality_command(4096, weights_id, profile, binaries, output)
            self.assertEqual(command[command.index("--profiles") + 1],
                             "bf16-reference,r9700-g16,r9700-g32")
            self.assertEqual(command[command.index("--expected-xattention-profile") + 1], profile)
            self.assertNotIn("--tokens", command)
            self.assertNotIn("--long", command)
            for group in (16, 32):
                binary = command[command.index(f"--g{group}-ppl-bin") + 1]
                route = "dense" if profile == "dense" else "xattention"
                self.assertIn(f"build-r9700-selection-panel-{route}-g{group}-20260921/", binary)
            self.assertEqual("--require-fp8-hybrid" in command, "four-role" in weights_id)

    def test_chunk4096_reference_reuses_without_a_python_dependency(self):
        with patch.object(campaign, "validate_reference") as validate, patch.object(
            campaign.subprocess, "run", side_effect=AssertionError("must not launch")
        ):
            campaign.reference({"selected_prefill_chunk": 4096}, None)
        validate.assert_called_once_with(4096)

    def test_other_chunk_requires_explicit_reference_environment(self):
        with patch.object(campaign.subprocess, "run") as execute:
            with self.assertRaisesRegex(ValueError, "Python 3.11 ROCm"):
                campaign.reference({"selected_prefill_chunk": 2048}, None)
        execute.assert_not_called()

    def test_bad_reference_environment_stops_before_output_or_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(campaign, "PACKAGE", root), patch.object(
                campaign.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "probe")
            ) as execute:
                with self.assertRaises(subprocess.CalledProcessError):
                    campaign.reference({"selected_prefill_chunk": 2048}, Path("python3.12"))
            execute.assert_called_once()
            self.assertEqual(execute.call_args.args[0][1], "-c")
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_fresh_reference_blocks_candidate_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(campaign, "PACKAGE", root), patch.object(
                campaign.subprocess, "run", side_effect=AssertionError("must not launch")
            ):
                with self.assertRaises(FileNotFoundError):
                    campaign.quality({"selected_prefill_chunk": 2048})
            self.assertEqual(list(root.iterdir()), [])

    def test_fresh_reference_requires_matching_passing_probe_before_scoring(self):
        for failure in (None, "oracle", "geometry", "interpreter", "interpreter-hash", "existing"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                python = root / "python3.11"
                python.write_bytes(b"reference interpreter")
                probe = root / "bf16-chunk2048-gdn-full-span.json"
                if failure == "existing":
                    probe.write_text("previous failed probe")
                stages = []

                def execute(command, **kwargs):
                    if command[1] == "-c":
                        stages.append("environment")
                    elif command[1].endswith("gdn_full_span_probe.py"):
                        stages.append("probe")
                        report = {
                            "all_pass": failure != "oracle",
                            "geometry": {"row_extents": [4095] if failure == "geometry" else [4095, 4096]},
                            "provenance": {"execution": {
                                "python_executable": str(root / "other") if failure == "interpreter" else str(python),
                                "python_executable_sha256": "changed" if failure == "interpreter-hash" else campaign.run.file_sha256(python),
                            }},
                        }
                        self.assertEqual(command[0], str(python))
                        probe.write_text(json.dumps(report))
                    else:
                        self.assertTrue(probe.exists())
                        stages.append("compare" if command[1].endswith("compare_bf16_repeats.py") else "score")

                with patch.object(campaign, "PACKAGE", root), \
                        patch.object(campaign, "validate_checkpoint_files"), \
                        patch.object(campaign, "validate_reference"), \
                        patch.object(campaign, "unchanged"), \
                        patch.object(campaign.subprocess, "run", side_effect=execute):
                    if failure is None:
                        campaign.reference({"selected_prefill_chunk": 2048}, python)
                        self.assertEqual(stages, ["environment", "probe", "score", "score", "compare"])
                    else:
                        with self.assertRaises(ValueError):
                            campaign.reference({"selected_prefill_chunk": 2048}, python)
                        self.assertEqual(stages, ["environment"] if failure == "existing" else ["environment", "probe"])
                        self.assertTrue(probe.exists())
                        if failure == "existing":
                            self.assertEqual(probe.read_text(), "previous failed probe")


if __name__ == "__main__":
    unittest.main()
