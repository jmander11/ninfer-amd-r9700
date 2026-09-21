#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from tools.ppl.prepare_selected_exact_token import REPO, prepare, selected_quality, sha
from tools.ppl.run import prepare_output_directory


class PrepareSelectedExactTokenTest(unittest.TestCase):
    def test_prepare_rejects_dangling_output_before_route_resolution(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "profiles/ppl") as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text("{}\n", encoding="utf-8")
            output = root / "prepared"
            output.symlink_to(root / "missing")
            with patch("tools.ppl.prepare_selected_exact_token.resolve_route") as resolver, \
                    self.assertRaisesRegex(ValueError, "already exists"):
                prepare(selection, output)
            resolver.assert_not_called()
            self.assertTrue(output.is_symlink())

    def test_ppl_runner_rejects_real_and_dangling_output_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            for target in (real, root / "missing"):
                output = root / ("linked" if target == real else "dangling")
                output.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(SystemExit, "symlinked PPL output"):
                    prepare_output_directory(output)
                self.assertTrue(output.is_symlink())

    def test_quality_requires_one_cell_at_each_exact_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quality = root / "quality.json"
            cells = [{"scheme": "r9700-g16", "schedule": "prefill",
                      "prompt_tokens": 8192, "quality_tier": "accuracy",
                      "quality_eligible": True, "pass": True} for _ in range(2)]
            quality.write_text(json.dumps({
                "artifact_type": "ninfer_r9700_ppl_campaign", "schema_version": 6,
                "pass": True, "prefill_chunk": 2048, "xattention_profile": "dense",
                "candidate_artifact": {"weights_id": "weights", "sha256": "a" * 64},
                "cells": cells,
            }), encoding="utf-8")
            authority = {"source_provenance": [{"candidate": "winner",
                          "quality": {"path": str(quality), "sha256": sha(quality)}}]}
            route = {"selected_prefill_chunk": 2048,
                     "execution_profile": {"xattention_profile": "dense"},
                     "artifact": {"weights_id": "weights", "sha256": "a" * 64}}
            selection = root / "selection.json"
            selection.write_text("{}", encoding="utf-8")
            with patch("tools.ppl.prepare_selected_exact_token.load_payload",
                       return_value=authority), patch(
                "tools.ppl.prepare_selected_exact_token.validate_terminal_production_authority"
            ), self.assertRaisesRegex(ValueError, "exact admitted"):
                selected_quality(selection, "winner", 16, route)

    def test_prepares_only_selected_c1_decode_route(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO / "profiles/ppl") as directory:
            root = Path(directory)
            selection = root / "selection.json"
            artifact = root / "selected.ninfer"
            build = root / "build"
            scorer = build / "apps/ninfer-ppl"
            quality = root / "quality.json"
            for path in (selection, artifact, scorer, quality):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")
            reference = root / "reference.json"
            repeat = root / "repeat.json"
            reference.write_text("{}")
            repeat.write_text("{}")
            quality.write_text(json.dumps({
                "reused_bf16_campaign": {"path": str(reference), "sha256": sha(reference)},
                "bf16_repeat_comparison": {"path": str(repeat), "sha256": sha(repeat)},
            }))
            route = {
                "winner": "selected-g32-dense",
                "artifact": {"path": str(artifact), "weights_id": "r9700-q4g64-n16k16-eval",
                             "sha256": "a" * 64},
                "build_directory": str(build), "cache_profile": {"value_group": 32},
                "execution_profile": {"xattention_profile": "dense"},
                "selected_prefill_chunk": 2048, "hybrid_width_tool": None,
            }
            output = root / "prepared"
            shards = {f"tensor{i}": f"model-{i:05d}-of-00018.safetensors" for i in range(1, 19)}
            with patch("tools.ppl.prepare_selected_exact_token.validate_bf16_repeat_comparison"), patch("tools.ppl.prepare_selected_exact_token.resolve_route", return_value=route), patch(
                "tools.ppl.prepare_selected_exact_token.selected_quality",
                return_value=({"path": str(quality), "sha256": "b" * 64},
                              {"tier": "capacity-speed", "profile": "r9700-g32"}),
            ), patch("tools.ppl.prepare_selected_exact_token.inspect_executable",
                     return_value={"path": str(scorer), "file_size_bytes": 1,
                                   "sha256": "c" * 64}), patch(
                "tools.ppl.prepare_selected_exact_token.validate_checkpoint_files",
                return_value=shards,
            ):
                plan = prepare(selection, output)
            self.assertEqual(plan["workload"]["concurrency"], 1)
            self.assertEqual(plan["workload"]["lengths"], [8192, 32768])
            self.assertEqual(plan["candidate_profile"], "r9700-g32")
            self.assertIn("--execution-parity-max-abs-nll", plan["command"])
            self.assertEqual(plan["command"].count("--no-position-extras"), 1)
            self.assertEqual(plan["command"][plan["command"].index("--reuse-bf16-campaign") + 1], str(reference))
            self.assertNotIn("--require-fp8-hybrid", plan["command"])
            self.assertTrue((output / "prepared.sha256").is_file())
            commands = (output / "commands.sh").read_text(encoding="utf-8")
            self.assertIn("-m tools.ppl.validate_selected_exact_token", commands)
            self.assertEqual(plan["python"]["launcher_path"],
                             "/ssdpool2nvme/local_llm/.venv-ninfer-r9700-py311/bin/python")
            self.assertIn("export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib", commands)
            self.assertIn("unset PYTHONPATH PYTHONHOME", commands)
            self.assertLess(commands.index("unset PYTHONPATH PYTHONHOME"),
                            commands.index(plan["python"]["launcher_path"]))
            self.assertIn("/sys/bus/pci/devices/0000:13:00.0/", commands)
            self.assertNotIn(str(REPO / "tools/ppl/validate_selected_exact_token.py"), commands)

    def test_selected_group_admission_not_sibling_campaign_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quality = root / "quality.json"
            selection = root / "selection.json"
            selection.write_text("{}")
            route = {"selected_prefill_chunk": 2048,
                     "execution_profile": {"xattention_profile": "dense"},
                     "artifact": {"weights_id": "weights", "sha256": "a" * 64}}
            campaign = {"artifact_type": "ninfer_r9700_ppl_campaign", "schema_version": 6,
                        "pass": False, "prefill_chunk": 2048, "xattention_profile": "dense",
                        "candidate_artifact": route["artifact"], "cells": [
                            {"scheme": f"r9700-g{group}", "schedule": "prefill",
                             "prompt_tokens": length, "quality_tier": "accuracy",
                             "quality_eligible": group == 16, "pass": group == 16}
                            for group in (16, 32) for length in (8192, 32768)]}
            quality.write_text(json.dumps(campaign))
            authority = {"source_provenance": [{"candidate": "winner", "quality": {
                "path": str(quality), "sha256": sha(quality)}}]}
            # Terminal validation owns raw numerical replay; this test isolates the
            # selected-group join after that validation has succeeded.
            with patch("tools.ppl.prepare_selected_exact_token.load_payload", return_value=authority), \
                    patch("tools.ppl.prepare_selected_exact_token.validate_terminal_production_authority"):
                _, selected = selected_quality(selection, "winner", 16, route)
                self.assertEqual(selected["profile"], "r9700-g16")
                with self.assertRaisesRegex(ValueError, "exact admitted"):
                    selected_quality(selection, "winner", 32, route)


if __name__ == "__main__":
    unittest.main()
