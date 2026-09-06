#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from tools.bench import prepare_selected_dflash as prepare_module
from tools.bench.prepare_selected_dflash import (
    _base_migration_authority,
    _common,
    _companion_conversion_command,
    _matrix_shell,
)

REPO = Path(__file__).resolve().parents[2]


class SelectedDflashPrepareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "selected_prefill_chunk": 2048,
            "benchmark": {"path": "/build/bench/ninfer_bench"},
            "companion": {"path": "/out/companion.ninfer"},
            "base_artifact": {"path": "/out/selected-base.ninfer"},
            "dflash_source": {"path": "/models/qwen3.8-27b-dflash2"},
            "conversion_python": {
                "launcher_path": "/venv/bin/python",
                "environment": {
                    "LD_LIBRARY_PATH": "/opt/rocm/lib:/opt/rocm/core-10.0/lib",
                },
            },
            "cache_group": 32,
            "text_prefill_attention_profile": "b128-s16-tau900",
            "recipe": "all-q4",
            "hybrid_width_tool": None,
        }

    def test_shortlist_is_exact_c1_and_does_not_invent_kw(self) -> None:
        command = _common(self.plan, "dflash-shortlist", Path("/fresh/shortlist"))
        self.assertNotIn("--dflash-draft-tokens", command)
        self.assertNotIn("--concurrency", command)
        self.assertEqual(command[command.index("--prefill-chunk") + 1], "2048")

    def test_frontier_command_is_capped_at_exact_c1_through_c4(self) -> None:
        command = _common(self.plan, "dflash-capacity", Path("/fresh/capacity"), 7, 12)
        values = [command[index + 1] for index, item in enumerate(command)
                  if item == "--concurrency"]
        self.assertEqual(values, ["1", "2", "3", "4"])
        self.assertEqual(command[command.index("--dflash-draft-tokens") + 1], "7")
        self.assertEqual(command[command.index("--dflash-verify-width") + 1], "12")

    def test_four_role_campaign_requires_and_binds_hybrid_planner(self) -> None:
        self.plan["recipe"] = "four-role"
        self.plan["hybrid_width_tool"] = {"path": "/build/tools/planner"}
        command = _common(self.plan, "dflash-shortlist", Path("/fresh/shortlist"))
        self.assertEqual(command.count("--require-fp8-hybrid"), 1)
        self.assertEqual(
            command[command.index("--hybrid-width-tool") + 1], "/build/tools/planner"
        )

    def test_hybrid_base_receipt_is_normalized_to_converter_authority(self) -> None:
        receipt = {"path": "/out/base.conversion.json", "sha256": "a" * 64,
                   "recipe_id": "r9700-q4g64-f8e4m3-four-role-n16k16-eval-v1",
                   "selection_sha256":
                       "b2ceeb63c581c0f26aab5a4d8c0958da34d836fcc5c47d377bce709eaf37e3e8",
                   "object_plan_sha256": "c" * 64, "source_index_sha256": "d" * 64,
                   "source_ranking_sha256": "e" * 64,
                   "source_artifact_sha256": "1" * 64,
                   "source_receipt_sha256": "2" * 64, "transcoder_sha256": "3" * 64}
        self.assertEqual(_base_migration_authority({
            "weights_id": "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
            "conversion_receipt": receipt,
        }), {"receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
             **{key: receipt[key] for key in receipt if key not in ("path", "sha256")}})

    def test_missing_companion_conversion_is_selected_only_for_every_recipe(self) -> None:
        cases = (
            ("all-q4", "/out/all-q4-n16k16.ninfer",
             "/out/all-q4-n16k16-dflash2.ninfer"),
            ("mixed", "/out/mixed-n16k16.ninfer",
             "/out/mixed-n16k16-dflash2.ninfer"),
            ("four-role", "/out/four-role-n16k16.ninfer",
             "/out/four-role-n16k16-dflash2.ninfer"),
        )
        for recipe, base, companion in cases:
            with self.subTest(recipe=recipe):
                self.plan["recipe"] = recipe
                self.plan["base_artifact"]["path"] = base
                self.plan["companion"]["path"] = companion
                command = _companion_conversion_command(
                    self.plan, companion_exists=False
                )
                self.assertEqual(command, [
                    "env", "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib",
                    "/venv/bin/python", "-m",
                    "tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4",
                    "--base", base,
                    "--dflash-model", "/models/qwen3.8-27b-dflash2",
                    "--out", companion,
                    "--device", "cuda",
                ])

    def test_existing_selected_companion_only_finalizes_its_missing_report(self) -> None:
        command = _companion_conversion_command(self.plan, companion_exists=True)
        self.assertIn("--finalize-report", command)
        self.assertNotIn("--device", command)

    def test_conversion_python_identity_fails_closed_and_binds_rocm_environment(self) -> None:
        with patch.object(prepare_module, "CONVERSION_PYTHON", Path("/missing/python")), \
                self.assertRaisesRegex(ValueError, "Python is unavailable"):
            prepare_module._conversion_python_identity()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "bin/python"
            launcher.parent.mkdir()
            launcher.write_text("python", encoding="utf-8")
            launcher.chmod(0o755)
            pyvenv = root / "pyvenv.cfg"
            pyvenv.write_text("venv", encoding="utf-8")
            runtime = {
                "runtime_executable": str(launcher), "python_version": "3.12.0",
                "torch": "2.9.1", "torch_hip": "7.2", "safetensors": "0.8.0",
            }
            completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(runtime), stderr="")
            with patch.object(prepare_module, "CONVERSION_PYTHON", launcher), patch.object(
                prepare_module.subprocess, "run", return_value=completed
            ) as run:
                identity = prepare_module._conversion_python_identity()
            self.assertEqual(identity["environment"], {
                "LD_LIBRARY_PATH": "/opt/rocm/lib:/opt/rocm/core-10.0/lib",
            })
            self.assertEqual(identity["pyvenv_cfg"]["path"], str(pyvenv))
            self.assertEqual(
                run.call_args.kwargs["env"]["LD_LIBRARY_PATH"],
                "/opt/rocm/lib:/opt/rocm/core-10.0/lib",
            )
            wrong_runtime = {**runtime, "runtime_executable": "/different/python"}
            completed.stdout = json.dumps(wrong_runtime)
            with patch.object(prepare_module, "CONVERSION_PYTHON", launcher), patch.object(
                prepare_module.subprocess, "run", return_value=completed
            ), self.assertRaisesRegex(ValueError, "malformed identity"):
                prepare_module._conversion_python_identity()

    def test_matrix_shell_resumes_only_an_existing_owned_output(self) -> None:
        command = _common(self.plan, "dflash-shortlist", Path("/fresh/shortlist"))
        shell = _matrix_shell(command, Path("/fresh/shortlist"))
        self.assertIn("[[ -L /fresh/shortlist ]]", shell)
        self.assertIn("elif [[ -e /fresh/shortlist ]]", shell)
        self.assertEqual(shell.count("--resume"), 1)
        self.assertIn("else", shell)

    def test_prepared_self_commands_and_cli_use_repo_module(self) -> None:
        source = (REPO / "tools/bench/prepare_selected_dflash.py").read_text(encoding="utf-8")
        package_prepare = (
            REPO / "profiles/bench/selected-dflash-prepare-20260905/prepare.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m tools.bench.prepare_selected_dflash prepare", package_prepare)
        for command in ("validate-plan", "advance-capacity", "advance-pareto"):
            self.assertIn(f'"tools.bench.prepare_selected_dflash", "{command}"', source)
        result = subprocess.run(
            [sys.executable, "-m", "tools.bench.prepare_selected_dflash", "--help"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
