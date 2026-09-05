#!/usr/bin/env python3

import unittest
from pathlib import Path
import subprocess
import sys

from tools.bench.prepare_selected_dflash import _common, _hybrid_base_authority, _matrix_shell

REPO = Path(__file__).resolve().parents[2]


class SelectedDflashPrepareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "selected_prefill_chunk": 2048,
            "benchmark": {"path": "/build/bench/ninfer_bench"},
            "companion": {"path": "/out/companion.ninfer"},
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
                   "recipe_id": "recipe", "selection_sha256": "b" * 64,
                   "object_plan_sha256": "c" * 64, "source_index_sha256": "d" * 64,
                   "source_ranking_sha256": "e" * 64}
        self.assertEqual(_hybrid_base_authority({
            "weights_id": "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
            "conversion_receipt": receipt,
        }), {"receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
             **{key: receipt[key] for key in receipt if key not in ("path", "sha256")}})

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
