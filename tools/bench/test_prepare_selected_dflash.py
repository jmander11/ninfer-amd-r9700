from __future__ import annotations

import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.bench import prepare_selected_dflash as prep


class PreparationTest(unittest.TestCase):
    def plan(self, root):
        return {"route": {"terminal_selection": {"path": "/base.json"},
            "base_artifact": {"path": "/base.ninfer", "weights_id": "r9700-q4g64-n16k16-eval"},
            "selected_prefill_chunk": 2048, "cache_group": 16,
            "text_prefill_attention_profile": "dense", "hybrid_base_authority": None},
            "build": {"benchmark": {"path": "/fresh/bench/ninfer_bench"}},
            "planner": {"tool": {"path": "/fresh/qualification/runtime_planner_qual"}},
            "conversion_python": {"path": "/explicit/python3.11", "environment": {
                "PYTHONPATH": "/cpu-packages", "PYTHONDONTWRITEBYTECODE": "1"}},
            "dflash_source": {"path": "/source"},
            "recipes": prep._recipe_paths(root, "r9700-q4g64-n16k16-eval")}

    def test_three_exact_cpu_recipe_conversions_have_separate_outputs(self):
        plan = self.plan(Path("/campaign"))
        outputs = []
        for recipe in prep.RECIPES:
            command = prep._companion_conversion_command(plan, recipe)
            self.assertIn("/explicit/python3.11", command)
            self.assertEqual(command[command.index("--device") + 1], "cpu")
            self.assertEqual(command[command.index("--matrix-recipe") + 1], recipe)
            self.assertEqual(command[command.index("--base") + 1], "/base.ninfer")
            outputs.append(command[command.index("--out") + 1])
        self.assertEqual(len(set(outputs)), 3)

    def test_subset_command_declares_capacity_decision_and_exact_width(self):
        plan = self.plan(Path("/campaign"))
        command = prep._common(plan, prep.RECIPES[2], "dflash-pareto", Path("/out"), 5, 6, [1, 3])
        self.assertEqual([command[i+1] for i, value in enumerate(command) if value == "--concurrency"], ["1", "3"])
        self.assertEqual(command[command.index("--prefill-chunk")+1], "2048")
        with self.assertRaises(ValueError):
            prep._common(plan, prep.RECIPES[0], "dflash-pareto", Path("/out"), 4, 6, [1])
        with self.assertRaises(ValueError):
            prep._common(plan, prep.RECIPES[0], "dflash-pareto", Path("/out"), 4, 5, [2])

    def test_terminal_gate_precedes_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            with patch.object(prep, "selected_base_route", side_effect=ValueError("nonterminal")):
                with self.assertRaisesRegex(ValueError, "nonterminal"):
                    prep.prepare(Path(directory), root, bench=Path("bench"), planner=Path("planner"))
            self.assertFalse(root.exists())

    def test_same_selected_base_evaluator_plan_roundtrips_and_revalidates_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "campaign"
            selected = parent / "base.json"
            selected.write_text("{}")
            fixture = self.plan(root)
            route = {**fixture["route"], "base_benchmark": {"sha256": "fresh"}}
            build = {"benchmark": {"path": "/fresh/bench/ninfer_bench", "sha256": "fresh"}}
            with patch.object(prep, "selected_base_route", return_value=route), \
                 patch.object(prep, "benchmark_profile", return_value=build), \
                 patch.object(prep, "_planner", return_value=fixture["planner"]), \
                 patch.object(prep, "_source_identity", return_value=fixture["dflash_source"]), \
                 patch.object(prep, "_conversion_python_identity", return_value=fixture["conversion_python"]):
                plan = prep.prepare(selected, root, bench=Path("bench"), planner=Path("planner"))
                self.assertEqual(plan["build"]["benchmark"]["sha256"],
                                 plan["route"]["base_benchmark"]["sha256"])
                self.assertEqual(prep._load_plan(root / "plan.json"), (plan, root))
                self.assertEqual(json.loads((root / "plan.json").read_text()), plan)
                self.assertEqual(len(list(root.glob("*/companion.ninfer"))), 0)
                self.assertEqual((root / "commands.sh").read_text().count("--device cpu"), 3)
                with patch.object(prep, "benchmark_profile", return_value={"benchmark": {"sha256": "changed"}}):
                    with self.assertRaises((ValueError, KeyError)):
                        prep._load_plan(root / "plan.json")

    def test_script_publication_never_overwrites_changed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "commands.sh"
            prep._write_script(output, ["true"])
            prep._write_script(output, ["true"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prep._write_script(output, ["false"])
            self.assertIn("true", output.read_text())

    def test_existing_failed_matrix_is_preserved_without_running_or_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "failed"
            output.mkdir()
            retained = output / "report.json"
            retained.write_text("failed retained evidence")
            (output / "failures.json").write_text("[]")
            marker = root / "invoked"
            script = "{ " + prep._matrix_shell(["touch", str(marker)], output) + "; } || test -f " + str(output / "failures.json")
            completed = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertEqual(retained.read_text(), "failed retained evidence")

    def test_advancement_uses_declared_subset_not_successful_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            cells = [(recipe, k, w, {"eligible_concurrency": [1, 3]})
                     for recipe in prep.RECIPES for k, w in prep.DFLASH_PRODUCTION_PROFILES]
            with patch.object(prep, "_load_plan", return_value=(plan, root)), \
                 patch.object(prep, "_evidence", return_value={recipe: {} for recipe in prep.RECIPES}), \
                 patch.object(prep, "_cells", return_value=cells), \
                 patch.object(prep, "performance_cells", return_value={"matched_speed_by_concurrency": {"1": {"pass": True}}}):
                prep.advance_pareto(root / "plan.json")
            script = (root / "pareto-and-select.sh").read_text()
            self.assertIn("--concurrency 1 --concurrency 3", script)
            self.assertNotIn("--concurrency 2", script)
            self.assertEqual(script.count("--pareto "), 6)
            self.assertEqual(script.count("--capacity "), 6)

    def test_no_k4_material_win_cannot_schedule_full_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            cells = [(recipe, k, w, {"eligible_concurrency": [1]})
                     for recipe in prep.RECIPES for k, w in prep.DFLASH_PRODUCTION_PROFILES]
            def performance(route, evidence, k, *args):
                return {"matched_speed_by_concurrency": {"1": {"pass": k == 5}}}
            with patch.object(prep, "_load_plan", return_value=(plan, root)), \
                 patch.object(prep, "_evidence", return_value={recipe: {} for recipe in prep.RECIPES}), \
                 patch.object(prep, "_cells", return_value=cells), \
                 patch.object(prep, "performance_cells", side_effect=performance):
                prep.advance_pareto(root / "plan.json")
            self.assertNotIn("--pareto ", (root / "pareto-and-select.sh").read_text())


if __name__ == "__main__":
    unittest.main()
