from __future__ import annotations

import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.bench import prepare_selected_dflash as prep
from tools.bench import run_ninfer_bench_matrix as matrix
from tools.bench import assemble_dflash_selection as selection


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
        followup = prep._common(plan, prep.RECIPES[0], "dflash-pareto", Path("/out"), 4, 5, [2])
        self.assertEqual(followup[followup.index("--concurrency") + 1], "2")

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

    def test_new_evaluator_owns_speculative_and_ordinary_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            old = str(root / "original-panel/bench/ninfer_bench")
            evaluator = str(root / "accounted/bench/ninfer_bench")
            plan["route"]["base_benchmark"] = {"path": old, "sha256": "original"}
            plan["build"]["benchmark"] = {"path": evaluator, "sha256": "recipe-aware"}
            for preset in ("dflash-shortlist", "dflash-pareto"):
                output = root / preset
                command = prep._common(plan, prep.RECIPES[0], preset, output,
                                       4, 5, [1])
                argv = command[command.index("tools.bench.run_ninfer_bench_matrix") + 1:]
                self.assertEqual(matrix.main([*argv, "--dry-run"]), 0)
                manifest = json.loads((output / "manifest.json").read_text())
                controls = [row for row in manifest["commands"] if "control" in row["suite"]]
                self.assertTrue(controls)
                for row in manifest["commands"]:
                    self.assertEqual(row["command"][0], evaluator)
                    self.assertNotIn(old, row["command"])
                for row in controls:
                    args = row["command"]
                    self.assertEqual(args[args.index("--draft-tokens") + 1], "0")
                    self.assertNotIn("--spec", args)
                    self.assertIn("--retain-token-ids", args)
            self.assertEqual(plan["route"]["base_benchmark"]["path"], old)

    def test_followup_dry_run_has_only_declared_non_c1_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "followup"
            command = prep._common(self.plan(root), prep.RECIPES[0], "dflash-pareto",
                                   output, 4, 5, [2, 3, 4])
            argv = command[command.index("tools.bench.run_ninfer_bench_matrix") + 1:]
            self.assertEqual(matrix.main([*argv, "--dry-run"]), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["concurrency"], [2, 3, 4])
            self.assertEqual(len(manifest["commands"]), 15)
            self.assertEqual({row["concurrency"] for row in manifest["commands"]}, {2, 3, 4})
            self.assertNotIn("dflash_selector_diagnostic", {row["suite"] for row in manifest["commands"]})
            # Replay the real runner's manifest/command corpus shape. Only raw GPU
            # result validation is replaced; no benchmark executable is launched.
            manifest["dry_run"] = False
            (output / "manifest.json").write_text(json.dumps(manifest))
            def rows(path, case, group, a4, a8, fp8, c, *args):
                if case.suite == "dflash_pareto_prefill":
                    return []
                phase = "whole_output" if "whole" in case.suite else "decode_output"
                mean = 100. if "control" in case.suite else 130.
                return [{"suite": case.suite, "label": str(prompt), "n_prompt": prompt,
                         "n_gen": 256, "concurrency": c, "requested_output_tokens": 257,
                         f"{phase}_tok_s_mean": mean, f"{phase}_tok_s_stddev": .1,
                         "spec_acceptance_length": 3.} for prompt in (8192, 32768)]
            with patch.object(selection, "_same_campaign"), \
                 patch.object(selection, "_records"), \
                 patch.object(selection, "_auxiliary", return_value={"parity": {}}), \
                 patch.object(selection, "report_rows", side_effect=rows):
                evidence = {"artifact": {}, "benchmark": {}}
                result = selection.performance_cells(self.plan(root)["route"], evidence,
                                                     4, 5, output, [2, 3, 4])
                self.assertEqual(result["corpus"]["sha256"], manifest["corpus_sha256"])
                self.assertEqual(result["declared_concurrency"], [2, 3, 4])
                self.assertTrue(all(row["pass"] for row in result["matched_speed_by_concurrency"].values()))
                manifest["corpus_sha256"] = "0" * 64
                (output / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "corpus differs"):
                    selection.performance_cells(self.plan(root)["route"], evidence,
                                                4, 5, output, [2, 3, 4])

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
            self.assertIn("--concurrency 3", script)
            self.assertNotIn("--concurrency 1", script)
            self.assertNotIn("--concurrency 2", script)
            self.assertEqual(script.count("--pareto "), 6)
            self.assertEqual(script.count("--capacity "), 6)

    def test_c1_only_capacity_reuses_screen_without_empty_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            cells = [(recipe, k, w, {"eligible_concurrency": [1]})
                     for recipe in prep.RECIPES for k, w in prep.DFLASH_PRODUCTION_PROFILES]
            with patch.object(prep, "_load_plan", return_value=(plan, root)), \
                 patch.object(prep, "_evidence", return_value={recipe: {} for recipe in prep.RECIPES}), \
                 patch.object(prep, "_cells", return_value=cells), \
                 patch.object(prep, "performance_cells", return_value={"matched_speed_by_concurrency": {"1": {"pass": True}}}):
                prep.advance_pareto(root / "plan.json")
            script = (root / "pareto-and-select.sh").read_text()
            self.assertNotIn("--pareto ", script)
            self.assertNotIn("--preset dflash-pareto", script)
            self.assertEqual(script.count("--c1 "), 6)

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
