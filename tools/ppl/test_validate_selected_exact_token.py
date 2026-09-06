#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ppl.run import file_sha256, sidecar_parity
from tools.ppl import validate_selected_exact_token as validator_module
from tools.ppl.validate_selected_exact_token import _identity, _unlink_if_owned, validate


class ValidateSelectedExactTokenTest(unittest.TestCase):
    def test_unlink_if_owned_removes_only_the_captured_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            path.write_text("owned", encoding="utf-8")
            owner = _identity(path)
            self.assertTrue(_unlink_if_owned(path, owner))
            self.assertFalse(path.exists())

            path.write_text("first", encoding="utf-8")
            stale_owner = _identity(path)
            replacement = path.with_name("replacement.json")
            replacement.write_text("replacement", encoding="utf-8")
            os.replace(replacement, path)
            self.assertFalse(_unlink_if_owned(path, stale_owner))
            self.assertEqual(path.read_text(encoding="utf-8"), "replacement")

    def test_main_preserves_dangling_output_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "admission.json"
            output.symlink_to(root / "missing")
            argv = ["validate_selected_exact_token.py", "--plan", str(root / "plan.json"),
                    "--campaign", str(root / "campaign.json"), "--out", str(output)]
            with patch.object(sys, "argv", argv), patch.object(
                    validator_module, "validate") as checker, self.assertRaises(SystemExit):
                validator_module.main()
            checker.assert_not_called()
            self.assertTrue(output.is_symlink())

    def test_main_rejects_symlinked_campaign_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            campaign_real = root / "campaign-real.json"
            campaign_link = root / "campaign.json"
            output = root / "admission.json"
            plan.write_text("{}\n", encoding="utf-8")
            campaign_real.write_text("{}\n", encoding="utf-8")
            campaign_link.symlink_to(campaign_real)
            argv = ["validate_selected_exact_token.py", "--plan", str(plan),
                    "--campaign", str(campaign_link), "--out", str(output)]
            with patch.object(sys, "argv", argv), patch.object(
                    validator_module, "validate") as checker, self.assertRaisesRegex(
                        SystemExit, "campaign authority must not be symlinked"):
                validator_module.main()
            checker.assert_not_called()
            self.assertFalse(os.path.lexists(output))

    def test_main_late_failure_preserves_replacement_output_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            campaign = root / "campaign.json"
            output = root / "admission.json"
            plan.write_text("{}\n", encoding="utf-8")
            campaign.write_text("{}\n", encoding="utf-8")
            value = {"status": "passed"}

            def validate_then_replace(*_args):
                if not output.exists():
                    return value
                output.unlink()
                output.write_text('{"foreign": true}\n', encoding="utf-8")
                raise ValueError("late revalidation failure")

            argv = ["validate_selected_exact_token.py", "--plan", str(plan),
                    "--campaign", str(campaign), "--out", str(output)]
            with patch.object(sys, "argv", argv), patch.object(
                    validator_module, "validate", side_effect=validate_then_replace
            ), self.assertRaisesRegex(SystemExit, "late revalidation failure"):
                validator_module.main()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"foreign": True})

    def test_main_link_race_preserves_replacement_output_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            campaign = root / "campaign.json"
            output = root / "admission.json"
            plan.write_text("{}\n", encoding="utf-8")
            campaign.write_text("{}\n", encoding="utf-8")
            value = {"status": "passed"}
            real_link = os.link

            def link_then_replace(source, destination):
                real_link(source, destination)
                Path(destination).unlink()
                Path(destination).write_text('{"foreign": true}\n', encoding="utf-8")

            argv = ["validate_selected_exact_token.py", "--plan", str(plan),
                    "--campaign", str(campaign), "--out", str(output)]
            with patch.object(sys, "argv", argv), patch.object(
                    validator_module, "validate", return_value=value
            ), patch.object(validator_module.os, "link", side_effect=link_then_replace), \
                    self.assertRaisesRegex(SystemExit, "hard-link inode differs"):
                validator_module.main()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"foreign": True})

    def fixture(self, root: Path) -> tuple[Path, Path]:
        bindings = {}
        for name in ("selection", "quality"):
            path = root / f"{name}.json"
            path.write_text("{}", encoding="utf-8")
            bindings[name] = {"path": str(path), "sha256": file_sha256(path)}
        source = root / "source.json"
        source.write_text(json.dumps({"metadata": {
            "config": {"sha256": "c" * 64}, "index": {"sha256": "d" * 64}}}))
        bindings["source"] = {"path": str(source), "sha256": file_sha256(source)}
        route = {
            "winner": "selected-g16-dense",
            "artifact": {"path": str(root / "selected.ninfer"),
                         "weights_id": "r9700-q4g64-n16k16-eval", "sha256": "a" * 64},
            "cache_profile": {"value_group": 16},
            "execution_profile": {"xattention_profile": "dense"},
            "selected_prefill_chunk": 2048,
        }
        selection_path = Path(bindings["selection"]["path"])
        selection_path.write_text(json.dumps({"selected_prefill_chunk": 2048}), encoding="utf-8")
        bindings["selection"]["sha256"] = file_sha256(selection_path)
        Path(route["artifact"]["path"]).write_text("artifact", encoding="utf-8")
        route["artifact"]["sha256"] = file_sha256(Path(route["artifact"]["path"]))
        scorer = root / "ninfer-ppl"
        bf16_scorer = root / "ppl.py"
        scorer.write_text("scorer", encoding="utf-8")
        bf16_scorer.write_text("reference", encoding="utf-8")
        python = root / "python"
        pyvenv = root / "pyvenv.cfg"
        corpus = root / "corpus.ids"
        corpus_manifest = root / "corpus.manifest.json"
        python.write_text("python", encoding="utf-8")
        pyvenv.write_text("venv", encoding="utf-8")
        corpus.write_text("1 2 3 4\n", encoding="utf-8")
        corpus_manifest.write_text("{}", encoding="utf-8")
        campaign_dir = root / "campaign"
        campaign_dir.mkdir()
        profile = "r9700-g16"
        quality_gate = 0.05
        plan = {
            "artifact_type": "ninfer_r9700_selected_exact_token_plan", "schema_version": 1,
            "status": "command_only_not_executed", "terminal_route": route,
            "terminal_selection": bindings["selection"], "quality_authority": bindings["quality"],
            "bf16_source_receipt": bindings["source"], "candidate_profile": "r9700-g16",
            "bf16_source": {"path": "/source", "shard_count": 18},
            "quality_tier": "capacity-speed",
            "quality_mean_nll_gate": quality_gate,
            "candidate_scorer": {"path": str(scorer), "sha256": file_sha256(scorer),
                                 "file_size_bytes": scorer.stat().st_size},
            "bf16_scorer": {"path": str(bf16_scorer), "sha256": file_sha256(bf16_scorer),
                            "file_size_bytes": bf16_scorer.stat().st_size},
            "corpus": {"path": str(corpus), "sha256": file_sha256(corpus),
                       "manifest_path": str(corpus_manifest),
                       "manifest_sha256": file_sha256(corpus_manifest)},
            "python": {"launcher_path": str(python), "sha256": file_sha256(python),
                       "pyvenv_cfg": {"path": str(pyvenv), "sha256": file_sha256(pyvenv)}},
            "workload": {"concurrency": 1, "lengths": [8192, 32768], "schedule": "decode",
                         "spec": "none", "draft_tokens": 0, "device": 0,
                         "execution_parity_max_abs_nll": 0.0},
            "outputs": {"campaign": str(campaign_dir),
                        "admission": str(root / "admission.json")},
        }
        plan["command"] = [
            str(python), str(validator_module.RUNNER),
            "--bf16-reference-ppl-bin", str(bf16_scorer),
            "--bf16-reference-weights", "/source",
            "--g16-ppl-bin", str(scorer), "--g16-weights", route["artifact"]["path"],
            "--ids", str(corpus), "--profiles", f"bf16-reference,{profile}",
            "--quality-tier", "capacity-speed", "--gate", f"{profile}={quality_gate}",
            "--schedule", "decode", "--prefill-chunk", "2048", "--device", "0",
            "--spec", "none",
            "--execution-parity-max-abs-nll", "0", "--no-position-extras",
            "--expected-q4-activation-bits", "8", "--expected-w8-activation-bits", "8",
            "--expected-fp8-qk-wmma", "1", "--expected-xattention-profile", "dense",
            "--out", str(campaign_dir),
        ]
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        cells = []
        shard_hashes = {str(i): "e" * 64 for i in range(18)}
        for tokens in (8192, 32768):
            raw = campaign_dir / f"{tokens}.decode.r9700-g16.json"
            scored = tokens - tokens // 2 - 1
            raw_value = {"scheme": "r9700-g16", "schedule": "decode",
                         "weights": route["artifact"]["path"],
                         "model_id": "qwen3.8-27b",
                         "weights_id": route["artifact"]["weights_id"],
                         "prompt_tokens": tokens, "skip_tokens": tokens // 2,
                         "tokens_scored": scored, "argmax_tokens": scored,
                         "spec": "none", "draft_tokens": 0, "device_graph": True,
                         "prefill_chunk": 2048, "non_finite": 0, "mean_nll": 1.0}
            raw.write_text(json.dumps(raw_value), encoding="utf-8")
            raw.with_suffix(".nllf32").write_bytes(struct.pack(f"<{scored}f", *([1.0] * scored)))
            raw.with_suffix(".argmaxi32").write_bytes(struct.pack(f"<{scored}i", *([1] * scored)))
            command = [str(scorer), "--weights", route["artifact"]["path"],
                       "--ids", str(corpus), "--scheme", "r9700-g16",
                       "--schedule", "decode", "--skip", "half", "--tokens", str(tokens),
                       "--prefill-chunk", "2048", "--device", "0", "--out-json", str(raw)]
            cells.append({**raw_value, "command": command,
                          "nll_sha256": file_sha256(raw.with_suffix(".nllf32")),
                          "argmax_sha256": file_sha256(raw.with_suffix(".argmaxi32"))})
            bf_raw = campaign_dir / f"{tokens}.decode.bf16-reference.json"
            bf_value = {**raw_value, "scheme": "bf16-reference",
                        "weights": "/source", "weights_id": "bf16-source",
                        "device_graph": False,
                        "source_config_sha256": "c" * 64,
                        "source_index_sha256": "d" * 64,
                        "source_shards_sha256": shard_hashes,
                        "source_tensor_count": 1199, "source_shard_count": 18}
            bf_raw.write_text(json.dumps(bf_value), encoding="utf-8")
            bf_raw.with_suffix(".nllf32").write_bytes(struct.pack(f"<{scored}f", *([1.0] * scored)))
            bf_raw.with_suffix(".argmaxi32").write_bytes(struct.pack(f"<{scored}i", *([1] * scored)))
            bf_command = [str(python), str(bf16_scorer), "--weights", "/source",
                          "--ids", str(corpus), "--scheme", "bf16-reference",
                          "--schedule", "decode", "--skip", "half", "--tokens", str(tokens),
                          "--prefill-chunk", "2048", "--device", "0", "--out-json", str(bf_raw)]
            cells.append({**bf_value, "command": bf_command,
                          "nll_sha256": file_sha256(bf_raw.with_suffix(".nllf32")),
                          "argmax_sha256": file_sha256(bf_raw.with_suffix(".argmaxi32"))})
        required = [(8192, "device_graph_parity"), (32768, "device_graph_parity")]
        labels = {"device_graph_parity": "eager"}
        for tokens, parity_key in required:
            raw = campaign_dir / f"{tokens}.decode.r9700-g16.{labels[parity_key]}.json"
            scored = tokens - tokens // 2 - 1
            raw_value = {"scheme": "r9700-g16", "schedule": "decode",
                         "weights": route["artifact"]["path"],
                         "model_id": "qwen3.8-27b",
                         "weights_id": route["artifact"]["weights_id"],
                         "prompt_tokens": tokens, "skip_tokens": tokens // 2,
                         "tokens_scored": scored, "argmax_tokens": scored,
                         "spec": "none", "draft_tokens": 0, "device_graph": False,
                         "prefill_chunk": 2048, "non_finite": 0, "mean_nll": 1.0}
            raw.write_text(json.dumps(raw_value), encoding="utf-8")
            raw.with_suffix(".nllf32").write_bytes(struct.pack(f"<{scored}f", *([1.0] * scored)))
            raw.with_suffix(".argmaxi32").write_bytes(struct.pack(f"<{scored}i", *([1] * scored)))
            primary = campaign_dir / f"{tokens}.decode.r9700-g16.json"
            parity = sidecar_parity(primary, raw, max_abs_nll=0.0)
            flags = {"device_graph_parity": ["--no-device-graph"]}[parity_key]
            command = [str(scorer), "--weights", route["artifact"]["path"],
                       "--ids", str(corpus), "--scheme", "r9700-g16",
                       "--schedule", "decode", "--skip", "half", "--tokens", str(tokens),
                       "--prefill-chunk", "2048", "--device", "0", "--out-json", str(raw),
                       *flags]
            cells.append({**raw_value, "command": command,
                          "nll_sha256": file_sha256(raw.with_suffix(".nllf32")),
                          "argmax_sha256": file_sha256(raw.with_suffix(".argmaxi32")),
                          parity_key: parity})
        campaign = {
            "artifact_type": "ninfer_r9700_ppl_campaign", "schema_version": 6, "pass": True,
            "model_id": "qwen3.8-27b", "lengths": [8192, 32768], "schedules": ["decode"],
            "reference_weights_id": "bf16-source",
            "prefill_chunk": 2048, "skip": "half",
            "weights_inputs": {"bf16-reference": "/source",
                               "r9700-g16": route["artifact"]["path"]},
            "scorers": {"bf16-reference": {}, "r9700-g16": {
                "path": str(scorer), "sha256": file_sha256(scorer), "bytes": scorer.stat().st_size}},
            "quality_tier": "capacity-speed", "execution_parity_max_abs_nll": 0.0,
            "spec": "none", "draft_tokens": 0,
            "position_extras_enabled": False,
            "xattention_profile": "dense",
            "corpus": {"path": str(corpus), "ids_sha256": file_sha256(corpus),
                       "manifest_path": str(corpus_manifest),
                       "manifest_sha256": file_sha256(corpus_manifest)},
            "reference_source": {"config_sha256": "c" * 64, "index_sha256": "d" * 64,
                                 "tensor_count": 1199, "shard_count": 18,
                                 "shards_sha256": shard_hashes},
            "candidate_artifact": {"weights_id": "r9700-q4g64-n16k16-eval",
                                   "sha256": route["artifact"]["sha256"]},
            "cells": cells,
        }
        campaign["scorers"]["bf16-reference"] = {
            "path": str(bf16_scorer), "sha256": file_sha256(bf16_scorer),
            "bytes": bf16_scorer.stat().st_size}
        campaign_path = campaign_dir / "results.json"
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        return plan_path, campaign_path

    def test_accepts_required_graph_eager_inventory_without_mtp_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan, campaign = self.fixture(Path(directory))
            plan_value = json.loads(plan.read_text())
            terminal = {"winner": plan_value["terminal_route"]["winner"],
                        "winner_artifact": plan_value["terminal_route"]["artifact"],
                        "winner_cache_profile": plan_value["terminal_route"]["cache_profile"],
                        "winner_execution_profile": plan_value["terminal_route"]["execution_profile"]}
            with patch("tools.ppl.validate_selected_exact_token.validate_terminal_production_authority",
                       return_value=(terminal, {})):
                result = validate(plan, campaign)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["compared_parity_cells"], 2)
            self.assertEqual(result["bf16_argmax_identity"], "diagnostic_only")
            self.assertEqual(result["mtp_diagnostics"], "optional_non_ranking_not_supplied")

    def test_rejects_mutated_duplicate_or_unknown_plan_arguments(self) -> None:
        for case in ("mutated-tier", "duplicate-option", "unknown-option"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                plan, campaign = self.fixture(Path(directory))
                value = json.loads(plan.read_text())
                if case == "mutated-tier":
                    index = value["command"].index("--quality-tier")
                    value["command"][index + 1] = "accuracy"
                elif case == "duplicate-option":
                    index = value["command"].index("--out")
                    value["command"][index:index] = ["--schedule", "decode"]
                else:
                    value["command"].append("--unexpected")
                plan.write_text(json.dumps(value))
                terminal = {
                    "winner": value["terminal_route"]["winner"],
                    "winner_artifact": value["terminal_route"]["artifact"],
                    "winner_cache_profile": value["terminal_route"]["cache_profile"],
                    "winner_execution_profile": value["terminal_route"]["execution_profile"],
                }
                with patch(
                    "tools.ppl.validate_selected_exact_token.validate_terminal_production_authority",
                    return_value=(terminal, {}),
                ), self.assertRaisesRegex(ValueError, "plan command differs"):
                    validate(plan, campaign)

    def test_rejects_one_greedy_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan, campaign = self.fixture(Path(directory))
            value = json.loads(campaign.read_text())
            variant = next(row for row in value["cells"] if "device_graph_parity" in row)
            variant["device_graph_parity"]["argmax_exact"] = False
            variant["device_graph_parity"]["argmax_mismatches"] = 1
            campaign.write_text(json.dumps(value))
            plan_value = json.loads(plan.read_text())
            terminal = {"winner": plan_value["terminal_route"]["winner"],
                        "winner_artifact": plan_value["terminal_route"]["artifact"],
                        "winner_cache_profile": plan_value["terminal_route"]["cache_profile"],
                        "winner_execution_profile": plan_value["terminal_route"]["execution_profile"]}
            with patch("tools.ppl.validate_selected_exact_token.validate_terminal_production_authority",
                       return_value=(terminal, {})), self.assertRaisesRegex(ValueError, "parity failed"):
                validate(plan, campaign)

    def test_rejects_misassembled_or_unbound_raw_inventory(self) -> None:
        cases = (
            "extra", "primary-parity", "wrong-flags", "hidden-mtp-before-output",
            "wrong-semantic-profile", "missing-bf-sidecar", "wrong-scorer",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                plan, campaign = self.fixture(Path(directory))
                value = json.loads(campaign.read_text())
                candidate = [row for row in value["cells"]
                             if row["scheme"] == "r9700-g16"]
                if case == "extra":
                    value["cells"].append(dict(candidate[0]))
                elif case == "primary-parity":
                    variant = next(row for row in candidate if "device_graph_parity" in row)
                    primary = next(row for row in candidate
                                   if ".eager." not in row["command"][row["command"].index("--out-json") + 1])
                    primary["device_graph_parity"] = variant.pop("device_graph_parity")
                elif case == "wrong-flags":
                    variant = next(row for row in candidate if "device_graph_parity" in row)
                    variant["command"].remove("--no-device-graph")
                elif case == "hidden-mtp-before-output":
                    primary = next(row for row in candidate if "device_graph_parity" not in row)
                    index = primary["command"].index("--out-json")
                    primary["command"][index:index] = ["--spec", "mtp", "--draft-tokens", "3"]
                elif case == "wrong-semantic-profile":
                    primary = next(row for row in candidate if "device_graph_parity" not in row)
                    primary["spec"] = "mtp"
                    primary["draft_tokens"] = 3
                    raw_path = Path(primary["command"][primary["command"].index("--out-json") + 1])
                    raw_value = json.loads(raw_path.read_text())
                    raw_value["spec"] = "mtp"
                    raw_value["draft_tokens"] = 3
                    raw_path.write_text(json.dumps(raw_value))
                elif case == "missing-bf-sidecar":
                    bf16 = next(row for row in value["cells"]
                                if row["scheme"] == "bf16-reference")
                    raw = Path(bf16["command"][bf16["command"].index("--out-json") + 1])
                    raw.with_suffix(".argmaxi32").unlink()
                elif case == "wrong-scorer":
                    candidate[0]["command"][0] = "/different/scorer"
                campaign.write_text(json.dumps(value))
                plan_value = json.loads(plan.read_text())
                terminal = {"winner": plan_value["terminal_route"]["winner"],
                            "winner_artifact": plan_value["terminal_route"]["artifact"],
                            "winner_cache_profile": plan_value["terminal_route"]["cache_profile"],
                            "winner_execution_profile": plan_value["terminal_route"]["execution_profile"]}
                with patch(
                    "tools.ppl.validate_selected_exact_token.validate_terminal_production_authority",
                    return_value=(terminal, {}),
                ), self.assertRaises(ValueError):
                    validate(plan, campaign)

    def test_hybrid_requires_bound_planner_and_command_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan, campaign = self.fixture(Path(directory))
            plan_value = json.loads(plan.read_text())
            campaign_value = json.loads(campaign.read_text())
            tool = Path(directory) / "hybrid-tool.py"
            tool.write_text("planner", encoding="utf-8")
            plan_value["terminal_route"]["artifact"]["weights_id"] = (
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
            )
            plan_value["terminal_route"]["hybrid_width_tool"] = {
                "path": str(tool), "sha256": file_sha256(tool)}
            plan_value["command"].insert(
                plan_value["command"].index("--out"), "--require-fp8-hybrid"
            )
            campaign_value["candidate_artifact"]["weights_id"] = (
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
            )
            for cell in campaign_value["cells"]:
                if cell["scheme"] != "r9700-g16":
                    continue
                cell["weights_id"] = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
                raw_path = Path(cell["command"][cell["command"].index("--out-json") + 1])
                raw = json.loads(raw_path.read_text())
                raw["weights_id"] = cell["weights_id"]
                raw_path.write_text(json.dumps(raw))
            campaign_value["required_candidate_identity"] = "fp8-hybrid-selection-authority"
            plan.write_text(json.dumps(plan_value))
            campaign.write_text(json.dumps(campaign_value))
            terminal = {"winner": plan_value["terminal_route"]["winner"],
                        "winner_artifact": plan_value["terminal_route"]["artifact"],
                        "winner_cache_profile": plan_value["terminal_route"]["cache_profile"],
                        "winner_execution_profile": plan_value["terminal_route"]["execution_profile"]}
            with patch("tools.ppl.validate_selected_exact_token.validate_terminal_production_authority",
                       return_value=(terminal, {})):
                self.assertEqual(validate(plan, campaign)["status"], "passed")
                plan_value["command"].remove("--require-fp8-hybrid")
                plan.write_text(json.dumps(plan_value))
                with self.assertRaisesRegex(ValueError, "hybrid identity"):
                    validate(plan, campaign)


if __name__ == "__main__":
    unittest.main()
