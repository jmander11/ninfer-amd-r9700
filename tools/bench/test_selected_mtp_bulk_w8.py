#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench import finalize_selected_mtp_bulk_w8 as finalizer
from tools.bench import prepare_selected_mtp_bulk_w8 as preparer


class SelectedMtpBulkW8Test(unittest.TestCase):
    def route(self, root: Path, weights: str = preparer.MIXED_ID) -> dict:
        build = root / "build"
        planner = build / "src/ninfer_r9700_runtime_planner_qual"
        planner.parent.mkdir(parents=True); planner.write_bytes(b"planner")
        artifact = root / "model.ninfer"; artifact.write_bytes(b"artifact")
        bench = build / "bench/ninfer_bench"; bench.parent.mkdir(); bench.write_bytes(b"aaCODEbb")
        selection = root / "selection.json"; selection.write_text("{}")
        corpus = root / "corpus.ids"; corpus.write_text("1 2 3\n")
        whole = root / "whole.json"
        whole.write_text(json.dumps({"corpus": str(corpus), "corpus_sha256": finalizer.sha(corpus)}))
        return {"terminal_selection": {"path": str(selection), "sha256": finalizer.sha(selection)},
                "winner": "mixed-dense-g16",
                "artifact": {"path": str(artifact), "sha256": finalizer.sha(artifact),
                             "file_size_bytes": artifact.stat().st_size, "weights_id": weights},
                "benchmark": {"path": str(bench), "sha256": finalizer.sha(bench),
                              "file_size_bytes": bench.stat().st_size},
                "build_directory": str(build), "cache_profile": {"value_group": 16},
                "execution_profile": {"xattention_profile": "dense"},
                "selected_prefill_chunk": 2048,
                "source_matrices": {"pareto-whole": {"path": str(whole),
                                                       "sha256": finalizer.sha(whole)}}}

    def raw(self, path: Path, samples: int = 64) -> None:
        rows = [{"rows": m, "columns": n, "dispatches_per_full_chunk": count,
                 "oracle": "independent FP64 represented A8G32-times-W8G32 formula",
                 "oracle_sample_count": samples, "oracle_max_bf16_steps": 2,
                 "candidate_median_ms": 1.0, "baseline_median_ms": 2.0,
                 "candidate_faster": True}
                for m, n, count in ((5120, 10240, 1), (1024, 5120, 2))]
        path.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_mtp_bulk_w8_operator_qualification",
            "schema_version": 1, "status": "passed", "tokens": 2048, "concurrency": 1,
            "power_profile": {"path": str(finalizer.POWER), "value": "auto"},
            "hardware": {"architecture": "gfx1201", "wave_size": 32},
            "candidate": {"activation": "signed A8G32", "weight": "W8G32_F16S",
                          "opcode": "v_wmma_i32_16x16x16_iu8"},
            "baseline": {"activation": "signed A8G32", "route": "a8w8g32_linear_wmma32"},
            "resources": {"vgpr": 50, "lds_bytes": 4352, "scratch_bytes": 0}, "shapes": rows}))
        value = json.loads(path.read_text())
        value["executable"] = {"path": str(preparer.QUALIFIER.resolve()),
                               "sha256": finalizer.sha(preparer.QUALIFIER)}
        sources = (("qualifier", finalizer.QUALIFIER_SOURCE),
                   ("contract_header", finalizer.REPO / "src/ops/r9700/linear/r9700_linear.h"),
                   ("kernel", finalizer.KERNEL_SOURCE),
                   ("dispatch_profile", finalizer.PROFILE_SOURCE))
        value["sources"] = [{"role": role, "path": str(source.resolve()),
                             "sha256": ("pre-admission-profile" if role == "dispatch_profile"
                                        else finalizer.sha(source))}
                            for role, source in sources]
        path.write_text(json.dumps(value))

    @staticmethod
    def trace_rows() -> list[dict]:
        region = "ninfer.mtp.prefill.mtp_chunk payload=2048"
        return [{"symbol": "ninfer::a8w8g32_linear_prefill_cta_kernel",
                 "roctx_region": region, "grid": {"x": x, "y": 32, "z": 1},
                 "workgroup": {"x": 512, "y": 1, "z": 1}}
                for x in (40960, 8192, 8192)]

    def prepare_package(self, root: Path, route: dict) -> tuple[Path, dict]:
        output = root / "package"
        operator = root / "operator.json"; self.raw(operator)
        with patch.object(preparer, "resolve", return_value=route):
            plan = preparer.prepare(Path(route["terminal_selection"]["path"]), operator, output)
        return output, plan

    def materialize(self, package: Path, plan: dict, route: dict, samples: int = 64) -> None:
        outputs = plan["outputs"]
        self.raw(Path(outputs["operator"]), samples)
        Path(outputs["benchmark_report"]).write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": finalizer.REPORT_SCHEMA_VERSION,
            "command": " ".join(plan["benchmark_command"]),
            "artifact": {"path": route["artifact"]["path"],
                         "file_size_bytes": route["artifact"]["file_size_bytes"]},
            "load": {"target": "qwen3_8_27b_r9700", "weights_id": preparer.MIXED_ID},
            "config": {"concurrency": 1, "prefill_chunk": 2048, "kv_value_group": 16,
                       "q4_activation_bits": 8, "w8_activation_bits": 8,
                       "fp8_qk_wmma_enabled": True,
                       "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                       "xattention_qualification": False, "spec": "mtp", "draft_tokens": 3,
                       "proposal_head": "optimized", "repetitions": 1, "warmup": 1},
            "tests": [{"kind": "pp", "n_prompt": 2048, "n_gen": 0,
                       "requested_output_tokens": 1}]}))
        Path(outputs["trace_database"]).parent.mkdir(); Path(outputs["trace_database"]).write_bytes(b"db")
        Path(outputs["power_before"]).write_text("auto\n"); Path(outputs["power_after"]).write_text("auto\n")
        Path(outputs["production_code_object"]).write_bytes(b"CODE")
        Path(outputs["production_assembly"]).write_text("""
0000000000001000 <a8w8g32_linear_prefill_cta_kernel>:
 v_wmma_i32_16x16x16_iu8 v[0:7], v[1:2], v[3:4], 0 neg_lo:[1,1,0]
 v_wmma_i32_16x16x16_iu8 v[0:7], v[1:2], v[3:4], v[0:7] neg_lo:[1,1,0]
0000000000001100 <next_kernel>:
 s_endpgm
""")
        Path(outputs["production_metadata"]).write_text("""
  - .args:
    .group_segment_fixed_size: 4352
    .name: a8w8g32_linear_prefill_cta_kernel
    .private_segment_fixed_size: 0
    .vgpr_count: 50
""")

    def invoke(self, route: dict, package: Path, plan: dict, out: Path):
        o = plan["outputs"]
        return finalizer.finalize(Path(route["terminal_selection"]["path"]), package / "plan.json",
            Path(o["operator"]), Path(o["production_assembly"]), Path(o["production_metadata"]),
            Path(o["production_code_object"]),
            Path(o["benchmark_report"]), Path(o["trace_database"]), Path(o["power_before"]),
            Path(o["power_after"]), preparer.QUALIFIER,
            Path(route["build_directory"]) / "src/ninfer_r9700_runtime_planner_qual", out,
            power_reader=lambda _: "auto")

    def test_prepare_non_mixed_is_inapplicable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root, "r9700-q4g64-mse-eval")
            with patch.object(preparer, "resolve", return_value=route), self.assertRaisesRegex(ValueError, "inapplicable"):
                preparer.prepare(Path(route["terminal_selection"]["path"]), root / "absent.json",
                                 root / "package")

    def test_prepare_is_exact_c1_and_trace_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root); package, plan = self.prepare_package(root, route)
            commands = (package / "commands.sh").read_text()
            self.assertEqual((plan["concurrency"], plan["selected_prefill_chunk"]), (1, 2048))
            self.assertNotIn("--mtp-bulk-selected-chunk", commands)
            self.assertIn("--kernel-trace", commands); self.assertNotIn("--concurrency 2", commands)
            self.assertIn("python3 -m tools.bench.finalize_selected_mtp_bulk_w8", commands)

    def test_finalize_binds_oracle_signed_isa_and_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root); package, plan = self.prepare_package(root, route)
            self.materialize(package, plan, route); out = package / "evidence.json"
            with patch.object(finalizer, "resolve", return_value=route), patch.object(
                    finalizer, "_parse_database", return_value=(self.trace_rows(), {"dispatch_count": 3})):
                value = self.invoke(route, package, plan, out)
            self.assertEqual(value["native_hardware"]["neg_lo"], [[1, 1, 0], [1, 1, 0]])
            self.assertEqual(value["selected_engine_execution"]["dispatch_count"], 3)

    def test_finalize_rejects_bypassed_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root); package, plan = self.prepare_package(root, route)
            self.materialize(package, plan, route, 0)
            with patch.object(finalizer, "resolve", return_value=route), patch.object(
                    finalizer, "_parse_database", return_value=(self.trace_rows(), {"dispatch_count": 3})), \
                    self.assertRaisesRegex(ValueError, "incomplete"):
                self.invoke(route, package, plan, package / "evidence.json")

    def test_finalize_rejects_mutated_trace_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root); package, plan = self.prepare_package(root, route)
            self.materialize(package, plan, route)
            mutated = json.loads((package / "plan.json").read_text())
            mutated["benchmark_command"][mutated["benchmark_command"].index("--device") + 1] = "1"
            (package / "plan.json").write_text(json.dumps(mutated))
            with patch.object(finalizer, "resolve", return_value=route), \
                    self.assertRaisesRegex(ValueError, "plan differs"):
                self.invoke(route, package, plan, package / "evidence.json")

    def test_finalize_rejects_duplicate_shape_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root); package, plan = self.prepare_package(root, route)
            self.materialize(package, plan, route)
            raw = Path(plan["outputs"]["operator"]); value = json.loads(raw.read_text())
            value["shapes"].append(dict(value["shapes"][0])); raw.write_text(json.dumps(value))
            with patch.object(finalizer, "resolve", return_value=route), \
                    self.assertRaisesRegex(ValueError, "incomplete"):
                self.invoke(route, package, plan, package / "evidence.json")

    def test_finalize_rejects_fallback_engine_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root); package, plan = self.prepare_package(root, route)
            self.materialize(package, plan, route); rows = self.trace_rows()
            rows[0]["symbol"] = "a8w8g32_linear_wmma32_kernel"
            with patch.object(finalizer, "resolve", return_value=route), patch.object(
                    finalizer, "_parse_database", return_value=(rows, {"dispatch_count": 3})), \
                    self.assertRaisesRegex(ValueError, "three native W8 CTA"):
                self.invoke(route, package, plan, package / "evidence.json")

    def test_raced_publication_is_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); route = self.route(root); package, plan = self.prepare_package(root, route)
            self.materialize(package, plan, route); out = package / "evidence.json"
            def raced_link(_source, destination): Path(destination).write_text("raced")
            with patch.object(finalizer, "resolve", return_value=route), patch.object(
                    finalizer, "_parse_database", return_value=(self.trace_rows(), {"dispatch_count": 3})), \
                    patch.object(finalizer.os, "link", side_effect=raced_link), \
                    self.assertRaisesRegex(ValueError, "not the owned"):
                self.invoke(route, package, plan, out)
            self.assertEqual(out.read_text(), "raced")


if __name__ == "__main__":
    unittest.main()
