"""CPU integration checks for retained capacity and fresh reporting identities."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ppl import benchmark_reporting_recovery as reporting
from tools.ppl import fp8_context_recovery as fp8
from tools.ppl.assemble_pareto import validate_chunk_candidate_bindings


class ReportingBridgeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=reporting.REPO)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = []
        self.frozen = {"files": {}}
        self.resource = {"hybrid_builds": []}
        self.bridge = {
            "artifact_type": "ninfer_r9700_benchmark_reporting_recovery", "schema_version": 1,
            "scope": reporting.SCOPE,
            "preserves": ["numerical_quality", "selected_prefill_chunk", "capacity"],
            "reuses_concurrent_prefill_metrics": False, "profiles": [],
        }
        for group in (16, 32):
            for attention in ("dense", "b128-s16-tau900"):
                row = {"kv_value_group": group, "xattention_profile": attention}
                for label in ("old_nonhybrid", "old_hybrid", "new"):
                    build = self.root / f"{label}-{group}-{attention}"
                    row[label + "_benchmark"] = self.write(build / "bench/ninfer_bench", label, executable=True)
                    row[label + "_planner"] = self.write(build / "src/ninfer_r9700_runtime_planner_qual",
                                                          label + "planner", executable=True)
                    field = "build_cache" if label == "new" else label + "_build_cache"
                    row[field] = self.write(build / "CMakeCache.txt",
                        f"NINFER_R9700_KV_VALUE_GROUP:STRING={group}\n"
                        f"NINFER_R9700_XATTENTION_PROFILE:STRING={attention}\n")
                    if label == "old_nonhybrid":
                        for item in (row[label + "_benchmark"], row[label + "_planner"], row[field]):
                            self.frozen["files"][str(Path(item["path"]).relative_to(reporting.REPO))] = item["sha256"]
                row["host_planner_check"] = self.write(self.root / f"check-{group}-{attention}.json", {
                    "command": [row["new_planner"]["path"], "--host-split512-routing"], "exit_code": 0,
                    "stdout": self.write(self.root / f"stdout-{group}-{attention}", "PASS\n"),
                    "stderr": self.write(self.root / f"stderr-{group}-{attention}", ""),
                })
                self.bridge["profiles"].append(row)
                hybrid = {"weights_id": fp8.HYBRID, "kv_value_group": group,
                          "xattention_profile": attention,
                          "new_benchmark": row["old_hybrid_benchmark"],
                          "new_planner": row["old_hybrid_planner"],
                          "build_cache": row["old_hybrid_build_cache"]}
                for recipe in ("allq4", "mixed", fp8.HYBRID):
                    old = row["old_nonhybrid_benchmark"]
                    source = {"weights_id": recipe, "kv_value_group": group,
                              "xattention_profile": attention,
                              "artifact": {"weights_id": recipe}, "benchmark_executable": old}
                    self.sources.append(source)
                    if recipe == fp8.HYBRID:
                        hybrid["old_benchmark"] = old
                self.resource["hybrid_builds"].append(hybrid)
        self.chunk = {"selected_prefill_chunk": 2048, "sources": self.sources}
        self.resource["chunk_selection"] = self.write(self.root / "chunk.json", self.chunk)
        self.resource["frozen_panel_inputs"] = self.write(self.root / "frozen.json", self.frozen)
        self.resource["source_review"] = self.write(self.root / "model-review.json", {"status": "unchanged_arithmetic"})
        self.bridge["fp8_context_resource_recovery"] = self.write(self.root / "resource.json", self.resource)
        self.review = {
            "artifact_type": "ninfer_r9700_benchmark_reporting_source_review", "schema_version": 1,
            "status": "reporting_only", "scope": reporting.SCOPE,
            "model_runtime_changed": False, "arithmetic_changed": False, "kernel_or_recipe_changed": False,
            "base_source_review": self.resource["source_review"],
            "sources": [fp8.identity(reporting.REPO / path) for path in reporting.SOURCE_FILES],
        }
        self.bridge["source_review"] = self.write(self.root / "review.json", self.review)
        self.addCleanup(patch.stopall)
        # The existing physical FP8 authority has its own tests. Keep real file,
        # cache, planner-receipt, matrix and new semantic-review validation here.
        patch.object(fp8, "validate_bridge", return_value=self.resource).start()
        patch.object(fp8, "check_review").start()

    def write(self, path, value, executable=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return fp8.executable_identity(path) if executable else fp8.identity(path)

    def provenance(self):
        result = []
        for index, original in enumerate(self.sources):
            row = reporting.mapping(self.bridge, original["kv_value_group"], original["xattention_profile"])
            source = {"artifact": original["artifact"], "cache_value_group": original["kv_value_group"],
                      "benchmark_executable": row["new_benchmark"],
                      "quality": {"representation": {"xattention_profile": original["xattention_profile"]}},
                      "matrices": {}}
            for preset in ("pareto-capacity", "pareto-whole"):
                manifest = {"artifact": source["artifact"],
                            "bench": reporting.expected_benchmark(source, preset, self.bridge)}
                if original["weights_id"] == fp8.HYBRID:
                    manifest["hybrid_shared_workspace_authority"] = {"tool": row[
                        "new_planner" if preset == "pareto-whole" else "old_hybrid_planner"]}
                source["matrices"][preset] = {**self.write(self.root / f"matrix-{index}-{preset}.json", manifest),
                                               "reports": []}
            result.append(source)
        return result

    def test_complete_bridge_and_twelve_old_capacity_new_whole_joins(self):
        self.assertEqual(reporting.validate_bridge_value(self.bridge, self.chunk), self.bridge)
        sources = self.provenance()
        validate_chunk_candidate_bindings(self.chunk, sources, self.resource, self.bridge)
        with self.assertRaisesRegex(ValueError, "prefill-chunk selection identity"):
            validate_chunk_candidate_bindings(self.chunk, sources, self.resource)

    def test_old_concurrent_reporter_cannot_supply_new_whole(self):
        sources = self.provenance()
        source = sources[0]
        binding = source["matrices"]["pareto-whole"]
        path = Path(binding["path"])
        manifest = json.loads(path.read_text())
        manifest["bench"] = self.bridge["profiles"][0]["old_nonhybrid_benchmark"]
        source["matrices"]["pareto-whole"] = self.write(path, manifest)
        with self.assertRaisesRegex(ValueError, "matrix artifact/benchmark differs"):
            validate_chunk_candidate_bindings(self.chunk, sources, self.resource, self.bridge)

    def test_selected_resolver_and_dflash_use_fresh_whole_planner(self):
        from tools.bench import assemble_dflash_selection as dflash
        source = self.provenance()[2]  # G16/dense hybrid.
        row = self.bridge["profiles"][0]
        artifact = self.write(self.root / "model.ninfer", "artifact", executable=True)
        artifact["weights_id"] = fp8.HYBRID
        source.update({"candidate": "winner", "artifact": artifact,
                       "fp8_context_resource_recovery": self.bridge["fp8_context_resource_recovery"],
                       "benchmark_reporting_recovery": self.write(self.root / "bridge.json", self.bridge)})
        cache = {"value_group": 16}
        execution = {"xattention_profile": "dense"}
        for preset, binding in source["matrices"].items():
            path = Path(binding["path"])
            manifest = json.loads(path.read_text())
            manifest.update({"artifact_type": "ninfer_bench_matrix_run", "schema_version": 14,
                             "preset": preset, "artifact": artifact, "concurrency": [1, 2, 3, 4],
                             "selected_prefill_chunk": 2048, "expected_kv_value_group": 16,
                             "expected_xattention_profile": "dense",
                             "required_candidate_identity": "fp8-hybrid-selection-authority"})
            source["matrices"][preset] = self.write(path, manifest)
        terminal = {"winner": "winner", "winner_artifact": artifact,
                    "winner_cache_profile": cache, "winner_execution_profile": execution}
        candidate = {"name": "winner", "weight_recipe": artifact,
                     "cache_profile": cache, "execution_profile": execution}
        base = {"terminal_production_selection": terminal, "source_provenance": [source],
                "candidates": [candidate], "selected_prefill_chunk": 2048}
        path = Path(self.write(self.root / "selection.json", base)["path"])
        build = Path(row["new_benchmark"]["path"]).parent.parent
        self.write(build / "CTestTestfile.cmake", "# fixture\n")
        spec = importlib.util.spec_from_file_location("reporting_route_test", reporting.REPO /
            "profiles/bench/post-terminal-focused-verification-20260905/resolve.py")
        resolver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resolver)
        with patch.object(resolver, "load_payload", return_value=base), \
                patch.object(resolver, "validate_terminal_production_authority", return_value=(terminal, candidate)), \
                patch.object(resolver, "inspect_artifact", return_value=artifact), \
                patch.object(resolver, "bind_n16_migration_receipt", return_value=artifact), \
                patch.object(resolver, "require_fp8_hybrid_artifact", return_value=artifact), \
                patch.object(resolver, "inspect_executable", side_effect=fp8.executable_identity), \
                patch.object(resolver, "validate_build_profile"), \
                patch.object(resolver, "validate_hybrid_shared_workspace_authority", side_effect=lambda value, _: value), \
                patch.object(dflash, "validate_hybrid_shared_workspace_authority", side_effect=lambda value, _: value):
            route = resolver.resolve(path)
            self.assertEqual(route["benchmark"], row["new_benchmark"])
            self.assertEqual(route["hybrid_width_tool"], row["new_planner"])
            self.assertEqual(dflash._selected_hybrid_authority(base, artifact, 2048)["tool"],
                             row["new_planner"])
            source.pop("benchmark_reporting_recovery")
            with self.assertRaisesRegex(ValueError, "different planners"):
                dflash._selected_hybrid_authority(base, artifact, 2048)

    def test_changed_semantics_profile_or_planner_result_reject(self):
        for change in ("model", "cache", "planner", "bench"):
            with self.subTest(change=change):
                bridge = copy.deepcopy(self.bridge)
                row = bridge["profiles"][0]
                if change == "model":
                    review = {**self.review, "model_runtime_changed": True}
                    bridge["source_review"] = self.write(self.root / "bad-review.json", review)
                elif change == "cache":
                    path = Path(row["build_cache"]["path"])
                    original = path.read_text()
                    row["build_cache"] = self.write(path, "NINFER_R9700_KV_VALUE_GROUP:STRING=32\n")
                elif change == "planner":
                    check = json.loads(Path(row["host_planner_check"]["path"]).read_text())
                    check["exit_code"] = 1
                    row["host_planner_check"] = self.write(self.root / "bad-check.json", check)
                else:
                    row["new_benchmark"]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    reporting.validate_bridge_value(bridge)
                if change == "cache":
                    path.write_text(original)


if __name__ == "__main__":
    unittest.main()
