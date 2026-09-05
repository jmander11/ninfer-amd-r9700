#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bench.reconcile_qwen3_8_27b_dispatches import _publish, _snapshot, reconcile
from tools.bench.model_qwen3_8_27b_roofline import model


class DispatchReconciliationTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        root.mkdir(parents=True)
        files = {}
        for name in ("database", "power_before", "power_after", "terminal_selection",
                     "artifact", "benchmark_executable", "corpus", "source_matrix"):
            path = root / name
            path.write_text(name, encoding="utf-8")
            files[name] = _snapshot(path, name)
        report = root / "source_report.json"
        report.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": 20,
            "tests": [{"kind": "pp", "n_prompt": 2048,
                       "prefill_tok_s_mean": 2048.0,
                       "prefill_seconds_mean": 1.0}],
        }), encoding="utf-8")
        evaluation = root / "evaluation.json"
        evaluation.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_low_context_prefill_evaluation",
            "schema_version": 1, "minimum_p2048_tok_s": 2000.0,
            "observed_p2048_tok_s": 2048.0, "passes_p2048_gate": True,
            "ladder": [{"prompt_tokens": 2048, "prefill_tok_s_mean": 2048.0}],
        }), encoding="utf-8")
        files["source_report"] = _snapshot(report, "source report")
        files["low_context_evaluation"] = _snapshot(evaluation, "evaluation")
        benchmark_report = root / "profile_report.json"
        benchmark_report.write_text("profile", encoding="utf-8")
        files["benchmark_report"] = _snapshot(benchmark_report, "profile report")
        plan = root / "plan.json"
        plan.write_text(json.dumps({"required_power_profile": {
            "sysfs_path": "/sys/exact/power_profile"}}), encoding="utf-8")
        workload = {
            "concurrency": 1, "prompt_tokens": 2048, "generated_tokens": 0,
            "spec": "none", "draft_tokens": 0, "dflash_verify_width": 0,
            "kv_value_group": 16, "xattention_profile": "dense", "prefill_chunk": 2048,
        }
        selected_artifact = {**files["artifact"], "weights_id": "selected"}
        trace = root / "trace.json"
        dispatches = [
            {"dispatch_id": "trace:7", "rocprof_dispatch_id": 7, "symbol": "q4_kernel",
             "roctx_region": "prefill/text", "start_ns": 10, "end_ns": 30,
             "duration_ns": 20, "stream_id": 1, "grid": {"x": 2, "y": 3, "z": 1},
             "workgroup": {"x": 64, "y": 1, "z": 1}, "resources": {
                 "sgpr_count": 128, "vgpr_count": 32, "accum_vgpr_count": None, "lds_bytes": 1024,
                 "scratch_bytes": 0, "static_lds_bytes": None,
                 "static_scratch_bytes": None}},
            {"dispatch_id": "trace:8", "rocprof_dispatch_id": 8, "symbol": "sampling_kernel",
             "roctx_region": "prefill/orchestration", "start_ns": 31, "end_ns": 36,
             "duration_ns": 5, "stream_id": 1, "grid": {"x": 1, "y": 1, "z": 1},
             "workgroup": {"x": 64, "y": 1, "z": 1}, "resources": {
                 "sgpr_count": None, "vgpr_count": None, "accum_vgpr_count": None, "lds_bytes": None,
                 "scratch_bytes": None, "static_lds_bytes": None,
                 "static_scratch_bytes": None}},
        ]
        selected_route = {"kind": "pp", "prompt_tokens": 2048, "concurrency": 1,
                          "prefill_chunk": 2048, "kv_value_group": 16,
                          "xattention_profile": "dense", "artifact": selected_artifact,
                          "benchmark_executable": files["benchmark_executable"],
                          "unprofiled_p2048": {
                              "report": files["source_report"],
                              "evaluation": files["low_context_evaluation"],
                              "minimum_tok_s": 2000.0, "observed_tok_s": 2048.0,
                              "passes_gate": True}}
        trace.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_selected_profile_trace", "schema_version": 1,
            "status": "valid_attribution_only", "profile_timing_admissible": False,
            "plan": _snapshot(plan, "plan"),
            "inputs": {name: files[name] for name in
                       ("benchmark_report", "database", "power_before", "power_after",
                        "terminal_selection", "artifact", "benchmark_executable", "corpus")},
            "authorities": {name: files[name] for name in
                            ("source_matrix", "source_report", "low_context_evaluation")},
            "workload": workload,
            "selected_route": selected_route,
            "power_profile": {"required": "auto", "before": "auto", "after": "auto"},
            "dispatches": dispatches,
            "aggregates": {"dispatch_count": 2, "independent_device_service_time_ns": 25,
                           "device_wall_union_ns": 25},
        }), encoding="utf-8")
        source = root / "target_schedule_source.cpp"
        source.write_text("fixed schedule", encoding="utf-8")
        schedule = root / "schedule.json"
        common = {"marker": "prefill/text", "layer": {"kind": "text", "index": 0},
                  "grid": {"x": 2, "y": 3, "z": 1},
                  "workgroup": {"x": 64, "y": 1, "z": 1}}
        schedule.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_8_27b_static_dispatch_schedule",
            "schema_version": 1, "workload": workload, "selected_route": selected_route,
            "dispatch_order": "trace_start_ns_then_dispatch_id",
            "sources": [_snapshot(source, "schedule source")],
            "dispatches": [
                {"schedule_index": 0, **common, "symbol": "q4_kernel",
                 "call": {"id": "text.0.linear", "ordinal": 0,
                          "dispatch_index": 0, "dispatch_count": 1},
                 "classification": "modeled", "role": "text_mlp_gate_up",
                 "operation": "q4_linear", "format": "a8q4g64_bf16_output",
                 "parameters": {"tokens": 2048, "rows": 34816, "columns": 5120}},
                {"schedule_index": 1, "marker": "prefill/orchestration",
                 "layer": {"kind": "orchestration", "index": None},
                 "symbol": "sampling_kernel", "grid": {"x": 1, "y": 1, "z": 1},
                 "workgroup": {"x": 64, "y": 1, "z": 1},
                 "call": {"id": "sampling", "ordinal": 1,
                          "dispatch_index": 0, "dispatch_count": 1},
                 "classification": "unsupported",
                 "role": "prefill_bonus_sampling", "operation": "categorical_sampling",
                 "reason": "sampling has no selected logical roof model"},
            ],
        }), encoding="utf-8")
        return trace, schedule

    def test_exact_reconciliation_and_explicit_uncovered_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, schedule = self.fixture(Path(directory) / "case")
            result = reconcile(trace, schedule)
            self.assertEqual(result["coverage"]["trace_dispatch_count"], 2)
            self.assertEqual(result["coverage"]["modeled"],
                             {"dispatch_count": 1, "duration_ns": 20})
            self.assertEqual(result["coverage"]["unsupported"],
                             {"dispatch_count": 1, "duration_ns": 5})
            self.assertEqual(len(result["timing_authority"]["dispatches"]), 1)
            self.assertEqual(result["dispatch_inventory"]["dispatches"][0]["role"],
                             "text_mlp_gate_up")
            self.assertEqual(result["unprofiled_performance"]["prefill_tok_s_mean"], 2048.0)

            output = Path(directory) / "reconciliation.json"
            output.write_text(json.dumps(result), encoding="utf-8")
            roofline = model(output)
            self.assertEqual(roofline["timing"]["profiled_dispatch_count"], 1)
            self.assertEqual(roofline["timing"]["uncovered_dispatch_count"], 1)

    def test_rejects_order_signature_mismatch_and_ambiguous_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, schedule = self.fixture(Path(directory) / "case")
            value = json.loads(schedule.read_text(encoding="utf-8"))
            value["dispatches"][0]["grid"]["x"] = 9
            schedule.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs at grid"):
                reconcile(trace, schedule)

            trace2, schedule2 = self.fixture(Path(directory) / "collision")
            value = json.loads(trace2.read_text(encoding="utf-8"))
            value["dispatches"][1].update({
                "symbol": "q4_kernel", "roctx_region": "prefill/text", "start_ns": 10,
                "end_ns": 15, "duration_ns": 5, "grid": {"x": 2, "y": 3, "z": 1}})
            trace2.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "indistinguishable dispatch collisions"):
                reconcile(trace2, schedule2)

    def test_rejects_call_multiplicity_and_changed_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, schedule = self.fixture(Path(directory) / "case")
            value = json.loads(schedule.read_text(encoding="utf-8"))
            value["dispatches"][0]["call"]["dispatch_count"] = 2
            schedule.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact contiguous multiplicity"):
                reconcile(trace, schedule)

            trace2, schedule2 = self.fixture(Path(directory) / "source")
            value = json.loads(schedule2.read_text(encoding="utf-8"))
            Path(value["sources"][0]["path"]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                reconcile(trace2, schedule2)

    def test_rejects_trace_plan_timing_resource_and_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, schedule = self.fixture(root / "plan")
            value = json.loads(trace.read_text(encoding="utf-8"))
            Path(value["plan"]["path"]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace plan path or SHA-256 changed"):
                reconcile(trace, schedule)

            trace, schedule = self.fixture(root / "wall")
            value = json.loads(trace.read_text(encoding="utf-8"))
            value["aggregates"]["device_wall_union_ns"] = 24
            trace.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wall-union aggregate differs"):
                reconcile(trace, schedule)

            trace, schedule = self.fixture(root / "resource")
            value = json.loads(trace.read_text(encoding="utf-8"))
            value["dispatches"][0]["resources"].pop("scratch_bytes")
            trace.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact trace fields"):
                reconcile(trace, schedule)

            trace, schedule = self.fixture(root / "report")
            value = json.loads(trace.read_text(encoding="utf-8"))
            report = Path(value["authorities"]["source_report"]["path"])
            report_value = json.loads(report.read_text(encoding="utf-8"))
            report_value["tests"][0]["prefill_tok_s_mean"] = 2047.0
            report.write_text(json.dumps(report_value), encoding="utf-8")
            # Keep this test focused on the semantic mismatch after simulating a newly
            # validated snapshot of the changed authority.
            snapshot = _snapshot(report, "source report")
            value["authorities"]["source_report"] = snapshot
            value["selected_route"]["unprofiled_p2048"]["report"] = snapshot
            trace.write_text(json.dumps(value), encoding="utf-8")
            schedule_value = json.loads(schedule.read_text(encoding="utf-8"))
            schedule_value["selected_route"] = value["selected_route"]
            schedule.write_text(json.dumps(schedule_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "performance authorities disagree"):
                reconcile(trace, schedule)

    def test_atomic_publish_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            _publish(output, {"ok": True})
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _publish(output, {"ok": False})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})

    def test_atomic_publish_refuses_dangling_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.symlink_to(Path(directory) / "absent")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _publish(output, {"ok": True})
            self.assertTrue(output.is_symlink())

    def test_hardware_trace_schema_two_needs_no_dense_low_context_surrogate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, schedule = self.fixture(Path(directory) / "case")
            trace_value = json.loads(trace.read_text(encoding="utf-8"))
            trace_value["schema_version"] = 2
            trace_value["authorities"] = {
                "source_matrix": trace_value["authorities"]["source_matrix"]}
            trace_value["selected_route"].pop("unprofiled_p2048")
            trace.write_text(json.dumps(trace_value), encoding="utf-8")
            schedule_value = json.loads(schedule.read_text(encoding="utf-8"))
            schedule_value["selected_route"] = trace_value["selected_route"]
            schedule.write_text(json.dumps(schedule_value), encoding="utf-8")
            result = reconcile(trace, schedule)
            self.assertIsNone(result["unprofiled_performance"])
            self.assertEqual(set(result["timing_authority"]["source"]),
                             {"trace_authority", "source_matrix"})

    def test_unmarked_dispatch_must_remain_explicitly_unmodeled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, schedule = self.fixture(Path(directory) / "case")
            trace_value = json.loads(trace.read_text(encoding="utf-8"))
            trace_value["dispatches"][1]["roctx_region"] = None
            trace.write_text(json.dumps(trace_value), encoding="utf-8")
            schedule_value = json.loads(schedule.read_text(encoding="utf-8"))
            schedule_value["dispatches"][1]["marker"] = None
            schedule.write_text(json.dumps(schedule_value), encoding="utf-8")
            result = reconcile(trace, schedule)
            self.assertIsNone(result["dispatches"][1]["marker"])
            schedule_value["dispatches"][1].update({
                "classification": "modeled", "role": "guessed", "operation": "guessed",
                "format": "guessed", "parameters": {}})
            schedule_value["dispatches"][1].pop("reason")
            schedule.write_text(json.dumps(schedule_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires an exact ROCTX marker"):
                reconcile(trace, schedule)


if __name__ == "__main__":
    unittest.main()
