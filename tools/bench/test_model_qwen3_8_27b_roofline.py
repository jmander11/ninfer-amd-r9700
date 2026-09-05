#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.bench.model_qwen3_8_27b_roofline import FORMATS, _counts, model
from tools.r9700.a8q4_prefill_traffic import (
    M64_N64,
    W8_M64_N64,
    Shape,
    requested_traffic,
    requested_w8_traffic,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QwenRooflineTest(unittest.TestCase):
    def fixture(self, root: Path, operations: list[tuple[str, str, dict]], *, overlap=False,
                auto=True, keep: dict | None = None) -> tuple[Path, Path, Path | None]:
        root.mkdir(parents=True, exist_ok=True)
        sparse_profile = any(operation.startswith("sparse_") for operation, _, _ in operations)
        timing_path = root / "timing.json"
        timing_dispatches = []
        inventory_dispatches = []
        for index, (operation, role, parameters) in enumerate(operations):
            begin = index * (50 if overlap else 100) + 1
            dispatch_id = f"d{index}"
            symbol = f"exact_{operation}_symbol"
            stage = "attention" if "attention" in operation or "sparse" in operation else "gdn"
            timing_dispatches.append({
                "dispatch_id": dispatch_id, "symbol": symbol, "stage": stage,
                "start_ns": begin, "end_ns": begin + 100, "duration_ns": 100,
            })
            inventory_dispatches.append({
                "dispatch_id": dispatch_id, "symbol": symbol, "stage": stage,
                "operation": operation, "role": role, "format": FORMATS[operation],
                "parameters": parameters,
            })
        workload = {
            "kind": "pp", "prompt_tokens": 2048, "concurrency": 1,
            "prefill_chunk": 2048, "artifact_path": "/exact/selected.ninfer",
            "artifact_sha256": "a" * 64, "weights_id": "selected",
            "executable_path": "/exact/ninfer_bench", "executable_sha256": "d" * 64,
            "kv_value_group": 16,
            "xattention_profile": "b128-s16-tau900" if sparse_profile else "dense",
        }
        analyzer_path = root / "analyzer.json"
        analyzer_path.write_text(json.dumps({
            "schema": "ninfer_selected_region_attribution", "schema_version": 1,
            "workload": {"kind": "pp", "prompt_tokens": 2048, "concurrency": 1,
                         "prefill_chunk": 2048},
            "operator_stage_attribution": {"complete": True},
        }), encoding="utf-8")
        report_path = root / "report.json"
        report_path.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": 20,
            "load": {"weights_id": "selected"},
            "config": {"concurrency": 1, "prefill_chunk": 2048, "kv_value_group": 16,
                       "xattention_qualification": sparse_profile,
                       **({"xattention_profile": "b128-s16-tau900"}
                          if sparse_profile else {})},
            "tests": [{"kind": "pp", "n_prompt": 2048,
                       "prefill_tok_s_mean": 2048.0, "prefill_seconds_mean": 1.0}],
        }), encoding="utf-8")
        timing_path.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_8_27b_dispatch_timing", "schema_version": 1,
            "power_profile": {
                "required": "auto", "observed": "auto" if auto else "profile_standard",
                "rechecked_after": "auto", "sysfs_path": "/sys/exact/power_profile",
            },
            "source": {"analyzer": str(analyzer_path),
                       "analyzer_sha256": sha256(analyzer_path),
                       "benchmark_report": str(report_path),
                       "benchmark_report_sha256": sha256(report_path)},
            "workload": workload, "dispatches": timing_dispatches,
        }), encoding="utf-8")
        inventory_path = root / "inventory.json"
        inventory_path.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_8_27b_dispatch_inventory", "schema_version": 1,
            "timing_authority_sha256": sha256(timing_path), "workload": workload,
            "dispatches": inventory_dispatches,
        }), encoding="utf-8")
        keep_path = None
        if keep is not None:
            keep_path = root / "keep.json"
            keep_path.write_text(json.dumps({
                "artifact_type": "ninfer_qwen3_8_27b_xattention_keep_counts",
                "schema_version": 1, "timing_authority_sha256": sha256(timing_path),
                "dispatches": [dict(dispatch_id="d0", **keep)],
            }), encoding="utf-8")
        evaluation_path = root / "evaluation.json"
        evaluation_path.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_low_context_prefill_evaluation",
            "schema_version": 1, "minimum_p2048_tok_s": 2000.0,
            "observed_p2048_tok_s": 2048.0, "passes_p2048_gate": True,
        }), encoding="utf-8")
        evaluation_snapshot = {"path": str(evaluation_path.resolve()),
                               "file_size_bytes": evaluation_path.stat().st_size,
                               "sha256": sha256(evaluation_path)}
        report_snapshot = {"path": str(report_path.resolve()),
                           "file_size_bytes": report_path.stat().st_size,
                           "sha256": sha256(report_path)}
        coverage = []
        trace_dispatches = []
        for index, (timing_row, inventory_row) in enumerate(
                zip(timing_dispatches, inventory_dispatches, strict=True)):
            marker = timing_row["stage"]
            common = {**timing_row, "rocprof_dispatch_id": index,
                      "marker": marker, "grid": {"x": 1, "y": 1, "z": 1},
                      "workgroup": {"x": 64, "y": 1, "z": 1}}
            coverage.append({**common, "classification": "modeled",
                             **{key: inventory_row[key] for key in
                                ("operation", "role", "format", "parameters")}})
            trace_dispatches.append({
                **{key: common[key] for key in
                   ("dispatch_id", "rocprof_dispatch_id", "symbol", "start_ns", "end_ns",
                    "duration_ns", "grid", "workgroup")},
                "roctx_region": marker,
            })
        trace_path = root / "trace.json"
        trace_path.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_selected_profile_trace", "schema_version": 1,
            "status": "valid_attribution_only", "profile_timing_admissible": False,
            "selected_route": {
                "kind": "pp", "prompt_tokens": 2048, "concurrency": 1,
                "prefill_chunk": 2048, "kv_value_group": 16,
                "xattention_profile": workload["xattention_profile"],
                "artifact": {"path": workload["artifact_path"],
                             "sha256": workload["artifact_sha256"], "weights_id": "selected"},
                "benchmark_executable": {"path": workload["executable_path"],
                                         "sha256": workload["executable_sha256"]},
                "unprofiled_p2048": {"evaluation": evaluation_snapshot,
                                     "report": report_snapshot,
                                     "minimum_tok_s": 2000.0,
                                     "observed_tok_s": 2048.0,
                                     "passes_gate": True},
            },
            "authorities": {"low_context_evaluation": evaluation_snapshot,
                            "source_report": report_snapshot},
            "dispatches": trace_dispatches,
            "aggregates": {"dispatch_count": len(trace_dispatches),
                           "independent_device_service_time_ns": 100 * len(trace_dispatches)},
        }), encoding="utf-8")
        schedule_path = root / "schedule.json"
        schedule_path.write_text("{}", encoding="utf-8")
        canonical_timing = json.dumps(
            json.loads(timing_path.read_text(encoding="utf-8")),
            sort_keys=True, separators=(",", ":")).encode()
        reconciliation_path = root / "reconciliation.json"
        reconciliation_path.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_8_27b_dispatch_reconciliation",
            "schema_version": 1,
            "trace_authority": {"path": str(trace_path.resolve()),
                                "file_size_bytes": trace_path.stat().st_size,
                                "sha256": sha256(trace_path)},
            "static_schedule": {"path": str(schedule_path.resolve()),
                                "file_size_bytes": schedule_path.stat().st_size,
                                "sha256": sha256(schedule_path)},
            "schedule_sources": [{"path": str(analyzer_path.resolve()),
                                  "file_size_bytes": analyzer_path.stat().st_size,
                                  "sha256": sha256(analyzer_path)}],
            "workload": workload,
            "unprofiled_performance": {
                "prompt_tokens": 2048, "prefill_tok_s_mean": 2048.0,
                "prefill_seconds_mean": 1.0,
                "low_context_evaluation": evaluation_snapshot,
                "p2048_report": report_snapshot,
            },
            "dispatches": coverage,
            "coverage": {
                "trace_dispatch_count": len(coverage),
                "trace_service_time_ns": 100 * len(coverage),
                "modeled": {"dispatch_count": len(coverage),
                            "duration_ns": 100 * len(coverage)},
                "unmodeled": {"dispatch_count": 0, "duration_ns": 0},
                "unsupported": {"dispatch_count": 0, "duration_ns": 0},
            },
            "timing_authority": json.loads(timing_path.read_text(encoding="utf-8")),
            "dispatch_inventory": {
                **json.loads(inventory_path.read_text(encoding="utf-8")),
                "timing_authority_sha256": hashlib.sha256(canonical_timing).hexdigest(),
            },
        }), encoding="utf-8")
        return reconciliation_path, inventory_path, keep_path

    def test_q4_w8_real_shape_accounting_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operations = [
                ("q4_linear", "mlp_gate_up",
                 {"tokens": 2048, "rows": 34816, "columns": 5120}),
                ("w8_linear", "attention_gate_value",
                 {"tokens": 2048, "rows": 7168, "columns": 5120}),
            ]
            timing, inventory, _ = self.fixture(root, operations, overlap=True)
            result = model(timing)
            rows = {row["operation"]: row for row in result["dispatches"]}
            q4 = requested_traffic(Shape(2048, 34816, 5120), M64_N64)
            w8 = requested_w8_traffic(Shape(2048, 7168, 5120), W8_M64_N64)
            self.assertEqual(rows["q4_linear"]["counts"]["logical_ops"], q4.logical_ops)
            self.assertEqual(rows["q4_linear"]["counts"]["issued_ops"]["iu4"],
                             q4.issued_iu4_ops)
            self.assertEqual(rows["q4_linear"]["counts"]["source_requested_bytes"],
                             q4.total_requested_bytes)
            self.assertEqual(rows["w8_linear"]["counts"]["issued_ops"]["iu8"],
                             w8.issued_iu8_ops)
            self.assertEqual(result["timing"]["independent_device_service_time_ns"], 200)
            self.assertEqual(result["timing"]["device_wall_union_ns"], 150)
            self.assertIsNone(result["physical_measurement"]["hbm_bandwidth_gbps"])

    def test_dense_causal_attention_formula(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timing, inventory, _ = self.fixture(Path(directory), [
                ("dense_attention", "full_attention", {
                    "tokens": 2, "context_start": 3, "query_heads": 2, "kv_heads": 1,
                    "head_dimension": 4, "value_group": 2,
                    "represented_metadata_bytes": 7, "source_metadata_bytes": 11,
                })])
            row = model(timing)["dispatches"][0]
            # Visible lengths 4 and 5: QK plus PV each count two FLOPs per D.
            self.assertEqual(row["counts"]["logical_ops"], 4 * 2 * 4 * 9)
            self.assertEqual(row["counts"]["represented_minimum_bytes"],
                             2*2*2*4 + 5*1*4 + 5*1*4//2 + 2*5*1*2 + 4*2*2*4 + 7)
            self.assertEqual(
                row["counts"]["source_requested_bytes"],
                2*2*2*4 + 2*9*(4 + 4 + 2*4) + 4*2*2*4 + 11,
            )
            self.assertEqual(row["counts"]["special_functions"]["softmax_exp_arguments"], 18)

    def test_sparse_pack_rank_and_keep_derived_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep_rows = {"d0": {"retained_key_counts_per_query": [1, 2, 3, 4],
                                 "represented_unique_kv_token_heads": 6,
                                 "consumer_tiles": [{"query_rows": 2, "key_rows": 16},
                                                    {"query_rows": 1, "key_rows": 8}]}}
            consumer = _counts("sparse_consumer", {
                "tokens": 2, "query_heads": 2, "kv_heads": 1, "head_dimension": 4,
                "value_group": 2, "represented_metadata_bytes": 0,
                "source_metadata_bytes": 0}, "d0", keep_rows)
            self.assertEqual(consumer["logical_ops"], 4 * 4 * 10)
            self.assertEqual(consumer["issued_ops"]["bf16_matrix"], 2*16*16*4*2)
            self.assertEqual(consumer["issued_ops"]["fp32_vector_arithmetic"], 2*4*10)

            pack = _counts("sparse_pack", {"context_tokens": 8, "kv_heads": 1,
                "head_dimension": 4, "represented_metadata_bytes": 0,
                "source_metadata_bytes": 0}, "pack", {})
            rank = _counts("sparse_rank", {
                "head_dimension": 4, "logical_qk_pairs": 20,
                "issued_wmma_16x16_tiles": 3, "represented_query_elements": 8,
                "represented_packed_key_elements": 32, "represented_scratch_bytes": 24,
                "source_query_elements": 16, "source_packed_key_elements": 64,
                "source_scratch_read_bytes": 11, "source_scratch_write_bytes": 13,
                "represented_metadata_bytes": 5, "source_metadata_bytes": 7,
                "estimator_exp_arguments": 7}, "rank", {})
            self.assertEqual(pack["represented_minimum_bytes"], 96)
            self.assertEqual(rank["logical_ops"], 160)
            self.assertEqual(rank["issued_ops"]["bf16_matrix"],
                             2*16*16*4*3)

    def test_gdn_and_elementwise_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            operations = [
                ("gdn_recurrence", "ordinary", {"tokens": 2, "query_heads": 2,
                    "value_heads": 4, "dimension": 8, "row_tiles": 4}),
                ("gdn_control", "control", {"tokens": 2, "value_heads": 4,
                                             "softplus_nonlinear_elements": 7}),
                ("gdn_conv", "conv", {"tokens": 2, "channels": 8}),
                ("residual_add", "residual", {"elements": 16}),
                ("silu_mul", "activation", {"elements": 16}),
                ("rmsnorm", "norm", {"elements": 16, "features": 8}),
                ("residual_rmsnorm", "residual_norm", {"elements": 16, "features": 8}),
                ("gated_rmsnorm", "gated_norm", {"elements": 16, "features": 8}),
            ]
            timing, inventory, _ = self.fixture(Path(directory), operations)
            rows = {row["operation"]: row for row in model(timing)["dispatches"]}
            self.assertEqual(rows["gdn_recurrence"]["counts"]["logical_ops"],
                             2*4*8*(7*8+4))
            self.assertEqual(rows["gdn_control"]["counts"]["logical_ops"], 0)
            self.assertEqual(rows["gdn_conv"]["counts"]["logical_ops"], 8*2*8)
            self.assertEqual(rows["residual_add"]["counts"]["represented_minimum_bytes"], 96)
            self.assertEqual(rows["silu_mul"]["counts"]["special_functions"],
                             {"exp_arguments": 16})
            self.assertEqual(rows["rmsnorm"]["counts"]["represented_minimum_bytes"], 80)
            self.assertEqual(rows["residual_rmsnorm"]["counts"]["represented_minimum_bytes"],
                             144)
            self.assertEqual(rows["gated_rmsnorm"]["counts"]["logical_ops"], 132)
            self.assertEqual(rows["gated_rmsnorm"]["counts"]["represented_minimum_bytes"],
                             112)
            self.assertEqual(rows["gated_rmsnorm"]["counts"]["source_requested_bytes"], 160)
            self.assertEqual(rows["gated_rmsnorm"]["counts"]["special_functions"],
                             {"rsqrt_arguments": 2, "exp_arguments": 16})

    def test_rejects_ambiguous_or_unbound_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timing, inventory, _ = self.fixture(root, [
                ("q4_linear", "projection", {"tokens": 64, "rows": 64, "columns": 64})])
            value = json.loads(timing.read_text(encoding="utf-8"))
            value["dispatch_inventory"]["dispatches"][0]["symbol"] = "wrong"
            timing.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory row d0 differs"):
                model(timing)

            timing2, inventory2, _ = self.fixture(root / "nonauto", [
                ("q4_linear", "projection", {"tokens": 64, "rows": 64, "columns": 64})],
                auto=False)
            with self.assertRaisesRegex(ValueError, "required/observed/rechecked_after auto"):
                model(timing2)

            timing3, inventory3, _ = self.fixture(root / "sparse", [
                ("sparse_consumer", "consumer", {"tokens": 1, "query_heads": 1,
                    "kv_heads": 1, "head_dimension": 4, "value_group": 2,
                    "represented_metadata_bytes": 0, "source_metadata_bytes": 0})])
            with self.assertRaisesRegex(ValueError, "dense C1 P2048 selected route"):
                model(timing3)

    def test_probe_reference_is_bound_and_source_hashes_are_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timing, inventory, _ = self.fixture(root, [
                ("residual_add", "residual", {"elements": 16})])
            probe = root / "probe.json"
            probe.write_text(json.dumps({
                "artifact_type": "ninfer_r9700_memory_probe", "schema_version": 1,
                "read_gbps": 636.0, "write_gbps": 588.0, "copy_gbps": 549.0,
            }), encoding="utf-8")
            result = model(timing, probe_path=probe)
            reference = result["peak_references"]["supplied_memory_probe_reference"]
            self.assertEqual(reference["sha256"], sha256(probe))
            self.assertEqual(
                result["dispatches"][0]["rates"][
                    "represented_to_supplied_copy_probe_ratio"],
                (96 / (100 / 1e9) / 1e9) / 549.0,
            )
            self.assertIn(
                "not proof of same-session collection",
                result["peak_references"]["interpretation"],
            )

            timing_value = json.loads(timing.read_text(encoding="utf-8"))
            analyzer = Path(timing_value["schedule_sources"][0]["path"])
            analyzer.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"schedule_sources\[0\].*changed"):
                model(timing)

    def test_atomic_reconciliation_covers_unmodeled_and_separates_whole_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reconciliation, _, _ = self.fixture(Path(directory), [
                ("residual_add", "residual", {"elements": 16}),
                ("silu_mul", "activation", {"elements": 16}),
            ])
            value = json.loads(reconciliation.read_text(encoding="utf-8"))
            uncovered = value["dispatches"][1]
            uncovered["classification"] = "unmodeled"
            uncovered["reason"] = "no accepted logical accounting contract"
            for key in ("role", "operation", "format", "parameters"):
                uncovered.pop(key)
            value["timing_authority"]["dispatches"] = value["timing_authority"]["dispatches"][:1]
            value["dispatch_inventory"]["dispatches"] = value["dispatch_inventory"]["dispatches"][:1]
            canonical = json.dumps(value["timing_authority"], sort_keys=True,
                                   separators=(",", ":")).encode()
            value["dispatch_inventory"]["timing_authority_sha256"] = hashlib.sha256(
                canonical).hexdigest()
            value["coverage"].update({
                "modeled": {"dispatch_count": 1, "duration_ns": 100},
                "unmodeled": {"dispatch_count": 1, "duration_ns": 100},
            })
            reconciliation.write_text(json.dumps(value), encoding="utf-8")
            result = model(reconciliation)
            self.assertEqual(result["timing"]["profiled_dispatch_count"], 1)
            self.assertEqual(result["timing"]["uncovered_dispatch_count"], 1)
            self.assertEqual(result["dispatch_coverage"]["unmodeled"],
                             {"dispatch_count": 1, "duration_ns": 100})
            self.assertEqual(result["uncovered_dispatches"][0]["reason"],
                             "no accepted logical accounting contract")
            self.assertEqual(result["unprofiled_whole_p2048"]["prefill_tok_s_mean"], 2048.0)
            self.assertIn("never profiler timing",
                          result["unprofiled_whole_p2048"]["authority"])

            value["coverage"]["trace_dispatch_count"] = True
            reconciliation.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage.trace_dispatch_count"):
                model(reconciliation)
            value["coverage"]["trace_dispatch_count"] = 2
            value["dispatches"] = value["dispatches"][:1]
            value["coverage"].update({"trace_dispatch_count": 1, "trace_service_time_ns": 100,
                                      "unmodeled": {"dispatch_count": 0, "duration_ns": 0}})
            reconciliation.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover every validated trace dispatch"):
                model(reconciliation)


if __name__ == "__main__":
    unittest.main()
