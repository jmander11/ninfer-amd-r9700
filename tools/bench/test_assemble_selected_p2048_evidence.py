#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bench.assemble_selected_p2048_evidence import (
    _publish, _snapshot, assemble, main, validate,
)
from tools.bench.prepare_whole_profile import DISPATCH_COUNTERS


class SelectedP2048EvidenceTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, list[Path]]:
        root.mkdir(parents=True, exist_ok=True)
        artifact = root / "selected.ninfer"
        executable = root / "ninfer_bench"
        source = root / "kernel.hip"
        artifact.write_bytes(b"artifact")
        executable.write_bytes(b"executable")
        source.write_text("selected specialization", encoding="utf-8")
        artifact_id, executable_id = _snapshot(artifact, "artifact"), _snapshot(
            executable, "executable")
        workload = {
            "kind": "pp", "prompt_tokens": 2048, "concurrency": 1,
            "prefill_chunk": 2048, "kv_value_group": 16, "xattention_profile": "dense",
            "artifact_path": artifact_id["path"], "artifact_sha256": artifact_id["sha256"],
            "weights_id": "selected", "executable_path": executable_id["path"],
            "executable_sha256": executable_id["sha256"],
        }
        dispatches = [
            {"dispatch_id": "trace:1", "marker": "prefill/text/linear",
             "symbol": "q4_prefill", "duration_ns": 80, "classification": "modeled",
             "operation": "q4_linear"},
            {"dispatch_id": "trace:2", "marker": "prefill/orchestration",
             "symbol": "bonus_sample", "duration_ns": 20, "classification": "unsupported",
             "reason": "not a logical roofline operation"},
        ]
        coverage = {
            "trace_dispatch_count": 2, "trace_service_time_ns": 100,
            "modeled": {"dispatch_count": 1, "duration_ns": 80},
            "unmodeled": {"dispatch_count": 0, "duration_ns": 0},
            "unsupported": {"dispatch_count": 1, "duration_ns": 20},
        }
        reconciliation = root / "reconciliation.json"
        reconciliation.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_8_27b_dispatch_reconciliation",
            "schema_version": 1, "workload": workload,
            "timing_authority": {"power_profile": {
                "required": "auto", "observed": "auto", "rechecked_after": "auto"}},
            "dispatches": dispatches, "coverage": coverage,
        }), encoding="utf-8")
        reconciliation_id = _snapshot(reconciliation, "reconciliation")
        roofline = root / "roofline.json"
        roofline.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_8_27b_roofline", "schema_version": 1,
            "dispatch_reconciliation": reconciliation_id, "workload": workload,
            "dispatch_coverage": coverage,
            "dispatches": [{"dispatch_id": "trace:1"}],
            "uncovered_dispatches": [{"dispatch_id": "trace:2"}],
            "unprofiled_whole_p2048": {"prompt_tokens": 2048,
                "prefill_tok_s_mean": 2100.0, "prefill_seconds_mean": 0.975},
            "physical_measurement": {"hbm_bytes": None, "hbm_bandwidth_gbps": None,
                "hbm_peak_fraction": None, "stall_fraction": None},
        }), encoding="utf-8")
        pmc = root / "pmc.json"
        pmc.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_selected_profile_pmc", "schema_version": 1,
            "status": "valid_attribution_only", "profile_timing_admissible": False,
            "workload": workload,
            "inputs": {"artifact": artifact_id, "benchmark_executable": executable_id},
            "power_profile": {"required": "profile_standard", "before": "profile_standard",
                              "after": "auto"},
            "roctx_stage_join": {"state": "exact_same_capture", "dispatches": [
                {"dispatch_id": 101, "stage": "prefill/text/linear", "symbol": "q4_prefill"},
                {"dispatch_id": 102, "stage": "prefill/orchestration", "symbol": "bonus_sample"},
            ], "stages": [
                {"stage": "prefill/text/linear", "dispatch_count": 1,
                 "counter_sums": {name: "20" for name in DISPATCH_COUNTERS},
                 "gl2_hit_ratio": {"state": "measured", "value": 0.8},
                 "tcp_hit_ratio": {"state": "measured", "value": 0.7}},
                {"stage": "prefill/orchestration", "dispatch_count": 1,
                 "counter_sums": {name: "4" for name in DISPATCH_COUNTERS},
                 "gl2_hit_ratio": {"state": "measured", "value": 0.5},
                 "tcp_hit_ratio": {"state": "measured", "value": 0.4}},
            ]},
        }), encoding="utf-8")
        route = {field: workload[field] for field in (
            "prompt_tokens", "concurrency", "prefill_chunk", "kv_value_group",
            "xattention_profile")}
        route.update({"artifact_sha256": artifact_id["sha256"],
                      "executable_sha256": executable_id["sha256"]})

        def static(name: str, operation: str, stage: str, symbol: str,
                   classification: str, arithmetic: str, opcode: str) -> Path:
            path = root / f"static-{name}.json"
            path.write_text(json.dumps({
                "artifact_type": "ninfer_r9700_operation_static_evidence",
                "schema_version": 1, "evidence_id": name, "selected_route": route,
                "operation_family": operation, "specialization": f"{name}-gfx1201",
                "dispatch_signatures": [{"stage": stage, "symbol": symbol}],
                "intended_hardware": {"classification": classification,
                    "arithmetic": arithmetic, "expected_opcodes": [opcode]},
                "static_proof": {"status": "passed", "code_symbol": symbol,
                    "executable_embedding": {"offset_bytes": 0,
                        "file_size_bytes": executable.stat().st_size,
                        "occurrence_count": 1}, "inputs": {
                    "code_object": _snapshot(executable, "code object"),
                    "assembly": _snapshot(source, "assembly"),
                    "metadata": _snapshot(source, "metadata"),
                    "checker": _snapshot(source, "checker"),
                    "sources": [_snapshot(source, "source")]},
                    "opcode_counts": {opcode: 2}, "resources": {
                        "lds_bytes": 1024, "vgpr_count": 32, "private_bytes": 0,
                        "scratch_bytes": 0, "flat_scratch": 0}, "zero_scratch": True},
                "memory_path": {"residency": "lds_reuse" if classification == "matrix"
                                                else "metadata_control",
                    "access_pattern": "coalesced staged reads" if classification == "matrix"
                                      else "bounded scalar control",
                    "overlap": "unproven" if classification == "matrix" else "not_applicable",
                    "evidence": "static source and disassembly classification"},
            }), encoding="utf-8")
            return path

        static_paths = [
            static("q4", "q4_linear", "prefill/text/linear", "q4_prefill",
                   "matrix", "signed_int4_matrix", "v_wmma_i32_16x16x32_iu4"),
            static("sampling", "categorical_sampling", "prefill/orchestration", "bonus_sample",
                   "memory_control", "integer_control", "s_endpgm"),
        ]
        return reconciliation, roofline, pmc, static_paths

    def test_complete_assembly_separates_timing_and_stage_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reconciliation, roofline, pmc, static_paths = self.fixture(Path(directory))
            result = assemble(reconciliation, roofline, pmc, static_paths)
            self.assertEqual(result["status"], "assembled_with_unproven_overlap")
            self.assertEqual(result["timing_semantics"]["selection_performance"]
                             ["authority"]["prefill_tok_s_mean"], 2100.0)
            self.assertIn("attribution only",
                          result["timing_semantics"]["dispatch_attribution"])
            self.assertEqual(result["pmc_stage_attribution"][0]["join_basis"],
                             "same-capture ROCTX stage plus exact profiler display-symbol "
                             "multiplicity; never cross-capture dispatch ID")
            self.assertIsNone(result["physical_measurement"]["hbm_bytes"])

    def test_rejects_silent_dispatch_subset_and_missing_static_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reconciliation, roofline, pmc, static_paths = self.fixture(Path(directory))
            value = json.loads(roofline.read_text(encoding="utf-8"))
            value["uncovered_dispatches"] = []
            roofline.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modeled/uncovered dispatch coverage"):
                assemble(reconciliation, roofline, pmc, static_paths)
            value["uncovered_dispatches"] = [{"dispatch_id": "trace:2"}]
            roofline.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks intended-hardware static evidence"):
                assemble(reconciliation, roofline, pmc, static_paths[:1])

    def test_rejects_weak_static_proof_and_cross_stage_pmc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reconciliation, roofline, pmc, static_paths = self.fixture(Path(directory))
            value = json.loads(static_paths[0].read_text(encoding="utf-8"))
            value["static_proof"]["resources"]["scratch_bytes"] = 1
            static_paths[0].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero private/scratch proof"):
                assemble(reconciliation, roofline, pmc, static_paths)

            reconciliation, roofline, pmc, static_paths = self.fixture(
                Path(directory) / "embedding")
            value = json.loads(static_paths[0].read_text(encoding="utf-8"))
            value["static_proof"]["executable_embedding"]["offset_bytes"] = 1
            static_paths[0].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not uniquely embedded"):
                assemble(reconciliation, roofline, pmc, static_paths)

            reconciliation, roofline, pmc, static_paths = self.fixture(
                Path(directory) / "second")
            value = json.loads(pmc.read_text(encoding="utf-8"))
            value["roctx_stage_join"]["stages"][0]["stage"] = "different-run-stage"
            pmc.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                assemble(reconciliation, roofline, pmc, static_paths)

    def test_rejects_pmc_symbol_or_multiplicity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reconciliation, roofline, pmc, static_paths = self.fixture(Path(directory))
            value = json.loads(pmc.read_text(encoding="utf-8"))
            value["roctx_stage_join"]["dispatches"][0]["symbol"] = "other_specialization"
            pmc.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace/static signature ownership"):
                assemble(reconciliation, roofline, pmc, static_paths)

            reconciliation, roofline, pmc, static_paths = self.fixture(
                Path(directory) / "multiplicity")
            value = json.loads(pmc.read_text(encoding="utf-8"))
            value["roctx_stage_join"]["dispatches"].append(
                {"dispatch_id": 103, "stage": "prefill/text/linear", "symbol": "q4_prefill"})
            value["roctx_stage_join"]["stages"][0]["dispatch_count"] = 2
            pmc.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "signature multiplicity differs"):
                assemble(reconciliation, roofline, pmc, static_paths)

    def test_rejects_two_code_specializations_for_one_dispatch_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reconciliation, roofline, pmc, static_paths = self.fixture(Path(directory))
            first = json.loads(static_paths[0].read_text(encoding="utf-8"))
            second = json.loads(static_paths[1].read_text(encoding="utf-8"))
            self.assertNotEqual(first["static_proof"]["code_symbol"],
                                second["static_proof"]["code_symbol"])
            second["dispatch_signatures"] = first["dispatch_signatures"]
            static_paths[1].write_text(json.dumps(second), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple static owners"):
                assemble(reconciliation, roofline, pmc, static_paths)

    def test_atomic_publish_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            _publish(path, {"first": True})
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _publish(path, {"first": False})
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                main(["--dispatch-reconciliation", "missing-reconciliation",
                      "--roofline", "missing-roofline", "--pmc", "missing-pmc",
                      "--static-report", "missing-static", "--out", str(path)])

    def test_final_validator_reconstructs_and_rejects_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reconciliation, roofline, pmc, static_paths = self.fixture(Path(directory))
            record = Path(directory) / "selected-evidence.json"
            record.write_text(json.dumps(assemble(reconciliation, roofline, pmc, static_paths)),
                              encoding="utf-8")
            self.assertEqual(validate(record, reconciliation, roofline, pmc, static_paths)
                             ["artifact_type"], "ninfer_r9700_selected_p2048_evidence")
            value = json.loads(record.read_text(encoding="utf-8"))
            value["physical_measurement"]["hbm_bytes"] = 1
            record.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from exact reconstructed authority"):
                validate(record, reconciliation, roofline, pmc, static_paths)


if __name__ == "__main__":
    unittest.main()
