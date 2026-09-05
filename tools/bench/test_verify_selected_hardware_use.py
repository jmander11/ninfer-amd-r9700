#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.bench.verify_selected_hardware_use import sha, snapshot, verify


Q4_CTA = "_Z_a8q4g64_linear_prefill_cta_kernel"
Q4_WAVE = "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_128a8q4g64_linear_wmma32_kernelEPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjj"
W8_CTA = "_Z_a8w8g32_linear_prefill_cta_kernel"
DENSE_QK = "_Z_dense_full_score_qk_kernel"
XRANK = "_Z_xattention_rank_kernel"
XCONSUMER = "_Z_xattention_flash_consumer_kernel"
FP8_GATE = "Cijk_fp8_gate"
FP8_ATTN = "Cijk_fp8_attention"


class SelectedHardwareUseTest(unittest.TestCase):
    def test_prepared_verifier_module_imports_from_repo(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        script = (
            repo / "profiles/bench/post-terminal-selected-hardware-use-20260905/verify.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m tools.bench.verify_selected_hardware_use", script)
        plan = json.loads((
            repo / "profiles/bench/post-terminal-selected-hardware-use-20260905/plan.json"
        ).read_text(encoding="utf-8"))
        self.assertNotIn("--mtp-proof", script)
        self.assertNotIn("mixed_mtp_bulk_preparation", plan)
        self.assertNotIn("mixed_mtp_bulk_execution", plan)
        result = subprocess.run(
            [sys.executable, "-m", "tools.bench.verify_selected_hardware_use", "--help"],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def write(self, path: Path, value: object) -> Path:
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def fixture(self, root: Path, weights: str = "r9700-q4-w8-mse-n16k16-eval",
                attention: str = "dense") -> dict:
        root.mkdir(parents=True, exist_ok=True)
        selection = self.write(root / "selection.json", {"placeholder": True})
        artifact = root / "model.ninfer"; artifact.write_bytes(b"model")
        executable = root / "ninfer_bench"; executable.write_bytes(b"bench")
        route = {
            "terminal_selection": snapshot(selection), "winner": "winner",
            "weights_id": weights,
            "artifact": {**snapshot(artifact), "weights_id": weights},
            "executable": snapshot(executable), "kv_value_group": 16,
            "xattention_profile": attention, "prefill_chunk": 2048,
        }
        symbols = [Q4_CTA, Q4_WAVE,
                   W8_CTA if weights == "r9700-q4-w8-mse-n16k16-eval" else Q4_CTA,
                   DENSE_QK if attention == "dense" else XRANK]
        if attention != "dense": symbols.append(XCONSUMER)
        if weights == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
            symbols.extend((FP8_GATE, FP8_GATE, FP8_ATTN, FP8_ATTN))
        workload = {
            "artifact_sha256": route["artifact"]["sha256"],
            "executable_sha256": route["executable"]["sha256"],
            "weights_id": weights, "kv_value_group": 16,
            "xattention_profile": attention, "prefill_chunk": 2048, "concurrency": 1,
        }
        terminal_input = root / "terminal-input.json"
        terminal_input.write_bytes(selection.read_bytes())
        route["terminal_selection"] = snapshot(terminal_input)
        trace_database = root / "trace.db"; trace_database.write_bytes(b"database")
        trace = self.write(root / "trace.json", {
            "artifact_type": "ninfer_r9700_selected_profile_trace", "schema_version": 1,
            "status": "valid_attribution_only", "profile_timing_admissible": False,
            "selected_route": {
                "artifact": {"sha256": route["artifact"]["sha256"]},
                "benchmark_executable": {"sha256": route["executable"]["sha256"]},
            }, "inputs": {"terminal_selection": route["terminal_selection"],
                            "database": snapshot(trace_database)},
        })
        schedule = self.write(root / "schedule.json", {"fixed": True})
        fp8_roles = iter(("mlp_gate_up", "gdn_query_key",
                          "attention_query_key", "attention_gate_value"))
        inventory_rows = [{"dispatch_id": f"d{index}", "symbol": symbol,
                           "operation": "fp8_linear" if symbol in (FP8_GATE, FP8_ATTN)
                           else ("w8_linear" if symbol == W8_CTA else "other"),
                           "resources": ({"vgpr_count": 88, "lds_bytes": 17152,
                                          "scratch_bytes": 0} if symbol == Q4_CTA else
                                         {"vgpr_count": 50, "lds_bytes": 4352,
                                          "scratch_bytes": 0} if symbol == W8_CTA else
                                         {"sgpr_count": 128, "vgpr_count": 192,
                                          "accum_vgpr_count": 0,
                                          "lds_bytes": (25088 if symbol == FP8_GATE else 12544),
                                          "scratch_bytes": 0} if symbol in (FP8_GATE, FP8_ATTN)
                                         else {}),
                           "role": next(fp8_roles) if symbol in (FP8_GATE, FP8_ATTN)
                           else "other"}
                          for index, symbol in enumerate(symbols)]
        reconciliation = self.write(root / "reconciliation.json", {
            "artifact_type": "ninfer_qwen3_8_27b_dispatch_reconciliation",
            "schema_version": 1, "workload": workload, "trace_authority": snapshot(trace),
            "static_schedule": snapshot(schedule),
            "dispatches": [{**row, "classification": "modeled"} for row in inventory_rows],
            "dispatch_inventory": {
                "artifact_type": "ninfer_qwen3_8_27b_dispatch_inventory", "schema_version": 1,
                "workload": workload, "dispatches": inventory_rows},
        })
        profile = ("dense" if attention == "dense" else "xattention-s16-tau900") + "-g16"
        proofs = {
            "q4_p2048_cta": {"code_symbol": Q4_CTA,
                               "opcode": "v_wmma_i32_16x16x32_iu4", "opcode_sites": 8,
                               "vgpr": 88, "lds_bytes": 17152, "private_bytes": 0,
                               "scratch_bytes": 0},
            "q4_wave32": {"code_symbol": Q4_WAVE,
                            "opcode": "v_wmma_i32_16x16x32_iu4", "opcode_sites": 4,
                            "vgpr": 64, "lds_bytes": 0, "private_bytes": 0},
            "w8_p2048_cta": {"code_symbol": W8_CTA,
                              "opcode": "v_wmma_i32_16x16x16_iu8", "opcode_sites": 2,
                              "vgpr": 50, "lds_bytes": 4352, "private_bytes": 0,
                              "scratch_bytes": 0},
            "ordinary_fp8_qk": {"fp8_wmma_opcode": "v_wmma_f32_16x16x16_fp8_fp8",
                                  "fp8_wmma_sites": 1,
                                  "packed_fp8_conversion_opcode": "v_cvt_pk_fp8_f32",
                                  "packed_fp8_conversion_sites": 8},
            "dense_initial_prefix_qk": {"code_symbol": DENSE_QK,
                                         "opcode": "v_wmma_f32_16x16x16_bf16",
                                         "opcode_sites": 16},
            "xattention_rank": {"code_symbol": XRANK,
                                  "opcode": "v_wmma_f32_16x16x16_bf16",
                                  "opcode_sites": 2},
            "xattention_flash_consumer": {"g16_code_symbol": XCONSUMER,
                                            "opcode": "v_wmma_f32_16x16x16_bf16",
                                            "opcode_sites_each": 16},
        }
        audit = self.write(root / "audit.json", {
            "schema": "ninfer.r9700.twelve_candidate_hardware_path_static_audit.v1",
            "status": "static_preflight_pass_terminal_dispatch_proof_pending",
            "benchmark_executables": [{"profile": profile,
                                        "sha256": route["executable"]["sha256"]}],
            "shared_symbol_proofs": proofs,
        })
        fp8 = []
        loaded_fp8 = []
        if weights == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
            capture_library = root / "capture.so"; capture_library.write_bytes(b"capture")
            capture_dir = root / "captured"; capture_dir.mkdir()
            for qualification, kernel, count, schema in (
                ("gate_up", FP8_GATE, 112, "ninfer.r9700.fp8_gate_up_hardware_proof.v1"),
                ("attention_qk_gate_value", FP8_ATTN, 160,
                 "ninfer.r9700.fp8_attention_qk_gate_value_hardware_proof.v1"),
            ):
                loaded = root / f"{qualification}.co"; loaded.write_bytes(kernel.encode())
                auxiliaries = {}
                for name in ("plan", "matched_report", "traced_report", "trace_database", "objdump"):
                    file = root / f"{qualification}-{name}"; file.write_bytes(name.encode())
                    auxiliaries[name] = {"path": str(file), "bytes": file.stat().st_size,
                                         "sha256": sha(file)}
                proof = self.write(root / f"{qualification}.json", {
                    "schema": schema, "qualification": qualification, "pass": True,
                    "fp8": {"kernel_name": kernel, "native_fp8_matrix_opcode_count": count,
                            "resources": {
                                "sgpr_count": 128, "architectural_vgpr_count": 192,
                                "accum_vgpr_count": 0,
                                "group_segment_lds_bytes": (
                                    25088 if qualification == "gate_up" else 12544),
                                "private_segment_bytes": 0,
                                "spill_counts": {"available": False, "sgpr": None,
                                                 "vgpr": None},
                            },
                            "code_object": {"sha256": sha(loaded),
                                            "source": {"path": str(loaded),
                                                       "bytes": loaded.stat().st_size,
                                                       "sha256": sha(loaded)}}},
                    **auxiliaries,
                })
                fp8.append(proof)
                loaded_fp8.append({
                    "kernel_name": kernel, "kernel_id": len(loaded_fp8) + 11,
                    "code_object_id": len(loaded_fp8) + 1,
                    "dispatch_ids": [row["dispatch_id"] for row in inventory_rows
                                     if row["symbol"] == kernel],
                    "resources": {
                        "sgpr_count": 128, "architectural_vgpr_count": 192,
                        "accum_vgpr_count": 0,
                        "group_segment_lds_bytes": (
                            25088 if qualification == "gate_up" else 12544),
                        "private_segment_bytes": 0,
                        "spill_counts": {"available": False, "sgpr": None,
                                         "vgpr": None},
                    },
                    "code_object": {"uri": f"memory://fixture/{qualification}",
                                    "source": snapshot(loaded),
                                    "sha256": sha(loaded), "bytes": loaded.stat().st_size},
                })
            trace_value = json.loads(trace.read_text())
            trace_value["loaded_fp8_code_objects"] = loaded_fp8
            trace_value["code_object_capture"] = {
                "library": snapshot(capture_library), "directory": str(capture_dir),
                "scope": "selected hybrid workload loaded code objects"}
            self.write(trace, trace_value)
            reconciliation_value = json.loads(reconciliation.read_text())
            reconciliation_value["trace_authority"] = snapshot(trace)
            self.write(reconciliation, reconciliation_value)
        return {"selection": selection, "route": route, "trace": trace,
                "reconciliation": reconciliation,
                "audit": audit, "fp8": fp8}

    def run_fixture(self, fixture: dict) -> dict:
        return verify(fixture["selection"], fixture["reconciliation"], fixture["audit"],
                      fixture["fp8"],
                      route_resolver=lambda _: fixture["route"],
                      reconciliation_validator=lambda _trace, _schedule:
                      json.loads(fixture["reconciliation"].read_text()),
                      loaded_fp8_validator=lambda _database, _capture, _dispatches:
                      json.loads(fixture["trace"].read_text()).get(
                          "loaded_fp8_code_objects", []),
                      embedded_validator=lambda *_args: {"validated": True},
                      fp8_resource_validator=lambda value: value["fp8"]["resources"])

    def test_mixed_dense_passes_and_rejects_missing_iu8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            result = self.run_fixture(fixture)
            self.assertEqual(result["status"], "passed")
            self.assertIsNone(result["physical_bandwidth_claim"])
            self.assertEqual(result["static_proof_names"].count("q4_wave32"), 1)
            self.assertEqual(set(result["authorities"]),
                             {"dispatch_reconciliation", "trace", "static_audit"})
            value = json.loads(fixture["reconciliation"].read_text())
            value["dispatch_inventory"]["dispatches"] = [
                row for row in value["dispatch_inventory"]["dispatches"]
                if W8_CTA not in row["symbol"]]
            value["dispatches"] = [row for row in value["dispatches"]
                                     if W8_CTA not in row["symbol"]]
            self.write(fixture["reconciliation"], value)
            with self.assertRaisesRegex(ValueError, "W8 dispatch presence"):
                self.run_fixture(fixture)

    def test_hybrid_sparse_requires_both_executed_loaded_elf_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
                                   "b128-s16-tau900")
            result = self.run_fixture(fixture)
            self.assertEqual(len(result["loaded_fp8_proofs"]), 2)
            with self.assertRaisesRegex(ValueError, "current DB/URI capture"):
                verify(fixture["selection"], fixture["reconciliation"], fixture["audit"],
                       fixture["fp8"],
                       route_resolver=lambda _: fixture["route"],
                       reconciliation_validator=lambda _trace, _schedule:
                       json.loads(fixture["reconciliation"].read_text()),
                       loaded_fp8_validator=lambda _database, _capture, _dispatches: [],
                       embedded_validator=lambda *_args: {"validated": True},
                       fp8_resource_validator=lambda value: value["fp8"]["resources"])
            with self.assertRaisesRegex(ValueError, "exactly two"):
                verify(fixture["selection"], fixture["reconciliation"], fixture["audit"],
                       fixture["fp8"][:1],
                       route_resolver=lambda _: fixture["route"],
                       reconciliation_validator=lambda _trace, _schedule:
                       json.loads(fixture["reconciliation"].read_text()),
                       loaded_fp8_validator=lambda _database, _capture, _dispatches:
                       json.loads(fixture["trace"].read_text()).get(
                           "loaded_fp8_code_objects", []),
                       embedded_validator=lambda *_args: {"validated": True},
                       fp8_resource_validator=lambda value: value["fp8"]["resources"])

            proof = json.loads(fixture["fp8"][0].read_text())
            proof["fp8"]["resources"]["architectural_vgpr_count"] = 191
            self.write(fixture["fp8"][0], proof)
            with self.assertRaisesRegex(ValueError, "exact FP8 resource envelope"):
                self.run_fixture(fixture)

    def test_hybrid_rejects_selected_dispatch_resource_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory),
                                   "r9700-q4g64-f8e4m3-four-role-n16k16-eval", "dense")
            value = json.loads(fixture["reconciliation"].read_text())
            rows = value["dispatch_inventory"]["dispatches"]
            next(row for row in rows if row["symbol"] == FP8_GATE)["resources"][
                "lds_bytes"] = 12544
            self.write(fixture["reconciliation"], value)
            with self.assertRaisesRegex(ValueError, "selected dispatch resources"):
                self.run_fixture(fixture)

    def test_rejects_concurrency_above_product_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            value = json.loads(fixture["reconciliation"].read_text())
            value["workload"]["concurrency"] = 5
            value["dispatch_inventory"]["workload"]["concurrency"] = 5
            self.write(fixture["reconciliation"], value)
            with self.assertRaisesRegex(ValueError, r"\[1,4\]"):
                self.run_fixture(fixture)

    def test_rejects_unrecomputed_inventory_and_missing_executed_q4_wave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "differs from current trace/schedule"):
                verify(
                    fixture["selection"], fixture["reconciliation"], fixture["audit"],
                    fixture["fp8"],
                    route_resolver=lambda _: fixture["route"],
                    reconciliation_validator=lambda _trace, _schedule: {},
                )
            value = json.loads(fixture["reconciliation"].read_text())
            value["dispatch_inventory"]["dispatches"] = [
                row for row in value["dispatch_inventory"]["dispatches"]
                if Q4_WAVE not in row["symbol"]]
            value["dispatches"] = [row for row in value["dispatches"]
                                   if Q4_WAVE not in row["symbol"]]
            self.write(fixture["reconciliation"], value)
            with self.assertRaisesRegex(ValueError, "required symbol"):
                self.run_fixture(fixture)

    def test_rejects_recipe_inconsistent_operation_and_conditional_proof_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary = self.fixture(root / "ordinary", "r9700-q4g64-mse-eval")
            value = json.loads(ordinary["reconciliation"].read_text())
            value["dispatch_inventory"]["dispatches"][0]["operation"] = "fp8_linear"
            self.write(ordinary["reconciliation"], value)
            with self.assertRaisesRegex(ValueError, "FP8-linear dispatch presence"):
                self.run_fixture(ordinary)

            hybrid = self.fixture(root / "hybrid",
                                  "r9700-q4g64-f8e4m3-four-role-n16k16-eval")
            audit = json.loads(hybrid["audit"].read_text())
            audit["shared_symbol_proofs"]["q4_wave32"]["opcode_sites"] = 3
            self.write(hybrid["audit"], audit)
            with self.assertRaisesRegex(ValueError, "Q4 wave32 static proof"):
                self.run_fixture(hybrid)


if __name__ == "__main__":
    unittest.main()
