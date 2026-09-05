#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.bench.assemble_selected_p2048_evidence import _static_report
from tools.bench.produce_operation_static_evidence import _publish, produce
from tools.bench.produce_operation_static_evidence import CTA_RECIPES
from tools.r9700.check_prefill_cta_static import PROFILES


class OperationStaticEvidenceProducerTest(unittest.TestCase):
    def files(self, root: Path, symbol: str, opcode: str, count: int,
              *, scratch: int = 0) -> tuple[Path, Path, Path, Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        artifact, executable, source = root / "model.ninfer", root / "ninfer_bench", root / "op.hip"
        artifact.write_bytes(b"artifact")
        executable.write_bytes(b"executable")
        source.write_text("source", encoding="utf-8")
        opcode_lines = "".join(f"\t{opcode} v0, v1, v2\n" for _ in range(count))
        body = f"""\t.globl {symbol} ; -- Begin function {symbol}
{symbol}:
{opcode_lines}
\t.amdhsa_kernel {symbol}
\t\t.amdhsa_group_segment_fixed_size 128
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_next_free_vgpr 24
\t.end_amdhsa_kernel
\t.set .Lselected.uses_flat_scratch, 0
; ScratchSize: {scratch}
"""
        assembly, metadata = root / "assembly.s", root / "metadata.s"
        assembly.write_text(body, encoding="utf-8")
        metadata.write_text(body, encoding="utf-8")
        return artifact, executable, source, assembly, metadata

    def args(self, root: Path, *, classification: str = "scalar_valu",
             opcode: str = "v_fma_f32", count: int = 2, scratch: int = 0) -> dict:
        artifact, executable, source, assembly, metadata = self.files(
            root, "scalar_kernel", opcode, count, scratch=scratch)
        return dict(
            artifact=artifact, executable=executable, code_object=executable,
            assembly=assembly, metadata=metadata,
            sources=[source], stage="prefill/text/gdn", dispatch_symbol="scalar_kernel",
            code_symbol="scalar_kernel",
            operation_family="gdn_recurrence", specialization="gdn-scalar-gfx1201",
            hardware_classification=classification, arithmetic="fp32_valu",
            opcodes=[(opcode, count)], max_lds_bytes=1024, max_vgpr_count=32,
            memory_residency="register_reuse", memory_access="state rows remain in registers",
            overlap="not_applicable", memory_evidence="source and selected disassembly",
            overlap_evidence=None, prefill_chunk=2048, kv_value_group=16,
            xattention_profile="dense", cta_recipe=None)

    def test_scalar_valu_report_has_exact_schema_without_false_wmma(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = produce(**self.args(Path(directory)))
            self.assertEqual(result["artifact_type"],
                             "ninfer_r9700_operation_static_evidence")
            self.assertEqual(result["intended_hardware"]["classification"], "scalar_valu")
            self.assertEqual(result["static_proof"]["opcode_counts"], {"v_fma_f32": 2})
            self.assertTrue(result["static_proof"]["zero_scratch"])
            self.assertEqual(set(result["static_proof"]["inputs"]),
                             {"code_object", "assembly", "metadata", "checker", "sources"})
            workload = {**result["selected_route"], "kind": "pp",
                        "weights_id": "selected",
                        "artifact_path": result["static_proof"]["inputs"]["sources"][0]["path"],
                        "executable_path": result["static_proof"]["inputs"]["code_object"]["path"]}
            self.assertEqual(_static_report(result, workload, "produced")["operation_family"],
                             "gdn_recurrence")

    def test_dispatch_display_and_code_object_symbols_are_distinct_authorities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            args["dispatch_symbol"] = "ordinary_kernel<true, 4u>"
            result = produce(**args)
            self.assertEqual(result["dispatch_signatures"][0]["symbol"],
                             "ordinary_kernel<true, 4u>")
            self.assertEqual(result["static_proof"]["code_symbol"], "scalar_kernel")

            args["stage"] = ["ninfer.gdn.prefill.gdn payload=0",
                             "ninfer.gdn.prefill.gdn payload=1"]
            result = produce(**args)
            self.assertEqual([row["stage"] for row in result["dispatch_signatures"]],
                             args["stage"])

            args["stage"] = ["duplicate", "duplicate"]
            with self.assertRaisesRegex(ValueError, "stages must be unique"):
                produce(**args)

    def test_zero_lds_ceiling_is_valid_but_bool_or_negative_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            assembly = args["assembly"]
            assembly.write_text(assembly.read_text(encoding="utf-8").replace(
                ".amdhsa_group_segment_fixed_size 128",
                ".amdhsa_group_segment_fixed_size 0"), encoding="utf-8")
            args["metadata"].write_text(assembly.read_text(encoding="utf-8"), encoding="utf-8")
            args["max_lds_bytes"] = 0
            result = produce(**args)
            self.assertEqual(result["static_proof"]["resources"]["lds_bytes"], 0)
            for invalid in (False, -1):
                args["max_lds_bytes"] = invalid
                with self.subTest(invalid=invalid), self.assertRaisesRegex(
                        ValueError, "nonnegative integer"):
                    produce(**args)

    def test_rejects_code_object_not_embedded_in_selected_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            detached = Path(directory) / "detached.co"
            detached.write_bytes(b"detached-device-object")
            args["code_object"] = detached
            with self.assertRaisesRegex(ValueError, "occur exactly once"):
                produce(**args)

    def test_dependency_safe_overlap_rejects_an_unchecked_hashed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            args["overlap"] = "proven_dependency_safe"
            with self.assertRaisesRegex(ValueError, "explicit evidence file"):
                produce(**args)
            evidence = Path(directory) / "overlap.json"
            evidence.write_text("{}", encoding="utf-8")
            args["overlap_evidence"] = evidence
            with self.assertRaisesRegex(ValueError, "only by the exact Q4 CTA checker"):
                produce(**args)

    def test_rejects_false_matrix_class_and_resource_or_scratch_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory) / "matrix", classification="matrix")
            with self.assertRaisesRegex(ValueError, "requires an exact WMMA opcode"):
                produce(**args)
            args = self.args(Path(directory) / "scratch", scratch=8)
            with self.assertRaisesRegex(ValueError, "zero private/scratch"):
                produce(**args)
            args = self.args(Path(directory) / "resource")
            args["max_vgpr_count"] = 20
            with self.assertRaisesRegex(ValueError, "exceed ceilings"):
                produce(**args)
            args = self.args(Path(directory) / "hidden-wmma")
            assembly = args["assembly"]
            assembly.write_text(assembly.read_text(encoding="utf-8").replace(
                "\t.amdhsa_kernel", "\tv_wmma_f32_16x16x16_bf16 v0, v1, v2\n\t.amdhsa_kernel"),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "containing WMMA"):
                produce(**args)

    def cta_files(self, root: Path, recipe: str, *, incumbent: bool = False):
        profile = PROFILES[recipe]
        symbol = profile.production_symbol
        opcodes = "\n".join(f"\t{profile.opcode} v[0:7], v[8:9], v[10:11], 0"
                            for _ in range(profile.opcode_count))
        prefetch = ("\n".join("\tglobal_load_b32 v0, v[0:1], off" for _ in range(4))
                    if recipe == "q4" else "")
        publication = ("\ts_wait_loadcnt 0x0\n\tds_store_b32 v0, v0\n"
                       "\tds_store_b32 v0, v0\n\tds_store_b32 v0, v0\n"
                       if recipe == "q4" else "")
        global_inv = "\n\tglobal_inv scope:SCOPE_SE" if incumbent else ""
        body = f"""\t.globl {symbol} ; -- Begin function {symbol}
{symbol}:
\ts_barrier_signal -1
\ts_barrier_wait -1{global_inv}
{prefetch}
{opcodes}
{publication}\
\ts_barrier_signal -1
\ts_barrier_wait -1{global_inv}
\t.amdhsa_kernel {symbol}
\t\t.amdhsa_group_segment_fixed_size {profile.lds_ceiling}
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_next_free_vgpr {profile.vgpr_ceiling}
\t.end_amdhsa_kernel
\t.set .Lselected.uses_flat_scratch, 0
; ScratchSize: 0
; Occupancy: {profile.occupancy or 16}
    .max_flat_workgroup_size: {profile.maximum_workgroup_size}
    .name: {symbol}
"""
        artifact, executable, source, assembly, metadata = self.files(
            root, "unused", "v_nop", 1)
        assembly.write_text(body, encoding="utf-8")
        metadata.write_text(body, encoding="utf-8")
        return artifact, executable, source, assembly, metadata

    def test_cta_mode_delegates_exact_challenger_gate(self) -> None:
        for recipe in CTA_RECIPES:
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                profile = PROFILES[recipe]
                artifact, executable, source, assembly, metadata = self.cta_files(
                    Path(directory), recipe)
                result = produce(
                    artifact=artifact, executable=executable, code_object=executable,
                    assembly=assembly,
                    metadata=metadata, sources=[source], stage="prefill/text/linear",
                    dispatch_symbol=profile.production_symbol,
                    code_symbol=profile.production_symbol,
                    operation_family=f"{recipe}_linear",
                    specialization=f"{recipe}-cta", hardware_classification="matrix",
                    arithmetic="signed_integer_matrix",
                    opcodes=[(profile.opcode, profile.opcode_count)],
                    max_lds_bytes=profile.lds_ceiling, max_vgpr_count=profile.vgpr_ceiling,
                    memory_residency="lds_reuse", memory_access="coalesced CTA staging",
                    overlap="unproven", memory_evidence="CTA checker and disassembly",
                    overlap_evidence=None, prefill_chunk=2048, kv_value_group=16,
                    xattention_profile="dense", cta_recipe=recipe)
                self.assertEqual(result["static_proof"]["resources"]["scratch_bytes"], 0)

                if recipe == "q4":
                    result = produce(
                        artifact=artifact, executable=executable, code_object=executable,
                        assembly=assembly, metadata=metadata, sources=[source],
                        stage="prefill/text/linear", dispatch_symbol=profile.production_symbol,
                        code_symbol=profile.production_symbol, operation_family="q4_linear",
                        specialization="q4-cta-overlap", hardware_classification="matrix",
                        arithmetic="signed_integer_matrix",
                        opcodes=[(profile.opcode, profile.opcode_count)],
                        max_lds_bytes=profile.lds_ceiling,
                        max_vgpr_count=profile.vgpr_ceiling,
                        memory_residency="lds_reuse",
                        memory_access="coalesced CTA staging",
                        overlap="proven_dependency_safe",
                        memory_evidence="exact Q4 CTA checker and assembly",
                        overlap_evidence=assembly, prefill_chunk=2048, kv_value_group=16,
                        xattention_profile="dense", cta_recipe="q4")
                    self.assertEqual(
                        result["memory_path"]["overlap_evidence"]["sha256"],
                        result["static_proof"]["inputs"]["assembly"]["sha256"])
                    unrelated = Path(directory) / "unrelated.json"
                    unrelated.write_text("{}", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "exact checked assembly"):
                        args = {
                            "artifact": artifact, "executable": executable,
                            "code_object": executable, "assembly": assembly,
                            "metadata": metadata, "sources": [source],
                            "stage": "prefill/text/linear",
                            "dispatch_symbol": profile.production_symbol,
                            "code_symbol": profile.production_symbol,
                            "operation_family": "q4_linear",
                            "specialization": "q4-cta-overlap",
                            "hardware_classification": "matrix",
                            "arithmetic": "signed_integer_matrix",
                            "opcodes": [(profile.opcode, profile.opcode_count)],
                            "max_lds_bytes": profile.lds_ceiling,
                            "max_vgpr_count": profile.vgpr_ceiling,
                            "memory_residency": "lds_reuse",
                            "memory_access": "coalesced CTA staging",
                            "overlap": "proven_dependency_safe",
                            "memory_evidence": "unrelated file is insufficient",
                            "overlap_evidence": unrelated, "prefill_chunk": 2048,
                            "kv_value_group": 16, "xattention_profile": "dense",
                            "cta_recipe": "q4",
                        }
                        produce(**args)

        with tempfile.TemporaryDirectory() as directory:
            profile = PROFILES["q4"]
            artifact, executable, source, assembly, metadata = self.cta_files(
                Path(directory), "q4", incumbent=True)
            with self.assertRaisesRegex(ValueError, "global_inv count 2, expected 0"):
                produce(
                    artifact=artifact, executable=executable, code_object=executable,
                    assembly=assembly,
                    metadata=metadata, sources=[source], stage="prefill/text/linear",
                    dispatch_symbol=profile.production_symbol,
                    code_symbol=profile.production_symbol,
                    operation_family="q4_linear",
                    specialization="q4-cta", hardware_classification="matrix",
                    arithmetic="signed_integer_matrix",
                    opcodes=[(profile.opcode, profile.opcode_count)],
                    max_lds_bytes=profile.lds_ceiling, max_vgpr_count=profile.vgpr_ceiling,
                    memory_residency="lds_reuse", memory_access="coalesced CTA staging",
                    overlap="unproven", memory_evidence="diagnostic incumbent",
                    overlap_evidence=None, prefill_chunk=2048, kv_value_group=16,
                    xattention_profile="dense", cta_recipe="q4")

    def test_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            _publish(path, {"first": True})
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _publish(path, {"first": False})


if __name__ == "__main__":
    unittest.main()
