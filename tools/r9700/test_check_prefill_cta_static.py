#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.r9700.check_prefill_cta_static import PROFILES, check

class PrefillCtaStaticTest(unittest.TestCase):
    def fixture(self, root: Path, recipe: str, *, incumbent: bool = False) -> tuple[Path, Path]:
        profile = PROFILES[recipe]
        symbol = profile.incumbent_symbol if incumbent else profile.production_symbol
        opcode_count = profile.incumbent_opcode_count if incumbent else profile.opcode_count
        lds = profile.incumbent_lds_ceiling if incumbent else profile.lds_ceiling
        vgpr = profile.incumbent_vgpr_ceiling if incumbent else profile.vgpr_ceiling
        control = " neg_lo:[1,1,0]" if recipe == "q4-a4-m64n128" else ""
        controls = ([" neg_lo:[0,1,0]"] * 4 + [" neg_lo:[1,1,0]"] * 4
                    if recipe == "q4" and not incumbent else [control] * opcode_count)
        opcodes = "\n".join(
            f"\t{profile.opcode} v[0:7], v[8:9], v[10:11], 0{control}"
            for control in controls)
        global_inv = ("\n\tglobal_inv scope:SCOPE_SE"
                      if incumbent and profile.incumbent_global_inv_count else "")
        prefetch_count = 6 if recipe == "q4" and not incumbent else 0
        pipeline_prefetch = (("\tglobal_load_b32 v0, v0, s[4:5]\n"
                              "\tglobal_load_b32 v1, v0, s[6:7]\n"
                              "\tglobal_load_b64 v[2:3], v1, s[16:17]\n"
                              "\tglobal_load_d16_b16 v4, v2, s[8:9]\n"
                              "\tglobal_load_d16_b16 v5, v3, s[18:19]\n") * 2
                             if prefetch_count else "")
        pipeline_publish = ("\ts_wait_loadcnt 0x0\n" +
                            "\tds_store_b32 v0, v1\n" * 3
                            if prefetch_count else "")
        body = f"""\t.globl {symbol} ; -- Begin function {symbol}
{symbol}:
\ts_barrier_signal -1
\ts_barrier_wait -1{global_inv}
{pipeline_prefetch}{opcodes}
{pipeline_publish}\ts_barrier_signal -1
\ts_barrier_wait -1{global_inv}
\t.amdhsa_kernel {symbol}
\t\t.amdhsa_group_segment_fixed_size {lds}
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_next_free_vgpr {vgpr}
\t.end_amdhsa_kernel
\t.set .Lselected.uses_flat_scratch, 0
; ScratchSize: 0
; Occupancy: {profile.occupancy or 16}
  - .args:
    .max_flat_workgroup_size: {profile.maximum_workgroup_size}
    .name: {symbol}
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
"""
        assembly, metadata = root / "assembly.s", root / "metadata.s"
        assembly.write_text(body, encoding="utf-8")
        metadata.write_text(body, encoding="utf-8")
        return assembly, metadata

    def test_accepts_exact_q4_and_w8_production_symbols(self) -> None:
        for recipe in ("q4", "w8"):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                result = check(recipe, "lds-scope", *self.fixture(Path(directory), recipe))
                self.assertEqual(result["global_inv_count"], 0)
                self.assertEqual(result["barrier_pair_count"], 2)

    def test_q4_production_is_pingpong_and_single_bank_n128_is_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = check("q4", "lds-scope", *self.fixture(Path(directory), "q4"))
            self.assertEqual(result["opcode_count"], 8)
            self.assertEqual(result["lds_bytes"], 17152)
            self.assertEqual(result["global_inv_count"], 0)
            self.assertEqual(result["n16_weight_b64_sites"], 2)
            self.assertEqual(result["scalar_base_load_sites"], 10)

    def test_q4_production_requires_n16_weight_load_and_signed_plane_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assembly, metadata = self.fixture(Path(directory), "q4")
            text = assembly.read_text(encoding="utf-8")
            assembly.write_text(text.replace("global_load_b64", "global_load_b32", 1),
                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "N16/K16 weight b64"):
                check("q4", "lds-scope", assembly, metadata)
            assembly.write_text(text.replace("neg_lo:[0,1,0]", "neg_lo:[1,1,0]", 1),
                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsigned-low"):
                check("q4", "lds-scope", assembly, metadata)

    def test_q4_production_requires_scalar_base_u32_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assembly, metadata = self.fixture(Path(directory), "q4")
            text = assembly.read_text(encoding="utf-8")
            assembly.write_text(
                text.replace("global_load_b32 v0, v0, s[4:5]",
                             "global_load_b32 v0, v[0:1], off", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "scalar-base role/order"):
                check("q4", "lds-scope", assembly, metadata)

    def test_accepts_exact_m128n128_challenger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = check("q4-m128n128", "m128n128",
                           *self.fixture(Path(directory), "q4-m128n128"))
            self.assertEqual(result["opcode_count"], 8)
            self.assertEqual(result["lds_bytes"], 12800)
            self.assertEqual(result["maximum_workgroup_size"], 1024)

    def test_accepts_exact_a4_m64n128_challenger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = check("q4-a4-m64n128", "a4-m64n128",
                           *self.fixture(Path(directory), "q4-a4-m64n128"))
            self.assertEqual(result["opcode_count"], 4)
            self.assertEqual(result["lds_bytes"], 6528)
            self.assertEqual(result["vgpr_count"], 84)
            self.assertEqual(result["occupancy"], 16)

    def test_m128n128_recipe_and_mode_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be selected together"):
                check("q4", "m128n128", *self.fixture(Path(directory), "q4"))
            with self.assertRaisesRegex(ValueError, "must be selected together"):
                check("q4-m128n128", "lds-scope",
                      *self.fixture(Path(directory), "q4-m128n128"))
            result = check("q4", "incumbent-diagnostic",
                           *self.fixture(Path(directory), "q4", incumbent=True))
            self.assertEqual(result["opcode_count"], 8)
            self.assertEqual(result["lds_bytes"], 8576)
            self.assertEqual(result["global_inv_count"], 0)

    def test_incumbent_is_diagnostic_only_and_modes_do_not_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory), "q4", incumbent=True)
            self.assertEqual(check("q4", "incumbent-diagnostic", *paths)["global_inv_count"], 0)
            with self.assertRaisesRegex(ValueError, "exact selected symbol.*found 0"):
                check("q4", "lds-scope", *paths)
            with self.assertRaisesRegex(ValueError, "unsupported static-gate mode"):
                check("q4", "incumbent", *paths)

    def test_rejects_symbol_ambiguity_and_wrong_native_opcode_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assembly, metadata = self.fixture(Path(directory), "w8")
            text = assembly.read_text(encoding="utf-8")
            assembly.write_text(text + text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one function body, found 2"):
                check("w8", "lds-scope", assembly, metadata)
            assembly.write_text(text.replace(PROFILES["w8"].opcode, "v_nop", 1),
                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count 1, expected 2"):
                check("w8", "lds-scope", assembly, metadata)

    def test_rejects_barrier_and_resource_regressions(self) -> None:
        mutations = (
            ("s_barrier_wait -1", "s_barrier", "signal=2 wait=1", True),
            (".amdhsa_next_free_vgpr 96", ".amdhsa_next_free_vgpr 97", "exceed", False),
            (".amdhsa_private_segment_fixed_size 0", ".amdhsa_private_segment_fixed_size 4",
             "all must be zero", False),
            (".uses_flat_scratch, 0", ".uses_flat_scratch, 1", "all must be zero", False),
            ("ScratchSize: 0", "ScratchSize: 8", "all must be zero", False),
            (".sgpr_spill_count: 0", ".sgpr_spill_count: 1", "all must be zero", False),
            (".vgpr_spill_count: 0", ".vgpr_spill_count: 1", "all must be zero", False),
        )
        for old, new, message, assembly_target in mutations:
            with self.subTest(old=old), tempfile.TemporaryDirectory() as directory:
                assembly, metadata = self.fixture(Path(directory), "q4")
                target = assembly if assembly_target else metadata
                target.write_text(target.read_text(encoding="utf-8").replace(old, new, 1),
                                  encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    check("q4", "lds-scope", assembly, metadata)

    def test_rejects_monolithic_barrier_even_with_two_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assembly, metadata = self.fixture(Path(directory), "w8")
            assembly.write_text(assembly.read_text(encoding="utf-8") + "\ns_barrier\n",
                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "monolithic s_barrier"):
                check("w8", "lds-scope", assembly, metadata)

    def test_rejects_unexpected_barrier_family_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assembly, metadata = self.fixture(Path(directory), "q4")
            assembly.write_text(assembly.read_text(encoding="utf-8") +
                                "\ns_barrier_signal_var s0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected barrier-family instruction"):
                check("q4", "lds-scope", assembly, metadata)

    def test_rejects_unpaired_barrier_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assembly, metadata = self.fixture(Path(directory), "q4")
            text = assembly.read_text(encoding="utf-8")
            first_wait = text.index("s_barrier_wait")
            second_signal = text.index("s_barrier_signal", text.index("s_barrier_signal") + 1)
            text = (text[:first_wait] + "s_barrier_signal" +
                    text[first_wait + len("s_barrier_wait"):second_signal] + "s_barrier_wait" +
                    text[second_signal + len("s_barrier_signal"):])
            assembly.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "two ordered barrier pairs"):
                check("q4", "lds-scope", assembly, metadata)

    def test_rejects_non_workgroup_barrier_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assembly, metadata = self.fixture(Path(directory), "w8")
            assembly.write_text(assembly.read_text(encoding="utf-8").replace(
                "s_barrier_signal -1", "s_barrier_signal 0", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "workgroup barrier id -1"):
                check("w8", "lds-scope", assembly, metadata)

if __name__ == "__main__":
    unittest.main()
