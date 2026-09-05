#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.r9700.check_dense_prefill_attention_static import (
    FULL_SCORE_PROFILES,
    check,
)


class DensePrefillAttentionStaticTest(unittest.TestCase):
    symbol = "_ZN6ninfer3ops5r97002kv26dense_full_score_qk_kernelILb0EEEv"

    def fixture(self, root: Path, *, lds: int = 8296, vgpr: int = 29,
                scratch: int = 0, occupancy: int = 15, maximum_workgroup: int = 192,
                wavefront: int = 32, wgp_mode: int = 1, vgpr_spills: int = 0,
                opcode: str = "v_wmma_f32_16x16x16_bf16",
                symbol: str | None = None, wmma_count: int = 16,
                additional: str = "") -> tuple[Path, Path]:
        symbol = self.symbol if symbol is None else symbol
        instructions = "\n".join(f"\t{opcode} v[0:7], v[8:11], v[12:15]"
                                 for _ in range(wmma_count))
        instructions += additional
        body = f"""\t.globl {symbol} ; -- Begin function {symbol}
{symbol}:
{instructions}
\t.amdhsa_kernel {symbol}
\t\t.amdhsa_group_segment_fixed_size {lds}
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_wavefront_size32 {1 if wavefront == 32 else 0}
\t\t.amdhsa_workgroup_processor_mode {wgp_mode}
\t\t.amdhsa_next_free_vgpr {vgpr}
\t.end_amdhsa_kernel
\t.set .Lselected.uses_flat_scratch, 0
; ScratchSize: {scratch}
; Occupancy: {occupancy}
amdhsa.kernels:
  - .args: []
    .max_flat_workgroup_size: {maximum_workgroup}
    .name:           {symbol}
    .private_segment_fixed_size: 0
    .sgpr_spill_count: 0
    .symbol:         {symbol}.kd
    .vgpr_spill_count: {vgpr_spills}
    .wavefront_size: {wavefront}
    .workgroup_processor_mode: {wgp_mode}
"""
        assembly, metadata = root / "assembly.s", root / "metadata.s"
        assembly.write_text(body, encoding="utf-8")
        metadata.write_text(body, encoding="utf-8")
        return assembly, metadata

    def invoke(self, paths: tuple[Path, Path], **changes):
        args = dict(assembly=paths[0], metadata=paths[1], symbol=self.symbol,
                    value_group=16, full_score_stage="qk_bk16")
        args.update(changes)
        return check(**args)

    def test_accepts_exact_production_bk16_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(self.fixture(Path(directory)))
            self.assertEqual(result["bf16_wmma_count"], 16)
            self.assertEqual(result["lds_bytes"], 8296)
            self.assertEqual(result["query_tile"], 16)
            self.assertEqual(result["maximum_workgroup_size"], 192)

    def test_accepts_exact_full_score_stage_profiles(self) -> None:
        qk_symbol = "_ZN6ninfer3ops5r97002kv26dense_full_score_qk_kernelILb0EEEv"
        with tempfile.TemporaryDirectory() as directory:
            profile = FULL_SCORE_PROFILES["qk_bk16"]
            paths = self.fixture(Path(directory), lds=profile["lds"], vgpr=29,
                                 occupancy=15,
                                 maximum_workgroup=profile["workgroup"], symbol=qk_symbol)
            result = check(assembly=paths[0], metadata=paths[1], symbol=qk_symbol,
                           value_group=16, full_score_stage="qk_bk16")
            self.assertEqual(result["full_score_stage"], "qk_bk16")

            bk32_symbol = ("_ZN6ninfer3ops5r97002kv"
                           "31dense_full_score_qk_bk32_kernelILb0EEEv")
            profile = FULL_SCORE_PROFILES["qk_bk32"]
            paths = self.fixture(
                Path(directory), lds=profile["lds"], vgpr=37, occupancy=15,
                maximum_workgroup=profile["workgroup"], symbol=bk32_symbol,
                wmma_count=profile["wmma"])
            result = check(assembly=paths[0], metadata=paths[1], symbol=bk32_symbol,
                           value_group=16, full_score_stage="qk_bk32")
            self.assertEqual(result["bf16_wmma_count"], 32)
            self.assertEqual(result["lds_bytes"], 16488)

            pv_symbol = ("_ZN6ninfer3ops5r97002kv26dense_full_score_pv_kernel"
                         "ILj16ELb0EEEv")
            profile = FULL_SCORE_PROFILES["pv_g16"]
            paths = self.fixture(
                Path(directory), lds=profile["lds"], vgpr=116,
                occupancy=12, maximum_workgroup=profile["workgroup"], symbol=pv_symbol,
                wmma_count=0, additional="\n\tv_exp_f32 v0, v1\n\tv_fmac_f32 v0, v1, v2")
            result = check(assembly=paths[0], metadata=paths[1], symbol=pv_symbol,
                           value_group=16, full_score_stage="pv")
            self.assertEqual(result["full_score_stage"], "pv")

    def test_full_score_pv_gate_rejects_wmma(self) -> None:
        symbol = ("_ZN6ninfer3ops5r97002kv26dense_full_score_pv_kernel"
                  "ILj16ELb0EEEv")
        with tempfile.TemporaryDirectory() as directory:
            profile = FULL_SCORE_PROFILES["pv_g16"]
            paths = self.fixture(Path(directory), lds=profile["lds"], vgpr=116,
                                 occupancy=12, maximum_workgroup=256, symbol=symbol,
                                 additional="\n\tv_exp_f32 v0, v1\n\tv_fmac_f32 v0, v1, v2")
            with self.assertRaisesRegex(ValueError, "count 16, expected 0"):
                check(assembly=paths[0], metadata=paths[1], symbol=symbol,
                      value_group=16, full_score_stage="pv")

    def test_rejects_wrong_symbol_or_opcode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "production full-score"):
                self.invoke(paths, symbol="qualification_kernel")
            assembly = paths[0]
            assembly.write_text(assembly.read_text(encoding="utf-8").replace(
                "v_wmma_f32_16x16x16_bf16", "v_nop", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count 15, expected 16"):
                self.invoke(paths)
            paths = self.fixture(Path(directory), opcode="v_wmma_i32_16x16x16_iu8")
            with self.assertRaisesRegex(ValueError, "count 0"):
                self.invoke(paths)

    def test_rejects_resource_scratch_and_execution_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "LDS size"):
                self.invoke(self.fixture(Path(directory), lds=8292))
            with self.assertRaisesRegex(ValueError, "resources fail"):
                self.invoke(self.fixture(Path(directory), vgpr=241))
            with self.assertRaisesRegex(ValueError, "resources fail"):
                self.invoke(self.fixture(Path(directory), occupancy=5))
            with self.assertRaisesRegex(ValueError, "all must be zero"):
                self.invoke(self.fixture(Path(directory), scratch=8))
            with self.assertRaisesRegex(ValueError, "all must be zero"):
                self.invoke(self.fixture(Path(directory), vgpr_spills=1))
            with self.assertRaisesRegex(ValueError, "execution mode fails"):
                self.invoke(self.fixture(Path(directory), maximum_workgroup=512))
            with self.assertRaisesRegex(ValueError, "execution mode fails"):
                self.invoke(self.fixture(Path(directory), wavefront=64))
            with self.assertRaisesRegex(ValueError, "execution mode fails"):
                self.invoke(self.fixture(Path(directory), wgp_mode=0))


if __name__ == "__main__":
    unittest.main()
