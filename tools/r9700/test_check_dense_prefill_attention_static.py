#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.r9700.check_dense_prefill_attention_static import PROFILE, check


class DensePrefillAttentionStaticTest(unittest.TestCase):
    symbol = ("_ZN6ninfer3ops5r97002kv20dense_prefill_kernelILj16EEEv"
              "NS2_22Fp8Int4KvAttentionArgsE")

    def fixture(self, root: Path, *, lds: int = PROFILE["lds"], vgpr: int = 246,
                scratch: int = 0, occupancy: int = 5, maximum_workgroup: int = 384,
                vgpr_spills: int = 0, bf16: int = 32, f16: int = 32, barriers: int = 3,
                additional: str = "") -> tuple[Path, Path]:
        symbol = self.symbol
        instructions = "\n".join(["\tv_wmma_f32_16x16x16_bf16 v[0:7], v[8:11], v[12:15]"] * bf16 +
                                 ["\tv_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[12:15]"] * f16 +
                                 ["\tv_exp_f32_e32 v0, v1"] +
                                 ["\ts_barrier_signal -1\n\ts_barrier_wait -1"] * barriers)
        instructions += additional
        body = f"""\t.globl {symbol} ; -- Begin function {symbol}
{symbol}:
{instructions}
\t.amdhsa_kernel {symbol}
\t\t.amdhsa_group_segment_fixed_size {lds}
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_wavefront_size32 1
\t\t.amdhsa_workgroup_processor_mode 1
\t\t.amdhsa_next_free_vgpr {vgpr}
\t.end_amdhsa_kernel
\t.set .Lselected.uses_flat_scratch, 0
; ScratchSize: {scratch}
; Occupancy: {occupancy}
amdhsa.kernels:
  - .args: []
    .max_flat_workgroup_size: {maximum_workgroup}
    .name:           {symbol}
    .sgpr_spill_count: 0
    .vgpr_spill_count: {vgpr_spills}
    .wavefront_size: 32
"""
        assembly, metadata = root / "assembly.s", root / "metadata.s"
        assembly.write_text(body, encoding="utf-8")
        metadata.write_text(body, encoding="utf-8")
        return assembly, metadata

    def run_check(self, paths: tuple[Path, Path], value_group: int = 16):
        return check(assembly=paths[0], metadata=paths[1], symbol=self.symbol,
                     value_group=value_group)

    def test_accepts_exact_fused_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_check(self.fixture(Path(directory)))
            self.assertEqual(result["bf16_wmma"], 32)
            self.assertEqual(result["f16_wmma"], 32)
            self.assertEqual(result["lds_bytes"], PROFILE["lds"])

    def test_rejects_wrong_group_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "G32 specialization"):
                self.run_check(self.fixture(Path(directory)), value_group=32)

    def test_rejects_resource_and_instruction_regressions(self) -> None:
        cases = (
            (dict(bf16=31), "WMMA counts"),
            (dict(f16=33), "WMMA counts"),
            (dict(additional="\n\tv_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[10:11]"),
             "unexpected matrix opcodes"),
            (dict(barriers=2), "barrier pairs"),
            (dict(lds=37537), "LDS size"),
            (dict(vgpr=257), "resources fail"),
            (dict(scratch=4), "all must be zero"),
            (dict(vgpr_spills=1), "all must be zero"),
            (dict(maximum_workgroup=256), "execution mode"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    self.run_check(self.fixture(Path(directory), **changes))


if __name__ == "__main__":
    unittest.main()
