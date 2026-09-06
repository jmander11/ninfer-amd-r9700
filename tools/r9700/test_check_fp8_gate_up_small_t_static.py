#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fp8_gate_up_small_t_static import SYMBOL, analyze  # noqa: E402


def fixture(*, vgpr: int = 54, sgpr: int = 19, wmma: int = 4,
            loads: int = 5, scratch: int = 0, wave: int = 32,
            vgpr_spills: int = 0) -> str:
    instructions = "\n".join(
        ["\tv_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[9:10], v[11:12], v[1:8]"] * wmma
        + ["\tglobal_load_b64 v[1:2], v[3:4], off"] * loads
    )
    return f"""; -- Begin function {SYMBOL}
{instructions}
\t.amdhsa_group_segment_fixed_size 0
\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_wavefront_size32 1
\t.amdhsa_next_free_vgpr {vgpr}
\t.amdhsa_next_free_sgpr {sgpr}
; TotalNumSgprs: 21
; NumVgprs: 54
; ScratchSize: {scratch}
; Occupancy: 16
\t.size {SYMBOL}, .-{SYMBOL}
---
  - .args:
      - .offset: 0
    .max_flat_workgroup_size: 128
    .name: qualified_{SYMBOL}
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .vgpr_count:     54
    .vgpr_spill_count: {vgpr_spills}
    .wavefront_size: {wave}
  - .args:
    .name: next_kernel
"""


class StaticCheckTest(unittest.TestCase):
    def test_accepts_exact_profile(self) -> None:
        self.assertEqual(analyze(fixture())["wmma_sites"], 4)

    def test_rejects_lost_chain(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "four independent"):
            analyze(fixture(wmma=3))

    def test_rejects_resource_regression(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "VGPR identity"):
            analyze(fixture(vgpr=65))

    def test_rejects_scratch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "resource profile"):
            analyze(fixture(scratch=16))

    def test_rejects_sgpr_change(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SGPR identity"):
            analyze(fixture(sgpr=20))

    def test_rejects_load_site_change(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly five"):
            analyze(fixture(loads=6))

    def test_rejects_wave_change(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wavefront_size"):
            analyze(fixture(wave=64))

    def test_rejects_spill(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "vgpr_spill_count"):
            analyze(fixture(vgpr_spills=1))


if __name__ == "__main__":
    unittest.main()
