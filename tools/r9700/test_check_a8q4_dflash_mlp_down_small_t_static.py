from __future__ import annotations

import unittest
from pathlib import Path

from tools.r9700.check_a8q4_dflash_mlp_down_small_t_static import check


ASSEMBLY = Path(__file__).resolve().parent / "build" / "a8q4_dflash_mlp_down_small_t.s"


class MlpDownSmallTStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assembly = ASSEMBLY.read_text()

    def test_current_gfx1201_assembly_passes(self) -> None:
        result = check(self.assembly)
        self.assertEqual(result["dot8"], 80)
        self.assertEqual(result["vgpr"], 33)

    def test_sgpr_spill_is_rejected(self) -> None:
        marker = "mlp_down_qual16candidate_kernelILj5EE"
        begin = self.assembly.find(".name:", self.assembly.find(".name:", 0))
        begin = self.assembly.find(".name:", self.assembly.find(marker, begin) - 80)
        spill = self.assembly.find(".sgpr_spill_count: 0", begin)
        self.assertGreaterEqual(spill, 0)
        mutated = self.assembly[:spill] + self.assembly[spill:].replace(
            ".sgpr_spill_count: 0", ".sgpr_spill_count: 1", 1
        )
        with self.assertRaisesRegex(ValueError, "resource identity mismatch"):
            check(mutated)

    def test_extra_t6_candidate_is_rejected(self) -> None:
        mutated = self.assembly + (
            "\n; -- Begin function fake_mlp_down_qual16candidate_kernelILj6EE\n"
        )
        with self.assertRaisesRegex(ValueError, "only the T5"):
            check(mutated)


if __name__ == "__main__":
    unittest.main()
