from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.r9700.check_gdn_grouped_heads_static import SYMBOL, check


class GroupedHeadsStaticTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        path = root / "candidate.s"
        path.write_text(f"""; -- Begin function {SYMBOL}
{SYMBOL}:
 s_barrier_signal -1
 s_barrier_wait -1
 s_barrier_signal -1
 s_barrier_wait -1
 s_barrier_signal -1
 s_barrier_wait -1
 s_barrier_signal -1
 s_barrier_wait -1
 .amdhsa_group_segment_fixed_size 1056
 .amdhsa_private_segment_fixed_size 0
 .amdhsa_next_free_vgpr 116
; ScratchSize: 0
; Occupancy: 12
 .max_flat_workgroup_size: 256
""", encoding="utf-8")
        return path

    def test_accepts_exact_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(check(self.fixture(Path(directory)))["vgpr"], 116)

    def test_rejects_spill_resource_and_barrier_drift(self) -> None:
        mutations = (("ScratchSize: 0", "ScratchSize: 4"),
                     ("Occupancy: 12", "Occupancy: 4"),
                     (" s_barrier_wait -1\n", "",))
        for old, new in mutations:
            with self.subTest(old=old), tempfile.TemporaryDirectory() as directory:
                path = self.fixture(Path(directory))
                path.write_text(path.read_text().replace(old, new, 1))
                with self.assertRaises(ValueError):
                    check(path)


if __name__ == "__main__":
    unittest.main()
