from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_rmsnorm_k256_prefill_static import check

SYMBOL = "_ZN6ninfer3ops5r97005eager12_GLOBAL__N_128rmsnorm_k256_token8_kernelEv"


def fixture(*, operation: str = "global_load_b128 v[0:3], v0, off", vgprs: int = 24,
            occupancy: int = 16, lds: int = 0, scratch: int = 0) -> str:
    return f"""\t.globl {SYMBOL} ; -- Begin function {SYMBOL}
{SYMBOL}:
  {operation}
  .amdhsa_group_segment_fixed_size {lds}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr {vgprs}
  .amdhsa_wavefront_size32 1
  .amdhsa_workgroup_processor_mode 1
; ScratchSize: {scratch}
; Occupancy: {occupancy}
  - .args: []
    .max_flat_workgroup_size: 256
    .name: {SYMBOL}
"""


class StaticCheckTest(unittest.TestCase):
    def run_check(self, text: str):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.s"
            path.write_text(text, encoding="utf-8")
            return check(path)

    def test_accepts(self):
        result = self.run_check(fixture())
        self.assertIs(result["production"], True)
        self.assertEqual(result["rows_minimum"], 128)

    def test_rejects_missing_vector_load(self):
        with self.assertRaisesRegex(ValueError, "128-bit"):
            self.run_check(fixture(operation="v_add_f32 v0, v1, v2"))

    def test_rejects_barrier(self):
        with self.assertRaisesRegex(ValueError, "barrier"):
            self.run_check(fixture(operation="global_load_b128 v[0:3], v0, off\n  s_barrier"))

    def test_rejects_resources(self):
        for text in (fixture(vgprs=65), fixture(occupancy=11), fixture(lds=16),
                     fixture(scratch=16)):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "resources|must be zero"):
                    self.run_check(text)


if __name__ == "__main__":
    unittest.main()
