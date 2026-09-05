from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_silu_mul_split_prefill_static import check


SYMBOL = "_ZN6ninfer3ops5r97005eager12_GLOBAL__N_137silu_mul_split17408_2d_kernelEv"


def fixture(*, vgprs: int = 24, occupancy: int = 16, lds: int = 0, scratch: int = 0,
            operation: str = "v_exp_f32 v0, v0", maximum_workgroup: int = 256) -> str:
    return f"""\t.globl {SYMBOL} ; -- Begin function {SYMBOL}
{SYMBOL}:
  {operation}
  .amdhsa_group_segment_fixed_size {lds}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr {vgprs}
  .amdhsa_wavefront_size32 1
  .amdhsa_workgroup_processor_mode 1
  .amdhsa_system_sgpr_workgroup_id_y 1
; ScratchSize: {scratch}
; Occupancy: {occupancy}
  - .args: []
    .max_flat_workgroup_size: {maximum_workgroup}
    .name: {SYMBOL}
"""


class StaticCheckTest(unittest.TestCase):
    def run_check(self, text: str):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.s"
            path.write_text(text, encoding="utf-8")
            return check(path)

    def test_accepts_exact_candidate(self):
        result = self.run_check(fixture())
        self.assertIs(result["production"], True)
        self.assertEqual(result["rows_minimum"], 128)

    def test_rejects_coordinate_division(self):
        with self.assertRaisesRegex(ValueError, "coordinate division"):
            self.run_check(fixture(operation="v_mul_hi_u32 v0, v1, v2"))

    def test_rejects_missing_exponential(self):
        with self.assertRaisesRegex(ValueError, "exponential"):
            self.run_check(fixture(operation="v_add_f32 v0, v1, v2"))

    def test_rejects_resources(self):
        for text in (fixture(vgprs=65), fixture(lds=16), fixture(scratch=16)):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "resources|must be zero"):
                    self.run_check(text)

    def test_rejects_geometry(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.run_check(fixture(maximum_workgroup=1024))


if __name__ == "__main__":
    unittest.main()
