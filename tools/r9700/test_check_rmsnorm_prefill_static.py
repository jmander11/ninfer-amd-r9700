from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_rmsnorm_prefill_static import check


SYMBOL = "_ZN6ninfer3ops5r97005eager12_GLOBAL__N_129rmsnorm_k5120_token8_kernelEv"


def fixture(*, vgprs: int = 48, occupancy: int = 16, vector_load: bool = True,
            scratch: int = 0, maximum_workgroup: int = 256) -> str:
    load = "  global_load_b128 v[0:3], v0, off\n" if vector_load else ""
    return f"""\t.globl {SYMBOL} ; -- Begin function {SYMBOL}
{SYMBOL}:
{load}  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr {vgprs}
  .amdhsa_wavefront_size32 1
  .amdhsa_workgroup_processor_mode 1
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
        self.assertEqual(self.run_check(fixture())["maximum_workgroup"], 256)

    def test_rejects_missing_vector_load(self):
        with self.assertRaisesRegex(ValueError, "128-bit vector"):
            self.run_check(fixture(vector_load=False))

    def test_rejects_resource_regression(self):
        with self.assertRaisesRegex(ValueError, "resources fail"):
            self.run_check(fixture(vgprs=121))

    def test_rejects_scratch(self):
        with self.assertRaisesRegex(ValueError, "must be zero"):
            self.run_check(fixture(scratch=16))

    def test_rejects_launch_geometry(self):
        with self.assertRaisesRegex(ValueError, "execution geometry"):
            self.run_check(fixture(maximum_workgroup=1024))


if __name__ == "__main__":
    unittest.main()
