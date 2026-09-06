from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_rmsnorm_decode_static import check


SYMBOL = "_ZN6ninfer3ops5r97005eager12_GLOBAL__N_137rmsnorm_k5120_rows4_cta_kernelEv"


def fixture(*, vgprs: int = 24, lds: int = 32, occupancy: int = 16,
            global_load: bool = True, shuffle: bool = True, scalar: bool = False,
            scratch: int = 0, maximum_workgroup: int = 256) -> str:
    load = "  global_load_u16 v0, v[1:2], off\n" if global_load else ""
    shuffle_instruction = "  ds_bpermute_b32 v0, v1, v2\n" if shuffle else ""
    scalar_instruction = "  s_load_u16 s0, s[2:3], 0\n" if scalar else ""
    return f"""\t.globl {SYMBOL} ; -- Begin function {SYMBOL}
{SYMBOL}:
{load}{shuffle_instruction}{scalar_instruction}  .amdhsa_group_segment_fixed_size {lds}
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
        result = self.run_check(fixture())
        self.assertEqual(result["lds"], 32)
        self.assertEqual(result["maximum_workgroup"], 256)

    def test_rejects_missing_global_load(self):
        with self.assertRaisesRegex(ValueError, "global input load"):
            self.run_check(fixture(global_load=False))

    def test_rejects_missing_shuffle(self):
        with self.assertRaisesRegex(ValueError, "shuffle reduction"):
            self.run_check(fixture(shuffle=False))

    def test_rejects_scalar_reduction(self):
        with self.assertRaisesRegex(ValueError, "scalar feature reduction"):
            self.run_check(fixture(scalar=True))

    def test_rejects_resource_regression(self):
        with self.assertRaisesRegex(ValueError, "resources fail"):
            self.run_check(fixture(vgprs=49))

    def test_rejects_lds_regression(self):
        with self.assertRaisesRegex(ValueError, "LDS/private/scratch fail"):
            self.run_check(fixture(lds=36))

    def test_rejects_scratch(self):
        with self.assertRaisesRegex(ValueError, "LDS/private/scratch fail"):
            self.run_check(fixture(scratch=16))

    def test_rejects_launch_geometry(self):
        with self.assertRaisesRegex(ValueError, "execution geometry"):
            self.run_check(fixture(maximum_workgroup=1024))


if __name__ == "__main__":
    unittest.main()
