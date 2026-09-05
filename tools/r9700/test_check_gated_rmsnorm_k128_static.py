from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_gated_rmsnorm_k128_static import check


SYMBOL = "_ZN6ninfer3ops5r97005eager12_GLOBAL__N_131gated_rmsnorm_k128_rows8_kernelEv"


def fixture(*, vgprs: int = 24, occupancy: int = 16, vector_load: bool = True,
            exponential: bool = True, scratch: int = 0, lds: int = 0,
            maximum_workgroup: int = 256, forbidden: str = "") -> str:
    load = "  global_load_b128 v[0:3], v0, off\n" if vector_load else ""
    exp = "  v_exp_f32 v0, v0\n" if exponential else ""
    return f"""\t.globl {SYMBOL} ; -- Begin function {SYMBOL}
{SYMBOL}:
{load}{exp}{forbidden}  .amdhsa_group_segment_fixed_size {lds}
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
            path = Path(directory) / "production.s"
            path.write_text(text, encoding="utf-8")
            return check(path)

    def test_accepts_exact_production_kernel(self):
        result = self.run_check(fixture())
        self.assertEqual(result["vgprs"], 24)
        self.assertEqual(result["maximum_workgroup"], 256)

    def test_rejects_missing_required_instructions(self):
        for text in (fixture(vector_load=False), fixture(exponential=False)):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "lacks"):
                    self.run_check(text)

    def test_rejects_resources(self):
        for text in (fixture(vgprs=25), fixture(occupancy=15), fixture(lds=4),
                     fixture(scratch=4)):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.run_check(text)

    def test_rejects_forbidden_instructions(self):
        for opcode in ("s_barrier\n", "v_wmma_f32_16x16x16_bf16 v[0:7], v[0:3], v[0:3], v[0:7]\n",
                       "global_atomic_add v0, v0, v1\n"):
            with self.subTest(opcode=opcode):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    self.run_check(fixture(forbidden="  " + opcode))

    def test_rejects_launch_geometry(self):
        with self.assertRaisesRegex(ValueError, "execution geometry"):
            self.run_check(fixture(maximum_workgroup=1024))


if __name__ == "__main__":
    unittest.main()
