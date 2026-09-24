from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_causal_conv_prefill_static import check


def fixture(*, vgprs: int = 48, scratch: int = 0, omit_tile: int | None = None,
            maximum_workgroup: int = 256) -> str:
    blocks = []
    for tile in (4, 8, 16, 32):
        if tile == omit_tile:
            continue
        symbol = ("_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_1"
                  f"33causal_conv1d_silu_prefill_kernelILj{tile}EEv")
        blocks.append(f"""; -- Begin function {symbol}
{symbol}:
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr {vgprs}
  .amdhsa_wavefront_size32 1
  .amdhsa_workgroup_processor_mode 1
; ScratchSize: {scratch}
; Occupancy: 8
""")
    metadata = []
    for tile in (4, 8, 16, 32):
        if tile == omit_tile:
            continue
        symbol = ("_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_1"
                  f"33causal_conv1d_silu_prefill_kernelILj{tile}EEv")
        metadata.append(f"""  - .args: []
    .max_flat_workgroup_size: {maximum_workgroup}
    .name: {symbol}
""")
    return "".join(blocks + metadata)


class StaticCheckTest(unittest.TestCase):
    def run_check(self, text: str):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.s"
            path.write_text(text, encoding="utf-8")
            return check(path)

    def test_accepts_exact_four_candidate_matrix(self):
        self.assertEqual(set(self.run_check(fixture())), {4, 8, 16, 32})

    def test_rejects_missing_specialization(self):
        with self.assertRaisesRegex(ValueError, "tile 16"):
            self.run_check(fixture(omit_tile=16))

    def test_rejects_resource_regression(self):
        with self.assertRaisesRegex(ValueError, "resources fail"):
            self.run_check(fixture(vgprs=241))

    def test_rejects_scratch(self):
        with self.assertRaisesRegex(ValueError, "must be zero"):
            self.run_check(fixture(scratch=16))

    def test_rejects_unbounded_launch_geometry(self):
        with self.assertRaisesRegex(ValueError, "maximum flat workgroup"):
            self.run_check(fixture(maximum_workgroup=1024))


if __name__ == "__main__":
    unittest.main()
