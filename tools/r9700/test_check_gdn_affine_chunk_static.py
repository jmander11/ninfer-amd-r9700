from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.r9700.check_gdn_affine_chunk_static import check


def fixture(*, scratch: int = 0, private: int = 0, vgprs: int = 80,
            occupancy: int = 8, lds: int = 16384, wave32: int = 1,
            wgp: int = 1, maximum_workgroup: int = 256,
            wmma: bool = False, omit: str | None = None,
            duplicate: str | None = None) -> str:
    blocks = []
    metadata = []
    for stage in ("build", "boundaries", "replay"):
        if stage == omit:
            continue
        symbol = f"_Z41affine_chunk_{stage}_qualification_kernelv"
        opcode = "  v_wmma_f32_16x16x16_bf16 v0, v1, v2, v3\n" if wmma else ""
        block = f"""\t.globl {symbol} ; -- Begin function {symbol}
{symbol}:
{opcode}  .amdhsa_group_segment_fixed_size {lds}
  .amdhsa_private_segment_fixed_size {private}
  .amdhsa_next_free_vgpr {vgprs}
  .amdhsa_wavefront_size32 {wave32}
  .amdhsa_workgroup_processor_mode {wgp}
; ScratchSize: {scratch}
; Occupancy: {occupancy}
"""
        blocks.append(block)
        if duplicate == stage:
            blocks.append(block.replace(symbol, symbol + "_duplicate"))
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

    def test_accepts_three_fp32_stages(self):
        self.assertEqual(set(self.run_check(fixture())), {"build", "boundaries", "replay"})

    def test_rejects_missing_stage(self):
        with self.assertRaisesRegex(ValueError, "replay"):
            self.run_check(fixture(omit="replay"))

    def test_rejects_scratch(self):
        with self.assertRaisesRegex(ValueError, "private/scratch"):
            self.run_check(fixture(scratch=32))

    def test_rejects_private_segment(self):
        with self.assertRaisesRegex(ValueError, "private/scratch"):
            self.run_check(fixture(private=32))

    def test_rejects_resource_regression(self):
        with self.assertRaisesRegex(ValueError, "resources fail"):
            self.run_check(fixture(vgprs=241))

    def test_rejects_occupancy_regression(self):
        with self.assertRaisesRegex(ValueError, "resources fail"):
            self.run_check(fixture(occupancy=5))

    def test_rejects_lds_regression(self):
        with self.assertRaisesRegex(ValueError, "resources fail"):
            self.run_check(fixture(lds=43521))

    def test_rejects_wrong_execution_geometry(self):
        for arguments in ({"wave32": 0}, {"wgp": 0}, {"maximum_workgroup": 128}):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "geometry fails"):
                    self.run_check(fixture(**arguments))

    def test_rejects_ambiguous_stage_symbol(self):
        with self.assertRaisesRegex(ValueError, "build: expected one exact stage symbol"):
            self.run_check(fixture(duplicate="build"))

    def test_rejects_matrix_profile(self):
        with self.assertRaisesRegex(ValueError, "matrix opcodes"):
            self.run_check(fixture(wmma=True))


if __name__ == "__main__":
    unittest.main()
