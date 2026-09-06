from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.bench.extract_embedded_code_object import extract
from tools.r9700.check_a8q4_n16k16_scalar_base_static import check

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/ops/r9700/linear/r9700_linear.hip"
ASM = ROOT / "tools/r9700/build/r9700_linear_scalar_base_product.s"
EXECUTABLE = (
    ROOT / "build-r9700-scalar-base-qualification/tests/"
    "ninfer_r9700_a8q4_scalar_base_product_qual"
)
OBJDUMP = "/opt/rocm/core-10.0/lib/llvm/bin/llvm-objdump"
READOBJ = "/opt/rocm/core-10.0/lib/llvm/bin/llvm-readobj"


class StaticCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.assembly = ASM.read_text()
        cls.temporary = tempfile.TemporaryDirectory(prefix="scalar-base-static-test-")
        code_object = Path(cls.temporary.name) / "product.hsaco"
        extract(
            EXECUTABLE,
            code_object,
            code_symbol="a8q4g64_linear_prefill_cta_scalar_base_qualification_kernel",
        )
        cls.candidate = subprocess.run(
            [OBJDUMP, "-d", "--mcpu=gfx1201", str(code_object)],
            check=True, text=True, capture_output=True,
        ).stdout
        cls.baseline = cls.candidate
        cls.notes = subprocess.run(
            [READOBJ, "--symbols", "--notes", str(code_object)],
            check=True, text=True, capture_output=True,
        ).stdout

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_product_emission(self):
        result = check(
            self.source, self.assembly, self.candidate, self.baseline, self.notes
        )
        self.assertEqual(result["logical_vgpr"], 88)
        self.assertEqual(result["qualified_tuples"], 32)
        self.assertEqual(result["weight_code_max_voffset"], 89_128_952)
        self.assertEqual(result["activation_code_max_final_byte"], 71_303_167)
        self.assertEqual(result["weight_code_max_final_byte"], 89_128_959)
        self.assertEqual(result["activation_scale_max_final_byte"], 4_456_447)
        self.assertEqual(result["weight_scale_max_final_byte"], 5_570_559)

    def reject_asm(self, old, new):
        symbol = "a8q4g64_linear_prefill_cta_scalar_base_qualification_kernel"
        start = self.assembly.index(f"{symbol}:")
        end = self.assembly.index(f".size\t{symbol}", start)
        section = self.assembly[start:end]
        self.assertIn(old, section)
        mutated = self.assembly[:start] + section.replace(old, new, 1) + self.assembly[end:]
        with self.assertRaises(RuntimeError):
            check(self.source, mutated)

    def test_wrong_target_rejected(self):
        with self.assertRaises(RuntimeError):
            check(
                self.source,
                self.assembly.replace(
                    "amdgcn-amd-amdhsa--gfx1201",
                    "amdgcn-amd-amdhsa--gfx1200",
                    1,
                ),
            )

    def test_wrong_base_rejected(self):
        self.reject_asm("s[16:17]", "s[14:15]")

    def test_width_rejected(self):
        self.reject_asm("global_load_b64", "global_load_b32")

    def test_policy_rejected(self):
        self.reject_asm("s[16:17]", "s[16:17] th:TH_LOAD_HT scope:SCOPE_DEV")

    def test_wait_barrier_rejected(self):
        self.reject_asm("s_barrier_wait", "s_nop 0")

    def test_iu4_rejected(self):
        self.reject_asm("v_wmma_i32_16x16x32_iu4", "v_wmma_i32_16x16x16_iu8")

    def test_loaded_object_wrong_base_rejected(self):
        with self.assertRaises(RuntimeError):
            check(self.source, self.assembly,
                  self.candidate.replace("s[16:17]", "s[14:15]", 1),
                  self.baseline, self.notes)

    def test_loaded_object_metadata_rejected(self):
        with self.assertRaises(RuntimeError):
            check(self.source, self.assembly, self.candidate, self.baseline,
                  self.notes.replace(".kernarg_segment_size: 72",
                                     ".kernarg_segment_size: 64"))

    def test_loaded_object_vgpr_ceiling_rejected(self):
        with self.assertRaises(RuntimeError):
            check(self.source, self.assembly, self.candidate, self.baseline,
                  self.notes.replace(".vgpr_count:     88", ".vgpr_count:     93"))

    def test_loaded_object_backedge_rejected(self):
        lines = self.candidate.splitlines()
        position = next(i for i, line in enumerate(lines) if "s_cbranch_vccnz" in line)
        lines[position] = lines[position].replace("s_cbranch_vccnz", "s_nop")
        with self.assertRaises(RuntimeError):
            check(self.source, self.assembly, "\n".join(lines),
                  self.baseline, self.notes)

    def test_loaded_object_scale_guard_rejected(self):
        lines = self.candidate.splitlines()
        position = next(i for i, line in enumerate(lines)
                        if "global_load_d16_b16" in line)
        guard = next(i for i in range(position - 1, position - 9, -1)
                     if "s_cbranch_execz" in lines[i])
        lines[guard] = lines[guard].replace("s_cbranch_execz", "s_nop")
        with self.assertRaises(RuntimeError):
            check(self.source, self.assembly, "\n".join(lines),
                  self.baseline, self.notes)

    def test_loaded_object_iu4_signedness_rejected(self):
        self.assertIn("neg_lo:[0,1,0]", self.candidate)
        with self.assertRaises(RuntimeError):
            check(
                self.source,
                self.assembly,
                self.candidate.replace("neg_lo:[0,1,0]", "neg_lo:[1,1,0]", 1),
                self.baseline,
                self.notes,
            )

    def test_loaded_object_arithmetic_rejected(self):
        with self.assertRaises(RuntimeError):
            check(self.source, self.assembly,
                  self.candidate.replace("v_fmac_f32_e32", "v_add_nc_u32_e32", 1),
                  self.baseline, self.notes)


if __name__ == "__main__":
    unittest.main()
