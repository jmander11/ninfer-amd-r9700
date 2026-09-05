import tempfile
import unittest
from pathlib import Path

from tools.r9700.check_a8q4_group_major_activation_static import check, source_gate


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/r9700/a8q4_group_major_activation_qual.hip"
ASSEMBLY = ROOT / "tools/r9700/build/a8q4_group_major_activation_qual.s"


class SourceGateTest(unittest.TestCase):
    def test_current_source(self):
        source_gate(SOURCE.read_text())

    def reject(self, old: str, new: str):
        value = SOURCE.read_text()
        self.assertIn(old, value)
        with self.assertRaises(RuntimeError):
            source_gate(value.replace(old, new, 1))

    def test_token_major_code_rejected(self):
        self.reject("(static_cast<std::size_t>(g)*tokens+token)*32U+byte",
                    "static_cast<std::size_t>(token)*999U+g*32U+byte")

    def test_token_major_scale_rejected(self):
        self.reject("static_cast<std::size_t>(g)*tokens+token",
                    "static_cast<std::size_t>(token)*999U+g")

    def test_missing_fused_publisher_rejected(self):
        self.reject("quantize<true>", "quantize<false>")

    def test_wrong_reconstruction_rejected(self):
        self.reject("float(low[nh][e]+16*high[nh][e])",
                    "float(low[nh][e]+8*high[nh][e])")

    def test_wrong_fused_boundary_rejected(self):
        self.reject("float(hip_bfloat16((a/(1.0F+expf(-a)))*b))",
                    "(a/(1.0F+expf(-a)))*b")

    def test_current_assembly(self):
        check(SOURCE.read_text(),ASSEMBLY.read_text())

    def test_wrong_iu4_signedness_rejected(self):
        asm=ASSEMBLY.read_text().replace("neg_lo:[0,1,0]","neg_lo:[1,1,0]",1)
        with self.assertRaises(RuntimeError):check(SOURCE.read_text(),asm)

    def test_resource_regression_rejected(self):
        asm=ASSEMBLY.read_text().replace(".amdhsa_group_segment_fixed_size 17152",
                                         ".amdhsa_group_segment_fixed_size 17156",1)
        with self.assertRaises(RuntimeError):check(SOURCE.read_text(),asm)

    def test_early_successor_drain_rejected(self):
        asm=ASSEMBLY.read_text();symbol="a8q4_group_major_prefill"
        start=asm.index(".protected\t"+symbol);wmma=asm.index("v_wmma_i32_16x16x32_iu4",start)
        load=max(asm.rfind("global_load_b32",start,wmma),asm.rfind("global_load_b64",start,wmma))
        end=asm.index("\n",load)+1
        asm=asm[:end]+"\ts_wait_loadcnt 0x0\n"+asm[end:]
        with self.assertRaises(RuntimeError):check(SOURCE.read_text(),asm)


if __name__ == "__main__":
    unittest.main()
