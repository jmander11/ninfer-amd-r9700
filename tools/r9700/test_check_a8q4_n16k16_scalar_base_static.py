from pathlib import Path
import unittest

from tools.r9700.check_a8q4_n16k16_scalar_base_static import check

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/r9700/a8q4_n16k16_scalar_base_qual.hip"
ASM = ROOT / "tools/r9700/build/a8q4_n16k16_scalar_base_qual.s"


class StaticCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.assembly = ASM.read_text()
        cls.candidate=(ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_candidate.objdump").read_text()
        cls.baseline=(ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_baseline.objdump").read_text()
        cls.notes=(ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_candidate.notes").read_text()

    def test_real_emission(self):
        self.assertEqual(check(self.source, self.assembly,self.candidate,self.baseline,self.notes)["max_voffset"], 44_564_472)

    def reject_asm(self, old, new):
        self.assertIn(old, self.assembly)
        with self.assertRaises(RuntimeError):
            check(self.source, self.assembly.replace(old, new, 1))

    def test_wrong_base_rejected(self):
        self.reject_asm("s[16:17]", "s[14:15]")

    def test_width_rejected(self):
        self.reject_asm("global_load_b64", "global_load_b32")

    def test_policy_rejected(self):
        self.reject_asm("s[16:17]", "s[16:17] th:TH_LOAD_HT scope:SCOPE_DEV")

    def test_vector_pair_rejected(self):
        self.reject_asm("v62, s[16:17]", "v[62:63], s[16:17]")

    def test_wait_barrier_rejected(self):
        self.reject_asm("s_barrier_wait", "s_nop 0")

    def test_iu4_rejected(self):
        self.reject_asm("v_wmma_i32_16x16x32_iu4", "v_wmma_i32_16x16x16_iu8")

    def test_offset_progression_rejected(self):
        with self.assertRaises(RuntimeError):
            check(self.source.replace("woff_cur+=512", "woff_cur+=256"), self.assembly)

    def test_signed_cursor_rejected(self):
        with self.assertRaises(RuntimeError):
            check(self.source.replace("std::uint32_t woff_cur", "std::int32_t woff_cur"), self.assembly)

    def test_loaded_object_wrong_base_rejected(self):
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate.replace("s[16:17]","s[14:15]",1),self.baseline,self.notes)

    def test_loaded_object_early_drain_rejected(self):
        marker="\tglobal_load_b32 v77"
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate.replace(marker,"\ts_wait_loadcnt 0x0\n"+marker,1),self.baseline,self.notes)

    def test_loaded_object_metadata_rejected(self):
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate,self.baseline,self.notes.replace(".kernarg_segment_size: 72",".kernarg_segment_size: 64"))

    def test_loaded_object_backedge_rejected(self):
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate.replace("s_cbranch_vccnz","s_nop",1),self.baseline,self.notes)

    def test_loaded_object_arithmetic_rejected(self):
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate.replace("v_fmac_f32_e32","v_add_nc_u32_e32",1),self.baseline,self.notes)

    def test_loaded_object_w_after_wmma_rejected(self):
        lines=self.candidate.splitlines();w=next(i for i,x in enumerate(lines) if "global_load_b64 v[53:54]" in x);line=lines.pop(w);first=next(i for i,x in enumerate(lines) if "v_wmma_i32_16x16x32_iu4" in x);lines.insert(first+1,line)
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,"\n".join(lines),self.baseline,self.notes)

    def test_loaded_object_barrier_order_rejected(self):
        lines=self.candidate.splitlines();signals=[i for i,x in enumerate(lines) if "s_barrier_signal" in x];waits=[i for i,x in enumerate(lines) if "s_barrier_wait" in x];lines[signals[1]],lines[waits[1]]=lines[waits[1]],lines[signals[1]]
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,"\n".join(lines),self.baseline,self.notes)

    def test_loaded_object_fmac_before_conversion_rejected(self):
        lines=self.candidate.splitlines();f=next(i for i,x in enumerate(lines) if "v_fmac_f32_e32" in x);line=lines.pop(f);c=next(i for i,x in enumerate(lines) if "v_cvt_f32_i32_e32" in x);lines.insert(c,line)
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,"\n".join(lines),self.baseline,self.notes)

    def test_exact_metadata_contract_rejected(self):
        for old,new in ((".offset:         8",".offset:         9"),(".value_kind:     global_buffer",".value_kind:     by_value"),(".kernarg_segment_align: 8",".kernarg_segment_align: 4"),(".sgpr_count:     23",".sgpr_count:     99"),(".uses_dynamic_stack: false",".uses_dynamic_stack: true")):
            with self.subTest(old=old),self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate,self.baseline,self.notes.replace(old,new,1))

    def test_loaded_object_destination_rejected(self):
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate.replace("v[53:54], v62","v[55:56], v62",1),self.baseline,self.notes)

    def test_loaded_object_branch_target_rejected(self):
        with self.assertRaises(RuntimeError):check(self.source,self.assembly,self.candidate.replace("s_cbranch_vccnz 65341","s_cbranch_vccnz 0",1),self.baseline,self.notes)


if __name__ == "__main__":
    unittest.main()
