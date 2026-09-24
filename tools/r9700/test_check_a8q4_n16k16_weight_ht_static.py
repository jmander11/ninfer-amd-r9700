from __future__ import annotations
import unittest
from pathlib import Path
from tools.r9700.check_a8q4_n16k16_weight_ht_static import check
from tools.r9700.patch_a8q4_n16k16_weight_ht_assembly import patch
ROOT=Path(__file__).resolve().parents[2];B=ROOT/"tools/r9700/build"
class TestStatic(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.base=(B/"a8q4_n16k16_weight_ht_baseline.hsaco").read_bytes();cls.cand=(B/"a8q4_n16k16_weight_ht_candidate.hsaco").read_bytes();cls.bd=(B/"a8q4_n16k16_weight_ht_baseline.objdump").read_text();cls.cd=(B/"a8q4_n16k16_weight_ht_candidate.objdump").read_text();cls.notes=(B/"a8q4_n16k16_weight_ht_candidate.notes").read_text()
 def test_real(self):self.assertEqual(check(self.base,self.cand,self.bd,self.cd,self.notes)["weight_ht_b64"],2)
 def test_patcher_exact(self):
  base,cand,_=patch((B/"a8q4_n16k16_weight_ht_fatbin.bin").read_bytes(),self.bd);self.assertEqual(base,self.base);self.assertEqual(cand,self.cand)
 def test_reject_unrelated_byte(self):
  x=bytearray(self.cand);x[100]^=1
  with self.assertRaisesRegex(RuntimeError,"equality"):check(self.base,bytes(x),self.bd,self.cd,self.notes)
 def test_reject_wrong_modifier_disassembly(self):
  x=self.cd.replace("TH_LOAD_HT","TH_LOAD_NT",1)
  with self.assertRaisesRegex(RuntimeError,"spelling"):check(self.base,self.cand,self.bd,x,self.notes)
if __name__=="__main__":unittest.main()
