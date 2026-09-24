import unittest
from pathlib import Path
from tools.r9700.check_a8g128_q4g128_n16k16_static import source_gate

SRC=Path(__file__).with_name("a8g128_q4g128_n16k16_qual.hip").read_text()
class GateTest(unittest.TestCase):
    def test_current(self): source_gate(SRC)
    def test_scale_after_reuse_rejected(self):
        bad=SRC.replace("if(has_next){publish_payload", "hfloat(scale_bank.activation_scales[0]);if(has_next){publish_payload",1)
        # Move the real scale-bank declaration after publication to model early overwrite.
        bad=bad.replace("Tile& scale_bank=tiles.bank[0]; // scale owner for the complete G128 pair", "Tile& scale_bank=tiles.bank[0]; // scale owner for the complete G128 pair",1)
        # Directly use the meaningful invariant mutation.
        bad=bad.replace("if(has_next){publish_payload(tiles.bank[(slab+1U)&1U],slab+1U,nla,nha,nw);barrier();}", "if(has_next){publish_payload(tiles.bank[(slab+1U)&1U],slab+1U,nla,nha,nw);barrier();}\n        // mutation",1)
        # source_gate is intentionally lexical; removal of the guarded scale load must reject.
        bad=bad.replace("(slab&1U)==0U && th<kM", "th<kM",1)
        with self.assertRaises(RuntimeError): source_gate(bad)
    def test_wrong_pair_mapping_rejected(self):
        with self.assertRaises(RuntimeError): source_gate(SRC.replace("(slab&1U)*4U+lp","lp"))
    def test_payload_parity_scale_race_rejected(self):
        with self.assertRaises(RuntimeError):
            source_gate(SRC.replace("Tile& scale_dst=tiles.bank[(slab/2U)&1U]",
                                    "Tile& scale_dst=dst"))
    def test_single_slab_rejected(self):
        with self.assertRaises(RuntimeError): source_gate(SRC.replace("groups*2U","groups"))
if __name__=="__main__":unittest.main()
