import unittest
from tools.r9700.run_a8q4_n16k16_scalar_base_gate import decision


def raw(control=4.0,candidate=3.8,spread=0.0):
    cells=[]
    for name,n,k,calls in (("value_z",12288,5120,48),("gdn_output",5120,6144,64),("mlp_down",5120,17408,64)):
        a=[128*(control+(i-3.5)*spread) for i in range(8)]
        b=[128*(candidate+(i-3.5)*spread) for i in range(8)]
        cells.append({"name":name,"tokens":2048,"rows":n,"columns":k,"calls":calls,"iterations_per_sample":128,"control_interval_samples_ms":a,"candidate_interval_samples_ms":b})
    return {"schema":"ninfer.r9700.a8q4-n16k16-scalar-base-measurement.v2","device":{"ordinal":0,"name":"AMD Radeon AI PRO R9700","architecture":"gfx1201"},"warmup_order":["A","B","B","A"],"pair_order":["AB","BA"]*4,"cells":cells}


class DecisionTest(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(decision(raw())["classification"],"pass")

    def test_weighted_miss_terminal(self):
        self.assertEqual(decision(raw(4.0,3.9))["classification"],"terminal_reject")

    def test_per_cell_regression_terminal(self):
        value=raw()
        value["cells"][0]["candidate_interval_samples_ms"]=[4.05*128]*8
        self.assertEqual(decision(value)["classification"],"terminal_reject")

    def test_unstable_inconclusive(self):
        value=raw()
        value["cells"][1]["candidate_interval_samples_ms"]=[3.8*128]*7+[5.0*128]
        self.assertEqual(decision(value)["classification"],"inconclusive")

    def test_wrong_inventory_rejected(self):
        value=raw();value["cells"].pop()
        with self.assertRaises(RuntimeError):decision(value)


if __name__=="__main__":unittest.main()
