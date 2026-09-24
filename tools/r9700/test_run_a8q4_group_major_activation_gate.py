import copy
import unittest
from pathlib import Path

from tools.r9700.run_a8q4_group_major_activation_gate import SHAPES, validate_raw


def fixture(control=12800.0,candidate=9600.0):
    cells=[]
    for role,rows,columns,calls in sorted(SHAPES):
        cells.append({"role":role,"tokens":2048,"rows":rows,"columns":columns,
                      "calls":calls,"control_samples_ms":[control]*7,
                      "candidate_samples_ms":[candidate]*7})
    return {"schema_version":1,"status":"measured",
            "device":{"ordinal":0,"name":"AMD Radeon AI PRO R9700","architecture":"gfx1201","runtime_version":1,"driver_version":1},
            "warmups":3,"launches_per_interval":128,"alternating_trials":14,
            "cells":cells}


class RawGateTest(unittest.TestCase):
    def test_regression_routes_eager_through_owned_stream(self):
        source=(Path(__file__).with_name("a8q4_group_major_activation_harness.hip")).read_text()
        self.assertIn("Stream route",source)
        self.assertIn("candidate(x,t,di,c,route.p)",source)
        self.assertIn("control(x,t,di,b,route.p)",source)
        self.assertNotIn("control(x,t,di,b,nullptr)",source)

    def test_pass_requires_whole_followup(self):
        decision=validate_raw(fixture(12800,9600))
        self.assertTrue(decision["direct_gate_pass"])
        self.assertFalse(decision["production_admission"])
        self.assertTrue(decision["whole_p2048_followup_required"])

    def test_ratio_rejects(self):
        raw=fixture(12800,12000);raw["cells"][0]["candidate_samples_ms"]=[13000]*7
        self.assertFalse(validate_raw(raw)["direct_gate_pass"])

    def test_saving_rejects(self):
        self.assertFalse(validate_raw(fixture(12800,12790))["direct_gate_pass"])

    def test_missing_and_duplicate_cells_rejected(self):
        raw=fixture();raw["cells"][1]=copy.deepcopy(raw["cells"][0])
        with self.assertRaises(ValueError):validate_raw(raw)

    def test_trial_protocol_rejected(self):
        raw=fixture();raw["cells"][0]["control_samples_ms"].pop()
        with self.assertRaises(ValueError):validate_raw(raw)

    def test_noise_is_inconclusive(self):
        raw=fixture();raw["cells"][0]["candidate_samples_ms"]=[8000,12000,8000,12000,8000,12000,8000]
        self.assertEqual(validate_raw(raw)["classification"],"inconclusive")


if __name__=="__main__":unittest.main()
