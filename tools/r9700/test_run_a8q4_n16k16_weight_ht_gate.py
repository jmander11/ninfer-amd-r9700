from __future__ import annotations
import json, os, tempfile, unittest
from pathlib import Path
from unittest import mock
from tools.r9700 import run_a8q4_n16k16_weight_ht_gate as gate

def measurement(candidate:float=400.0)->dict:
 return {"schema":"ninfer.r9700.a8q4-n16k16-weight-ht-measurement.v1",
  "device":{"ordinal":0,"name":"AMD Radeon AI PRO R9700","architecture":"gfx1201"},
  "tokens":2048,"rows":5120,"columns":17408,"calls":64,"iterations_per_sample":128,
  "control_interval_samples_ms":[512.0]*7,"candidate_interval_samples_ms":[candidate]*7}

class TestGate(unittest.TestCase):
 def test_power_namespaces_and_identity(self):
  state={"card0":{"Card Series":"AMD Radeon AI PRO R9700",
                   "GFX Version":"gfx1201","Performance Level":"auto"}}
  gate.validate_power_state(state,"auto")
  for bad,level in [({"card2":state["card0"]},"auto"),(state,"manual"),
                    ({"card0":{**state["card0"],"GFX Version":"gfx1200"}},"auto")]:
   with self.assertRaisesRegex(RuntimeError,"ROCm device0"):
    gate.validate_power_state(bad,level)
 def test_report_power_uses_rocm_card0_and_drm_card2(self):
  static_command=[str(gate.PY),"-m","tools.r9700.check_a8q4_n16k16_weight_ht_static",
                  "--baseline",str(gate.BASE_ASM),"--candidate",str(gate.ASM),
                  "--baseline-disassembly",str(gate.BASE_DISASM),
                  "--candidate-disassembly",str(gate.DISASM),"--notes",str(gate.NOTES)]
  power={"command":["/opt/rocm/bin/rocm-smi","-d","0","--showproductname",
                    "--showprofile","--showperflevel","--json"],
         "stdout":{"card0":{"Card Series":"AMD Radeon AI PRO R9700",
                              "GFX Version":"gfx1201","Performance Level":"auto"}},
         "sysfs_path":str(gate.POWER),"sysfs_value":"auto"}
  raw=measurement()
  report={"schema":"ninfer.r9700.a8q4-n16k16-weight-device-ht-gate.v1",
          "command":[str(gate.PY),"-m","tools.r9700.run_a8q4_n16k16_weight_ht_gate",
                     "--benchmark","--output",str(gate.REPORT)],
          "hashes_sha256":{str(p.relative_to(gate.ROOT)):gate.sha(p) for p in gate.INPUTS},
          "static":{"command":static_command,
                    "stdout":"PASS exact_hsaco_equality=true"},
          "decision":gate.validate_measurement(raw),"measurement":raw,
          "bound_regression":{"command":[str(gate.BINARY),"--regression"],
                              "returncode":0,"stdout":"PASS regression","stderr":""},
          "resources":{"logical_vgpr":92,"architectural_vgpr":96,"lds_bytes":17152,
                       "occupancy_waves_per_eu":16,"scratch_bytes":0,"sgpr_spills":0,
                       "vgpr_spills":0,"native_signed_iu4_sites":8,
                       "weight_ht_b64_static_sites":2,"activation_policy":"default",
                       "scale_policy":"default","complete_hsaco_otherwise_identical":True,
                       "production_kernarg_bytes":72},
          "power_before":power,"power_after":power}
  gate.validate_report_payload(report)
  report["power_after"]={**power,"stdout":{"card2":power["stdout"]["card0"]}}
  with self.assertRaisesRegex(RuntimeError,"power_after"):
   gate.validate_report_payload(report)
 def test_pass_thresholds(self):
  d=gate.validate_measurement(measurement(400.0))
  self.assertTrue(d["admission_pass"]);self.assertEqual(d["classification"],"pass")
 def test_first_miss_is_terminal(self):
  d=gate.validate_measurement(measurement(480.0))
  self.assertFalse(d["admission_pass"]);self.assertEqual(d["classification"],"terminal_reject")
 def test_unstable_is_inconclusive(self):
  x=measurement(480.0);x["candidate_interval_samples_ms"][-1]=700.0
  d=gate.validate_measurement(x)
  self.assertFalse(d["admission_pass"]);self.assertEqual(d["classification"],"inconclusive")
 def test_reject_wrong_shape_and_short_interval(self):
  x=measurement();x["rows"]=6144
  with self.assertRaisesRegex(RuntimeError,"workload"):gate.validate_measurement(x)
  x=measurement();x["candidate_interval_samples_ms"]=[20.0]*7
  with self.assertRaisesRegex(RuntimeError,"too short"):gate.validate_measurement(x)
 def test_atomic_create_only_and_late_rollback(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"report.json";payload={"x":1}
   with mock.patch.object(gate,"validate_report_payload",return_value=None):gate.publish(p,payload)
   self.assertEqual(json.loads(p.read_text()),payload)
   with self.assertRaises(RuntimeError):gate.publish(p,payload)
   q=Path(td)/"late.json"
   with mock.patch.object(gate,"validate_report_payload",side_effect=RuntimeError("late")):
    with self.assertRaisesRegex(RuntimeError,"late"):gate.publish(q,payload)
   self.assertFalse(os.path.lexists(q));self.assertFalse(any(Path(td).glob(".*.pending")))

if __name__=="__main__":unittest.main()
