import unittest
from tools.r9700.run_a8g128_q4g128_n16k16_gate import BATCH_CONTRACT, CONTROL_CONTRACT, STREAM_CONTRACT, TOKENS, validate

def sample():
 rows=[]
 for role,n,k,c in (("value_z",12288,5120,48),("gdn_output",5120,6144,48),("mlp_down",5120,17408,64)):
  for t in TOKENS:
   boundary="canonical_a8g64_prepare_plus_gemm" if role!="mlp_down" else ("canonical_fused_silu_a8g64_prepare_plus_gemm" if t==2048 else "product_eager_silu_mul_bf16_strided_plus_canonical_a8g64_prepare_plus_gemm")
   iterations=128 if t==2048 else 1024
   rows.append({"role":role,"rows":n,"columns":k,"calls":c,"tokens":t,"iterations_per_sample":iterations,"control_boundary":boundary,"control_ms":2.0,"candidate_ms":1.0,"ratio":.5,"control_interval_samples_ms":[2.0*iterations]*7,"candidate_interval_samples_ms":[1.0*iterations]*7})
 return {"schema_version":2,"status":"measured","timing_weight_contract":"identical_dense_logical_signed_q4_codes_and_paired_equal_scales","timing_stream_contract":STREAM_CONTRACT,"timing_batch_contract":BATCH_CONTRACT,"control_boundary_contract":CONTROL_CONTRACT,"device":{"ordinal":0,"name":"AMD Radeon AI PRO R9700","architecture":"gfx1201","runtime_version":1,"driver_version":1},"trials":rows}
class Test(unittest.TestCase):
 def test_pass(self):self.assertTrue(validate(sample())["admission_pass"])
 def test_missing_cell(self):
  x=sample();x["trials"].pop()
  with self.assertRaises(RuntimeError):validate(x)
 def test_default_stream_contract_rejected(self):
  x=sample();x["timing_stream_contract"]="default_stream_events"
  with self.assertRaises(RuntimeError):validate(x)
 def test_regression_rejected(self):
  x=sample();it=x["trials"][0]["iterations_per_sample"];x["trials"][0]["candidate_ms"]=3;x["trials"][0]["candidate_interval_samples_ms"]=[3.0*it]*7;x["trials"][0]["ratio"]=1.5
  self.assertEqual(validate(x)["classification"],"reject")
 def test_wrong_control_boundary_rejected(self):
  x=sample();x["trials"][-2]["control_boundary"]="qualification_local_fused"
  with self.assertRaises(RuntimeError):validate(x)
 def test_single_iteration_p2048_rejected(self):
  x=sample();x["trials"][-1]["iterations_per_sample"]=1
  with self.assertRaises(RuntimeError):validate(x)
 def test_unstable_samples_are_inconclusive(self):
  x=sample();r=x["trials"][0];r["control_interval_samples_ms"][0]*=1.2
  self.assertEqual(validate(x)["classification"],"inconclusive")
 def test_weighted_upper_below_gate_rejects(self):
  x=sample()
  for r in x["trials"]:
   it=r["iterations_per_sample"];r["candidate_ms"]=1.99;r["ratio"]=.995;r["candidate_interval_samples_ms"]=[1.99*it]*7
  self.assertEqual(validate(x)["classification"],"reject")
if __name__=="__main__":unittest.main()
