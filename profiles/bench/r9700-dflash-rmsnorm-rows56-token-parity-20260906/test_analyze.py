import copy, importlib.util, math, unittest
from unittest import mock
from pathlib import Path
P=Path(__file__).with_name('analyze.py');S=importlib.util.spec_from_file_location('rows56_parity_analyze',P);A=importlib.util.module_from_spec(S);S.loader.exec_module(A)
def spec(d=True):
 return {'enabled':d,'draft_window':4 if d else 0,'rounds':20 if d else 0,'drafted_tokens':80 if d else 0,'accepted_tokens':7 if d else 0,'fallback_steps':0,'acceptance_rate':0.0875 if d else None,'acceptance_length':1.35 if d else None,'accepted_per_position':[4,2,1,0] if d else []}
class T(unittest.TestCase):
 def test_exact_commands(self):
  for r in ('control-ordinary','control-dflash','candidate-dflash'):
   c=A.command(r,A.RESULTS/f'{r}.json');self.assertIn('129,27',c);self.assertEqual(c[0],str(A.CANDIDATE if r.startswith('candidate') else A.CONTROL));self.assertEqual('--spec' in c,r.endswith('dflash'))
 def test_token_domain(self):
  self.assertEqual(A.token_list(A.ORDINARY,'x'),A.ORDINARY)
  for v in ([True]*28,[0]*27+[248077]):
   with self.assertRaisesRegex(RuntimeError,'token list'):A.token_list(v,'x')
 def test_spec_accounting(self):
  A.validate_spec(spec(),True,'x')
  v=spec();v['accepted_per_position']=[4,2,0,0]
  with self.assertRaisesRegex(RuntimeError,'accounting'):A.validate_spec(v,True,'x')
 def test_nonfinite_rate(self):
  v=spec();v['acceptance_rate']=math.nan
  with self.assertRaisesRegex(RuntimeError,'rate'):A.validate_spec(v,True,'x')
 def test_recomputes_rates_and_requires_test_rep_equality(self):
  v=spec();v['acceptance_rate']=0.1
  with self.assertRaisesRegex(RuntimeError,'derived rate'):A.validate_spec(v,True,'x')
  v=spec();v['drafted_tokens']=79
  with self.assertRaisesRegex(RuntimeError,'derived rate'):A.validate_spec(v,True,'x')
  test=spec();rep=copy.deepcopy(test);rep['accepted_per_position']=[3,3,1,0]
  with self.assertRaisesRegex(RuntimeError,'test/rep speculative summaries differ'):A.validate_spec_pair(test,rep,True,'x')
 def test_exact_baseline_lists(self):
  self.assertEqual(A.ORDINARY[:27],A.DFLASH[:27]);self.assertEqual((A.ORDINARY[27],A.DFLASH[27]),(95946,98003))
 def test_classification_and_baseline_gates(self):
  restored={'control-ordinary':A.ORDINARY,'control-dflash':A.DFLASH,'candidate-dflash':A.ORDINARY}
  self.assertEqual(A.classify(restored),(True,None))
  rejected=dict(restored);rejected['candidate-dflash']=A.DFLASH
  self.assertEqual(A.classify(rejected),(False,27))
  for role in ('control-ordinary','control-dflash'):
   changed=dict(restored);changed[role]=[0]*28
   with self.subTest(role=role),self.assertRaisesRegex(RuntimeError,'retained authority'):A.classify(changed)
 def test_analyze_wires_all_three_roles(self):
  for candidate,status,index in ((A.ORDINARY,'restored_exact_token_parity_through_index27',None),(A.DFLASH,'not_restored',27)):
   tokens={'control-ordinary':A.ORDINARY,'control-dflash':A.DFLASH,'candidate-dflash':candidate}
   with self.subTest(status=status),mock.patch.object(A,'validate_authorities',return_value={'bound':True}),mock.patch.object(A,'validate_process',side_effect=lambda role:{'role':role}),mock.patch.object(A,'validate_report',side_effect=lambda role:({'role':role},tokens[role])):
    value=A.analyze()
   self.assertEqual(value['status'],status);self.assertEqual(value['candidate_first_difference_index'],index)
if __name__=='__main__':unittest.main()
