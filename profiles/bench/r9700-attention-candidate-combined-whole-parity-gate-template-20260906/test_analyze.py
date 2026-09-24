#!/usr/bin/env python3
from __future__ import annotations
import copy,importlib.util,unittest
from pathlib import Path
P=Path(__file__).parent
s=importlib.util.spec_from_file_location('combined_gate',P/'analyze.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class TestCombinedGate(unittest.TestCase):
 def test_bound_plan_and_build(self):m.validate_plan()
 def test_plan_semantic_mutations_rejected(self):
  cases=(('claim','changed'),('selectors',{}),('decision',{**m.REQUIRED_DECISION,'pass_status':'target_parity_failed'}),('limitations',[]),('results_directory','elsewhere'))
  for key,value in cases:
   plan=copy.deepcopy(m.PLAN);plan[key]=value
   with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_plan_values(plan,m.BUILD)
 def test_exact_three_commands(self):
  for stem in m.ARMS:
   command=m.command(stem,Path('/tmp/report.json'))
   self.assertIn('--no-device-graph',command);self.assertEqual(command[command.index('--concurrency')+1],'1')
  self.assertIn('--isolate-prompt-decode',m.command('ordinary-append'))
  self.assertIn('--lm-head-draft',m.command('dflash-fresh'))
 def test_trace_roles(self):
  self.assertEqual(m.trace_env('ordinary-fresh')['NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE'],'target-ordinary-frontier130')
  self.assertEqual(m.trace_env('dflash-fresh')['NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE'],'target-dflash-frontier130-column0')
  self.assertNotIn('NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE',m.trace_env('ordinary-append'))
 def test_spec_contracts(self):
  ordinary={'enabled':False,'draft_window':0,'rounds':0,'drafted_tokens':0,'accepted_tokens':0,'fallback_steps':0,'acceptance_rate':None,'acceptance_length':None,'accepted_per_position':[]};m.validate_spec(ordinary,False)
  d={'enabled':True,'draft_window':4,'rounds':10,'drafted_tokens':35,'accepted_tokens':8,'fallback_steps':9,'acceptance_rate':8/35,'acceptance_length':1.8,'accepted_per_position':[4,2,1,1]};m.validate_spec(d,True)
  for mutation in ({**d,'accepted_tokens':True},{**d,'acceptance_rate':0.5},{**d,'accepted_per_position':[8,0,0,1]}):
   with self.assertRaises(RuntimeError):m.validate_spec(mutation,True)
 def test_pass_and_fail_classification(self):
  base=[24178]+list(range(27));self.assertEqual(m.classify_sequences(base,base,base,'exact')[0],m.PLAN['decision']['pass_status'])
  changed=base.copy();changed[21]+=1
  result=m.classify_sequences(base,base,changed,'exact');self.assertEqual((result[0],result[4]),('target_parity_failed',21))
  self.assertEqual(m.classify_sequences(base,base,base,'mismatch')[0],'text_parity_failed')
 def test_token_domain_is_strict(self):
  base=[24178]+list(range(27))
  for bad in (True,-1,m.TOKEN_DOMAIN):
   changed=base.copy();changed[2]=bad
   with self.assertRaises(RuntimeError):m.classify_sequences(base,changed,base,'exact')

if __name__=='__main__':unittest.main()
