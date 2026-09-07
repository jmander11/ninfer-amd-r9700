#!/usr/bin/env python3
import copy,importlib.util,unittest
from pathlib import Path
from unittest import mock
P=Path(__file__).parent/'analyze.py';S=importlib.util.spec_from_file_location('e2p',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
class T(unittest.TestCase):
 def test_command_geometry(self):
  self.assertIn('--whole-pg',M.command('layer0-fresh'));self.assertIn('--isolate-prompt-decode',M.command('layer1-append'))
  self.assertEqual(M.trace_env('layer0-fresh')['NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER'],'0')
  self.assertEqual(M.trace_env('layer1-append')['NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER'],'1')
 def test_spec_exact(self):M.validate_spec({'enabled':False,'draft_window':0,'rounds':0,'drafted_tokens':0,'accepted_tokens':0,'fallback_steps':0,'acceptance_rate':None,'acceptance_length':None,'accepted_per_position':[]})
 def test_spec_bool_rejected(self):
   with self.assertRaises(RuntimeError):M.validate_spec({'enabled':False,'draft_window':0,'rounds':False,'drafted_tokens':0,'accepted_tokens':0,'fallback_steps':0,'acceptance_rate':None,'acceptance_length':None,'accepted_per_position':[]})
 def test_candidate_mutation_rejected(self):
  old=M.load_json
  def changed(path):
   x=old(path)
   if path.name=='plan.json':x=copy.deepcopy(x);x['candidate']['algorithm_fingerprint']='00'
   return x
  with mock.patch.object(M,'load_json',side_effect=changed):
   with self.assertRaises(RuntimeError):M.validate_plan()
 def test_result_classification_gate(self):
  with mock.patch.object(M,'validate_plan',return_value=({'candidate':{},'limitations':[]},{})),mock.patch.object(M,'validate_report',return_value={}),mock.patch.object(M,'validate_process',return_value={}),mock.patch.object(M,'load',return_value=({'token':24178},b'')),mock.patch.object(M,'load_gdn',return_value=({'token':24178},b'')),mock.patch.object(M,'load_recurrent_state',side_effect=lambda p,r:({'text_layer':int(p.name[5])},b'')),mock.patch.object(Path,'read_bytes',return_value=b'x'),mock.patch.object(M,'compare',return_value={'classification':'bad'}),mock.patch.object(M,'compare_gdn',return_value={'classification':'layer1_gdn_boundaries_exact'}),mock.patch.object(M,'compare_recurrent_state',side_effect=[{'classification':'layer0_recurrent_prefix_state_exact'},{'classification':'layer1_recurrent_prefix_state_exact'}]):
   with self.assertRaises(RuntimeError):M.analyze()
 def test_success_decision(self):
  layer={'classification':'first_visible_post_mixer_layer_3','first_difference':{'snapshot_index':7,'layer':3,'half':'post_mixer','first_hidden_index':0,'mismatch_count':3467,'left_bf16_bits':48413,'right_bf16_bits':48414,'preceding_boundary_exact':True}};gdn={'classification':'layer1_gdn_boundaries_exact'}
  states=[{'classification':'layer0_recurrent_prefix_state_exact'},{'classification':'layer1_recurrent_prefix_state_exact'}]
  with mock.patch.object(M,'validate_plan',return_value=({'candidate':{'semantic_identity':'e2-at-t129'},'limitations':['functional']},{})),mock.patch.object(M,'validate_report',return_value={}),mock.patch.object(M,'validate_process',return_value={}),mock.patch.object(M,'load',return_value=({'token':24178},b'')),mock.patch.object(M,'load_gdn',return_value=({'token':24178},b'')),mock.patch.object(M,'load_recurrent_state',side_effect=lambda p,r:({'text_layer':int(p.name[5])},b'')),mock.patch.object(Path,'read_bytes',return_value=b'x'),mock.patch.object(M,'compare',return_value=layer),mock.patch.object(M,'compare_gdn',return_value=gdn),mock.patch.object(M,'compare_recurrent_state',side_effect=states),mock.patch.object(M,'identity',return_value={}):
   out=M.analyze();self.assertEqual(out['status'],'partial_success_layer01_exact_whole_text_blocked_at_layer3_full_attention');self.assertEqual(out['generated_token_ids_by_lane'],[[96558,96917]])
if __name__=='__main__':unittest.main()
