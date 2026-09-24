#!/usr/bin/env python3
"""Fail-closed analyzer for the exact three-load combined attention parity gate."""
from __future__ import annotations
import hashlib,importlib.util,json,math,os,sys
from pathlib import Path
ROOT=Path('/ssdpool2nvme/local_llm/ninfer-amd-r9700');PACKAGE=Path(__file__).resolve().parent;RESULTS=PACKAGE/'results'
PLAN=json.loads((PACKAGE/'plan.json').read_text());BUILD=json.loads((PACKAGE/'build-provenance.json').read_text())
SOURCE=Path(PLAN['source']['worktree']);EXE=Path(BUILD['outputs']['benchmark']['path']);ARTIFACT=Path(PLAN['artifact']['path']);CORPUS=Path(PLAN['history']['source_corpus']['path']);HISTORY=Path(PLAN['history']['p129_fixture']['path'])
ARMS=('ordinary-fresh','ordinary-append','dflash-fresh');TOKEN_DOMAIN=248077
PASS_STATUS='exact_combined_attention_candidate_whole_semantic_parity'
REQUIRED_CLAIM='functional-only three-arm gate for exact Text fresh-versus-append continuation and target ordinary-versus-DFlash K4/W5 accept-commit parity'
REQUIRED_SELECTORS={'NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE':'1','NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE':'1','NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE':'1','NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE':'1','NINFER_R9700_ATTENTION_PARITY_CANDIDATE':'1','NINFER_R9700_DFLASH_SMALL_T_CANDIDATE':'0','NINFER_R9700_Q4_ACTIVATION_BITS':'8','NINFER_R9700_W8_ACTIVATION_BITS':'8','NINFER_R9700_FP8_QK_WMMA':'1','NINFER_R9700_KV_VALUE_GROUP':'16'}
REQUIRED_DECISION={'pass_status':PASS_STATUS,'text_requirement':'ordinary-fresh and ordinary-append retained 28-token sequences and P129 tail traces are exact','target_requirement':'ordinary-fresh and dflash-fresh retained 28-token sequences are exact and the DFlash decision trace reconstructs all licensed tokens/frontiers/counters','localization':'retain Text tail, target layer-boundary, layer3-attention and decision-trace evidence','model_loads':3}
REQUIRED_LIMITATIONS=['functional eager C1 diagnostic only; synchronous trace copies invalidate all timing','qualifies only the exact combined selector build and does not isolate causal credit among selectors','exact public continuation and accept-commit accounting do not assert bitwise equality of every private cache/state element','no production routing or performance claim']
def fail(m):raise RuntimeError(m)
def pairs(items):
 out={}
 for k,v in items:
  if k in out:fail(f'duplicate JSON key: {k}')
  out[k]=v
 return out
def load(p):
 x=json.loads(Path(p).read_text(),object_pairs_hook=pairs,parse_constant=lambda v:fail(f'nonfinite JSON {v}'))
 if not isinstance(x,dict):fail(f'not an object: {p}')
 return x
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity(p):
 p=Path(p).resolve(strict=True);return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}
def exact(x):return identity(Path(x['path']))==x
def validate_closure(path,seen=None):
 path=Path(path).resolve(strict=True);seen=set() if seen is None else seen
 if path in seen:return
 seen.add(path);listed=set();count=0
 for line in path.read_text().splitlines():
  fields=line.split('  ',1)
  if len(fields)!=2 or len(fields[0])!=64 or any(c not in '0123456789abcdef' for c in fields[0]):fail(f'malformed closure {path}')
  member=Path(fields[1])
  if not member.is_absolute():
   rooted=ROOT/member
   member=rooted if rooted.exists() or rooted.is_symlink() else path.parent/member
  member=member.resolve(strict=True)
  if member in listed or digest(member)!=fields[0]:fail(f'closure member mismatch {member}')
  listed.add(member);count+=1
 if count==0:fail(f'empty closure {path}')
def integer(v,n):
 if isinstance(v,bool) or not isinstance(v,int):fail(f'{n} not integer')
 return v
def finite(v,n):
 if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):fail(f'{n} not finite')
 return float(v)
def token(v,n):
 v=integer(v,n)
 if not 0<=v<TOKEN_DOMAIN:fail(f'{n} outside token domain')
 return v
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
TAIL=module('combined_tail',SOURCE/'tools/bench/compare_prefill_p129_tail_trace.py')
LAYER=module('combined_layer',SOURCE/'tools/bench/compare_qwen3_layer_boundary_trace.py')
ATTN=module('combined_attention',SOURCE/'tools/bench/analyze_qwen3_layer3_attention_trace.py')
def command(stem,report=None):
 report=RESULTS/f'{stem}.json' if report is None else Path(report)
 corpus=CORPUS if stem=='ordinary-append' else HISTORY
 c=[str(EXE),'--weights',str(ARTIFACT),'--corpus',str(corpus),'--device','0','--concurrency','1']
 c+=(['-pg','128,27'] if stem=='ordinary-append' else ['--whole-pg','129,27'])
 c+=['--prefill-chunk','4096','--kv-capacity','workload']
 if stem=='dflash-fresh':c+=['--spec','dflash','--draft-tokens','4','--dflash-verify-width','5','--lm-head-draft']
 else:c+=['--draft-tokens','0']
 c+=['--retain-token-ids']
 if stem=='ordinary-append':c+=['--isolate-prompt-decode']
 return c+['--no-device-graph','--output','json','--output-file',str(report),'-r','1','--warmup','0']
def trace_env(stem):
 b=RESULTS/stem;out={'NINFER_DFLASH_DECISION_TRACE_OUT':str(b)+'.decision.json'}
 if stem!='dflash-fresh':out|={'NINFER_QWEN3_PREFILL_P129_TRACE':'1','NINFER_QWEN3_PREFILL_P129_TRACE_OUT':str(b)+'.tail.json'}
 if stem!='ordinary-append':
  role='target-ordinary-frontier130' if stem=='ordinary-fresh' else 'target-dflash-frontier130-column0'
  out|={'NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE':role,'NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST':str(b)+'.layer.json','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR':str(b)+'.layer.bin','NINFER_QWEN3_LAYER3_ATTENTION_TRACE_MANIFEST':str(b)+'.attention.json','NINFER_QWEN3_LAYER3_ATTENTION_TRACE_SIDECAR':str(b)+'.attention.bin','NINFER_QWEN3_LAYER3_TRACE_SOURCE_COMMIT':PLAN['source']['commit'],'NINFER_QWEN3_LAYER3_TRACE_SOURCE_TREE':PLAN['source']['tree'],'NINFER_QWEN3_LAYER3_TRACE_EXECUTABLE_SHA256':BUILD['outputs']['benchmark']['sha256'],'NINFER_QWEN3_LAYER3_TRACE_ARTIFACT_SHA256':PLAN['artifact']['sha256'],'NINFER_QWEN3_LAYER3_TRACE_HISTORY_SHA256':PLAN['history']['p129_fixture']['sha256'],'NINFER_QWEN3_LAYER3_TRACE_CORPUS_SHA256':PLAN['history']['source_corpus']['sha256'],'NINFER_QWEN3_LAYER3_TRACE_POWER_SOURCE':PLAN['hardware']['power_profile_source']}
 return out
def validate_plan_values(plan,build):
 if plan.get('artifact_type')!='ninfer_r9700_attention_candidate_combined_whole_parity_gate_plan' or plan.get('schema_version')!=2 or plan.get('status')!='prepared_cpu_only_awaiting_gpu_execution' or plan.get('timing_evidence_eligible') is not False or plan.get('production_routing_authorized') is not False:fail('plan authority')
 if plan.get('claim')!=REQUIRED_CLAIM or plan.get('selectors')!=REQUIRED_SELECTORS or plan.get('decision')!=REQUIRED_DECISION or plan.get('limitations')!=REQUIRED_LIMITATIONS or plan.get('results_directory')!='results':fail('plan semantic contract')
 if plan.get('source')!=build.get('source') or plan.get('arms')!=list(ARMS) or plan.get('build_receipt')!='build-provenance.json':fail('plan binding')
 if plan.get('workload')!={'concurrency':1,'maximum_product_concurrency':4,'prefill_chunk':4096,'warmup':0,'repetitions':1,'device_graph':False,'decode_tokens':27,'retained_tokens':28,'retain_token_ids':True,'dflash_draft_tokens':4,'dflash_verify_width':5,'dflash_proposal_head':'optimized'}:fail('workload')
 if plan.get('hardware')!={'device':0,'name':'AMD Radeon AI PRO R9700','architecture':'gfx1201','wave_size':32,'pci':'0000:13:00.0','power_profile_source':'/sys/class/drm/card2/device/power_dpm_force_performance_level','power_profile':'auto'}:fail('hardware plan')
def validate_plan():
 validate_plan_values(PLAN,BUILD)
 for group in ('artifact',):
  if not exact({k:PLAN[group][k] for k in ('path','bytes','sha256')}):fail(group)
 for x in PLAN['history'].values():
  if isinstance(x,dict) and not exact(x):fail('history identity')
 if [int(x) for x in HISTORY.read_text().split()]!=[int(x) for x in CORPUS.read_text().split()][:128]+[24178]:fail('history derivation')
 for x in PLAN['prior_authorities'].values():
  if not exact(x):fail('prior authority')
  if Path(x['path']).suffix=='.sha256':validate_closure(x['path'])
 dependency=PLAN.get('validator_dependencies',{}).get('semantic_trace_analyzer')
 if dependency is None or not exact(dependency):fail('semantic validator dependency')
 for group in ('outputs','toolchain'):
  for x in BUILD[group].values():
   if not exact(x):fail(f'build {group}')
 for name in BUILD['transcripts'].values():
  if not (PACKAGE/name).is_file():fail('build transcript missing')
 cache=Path(BUILD['outputs']['cmake_cache']['path']).read_text();cc=Path(BUILD['outputs']['compile_commands']['path']).read_text()
 for k,v in BUILD['profile'].items():
  if f'{k}:' not in cache or f'={v}' not in cache:fail(f'cache profile {k}')
 for k,v in PLAN['selectors'].items():
  if f'-D{k}={v}' not in cc:fail(f'compiled selector {k}')
 return PLAN,BUILD
def validate_spec(x,dflash):
 keys={'enabled','draft_window','rounds','drafted_tokens','accepted_tokens','fallback_steps','acceptance_rate','acceptance_length','accepted_per_position'}
 if not isinstance(x,dict) or set(x)!=keys or x['enabled'] is not dflash:fail('spec schema')
 if not dflash:
  expected={'enabled':False,'draft_window':0,'rounds':0,'drafted_tokens':0,'accepted_tokens':0,'fallback_steps':0,'acceptance_rate':None,'acceptance_length':None,'accepted_per_position':[]}
  if x!=expected:fail('ordinary spec')
  return
 for k in ('draft_window','rounds','drafted_tokens','accepted_tokens','fallback_steps'):integer(x[k],k)
 if x['draft_window']!=4 or x['rounds']<=0 or x['drafted_tokens']<=0 or x['accepted_tokens']>x['drafted_tokens'] or x['drafted_tokens']>4*x['rounds']:fail('dflash counts')
 pos=x['accepted_per_position']
 if not isinstance(pos,list) or len(pos)!=4 or any(integer(v,'position')<0 for v in pos) or sum(pos)!=x['accepted_tokens']:fail('accepted positions')
 if not math.isclose(finite(x['acceptance_rate'],'rate'),x['accepted_tokens']/x['drafted_tokens'],abs_tol=1e-9,rel_tol=0) or not math.isclose(finite(x['acceptance_length'],'length'),(x['rounds']+x['accepted_tokens'])/x['rounds'],abs_tol=1e-9,rel_tol=0):fail('spec rates')
def validate_report(stem):
 p=RESULTS/f'{stem}.json';r=load(p);d=stem=='dflash-fresh';a=stem=='ordinary-append'
 if set(r)!={'schema_version','artifact_type','tool','command','environment','artifact','load','memory','config','tests'} or r['schema_version']!=20 or r['artifact_type']!='ninfer_bench_report' or r['command']!=' '.join(command(stem,p)):fail(f'report identity {stem}')
 e=r['environment']
 if e.get('gpu_name')!='AMD Radeon AI PRO R9700' or e.get('architecture_name')!='gfx1201' or e.get('device_id')!=0 or not e.get('hip_runtime_version') or not e.get('hip_driver_version'):fail('report hardware')
 if r['artifact']!={'path':str(ARTIFACT),'file_size_bytes':PLAN['artifact']['bytes']} or r['load'].get('weights_id')!=PLAN['artifact']['weights_id']:fail('report artifact')
 c=r['config'];expected={'max_context':166 if d else 156,'prefill_chunk':4096,'kv_cache_format':'fp8-k-int4-v','kv_value_group':16,'q4_activation_bits':8,'dflash_small_t_candidate':False,'dflash_mlp_down_t5_candidate':True,'dflash_rmsnorm_rows56_candidate':True,'attention_parity_candidate':True,'w8_activation_bits':8,'fp8_qk_wmma_enabled':True,'xattention_qualification':False,'concurrency':1,'spec':'dflash' if d else 'none','draft_tokens':4 if d else 0,'speculative_execution':d,'dflash_verify_width':5 if d else 0,'proposal_head':'optimized' if d else 'full','use_device_graph':False,'retain_token_ids':True,'isolate_prompt_decode':a,'decode_path':'dflash_eager' if d else 'eager','repetitions':1,'warmup':0,'corpus_path':str(CORPUS if a else HISTORY),'corpus_tokens':65536 if a else 129}
 for k,v in expected.items():
  if c.get(k)!=v:fail(f'config {stem}/{k}')
 tests=r['tests'];t=tests[0] if isinstance(tests,list) and len(tests)==1 else None
 if not isinstance(t,dict) or t.get('label')!=('pp128+tg27' if a else 'whole-pp129+tg27') or t.get('kind')!=('pp+tg' if a else 'whole') or t.get('n_prompt')!=(128 if a else 129) or t.get('n_gen')!=27 or t.get('requested_output_tokens')!=28:fail('test geometry')
 reps=t.get('reps');rep=reps[0] if isinstance(reps,list) and len(reps)==1 else None;lanes=rep.get('generated_token_ids_by_lane') if isinstance(rep,dict) else None
 if not isinstance(lanes,list) or len(lanes)!=1 or len(lanes[0])!=28:fail('retained tokens')
 outputs=[token(v,'output token') for v in lanes[0]]
 if rep.get('generated_output_tokens')!=28 or rep.get('decode_output_tokens')!=27 or rep.get('decode_engine_tokens')!=27:fail('output accounting')
 validate_spec(rep.get('speculative'),d)
 if t.get('speculative')!=rep.get('speculative'):fail('aggregate spec')
 if d and rep['speculative']['rounds']+rep['speculative']['accepted_tokens']+rep['speculative']['fallback_steps']!=27:fail('engine accounting')
 return identity(p),outputs,rep['speculative']
def validate_process(stem):
 p=load(RESULTS/f'{stem}.process.json')
 if p.get('command')!=command(stem) or p.get('trace_environment')!=trace_env(stem) or p.get('exit_code')!=0 or p.get('power_before')!='auto' or p.get('power_after')!='auto' or p.get('injection_environment')!=[]:fail(f'process {stem}')
 for k in ('stdout','stderr'):
  if p.get(k)!=identity(RESULTS/f'{stem}.{k}'):fail('process stream')
 if (RESULTS/f'{stem}.stdout').read_text()!=f'wrote {RESULTS/f"{stem}.json"}\n':fail('stdout')
 context=166 if stem=='dflash-fresh' else 156;label='pp128+tg27' if stem=='ordinary-append' else 'whole-pp129+tg27'
 expected=f'[ninfer_bench] loading {ARTIFACT} (max_context={context}, concurrency=1, kv_format=fp8-k-int4-v)\n[ninfer_bench] test 1/1 {label}: warmup=0 reps=1\n'
 if (RESULTS/f'{stem}.stderr').read_text()!=expected:fail('stderr')
 if integer(p.get('finished_unix_ns'),'finish')<=integer(p.get('started_unix_ns'),'start'):fail('process interval')
 return identity(RESULTS/f'{stem}.process.json')
def validate_dflash_events(events,outputs,spec):
 # Reuse the proven semantic-trace validator; this exact trace is independently closure-bound.
 old=module('semantic_validator',Path(PLAN['validator_dependencies']['semantic_trace_analyzer']['path']))
 return old.validate_decision_trace(RESULTS/'dflash-fresh.decision.json','dflash',outputs,spec)
def classify_sequences(ordinary_fresh,ordinary_append,dflash_fresh,tail_classification):
 for name,values in (('ordinary fresh',ordinary_fresh),('ordinary append',ordinary_append),('dflash fresh',dflash_fresh)):
  if not isinstance(values,list) or len(values)!=28:fail(f'{name} sequence length')
  for value in values:token(value,f'{name} token')
 ti=next((i for i,(x,y) in enumerate(zip(ordinary_fresh,ordinary_append,strict=True)) if x!=y),None)
 di=next((i for i,(x,y) in enumerate(zip(ordinary_fresh,dflash_fresh,strict=True)) if x!=y),None)
 text_exact=ti is None and tail_classification=='exact';target_exact=di is None
 status='exact_combined_attention_candidate_whole_semantic_parity' if text_exact and target_exact else 'text_and_target_parity_failed' if not text_exact and not target_exact else 'text_parity_failed' if not text_exact else 'target_parity_failed'
 return status,text_exact,target_exact,ti,di
def analyze():
 validate_plan();reports={};processes={};seq={};spec={}
 for s in ARMS:reports[s],seq[s],spec[s]=validate_report(s);processes[s]=validate_process(s)
 old=module('semantic_validator2',Path(PLAN['validator_dependencies']['semantic_trace_analyzer']['path']))
 ordinary_decision=old.validate_decision_trace(RESULTS/'ordinary-fresh.decision.json','ordinary',seq['ordinary-fresh'],spec['ordinary-fresh'])
 dflash_decision=validate_dflash_events(load(RESULTS/'dflash-fresh.decision.json').get('events'),seq['dflash-fresh'],spec['dflash-fresh'])
 tail=TAIL.compare(load(RESULTS/'ordinary-fresh.tail.json'),load(RESULTS/'ordinary-append.tail.json'))
 layer=LAYER.compare(RESULTS/'ordinary-fresh.layer.json',RESULTS/'dflash-fresh.layer.json','target')
 attention=ATTN.analyze(RESULTS/'ordinary-fresh.attention.json',RESULTS/'dflash-fresh.attention.json')
 status,text_exact,target_exact,ti,di=classify_sequences(seq['ordinary-fresh'],seq['ordinary-append'],seq['dflash-fresh'],tail.get('classification'))
 return {'artifact_type':'ninfer_r9700_attention_candidate_combined_whole_parity_evidence','schema_version':1,'status':status,'timing_evidence_eligible':False,'production_routing_authorized':False,'plan':identity(PACKAGE/'plan.json'),'build':identity(PACKAGE/'build-provenance.json'),'reports':reports,'processes':processes,'sequences':seq,'text':{'exact':text_exact,'first_public_token_difference':ti,'tail_comparison':tail},'target':{'exact':target_exact,'first_public_token_difference':di,'ordinary_decision_trace':ordinary_decision,'dflash_decision_trace':dflash_decision,'layer_comparison':layer,'attention_comparison':attention},'limitations':PLAN['limitations']}
def write_new(path,data):
 b=data.encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644)
 try:
  at=0
  while at<len(b):
   n=os.write(fd,b[at:]);
   if n<=0:fail('write made no progress')
   at+=n
  os.fsync(fd)
 finally:os.close(fd)
if __name__=='__main__':
 if len(sys.argv)!=2:fail('usage: analyze.py OUTPUT')
 write_new(Path(sys.argv[1]),json.dumps(analyze(),indent=2,sort_keys=True,allow_nan=False)+'\n')
