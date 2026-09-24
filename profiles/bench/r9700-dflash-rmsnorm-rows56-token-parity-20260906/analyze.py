#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, sys
from pathlib import Path

ROOT=Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PKG=ROOT/"profiles/bench/r9700-dflash-rmsnorm-rows56-token-parity-20260906"
RESULTS=PKG/"results"
ART=ROOT/"out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
HIST=ROOT/"profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids"
CONTROL=ROOT/"build-r9700-dflash-rmsnorm-rows56-control-20260906/bench/ninfer_bench"
CANDIDATE=ROOT/"build-r9700-dflash-rmsnorm-rows56-candidate-20260906/bench/ninfer_bench"
ORDINARY=[96558,96917,96590,95762,98639,96917,3709,96342,104195,110896,96341,116256,1710,109357,96079,140293,104456,96560,96457,96590,3709,96723,109171,3709,98015,99937,115103,95946]
DFLASH=[96558,96917,96590,95762,98639,96917,3709,96342,104195,110896,96341,116256,1710,109357,96079,140293,104456,96560,96457,96590,3709,96723,109171,3709,98015,99937,115103,98003]

def fail(s): raise RuntimeError(s)
def pairs(items):
 d={}
 for k,v in items:
  if k in d: fail(f"duplicate JSON key: {k}")
  d[k]=v
 return d
def load(p):
 v=json.loads(p.read_text(),object_pairs_hook=pairs,parse_constant=lambda x:fail(f"nonfinite JSON: {x}"))
 if not isinstance(v,dict): fail(f"not object: {p}")
 return v
def ident(p):
 p=p.resolve(strict=True)
 with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
def command(role,report):
 exe=CANDIDATE if role=='candidate-dflash' else CONTROL
 c=[str(exe),'--weights',str(ART),'--corpus',str(HIST),'--device','0','--concurrency','1','--whole-pg','129,27','--prefill-chunk','4096','--kv-capacity','workload']
 if role.endswith('dflash'): c += ['--spec','dflash','--draft-tokens','4','--dflash-verify-width','5','--lm-head-draft']
 else: c += ['--draft-tokens','0']
 return c+['--retain-token-ids','--no-device-graph','--output','json','--output-file',str(report),'-r','1','--warmup','0']
def token_list(v,label):
 if not isinstance(v,list) or len(v)!=28 or any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<248077 for x in v): fail(f"token list differs: {label}")
 return v
def validate_spec(v,dflash,label):
 keys={'enabled','draft_window','rounds','drafted_tokens','accepted_tokens','fallback_steps','acceptance_rate','acceptance_length','accepted_per_position'}
 if not isinstance(v,dict) or set(v)!=keys or v.get('enabled') is not dflash: fail(f"spec keys differ: {label}")
 if not dflash:
  if v!={'enabled':False,'draft_window':0,'rounds':0,'drafted_tokens':0,'accepted_tokens':0,'fallback_steps':0,'acceptance_rate':None,'acceptance_length':None,'accepted_per_position':[]}: fail(f"ordinary spec differs: {label}")
  return
 for k in ('draft_window','rounds','drafted_tokens','accepted_tokens','fallback_steps'):
  if isinstance(v.get(k),bool) or not isinstance(v.get(k),int) or v[k]<0: fail(f"spec integer differs: {label}/{k}")
 if v['draft_window']!=4 or v['rounds']+v['accepted_tokens']+v['fallback_steps']!=27 or v['accepted_tokens']>v['drafted_tokens'] or v['drafted_tokens']>v['rounds']*4 or not isinstance(v['accepted_per_position'],list) or len(v['accepted_per_position'])!=4 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in v['accepted_per_position']) or sum(v['accepted_per_position'])!=v['accepted_tokens']: fail(f"spec accounting differs: {label}")
 for k in ('acceptance_rate','acceptance_length'):
  if isinstance(v.get(k),bool) or not isinstance(v.get(k),(int,float)) or not math.isfinite(v[k]): fail(f"spec rate differs: {label}/{k}")
 if v['drafted_tokens']==0 or v['rounds']==0: fail(f"spec denominator differs: {label}")
 if not math.isclose(v['acceptance_rate'],v['accepted_tokens']/v['drafted_tokens'],rel_tol=0.0,abs_tol=1e-9) or not math.isclose(v['acceptance_length'],(v['rounds']+v['accepted_tokens'])/v['rounds'],rel_tol=0.0,abs_tol=1e-9): fail(f"spec derived rate differs: {label}")
def validate_spec_pair(test,rep,dflash,label):
 validate_spec(test,dflash,label+'/test');validate_spec(rep,dflash,label+'/rep')
 if test!=rep: fail(f"test/rep speculative summaries differ: {label}")
def validate_report(role):
 p=RESULTS/f'{role}.json'; j=load(p); d=role.endswith('dflash'); selector=role=='candidate-dflash'
 if j.get('schema_version')!=20 or j.get('artifact_type')!='ninfer_bench_report' or j.get('tool')!='ninfer_bench' or j.get('command')!=' '.join(command(role,p)): fail(f"report identity differs: {role}")
 if j.get('environment')!={'gpu_name':'AMD Radeon AI PRO R9700','architecture_name':'gfx1201','hip_runtime_version':'7.15.26333','hip_driver_version':'7.15.26333','device_id':0}: fail(f"hardware differs: {role}")
 if j.get('artifact')!={'path':str(ART),'file_size_bytes':22763026944}: fail(f"artifact differs: {role}")
 c=j.get('config',{})
 expected={'max_context':166 if d else 156,'prefill_chunk':4096,'kv_cache_format':'fp8-k-int4-v','kv_value_group':16,'q4_activation_bits':8,'q4_prefill_cta_profile':'m64n128-pingpong-n16-k16-scalar-base-production','dflash_small_t_candidate':False,'dflash_mlp_down_t5_candidate':True,'dflash_rmsnorm_rows56_candidate':selector,'w8_activation_bits':8,'fp8_qk_wmma_enabled':True,'fp8_qk_wmma_profile':'t1-ge64-t2-ge320-t3plus-stream-v1','fp8_qk_wmma_t1_min_context':64,'fp8_qk_wmma_t2_min_context':320,'xattention_qualification':False,'concurrency':1,'pending_timeout_ms':4294967295,'pending_deadline':'unbounded','spec':'dflash' if d else 'none','draft_tokens':4 if d else 0,'speculative_execution':d,'dflash_verify_width_requested':5 if d else 0,'dflash_verify_width':5 if d else 0,'proposal_head':'optimized' if d else 'full','use_device_graph':False,'retain_token_ids':True,'isolate_prompt_decode':False,'decode_path':'dflash_eager' if d else 'eager','decode_graph_prime':{'primed':False,'output_tokens':0},'repetitions':1,'warmup':0,'corpus_path':str(HIST),'corpus_tokens':129}
 for k,v in expected.items():
  if c.get(k)!=v: fail(f"config differs: {role}/{k}")
 if c.get('kv_plane_layouts')!={'key':'token-fastest-head-major','value':'feature-fastest-page-major','value_scale':'feature-fastest-page-major'}: fail(f"KV layout differs: {role}")
 ts=j.get('tests'); t=ts[0] if isinstance(ts,list) and len(ts)==1 else None
 if not isinstance(t,dict) or t.get('label')!='whole-pp129+tg27' or t.get('kind')!='whole' or t.get('n_prompt')!=129 or t.get('n_gen')!=27 or t.get('requested_output_tokens')!=28: fail(f"test geometry differs: {role}")
 rs=t.get('reps'); r=rs[0] if isinstance(rs,list) and len(rs)==1 else None
 if not isinstance(r,dict) or r.get('generated_output_tokens')!=28 or r.get('decode_output_tokens')!=27 or r.get('decode_engine_tokens')!=27: fail(f"rep counts differ: {role}")
 lanes=r.get('generated_token_ids_by_lane'); tokens=token_list(lanes[0],role) if isinstance(lanes,list) and len(lanes)==1 else fail(f"lanes differ: {role}")
 validate_spec_pair(t.get('speculative'),r.get('speculative'),d,role)
 return ident(p),tokens
def validate_process(role):
 p=RESULTS/f'{role}.process.json'; v=load(p); report=RESULTS/f'{role}.json'
 if set(v)!={'command','exit_code','started_unix_ns','finished_unix_ns','power_before','power_after','injection_environment_present','executable','stdout','stderr'} or v.get('command')!=command(role,report) or v.get('exit_code')!=0 or v.get('power_before')!='auto' or v.get('power_after')!='auto' or v.get('injection_environment_present')!=[] or v.get('executable')!=ident(CANDIDATE if role=='candidate-dflash' else CONTROL): fail(f"process differs: {role}")
 for s in ('stdout','stderr'):
  if v.get(s)!=ident(RESULTS/f'{role}.{s}'): fail(f"stream differs: {role}/{s}")
 if (RESULTS/f'{role}.stdout').read_text()!=f'wrote {report}\n': fail(f"stdout differs: {role}")
 maxc=166 if role.endswith('dflash') else 156
 err=f'[ninfer_bench] loading {ART} (max_context={maxc}, concurrency=1, kv_format=fp8-k-int4-v)\n[ninfer_bench] test 1/1 whole-pp129+tg27: warmup=0 reps=1\n'
 if (RESULTS/f'{role}.stderr').read_text()!=err: fail(f"stderr differs: {role}")
 if any(isinstance(v.get(k),bool) or not isinstance(v.get(k),int) for k in ('started_unix_ns','finished_unix_ns')) or v['finished_unix_ns']<=v['started_unix_ns']: fail(f"process interval differs: {role}")
 return ident(p)
def validate_authorities():
 plan=load(PKG/'plan.json')
 if plan.get('artifact_type')!='ninfer_r9700_dflash_rmsnorm_rows56_token_parity_plan' or plan.get('schema_version')!=1 or plan.get('status')!='prepared_cpu_only_no_gpu_execution' or plan.get('source')!={'commit':'c8e1bc418b80c23364460801b7c0c017bdab87b8','tree':'5cda68536859bda53f091c1fc74593b246061598'} or plan.get('production_routing_authorized') is not False or plan.get('arms')!=['control-ordinary','control-dflash','candidate-dflash'] or plan.get('decision')!='credit restoration only when fresh ordinary and selector-off exactly reproduce the two retained 28-token authorities and selector-on DFlash exactly equals ordinary': fail('plan authority differs')
 if plan.get('workload')!={'device':0,'concurrency':1,'maximum_product_concurrency':4,'prompt_tokens':129,'decode_tokens':27,'retained_tokens':28,'prefill_chunk':4096,'device_graph':False,'warmup':0,'repetitions':1,'dflash':'K4/W5','power_profile':'auto'}: fail('workload authority differs')
 if ident(Path(plan['matched_build_receipt']['path']))!=plan['matched_build_receipt'] or ident(Path(plan['artifact']['path']))!=plan['artifact'] or ident(Path(plan['history']['path']))!=plan['history']: fail('input authority differs')
 numerical=plan.get('matched_gpu_numerical_authority',{})
 if set(numerical)!={'result','closure'} or any(ident(Path(v['path']))!=v for v in numerical.values()): fail('matched GPU numerical authority differs')
 result=load(Path(numerical['result']['path']))
 if result.get('status')!='passed' or result.get('source_commit')!='c8e1bc418b80c23364460801b7c0c017bdab87b8' or result.get('candidate',{}).get('selector')!=1 or result.get('candidate',{}).get('result')!='pass' or result.get('candidate',{}).get('maximum_bf16_steps')!=1 or result.get('candidate',{}).get('graph_rows')!=[5,6]: fail('candidate production numerical result differs')
 for value in plan['prior_authority'].values():
  if ident(Path(value['path']))!=value: fail('prior evidence authority differs')
 prior_o=load(Path(plan['prior_authority']['ordinary_report']['path']))['tests'][0]['reps'][0]['generated_token_ids_by_lane'][0]
 prior_d=load(Path(plan['prior_authority']['dflash_report']['path']))['tests'][0]['reps'][0]['generated_token_ids_by_lane'][0]
 if prior_o!=ORDINARY or prior_d!=DFLASH: fail('prior token authority differs')
 return {'matched_build_receipt':plan['matched_build_receipt'],'matched_gpu_numerical_authority':numerical,'prior_capture_closure':plan['prior_authority']['capture_closure'],'prior_analysis_closure':plan['prior_authority']['analysis_closure'],'layer_analysis_closure':plan['prior_authority']['layer_analysis_closure']}
def classify(tokens):
 if tokens['control-ordinary']!=ORDINARY: fail('fresh ordinary did not reproduce retained authority')
 if tokens['control-dflash']!=DFLASH: fail('selector-off DFlash did not reproduce retained authority')
 restored=tokens['candidate-dflash']==tokens['control-ordinary']
 first=next((i for i,(a,b) in enumerate(zip(tokens['control-ordinary'],tokens['candidate-dflash'])) if a!=b),None)
 return restored,first
def analyze():
 authority=validate_authorities()
 reports={};processes={};tokens={}
 for role in ('control-ordinary','control-dflash','candidate-dflash'):
  processes[role]=validate_process(role); reports[role],tokens[role]=validate_report(role)
 restored,first=classify(tokens)
 return {'artifact_type':'ninfer_r9700_dflash_rmsnorm_rows56_token_parity_evidence','schema_version':1,'status':'restored_exact_token_parity_through_index27' if restored else 'not_restored','timing_evidence_eligible':False,'production_routing_authorized':False,'source_commit':'c8e1bc418b80c23364460801b7c0c017bdab87b8','authority':authority,'baseline_reproduced':True,'selector_on_matches_ordinary_all_28_tokens':restored,'candidate_first_difference_index':first,'ordinary_index27':95946,'selector_off_dflash_index27':98003,'candidate_index27':tokens['candidate-dflash'][27],'reports':reports,'processes':processes,'limitations':['functional token parity only; timings are ineligible','does not prove hidden-state exactness or performance','does not authorize production routing']}
def main():
 if len(sys.argv)!=3 or sys.argv[1]!='--summary' or Path(sys.argv[2])!=RESULTS/'summary.json' or Path(sys.argv[2]).exists() or Path(sys.argv[2]).is_symlink(): fail('summary path differs or is not fresh')
 out=json.dumps(analyze(),indent=2,allow_nan=False)+'\n'; fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644); os.write(fd,out.encode()); os.fsync(fd); os.close(fd); return 0
if __name__=='__main__': raise SystemExit(main())
