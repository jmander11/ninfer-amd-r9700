#!/usr/bin/env python3
"""Fail-closed analysis for the exact e2-at-T129 Text parity gate."""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ROOT=Path('/ssdpool2nvme/local_llm/ninfer-amd-r9700')
RESULTS=PACKAGE/'results'
SOURCE=Path('/ssdpool2nvme/local_llm/ninfer-amd-r9700-fp8-e2-text-parity-src-25a27bb2')
EXE=ROOT/'build-r9700-fp8-e2-text-parity-25a27bb2-fresh-20260906/bench/ninfer_bench'
ARTIFACT=ROOT/'out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer'
CORPUS=ROOT/'bench/fixtures/bench_corpus.ids'
HISTORY=ROOT/'profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids'
sys.path.insert(0,str(SOURCE/'tools/bench'))
from compare_qwen3_layer_boundary_trace import compare,compare_gdn,compare_recurrent_state,load,load_gdn,load_recurrent_state

ARMS=('layer0-fresh','layer0-append','layer1-fresh','layer1-append')
ROLES={'layer0-fresh':('text-fresh-frontier129-column128',129,128,129,128),'layer0-append':('text-append-frontier129-column0',1,0,129,128),'layer1-fresh':('text-fresh-frontier129-column128',129,128,129,128),'layer1-append':('text-append-frontier129-column0',1,0,129,128)}
TOKENS=[[96558,96917]]

def fail(message): raise RuntimeError(message)
def pairs(items):
 out={}
 for k,v in items:
  if k in out: fail(f'duplicate key {k}')
  out[k]=v
 return out
def load_json(path):
 value=json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=lambda x:fail(f'nonfinite {x}'))
 if not isinstance(value,dict): fail(f'not object: {path}')
 return value
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity(path):
 p=path.resolve(strict=True);return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}
def command(stem):
 fresh=stem.endswith('fresh'); corpus=HISTORY if fresh else CORPUS
 cmd=[str(EXE),'--weights',str(ARTIFACT),'--corpus',str(corpus),'--device','0','--concurrency','1']
 cmd += ['--whole-pg','129,1'] if fresh else ['-pg','128,1']
 cmd += ['--prefill-chunk','4096','--kv-capacity','workload','--draft-tokens','0','--retain-token-ids']
 if not fresh:cmd += ['--isolate-prompt-decode']
 return cmd+['--no-device-graph','--output','json','-r','1','--warmup','0']
def trace_env(stem):
 base=RESULTS/stem; role=ROLES[stem][0]; layer=stem[5]
 return {'NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE':role,'NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST':str(base)+'.trace.json','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR':str(base)+'.trace.bin','NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST':str(base)+'.gdn.json','NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR':str(base)+'.gdn.bin','NINFER_QWEN3_GDN_STATE_TRACE_MANIFEST':str(base)+'.state.json','NINFER_QWEN3_GDN_STATE_TRACE_SIDECAR':str(base)+'.state.bin','NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER':layer}
def exact_identity(expected):
 return identity(Path(expected['path']))==expected
def validate_plan():
 p=load_json(PACKAGE/'plan.json'); b=load_json(PACKAGE/'build-provenance.json')
 if p.get('status')!='completed_partial_success_whole_text_parity_blocked' or p.get('claim')!='functional-only localization of the exact selector-one combined candidate build: public tokens and layer0/layer1 GDN state are exact while the first residual difference is layer3 post-mixer' or p.get('timing_evidence_eligible') is not False or p.get('production_routing_authorized') is not False:fail('plan authority differs')
 if p.get('source')!=b.get('source') or p.get('build_receipt')!='build-provenance.json':fail('source/build binding differs')
 if p.get('candidate')!={'semantic_identity':'e2-at-t129','algorithm_fingerprint':'e2e00100000000000000000000000000','matrix_shape':[34816,5120],'selected_tokens':[129],'unchanged_default_tokens':[128],'selector_cmake_key':'NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE','selector_value':'1','scope_contract':'T128 exact-validates and retains default e2; only N34816/K5120/T129 selects the same exact viable e2; every other shape and width retains default selection'}:fail('candidate differs')
 if p.get('workload')!={'concurrency':1,'maximum_product_concurrency':4,'prefill_chunk':4096,'warmup':0,'repetitions':1,'device_graph':False,'spec':'none','draft_tokens':0,'retain_token_ids':True}:fail('workload differs')
 if p.get('hardware')!={'device':0,'name':'AMD Radeon AI PRO R9700','architecture':'gfx1201','wave_size':32,'pci':'0000:13:00.0','power_profile':'auto'}:fail('hardware plan differs')
 if p.get('arms')!={k:{'history':'fresh-T129' if k.endswith('fresh') else 'append-P128-plus-T1','trace_role':ROLES[k][0],'state_layer':int(k[5]),'selected_column':ROLES[k][2]} for k in ARMS}:fail('arms differ')
 if p.get('acceptance')!={'all_report_tokens':TOKENS,'layer0_recurrent_state':'exact','layer1_recurrent_state':'exact','selected_column_layer_boundaries':'first_visible_post_mixer_layer_3','selected_column_layer1_gdn_fields':'exact','repeat_sidecars':'exact'}:fail('acceptance differs')
 if p.get('limitations')!=['four candidate captures are required because the committed trace emits one selected recurrent-state layer per process','prior operator and owner evidence motivates e2, but this gate proves only exact parity of the combined selector-one build and does not isolate e2 causality','functional eager C1 diagnostic only; trace D2H invalidates all timing','the result is consistent with e2 removing the prior layer0/layer1 divergence, but the combined-build experiment does not isolate e2 causality','whole Text parity remains blocked at the first full-attention layer3 post-mixer','no production routing or performance claim']:fail('limitations differ')
 history=p.get('history',{})
 if history.get('seed_token')!=24178 or history.get('rule')!='source corpus tokens [0,128) followed by retained fresh-P128 ordinary greedy seed':fail('history semantics differ')
 for key in ('source_corpus','p129_fixture'):
  if not exact_identity(history.get(key,{})):fail(f'history identity differs: {key}')
 if [int(x) for x in HISTORY.read_text().split()] != [int(x) for x in CORPUS.read_text().split()][:128]+[24178]:fail('history derivation differs')
 if not exact_identity({k:p['artifact'][k] for k in ('path','bytes','sha256')}):fail('artifact differs')
 for x in p['prior_control_authorities'].values():
  if not exact_identity(x):fail('prior authority differs')
 for group in ('outputs','toolchain_runtime'):
  for x in b[group].values():
   if not exact_identity(x):fail(f'build identity differs: {group}')
 for x in b.get('build_process',{}).values():
  if isinstance(x,dict) and identity(Path(x['path']))!={k:x[k] for k in ('path','bytes','sha256')}:fail('build process identity differs')
 compressed=b['build_process']['configure_stdout']; raw=gzip.decompress(Path(compressed['path']).read_bytes())
 if compressed.get('encoding')!='gzip-n' or len(raw)!=compressed.get('uncompressed_bytes') or hashlib.sha256(raw).hexdigest()!=compressed.get('uncompressed_sha256'):fail('configure transcript differs')
 if b.get('normalized_profile_sha256')!='995b25a06a910e972cadf711c8d376a123c089f2626c8c797c187e7164fef34f':fail('normalized profile differs')
 process=b.get('build_process',{})
 if process.get('configure_exit_code')!=0 or process.get('build_exit_code')!=0 or not all(type(process.get(k)) is int for k in ('configure_started_unix_ns','configure_finished_unix_ns','build_finished_unix_ns')) or not process['configure_started_unix_ns']<process['configure_finished_unix_ns']<process['build_finished_unix_ns']:fail('build process differs')
 build_dir='/ssdpool2nvme/local_llm/ninfer-amd-r9700/build-r9700-fp8-e2-text-parity-25a27bb2-fresh-20260906'
 expected_config=['cmake','-S',str(SOURCE),'-B',build_dir,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++','-DNINFER_BUILD_BENCHMARKS=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']+[f'-D{k}={v}' for k,v in b['profile'].items()]
 expected_build=['cmake','--build',build_dir,'--target','ninfer_bench','ninfer_qwen3_runtime_mechanisms_test','ninfer_qwen3_prefill_tail_trace_path_test','-j','16']
 if b.get('build_directory')!=build_dir or b.get('configure_command')!=expected_config or b.get('build_command')!=expected_build:fail('build commands differ')
 if b.get('benchmark_elf',{}).get('gnu_build_id') is not None or b.get('benchmark_elf',{}).get('fresh_benchmark_is_distinct_artifact') is not True:fail('ELF identity differs')
 if b.get('normalized_selector_delta')!={'control_value':'0','candidate_value':'1','only_difference':'NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE','control_not_reexecuted':'prior selector-zero authorities motivate attribution but are not a matched causal control for this combined build'}:fail('normalized selector delta differs')
 for relative,expected in b.get('source_files',{}).items():
  blob=(SOURCE/relative).read_bytes()
  if len(blob)!=expected.get('bytes') or hashlib.sha256(blob).hexdigest()!=expected.get('sha256'):fail(f'source identity differs: {relative}')
  if subprocess.check_output(['git','rev-parse',f"{p['source']['commit']}:{relative}"],cwd=ROOT,text=True).strip()!=b.get('source_blob_ids',{}).get(relative):fail(f'source blob differs: {relative}')
 cache=Path(b['outputs']['cmake_cache']['path']).read_text(); cc=Path(b['outputs']['compile_commands']['path']).read_text()
 for k,v in b['profile'].items():
  if f'{k}:' not in cache or f'={v}' not in cache:fail(f'cache profile differs: {k}')
 for k in ('NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE','NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE','NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE','NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE'):
  if f'-D{k}=1' not in cc:fail(f'compiled selector differs: {k}')
 return p,b
def validate_spec(x):
 expected={'enabled':False,'draft_window':0,'rounds':0,'drafted_tokens':0,'accepted_tokens':0,'fallback_steps':0,'acceptance_rate':None,'acceptance_length':None,'accepted_per_position':[]}
 if not isinstance(x,dict) or set(x)!=set(expected) or any(type(x[k]) is not type(v) or x[k]!=v for k,v in expected.items()):fail('spec differs')
def validate_report(stem):
 r=load_json(RESULTS/f'{stem}.json'); fresh=stem.endswith('fresh')
 if r.get('schema_version')!=20 or r.get('artifact_type')!='ninfer_bench_report' or r.get('command')!=' '.join(command(stem)):fail('report identity differs')
 if r.get('environment')!={'gpu_name':'AMD Radeon AI PRO R9700','architecture_name':'gfx1201','hip_runtime_version':'7.15.26333','hip_driver_version':'7.15.26333','device_id':0}:fail('hardware differs')
 if r.get('artifact')!={'path':str(ARTIFACT),'file_size_bytes':22763026944}:fail('report artifact differs')
 c=r.get('config',{})
 expected={'max_context':130,'prefill_chunk':4096,'kv_cache_format':'fp8-k-int4-v','kv_value_group':16,'q4_activation_bits':8,'dflash_small_t_candidate':False,'dflash_mlp_down_t5_candidate':True,'dflash_rmsnorm_rows56_candidate':True,'w8_activation_bits':8,'fp8_qk_wmma_enabled':True,'xattention_qualification':False,'concurrency':1,'spec':'none','draft_tokens':0,'speculative_execution':False,'dflash_verify_width':0,'proposal_head':'full','use_device_graph':False,'retain_token_ids':True,'isolate_prompt_decode':not fresh,'decode_path':'eager','repetitions':1,'warmup':0,'corpus_path':str(HISTORY if fresh else CORPUS),'corpus_tokens':129 if fresh else 65536}
 for k,v in expected.items():
  if c.get(k)!=v:fail(f'config differs {stem}/{k}')
 tests=r.get('tests'); t=tests[0] if isinstance(tests,list) and len(tests)==1 else None
 if not isinstance(t,dict) or t.get('label')!=('whole-pp129+tg1' if fresh else 'pp128+tg1') or t.get('kind')!=('whole' if fresh else 'pp+tg') or t.get('n_prompt')!=(129 if fresh else 128) or t.get('n_gen')!=1 or t.get('requested_output_tokens')!=2:fail(f'test contract differs: {stem}')
 reps=t.get('reps') if isinstance(t,dict) else None; rep=reps[0] if isinstance(reps,list) and len(reps)==1 else None
 if not isinstance(rep,dict) or rep.get('generated_token_ids_by_lane')!=TOKENS or rep.get('generated_output_tokens')!=2 or rep.get('decode_output_tokens')!=1 or rep.get('decode_engine_tokens')!=1:fail(f'tokens differ: {stem}')
 validate_spec(t.get('speculative'));validate_spec(rep.get('speculative'))
 for v in rep.get('timings',{}).values():
  if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:fail('timing invalid')
 return identity(RESULTS/f'{stem}.json')
def validate_process(stem):
 p=load_json(RESULTS/f'{stem}.process.json')
 if p.get('command')!=command(stem) or p.get('trace_environment')!=trace_env(stem) or p.get('exit_code')!=0 or p.get('power_before')!='auto' or p.get('power_after')!='auto' or p.get('instrumentation_environment_present')!=[]:fail(f'process differs: {stem}')
 if p.get('stdout')!=identity(RESULTS/f'{stem}.stdout') or p.get('stderr')!=identity(RESULTS/f'{stem}.stderr') or (RESULTS/f'{stem}.stdout').read_bytes()!=(RESULTS/f'{stem}.json').read_bytes():fail('process IO differs')
 label='whole-pp129+tg1' if stem.endswith('fresh') else 'pp128+tg1'
 expected=f'[ninfer_bench] loading {ARTIFACT} (max_context=130, concurrency=1, kv_format=fp8-k-int4-v)\n[ninfer_bench] test 1/1 {label}: warmup=0 reps=1\n'
 if (RESULTS/f'{stem}.stderr').read_text()!=expected:fail(f'stderr differs: {stem}')
 if not isinstance(p.get('started_unix_ns'),int) or isinstance(p['started_unix_ns'],bool) or not isinstance(p.get('finished_unix_ns'),int) or p['finished_unix_ns']<=p['started_unix_ns']:fail('process interval differs')
 return identity(RESULTS/f'{stem}.process.json')
def analyze():
 plan,build=validate_plan(); reports={};processes={}
 for stem in ARMS:
  reports[stem]=validate_report(stem);processes[stem]=validate_process(stem)
  m,_=load_recurrent_state(RESULTS/f'{stem}.state.json',ROLES[stem][0])
  lm,_=load(RESULTS/f'{stem}.trace.json',ROLES[stem]);gm,_=load_gdn(RESULTS/f'{stem}.gdn.json',ROLES[stem])
  if lm.get('token')!=24178 or gm.get('token')!=24178:fail('trace token differs')
  if m['text_layer']!=int(stem[5]):fail('state layer differs')
 for suffix in ('trace.bin','gdn.bin'):
  if (RESULTS/f'layer0-fresh.{suffix}').read_bytes()!=(RESULTS/f'layer1-fresh.{suffix}').read_bytes() or (RESULTS/f'layer0-append.{suffix}').read_bytes()!=(RESULTS/f'layer1-append.{suffix}').read_bytes():fail('repeat sidecar differs')
 layer=compare(RESULTS/'layer0-fresh.trace.json',RESULTS/'layer0-append.trace.json','text')
 gdn=compare_gdn(RESULTS/'layer0-fresh.gdn.json',RESULTS/'layer0-append.gdn.json','text')
 states={str(n):compare_recurrent_state(RESULTS/f'layer{n}-fresh.state.json',RESULTS/f'layer{n}-append.state.json') for n in (0,1)}
 expected_layer={'snapshot_index':7,'layer':3,'half':'post_mixer','first_hidden_index':0,'mismatch_count':3467,'left_bf16_bits':48413,'right_bf16_bits':48414,'preceding_boundary_exact':True}
 if layer.get('classification')!='first_visible_post_mixer_layer_3' or layer.get('first_difference')!=expected_layer or gdn['classification']!='layer1_gdn_boundaries_exact' or any(states[str(n)]['classification']!=f'layer{n}_recurrent_prefix_state_exact' for n in (0,1)):fail('captured diagnostic classification differs')
 return {'artifact_type':'ninfer_r9700_fp8_e2_t129_text_parity_evidence','schema_version':1,'status':'partial_success_layer01_exact_whole_text_blocked_at_layer3_full_attention','timing_evidence_eligible':False,'production_routing_authorized':False,'candidate':plan['candidate'],'build':identity(PACKAGE/'build-provenance.json'),'capture_result_closure':identity(RESULTS/'result.sha256'),'reports':reports,'processes':processes,'layer_comparison':layer,'gdn_comparison':gdn,'state_comparisons':states,'generated_token_ids_by_lane':TOKENS,'conclusion':'public tokens, layer0/layer1 recurrent states, and layer1 GDN selected-column boundaries are exact; first residual difference is layer3 post-mixer','limitations':plan['limitations']}
if __name__=='__main__':
 data=json.dumps(analyze(),indent=2,sort_keys=True,allow_nan=False)+'\n'
 if len(sys.argv)!=2:fail('usage: analyze.py OUTPUT')
 fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644)
 try:
  encoded=data.encode();at=0
  while at<len(encoded):at+=os.write(fd,encoded[at:])
  os.fsync(fd)
 finally:os.close(fd)
