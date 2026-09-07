#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
import analyze
P=analyze.PACKAGE;R=analyze.RESULTS;POWER=Path(analyze.PLAN['hardware']['power_profile_source'])
EXACT={'LD_PRELOAD','LD_AUDIT','HIP_FORCE_QUEUE_PROFILING','AMD_SERIALIZE_KERNEL','AMD_SERIALIZE_COPY','ROC_SERIALIZE_KERNEL','GPU_DUMP_CODE_OBJECT','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'}
PREFIX=('ROCPROF','ROCP_','ROCTRACER_','ROCTX_','HSA_TOOLS_','HIP_TRACE_','AQLPROFILE_','ATT_PROFILE')
def fail(m):raise RuntimeError(m)
def injected():return sorted(k for k in os.environ if k in EXACT or k.startswith(PREFIX) or (k.startswith('NINFER_') and 'TRACE' in k))
def power():
 v=POWER.read_text().strip()
 if v!='auto':fail(f'power profile {v!r}, expected auto')
 return v
def write(path,data):
 if isinstance(data,str):data=data.encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644)
 try:
  at=0
  while at<len(data):
   n=os.write(fd,data[at:])
   if n<=0:fail('write made no progress')
   at+=n
  os.fsync(fd)
 finally:os.close(fd)
def preflight():
 if injected():fail(f'injection environment not clean: {injected()}')
 plan,build=analyze.validate_plan()
 if R.exists() or R.is_symlink():fail('results path already exists')
 source=Path(plan['source']['worktree'])
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=plan['source']['commit'] or subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=source,text=True).strip()!=plan['source']['tree'] or subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True):fail('source worktree differs')
 d=Path('/sys/class/drm/card2/device')
 if d.resolve().name!='0000:13:00.0' or (d/'vendor').read_text().strip()!='0x1002' or (d/'device').read_text().strip()!='0x7551':fail('device0 PCI identity differs')
 power();env={**os.environ,'LD_LIBRARY_PATH':'/opt/rocm/core-10.0/lib','PYTHONDONTWRITEBYTECODE':'1'}
 ldd=subprocess.check_output(['ldd',str(analyze.EXE)],env=env,text=True)
 if 'not found' in ldd or 'libamdhip64.so.7 => /opt/rocm/core-10.0/lib/libamdhip64.so.7' not in ldd or 'libhipblaslt.so.1 => /opt/rocm/core-10.0/lib/libhipblaslt.so.1' not in ldd:fail('runtime DSO resolution differs')
 for k in ('bench_support','runtime_mechanisms','decision_trace','trace_path'):
  subprocess.run([build['outputs'][k]['path']],cwd=source,env=env,check=True)
 subprocess.run([build['outputs']['runtime_planner']['path'],'--host-attention-parity-routing'],cwd=source,env=env,check=True)
 subprocess.run(['/usr/bin/python3',str(P/'test_analyze.py')],cwd=source,env=env,check=True)
def suffixes(stem):
 x=['json','stdout','stderr','process.json','decision.json']
 if stem!='dflash-fresh':x+=['tail.json']
 if stem!='ordinary-append':x+=['layer.json','layer.bin','attention.json','attention.bin']
 return x
def run_arm(stem):
 paths=[R/f'{stem}.{x}' for x in suffixes(stem)]
 if any(x.exists() or x.is_symlink() for x in paths):fail(f'arm output exists: {stem}')
 env={**os.environ,'LD_LIBRARY_PATH':'/opt/rocm/core-10.0/lib',**analyze.trace_env(stem)};cmd=analyze.command(stem);before=power();start=time.time_ns();done=subprocess.run(cmd,cwd=analyze.ROOT,env=env,capture_output=True,text=True);end=time.time_ns();after=power()
 write(R/f'{stem}.stdout',done.stdout);write(R/f'{stem}.stderr',done.stderr)
 write(R/f'{stem}.process.json',json.dumps({'command':cmd,'trace_environment':analyze.trace_env(stem),'exit_code':done.returncode,'started_unix_ns':start,'finished_unix_ns':end,'power_before':before,'power_after':after,'injection_environment':[],'stdout':analyze.identity(R/f'{stem}.stdout'),'stderr':analyze.identity(R/f'{stem}.stderr')},indent=2)+'\n')
 if done.returncode or any(not x.is_file() or x.is_symlink() for x in paths):fail(f'arm failed: {stem}')
def comparator(name,cmd):
 done=subprocess.run(cmd,cwd=analyze.ROOT,capture_output=True,text=True);write(R/f'{name}.stdout',done.stdout);write(R/f'{name}.stderr',done.stderr);write(R/f'{name}.process.json',json.dumps({'command':cmd,'exit_code':done.returncode},indent=2)+'\n')
 if done.returncode:fail(f'comparator failed: {name}')
def close():
 out=R/'result.sha256'
 if out.exists() or out.is_symlink():return
 write(out,''.join(f'{analyze.digest(p)}  {p.name}\n' for p in sorted(R.iterdir()) if p.is_file()))
def main():
 a=argparse.ArgumentParser();a.add_argument('--preflight-only',action='store_true');x=a.parse_args();preflight()
 if x.preflight_only:print('combined attention whole-parity GPU preflight PASS');return 0
 R.mkdir()
 try:
  for stem in analyze.ARMS:run_arm(stem)
  comparator('tail-comparator',['/usr/bin/python3',str(analyze.SOURCE/'tools/bench/compare_prefill_p129_tail_trace.py'),'--fresh',str(R/'ordinary-fresh.tail.json'),'--append',str(R/'ordinary-append.tail.json'),'--out',str(R/'tail-comparison.json')])
  comparator('layer-comparator',['/usr/bin/python3',str(analyze.SOURCE/'tools/bench/compare_qwen3_layer_boundary_trace.py'),'--left',str(R/'ordinary-fresh.layer.json'),'--right',str(R/'dflash-fresh.layer.json'),'--diagnostic','target','--out',str(R/'layer-comparison.json')])
  comparator('attention-comparator',['/usr/bin/python3',str(analyze.SOURCE/'tools/bench/analyze_qwen3_layer3_attention_trace.py'),'--ordinary',str(R/'ordinary-fresh.attention.json'),'--dflash',str(R/'dflash-fresh.attention.json'),'--output',str(R/'attention-comparison.json')])
  summary=analyze.analyze();analyze.write_new(R/'summary.json',json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n');close()
  if summary['status']!=analyze.PASS_STATUS:fail(f"functional gate did not pass: {summary['status']}")
 except BaseException:
  if R.is_dir():close()
  raise
 print('combined attention whole-parity capture complete');return 0
if __name__=='__main__':raise SystemExit(main())
