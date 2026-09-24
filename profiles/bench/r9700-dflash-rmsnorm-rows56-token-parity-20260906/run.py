#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
PKG=Path(__file__).resolve().parent; ROOT=Path('/ssdpool2nvme/local_llm/ninfer-amd-r9700'); RESULTS=PKG/'results'
POWER=Path('/sys/class/drm/card2/device/power_dpm_force_performance_level')
sys.path.insert(0,str(PKG)); import analyze
EXACT={'LD_PRELOAD','LD_AUDIT','HIP_FORCE_QUEUE_PROFILING','AMD_SERIALIZE_KERNEL','AMD_SERIALIZE_COPY','ROC_SERIALIZE_KERNEL','GPU_DUMP_CODE_OBJECT','NINFER_DFLASH_DECISION_TRACE_OUT','NINFER_QWEN3_PREFILL_P129_TRACE','NINFER_QWEN3_PREFILL_P129_TRACE_OUT','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR'}
PREFIX=('ROCPROF','ROCP_','ROCTRACER_','ROCTX_','HSA_TOOLS_','HIP_TRACE_','AQLPROFILE_','ATT_PROFILE')
def fail(s): raise RuntimeError(s)
def injected(): return sorted(k for k in os.environ if k in EXACT or k.startswith(PREFIX))
def power():
 v=POWER.read_text().strip()
 if v!='auto': fail(f'power profile is {v!r}, expected auto')
 return v
def ident(p):
 p=p.resolve(strict=True)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
def exclusive(p,data):
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644)
 try:
  b=data.encode() if isinstance(data,str) else data; n=0
  while n<len(b): n+=os.write(fd,b[n:])
  os.fsync(fd)
 finally: os.close(fd)
def validate():
 if injected(): fail('profiling/trace injection environment must be clean')
 plan=json.loads((PKG/'plan.json').read_text()); receipt=json.loads(Path(plan['matched_build_receipt']['path']).read_text())
 if ident(Path(plan['matched_build_receipt']['path']))!=plan['matched_build_receipt'] or receipt.get('source')!={'commit':'c8e1bc418b80c23364460801b7c0c017bdab87b8','detached_checkout':'/ssdpool2nvme/local_llm/ninfer-amd-r9700-dflash-rmsnorm-rows56-build-source','tree':'5cda68536859bda53f091c1fc74593b246061598'} or receipt.get('configs_differ_only_in_dflash_rmsnorm_rows56_candidate') is not True: fail('matched receipt differs')
 for role,path,sha,size,selector in [('control',analyze.CONTROL,'5fbdbc70bd70dcf5af5f69b5cf91f5cbe9134dc05fdf57dfaa622d22fc1e0b43',5166640,0),('candidate',analyze.CANDIDATE,'a1c2a1a90550373aab2663594b1854ee21e7f2ca32612b0fbd4b1eeb4780b7eb',5166672,1)]:
  if ident(path)!={'path':str(path),'bytes':size,'sha256':sha} or receipt['builds'][role]['benchmark_executable']!=ident(path) or receipt['builds'][role]['NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE']!=selector: fail(f'{role} executable differs')
 for key in ('artifact','history'):
  if ident(Path(plan[key]['path']))!=plan[key]: fail(f'{key} differs')
 for v in plan['prior_authority'].values():
  if ident(Path(v['path']))!=v: fail('prior authority differs')
 for v in plan['matched_gpu_numerical_authority'].values():
  if ident(Path(v['path']))!=v: fail('matched GPU numerical authority differs')
 subprocess.run(['/usr/bin/python3',str(ROOT/'profiles/bench/r9700-dflash-rmsnorm-rows56-matched-builds-20260906/validate.py')],check=True,cwd=ROOT)
 for exe in (analyze.CONTROL,analyze.CANDIDATE):
  text=subprocess.check_output(['ldd',str(exe)],text=True,env={**os.environ,'LD_LIBRARY_PATH':'/opt/rocm/lib'})
  if 'not found' in text or 'libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1' not in text or 'libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7' not in text: fail('runtime DSO binding differs')
 if Path('/sys/class/drm/card2/device/vendor').read_text().strip()!='0x1002' or Path('/sys/class/drm/card2/device/device').read_text().strip()!='0x7551' or Path('/sys/class/drm/card2/device').resolve().name!='0000:13:00.0': fail('R9700 PCI identity differs')
 power()
 subprocess.run(['/usr/bin/python3',str(PKG/'test_analyze.py')],check=True,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
def run_one(role):
 report=RESULTS/f'{role}.json'; paths=[report,RESULTS/f'{role}.stdout',RESULTS/f'{role}.stderr',RESULTS/f'{role}.process.json']
 if any(p.exists() or p.is_symlink() for p in paths): fail(f'{role} output exists')
 cmd=analyze.command(role,report); env=os.environ.copy(); env['LD_LIBRARY_PATH']='/opt/rocm/lib'; before=power(); start=time.time_ns(); x=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True); end=time.time_ns(); after=power()
 exclusive(paths[1],x.stdout); exclusive(paths[2],x.stderr)
 rec={'command':cmd,'exit_code':x.returncode,'started_unix_ns':start,'finished_unix_ns':end,'power_before':before,'power_after':after,'injection_environment_present':[],'executable':ident(analyze.CANDIDATE if role=='candidate-dflash' else analyze.CONTROL),'stdout':ident(paths[1]),'stderr':ident(paths[2])}
 exclusive(paths[3],json.dumps(rec,indent=2)+'\n')
 if x.returncode or not report.is_file(): close(); fail(f'{role} failed; partial evidence retained')
def close():
 out=RESULTS/'result.sha256'
 if out.exists() or out.is_symlink(): return
 files=sorted(p for p in RESULTS.iterdir() if p.is_file() and p.name!='result.sha256')
 exclusive(out,''.join(f"{ident(p)['sha256']}  {p}\n" for p in files))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();validate()
 if a.preflight_only: print('rows56 token-parity preflight PASS');return 0
 if RESULTS.exists() or RESULTS.is_symlink(): fail('results path must be fresh')
 RESULTS.mkdir()
 try:
  for r in ('control-ordinary','control-dflash','candidate-dflash'):run_one(r)
  subprocess.run(['/usr/bin/python3',str(PKG/'analyze.py'),'--summary',str(RESULTS/'summary.json')],check=True,cwd=ROOT)
  close();print('rows56 token-parity functional experiment PASS');return 0
 except Exception:
  close();raise
if __name__=='__main__':raise SystemExit(main())
