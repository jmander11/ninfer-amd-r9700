#!/usr/bin/env python3
"""Create-only four-arm runner for the e2 Text parity gate."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
import analyze

PACKAGE=Path(__file__).resolve().parent; RESULTS=PACKAGE/'results'
POWER=Path('/sys/class/drm/card2/device/power_dpm_force_performance_level')
EXACT={'LD_PRELOAD','LD_AUDIT','HIP_FORCE_QUEUE_PROFILING','AMD_SERIALIZE_KERNEL','AMD_SERIALIZE_COPY','ROC_SERIALIZE_KERNEL','GPU_DUMP_CODE_OBJECT','NINFER_DFLASH_DECISION_TRACE_OUT','NINFER_QWEN3_PREFILL_P129_TRACE','NINFER_QWEN3_PREFILL_P129_TRACE_OUT','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST','NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR','NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST','NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR','NINFER_QWEN3_GDN_STATE_TRACE_MANIFEST','NINFER_QWEN3_GDN_STATE_TRACE_SIDECAR','NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER'}
EXACT.update({'HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'})
PREFIX=('ROCPROF','ROCP_','ROCTRACER_','ROCTX_','HSA_TOOLS_','HIP_TRACE_','AQLPROFILE_','ATT_PROFILE','NINFER_QWEN3_')
def fail(m):raise RuntimeError(m)
def injected():return sorted(k for k in os.environ if k in EXACT or k.startswith(PREFIX))
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
 if injected():fail(f'injection environment is not clean: {injected()}')
 plan,build=analyze.validate_plan()
 if RESULTS.exists() or RESULTS.is_symlink():fail('results already exists')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['source']['worktree'],text=True).strip()!=plan['source']['commit'] or subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=plan['source']['worktree'],text=True).strip()!=plan['source']['tree'] or subprocess.check_output(['git','status','--porcelain'],cwd=plan['source']['worktree'],text=True):fail('source worktree differs')
 process=build['build_process'];mtime=analyze.EXE.stat().st_mtime_ns
 if not process['configure_started_unix_ns']<mtime<=process['build_finished_unix_ns']:fail('benchmark mtime is outside fresh build interval')
 env={**os.environ,'LD_LIBRARY_PATH':'/opt/rocm/core-10.0/lib','PYTHONDONTWRITEBYTECODE':'1'}
 ldd=subprocess.check_output(['ldd',str(analyze.EXE)],env=env,text=True)
 if 'not found' in ldd or 'libamdhip64.so.7 => /opt/rocm/core-10.0/lib/libamdhip64.so.7' not in ldd or 'libhipblaslt.so.1 => /opt/rocm/core-10.0/lib/libhipblaslt.so.1' not in ldd:fail('runtime DSO resolution differs')
 power()
 device=Path('/sys/class/drm/card2/device')
 if device.resolve().name!='0000:13:00.0' or (device/'vendor').read_text().strip()!='0x1002' or (device/'device').read_text().strip()!='0x7551':fail('R9700 PCI identity differs')
 for cmd in ([build['outputs']['runtime_test']['path']],[build['outputs']['trace_path_test']['path']],['/usr/bin/python3',str(PACKAGE/'test_analyze.py')]):subprocess.run(cmd,cwd=analyze.ROOT,env=env,check=True)
def run_one(stem):
 suffixes=('json','stdout','stderr','process.json','trace.json','trace.bin','gdn.json','gdn.bin','state.json','state.bin')
 paths=[RESULTS/f'{stem}.{s}' for s in suffixes]
 if any(p.exists() or p.is_symlink() for p in paths):fail(f'output exists: {stem}')
 env={**os.environ,'LD_LIBRARY_PATH':'/opt/rocm/core-10.0/lib',**analyze.trace_env(stem)}
 before=power(); started=time.time_ns(); done=subprocess.run(analyze.command(stem),cwd=analyze.ROOT,env=env,capture_output=True,text=True); finished=time.time_ns(); after=power()
 write(RESULTS/f'{stem}.stdout',done.stdout);write(RESULTS/f'{stem}.stderr',done.stderr);write(RESULTS/f'{stem}.json',done.stdout)
 write(RESULTS/f'{stem}.process.json',json.dumps({'command':analyze.command(stem),'trace_environment':analyze.trace_env(stem),'exit_code':done.returncode,'started_unix_ns':started,'finished_unix_ns':finished,'power_before':before,'power_after':after,'instrumentation_environment_present':[],'stdout':analyze.identity(RESULTS/f'{stem}.stdout'),'stderr':analyze.identity(RESULTS/f'{stem}.stderr')},indent=2)+'\n')
 if done.returncode or any(not p.is_file() for p in paths):close();fail(f'arm failed: {stem}')
def close():
 out=RESULTS/'result.sha256'
 if out.exists() or out.is_symlink():return
 files=sorted(p for p in RESULTS.iterdir() if p.is_file())
 write(out,''.join(f'{analyze.digest(p)}  {p}\n' for p in files))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();preflight()
 if a.preflight_only:print('e2 Text parity GPU preflight PASS');return 0
 RESULTS.mkdir(mode=0o755)
 try:
  for stem in analyze.ARMS:run_one(stem)
  summary=json.dumps(analyze.analyze(),indent=2,sort_keys=True,allow_nan=False)+'\n';write(RESULTS/'summary.json',summary);close()
 except BaseException:
  if RESULTS.exists():close()
  raise
 print('e2 Text parity gate PASS');return 0
if __name__=='__main__':raise SystemExit(main())
