#!/usr/bin/env python3
"""Run and immutably publish the exact-P2048 scalar-base Q4 gate."""
from __future__ import annotations
import argparse,hashlib,json,math,os,re,stat,statistics,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PY=Path("/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python")
REPORT=ROOT/"profiles/bench/r9700-a8q4-n16k16-scalar-base-p2048-ab-20260905.json"
POWER=Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
BINARY=ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_harness"
SOURCE=ROOT/"tools/r9700/a8q4_n16k16_scalar_base_qual.hip"
HEADER=ROOT/"tools/r9700/a8q4_n16k16_scalar_base_qual.h"
ASSEMBLY=ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_qual.s"
CANDIDATE=ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_candidate.hsaco"
BASELINE=ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_baseline.hsaco"
NOTES=ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_candidate.notes"
BASE_DISASM=ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_baseline.objdump"
CAND_DISASM=ROOT/"tools/r9700/build/a8q4_n16k16_scalar_base_candidate.objdump"
CHECKER=ROOT/"tools/r9700/check_a8q4_n16k16_scalar_base_static.py"
PROD_OBJECT=ROOT/"build-r9700/src/CMakeFiles/ninfer_r9700_core.dir/ops/r9700/linear/r9700_linear.hip.o"
PROD_SOURCE=ROOT/"src/ops/r9700/linear/r9700_linear.hip"
HARNESS=ROOT/"tools/r9700/a8q4_n16k16_scalar_base_harness.hip"
INPUTS=(SOURCE,HEADER,ASSEMBLY,CANDIDATE,BASELINE,NOTES,BASE_DISASM,CAND_DISASM,CHECKER,PROD_OBJECT,PROD_SOURCE,HARNESS,BINARY,Path(__file__).resolve())
SHAPES=[("value_z",12288,5120,48),("gdn_output",5120,6144,64),("mlp_down",5120,17408,64)]

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def run(c:list[str])->subprocess.CompletedProcess[str]:return subprocess.run(c,cwd=ROOT,text=True,capture_output=True,check=False)
def mad(v:list[float])->float:
 m=statistics.median(v);return statistics.median(abs(x-m) for x in v)
def power()->dict:
 c=["/opt/rocm/bin/rocm-smi","-d","0","--showproductname","--showprofile","--showperflevel","--json"];r=run(c)
 if r.returncode:raise RuntimeError("rocm-smi failed")
 j=json.loads(r.stdout);level=POWER.read_text().strip()
 if set(j)!={"card0"} or j["card0"].get("Card Series")!="AMD Radeon AI PRO R9700" or j["card0"].get("GFX Version")!="gfx1201" or j["card0"].get("Performance Level")!="auto" or level!="auto":raise RuntimeError("device0 R9700/gfx1201 must be exact auto")
 return {"command":c,"stdout":j,"sysfs_path":str(POWER),"sysfs_value":level}
def validate_power(value:dict)->None:
 expected=["/opt/rocm/bin/rocm-smi","-d","0","--showproductname","--showprofile","--showperflevel","--json"]
 j=value.get("stdout",{})
 if value.get("command")!=expected or value.get("sysfs_path")!=str(POWER) or value.get("sysfs_value")!="auto" or set(j)!={"card0"} or j["card0"].get("Card Series")!="AMD Radeon AI PRO R9700" or j["card0"].get("GFX Version")!="gfx1201" or j["card0"].get("Performance Level")!="auto":raise RuntimeError("power evidence")
def decision(raw:dict)->dict:
 if raw.get("schema")!="ninfer.r9700.a8q4-n16k16-scalar-base-measurement.v2":raise RuntimeError("measurement schema")
 if raw.get("device")!={"ordinal":0,"name":"AMD Radeon AI PRO R9700","architecture":"gfx1201"}:raise RuntimeError("device")
 if raw.get("warmup_order")!=["A","B","B","A"] or raw.get("pair_order")!=["AB","BA"]*4:raise RuntimeError("balanced order")
 cells=raw.get("cells",[])
 if len(cells)!=3:raise RuntimeError("cell count")
 out=[];weighted=0.0;stable=True;ratios=[]
 for cell,expected in zip(cells,SHAPES):
  name,n,k,calls=expected
  if {x:cell.get(x) for x in ("name","tokens","rows","columns","calls","iterations_per_sample")}!={"name":name,"tokens":2048,"rows":n,"columns":k,"calls":calls,"iterations_per_sample":128}:raise RuntimeError("cell identity")
  a=cell.get("control_interval_samples_ms",[]);b=cell.get("candidate_interval_samples_ms",[])
  if len(a)!=8 or len(b)!=8 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in a+b):raise RuntimeError("trials")
  if min(a+b)<100:raise RuntimeError("interval too short")
  a=[x/128 for x in a];b=[x/128 for x in b];am=statistics.median(a);bm=statistics.median(b);amd=mad(a);bmd=mad(b)
  s=amd/am<=.02 and bmd/bm<=.02 and (max(a)-min(a))/am<=.06 and (max(b)-min(b))/bm<=.06
  savings=[x-y for x,y in zip(a,b)];pair_ratios=[y/x for x,y in zip(a,b)];sm=statistics.median(savings);smd=mad(savings);rm=statistics.median(pair_ratios);rmd=mad(pair_ratios);upper=rm+4.4478*rmd
  stable&=s;ratios.append(upper);weighted+=sm*calls
  out.append({"name":name,"control_median_ms":am,"candidate_median_ms":bm,"control_mad_ms":amd,"candidate_mad_ms":bmd,"paired_saving_median_ms":sm,"paired_saving_mad_ms":smd,"paired_ratio_median":rm,"paired_ratio_mad":rmd,"robust_ratio_upper":upper,"stability_pass":s})
 weighted_uncertainty=4.4478*sum(calls*cell["paired_saving_mad_ms"] for cell,(_,_,_,calls) in zip(out,SHAPES));lower=weighted-weighted_uncertainty
 admitted=stable and max(ratios)<=1.01 and lower>=25.0
 return {"classification":"pass" if admitted else ("inconclusive" if not stable else "terminal_reject"),"first_stable_timing_miss_is_terminal":True,"stability_pass":stable,"robust_mad_factor":4.4478,"cells":out,"weighted_paired_saving_point_ms":weighted,"weighted_saving_uncertainty_ms":weighted_uncertainty,"robust_weighted_saving_lower_ms":lower,"weighted_saving_lower_ms_min":25.0,"every_cell_robust_ratio_upper_max":1.01,"admission_pass":admitted}
def validate(p:dict)->None:
 if p.get("schema")!="ninfer.r9700.a8q4-n16k16-scalar-base-gate.v1":raise RuntimeError("report schema")
 expected=[str(PY),"-m","tools.r9700.run_a8q4_n16k16_scalar_base_gate","--benchmark","--output",str(REPORT)]
 if p.get("command")!=expected:raise RuntimeError("command")
 keys={str(x.relative_to(ROOT)) for x in INPUTS}
 if set(p.get("hashes_sha256",{}))!=keys:raise RuntimeError("hash keyset")
 for n,d in p["hashes_sha256"].items():
  q=ROOT/n
  if not q.is_file() or q.is_symlink() or sha(q)!=d:raise RuntimeError("input hash")
 if p.get("decision")!=decision(p.get("measurement",{})):raise RuntimeError("decision")
 r=p.get("bound_regression",{})
 if r.get("command")!=[str(BINARY),"--regression"] or r.get("returncode")!=0 or r.get("stderr") or not r.get("stdout","").startswith("PASS exact_shapes=3"):raise RuntimeError("regression")
 s=p.get("static",{})
 static_command=[str(PY),"-m","tools.r9700.check_a8q4_n16k16_scalar_base_static","--source",str(SOURCE),"--assembly",str(ASSEMBLY),"--baseline",str(BASELINE),"--candidate",str(CANDIDATE)]
 if s.get("command")!=static_command or not s.get("stdout","").startswith("PASS logical_vgpr=90 architectural_vgpr=96 lds_bytes=17152 occupancy=16 iu4=8 max_voffset=44564472"):raise RuntimeError("static")
 if p.get("resources")!={"logical_vgpr":90,"architectural_vgpr":96,"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"sgpr_spills":0,"vgpr_spills":0,"native_signed_iu4_sites":8,"max_zero_extended_voffset":44564472,"production_kernarg_bytes":72}:raise RuntimeError("resources")
 for key in ("power_before","power_after"):validate_power(p.get(key,{}))
def publish(path:Path,p:dict)->None:
 parent=path.parent
 if not parent.is_dir() or parent.is_symlink() or os.path.lexists(path):raise RuntimeError("fresh output")
 fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".pending",dir=parent);tmp=Path(name);owned=None;linked=False
 try:
  s=os.fstat(fd);owned=(s.st_dev,s.st_ino,s.st_uid)
  data=(json.dumps(p,indent=2,sort_keys=True)+"\n").encode();view=memoryview(data)
  while view:
   n=os.write(fd,view)
   if n<=0:raise RuntimeError("short write")
   view=view[n:]
  os.fsync(fd);os.fchmod(fd,0o444)
  if (lambda s:(s.st_dev,s.st_ino,s.st_uid))(os.lstat(tmp))!=owned:raise RuntimeError("pending replaced")
  os.link(tmp,path);linked=True
  if (lambda s:(s.st_dev,s.st_ino,s.st_uid))(os.lstat(path))!=owned or path.read_bytes()!=data:raise RuntimeError("readback")
  validate(json.loads(path.read_text()))
  d=os.open(parent,os.O_RDONLY);os.fsync(d);os.close(d)
  for q in (tmp,path):
   if (lambda s:(s.st_dev,s.st_ino,s.st_uid))(os.lstat(q))!=owned:raise RuntimeError("inode race")
  if path.read_bytes()!=data:raise RuntimeError("late bytes")
 except Exception:
  if linked and os.path.lexists(path) and (lambda s:(s.st_dev,s.st_ino,s.st_uid))(os.lstat(path))==owned:os.unlink(path)
  raise
 finally:
  os.close(fd)
  if owned and os.path.lexists(tmp) and (lambda s:(s.st_dev,s.st_ino,s.st_uid))(os.lstat(tmp))==owned:os.unlink(tmp)
  d=os.open(parent,os.O_RDONLY);os.fsync(d);os.close(d)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--regression",action="store_true");ap.add_argument("--benchmark",action="store_true");ap.add_argument("--output",type=Path);a=ap.parse_args()
 if a.regression==a.benchmark:raise RuntimeError("select one mode")
 output=None
 if a.benchmark:
  if a.output is None:raise RuntimeError("output required")
  output=Path(os.path.abspath(a.output))
  if output!=REPORT or os.path.lexists(output):raise RuntimeError("exact fresh output required")
 for p in INPUTS:
  if not p.is_file() or p.is_symlink():raise RuntimeError(f"missing input {p}")
 before={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
 sc=[sys.executable,"-m","tools.r9700.check_a8q4_n16k16_scalar_base_static","--source",str(SOURCE),"--assembly",str(ASSEMBLY),"--baseline",str(BASELINE),"--candidate",str(CANDIDATE)];sg=run(sc)
 if sg.returncode:raise RuntimeError("static gate failed")
 if a.regression:
  r=run([str(BINARY),"--regression"])
  if r.returncode or r.stderr or not r.stdout.startswith("PASS exact_shapes=3"):raise RuntimeError(r.stdout+r.stderr)
  print(r.stdout,end="");return 0
 pb=power();reg=run([str(BINARY),"--regression"])
 if reg.returncode or reg.stderr or not reg.stdout.startswith("PASS exact_shapes=3"):raise RuntimeError(reg.stdout+reg.stderr)
 measured=run([str(BINARY),"--benchmark"]);pa=power()
 if measured.returncode or measured.stderr:raise RuntimeError(measured.stdout+measured.stderr)
 raw=json.loads(measured.stdout);dec=decision(raw)
 if before!={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}:raise RuntimeError("input changed")
 report={"schema":"ninfer.r9700.a8q4-n16k16-scalar-base-gate.v1","command":[sys.executable,"-m","tools.r9700.run_a8q4_n16k16_scalar_base_gate","--benchmark","--output",str(output)],"hashes_sha256":before,"static":{"command":sc,"stdout":sg.stdout},"resources":{"logical_vgpr":90,"architectural_vgpr":96,"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"sgpr_spills":0,"vgpr_spills":0,"native_signed_iu4_sites":8,"max_zero_extended_voffset":44564472,"production_kernarg_bytes":72},"bound_regression":{"command":reg.args,"returncode":reg.returncode,"stdout":reg.stdout,"stderr":reg.stderr},"power_before":pb,"power_after":pa,"measurement":raw,"decision":dec}
 assert output is not None;publish(output,report);return 0
if __name__=="__main__":raise SystemExit(main())
