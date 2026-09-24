#!/usr/bin/env python3
"""Run and durably publish the exact-P2048 W-only DEVICE_HT qualifier."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, stat, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/"tools/r9700/build/a8q4_n16k16_weight_ht_harness"
BASE_ASM=ROOT/"tools/r9700/build/a8q4_n16k16_weight_ht_baseline.hsaco"
ASM=ROOT/"tools/r9700/build/a8q4_n16k16_weight_ht_candidate.hsaco"
BASE_DISASM=ROOT/"tools/r9700/build/a8q4_n16k16_weight_ht_baseline.objdump"
DISASM=ROOT/"tools/r9700/build/a8q4_n16k16_weight_ht_candidate.objdump"
NOTES=ROOT/"tools/r9700/build/a8q4_n16k16_weight_ht_candidate.notes"
FATBIN=ROOT/"tools/r9700/build/a8q4_n16k16_weight_ht_fatbin.bin"
PRODUCTION_OBJECT=ROOT/"build-r9700/src/CMakeFiles/ninfer_r9700_core.dir/ops/r9700/linear/r9700_linear.hip.o"
PRODUCTION_BINARY=ROOT/"build-r9700/bench/ninfer_bench"
CHECKER=ROOT/"tools/r9700/check_a8q4_n16k16_weight_ht_static.py"
VALIDATOR=ROOT/"tools/r9700/validate_a8q4_n16k16_weight_ht_report.py"
INPUTS=(ROOT/"tools/r9700/patch_a8q4_n16k16_weight_ht_assembly.py",
 ROOT/"tools/r9700/a8q4_n16k16_weight_ht_harness.hip",CHECKER,VALIDATOR,
 Path(__file__).resolve(),ROOT/"tools/r9700/a8q4_n16k16_layout_qual.hip",
 ROOT/"src/ops/r9700/linear/r9700_linear.hip",
 ROOT/"src/ops/r9700/linear/r9700_linear.h",PRODUCTION_OBJECT,PRODUCTION_BINARY,
 FATBIN,BASE_ASM,ASM,BASE_DISASM,DISASM,NOTES,BINARY)
RATIO_GATE=.86874
SAVING_GATE=.390625
PY=Path("/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python")
REPORT=ROOT/"profiles/bench/r9700-a8q4-n16k16-weight-device-ht-p2048-ab-20260905.json"
POWER=Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd:list[str])->subprocess.CompletedProcess[str]:return subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
def validate_power_state(j:object,level:str)->None:
 if not isinstance(j,dict) or set(j)!={"card0"} or not isinstance(j["card0"],dict) or \
    j["card0"].get("Card Series")!="AMD Radeon AI PRO R9700" or \
    j["card0"].get("GFX Version")!="gfx1201" or \
    j["card0"].get("Performance Level")!="auto" or level!="auto":
  raise RuntimeError("ROCm device0 R9700/gfx1201 and DRM card2 must both be in exact auto mode")
def power()->dict:
 c=["/opt/rocm/bin/rocm-smi","-d","0","--showproductname","--showprofile","--showperflevel","--json"]
 r=run(c)
 if r.returncode:raise RuntimeError("rocm-smi failed")
 j=json.loads(r.stdout);level=POWER.read_text().strip()
 validate_power_state(j,level)
 return {"command":c,"stdout":j,"sysfs_path":str(POWER),"sysfs_value":level}
def mad(v:list[float])->float:
 m=statistics.median(v);return statistics.median(abs(x-m) for x in v)
def validate_measurement(raw:dict)->dict:
 if raw.get("schema")!="ninfer.r9700.a8q4-n16k16-weight-ht-measurement.v1":raise RuntimeError("measurement schema")
 if {k:raw.get(k) for k in ("tokens","rows","columns","calls","iterations_per_sample")}!={"tokens":2048,"rows":5120,"columns":17408,"calls":64,"iterations_per_sample":128}:raise RuntimeError("exact workload")
 d=raw.get("device",{})
 if d.get("ordinal")!=0 or "R9700" not in d.get("name","") or d.get("architecture")!="gfx1201":raise RuntimeError("device identity")
 a=raw.get("control_interval_samples_ms",[]);b=raw.get("candidate_interval_samples_ms",[])
 if len(a)!=7 or len(b)!=7 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in a+b):raise RuntimeError("raw trial inventory")
 if min(a+b)<100:raise RuntimeError("event interval is too short")
 aa=[x/128 for x in a];bb=[x/128 for x in b];am=statistics.median(aa);bm=statistics.median(bb)
 amd,bmd=mad(aa),mad(bb);stable=amd/am<=.02 and bmd/bm<=.02 and (max(aa)-min(aa))/am<=.06 and (max(bb)-min(bb))/bm<=.06
 ratio=bm/am;saving=am-bm
 passed=stable and (ratio<=RATIO_GATE or saving>=SAVING_GATE)
 classification="pass" if passed else ("inconclusive" if not stable else "terminal_reject")
 return {"classification":classification,"first_stable_timing_miss_is_terminal":True,
  "stability_pass":stable,"control_median_ms":am,"candidate_median_ms":bm,
  "control_mad_ms":amd,"candidate_mad_ms":bmd,"candidate_over_control":ratio,
  "saving_ms_per_call":saving,"candidate_over_control_max":RATIO_GATE,
  "saving_ms_per_call_min":SAVING_GATE,"admission_pass":passed}
def validate_report_payload(p:dict)->None:
 if p.get("schema")!="ninfer.r9700.a8q4-n16k16-weight-device-ht-gate.v1":raise RuntimeError("report schema")
 expected_command=[str(PY),"-m","tools.r9700.run_a8q4_n16k16_weight_ht_gate","--benchmark","--output",str(REPORT)]
 if p.get("command")!=expected_command:raise RuntimeError("report command")
 expected_hashes={str(x.relative_to(ROOT)) for x in INPUTS}
 if set(p.get("hashes_sha256",{}))!=expected_hashes:raise RuntimeError("hash keyset")
 expected_static=[str(PY),"-m","tools.r9700.check_a8q4_n16k16_weight_ht_static","--baseline",str(BASE_ASM),"--candidate",str(ASM),"--baseline-disassembly",str(BASE_DISASM),"--candidate-disassembly",str(DISASM),"--notes",str(NOTES)]
 s=p.get("static",{})
 if s.get("command")!=expected_static or not s.get("stdout","").startswith("PASS ") or "exact_hsaco_equality=true" not in s.get("stdout",""):raise RuntimeError("static contract")
 if p.get("decision")!=validate_measurement(p.get("measurement",{})):raise RuntimeError("decision recomputation")
 r=p.get("bound_regression",{})
 if r.get("command")!=[str(BINARY),"--regression"] or r.get("returncode")!=0 or r.get("stderr") or not r.get("stdout","").startswith("PASS "):raise RuntimeError("bound regression")
 if p.get("resources")!={"logical_vgpr":92,"architectural_vgpr":96,"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"sgpr_spills":0,"vgpr_spills":0,"native_signed_iu4_sites":8,"weight_ht_b64_static_sites":2,"activation_policy":"default","scale_policy":"default","complete_hsaco_otherwise_identical":True,"production_kernarg_bytes":72}:raise RuntimeError("resource/static contract")
 expected_power=["/opt/rocm/bin/rocm-smi","-d","0","--showproductname","--showprofile","--showperflevel","--json"]
 for name in ("power_before","power_after"):
  q=p.get(name,{})
  if q.get("command")!=expected_power or q.get("sysfs_path")!=str(POWER) or q.get("sysfs_value")!="auto":raise RuntimeError(name)
  try:validate_power_state(q.get("stdout"),q.get("sysfs_value"))
  except RuntimeError as error:raise RuntimeError(name) from error
 for n,d in p.get("hashes_sha256",{}).items():
  q=ROOT/n
  if not q.is_file() or q.is_symlink() or sha(q)!=d:raise RuntimeError(f"input hash {n}")
def publish(path:Path,p:dict)->None:
 path=Path(os.path.abspath(path));parent=path.parent
 if not parent.is_dir() or parent.is_symlink() or os.path.lexists(path):raise RuntimeError("output must be a fresh leaf in a real directory")
 fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".pending",dir=parent);tmp=Path(name);owned=None;final=None
 try:
  s=os.fstat(fd);owned=(s.st_dev,s.st_ino,s.st_uid)
  if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or s.st_nlink!=1:raise RuntimeError("pending inode")
  data=(json.dumps(p,indent=2,sort_keys=True)+"\n").encode();view=memoryview(data)
  while view:
   written=os.write(fd,view)
   if written<=0:raise RuntimeError("short report write")
   view=view[written:]
  os.fsync(fd);os.fchmod(fd,0o444)
  s=os.lstat(tmp)
  if (s.st_dev,s.st_ino,s.st_uid)!=owned:raise RuntimeError("pending replaced")
  os.link(tmp,path);final=owned
  s=os.lstat(path)
  if (s.st_dev,s.st_ino,s.st_uid)!=owned or path.read_bytes()!=data:raise RuntimeError("publication readback")
  validate_report_payload(json.loads(path.read_text()))
  d=os.open(parent,os.O_RDONLY)
  try:os.fsync(d)
  finally:os.close(d)
  for q in (tmp,path):
   s=os.lstat(q)
   if (s.st_dev,s.st_ino,s.st_uid)!=owned:raise RuntimeError("publication inode changed")
  if path.read_bytes()!=data:raise RuntimeError("publication bytes changed")
 except Exception:
  if final and os.path.lexists(path):
   s=os.lstat(path)
   if (s.st_dev,s.st_ino,s.st_uid)==final:os.unlink(path)
  raise
 finally:
  os.close(fd)
  if owned and os.path.lexists(tmp):
   s=os.lstat(tmp)
   if (s.st_dev,s.st_ino,s.st_uid)==owned:os.unlink(tmp)
  d=os.open(parent,os.O_RDONLY)
  try:os.fsync(d)
  finally:os.close(d)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--regression",action="store_true");ap.add_argument("--benchmark",action="store_true");ap.add_argument("--output",type=Path);a=ap.parse_args()
 if a.regression==a.benchmark:raise RuntimeError("select exactly one mode")
 output=None
 if a.benchmark:
  if a.output is None:raise RuntimeError("--benchmark requires --output")
  output=Path(os.path.abspath(a.output))
  if output!=REPORT or os.path.lexists(output):raise RuntimeError("exact fresh immutable output path")
 for p in INPUTS:
  if not p.is_file() or p.is_symlink():raise RuntimeError(f"missing/nonregular input {p}")
 before={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
 sg=run([sys.executable,"-m","tools.r9700.check_a8q4_n16k16_weight_ht_static","--baseline",str(BASE_ASM),"--candidate",str(ASM),"--baseline-disassembly",str(BASE_DISASM),"--candidate-disassembly",str(DISASM),"--notes",str(NOTES)])
 if sg.returncode:raise RuntimeError(f"static gate failed: {sg.stdout}{sg.stderr}")
 if a.regression:
  r=run([str(BINARY),"--regression"])
  if r.returncode or r.stderr or not r.stdout.startswith("PASS "):raise RuntimeError(f"regression failed: {r.stdout}{r.stderr}")
  print(r.stdout,end="");return 0
 pb=power();reg=run([str(BINARY),"--regression"])
 if reg.returncode or reg.stderr or not reg.stdout.startswith("PASS "):raise RuntimeError(f"bound regression failed: {reg.stdout}{reg.stderr}")
 measured=run([str(BINARY),"--benchmark"]);pa=power()
 if measured.returncode or measured.stderr:raise RuntimeError(f"benchmark failed: {measured.stdout}{measured.stderr}")
 raw=json.loads(measured.stdout);decision=validate_measurement(raw)
 after={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
 if before!=after:raise RuntimeError("executed input changed")
 fields=dict(re.findall(r"\b(vgpr|architectural_vgpr|lds|iu4|weight_ht_b64|occupancy|scratch|spills)=(\d+)\b",sg.stdout))
 if fields!={"vgpr":"92","architectural_vgpr":"96","lds":"17152","iu4":"8","weight_ht_b64":"2","occupancy":"16","scratch":"0","spills":"0"}:raise RuntimeError("static resource emission")
 assert output is not None
 report={"schema":"ninfer.r9700.a8q4-n16k16-weight-device-ht-gate.v1","command":[sys.executable,"-m","tools.r9700.run_a8q4_n16k16_weight_ht_gate","--benchmark","--output",str(output)],"hashes_sha256":before,"static":{"command":sg.args,"stdout":sg.stdout},"resources":{"logical_vgpr":92,"architectural_vgpr":96,"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"sgpr_spills":0,"vgpr_spills":0,"native_signed_iu4_sites":8,"weight_ht_b64_static_sites":2,"activation_policy":"default","scale_policy":"default","complete_hsaco_otherwise_identical":True,"production_kernarg_bytes":72},"bound_regression":{"command":reg.args,"returncode":reg.returncode,"stdout":reg.stdout,"stderr":reg.stderr},"power_before":pb,"power_after":pa,"measurement":raw,"decision":decision}
 publish(output,report);return 0
if __name__=="__main__":raise SystemExit(main())
