#!/usr/bin/env python3
"""Run and durably publish the disconnected exact-G128 qualification gate."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, stat, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/"tools/r9700/build/a8g128_q4g128_n16k16_harness"
ASM=ROOT/"tools/r9700/build/a8g128_q4g128_n16k16_qual.s"
SOURCE=ROOT/"tools/r9700/a8g128_q4g128_n16k16_qual.hip"
CHECKER=ROOT/"tools/r9700/check_a8g128_q4g128_n16k16_static.py"
VALIDATOR=ROOT/"tools/r9700/validate_a8g128_q4g128_n16k16_report.py"
INPUTS=(SOURCE,ROOT/"tools/r9700/a8g128_q4g128_n16k16_qual.h",
 ROOT/"tools/r9700/a8g128_q4g128_n16k16_harness.hip",CHECKER,VALIDATOR,Path(__file__).resolve(),
 ROOT/"src/ops/r9700/linear/r9700_linear.hip",ROOT/"src/ops/r9700/linear/r9700_linear.h",
 ROOT/"src/ops/r9700/eager/eager_ops.hip",ROOT/"src/ops/r9700/eager/eager_ops.h",ASM,BINARY)
SHAPES={("value_z",12288,5120,48),("gdn_output",5120,6144,48),("mlp_down",5120,17408,64)}
TOKENS=(1,2,3,4,8,12,16,24,36,48,2048)
CONTROL_CONTRACT="nonfused=canonical_a8g64_prepare_plus_gemm;mlp_down_T_lt_2048=product_eager_silu_mul_bf16_strided_plus_canonical_a8g64_prepare_plus_gemm;mlp_down_T_2048=canonical_fused_silu_a8g64_prepare_plus_gemm"
STREAM_CONTRACT="one_owned_nondefault_stream_per_cell;both_routes_and_events_on_same_stream"
BATCH_CONTRACT="T_lt_2048=1024_calls_per_event_interval;T_2048=128_calls_per_event_interval;reported_samples_are_raw_interval_ms"
UNCERTAINTY_FACTOR=3.0*1.4826

def mad(values:list[float])->float:
 center=statistics.median(values)
 return statistics.median(abs(v-center) for v in values)

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd:list[str])->subprocess.CompletedProcess[str]:return subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
def power()->dict:
 c=["/opt/rocm/bin/rocm-smi","-d","0","--showproductname","--showprofile","--showperflevel","--json"]
 r=run(c)
 if r.returncode:raise RuntimeError("rocm-smi failed")
 j=json.loads(r.stdout);flat=json.dumps(j).lower()
 if "r9700" not in flat or "auto" not in flat:raise RuntimeError("device0 must be R9700 in auto")
 return {"command":c,"stdout":j}
def validate(raw:dict)->dict:
 if raw!={**raw,"schema_version":2,"status":"measured"}:raise RuntimeError("schema/status")
 if raw.get("timing_weight_contract")!="identical_dense_logical_signed_q4_codes_and_paired_equal_scales":raise RuntimeError("timing weight fairness")
 if raw.get("control_boundary_contract")!=CONTROL_CONTRACT:raise RuntimeError("control boundary contract")
 if raw.get("timing_stream_contract")!=STREAM_CONTRACT:raise RuntimeError("timing stream contract")
 if raw.get("timing_batch_contract")!=BATCH_CONTRACT:raise RuntimeError("timing batch contract")
 d=raw.get("device",{})
 if d.get("ordinal")!=0 or "R9700" not in d.get("name","") or d.get("architecture")!="gfx1201":raise RuntimeError("HIP device identity")
 rows=raw.get("trials",[])
 expected={(r,n,k,c,t) for r,n,k,c in SHAPES for t in TOKENS}
 got={(x.get("role"),x.get("rows"),x.get("columns"),x.get("calls"),x.get("tokens")) for x in rows}
 if len(rows)!=33 or got!=expected:raise RuntimeError("exact timing inventory")
 normalized={}
 stability_rows=[]
 for x in rows:
  iterations=128 if x["tokens"]==2048 else 1024
  if x.get("iterations_per_sample")!=iterations:raise RuntimeError("iterations per sample")
  boundary="canonical_a8g64_prepare_plus_gemm" if x["role"]!="mlp_down" else ("canonical_fused_silu_a8g64_prepare_plus_gemm" if x["tokens"]==2048 else "product_eager_silu_mul_bf16_strided_plus_canonical_a8g64_prepare_plus_gemm")
  if x.get("control_boundary")!=boundary:raise RuntimeError("per-cell control boundary")
  if len(x.get("control_interval_samples_ms",[]))!=7 or len(x.get("candidate_interval_samples_ms",[]))!=7:raise RuntimeError("raw trials")
  if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in x["control_interval_samples_ms"]+x["candidate_interval_samples_ms"]):raise RuntimeError("raw interval value")
  minimum_interval=100.0 if x["tokens"]==2048 else 32.0
  if min(x["control_interval_samples_ms"]+x["candidate_interval_samples_ms"])<minimum_interval:raise RuntimeError("event interval too short")
  cs=[v/iterations for v in x["control_interval_samples_ms"]];ns=[v/iterations for v in x["candidate_interval_samples_ms"]]
  cm=statistics.median(cs);nm=statistics.median(ns);cmd=mad(cs);nmd=mad(ns)
  if abs(cm-x["control_ms"])>1e-6 or abs(nm-x["candidate_ms"])>1e-6 or abs(nm/cm-x["ratio"])>1e-6:raise RuntimeError("median/ratio")
  stable=cmd/cm<=.02 and nmd/nm<=.02 and (max(cs)-min(cs))/cm<=.06 and (max(ns)-min(ns))/nm<=.06
  key=(x["role"],x["tokens"]);normalized[key]=(x,cs,ns,cm,nm,cmd,nmd)
  stability_rows.append({"role":x["role"],"tokens":x["tokens"],"control_mad_fraction":cmd/cm,"candidate_mad_fraction":nmd/nm,"control_range_fraction":(max(cs)-min(cs))/cm,"candidate_range_fraction":(max(ns)-min(ns))/nm,"stable":stable})
 bounds=[]
 for x,cs,ns,cm,nm,cmd,nmd in normalized.values():
  cu=UNCERTAINTY_FACTOR*cmd;nu=UNCERTAINTY_FACTOR*nmd
  if cm<=cu or nm<=nu:raise RuntimeError("uncertainty exceeds positive timing")
  bounds.append({"role":x["role"],"tokens":x["tokens"],"control_median_ms":cm,"candidate_median_ms":nm,"control_uncertainty_ms":cu,"candidate_uncertainty_ms":nu,"lower_ratio":(nm-nu)/(cm+cu),"point_ratio":nm/cm,"upper_ratio":(nm+nu)/(cm-cu)})
 p=[x for x in rows if x["tokens"]==2048]
 control_series=[sum(normalized[(x["role"],2048)][1][q]*x["calls"] for x in p) for q in range(7)]
 candidate_series=[sum(normalized[(x["role"],2048)][2][q]*x["calls"] for x in p) for q in range(7)]
 control_median=statistics.median(control_series);candidate_median=statistics.median(candidate_series)
 control_mad=mad(control_series);candidate_mad=mad(candidate_series)
 aggregate_stable=control_mad/control_median<=.01 and candidate_mad/candidate_median<=.01
 saving=sum(x["calls"]*(normalized[(x["role"],2048)][3]-normalized[(x["role"],2048)][4]) for x in p)
 control_uncertainty=UNCERTAINTY_FACTOR*sum(x["calls"]*normalized[(x["role"],2048)][5] for x in p)
 candidate_uncertainty=UNCERTAINTY_FACTOR*sum(x["calls"]*normalized[(x["role"],2048)][6] for x in p)
 saving_lower=saving-control_uncertainty-candidate_uncertainty
 saving_upper=saving+control_uncertainty+candidate_uncertainty
 stable=all(x["stable"] for x in stability_rows) and aggregate_stable
 upper_nonregression=all(x["upper_ratio"]<=1.01 for x in bounds)
 material_regression=any(x["lower_ratio"]>1.01 for x in bounds)
 if not stable:classification="inconclusive"
 elif upper_nonregression and saving_lower>=25.0:classification="pass"
 elif material_regression or saving_upper<25.0:classification="reject"
 else:classification="inconclusive"
 weighted={}
 for t in TOKENS:
  cells=[x for x in rows if x["tokens"]==t]
  control=sum(x["calls"]*normalized[(x["role"],t)][3] for x in cells)
  candidate=sum(x["calls"]*normalized[(x["role"],t)][4] for x in cells)
  weighted[str(t)]={"control_ms":control,"candidate_ms":candidate,
                    "saving_ms":control-candidate,"ratio":candidate/control}
 return {"classification":classification,"uncertainty":{"definition":"3*1.4826*MAD","factor":UNCERTAINTY_FACTOR},"stability_pass":stable,"per_cell_stability":stability_rows,
         "per_cell_ratio_bounds":bounds,"every_cell_upper_ratio_le_1_01":upper_nonregression,
         "material_per_cell_regression":material_regression,"p2048_weighted_saving_ms":saving,
         "p2048_weighted_saving_lower_ms":saving_lower,"p2048_weighted_saving_upper_ms":saving_upper,
         "p2048_weighted_saving_gate_ms":25.0,"p2048_weighted_control_uncertainty_ms":control_uncertainty,"p2048_weighted_candidate_uncertainty_ms":candidate_uncertainty,"p2048_aggregate":{"purpose":"diagnostic_stability_only; shape sample indices are sequential, not joint","control_samples_ms":control_series,"candidate_samples_ms":candidate_series,"control_median_ms":control_median,"candidate_median_ms":candidate_median,"control_mad_ms":control_mad,"candidate_mad_ms":candidate_mad,"control_mad_fraction":control_mad/control_median,"candidate_mad_fraction":candidate_mad/candidate_median,"stable":aggregate_stable},
         "weighted_by_tokens":weighted,"repeat_policy":"at_most_one_repeat_on_frozen_bytes_if_inconclusive",
         "admission_pass":classification=="pass"}
def validate_report_payload(payload:dict)->None:
 if payload.get("schema")!="ninfer.r9700.a8g128-q4g128-n16k16-gate.v2":raise RuntimeError("report schema")
 if payload.get("decision")!=validate(payload.get("measurement",{})):raise RuntimeError("report decision recomputation")
 reg=payload.get("bound_regression",{})
 if reg.get("returncode")!=0 or reg.get("stderr") or not reg.get("stdout","").startswith("PASS "):raise RuntimeError("bound regression evidence")
 r=payload.get("resource_contract",{})
 if r.get("prefill")!={"vgpr":r.get("prefill",{}).get("vgpr"),"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"spills":0,"static_iu4_sites":8,"dynamic_iu4_per_g128":16} or r["prefill"]["vgpr"]>96:raise RuntimeError("prefill resource contract")
 if r.get("wave")!={"vgpr":r.get("wave",{}).get("vgpr"),"lds_bytes":0,"occupancy_waves_per_eu":16,"scratch_bytes":0,"spills":0,"static_iu4_sites":8} or r["wave"]["vgpr"]>96:raise RuntimeError("wave resource contract")
 for name,digest in payload.get("hashes_sha256",{}).items():
  p=ROOT/name
  if not p.is_file() or p.is_symlink() or sha(p)!=digest:raise RuntimeError(f"report input hash {name}")
def publish(path:Path,payload:dict)->None:
 path=Path(os.path.abspath(path));parent=path.parent
 if not parent.is_dir() or parent.is_symlink():raise RuntimeError("output parent must be a real existing directory")
 if os.path.lexists(path):raise FileExistsError(path)
 fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".pending",dir=parent);tmp=Path(name);owned=None;final_owned=None
 try:
  st=os.fstat(fd);owned=(st.st_dev,st.st_ino,st.st_uid)
  if not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid() or st.st_nlink!=1:raise RuntimeError("pending inode")
  data=(json.dumps(payload,indent=2,sort_keys=True)+"\n").encode();os.write(fd,data);os.fsync(fd);os.fchmod(fd,0o444)
  if (os.lstat(tmp).st_dev,os.lstat(tmp).st_ino,os.lstat(tmp).st_uid)!=owned:raise RuntimeError("pending replaced")
  os.link(tmp,path);final_owned=owned
  fs=os.lstat(path)
  if (fs.st_dev,fs.st_ino,fs.st_uid)!=owned or path.read_bytes()!=data:raise RuntimeError("publication readback")
  validate_report_payload(json.loads(path.read_text()))
  dfd=os.open(parent,os.O_RDONLY)
  try:os.fsync(dfd)
  finally:os.close(dfd)
  fs=os.lstat(path)
  if (fs.st_dev,fs.st_ino,fs.st_uid)!=owned or path.read_bytes()!=data:raise RuntimeError("post-validation publication changed")
 except Exception:
  if final_owned and os.path.lexists(path):
   s=os.lstat(path)
   if (s.st_dev,s.st_ino,s.st_uid)==final_owned:os.unlink(path)
  raise
 finally:
  os.close(fd)
  if owned and os.path.lexists(tmp):
   s=os.lstat(tmp)
   if (s.st_dev,s.st_ino,s.st_uid)==owned:os.unlink(tmp)
  dfd=os.open(parent,os.O_RDONLY)
  try:os.fsync(dfd)
  finally:os.close(dfd)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--regression",action="store_true");ap.add_argument("--benchmark",action="store_true");ap.add_argument("--output",type=Path);a=ap.parse_args()
 if a.regression==a.benchmark:raise RuntimeError("select exactly one mode")
 for p in INPUTS:
  if not p.is_file() or p.is_symlink():raise RuntimeError(f"missing/nonregular input {p}")
 before={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
 sg=run([sys.executable,str(CHECKER),"--source",str(SOURCE),"--assembly",str(ASM)])
 if sg.returncode:raise RuntimeError(f"static gate failed: {sg.stderr}")
 if a.regression:
  r=run([str(BINARY),"--regression"])
  if r.returncode or not r.stdout.startswith("PASS "):raise RuntimeError(f"regression failed: {r.stdout}{r.stderr}")
  print(r.stdout,end="");return 0
 if a.output is None:raise RuntimeError("--benchmark requires --output")
 regression=run([str(BINARY),"--regression"])
 if regression.returncode or regression.stderr or not regression.stdout.startswith("PASS "):raise RuntimeError(f"bound regression failed: {regression.stdout}{regression.stderr}")
 pb=power();r=run([str(BINARY),"--benchmark"]);pa=power()
 if r.returncode or r.stderr:raise RuntimeError(f"benchmark failed: {r.stdout}{r.stderr}")
 raw=json.loads(r.stdout);decision=validate(raw)
 after={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
 if before!=after:raise RuntimeError("executed input changed")
 fields=dict(re.findall(r"\b(prefill_vgpr|prefill_lds|wave_vgpr|wave_lds)=(\d+)\b",sg.stdout))
 if set(fields)!={"prefill_vgpr","prefill_lds","wave_vgpr","wave_lds"}:raise RuntimeError("static resource emission")
 report={"schema":"ninfer.r9700.a8g128-q4g128-n16k16-gate.v2","command":[sys.executable,"-m","tools.r9700.run_a8g128_q4g128_n16k16_gate","--benchmark","--output",str(Path(os.path.abspath(a.output)))],"hashes_sha256":before,"static":{"command":sg.args,"stdout":sg.stdout},"bound_regression":{"command":regression.args,"stdout":regression.stdout,"stderr":regression.stderr,"returncode":regression.returncode},"power_before":pb,"power_after":pa,"resource_contract":{"prefill":{"vgpr":int(fields["prefill_vgpr"]),"lds_bytes":int(fields["prefill_lds"]),"occupancy_waves_per_eu":16,"scratch_bytes":0,"spills":0,"static_iu4_sites":8,"dynamic_iu4_per_g128":16},"wave":{"vgpr":int(fields["wave_vgpr"]),"lds_bytes":int(fields["wave_lds"]),"occupancy_waves_per_eu":16,"scratch_bytes":0,"spills":0,"static_iu4_sites":8}},"measurement":raw,"decision":decision}
 publish(a.output,report);return 0
if __name__=="__main__":raise SystemExit(main())
