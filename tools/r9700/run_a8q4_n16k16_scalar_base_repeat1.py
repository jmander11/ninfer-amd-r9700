#!/usr/bin/env python3
"""Execute the one allowed independent repeat of the frozen scalar-base gate."""
from __future__ import annotations
import argparse,hashlib,json,os,stat,subprocess,sys,tempfile
from pathlib import Path
from tools.r9700 import run_a8q4_n16k16_scalar_base_gate as base

ROOT=base.ROOT
FIRST=ROOT/"profiles/bench/r9700-a8q4-n16k16-scalar-base-p2048-ab-20260905.json"
REPEAT=ROOT/"profiles/bench/r9700-a8q4-n16k16-scalar-base-p2048-ab-repeat1-20260905.json"
BASE_CLOSURE=ROOT/"profiles/bench/a8q4-n16k16-scalar-base-prepare-20260905/prepared.sha256"
FIRST_SHA="1941b3c6a262a61928d3971fc2b0afe06f7dac16db68a4221e79ae42940afb9b"
BASE_CLOSURE_SHA="fee60286307a97bcd4b51b4dda3541a22b54b20671c6c3dbd3031c6a46665c8c"
INPUTS=base.INPUTS+(FIRST,BASE_CLOSURE,Path(__file__).resolve())

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def run(command:list[str])->subprocess.CompletedProcess[str]:return subprocess.run(command,cwd=ROOT,text=True,capture_output=True,check=False)
def validate_first()->dict:
 if not FIRST.is_file() or FIRST.is_symlink() or sha(FIRST)!=FIRST_SHA:raise RuntimeError("immutable first report")
 if not BASE_CLOSURE.is_file() or BASE_CLOSURE.is_symlink() or sha(BASE_CLOSURE)!=BASE_CLOSURE_SHA:raise RuntimeError("frozen base closure")
 value=json.loads(FIRST.read_text());base.validate(value)
 if value.get("decision",{}).get("classification")!="inconclusive" or value["decision"].get("stability_pass") is not False:raise RuntimeError("first report does not authorize one repeat")
 return value
def validate(value:dict)->None:
 if value.get("schema")!="ninfer.r9700.a8q4-n16k16-scalar-base-repeat1-gate.v1" or value.get("repeat_ordinal")!=1 or value.get("pooled_with_first") is not False:raise RuntimeError("repeat schema")
 command=[str(base.PY),"-m","tools.r9700.run_a8q4_n16k16_scalar_base_repeat1","--benchmark","--output",str(REPEAT)]
 if value.get("command")!=command:raise RuntimeError("repeat command")
 first=value.get("first_report",{})
 if first!={"path":str(FIRST.relative_to(ROOT)),"sha256":FIRST_SHA,"classification":"inconclusive","pooled":False}:raise RuntimeError("first report binding")
 expected={str(path.relative_to(ROOT)) for path in INPUTS}
 if set(value.get("hashes_sha256",{}))!=expected:raise RuntimeError("hash keyset")
 for name,digest in value["hashes_sha256"].items():
  path=ROOT/name
  if not path.is_file() or path.is_symlink() or sha(path)!=digest:raise RuntimeError(f"input hash {name}")
 if value.get("decision")!=base.decision(value.get("measurement",{})):raise RuntimeError("independent repeat decision")
 regression=value.get("bound_regression",{})
 if regression.get("command")!=[str(base.BINARY),"--regression"] or regression.get("returncode")!=0 or regression.get("stderr") or not regression.get("stdout","").startswith("PASS exact_shapes=3"):raise RuntimeError("bound regression")
 static=value.get("static",{})
 static_command=[str(base.PY),"-m","tools.r9700.check_a8q4_n16k16_scalar_base_static","--source",str(base.SOURCE),"--assembly",str(base.ASSEMBLY),"--baseline",str(base.BASELINE),"--candidate",str(base.CANDIDATE)]
 if static.get("command")!=static_command or not static.get("stdout","").startswith("PASS logical_vgpr=90 architectural_vgpr=96"):raise RuntimeError("static")
 if value.get("resources")!={"logical_vgpr":90,"architectural_vgpr":96,"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"sgpr_spills":0,"vgpr_spills":0,"native_signed_iu4_sites":8,"max_zero_extended_voffset":44564472,"production_kernarg_bytes":72}:raise RuntimeError("resources")
 base.validate_power(value.get("power_before",{}));base.validate_power(value.get("power_after",{}))
def publish(path:Path,value:dict)->None:
 parent=path.parent
 if not parent.is_dir() or parent.is_symlink() or os.path.lexists(path):raise RuntimeError("fresh repeat output")
 fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".pending",dir=parent);pending=Path(name);owned=None;linked=False
 try:
  info=os.fstat(fd);owned=(info.st_dev,info.st_ino,info.st_uid);data=(json.dumps(value,indent=2,sort_keys=True)+"\n").encode();view=memoryview(data)
  while view:
   count=os.write(fd,view)
   if count<=0:raise RuntimeError("short write")
   view=view[count:]
  os.fsync(fd);os.fchmod(fd,0o444)
  if (lambda x:(x.st_dev,x.st_ino,x.st_uid))(os.lstat(pending))!=owned:raise RuntimeError("pending replaced")
  os.link(pending,path);linked=True
  if (lambda x:(x.st_dev,x.st_ino,x.st_uid))(os.lstat(path))!=owned or path.read_bytes()!=data:raise RuntimeError("publication readback")
  validate(json.loads(path.read_text()));directory=os.open(parent,os.O_RDONLY);os.fsync(directory);os.close(directory)
  for leaf in (pending,path):
   if (lambda x:(x.st_dev,x.st_ino,x.st_uid))(os.lstat(leaf))!=owned:raise RuntimeError("publication race")
  if path.read_bytes()!=data:raise RuntimeError("late readback")
 except Exception:
  if linked and os.path.lexists(path) and (lambda x:(x.st_dev,x.st_ino,x.st_uid))(os.lstat(path))==owned:os.unlink(path)
  raise
 finally:
  os.close(fd)
  if owned and os.path.lexists(pending) and (lambda x:(x.st_dev,x.st_ino,x.st_uid))(os.lstat(pending))==owned:os.unlink(pending)
  directory=os.open(parent,os.O_RDONLY);os.fsync(directory);os.close(directory)
def main()->int:
 parser=argparse.ArgumentParser();parser.add_argument("--benchmark",action="store_true");parser.add_argument("--output",type=Path);args=parser.parse_args()
 if not args.benchmark or args.output is None:raise RuntimeError("exact --benchmark --output required")
 output=Path(os.path.abspath(args.output))
 if output!=REPEAT or os.path.lexists(output):raise RuntimeError("exact fresh repeat1 output required")
 first=validate_first()
 for path in INPUTS:
  if not path.is_file() or path.is_symlink():raise RuntimeError(f"input {path}")
 before={str(path.relative_to(ROOT)):sha(path) for path in INPUTS}
 static_command=[sys.executable,"-m","tools.r9700.check_a8q4_n16k16_scalar_base_static","--source",str(base.SOURCE),"--assembly",str(base.ASSEMBLY),"--baseline",str(base.BASELINE),"--candidate",str(base.CANDIDATE)];static=run(static_command)
 if static.returncode:raise RuntimeError(static.stdout+static.stderr)
 power_before=base.power();regression=run([str(base.BINARY),"--regression"])
 if regression.returncode or regression.stderr or not regression.stdout.startswith("PASS exact_shapes=3"):raise RuntimeError(regression.stdout+regression.stderr)
 measured=run([str(base.BINARY),"--benchmark"]);power_after=base.power()
 if measured.returncode or measured.stderr:raise RuntimeError(measured.stdout+measured.stderr)
 measurement=json.loads(measured.stdout);decision=base.decision(measurement)
 if before!={str(path.relative_to(ROOT)):sha(path) for path in INPUTS}:raise RuntimeError("frozen input changed")
 value={"schema":"ninfer.r9700.a8q4-n16k16-scalar-base-repeat1-gate.v1","repeat_ordinal":1,"pooled_with_first":False,"command":[sys.executable,"-m","tools.r9700.run_a8q4_n16k16_scalar_base_repeat1","--benchmark","--output",str(output)],"first_report":{"path":str(FIRST.relative_to(ROOT)),"sha256":FIRST_SHA,"classification":first["decision"]["classification"],"pooled":False},"hashes_sha256":before,"static":{"command":static.args,"stdout":static.stdout},"resources":{"logical_vgpr":90,"architectural_vgpr":96,"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"sgpr_spills":0,"vgpr_spills":0,"native_signed_iu4_sites":8,"max_zero_extended_voffset":44564472,"production_kernarg_bytes":72},"bound_regression":{"command":regression.args,"returncode":regression.returncode,"stdout":regression.stdout,"stderr":regression.stderr},"power_before":power_before,"power_after":power_after,"measurement":measurement,"decision":decision}
 publish(output,value);return 0
if __name__=="__main__":raise SystemExit(main())
