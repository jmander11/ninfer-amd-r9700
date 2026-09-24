#!/usr/bin/env python3
"""Validate and durably publish the disconnected group-major A8G64 gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/"tools/r9700/build/a8q4_group_major_activation_harness"
ASM=ROOT/"tools/r9700/build/a8q4_group_major_activation_qual.s"
SOURCE=ROOT/"tools/r9700/a8q4_group_major_activation_qual.hip"
CHECKER=ROOT/"tools/r9700/check_a8q4_group_major_activation_static.py"
INPUTS=(SOURCE,ROOT/"tools/r9700/a8q4_group_major_activation_qual.h",
 ROOT/"tools/r9700/a8q4_group_major_activation_harness.hip",CHECKER,
 ROOT/"tools/r9700/test_check_a8q4_group_major_activation_static.py",
 ROOT/"tools/r9700/build_a8q4_group_major_activation.sh",Path(__file__).resolve(),
 ROOT/"tools/r9700/validate_a8q4_group_major_activation_report.py",
 ROOT/"profiles/bench/a8q4-group-major-activation-prepare-20260905/plan.json",
 ROOT/"profiles/bench/a8q4-group-major-activation-prepare-20260905/build.sh",
 ROOT/"profiles/bench/a8q4-group-major-activation-prepare-20260905/run.sh",
 ROOT/"tools/r9700/test_run_a8q4_group_major_activation_gate.py",
 ROOT/"src/ops/r9700/linear/r9700_linear.hip",
 ROOT/"src/ops/r9700/linear/r9700_linear.h",ASM,BINARY)
SHAPES={("value_z",12288,5120,48),("gdn_attention_output",5120,6144,64),
        ("mlp_down",5120,17408,64)}
POWER=Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
OUTPUT=ROOT/"profiles/bench/r9700-a8q4-group-major-activation-p2048-ab-20260905.json"

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def command(args:list[str])->subprocess.CompletedProcess[str]:
    return subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=False)

def power()->dict:
    cmd=["/opt/rocm/bin/rocm-smi","-d","0","--showproductname","--showprofile","--showperflevel","--json"]
    got=command(cmd)
    if got.returncode:raise RuntimeError("rocm-smi failed")
    data=json.loads(got.stdout);flat=json.dumps(data).lower();level=POWER.read_text().strip()
    if not isinstance(data,dict) or len(data)!=1 or "r9700" not in flat or level!="auto":raise RuntimeError("device0 must be R9700 in auto")
    return {"command":cmd,"result":data,"sysfs_path":str(POWER),"sysfs_value":level}

def validate_raw(raw:dict)->dict:
    if not isinstance(raw,dict) or raw.get("schema_version")!=1 or raw.get("status")!="measured":raise ValueError("raw schema")
    if raw.get("warmups")!=3 or raw.get("launches_per_interval")!=128 or raw.get("alternating_trials")!=14:raise ValueError("timing protocol")
    dev=raw.get("device")
    if not isinstance(dev,dict) or dev.get("ordinal")!=0 or "R9700" not in dev.get("name","") or dev.get("architecture")!="gfx1201":raise ValueError("device")
    cells=raw.get("cells")
    if not isinstance(cells,list) or len(cells)!=3:raise ValueError("cell count")
    seen=set();rows=[];stable=True
    for cell in cells:
        if not isinstance(cell,dict):raise ValueError("cell")
        key=(cell.get("role"),cell.get("rows"),cell.get("columns"),cell.get("calls"))
        if key not in SHAPES or key in seen or cell.get("tokens")!=2048:raise ValueError("cell identity")
        seen.add(key)
        control=cell.get("control_samples_ms");candidate=cell.get("candidate_samples_ms")
        if not isinstance(control,list) or not isinstance(candidate,list) or len(control)!=7 or len(candidate)!=7:raise ValueError("seven alternating samples")
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in control+candidate):raise ValueError("sample")
        if min(control+candidate)<100:raise ValueError("event interval too short")
        cm=statistics.median(control)/128;nm=statistics.median(candidate)/128
        cmad=statistics.median(abs(v/128-cm) for v in control);nmad=statistics.median(abs(v/128-nm) for v in candidate)
        cell_stable=cmad/cm<=.02 and nmad/nm<=.02 and (max(control)-min(control))/(128*cm)<=.06 and (max(candidate)-min(candidate))/(128*nm)<=.06
        stable=stable and cell_stable
        rows.append({"role":key[0],"rows":key[1],"columns":key[2],"calls":key[3],
                     "control_ms":cm,"candidate_ms":nm,"ratio":nm/cm,
                     "saving_ms":key[3]*(cm-nm),"stable":cell_stable,
                     "control_mad_fraction":cmad/cm,"candidate_mad_fraction":nmad/nm})
    if seen!=SHAPES:raise ValueError("shape inventory")
    saving=sum(x["saving_ms"] for x in rows)
    per_cell=all(x["ratio"]<=1.01 for x in rows)
    admitted=stable and per_cell and saving>=25.0
    return {"cells":rows,"every_cell_ratio_le_1_01":per_cell,
            "weighted_p2048_saving_ms":saving,"weighted_p2048_gate_ms":25.0,
            "stability_pass":stable,"direct_gate_pass":admitted,"whole_p2048_followup_required":True,
            "whole_p2048_minimum_saving_ms":25.0,
            "production_admission":False,
            "classification":"direct_pass_pending_whole" if admitted else ("reject" if stable else "inconclusive")}

def validate_report(report:dict)->None:
    if report.get("schema")!="ninfer.r9700.a8q4-group-major-activation-gate.v1":raise ValueError("report schema")
    if report.get("decision")!=validate_raw(report.get("measurement")):raise ValueError("decision")
    regression=report.get("bound_regression")
    if not isinstance(regression,dict) or regression.get("returncode")!=0 or regression.get("stderr") or not regression.get("stdout","").startswith("PASS shapes=3 T1_all_roles=true P2048=true codec_exact=true full_output_bit_parity=true"):raise ValueError("regression")
    resources=report.get("resources")
    if resources!={"prefill_vgpr":88,"wave_vgpr":51,"ordinary_quantizer_vgpr":10,"fused_quantizer_vgpr":13,"lds_bytes":17152,
                  "occupancy_waves_per_eu":16,"scratch_bytes":0,"spills":0,
                  "prefill_static_iu4_sites":8,"activation_b32_sites":2}:raise ValueError("resources")
    hashes=report.get("hashes_sha256")
    expected={str(p.relative_to(ROOT)) for p in INPUTS}
    if not isinstance(hashes,dict) or set(hashes)!=expected:raise ValueError("hash inventory")
    for name,digest in hashes.items():
        path=ROOT/name
        if not path.is_file() or path.is_symlink() or sha(path)!=digest:raise ValueError(f"hash {name}")
    if report.get("output")!=str(OUTPUT):raise ValueError("fixed output")
    exact=[sys.executable,"-m","tools.r9700.run_a8q4_group_major_activation_gate","--benchmark"]
    if report.get("command")!=exact:raise ValueError("command")
    for name in ("power_before","power_after"):
        value=report.get(name)
        if not isinstance(value,dict) or value.get("sysfs_path")!=str(POWER) or value.get("sysfs_value")!="auto":raise ValueError(name)
    static=report.get("static")
    if not isinstance(static,dict) or not static.get("stdout","").startswith("PASS layout=group-major-g64"):raise ValueError("static")

def unlink_owned(path:Path,owner:tuple[int,int,int]|None)->None:
    if owner is None or not os.path.lexists(path):return
    s=os.lstat(path)
    if stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino,s.st_uid)==owner:os.unlink(path)

def publish(path:Path,payload:dict)->None:
    path=Path(os.path.abspath(path));parent=path.parent
    if not parent.is_dir() or parent.is_symlink() or os.path.lexists(path):raise FileExistsError(path)
    data=(json.dumps(payload,indent=2,sort_keys=True)+"\n").encode()
    fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".pending",dir=parent);pending=Path(name);owner=None;linked=False
    try:
        s=os.fstat(fd);owner=(s.st_dev,s.st_ino,s.st_uid)
        view=memoryview(data)
        while view:view=view[os.write(fd,view):]
        os.fsync(fd);os.fchmod(fd,0o444)
        ps=os.lstat(pending)
        if (ps.st_dev,ps.st_ino,ps.st_uid)!=owner:raise RuntimeError("pending replaced")
        os.link(pending,path);linked=True
        fs=os.lstat(path)
        if (fs.st_dev,fs.st_ino,fs.st_uid)!=owner or path.read_bytes()!=data:raise RuntimeError("readback")
        validate_report(json.loads(path.read_text()))
        dfd=os.open(parent,os.O_RDONLY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
        fs=os.lstat(path);ps=os.lstat(pending)
        if (fs.st_dev,fs.st_ino,fs.st_uid)!=owner or (ps.st_dev,ps.st_ino,ps.st_uid)!=owner or path.read_bytes()!=data:raise RuntimeError("late replacement")
        unlink_owned(pending,owner)
        dfd=os.open(parent,os.O_RDONLY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
        fs=os.lstat(path)
        if (fs.st_dev,fs.st_ino,fs.st_uid)!=owner:raise RuntimeError("final replaced")
    except Exception:
        if linked:unlink_owned(path,owner)
        raise
    finally:
        os.close(fd);unlink_owned(pending,owner)

def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--regression",action="store_true")
    parser.add_argument("--benchmark",action="store_true")
    args=parser.parse_args()
    if args.regression==args.benchmark:raise RuntimeError("select exactly one mode")
    for path in INPUTS:
        if not path.is_file() or path.is_symlink():raise RuntimeError(f"missing input {path}")
    before={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
    static=command([sys.executable,"-m","tools.r9700.check_a8q4_group_major_activation_static","--source",str(SOURCE),"--assembly",str(ASM)])
    if static.returncode or not static.stdout.startswith("PASS "):raise RuntimeError(static.stdout+static.stderr)
    regression=command([str(BINARY),"--regression"])
    if regression.returncode or regression.stderr or not regression.stdout.startswith("PASS "):raise RuntimeError(regression.stdout+regression.stderr)
    if args.regression:print(regression.stdout,end="");return 0
    if os.path.lexists(OUTPUT):raise FileExistsError(OUTPUT)
    pb=power();measured=command([str(BINARY),"--benchmark"]);pa=power()
    if measured.returncode or measured.stderr:raise RuntimeError(measured.stdout+measured.stderr)
    raw=json.loads(measured.stdout);decision=validate_raw(raw)
    after={str(p.relative_to(ROOT)):sha(p) for p in INPUTS}
    if before!=after:raise RuntimeError("executed input changed")
    output=str(OUTPUT)
    fields=dict(re.findall(r"\b(prefill_vgpr|wave_vgpr|ordinary_quantizer_vgpr|fused_quantizer_vgpr)=(\d+)\b",static.stdout))
    if set(fields)!={"prefill_vgpr","wave_vgpr","ordinary_quantizer_vgpr","fused_quantizer_vgpr"}:raise RuntimeError("static resource parse")
    report={"schema":"ninfer.r9700.a8q4-group-major-activation-gate.v1",
      "command":[sys.executable,"-m","tools.r9700.run_a8q4_group_major_activation_gate","--benchmark"],
      "output":output,"hashes_sha256":before,"static":{"command":static.args,"stdout":static.stdout},
      "resources":{"prefill_vgpr":int(fields["prefill_vgpr"]),"wave_vgpr":int(fields["wave_vgpr"]),"ordinary_quantizer_vgpr":int(fields["ordinary_quantizer_vgpr"]),"fused_quantizer_vgpr":int(fields["fused_quantizer_vgpr"]),"lds_bytes":17152,"occupancy_waves_per_eu":16,"scratch_bytes":0,"spills":0,"prefill_static_iu4_sites":8,"activation_b32_sites":2},
      "bound_regression":{"command":regression.args,"stdout":regression.stdout,"stderr":regression.stderr,"returncode":regression.returncode},
      "power_before":pb,"power_after":pa,"measurement":raw,"decision":decision}
    publish(OUTPUT,report)
    return 0

if __name__=="__main__":raise SystemExit(main())
