#!/usr/bin/env python3
"""Preflight and execute the repaired immutable two-arm layer-3 attention trace."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent; RESULTS=PACKAGE/"results"
sys.path.insert(0,str(PACKAGE)); import analyze
PLAN=analyze.PLAN; BUILD=analyze.BUILD; SOURCE=analyze.SOURCE
POWER=Path(PLAN["hardware"]["power_profile_source"])
EXACT={"LD_PRELOAD","LD_AUDIT","HIP_FORCE_QUEUE_PROFILING","AMD_SERIALIZE_KERNEL","AMD_SERIALIZE_COPY","ROC_SERIALIZE_KERNEL","GPU_DUMP_CODE_OBJECT","NINFER_DFLASH_DECISION_TRACE_OUT","NINFER_QWEN3_PREFILL_P129_TRACE","NINFER_QWEN3_PREFILL_P129_TRACE_OUT","NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE","NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST","NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR","NINFER_QWEN3_LAYER3_ATTENTION_TRACE_MANIFEST","NINFER_QWEN3_LAYER3_ATTENTION_TRACE_SIDECAR","NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST","NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR","NINFER_QWEN3_GDN_STATE_TRACE_MANIFEST","NINFER_QWEN3_GDN_STATE_TRACE_SIDECAR","NINFER_QWEN3_GDN_STATE_TRACE_TEXT_LAYER"}
PREFIX=("ROCPROF","ROCP_","ROCTRACER_","ROCTX_","HSA_TOOLS_","HIP_TRACE_","AQLPROFILE_","ATT_PROFILE")

def fail(m): raise RuntimeError(m)
def injected(): return sorted(k for k in os.environ if k in EXACT or k.startswith(PREFIX) or (k.startswith("NINFER_") and "TRACE" in k))
def power():
    value=POWER.read_text().strip()
    if value!="auto": fail(f"power profile {value!r}, expected auto")
    return value
def write_new(path,text):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644); data=text.encode()
    try:
        off=0
        while off<len(data):
            n=os.write(fd,data[off:])
            if n<=0: fail(f"write failed: {path}")
            off+=n
        os.fsync(fd)
    finally: os.close(fd)
def validate_inputs():
    if injected(): fail("profiling/trace injection environment must be clean")
    if set(PLAN)!={"artifact_type","schema_version","status","source","claim","production_routing_authorized","artifact","history","corpus","hardware","selectors","common","arms","trace","expected_generated_tokens","limitations","failed_attempt1","failed_attempt2","failed_attempt3","results_directory"}: fail("plan schema")
    if PLAN["source"]!={"commit":"617672eb7186f038f26412661d9f586454774450","tree":"9a793c64218c710f7fc1df765edc0d4d81950015"}: fail("plan source")
    if PLAN["selectors"]!={"dflash_small_t_candidate":False,"dflash_mlp_down_t5_candidate":True,"dflash_rmsnorm_rows56_candidate":True,"gdn_verify_wave_qk_candidate":True}: fail("selector plan")
    failed_status={"failed_attempt1":"invalid orchestration after ordinary arm; no functional comparison claim","failed_attempt2":"invalid orchestration after ordinary arm; no functional comparison claim","failed_attempt3":"invalid source trace contract after ordinary arm; no functional comparison claim"}
    for name in ("failed_attempt1","failed_attempt2","failed_attempt3"):
        failed=PLAN[name]; failed_path=PACKAGE/failed["path"]
        if failed.get("status")!=failed_status[name] or analyze.identity(failed_path)!={"path":str(failed_path.resolve()),"bytes":failed["bytes"],"sha256":failed["sha256"]}: fail(f"{name} closure")
    for key in ("artifact","history","corpus"):
        expected=PLAN[key]; actual=analyze.identity(Path(expected["path"])); expected_id={k:expected[k] for k in ("path","bytes","sha256")}
        if actual!=expected_id: fail(f"input identity: {key}")
    history=[int(x) for x in Path(PLAN["history"]["path"]).read_text().split()]
    corpus=[int(x) for x in Path(PLAN["corpus"]["path"]).read_text().split()]
    if len(history)!=129 or history[:128]!=corpus[:128] or history[128]!=24178: fail("history derivation")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=SOURCE,text=True).strip()!=PLAN["source"]["commit"] or subprocess.check_output(["git","rev-parse","HEAD^{tree}"],cwd=SOURCE,text=True).strip()!=PLAN["source"]["tree"] or subprocess.check_output(["git","status","--porcelain"],cwd=SOURCE,text=True): fail("source worktree")
    for group in ("outputs","tools"):
        for name, expected in BUILD[group].items():
            if analyze.identity(Path(expected["path"]))!=expected: fail(f"build identity: {group}/{name}")
    cache=Path(BUILD["outputs"]["cache"]["path"]).read_text(); commands=Path(BUILD["outputs"]["compile_commands"]["path"]).read_text()
    compile_bound={"NINFER_R9700_KV_VALUE_GROUP","NINFER_R9700_Q4_ACTIVATION_BITS","NINFER_R9700_W8_ACTIVATION_BITS","NINFER_R9700_FP8_QK_WMMA","NINFER_R9700_DFLASH_SMALL_T_CANDIDATE","NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE","NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE","NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE"}
    for name,value in BUILD["profile"].items():
        kind="BOOL" if value in ("OFF","ON") else "STRING"
        if f"{name}:{kind}={value}" not in cache or (name in compile_bound and f"-D{name}={value}" not in commands): fail(f"compiled profile: {name}")
    ldd=subprocess.check_output(["ldd",str(analyze.EXE)],env={**os.environ,"LD_LIBRARY_PATH":"/opt/rocm/lib"},text=True)
    if "not found" in ldd or "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7" not in ldd or "libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1" not in ldd: fail("runtime DSO resolution")
    if Path("/sys/class/drm/card2/device/vendor").read_text().strip()!="0x1002" or Path("/sys/class/drm/card2/device/device").read_text().strip()!="0x7551" or Path("/sys/class/drm/card2/device").resolve().name!="0000:13:00.0": fail("hardware identity")
    power()
    tests=[[BUILD["outputs"][k]["path"]] for k in ("runtime_test","decision_test","path_test")]
    tests += [["/usr/bin/python3",str(SOURCE/"tools/bench/test_compare_qwen3_layer_boundary_trace.py")],["/usr/bin/python3",str(SOURCE/"tools/bench/test_analyze_qwen3_layer3_attention_trace.py")],["/usr/bin/python3",str(PACKAGE/"test_analyze.py")]]
    for cmd in tests: subprocess.run(cmd,cwd=SOURCE,check=True)
def close_results():
    paths=sorted(p for p in RESULTS.iterdir() if p.is_file() and p.name!="result.sha256")
    write_new(RESULTS/"result.sha256","".join(f"{analyze.digest(p)}  {p.name}\n" for p in paths))
def run_arm(stem):
    paths={suffix:RESULTS/f"{stem}.{suffix}" for suffix in ("json","stdout","stderr","process.json","trace.json","trace.bin","attention.json","attention.bin")}
    if any(p.exists() or p.is_symlink() for p in paths.values()): fail(f"refusing overwrite: {stem}")
    trace=analyze.trace_environment(stem)
    env={**os.environ,"LD_LIBRARY_PATH":"/opt/rocm/lib",**trace}; cmd=analyze.command(stem,paths["json"]); before=power(); start=time.time_ns(); done=subprocess.run(cmd,cwd=PACKAGE.parent.parent.parent,env=env,capture_output=True,text=True); end=time.time_ns(); after=power()
    write_new(paths["stdout"],done.stdout); write_new(paths["stderr"],done.stderr)
    write_new(paths["process.json"],json.dumps({"command":cmd,"trace_environment":trace,"exit_code":done.returncode,"started_unix_ns":start,"finished_unix_ns":end,"power_before":before,"power_after":after,"injection_environment":[],"stdout":analyze.identity(paths["stdout"]),"stderr":analyze.identity(paths["stderr"])},indent=2)+"\n")
    expected_stdout=f"wrote {paths['json']}\n"
    if done.returncode or any(not paths[k].is_file() for k in ("json","trace.json","trace.bin","attention.json","attention.bin")) or paths["json"].is_symlink() or done.stdout!=expected_stdout: fail(f"arm failed: {stem}")
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args(); validate_inputs()
    if args.preflight_only: print("layer3 attention trace package preflight PASS"); return 0
    if RESULTS.exists() or RESULTS.is_symlink(): fail("results path must be fresh")
    RESULTS.mkdir()
    try:
        for stem in analyze.ROLES: run_arm(stem)
        comp=RESULTS/"attention-comparison.json"; cmd=["/usr/bin/python3",str(SOURCE/"tools/bench/analyze_qwen3_layer3_attention_trace.py"),"--ordinary",str(RESULTS/"target-ordinary.attention.json"),"--dflash",str(RESULTS/"target-dflash.attention.json"),"--output",str(comp)]
        done=subprocess.run(cmd,capture_output=True,text=True); write_new(RESULTS/"attention-comparator.stdout",done.stdout); write_new(RESULTS/"attention-comparator.stderr",done.stderr); write_new(RESULTS/"attention-comparator.process.json",json.dumps({"command":cmd,"exit_code":done.returncode},indent=2)+"\n")
        if done.returncode or not comp.is_file(): fail("attention comparator failed")
        analyze.validate_all(RESULTS/"summary.json"); close_results(); print("layer3 attention trace capture PASS"); return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS/"result.sha256").exists(): close_results()
        raise
if __name__=="__main__": raise SystemExit(main())
