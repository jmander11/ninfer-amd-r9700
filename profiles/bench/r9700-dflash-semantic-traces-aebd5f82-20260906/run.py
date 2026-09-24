#!/usr/bin/env python3
"""Execute the bounded aebd5f82 functional semantic-trace package."""

from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent; ROOT=Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
RESULTS=PACKAGE/"results"; POWER=Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
sys.path.insert(0,str(PACKAGE)); import analyze  # noqa: E402
EXACT={"LD_PRELOAD","LD_AUDIT","HIP_FORCE_QUEUE_PROFILING","AMD_SERIALIZE_KERNEL","AMD_SERIALIZE_COPY","GPU_DUMP_CODE_OBJECT","NINFER_DFLASH_DECISION_TRACE_OUT","NINFER_QWEN3_PREFILL_P129_TRACE","NINFER_QWEN3_PREFILL_P129_TRACE_OUT"}
PREFIX=("ROCPROF","ROCP_","ROCTRACER_","ROCTX_","HSA_TOOLS_","HIP_TRACE_","AQLPROFILE_","ATT_PROFILE")

def fail(message): raise RuntimeError(message)
def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()
def ident(path):
    p=Path(path).resolve(strict=True)
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}
def load(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict): fail(f"not object: {path}")
    return value
def power():
    value=POWER.read_text().strip()
    if value!="auto": fail(f"power profile is {value!r}, expected auto")
    return value
def injected(): return sorted(k for k in os.environ if k in EXACT or k.startswith(PREFIX))
def require_fresh_results(path):
    if Path(path).exists() or Path(path).is_symlink(): fail("results path already exists")

def validate_inputs():
    if injected(): fail("profiling/trace injection environment must be clean before preflight")
    plan=load(PACKAGE/"plan.json"); build=load(PACKAGE/"build-provenance.json")
    if plan.get("source")!={"commit":"aebd5f82b91a756429651b2795e3ad24c1058cb3","tree":"cd5a2e3e872759d88c094f56e0f7e6416cf1340e"} or plan.get("production_routing_authorized") is not False or plan.get("decision",{}).get("timing_evidence_eligible") is not False: fail("plan authority differs")
    if subprocess.check_output(["git","rev-parse","aebd5f82^{commit}"],cwd=ROOT,text=True).strip()!=plan["source"]["commit"] or subprocess.check_output(["git","rev-parse","aebd5f82^{tree}"],cwd=ROOT,text=True).strip()!=plan["source"]["tree"]: fail("source commit/tree differs")
    changed=subprocess.run(["git","diff","--quiet",plan["source"]["commit"],"--","src/targets/qwen3/impl/runtime/decision_trace.h","src/targets/qwen3/impl/runtime/prefill_tail_trace.h","src/targets/qwen3/impl/runtime/prefill_tail_trace_path.h","src/targets/qwen3/impl/runtime/program_impl.h","src/targets/qwen3/impl/runtime/text_context_impl.h","tools/bench/compare_prefill_p129_tail_trace.py","bench/targets/qwen3_8_27b/ninfer_bench.cpp","bench/targets/qwen3_8_27b/ninfer_bench_support.cpp"],cwd=ROOT)
    if changed.returncode: fail("relevant source differs from aebd5f82")
    for key in ("benchmark","cmake_cache","compile_commands"):
        if ident(build["outputs"][key]["path"])!=build["outputs"][key]: fail(f"build {key} identity differs")
    for key in ("cxx","hip_compiler","hipcc","python","hipblaslt","amdhip"):
        if ident(build["toolchain_runtime"][key]["path"])!=build["toolchain_runtime"][key]: fail(f"toolchain/runtime {key} identity differs")
    for key in ("artifact",):
        expected=plan[key]; actual=ident(expected["path"])
        if actual["bytes"]!=expected["bytes"] or actual["sha256"]!=expected["sha256"]: fail(f"{key} identity differs")
    for key in ("source_corpus","p129_fixture","seed_report"):
        expected=plan["history"][key]; actual=ident(expected["path"])
        if actual!=expected: fail(f"history {key} identity differs")
    source=[int(x) for x in Path(plan["history"]["source_corpus"]["path"]).read_text().split()]
    history=[int(x) for x in Path(plan["history"]["p129_fixture"]["path"]).read_text().split()]
    if len(history)!=129 or history[:128]!=source[:128] or history[128]!=24178: fail("P129 history/seed differs")
    seed=load(plan["history"]["seed_report"]["path"])
    tests=seed.get("tests")
    whole=next((test for test in tests if test.get("label")=="whole-pp128+tg64"),None) if isinstance(tests,list) else None
    lanes=whole.get("reps",[{}])[0].get("generated_token_ids_by_lane") if isinstance(whole,dict) and whole.get("reps") else None
    if not isinstance(lanes,list) or len(lanes)!=1 or not lanes[0] or lanes[0][0]!=24178: fail("seed report does not prove seed token 24178")
    ldd=subprocess.check_output(["ldd",str(analyze.EXE)],env={**os.environ,"LD_LIBRARY_PATH":"/opt/rocm/lib"},text=True)
    if "not found" in ldd or "libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1" not in ldd or "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7" not in ldd: fail("runtime DSO resolution differs")
    if Path("/sys/class/drm/card2/device/vendor").read_text().strip()!="0x1002" or Path("/sys/class/drm/card2/device/device").read_text().strip()!="0x7551" or Path("/sys/class/drm/card2/device").resolve().name!="0000:13:00.0": fail("R9700 PCI identity differs")
    power()

def close_results():
    files=sorted(p for p in RESULTS.rglob("*") if p.is_file() and p.name!="result.sha256")
    lines=[f"{sha(p)}  {p}\n" for p in files]
    (RESULTS/"result.sha256").write_text("".join(lines))

def run_one(stem,kind,trace_name,trace_environment):
    report=RESULTS/f"{stem}.json"; trace=RESULTS/trace_name
    command=analyze.command(kind,report); stdout=RESULTS/f"{stem}.stdout"; stderr=RESULTS/f"{stem}.stderr"; record=RESULTS/f"{stem}.process.json"
    for path in (report,trace,stdout,stderr,record):
        if path.exists() or path.is_symlink(): fail(f"refusing overwrite: {path}")
    env=os.environ.copy(); env["LD_LIBRARY_PATH"]="/opt/rocm/lib"; env.update(trace_environment(trace))
    before=power(); started=time.time_ns(); proc=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True); finished=time.time_ns(); after=power()
    stdout.write_text(proc.stdout); stderr.write_text(proc.stderr)
    record.write_text(json.dumps({"command":command,"trace_environment":trace_environment(trace),"exit_code":proc.returncode,"started_unix_ns":started,"finished_unix_ns":finished,"power_before":before,"power_after":after,"instrumentation_environment_present":[],"stdout":ident(stdout),"stderr":ident(stderr)},indent=2)+"\n")
    if proc.returncode or not report.is_file() or not trace.is_file(): close_results(); fail(f"{stem} failed; partial evidence retained")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args()
    validate_inputs()
    if args.preflight_only: print("semantic trace package preflight PASS"); return 0
    require_fresh_results(RESULTS)
    RESULTS.mkdir()
    try:
        run_one("decision-ordinary","decision-ordinary","decision-ordinary.trace.json",lambda p:{"NINFER_DFLASH_DECISION_TRACE_OUT":str(p)})
        run_one("decision-dflash","decision-dflash","decision-dflash.trace.json",lambda p:{"NINFER_DFLASH_DECISION_TRACE_OUT":str(p)})
        run_one("tail-fresh","tail-fresh","tail-fresh.trace.json",lambda p:{"NINFER_QWEN3_PREFILL_P129_TRACE":"1","NINFER_QWEN3_PREFILL_P129_TRACE_OUT":str(p)})
        run_one("tail-append","tail-append","tail-append.trace.json",lambda p:{"NINFER_QWEN3_PREFILL_P129_TRACE":"1","NINFER_QWEN3_PREFILL_P129_TRACE_OUT":str(p)})
        comparator=ROOT/"tools/bench/compare_prefill_p129_tail_trace.py"; out=RESULTS/"tail-comparison.json"
        command=["/usr/bin/python3",str(comparator),"--fresh",str(RESULTS/"tail-fresh.trace.json"),"--append",str(RESULTS/"tail-append.trace.json"),"--out",str(out)]
        proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (RESULTS/"tail-comparator.stdout").write_text(proc.stdout); (RESULTS/"tail-comparator.stderr").write_text(proc.stderr)
        (RESULTS/"tail-comparator.process.json").write_text(json.dumps({"command":command,"exit_code":proc.returncode,"stdout":ident(RESULTS/"tail-comparator.stdout"),"stderr":ident(RESULTS/"tail-comparator.stderr")},indent=2)+"\n")
        if proc.returncode or not out.is_file(): close_results(); fail("tail comparator failed")
        old=sys.argv; sys.argv=["analyze.py","--summary",str(RESULTS/"summary.json")]
        try: analyze.main()
        finally: sys.argv=old
        close_results(); print("semantic trace capture and analysis PASS"); return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS/"result.sha256").exists(): close_results()
        raise

if __name__=="__main__": raise SystemExit(main())
