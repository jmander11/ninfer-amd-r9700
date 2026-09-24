#!/usr/bin/env python3
"""Fail-closed validation and summary for the repaired two-arm layer-3 trace."""

from __future__ import annotations
import hashlib, importlib.util, json, math, os, sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
RESULTS = PACKAGE / "results"
PLAN = json.loads((PACKAGE / "plan.json").read_text())
BUILD = json.loads((PACKAGE / "build-provenance.json").read_text())
SOURCE = Path(BUILD["source"]["worktree"])
EXE = Path(BUILD["outputs"]["benchmark"]["path"])
ARTIFACT = Path(PLAN["artifact"]["path"])
HISTORY = Path(PLAN["history"]["path"])
ROLES = {"target-ordinary":"target-ordinary-frontier130", "target-dflash":"target-dflash-frontier130-column0"}

def fail(message): raise RuntimeError(message)
def pairs(items):
    out = {}
    for key, value in items:
        if key in out: fail(f"duplicate JSON key: {key}")
        out[key] = value
    return out
def load(path):
    value=json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:fail(f"nonfinite JSON: {x}"))
    if not isinstance(value,dict): fail(f"not an object: {path}")
    return value
def digest(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()
def identity(path):
    path=Path(path).resolve(strict=True); return {"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path)}
def validate_speculative(value, expected):
    keys={"enabled","draft_window","rounds","drafted_tokens","accepted_tokens","fallback_steps","acceptance_rate","acceptance_length","accepted_per_position"}
    if not isinstance(value,dict) or set(value)!=keys or value.get("enabled") is not expected: fail("speculative schema")
    if not expected:
        exact={"enabled":False,"draft_window":0,"rounds":0,"drafted_tokens":0,"accepted_tokens":0,"fallback_steps":0,"acceptance_rate":None,"acceptance_length":None,"accepted_per_position":[]}
        if value!=exact: fail("ordinary speculative accounting")
        return
    for key in ("draft_window","rounds","drafted_tokens","accepted_tokens","fallback_steps"):
        if isinstance(value.get(key),bool) or not isinstance(value.get(key),int) or value[key]<0: fail(f"speculative integer: {key}")
    if value["draft_window"]!=4 or value["accepted_tokens"]>value["drafted_tokens"] or value["drafted_tokens"]>4*value["rounds"]: fail("speculative count invariant")
    positions=value["accepted_per_position"]
    if not isinstance(positions,list) or len(positions)!=4 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in positions) or sum(positions)!=value["accepted_tokens"]: fail("speculative positions")
    rate=None if value["drafted_tokens"]==0 else value["accepted_tokens"]/value["drafted_tokens"]
    length=None if value["rounds"]==0 else (value["rounds"]+value["accepted_tokens"])/value["rounds"]
    for key,expected_value in (("acceptance_rate",rate),("acceptance_length",length)):
        actual=value[key]
        if expected_value is None:
            if actual is not None: fail(f"speculative {key}")
        elif isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,expected_value,rel_tol=1e-12,abs_tol=1e-12): fail(f"speculative {key}")
def command(stem, report):
    cmd=[str(EXE),"--weights",str(ARTIFACT),"--corpus",str(HISTORY),"--device","0","--concurrency","1","--whole-pg","129,1","--prefill-chunk","4096","--kv-capacity","workload","--retain-token-ids","--output","json","--output-file",str(report),"-r","1","--warmup","0","--no-device-graph"]
    if stem=="target-dflash": cmd += ["--spec","dflash","--draft-tokens","4","--dflash-verify-width","5","--lm-head-draft"]
    else: cmd += ["--draft-tokens","0"]
    return cmd
def trace_environment(stem):
    return {"NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE":ROLES[stem],"NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST":str(RESULTS/f"{stem}.trace.json"),"NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR":str(RESULTS/f"{stem}.trace.bin"),"NINFER_QWEN3_LAYER3_ATTENTION_TRACE_MANIFEST":str(RESULTS/f"{stem}.attention.json"),"NINFER_QWEN3_LAYER3_ATTENTION_TRACE_SIDECAR":str(RESULTS/f"{stem}.attention.bin"),"NINFER_QWEN3_LAYER3_TRACE_SOURCE_COMMIT":PLAN["source"]["commit"],"NINFER_QWEN3_LAYER3_TRACE_SOURCE_TREE":PLAN["source"]["tree"],"NINFER_QWEN3_LAYER3_TRACE_EXECUTABLE_SHA256":BUILD["outputs"]["benchmark"]["sha256"],"NINFER_QWEN3_LAYER3_TRACE_ARTIFACT_SHA256":PLAN["artifact"]["sha256"],"NINFER_QWEN3_LAYER3_TRACE_HISTORY_SHA256":PLAN["history"]["sha256"],"NINFER_QWEN3_LAYER3_TRACE_CORPUS_SHA256":PLAN["corpus"]["sha256"],"NINFER_QWEN3_LAYER3_TRACE_POWER_SOURCE":PLAN["hardware"]["power_profile_source"]}
def layer_expected(stem):
    return (ROLES[stem],1 if stem=="target-ordinary" else 5,0,130,129)
def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

LAYER=_module("frozen_layer_compare",SOURCE/"tools/bench/compare_qwen3_layer_boundary_trace.py")
ATTN=_module("frozen_attention_compare",SOURCE/"tools/bench/analyze_qwen3_layer3_attention_trace.py")

def validate_report(stem):
    path=RESULTS/f"{stem}.json"; value=load(path)
    if set(value)!={"schema_version","artifact_type","tool","command","environment","artifact","load","memory","config","tests"}: fail(f"report schema: {stem}")
    if value["schema_version"]!=20 or value["artifact_type"]!="ninfer_bench_report" or value["tool"]!="ninfer_bench" or value["command"]!=" ".join(command(stem,path)): fail(f"report identity: {stem}")
    if value["environment"]!={"gpu_name":"AMD Radeon AI PRO R9700","architecture_name":"gfx1201","hip_runtime_version":"7.15.26333","hip_driver_version":"7.15.26333","device_id":0}: fail(f"hardware: {stem}")
    if value["artifact"]!={"path":str(ARTIFACT),"file_size_bytes":PLAN["artifact"]["bytes"]}: fail(f"artifact: {stem}")
    config=value["config"]; spec=stem=="target-dflash"
    load_record=value.get("load",{}); expected_load=(22287117113,22274271744,839) if spec else (20720607545,20707762176,771)
    if set(load_record)!={"target","weights_id","load_seconds","upload_seconds","artifact_bytes_read","host_to_device_bytes","peak_staging_bytes","tensor_count","resource_count"} or load_record.get("target")!="qwen3_8_27b_r9700" or load_record.get("weights_id")!=PLAN["artifact"]["weights_id"] or tuple(load_record.get(k) for k in ("artifact_bytes_read","host_to_device_bytes","tensor_count"))!=expected_load or load_record.get("peak_staging_bytes")!=268435456 or load_record.get("resource_count")!=6: fail(f"load record: {stem}")
    if any(isinstance(load_record.get(k),bool) or not isinstance(load_record.get(k),(int,float)) or not math.isfinite(load_record[k]) or load_record[k]<=0 for k in ("load_seconds","upload_seconds")): fail(f"load timing: {stem}")
    config_keys={"max_context","prefill_chunk","kv_cache_format","kv_value_group","kv_plane_layouts","q4_activation_bits","q4_prefill_cta_profile","dflash_small_t_candidate","dflash_mlp_down_t5_candidate","dflash_rmsnorm_rows56_candidate","w8_activation_bits","fp8_qk_wmma_enabled","fp8_qk_wmma_profile","fp8_qk_wmma_t1_min_context","fp8_qk_wmma_t2_min_context","xattention_qualification","concurrency","pending_timeout_ms","pending_deadline","spec","draft_tokens","speculative_execution","dflash_verify_width_requested","dflash_verify_width","proposal_head","use_device_graph","retain_token_ids","isolate_prompt_decode","decode_path","decode_graph_prime","repetitions","warmup","corpus_path","corpus_tokens"}
    if not isinstance(config,dict) or set(config)!=config_keys: fail(f"config schema: {stem}")
    exact={"max_context":140 if spec else 130,"prefill_chunk":4096,"kv_cache_format":"fp8-k-int4-v","kv_value_group":16,"kv_plane_layouts":{"key":"token-fastest-head-major","value":"feature-fastest-page-major","value_scale":"feature-fastest-page-major"},"q4_activation_bits":8,"q4_prefill_cta_profile":"m64n128-pingpong-n16-k16-scalar-base-production","w8_activation_bits":8,"fp8_qk_wmma_enabled":True,"fp8_qk_wmma_profile":"t1-ge64-t2-ge320-t3plus-stream-v1","fp8_qk_wmma_t1_min_context":64,"fp8_qk_wmma_t2_min_context":320,"xattention_qualification":False,"concurrency":1,"pending_timeout_ms":4294967295,"pending_deadline":"unbounded","spec":"dflash" if spec else "none","draft_tokens":4 if spec else 0,"speculative_execution":spec,"dflash_verify_width_requested":5 if spec else 0,"dflash_verify_width":5 if spec else 0,"proposal_head":"optimized" if spec else "full","use_device_graph":False,"retain_token_ids":True,"isolate_prompt_decode":False,"decode_path":"dflash_eager" if spec else "eager","decode_graph_prime":{"primed":False,"output_tokens":0},"repetitions":1,"warmup":0,"corpus_path":str(HISTORY),"corpus_tokens":129,"dflash_small_t_candidate":False,"dflash_mlp_down_t5_candidate":True,"dflash_rmsnorm_rows56_candidate":True}
    for key, expected in exact.items():
        if config.get(key)!=expected: fail(f"config {key}: {stem}")
    for key in ("planned_slack_bytes","available_after_startup_bytes"):
        amount=value.get("memory",{}).get(key)
        if isinstance(amount,bool) or not isinstance(amount,int) or amount<1674508: fail(f"memory {key}: {stem}")
    tests=value["tests"]
    if not isinstance(tests,list) or len(tests)!=1: fail(f"tests: {stem}")
    test=tests[0]
    test_keys={"label","kind","n_prompt","n_gen","requested_output_tokens","prefill_tok_s_mean","prefill_tok_s_stddev","decode_output_tok_s_mean","decode_output_tok_s_stddev","decode_engine_tok_s_mean","decode_engine_tok_s_stddev","whole_output_tok_s_mean","whole_output_tok_s_stddev","prepare_seconds_mean","prepare_seconds_stddev","prefill_seconds_mean","prefill_seconds_stddev","decode_seconds_mean","decode_seconds_stddev","total_seconds_mean","total_seconds_stddev","workspace_peak_bytes","workspace_allocator_peak_bytes","speculative","reps"}
    if not isinstance(test,dict) or set(test)!=test_keys: fail(f"test schema: {stem}")
    if test.get("label")!="whole-pp129+tg1" or test.get("kind")!="whole" or test.get("n_prompt")!=129 or test.get("n_gen")!=1 or test.get("requested_output_tokens")!=2: fail(f"test geometry: {stem}")
    reps=test.get("reps"); rep=reps[0] if isinstance(reps,list) and len(reps)==1 else None
    if not isinstance(rep,dict) or set(rep)!={"generated_output_tokens","decode_output_tokens","decode_engine_tokens","generated_token_ids_by_lane","timings","speculative"} or rep.get("generated_output_tokens")!=2 or rep.get("decode_output_tokens")!=1 or rep.get("decode_engine_tokens")!=1 or rep.get("generated_token_ids_by_lane")!=[[96558,96917]]: fail(f"tokens: {stem}")
    timings=rep.get("timings")
    if not isinstance(timings,dict) or set(timings)!={"prepare_seconds","vision_seconds","prefill_seconds","decode_seconds","total_seconds"} or any(isinstance(timings.get(k),bool) or not isinstance(timings.get(k),(int,float)) or not math.isfinite(timings[k]) or timings[k]<0 for k in timings): fail(f"timing encoding: {stem}")
    speculative=rep.get("speculative",{})
    if test.get("speculative")!=speculative: fail(f"aggregate speculative accounting: {stem}")
    validate_speculative(speculative,spec)
    if spec and speculative["rounds"]+speculative["accepted_tokens"]+speculative["fallback_steps"]!=1: fail("decode engine/spec accounting")
    return identity(path)

def validate_process_header(stem,p):
    process_keys={"command","trace_environment","exit_code","started_unix_ns","finished_unix_ns","power_before","power_after","injection_environment","stdout","stderr"}
    if set(p)!=process_keys or p.get("command")!=command(stem,RESULTS/f"{stem}.json") or p.get("trace_environment")!=trace_environment(stem) or p.get("exit_code")!=0 or p.get("power_before")!="auto" or p.get("power_after")!="auto" or p.get("injection_environment")!=[]: fail(f"process: {stem}")
    start=p.get("started_unix_ns"); finish=p.get("finished_unix_ns")
    if isinstance(start,bool) or not isinstance(start,int) or start<=0 or isinstance(finish,bool) or not isinstance(finish,int) or finish<=start: fail(f"process timestamps: {stem}")
def validate_process(stem):
    p=load(RESULTS/f"{stem}.process.json")
    validate_process_header(stem,p)
    for kind in ("stdout","stderr"):
        if p.get(kind)!=identity(RESULTS/f"{stem}.{kind}"): fail(f"process {kind}: {stem}")
    expected_stdout=f"wrote {RESULTS/f'{stem}.json'}\n"
    if (RESULTS/f"{stem}.stdout").read_text()!=expected_stdout: fail(f"stdout: {stem}")
    stderr=(RESULTS/f"{stem}.stderr").read_text().splitlines(); context=140 if stem=="target-dflash" else 130
    expected=[f"[ninfer_bench] loading {ARTIFACT} (max_context={context}, concurrency=1, kv_format=fp8-k-int4-v)","[ninfer_bench] test 1/1 whole-pp129+tg1: warmup=0 reps=1"]
    if stderr!=expected: fail(f"stderr: {stem}")
    return identity(RESULTS/f"{stem}.process.json")

def validate_all(output):
    reports={s:validate_report(s) for s in ROLES}; processes={s:validate_process(s) for s in ROLES}
    layer={}; attention={}
    for stem in ROLES:
        manifest=RESULTS/f"{stem}.trace.json"; manifest_value,_=LAYER.load(manifest,layer_expected(stem))
        if manifest_value.get("token")!=96558: fail(f"layer trace token differs: {stem}")
        layer[stem]={"manifest":identity(manifest),"sidecar":identity(RESULTS/f"{stem}.trace.bin")}
        kind="ordinary" if stem=="target-ordinary" else "dflash"
        ATTN._load(RESULTS/f"{stem}.attention.json",kind)
        attention[stem]={"manifest":identity(RESULTS/f"{stem}.attention.json"),"sidecar":identity(RESULTS/f"{stem}.attention.bin")}
    comparison=load(RESULTS/"attention-comparison.json")
    if comparison!=ATTN.analyze(RESULTS/"target-ordinary.attention.json",RESULTS/"target-dflash.attention.json"): fail("comparison differs from source analyzer")
    comparator=load(RESULTS/"attention-comparator.process.json")
    expected_comparator=["/usr/bin/python3",str(SOURCE/"tools/bench/analyze_qwen3_layer3_attention_trace.py"),"--ordinary",str(RESULTS/"target-ordinary.attention.json"),"--dflash",str(RESULTS/"target-dflash.attention.json"),"--output",str(RESULTS/"attention-comparison.json")]
    if comparator!={"command":expected_comparator,"exit_code":0} or (RESULTS/"attention-comparator.stdout").read_bytes() or (RESULTS/"attention-comparator.stderr").read_bytes(): fail("comparator process differs")
    summary={"artifact_type":"ninfer_qwen3_layer3_attention_trace_evidence","schema_version":1,"status":"valid_functional_localization","timing_evidence_eligible":False,"production_routing_authorized":False,"source":PLAN["source"],"selectors":PLAN["selectors"],"plan":identity(PACKAGE/"plan.json"),"build_provenance":identity(PACKAGE/"build-provenance.json"),"raw_capture_closure":identity(RESULTS/"capture-result.sha256"),"offline_recovery":identity(RESULTS/"capture-attempt.json"),"reports":reports,"processes":processes,"layer_traces":layer,"attention_traces":attention,"attention_comparison":identity(RESULTS/"attention-comparison.json"),"attention_comparator":{"process":identity(RESULTS/"attention-comparator.process.json"),"stdout":identity(RESULTS/"attention-comparator.stdout"),"stderr":identity(RESULTS/"attention-comparator.stderr")},"classification":comparison["classification"],"limitations":PLAN["limitations"]}
    data=json.dumps(summary,indent=2,sort_keys=True)+"\n"
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644); encoded=data.encode()
    try:
        offset=0
        while offset<len(encoded):
            written=os.write(fd,encoded[offset:])
            if written<=0: fail("summary write made no progress")
            offset+=written
        os.fsync(fd)
    finally: os.close(fd)

if __name__=="__main__": validate_all(Path(sys.argv[1]) if len(sys.argv)>1 else RESULTS/"summary.json")
