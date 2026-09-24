#!/usr/bin/env python3
"""Validate and summarize the bounded aebd5f82 semantic traces."""

from __future__ import annotations

import argparse, hashlib, json, math, shlex, struct, sys
from pathlib import Path

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-dflash-semantic-traces-aebd5f82-20260906"
RESULTS = PACKAGE / "results"
EXE = ROOT / "build-r9700-semantic-traces-aebd5f82-final-20260906/bench/ninfer_bench"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
SOURCE_CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
HISTORY = ROOT / "profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids"
EXPECTED_EXE = "49eae026ba04a7d7bc7a0bed6a137e94c4bed34439d74304a9eb344679f8473b"
EXPECTED_ARTIFACT = "d8fc77c36cf17c92e96d67b9a6b5a1826a1ade4f59d59c003b2368fe981fc512"
CAPTURE_ANALYZER_SHA = "f8c17bf643aad048455ef08b86d90a64029d819ffe67c61de742270064c53e3c"
CAPTURE_PREPARED_SHA = "c45e9a346001c0824182508561aa4b7b2ed4dd9cd9a0e6b0cae5996799455a6c"
CAPTURE_RESULT_SHA = "d517d347a5bec7721c8fb043bd4937b266879779526404e59ec2330ce2218bfe"
DECISION_ENV = {"NINFER_DFLASH_DECISION_TRACE_OUT": None}
TAIL_ENV = {"NINFER_QWEN3_PREFILL_P129_TRACE": "1",
            "NINFER_QWEN3_PREFILL_P129_TRACE_OUT": None}
sys.path.insert(0, str(ROOT))
from tools.bench.compare_prefill_p129_tail_trace import compare as compare_tail  # noqa: E402

def fail(message: str) -> None: raise RuntimeError(message)

def pairs(items):
    out = {}
    for key, value in items:
        if key in out: fail(f"duplicate JSON key: {key}")
        out[key] = value
    return out

def load(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict): fail(f"not a JSON object: {path}")
    return value

def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}

def integer(value, label: str, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low: fail(f"invalid {label}")
    return value

def token(value, label: str) -> int:
    checked=integer(value,label)
    if checked >= 248077: fail(f"invalid {label}")
    return checked

def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"invalid {label}")
    return float(value)

def bf16(bits: int) -> float: return struct.unpack("<f", struct.pack("<I", bits << 16))[0]

def decision(value: object, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"top1_token", "top1_bf16_bits", "top1_logit", "top2_token", "top2_bf16_bits", "top2_logit", "margin"}: fail(f"invalid {label}")
    t1, t2 = integer(value["top1_token"], label), integer(value["top2_token"], label)
    b1, b2 = integer(value["top1_bf16_bits"], label), integer(value["top2_bf16_bits"], label)
    if b1 > 65535 or b2 > 65535 or t1 >= 248077 or t2 >= 248077 or t1 == t2: fail(f"invalid {label} domain")
    v1, v2 = finite(value["top1_logit"], label), finite(value["top2_logit"], label)
    margin = finite(value["margin"], label)
    if v1 != bf16(b1) or v2 != bf16(b2) or v1 < v2 or not math.isclose(margin, v1-v2, abs_tol=1e-7): fail(f"inconsistent {label}")
    return {"top1_token": t1, "top1_bits": b1, "top2_token": t2, "top2_bits": b2, "margin": margin}

def command(kind: str, report: Path) -> list[str]:
    corpus = HISTORY if kind != "tail-append" else SOURCE_CORPUS
    cmd = [str(EXE), "--weights", str(ARTIFACT), "--corpus", str(corpus), "--device", "0", "--concurrency", "1"]
    if kind in ("decision-ordinary", "decision-dflash"): cmd += ["--whole-pg", "129,27"]
    elif kind == "tail-fresh": cmd += ["--whole-pg", "129,1"]
    else: cmd += ["-pg", "128,1"]
    cmd += ["--prefill-chunk", "4096", "--kv-capacity", "workload"]
    if kind == "decision-dflash": cmd += ["--spec", "dflash", "--draft-tokens", "4", "--dflash-verify-width", "5", "--lm-head-draft"]
    else: cmd += ["--draft-tokens", "0"]
    cmd += ["--retain-token-ids"]
    if kind == "tail-append": cmd += ["--isolate-prompt-decode"]
    cmd += ["--no-device-graph", "--output", "json", "--output-file", str(report), "-r", "1", "--warmup", "0"]
    return cmd

def validate_plan() -> None:
    plan = load(PACKAGE / "plan.json")
    exact = {
        "artifact_type": "ninfer_r9700_dflash_semantic_traces_plan",
        "schema_version": 1,
        "status": "prepared_cpu_only_no_gpu_execution",
        "claim": "functional-only decision and P129 Text-prefill boundary diagnostics; all timings are ineligible",
        "production_routing_authorized": False,
        "results_directory": "results",
    }
    if any(plan.get(key) != value for key, value in exact.items()): fail("plan authority differs")
    if plan.get("source") != {"commit":"aebd5f82b91a756429651b2795e3ad24c1058cb3","tree":"cd5a2e3e872759d88c094f56e0f7e6416cf1340e"}: fail("plan source differs")
    if plan.get("common") != {"concurrency":1,"maximum_product_concurrency":4,"prefill_chunk":4096,"kv_capacity":"workload","repetitions":1,"warmup":0,"device_graph":False,"retain_token_ids":True}: fail("plan workload differs")
    if plan.get("decision") != {"classification_only":True,"timing_evidence_eligible":False,"ordinary_and_dflash_history_identical":True,"tail_comparison_precedes_any_layer_instrumentation":True,"no_production_or_performance_claim":True}: fail("plan decision differs")

def validate_process(stem: str, expected_command: list[str]) -> dict:
    record = load(RESULTS / f"{stem}.process.json")
    if record.get("command") != expected_command or record.get("exit_code") != 0 or record.get("power_before") != "auto" or record.get("power_after") != "auto" or record.get("instrumentation_environment_present") != []: fail(f"process identity differs: {stem}")
    if record.get("stdout") != identity(RESULTS / f"{stem}.stdout") or record.get("stderr") != identity(RESULTS / f"{stem}.stderr"): fail(f"process streams differ: {stem}")
    validate_benchmark_stderr(stem, RESULTS / f"{stem}.stderr")
    trace = RESULTS / f"{stem}.trace.json"
    expected_env = ({"NINFER_DFLASH_DECISION_TRACE_OUT":str(trace)} if stem.startswith("decision-") else
                    {"NINFER_QWEN3_PREFILL_P129_TRACE":"1",
                     "NINFER_QWEN3_PREFILL_P129_TRACE_OUT":str(trace)})
    if record.get("trace_environment") != expected_env: fail(f"trace environment differs: {stem}")
    if integer(record.get("finished_unix_ns"), stem) <= integer(record.get("started_unix_ns"), stem): fail(f"invalid process interval: {stem}")
    return identity(RESULTS / f"{stem}.process.json")

def validate_benchmark_stderr(stem: str, path: Path) -> None:
    contexts={"decision-ordinary":156,"decision-dflash":166,"tail-fresh":130,"tail-append":130}
    labels={"decision-ordinary":"whole-pp129+tg27","decision-dflash":"whole-pp129+tg27",
            "tail-fresh":"whole-pp129+tg1","tail-append":"pp128+tg1"}
    if stem not in contexts: fail(f"unknown benchmark stderr role: {stem}")
    expected=(f"[ninfer_bench] loading {ARTIFACT} (max_context={contexts[stem]}, concurrency=1, kv_format=fp8-k-int4-v)\n"
              f"[ninfer_bench] test 1/1 {labels[stem]}: warmup=0 reps=1\n")
    if path.read_text() != expected: fail(f"benchmark stderr differs: {stem}")

def validate_analysis_repair() -> dict:
    repair=load(PACKAGE/"analysis-repair.json")
    if repair.get("artifact_type") != "ninfer_r9700_dflash_semantic_trace_analysis_repair" or repair.get("schema_version") != 1 or repair.get("status") != "analysis_only_no_gpu_rerun" or repair.get("cause") != "capture analyzer rejected exact normal two-line ninfer_bench informational stderr and treated finite decimal speculative-rate serialization as exact rational equality" or repair.get("capture_analyzer_sha256") != CAPTURE_ANALYZER_SHA or repair.get("capture_prepared") != identity(PACKAGE/"prepared.sha256") or repair.get("capture_result") != identity(RESULTS/"result.sha256") or repair.get("repaired_analyzer") != identity(Path(__file__)) or repair.get("raw_capture_modified") is not False or repair.get("gpu_rerun_permitted") is not False: fail("analysis repair provenance differs")
    if repair["capture_prepared"]["sha256"] != CAPTURE_PREPARED_SHA or repair["capture_result"]["sha256"] != CAPTURE_RESULT_SHA: fail("capture closure identity differs")
    return identity(PACKAGE/"analysis-repair.json")

def validate_report(stem: str, kind: str) -> tuple[dict, list[int], dict]:
    path = RESULTS / f"{stem}.json"; report = load(path); expected = command(kind, path)
    validate_process(stem, expected)
    dflash = kind == "decision-dflash"; append = kind == "tail-append"
    prompt, gen = ((128,1) if append else (129, 27 if kind.startswith("decision") else 1))
    label = "pp128+tg1" if append else f"whole-pp129+tg{gen}"
    if report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or report.get("tool") != "ninfer_bench" or shlex.split(report.get("command", "")) != expected or report.get("artifact") != {"path": str(ARTIFACT), "file_size_bytes": 22763026944} or report.get("load", {}).get("weights_id") != "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval": fail(f"report provenance differs: {stem}")
    env = report.get("environment", {})
    if env.get("gpu_name") != "AMD Radeon AI PRO R9700" or env.get("architecture_name") != "gfx1201" or env.get("device_id") != 0 or not env.get("hip_runtime_version") or not env.get("hip_driver_version"): fail(f"hardware differs: {stem}")
    cfg = report.get("config", {})
    expected_corpus = SOURCE_CORPUS if append else HISTORY
    if cfg.get("prefill_chunk") != 4096 or cfg.get("kv_cache_format") != "fp8-k-int4-v" or cfg.get("kv_value_group") != 16 or cfg.get("q4_activation_bits") != 8 or cfg.get("q4_prefill_cta_profile") != "m64n128-pingpong-n16-k16-scalar-base-production" or cfg.get("dflash_small_t_candidate") is not False or cfg.get("dflash_mlp_down_t5_candidate") is not False or cfg.get("w8_activation_bits") != 8 or cfg.get("fp8_qk_wmma_enabled") is not True or cfg.get("concurrency") != 1 or cfg.get("spec") != ("dflash" if dflash else "none") or cfg.get("draft_tokens") != (4 if dflash else 0) or cfg.get("speculative_execution") is not dflash or cfg.get("dflash_verify_width") != (5 if dflash else 0) or cfg.get("proposal_head") != ("optimized" if dflash else "full") or cfg.get("use_device_graph") is not False or cfg.get("retain_token_ids") is not True or cfg.get("isolate_prompt_decode") is not append or cfg.get("decode_path") != ("dflash_eager" if dflash else "eager") or cfg.get("repetitions") != 1 or cfg.get("warmup") != 0 or cfg.get("corpus_path") != str(expected_corpus) or cfg.get("corpus_tokens") != (65536 if append else 129): fail(f"report config differs: {stem}")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1: fail(f"test inventory differs: {stem}")
    test = tests[0]
    if test.get("label") != label or test.get("kind") != ("pp+tg" if append else "whole") or test.get("n_prompt") != prompt or test.get("n_gen") != gen or test.get("requested_output_tokens") != gen+1: fail(f"test identity differs: {stem}")
    reps = test.get("reps")
    if not isinstance(reps, list) or len(reps) != 1: fail(f"rep inventory differs: {stem}")
    rep = reps[0]; lanes = rep.get("generated_token_ids_by_lane")
    if not isinstance(lanes, list) or len(lanes) != 1 or len(lanes[0]) != gen+1 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 or x>=248077 for x in lanes[0]): fail(f"retained outputs differ: {stem}")
    spec = rep.get("speculative", {})
    if spec.get("enabled") is not dflash or spec.get("draft_window") != (4 if dflash else 0): fail(f"spec identity differs: {stem}")
    if not dflash and spec != {"enabled":False,"draft_window":0,"rounds":0,"drafted_tokens":0,"accepted_tokens":0,"fallback_steps":0,"acceptance_rate":None,"acceptance_length":None,"accepted_per_position":[]}: fail(f"ordinary speculative counters differ: {stem}")
    if dflash:
        for key in ("rounds","drafted_tokens","accepted_tokens","fallback_steps"):
            integer(spec.get(key), f"{stem} {key}")
        positions=spec.get("accepted_per_position")
        if not isinstance(positions,list) or len(positions)!=4:
            fail(f"spec position accounting differs: {stem}")
        for index,value in enumerate(positions): integer(value,f"{stem} accepted position {index}")
        if spec["rounds"] + spec["accepted_tokens"] + spec["fallback_steps"] != gen: fail(f"spec accounting differs: {stem}")
        if not math.isclose(finite(spec.get("acceptance_rate"),stem),spec["accepted_tokens"]/spec["drafted_tokens"],rel_tol=0,abs_tol=1e-9) or not math.isclose(finite(spec.get("acceptance_length"),stem),1+spec["accepted_tokens"]/spec["rounds"],rel_tol=0,abs_tol=1e-9): fail(f"spec rates differ: {stem}")
    return identity(path), lanes[0], spec

def validate_decision_trace(path: Path, kind: str, outputs: list[int], report_spec: dict) -> dict:
    trace = load(path)
    if set(trace) != {"artifact_type","schema_version","diagnostic_only","timing_eligible","production_routing_authorized","events"} or trace.get("artifact_type") != "ninfer_dflash_target_decision_trace" or trace.get("schema_version") != 1 or trace.get("diagnostic_only") is not True or trace.get("timing_eligible") is not False or trace.get("production_routing_authorized") is not False: fail("decision trace identity differs")
    events = trace.get("events")
    if not isinstance(events, list) or not events: fail("decision trace has no events")
    if kind == "ordinary":
        if len(events) != 27: fail("ordinary decision trace must cover output indices 1..27")
        for index, event in enumerate(events, 1):
            token(event.get("sampled_token"),f"ordinary sampled token {index}")
            if set(event) != {"kind","logits_stage","lane","token_domain","input_cache_position","absolute_frontier","sampled_token","sampled_matches_top1","next_frontier","decision"} or event.get("kind") != "ordinary" or event.get("logits_stage") != "pre_sample" or event.get("lane") != 0 or event.get("token_domain") != 248077 or event.get("input_cache_position") != 128+index or event.get("absolute_frontier") != 129+index or event.get("next_frontier") != 129+index or event.get("sampled_token") != outputs[index] or event.get("sampled_matches_top1") is not True: fail(f"ordinary event differs at output {index}")
            decision(event.get("decision"), f"ordinary[{index}]")
    else:
        frontier = 129; context_frontier = 129; generated = []
        rounds=drafted=accepted=fallback=0; accepted_positions=[0,0,0,0]
        for event_index, event in enumerate(events):
            if set(event) != {"kind","lane","token_domain","verify_width","tree_verify","anchor","base_frontier","context_frontier","proposal_extent","accepted_drafts","accepted_column","licensed_tokens","next_frontier","columns"}: fail(f"DFlash event fields differ: {event_index}")
            licensed = event.get("licensed_tokens")
            token(event.get("anchor"),f"DFlash anchor {event_index}")
            if isinstance(licensed,list):
                for index,value in enumerate(licensed): token(value,f"DFlash licensed token {event_index}/{index}")
            extent = event.get("proposal_extent")
            accepted_drafts=event.get("accepted_drafts"); accepted_column=event.get("accepted_column")
            expected_extent=min(4,27-len(generated)-1)
            if event.get("kind") != "dflash" or event.get("lane") != 0 or event.get("token_domain") != 248077 or event.get("verify_width") != 5 or event.get("tree_verify") is not False or event.get("anchor") != outputs[len(generated)] or event.get("base_frontier") != frontier or event.get("context_frontier") != context_frontier or isinstance(extent,bool) or not isinstance(extent,int) or extent != expected_extent or isinstance(accepted_drafts,bool) or not isinstance(accepted_drafts,int) or not 0 <= accepted_drafts <= extent or isinstance(accepted_column,bool) or not isinstance(accepted_column,int) or accepted_column != accepted_drafts or not isinstance(licensed, list) or len(licensed) != accepted_drafts+1 or event.get("next_frontier") != frontier+len(licensed): fail(f"DFlash event contract differs: {event_index}")
            columns = event.get("columns")
            if not isinstance(columns, list) or len(columns) != extent+1: fail(f"DFlash columns differ: {event_index}")
            for column_index, column in enumerate(columns):
                token(column.get("verify_token"),f"DFlash verify token {event_index}/{column_index}")
                token(column.get("target_argmax"),f"DFlash target argmax {event_index}/{column_index}")
                if set(column) != {"column","logits_stage","verify_token","input_cache_position","absolute_frontier","target_argmax","target_argmax_matches_top1","decision"} or column.get("column") != column_index or column.get("logits_stage") != "pre_accept_target_verify" or column.get("input_cache_position") != frontier+column_index or column.get("absolute_frontier") != frontier+column_index+1 or column.get("target_argmax_matches_top1") is not True: fail(f"DFlash column differs: {event_index}/{column_index}")
                ranked = decision(column.get("decision"), f"dflash[{event_index}][{column_index}]")
                if column.get("target_argmax") != ranked["top1_token"]: fail("DFlash argmax differs from ranked logit")
            if columns[0].get("verify_token") != event.get("anchor"): fail(f"DFlash anchor column differs: {event_index}")
            for index in range(accepted_drafts):
                if licensed[index] != columns[index+1].get("verify_token"): fail(f"DFlash accepted draft differs: {event_index}/{index}")
            if licensed[accepted_drafts] != columns[accepted_drafts].get("target_argmax"): fail(f"DFlash terminal token differs: {event_index}")
            if extent: rounds+=1; drafted+=extent
            else: fallback+=1
            accepted+=accepted_drafts
            for index in range(accepted_drafts): accepted_positions[index]+=1
            generated.extend(licensed); context_frontier = frontier; frontier += len(licensed)
        if frontier != 156 or generated != outputs[1:]: fail("DFlash licensed-token chain does not cover outputs 1..27")
        exact_spec={"enabled":True,"draft_window":4,"rounds":rounds,"drafted_tokens":drafted,"accepted_tokens":accepted,"fallback_steps":fallback,"accepted_per_position":accepted_positions}
        if any(report_spec.get(key)!=value for key,value in exact_spec.items()) or not math.isclose(report_spec["acceptance_rate"],accepted/drafted,rel_tol=0,abs_tol=1e-9) or not math.isclose(report_spec["acceptance_length"],1+accepted/rounds,rel_tol=0,abs_tol=1e-9): fail("DFlash trace acceptance differs from report")
    return identity(path)

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--summary",type=Path,required=True); args=parser.parse_args()
    if args.summary != RESULTS/"summary.json" or args.summary.exists() or args.summary.is_symlink(): fail("summary output is not fresh/exact")
    validate_plan()
    repair_identity=validate_analysis_repair()
    if identity(EXE)["sha256"] != EXPECTED_EXE or identity(ARTIFACT)["sha256"] != EXPECTED_ARTIFACT: fail("bound executable/artifact differs")
    reports={}; sequences={}; report_specs={}
    for stem,kind in (("decision-ordinary","decision-ordinary"),("decision-dflash","decision-dflash"),("tail-fresh","tail-fresh"),("tail-append","tail-append")):
        reports[stem],sequences[stem],report_specs[stem]=validate_report(stem,kind)
    ordinary=sequences["decision-ordinary"]; dflash=sequences["decision-dflash"]
    if ordinary[:27] != dflash[:27]: fail("ordinary/DFlash no longer share exact outputs 0..26")
    first_difference=next((i for i,(a,b) in enumerate(zip(ordinary,dflash,strict=True)) if a!=b),None)
    traces={
      "ordinary":validate_decision_trace(RESULTS/"decision-ordinary.trace.json","ordinary",ordinary,report_specs["decision-ordinary"]),
      "dflash":validate_decision_trace(RESULTS/"decision-dflash.trace.json","dflash",dflash,report_specs["decision-dflash"]),
      "tail_fresh":identity(RESULTS/"tail-fresh.trace.json"), "tail_append":identity(RESULTS/"tail-append.trace.json")}
    comparison=load(RESULTS/"tail-comparison.json")
    if comparison.get("artifact_type") != "ninfer_qwen3_text_prefill_p129_tail_comparison" or comparison.get("diagnostic_only") is not True or comparison.get("timing_evidence_eligible") is not False: fail("tail comparison identity differs")
    recomputed = compare_tail(load(RESULTS/"tail-fresh.trace.json"),
                              load(RESULTS/"tail-append.trace.json"))
    if comparison != recomputed: fail("tail comparison differs from exact trace inputs")
    comparator_command=["/usr/bin/python3",str(ROOT/"tools/bench/compare_prefill_p129_tail_trace.py"),"--fresh",str(RESULTS/"tail-fresh.trace.json"),"--append",str(RESULTS/"tail-append.trace.json"),"--out",str(RESULTS/"tail-comparison.json")]
    comparator_process=load(RESULTS/"tail-comparator.process.json")
    if comparator_process.get("command") != comparator_command or comparator_process.get("exit_code") != 0 or comparator_process.get("stdout") != identity(RESULTS/"tail-comparator.stdout") or comparator_process.get("stderr") != identity(RESULTS/"tail-comparator.stderr") or (RESULTS/"tail-comparator.stdout").read_bytes() or (RESULTS/"tail-comparator.stderr").read_bytes(): fail("tail comparator process differs")
    summary={"artifact_type":"ninfer_r9700_dflash_semantic_trace_evidence","schema_version":1,"status":"valid_functional_diagnostic","timing_evidence_eligible":False,"production_routing_authorized":False,"source_commit":"aebd5f82b91a756429651b2795e3ad24c1058cb3","analysis_repair":repair_identity,"capture_result_closure":identity(RESULTS/"result.sha256"),"executable":identity(EXE),"artifact":identity(ARTIFACT),"history":{"fixture":identity(HISTORY),"seed_token":24178},"decision_campaign":{"common_prefix_last_index":26,"first_output_difference_index":first_difference,"ordinary_output_index27":ordinary[27],"dflash_output_index27":dflash[27]},"prefill_tail_comparison":comparison,"tail_comparator_process":identity(RESULTS/"tail-comparator.process.json"),"reports":reports,"traces":traces,"limitations":["functional diagnostic only; synchronous trace copies invalidate all timings","does not identify stale state without a first differing semantic boundary","does not authorize production routing or recipe selection"]}
    args.summary.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n"); return 0

if __name__=="__main__": raise SystemExit(main())
