#!/usr/bin/env python3
"""Fail-closed analyzer for the exact e915a5e4 layer-1 GDN diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
SOURCE = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700-layer1-gdn-src-e915a5e4")
RESULTS = PACKAGE / "results"
EXE = ROOT / "build-r9700-layer1-gdn-e915a5e4-20260906/bench/ninfer_bench"
COMPARATOR = SOURCE / "tools/bench/compare_qwen3_layer_boundary_trace.py"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
HISTORY = ROOT / "profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
sys.path.insert(0, str(SOURCE / "tools/bench"))
from compare_qwen3_layer_boundary_trace import compare_gdn, load, load_gdn  # noqa: E402

ROLES = {
    "target-ordinary": "target-ordinary-frontier130",
    "target-dflash": "target-dflash-frontier130-column0",
    "text-fresh": "text-fresh-frontier129-column128",
    "text-append": "text-append-frontier129-column0",
}
GDN_FIELDS = [
    {"name":"h","dtype":"bf16","elements":5120,"offset":0,"bytes":10240},
    {"name":"g","dtype":"fp32","elements":48,"offset":10240,"bytes":192},
    {"name":"beta","dtype":"fp32","elements":48,"offset":10432,"bytes":192},
    {"name":"z","dtype":"bf16","elements":6144,"offset":10624,"bytes":12288},
    {"name":"q","dtype":"bf16","elements":2048,"offset":22912,"bytes":4096},
    {"name":"k","dtype":"bf16","elements":2048,"offset":27008,"bytes":4096},
    {"name":"v","dtype":"bf16","elements":6144,"offset":31104,"bytes":12288},
    {"name":"o","dtype":"bf16","elements":6144,"offset":43392,"bytes":12288},
    {"name":"on","dtype":"bf16","elements":6144,"offset":55680,"bytes":12288},
    {"name":"x","dtype":"bf16","elements":5120,"offset":67968,"bytes":10240},
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            fail(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict):
        fail(f"not object: {path}")
    return value


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def command(stem: str, report: Path) -> list[str]:
    if stem not in ROLES:
        fail("unknown arm")
    corpus = CORPUS if stem == "text-append" else HISTORY
    result = [str(EXE), "--weights", str(ARTIFACT), "--corpus", str(corpus), "--device", "0",
              "--concurrency", "1"]
    result += (["-pg", "128,1"] if stem == "text-append" else ["--whole-pg", "129,1"])
    result += ["--prefill-chunk", "4096", "--kv-capacity", "workload"]
    if stem == "target-dflash":
        result += ["--spec", "dflash", "--draft-tokens", "4", "--dflash-verify-width", "5",
                   "--lm-head-draft"]
    else:
        result += ["--draft-tokens", "0"]
    result += ["--retain-token-ids"]
    if stem == "text-append":
        result += ["--isolate-prompt-decode"]
    result += ["--no-device-graph", "--output", "json", "-r", "1", "--warmup", "0"]
    return result


def comparator_command(label: str, left: str, right: str, output: Path) -> list[str]:
    if label not in {"target", "text"}:
        fail("unknown comparison")
    return ["/usr/bin/python3", str(COMPARATOR), "--left", str(RESULTS / f"{left}.gdn.json"),
            "--right", str(RESULTS / f"{right}.gdn.json"), "--diagnostic", label,
            "--gdn-detail", "--out", str(output)]


def validate_speculative(value: object, expected: bool, stem: str) -> None:
    keys = {"enabled", "draft_window", "rounds", "drafted_tokens", "accepted_tokens",
            "fallback_steps", "acceptance_rate", "acceptance_length", "accepted_per_position"}
    if not isinstance(value, dict) or set(value) != keys or value.get("enabled") is not expected:
        fail(f"speculative schema differs: {stem}")
    if not expected:
        exact = {"enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
                 "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
                 "acceptance_length": None, "accepted_per_position": []}
        if value != exact:
            fail(f"ordinary speculative accounting differs: {stem}")
        return
    for key in ("draft_window", "rounds", "drafted_tokens", "accepted_tokens", "fallback_steps"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 0:
            fail(f"speculative integer differs: {stem}/{key}")
    if value["draft_window"] != 4 or value["rounds"] + value["accepted_tokens"] + value["fallback_steps"] != 1:
        fail(f"speculative accounting differs: {stem}")
    if value["drafted_tokens"] == 0:
        if (value["rounds"] != 0 or value["accepted_tokens"] != 0 or
                value["fallback_steps"] != 1 or value["acceptance_rate"] is not None or
                value["acceptance_length"] is not None or value["accepted_per_position"] != [0,0,0,0]):
            fail(f"zero-draft accounting differs: {stem}")
    else:
        if (value["drafted_tokens"] > 4 * value["rounds"] or
                value["accepted_tokens"] > value["drafted_tokens"] or
                not isinstance(value["accepted_per_position"], list) or
                len(value["accepted_per_position"]) != 4 or
                any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in value["accepted_per_position"]) or
                sum(value["accepted_per_position"]) != value["accepted_tokens"]):
            fail(f"positive speculative accounting differs: {stem}")
        for key in ("acceptance_rate", "acceptance_length"):
            if isinstance(value.get(key), bool) or not isinstance(value.get(key), (int, float)) or not math.isfinite(value[key]):
                fail(f"speculative rate differs: {stem}/{key}")
        if (not math.isclose(value["acceptance_rate"],
                             value["accepted_tokens"] / value["drafted_tokens"],
                             rel_tol=1e-12, abs_tol=1e-12) or
                not math.isclose(value["acceptance_length"],
                                 (value["rounds"] + value["accepted_tokens"]) /
                                 value["rounds"],
                                 rel_tol=1e-12, abs_tol=1e-12)):
            fail(f"positive speculative rate formula differs: {stem}")


def validate_report(stem: str) -> tuple[dict, list[int]]:
    path = RESULTS / f"{stem}.json"
    report = load_json(path)
    if (set(report) != {"schema_version", "artifact_type", "tool", "command", "environment",
                        "artifact", "load", "memory", "config", "tests"} or
            report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or report.get("command") != " ".join(command(stem, path))):
        fail(f"report identity differs: {stem}")
    if report.get("environment") != {"gpu_name":"AMD Radeon AI PRO R9700",
            "architecture_name":"gfx1201", "hip_runtime_version":"7.15.26333",
            "hip_driver_version":"7.15.26333", "device_id":0}:
        fail(f"report hardware differs: {stem}")
    if report.get("artifact") != {"path":str(ARTIFACT), "file_size_bytes":22763026944}:
        fail(f"report artifact differs: {stem}")
    expected_spec = stem == "target-dflash"
    expected_load = ((22287117113, 22274271744, 839) if expected_spec else
                     (20720607545, 20707762176, 771))
    load_record = report.get("load")
    if (not isinstance(load_record, dict) or set(load_record) != {"target", "weights_id",
            "load_seconds", "upload_seconds", "artifact_bytes_read", "host_to_device_bytes",
            "peak_staging_bytes", "tensor_count", "resource_count"} or
            load_record.get("target") != "qwen3_8_27b_r9700" or
            load_record.get("weights_id") !=
            "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval" or
            load_record.get("artifact_bytes_read") != expected_load[0] or
            load_record.get("host_to_device_bytes") != expected_load[1] or
            load_record.get("peak_staging_bytes") != 268435456 or
            load_record.get("tensor_count") != expected_load[2] or
            load_record.get("resource_count") != 6):
        fail(f"report load contract differs: {stem}")
    for key in ("load_seconds", "upload_seconds"):
        if (isinstance(load_record.get(key), bool) or
                not isinstance(load_record.get(key), (int, float)) or
                not math.isfinite(load_record[key]) or load_record[key] <= 0):
            fail(f"report load timing differs: {stem}/{key}")
    config = report.get("config")
    expected_context = 140 if expected_spec else 130
    expected_corpus = str(CORPUS if stem == "text-append" else HISTORY)
    config_keys = {"max_context", "prefill_chunk", "kv_cache_format", "kv_value_group",
        "kv_plane_layouts", "q4_activation_bits", "q4_prefill_cta_profile",
        "dflash_small_t_candidate", "dflash_mlp_down_t5_candidate",
        "dflash_rmsnorm_rows56_candidate", "w8_activation_bits", "fp8_qk_wmma_enabled",
        "fp8_qk_wmma_profile", "fp8_qk_wmma_t1_min_context", "fp8_qk_wmma_t2_min_context",
        "xattention_qualification", "concurrency", "pending_timeout_ms", "pending_deadline",
        "spec", "draft_tokens", "speculative_execution", "dflash_verify_width_requested",
        "dflash_verify_width", "proposal_head", "use_device_graph", "retain_token_ids",
        "isolate_prompt_decode", "decode_path", "decode_graph_prime", "repetitions", "warmup",
        "corpus_path", "corpus_tokens"}
    if (not isinstance(config, dict) or set(config) != config_keys or
            config.get("concurrency") != 1 or
            config.get("max_context") != expected_context or config.get("prefill_chunk") != 4096 or
            config.get("kv_cache_format") != "fp8-k-int4-v" or config.get("kv_value_group") != 16 or
            config.get("kv_plane_layouts") != {"key":"token-fastest-head-major",
                "value":"feature-fastest-page-major", "value_scale":"feature-fastest-page-major"} or
            config.get("q4_activation_bits") != 8 or
            config.get("q4_prefill_cta_profile") !=
            "m64n128-pingpong-n16-k16-scalar-base-production" or
            config.get("w8_activation_bits") != 8 or
            config.get("fp8_qk_wmma_enabled") is not True or
            config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1" or
            config.get("fp8_qk_wmma_t1_min_context") != 64 or
            config.get("fp8_qk_wmma_t2_min_context") != 320 or
            config.get("dflash_small_t_candidate") is not False or
            config.get("dflash_mlp_down_t5_candidate") is not True or
            config.get("dflash_rmsnorm_rows56_candidate") is not True or
            config.get("xattention_qualification") is not False or
            config.get("pending_timeout_ms") != 4294967295 or
            config.get("pending_deadline") != "unbounded" or
            config.get("use_device_graph") is not False or config.get("retain_token_ids") is not True or
            config.get("isolate_prompt_decode") is not (stem == "text-append") or
            config.get("corpus_path") != expected_corpus or
            config.get("corpus_tokens") != (65536 if stem == "text-append" else 129) or
            config.get("repetitions") != 1 or config.get("warmup") != 0 or
            config.get("speculative_execution") is not expected_spec or
            config.get("decode_graph_prime") != {"primed":False, "output_tokens":0}):
        fail(f"report configuration differs: {stem}")
    if expected_spec:
        if (config.get("spec") != "dflash" or config.get("draft_tokens") != 4 or
                config.get("dflash_verify_width_requested") != 5 or
                config.get("dflash_verify_width") != 5 or config.get("proposal_head") != "optimized" or
                config.get("decode_path") != "dflash_eager"):
            fail("DFlash K4/W5 report differs")
    elif (config.get("spec") != "none" or config.get("draft_tokens") != 0 or
          config.get("dflash_verify_width_requested") != 0 or
          config.get("dflash_verify_width") != 0 or config.get("proposal_head") != "full" or
          config.get("decode_path") != "eager"):
        fail(f"ordinary report differs: {stem}")
    memory = report.get("memory")
    if (not isinstance(memory, dict) or isinstance(memory.get("planned_slack_bytes"), bool) or
            not isinstance(memory.get("planned_slack_bytes"), int) or
            memory["planned_slack_bytes"] < 1399180 or
            isinstance(memory.get("available_after_startup_bytes"), bool) or
            not isinstance(memory.get("available_after_startup_bytes"), int) or
            memory["available_after_startup_bytes"] < 1399180):
        fail(f"trace staging headroom differs: {stem}")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        fail(f"report test count differs: {stem}")
    test = tests[0]
    test_keys = {"label", "kind", "n_prompt", "n_gen", "requested_output_tokens",
        "prefill_tok_s_mean", "prefill_tok_s_stddev", "decode_output_tok_s_mean",
        "decode_output_tok_s_stddev", "decode_engine_tok_s_mean", "decode_engine_tok_s_stddev",
        "whole_output_tok_s_mean", "whole_output_tok_s_stddev", "prepare_seconds_mean",
        "prepare_seconds_stddev", "prefill_seconds_mean", "prefill_seconds_stddev",
        "decode_seconds_mean", "decode_seconds_stddev", "total_seconds_mean",
        "total_seconds_stddev", "workspace_peak_bytes", "workspace_allocator_peak_bytes",
        "speculative", "reps"}
    expected_label = "pp128+tg1" if stem == "text-append" else "whole-pp129+tg1"
    expected_kind = "pp+tg" if stem == "text-append" else "whole"
    if (not isinstance(test, dict) or set(test) != test_keys or test.get("label") != expected_label or
            test.get("kind") != expected_kind or
            test.get("n_prompt") != (128 if stem == "text-append" else 129) or
            test.get("n_gen") != 1 or test.get("requested_output_tokens") != 2):
        fail(f"test contract differs: {stem}")
    reps = test.get("reps") if isinstance(test, dict) else None
    rep = reps[0] if isinstance(reps, list) and len(reps) == 1 else None
    if (not isinstance(rep, dict) or set(rep) != {"generated_output_tokens", "decode_output_tokens",
            "decode_engine_tokens", "generated_token_ids_by_lane", "timings", "speculative"} or
            rep.get("generated_output_tokens") != 2 or rep.get("decode_output_tokens") != 1):
        fail(f"repetition contract differs: {stem}")
    lanes = rep.get("generated_token_ids_by_lane") if isinstance(rep, dict) else None
    if not isinstance(lanes, list) or len(lanes) != 1 or lanes[0] != [96558, 96917]:
        fail(f"retained tokens differ: {stem}")
    validate_speculative(test.get("speculative"), expected_spec, stem)
    validate_speculative(rep.get("speculative"), expected_spec, stem)
    if test.get("speculative") != rep.get("speculative"):
        fail(f"test/repetition speculative accounting differs: {stem}")
    if rep.get("decode_engine_tokens") != (rep["speculative"]["rounds"] +
            rep["speculative"]["accepted_tokens"] + rep["speculative"]["fallback_steps"]
            if expected_spec else 1):
        fail(f"decode engine-token accounting differs: {stem}")
    timings = rep.get("timings")
    if not isinstance(timings, dict) or set(timings) != {"prepare_seconds", "vision_seconds",
            "prefill_seconds", "decode_seconds", "total_seconds"}:
        fail(f"timing schema differs: {stem}")
    for key, value in timings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            fail(f"timing value differs: {stem}/{key}")
    if (timings["vision_seconds"] != 0 or
            (timings["prefill_seconds"] <= 0 and stem != "text-append") or
            (timings["prefill_seconds"] != 0 and stem == "text-append") or
            timings["decode_seconds"] <= 0 or timings["total_seconds"] <
            timings["prepare_seconds"] + timings["prefill_seconds"] + timings["decode_seconds"]):
        fail(f"timing relationship differs: {stem}")
    expected_means = {
        "prepare_seconds_mean":timings["prepare_seconds"],
        "prefill_seconds_mean":timings["prefill_seconds"],
        "decode_seconds_mean":timings["decode_seconds"], "total_seconds_mean":timings["total_seconds"],
        "decode_output_tok_s_mean":rep["decode_output_tokens"] / timings["decode_seconds"],
        "decode_engine_tok_s_mean":rep["decode_engine_tokens"] / timings["decode_seconds"],
    }
    if stem != "text-append":
        expected_means["prefill_tok_s_mean"] = test["n_prompt"] / timings["prefill_seconds"]
        expected_means["whole_output_tok_s_mean"] = \
            rep["generated_output_tokens"] / timings["total_seconds"]
    for key, expected_value in expected_means.items():
        if (isinstance(test.get(key), bool) or not isinstance(test.get(key), (int, float)) or
                not math.isfinite(test[key]) or not math.isclose(test[key], expected_value,
                                                                 rel_tol=2e-9, abs_tol=1e-12)):
            fail(f"aggregate mean differs: {stem}/{key}")
    if stem == "text-append" and any(test.get(key) is not None for key in
            ("prefill_tok_s_mean", "prefill_tok_s_stddev", "whole_output_tok_s_mean",
             "whole_output_tok_s_stddev")):
        fail("isolated pp+tg undefined rate differs")
    for key in (("decode_output_tok_s_stddev", "decode_engine_tok_s_stddev",
                "prepare_seconds_stddev", "prefill_seconds_stddev", "decode_seconds_stddev",
                "total_seconds_stddev") + (() if stem == "text-append" else
                ("prefill_tok_s_stddev", "whole_output_tok_s_stddev"))):
        if test.get(key) != 0:
            fail(f"single-repetition stddev differs: {stem}/{key}")
    if (isinstance(test.get("workspace_peak_bytes"), bool) or
            not isinstance(test.get("workspace_peak_bytes"), int) or test["workspace_peak_bytes"] <= 0 or
            isinstance(test.get("workspace_allocator_peak_bytes"), bool) or
            not isinstance(test.get("workspace_allocator_peak_bytes"), int) or
            not 0 < test["workspace_allocator_peak_bytes"] <= test["workspace_peak_bytes"]):
        fail(f"workspace peak differs: {stem}")
    return identity(path), lanes[0]


def validate_process(stem: str) -> dict:
    record_path = RESULTS / f"{stem}.process.json"
    record = load_json(record_path)
    expected_env = {
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": ROLES[stem],
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST": str(RESULTS / f"{stem}.trace.json"),
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR": str(RESULTS / f"{stem}.trace.bin"),
        "NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST": str(RESULTS / f"{stem}.gdn.json"),
        "NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR": str(RESULTS / f"{stem}.gdn.bin"),
    }
    if (set(record) != {"command", "trace_environment", "exit_code", "started_unix_ns",
            "finished_unix_ns", "power_before", "power_after",
            "instrumentation_environment_present", "stdout", "stderr"} or
            record.get("command") != command(stem, RESULTS / f"{stem}.json") or
            record.get("trace_environment") != expected_env or record.get("exit_code") != 0 or
            record.get("power_before") != "auto" or record.get("power_after") != "auto" or
            record.get("instrumentation_environment_present") != [] or
            record.get("stdout") != identity(RESULTS / f"{stem}.stdout") or
            record.get("stderr") != identity(RESULTS / f"{stem}.stderr") or
            (RESULTS / f"{stem}.stdout").read_bytes() != (RESULTS / f"{stem}.json").read_bytes()):
        fail(f"process record differs: {stem}")
    expected_context = 140 if stem == "target-dflash" else 130
    expected_label = "pp128+tg1" if stem == "text-append" else "whole-pp129+tg1"
    expected_stderr = (
        f"[ninfer_bench] loading {ARTIFACT} (max_context={expected_context}, concurrency=1, "
        "kv_format=fp8-k-int4-v)\n"
        f"[ninfer_bench] test 1/1 {expected_label}: warmup=0 reps=1\n")
    if (RESULTS / f"{stem}.stderr").read_text() != expected_stderr:
        fail(f"benchmark stderr differs: {stem}")
    if (not isinstance(record.get("started_unix_ns"), int) or
            not isinstance(record.get("finished_unix_ns"), int) or
            record["finished_unix_ns"] <= record["started_unix_ns"]):
        fail(f"process interval differs: {stem}")
    return identity(record_path)


def validate_comparator(label: str, left: str, right: str, expected: dict) -> dict:
    output = RESULTS / f"{label}-gdn-comparison.json"
    stdout = RESULTS / f"{label}-gdn-comparator.stdout"
    stderr = RESULTS / f"{label}-gdn-comparator.stderr"
    process_path = RESULTS / f"{label}-gdn-comparator.process.json"
    process = load_json(process_path)
    if (set(process) != {"command", "exit_code", "stdout", "stderr"} or
            process.get("command") != comparator_command(label, left, right, output) or
            process.get("exit_code") != 0 or process.get("stdout") != identity(stdout) or
            process.get("stderr") != identity(stderr) or stdout.read_bytes() or stderr.read_bytes() or
            load_json(output) != expected):
        fail(f"comparator process differs: {label}")
    return identity(process_path)


def analyze() -> dict:
    reports, processes, tokens, layer_manifests, layer_sidecars = {}, {}, {}, {}, {}
    gdn_manifests, gdn_sidecars = {}, {}
    expected_roles = {
        "target-ordinary": ("target-ordinary-frontier130",1,0,130,129),
        "target-dflash": ("target-dflash-frontier130-column0",5,0,130,129),
        "text-fresh": ("text-fresh-frontier129-column128",129,128,129,128),
        "text-append": ("text-append-frontier129-column0",1,0,129,128),
    }
    for stem in ROLES:
        processes[stem] = validate_process(stem)
        reports[stem], tokens[stem] = validate_report(stem)
        layer_path, gdn_path = RESULTS / f"{stem}.trace.json", RESULTS / f"{stem}.gdn.json"
        layer_manifest, _ = load(layer_path, expected_roles[stem])
        gdn_manifest, _ = load_gdn(gdn_path, expected_roles[stem])
        expected_token = 96558 if stem.startswith("target-") else 24178
        if layer_manifest["token"] != expected_token or gdn_manifest["token"] != expected_token:
            fail(f"selected trace token differs: {stem}")
        layer_manifests[stem], layer_sidecars[stem] = identity(layer_path), identity(RESULTS / f"{stem}.trace.bin")
        gdn_manifests[stem], gdn_sidecars[stem] = identity(gdn_path), identity(RESULTS / f"{stem}.gdn.bin")
    target = compare_gdn(RESULTS / "target-ordinary.gdn.json", RESULTS / "target-dflash.gdn.json", "target")
    text = compare_gdn(RESULTS / "text-fresh.gdn.json", RESULTS / "text-append.gdn.json", "text")
    comparator_processes = {
        "target": validate_comparator("target", "target-ordinary", "target-dflash", target),
        "text": validate_comparator("text", "text-fresh", "text-append", text),
    }
    return {
        "artifact_type":"ninfer_qwen3_layer1_gdn_detail_trace_evidence", "schema_version":1,
        "status":"valid_functional_localization", "timing_evidence_eligible":False,
        "production_routing_authorized":False,
        "source_commit":"e915a5e45a51c07784b36aede8650352bfe5e374",
        "authorities":{"plan":identity(PACKAGE / "plan.json"),
                       "build_provenance":identity(PACKAGE / "build-provenance.json"),
                       "prepared_closure":identity(PACKAGE / "prepared.sha256"),
                       "comparator":identity(COMPARATOR)},
        "selectors":{"dflash_small_t_candidate":False,
                     "dflash_mlp_down_t5_candidate":True,
                     "dflash_rmsnorm_rows56_candidate":True,
                     "ordinary_w1_uses_rows56_or_t5_candidates":False},
        "claim":"eager-only functional layer1 GDN localization; all captured timings are ineligible",
        "reports":reports, "processes":processes, "tokens":tokens,
        "layer_manifests":layer_manifests, "layer_sidecars":layer_sidecars,
        "gdn_manifests":gdn_manifests, "gdn_sidecars":gdn_sidecars,
        "comparator_processes":comparator_processes,
        "target_gdn_comparison":target, "text_gdn_comparison":text,
        "limitations":["functional localization only; synchronous final D2H invalidates timing",
                       "captures selected columns and no persistent convolution or recurrence state",
                       "no production routing or performance claim"],
    }


def write_summary(path: Path) -> None:
    if path != RESULTS / "summary.json" or path.exists() or path.is_symlink():
        fail("summary output must be the exact fresh path")
    payload = (json.dumps(analyze(), indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        offset = 0
        while offset != len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                fail("summary write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--summary":
        fail("usage differs")
    write_summary(Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
