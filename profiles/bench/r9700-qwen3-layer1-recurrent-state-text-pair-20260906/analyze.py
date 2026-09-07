#!/usr/bin/env python3
"""Fail-closed analyzer for the exact two-arm layer-1 recurrent-state diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
RESULTS = PACKAGE / "results"
SOURCE = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700-recurrent-state-src-c3d44daa")
EXE = ROOT / "build-r9700-recurrent-state-c3d44daa-20260906/bench/ninfer_bench"
COMPARATOR = SOURCE / "tools/bench/compare_qwen3_layer_boundary_trace.py"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
HISTORY = ROOT / "profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
sys.path.insert(0, str(SOURCE / "tools/bench"))
from compare_qwen3_layer_boundary_trace import (  # noqa: E402
    compare_gdn, compare_recurrent_state, load, load_gdn, load_recurrent_state,
)

ROLES = {
    "text-fresh": ("text-fresh-frontier129-column128", 129, 128, 129, 128),
    "text-append": ("text-append-frontier129-column0", 1, 0, 129, 128),
}
EXPECTED_TOKENS = [96558, 96917]


def fail(message: str) -> None:
    raise RuntimeError(message)


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict):
        fail(f"not an object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest(resolved)}


def command(stem: str) -> list[str]:
    if stem not in ROLES:
        fail("unknown arm")
    corpus = CORPUS if stem == "text-append" else HISTORY
    result = [str(EXE), "--weights", str(ARTIFACT), "--corpus", str(corpus), "--device", "0",
              "--concurrency", "1"]
    result += (["-pg", "128,1"] if stem == "text-append" else ["--whole-pg", "129,1"])
    result += ["--prefill-chunk", "4096", "--kv-capacity", "workload", "--draft-tokens", "0",
               "--retain-token-ids"]
    if stem == "text-append":
        result += ["--isolate-prompt-decode"]
    result += ["--no-device-graph", "--output", "json", "-r", "1", "--warmup", "0"]
    return result


def comparator_command(kind: str, output: Path) -> list[str]:
    if kind == "gdn":
        flag = "--gdn-detail"
        suffix = "gdn"
    elif kind == "state":
        flag = "--recurrent-state"
        suffix = "state"
    else:
        fail("unknown comparator kind")
    return ["/usr/bin/python3", str(COMPARATOR),
            "--left", str(RESULTS / f"text-fresh.{suffix}.json"),
            "--right", str(RESULTS / f"text-append.{suffix}.json"),
            "--diagnostic", "text", flag, "--out", str(output)]


def validate_speculative(value: object, stem: str) -> None:
    expected = {"enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
                "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
                "acceptance_length": None, "accepted_per_position": []}
    if value != expected:
        fail(f"speculative accounting differs: {stem}")


def validate_expected_state_comparison(value: dict) -> None:
    detail = value.get("first_difference") if isinstance(value, dict) else None
    if (value.get("classification") != "first_difference_layer1_recurrent_prefix_state" or
            not isinstance(detail, dict) or detail.get("first_element_index") != 599 or
            detail.get("mismatch_count") != 402925 or detail.get("left_bits") != 950422399 or
            detail.get("right_bits") != 950422400 or
            not math.isclose(detail.get("maximum_absolute_difference", math.nan),
                             0.000046528875827789307, rel_tol=0.0, abs_tol=0.0)):
        fail("recurrent-state result differs from the retained preliminary classification")


def validate_expected_gdn_comparison(value: dict) -> None:
    detail = value.get("first_difference") if isinstance(value, dict) else None
    if (value.get("classification") != "first_difference_gdn_recurrence" or
            not isinstance(detail, dict) or detail.get("field") != "o" or
            detail.get("first_element_index") != 154 or detail.get("mismatch_count") != 106 or
            detail.get("left_bits") != 14405 or detail.get("right_bits") != 14406 or
            not math.isclose(detail.get("maximum_absolute_difference", math.nan),
                             0.000003814697265625, rel_tol=0.0, abs_tol=0.0)):
        fail("GDN output result differs from the retained classification")


def validate_report(stem: str) -> tuple[dict, list[int]]:
    path = RESULTS / f"{stem}.json"
    report = load_json(path)
    if (set(report) != {"schema_version", "artifact_type", "tool", "command", "environment",
                        "artifact", "load", "memory", "config", "tests"} or
            report.get("schema_version") != 20 or
            report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or
            report.get("command") != " ".join(command(stem))):
        fail(f"report identity differs: {stem}")
    if report.get("environment") != {"gpu_name": "AMD Radeon AI PRO R9700",
            "architecture_name": "gfx1201", "hip_runtime_version": "7.15.26333",
            "hip_driver_version": "7.15.26333", "device_id": 0}:
        fail(f"report hardware differs: {stem}")
    if report.get("artifact") != {"path": str(ARTIFACT), "file_size_bytes": 22763026944}:
        fail(f"report artifact differs: {stem}")
    load_record = report.get("load")
    if (not isinstance(load_record, dict) or set(load_record) != {"target", "weights_id",
            "load_seconds", "upload_seconds", "artifact_bytes_read", "host_to_device_bytes",
            "peak_staging_bytes", "tensor_count", "resource_count"} or
            load_record.get("target") != "qwen3_8_27b_r9700" or
            load_record.get("weights_id") !=
            "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval" or
            load_record.get("artifact_bytes_read") != 20720607545 or
            load_record.get("host_to_device_bytes") != 20707762176 or
            load_record.get("peak_staging_bytes") != 268435456 or
            load_record.get("tensor_count") != 771 or load_record.get("resource_count") != 6):
        fail(f"report load contract differs: {stem}")
    for key in ("load_seconds", "upload_seconds"):
        value = load_record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            fail(f"report load timing differs: {stem}/{key}")
    config = report.get("config")
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
    expected_corpus = str(CORPUS if stem == "text-append" else HISTORY)
    if (not isinstance(config, dict) or set(config) != config_keys or
            config.get("max_context") != 130 or config.get("prefill_chunk") != 4096 or
            config.get("kv_cache_format") != "fp8-k-int4-v" or config.get("kv_value_group") != 16 or
            config.get("kv_plane_layouts") != {"key": "token-fastest-head-major",
                "value": "feature-fastest-page-major", "value_scale": "feature-fastest-page-major"} or
            config.get("q4_activation_bits") != 8 or
            config.get("q4_prefill_cta_profile") !=
            "m64n128-pingpong-n16-k16-scalar-base-production" or
            config.get("dflash_small_t_candidate") is not False or
            config.get("dflash_mlp_down_t5_candidate") is not False or
            config.get("dflash_rmsnorm_rows56_candidate") is not False or
            config.get("w8_activation_bits") != 8 or config.get("fp8_qk_wmma_enabled") is not True or
            config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1" or
            config.get("fp8_qk_wmma_t1_min_context") != 64 or
            config.get("fp8_qk_wmma_t2_min_context") != 320 or
            config.get("xattention_qualification") is not False or config.get("concurrency") != 1 or
            config.get("pending_timeout_ms") != 4294967295 or
            config.get("pending_deadline") != "unbounded" or config.get("spec") != "none" or
            config.get("draft_tokens") != 0 or config.get("speculative_execution") is not False or
            config.get("dflash_verify_width_requested") != 0 or config.get("dflash_verify_width") != 0 or
            config.get("proposal_head") != "full" or config.get("use_device_graph") is not False or
            config.get("retain_token_ids") is not True or
            config.get("isolate_prompt_decode") is not (stem == "text-append") or
            config.get("decode_path") != "eager" or
            config.get("decode_graph_prime") != {"primed": False, "output_tokens": 0} or
            config.get("repetitions") != 1 or config.get("warmup") != 0 or
            config.get("corpus_path") != expected_corpus or
            config.get("corpus_tokens") != (65536 if stem == "text-append" else 129)):
        fail(f"report configuration differs: {stem}")
    memory = report.get("memory")
    if (not isinstance(memory, dict) or isinstance(memory.get("planned_slack_bytes"), bool) or
            not isinstance(memory.get("planned_slack_bytes"), int) or
            memory["planned_slack_bytes"] < 4544908 or
            isinstance(memory.get("available_after_startup_bytes"), bool) or
            not isinstance(memory.get("available_after_startup_bytes"), int) or
            memory["available_after_startup_bytes"] < 4544908):
        fail(f"trace staging headroom differs: {stem}")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1 or not isinstance(tests[0], dict):
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
    if (set(test) != test_keys or test.get("label") != expected_label or
            test.get("kind") != expected_kind or
            test.get("n_prompt") != (128 if stem == "text-append" else 129) or
            test.get("n_gen") != 1 or test.get("requested_output_tokens") != 2):
        fail(f"test contract differs: {stem}")
    reps = test.get("reps")
    rep = reps[0] if isinstance(reps, list) and len(reps) == 1 else None
    if (not isinstance(rep, dict) or set(rep) != {"generated_output_tokens", "decode_output_tokens",
            "decode_engine_tokens", "generated_token_ids_by_lane", "timings", "speculative"} or
            rep.get("generated_output_tokens") != 2 or
            rep.get("decode_output_tokens") != 1 or rep.get("decode_engine_tokens") != 1 or
            rep.get("generated_token_ids_by_lane") != [EXPECTED_TOKENS]):
        fail(f"repetition/token contract differs: {stem}")
    validate_speculative(test.get("speculative"), stem)
    validate_speculative(rep.get("speculative"), stem)
    if test.get("speculative") != rep.get("speculative"):
        fail(f"test/repetition speculative accounting differs: {stem}")
    timings = rep.get("timings")
    if not isinstance(timings, dict) or set(timings) != {"prepare_seconds", "vision_seconds",
            "prefill_seconds", "decode_seconds", "total_seconds"}:
        fail(f"timing schema differs: {stem}")
    for key, value in timings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            fail(f"timing value differs: {stem}/{key}")
    if (timings["vision_seconds"] != 0 or timings["decode_seconds"] <= 0 or
            (stem == "text-fresh" and timings["prefill_seconds"] <= 0) or
            (stem == "text-append" and timings["prefill_seconds"] != 0)):
        fail(f"timing relationship differs: {stem}")
    expected_means = {
        "prepare_seconds_mean": timings["prepare_seconds"],
        "prefill_seconds_mean": timings["prefill_seconds"],
        "decode_seconds_mean": timings["decode_seconds"],
        "total_seconds_mean": timings["total_seconds"],
        "decode_output_tok_s_mean": 1 / timings["decode_seconds"],
        "decode_engine_tok_s_mean": 1 / timings["decode_seconds"],
    }
    if stem == "text-fresh":
        expected_means["prefill_tok_s_mean"] = 129 / timings["prefill_seconds"]
        expected_means["whole_output_tok_s_mean"] = 2 / timings["total_seconds"]
    for key, expected in expected_means.items():
        value = test.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or
                not math.isclose(value, expected, rel_tol=2e-9, abs_tol=1e-12)):
            fail(f"aggregate mean differs: {stem}/{key}")
    if stem == "text-append" and any(test.get(key) is not None for key in
            ("prefill_tok_s_mean", "prefill_tok_s_stddev", "whole_output_tok_s_mean",
             "whole_output_tok_s_stddev")):
        fail("isolated undefined rate differs")
    stddevs = ("decode_output_tok_s_stddev", "decode_engine_tok_s_stddev",
               "prepare_seconds_stddev", "prefill_seconds_stddev", "decode_seconds_stddev",
               "total_seconds_stddev")
    if stem == "text-fresh":
        stddevs += ("prefill_tok_s_stddev", "whole_output_tok_s_stddev")
    if any(test.get(key) != 0 for key in stddevs):
        fail(f"single-repetition stddev differs: {stem}")
    if (isinstance(test.get("workspace_peak_bytes"), bool) or
            not isinstance(test.get("workspace_peak_bytes"), int) or test["workspace_peak_bytes"] <= 0 or
            isinstance(test.get("workspace_allocator_peak_bytes"), bool) or
            not isinstance(test.get("workspace_allocator_peak_bytes"), int) or
            not 0 < test["workspace_allocator_peak_bytes"] <= test["workspace_peak_bytes"]):
        fail(f"workspace peak differs: {stem}")
    return identity(path), EXPECTED_TOKENS


def expected_trace_environment(stem: str) -> dict[str, str]:
    role = ROLES[stem][0]
    return {
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": role,
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST": str(RESULTS / f"{stem}.trace.json"),
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR": str(RESULTS / f"{stem}.trace.bin"),
        "NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST": str(RESULTS / f"{stem}.gdn.json"),
        "NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR": str(RESULTS / f"{stem}.gdn.bin"),
        "NINFER_QWEN3_GDN_STATE_TRACE_MANIFEST": str(RESULTS / f"{stem}.state.json"),
        "NINFER_QWEN3_GDN_STATE_TRACE_SIDECAR": str(RESULTS / f"{stem}.state.bin"),
    }


def validate_process(stem: str) -> dict:
    path = RESULTS / f"{stem}.process.json"
    value = load_json(path)
    if (set(value) != {"command", "trace_environment", "exit_code", "started_unix_ns",
            "finished_unix_ns", "power_before", "power_after",
            "instrumentation_environment_present", "stdout", "stderr"} or
            value.get("command") != command(stem) or
            value.get("trace_environment") != expected_trace_environment(stem) or
            value.get("exit_code") != 0 or value.get("power_before") != "auto" or
            value.get("power_after") != "auto" or
            value.get("instrumentation_environment_present") != [] or
            value.get("stdout") != identity(RESULTS / f"{stem}.stdout") or
            value.get("stderr") != identity(RESULTS / f"{stem}.stderr") or
            (RESULTS / f"{stem}.stdout").read_bytes() != (RESULTS / f"{stem}.json").read_bytes()):
        fail(f"process record differs: {stem}")
    if (isinstance(value.get("started_unix_ns"), bool) or
            not isinstance(value.get("started_unix_ns"), int) or
            isinstance(value.get("finished_unix_ns"), bool) or
            not isinstance(value.get("finished_unix_ns"), int) or
            value["finished_unix_ns"] <= value["started_unix_ns"]):
        fail(f"process interval differs: {stem}")
    label = "pp128+tg1" if stem == "text-append" else "whole-pp129+tg1"
    expected_stderr = (f"[ninfer_bench] loading {ARTIFACT} (max_context=130, concurrency=1, "
                       "kv_format=fp8-k-int4-v)\n"
                       f"[ninfer_bench] test 1/1 {label}: warmup=0 reps=1\n")
    if (RESULTS / f"{stem}.stderr").read_text() != expected_stderr:
        fail(f"benchmark stderr differs: {stem}")
    return identity(path)


def validate_comparator(kind: str, expected: dict) -> dict:
    output = RESULTS / f"text-{kind}-comparison.json"
    stdout = RESULTS / f"text-{kind}-comparator.stdout"
    stderr = RESULTS / f"text-{kind}-comparator.stderr"
    process_path = RESULTS / f"text-{kind}-comparator.process.json"
    process = load_json(process_path)
    if (set(process) != {"command", "exit_code", "stdout", "stderr"} or
            process.get("command") != comparator_command(kind, output) or
            process.get("exit_code") != 0 or process.get("stdout") != identity(stdout) or
            process.get("stderr") != identity(stderr) or stdout.read_bytes() or stderr.read_bytes() or
            load_json(output) != expected):
        fail(f"comparator record differs: {kind}")
    return identity(process_path)


def analyze() -> dict:
    reports, processes, manifests, sidecars = {}, {}, {}, {}
    for stem, expected in ROLES.items():
        reports[stem], _ = validate_report(stem)
        processes[stem] = validate_process(stem)
        layer_path = RESULTS / f"{stem}.trace.json"
        gdn_path = RESULTS / f"{stem}.gdn.json"
        state_path = RESULTS / f"{stem}.state.json"
        layer, _ = load(layer_path, expected)
        gdn, _ = load_gdn(gdn_path, expected)
        state, _ = load_recurrent_state(state_path, expected[0])
        if layer.get("token") != 24178 or gdn.get("token") != 24178 or state.get("selected_token") != 24178:
            fail(f"selected input token differs: {stem}")
        manifests[stem] = {name: identity(path) for name, path in
                           (("layer", layer_path), ("gdn", gdn_path), ("state", state_path))}
        sidecars[stem] = {name: identity(RESULTS / f"{stem}.{name}.bin")
                          for name in ("trace", "gdn", "state")}
    gdn_comparison = compare_gdn(RESULTS / "text-fresh.gdn.json",
                                 RESULTS / "text-append.gdn.json", "text")
    state_comparison = compare_recurrent_state(RESULTS / "text-fresh.state.json",
                                               RESULTS / "text-append.state.json")
    validate_expected_state_comparison(state_comparison)
    validate_expected_gdn_comparison(gdn_comparison)
    comparator_processes = {
        "gdn": validate_comparator("gdn", gdn_comparison),
        "state": validate_comparator("state", state_comparison),
    }
    return {
        "artifact_type": "ninfer_qwen3_layer1_recurrent_state_text_pair_evidence",
        "schema_version": 1, "status": "valid_functional_state_comparison",
        "timing_evidence_eligible": False, "production_routing_authorized": False,
        "claim": "functional-only comparison of the exact layer1 FP32 recurrent state at frontier 128 and the existing selected-column GDN boundaries",
        "authorities": {"plan": identity(PACKAGE / "plan.json"),
                        "build_provenance": identity(PACKAGE / "build-provenance.json"),
                        "prepared_closure": identity(PACKAGE / "prepared.sha256"),
                        "comparator": identity(COMPARATOR)},
        "reports": reports, "processes": processes, "manifests": manifests,
        "sidecars": sidecars, "comparator_processes": comparator_processes,
        "gdn_comparison": gdn_comparison, "recurrent_state_comparison": state_comparison,
        "limitations": ["diagnostic eager execution and final synchronous D2H make all timings ineligible",
                        "state equality or divergence localizes the recurrence frontier but does not by itself identify an earlier transition",
                        "no production routing or performance claim"],
    }


def write_summary(path: Path) -> None:
    if path != RESULTS / "summary.json" or path.exists() or path.is_symlink():
        fail("summary output path differs or exists")
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


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--summary":
        fail("usage: analyze.py --summary RESULTS/summary.json")
    write_summary(Path(sys.argv[2]))
