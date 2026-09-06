#!/usr/bin/env python3
"""Fail-closed analyzer for the exact 7f110687 layer-boundary diagnostic."""

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
REPAIR = PACKAGE / "analysis-repair.json"
EXE = ROOT / "build-r9700-layer-boundary-43e5e4cc-20260906/bench/ninfer_bench"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
HISTORY = ROOT / "profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
sys.path.insert(0, str(ROOT / "tools/bench"))
from compare_qwen3_layer_boundary_trace import compare  # noqa: E402


def fail(message: str) -> None:
    raise RuntimeError(message)


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict):
        fail(f"not a JSON object: {path}")
    return value


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def exclusive_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
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


def command(stem: str, report: Path) -> list[str]:
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


def validate_output_tokens(stem: str, values: object) -> list[int]:
    if not isinstance(values, list) or any(isinstance(token, bool) or not isinstance(token, int) or
                                           token < 0 or token >= 248077 for token in values):
        fail(f"retained token domain differs: {stem}")
    if values != [96558, 96917]:
        fail(f"retained output sequence differs: {stem}")
    return values


def validate_decode_path(config: dict, expected_spec: bool, stem: str) -> None:
    expected = "dflash_eager" if expected_spec else "eager"
    if config.get("decode_path") != expected:
        fail(f"decode path differs: {stem}")


def validate_trace_input_token(stem: str, value: object) -> int:
    expected = 96558 if stem.startswith("target-") else 24178
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        fail(f"trace selected input token differs: {stem}")
    return value


def validate_speculative(spec: object, expected_spec: bool, stem: str, scope: str) -> None:
    spec_keys = {"enabled", "draft_window", "rounds", "drafted_tokens", "accepted_tokens",
                 "fallback_steps", "acceptance_rate", "acceptance_length",
                 "accepted_per_position"}
    if not isinstance(spec, dict) or set(spec) != spec_keys or \
            spec.get("enabled") is not expected_spec:
        fail(f"speculative report keys differ: {stem}/{scope}")
    if not expected_spec:
        if spec != {"enabled": False, "draft_window": 0, "rounds": 0,
                    "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
                    "acceptance_rate": None, "acceptance_length": None,
                    "accepted_per_position": []}:
            fail(f"ordinary speculative report differs: {stem}/{scope}")
        return
    for key in ("draft_window", "rounds", "drafted_tokens", "accepted_tokens",
                "fallback_steps"):
        if isinstance(spec.get(key), bool) or not isinstance(spec.get(key), int) or spec[key] < 0:
            fail(f"speculative integer differs: {stem}/{scope}/{key}")
    if spec["draft_window"] != 4 or \
            spec["rounds"] + spec["accepted_tokens"] + spec["fallback_steps"] != 1:
        fail(f"speculative accounting differs: {stem}/{scope}")
    if spec["drafted_tokens"] == 0:
        if (spec["rounds"] != 0 or spec["accepted_tokens"] != 0 or
                spec["fallback_steps"] != 1 or spec["acceptance_rate"] is not None or
                spec["acceptance_length"] is not None or
                spec["accepted_per_position"] != [0, 0, 0, 0]):
            fail(f"zero-draft speculative report differs: {stem}/{scope}")
        return
    for key in ("acceptance_rate", "acceptance_length"):
        if isinstance(spec.get(key), bool) or not isinstance(spec.get(key), (int, float)) or \
                not math.isfinite(spec[key]):
            fail(f"speculative rate differs: {stem}/{scope}/{key}")


def validate_analysis_repair() -> dict:
    repair = load(REPAIR)
    expected_keys = {"artifact_type", "schema_version", "status", "cause", "gpu_rerun",
                     "capture_prepared_closure", "capture_result_closure",
                     "capture_analyzer", "repaired_analyzer"}
    if (set(repair) != expected_keys or
            repair.get("artifact_type") != "ninfer_qwen3_layer_boundary_trace_analysis_repair" or
            repair.get("schema_version") != 1 or
            repair.get("status") != "analysis_only_no_gpu_rerun" or
            repair.get("cause") !=
            "the capture analyzer required finite DFlash acceptance metrics when zero drafted tokens correctly serialize undefined rates as null" or
            repair.get("gpu_rerun") is not False or
            repair.get("capture_prepared_closure") != identity(PACKAGE / "prepared.sha256") or
            repair.get("capture_result_closure") != identity(RESULTS / "result.sha256") or
            repair.get("capture_analyzer") != {
                "path": str(PACKAGE / "analyze.py"), "bytes": 19381,
                "sha256": "e04839b3ad51f417a8b5a341f4cac7c481e1353bc479388c18df93244a402c1d"} or
            repair.get("repaired_analyzer") != identity(Path(__file__))):
        fail("analysis-only repair provenance differs")
    return identity(REPAIR)


def validate_process(stem: str, expected_command: list[str], manifest: Path,
                     sidecar: Path) -> dict:
    record = load(RESULTS / f"{stem}.process.json")
    if set(record) != {"command", "trace_environment", "exit_code", "started_unix_ns",
                       "finished_unix_ns", "power_before", "power_after",
                       "instrumentation_environment_present", "stdout", "stderr"}:
        fail(f"process keys differ: {stem}")
    expected_env = {
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": {
            "target-ordinary": "target-ordinary-frontier130",
            "target-dflash": "target-dflash-frontier130-column0",
            "text-fresh": "text-fresh-frontier129-column128",
            "text-append": "text-append-frontier129-column0",
        }[stem],
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST": str(manifest),
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR": str(sidecar),
    }
    if (record.get("command") != expected_command or record.get("trace_environment") != expected_env or
            record.get("exit_code") != 0 or record.get("power_before") != "auto" or
            record.get("power_after") != "auto" or
            record.get("instrumentation_environment_present") != []):
        fail(f"process contract differs: {stem}")
    for stream in ("stdout", "stderr"):
        if record.get(stream) != identity(RESULTS / f"{stem}.{stream}"):
            fail(f"process stream identity differs: {stem}")
    expected_context = 140 if stem == "target-dflash" else 130
    expected_label = "pp128+tg1" if stem == "text-append" else "whole-pp129+tg1"
    expected_stderr = (
        f"[ninfer_bench] loading {ARTIFACT} (max_context={expected_context}, concurrency=1, "
        "kv_format=fp8-k-int4-v)\n"
        f"[ninfer_bench] test 1/1 {expected_label}: warmup=0 reps=1\n"
    )
    if (RESULTS / f"{stem}.stderr").read_text() != expected_stderr:
        fail(f"benchmark stderr differs: {stem}")
    if (RESULTS / f"{stem}.stdout").read_bytes() != (RESULTS / f"{stem}.json").read_bytes():
        fail(f"benchmark JSON stdout differs from retained report: {stem}")
    if not isinstance(record.get("started_unix_ns"), int) or isinstance(record["started_unix_ns"], bool) or \
            not isinstance(record.get("finished_unix_ns"), int) or isinstance(record["finished_unix_ns"], bool) or \
            record["finished_unix_ns"] <= record["started_unix_ns"]:
        fail(f"process interval differs: {stem}")
    return identity(RESULTS / f"{stem}.process.json")


def validate_report(stem: str) -> tuple[dict, list[int]]:
    path = RESULTS / f"{stem}.json"
    report = load(path)
    if set(report) != {"schema_version", "artifact_type", "tool", "command", "environment",
                       "artifact", "load", "memory", "config", "tests"}:
        fail(f"benchmark report keys differ: {stem}")
    if report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or \
            report.get("tool") != "ninfer_bench" or report.get("command") != " ".join(command(stem, path)):
        fail(f"benchmark report identity differs: {stem}")
    environment = report.get("environment")
    if not isinstance(environment, dict) or set(environment) != {"gpu_name", "architecture_name",
            "hip_runtime_version", "hip_driver_version", "device_id"} or \
            environment != {"gpu_name": "AMD Radeon AI PRO R9700", "architecture_name": "gfx1201",
                            "hip_runtime_version": "7.15.26333", "hip_driver_version": "7.15.26333",
                            "device_id": 0}:
        fail(f"benchmark hardware differs: {stem}")
    artifact = report.get("artifact")
    if artifact != {"path": str(ARTIFACT), "file_size_bytes": 22763026944}:
        fail(f"benchmark artifact differs: {stem}")
    config = report.get("config")
    config_keys = {"max_context", "prefill_chunk", "kv_cache_format", "kv_value_group",
                   "kv_plane_layouts", "q4_activation_bits", "q4_prefill_cta_profile",
                   "dflash_small_t_candidate", "dflash_mlp_down_t5_candidate",
                   "w8_activation_bits", "fp8_qk_wmma_enabled", "fp8_qk_wmma_profile",
                   "fp8_qk_wmma_t1_min_context", "fp8_qk_wmma_t2_min_context",
                   "xattention_qualification", "concurrency", "pending_timeout_ms",
                   "pending_deadline", "spec", "draft_tokens", "speculative_execution",
                   "dflash_verify_width_requested", "dflash_verify_width", "proposal_head",
                   "use_device_graph", "retain_token_ids", "isolate_prompt_decode",
                   "decode_path", "decode_graph_prime", "repetitions", "warmup",
                   "corpus_path", "corpus_tokens"}
    expected_spec = stem == "target-dflash"
    expected_corpus = str(CORPUS if stem == "text-append" else HISTORY)
    expected_max_context = 140 if expected_spec else 130
    if (not isinstance(config, dict) or set(config) != config_keys or config.get("concurrency") != 1 or
            config.get("max_context") != expected_max_context or config.get("prefill_chunk") != 4096 or
            config.get("kv_cache_format") != "fp8-k-int4-v" or config.get("kv_value_group") != 16 or
            config.get("kv_plane_layouts") != {"key": "token-fastest-head-major",
                "value": "feature-fastest-page-major", "value_scale": "feature-fastest-page-major"} or
            config.get("q4_activation_bits") != 8 or
            config.get("q4_prefill_cta_profile") != "m64n128-pingpong-n16-k16-scalar-base-production" or
            config.get("w8_activation_bits") != 8 or config.get("fp8_qk_wmma_enabled") is not True or
            config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1" or
            config.get("fp8_qk_wmma_t1_min_context") != 64 or
            config.get("fp8_qk_wmma_t2_min_context") != 320 or
            config.get("xattention_qualification") is not False or
            config.get("pending_timeout_ms") != 4294967295 or
            config.get("pending_deadline") != "unbounded" or
            config.get("use_device_graph") is not False or config.get("retain_token_ids") is not True or
            config.get("corpus_tokens") != (65536 if stem == "text-append" else 129) or
            config.get("corpus_path") != expected_corpus or config.get("repetitions") != 1 or
            config.get("warmup") != 0 or
            config.get("decode_graph_prime") != {"primed": False, "output_tokens": 0} or
            config.get("isolate_prompt_decode") is not (stem == "text-append") or
            config.get("speculative_execution") is not expected_spec or
            config.get("dflash_small_t_candidate") is not False or
            config.get("dflash_mlp_down_t5_candidate") is not False):
        fail(f"benchmark configuration differs: {stem}")
    if expected_spec and (config.get("spec") != "dflash" or config.get("draft_tokens") != 4 or
                          config.get("dflash_verify_width_requested") != 5 or
                          config.get("dflash_verify_width") != 5 or
                          config.get("proposal_head") != "optimized"):
        fail("DFlash K4/W5 configuration differs")
    if not expected_spec and (config.get("spec") != "none" or config.get("draft_tokens") != 0 or
                              config.get("dflash_verify_width_requested") != 0 or
                              config.get("dflash_verify_width") != 0 or
                              config.get("proposal_head") != "full"):
        fail(f"ordinary decode configuration differs: {stem}")
    memory = report.get("memory")
    if (not isinstance(memory, dict) or memory.get("max_context") != expected_max_context or
            memory.get("device") != 0 or memory.get("kv_cache_format") != "fp8-k-int4-v" or
            isinstance(memory.get("planned_slack_bytes"), bool) or
            not isinstance(memory.get("planned_slack_bytes"), int) or
            memory["planned_slack_bytes"] < 1320972 or
            isinstance(memory.get("available_after_startup_bytes"), bool) or
            not isinstance(memory.get("available_after_startup_bytes"), int) or
            memory["available_after_startup_bytes"] < 1320972):
        fail(f"trace device-staging headroom differs: {stem}")
    validate_decode_path(config, expected_spec, stem)
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        fail(f"benchmark test count differs: {stem}")
    test = tests[0]
    test_keys = {"label", "kind", "n_prompt", "n_gen", "requested_output_tokens",
                 "prefill_tok_s_mean", "prefill_tok_s_stddev", "decode_output_tok_s_mean",
                 "decode_output_tok_s_stddev", "decode_engine_tok_s_mean",
                 "decode_engine_tok_s_stddev", "whole_output_tok_s_mean",
                 "whole_output_tok_s_stddev", "prepare_seconds_mean", "prepare_seconds_stddev",
                 "prefill_seconds_mean", "prefill_seconds_stddev", "decode_seconds_mean",
                 "decode_seconds_stddev", "total_seconds_mean", "total_seconds_stddev",
                 "workspace_peak_bytes", "workspace_allocator_peak_bytes", "speculative", "reps"}
    if not isinstance(test, dict) or set(test) != test_keys:
        fail(f"benchmark test keys differ: {stem}")
    label = "pp128+tg1" if stem == "text-append" else "whole-pp129+tg1"
    if test.get("label") != label or test.get("n_prompt") != (128 if stem == "text-append" else 129) or \
            test.get("n_gen") != 1 or test.get("requested_output_tokens") != 2:
        fail(f"benchmark test geometry differs: {stem}")
    reps = test.get("reps")
    rep = reps[0] if isinstance(reps, list) and len(reps) == 1 else None
    if not isinstance(rep, dict) or set(rep) != {"generated_output_tokens", "decode_output_tokens",
            "decode_engine_tokens", "generated_token_ids_by_lane", "timings", "speculative"}:
        fail(f"benchmark repetition keys differ: {stem}")
    lanes = rep.get("generated_token_ids_by_lane")
    if not isinstance(lanes, list) or len(lanes) != 1 or not isinstance(lanes[0], list):
        fail(f"retained tokens differ: {stem}")
    expected_tokens = 2
    if len(lanes[0]) != expected_tokens:
        fail(f"retained token count differs: {stem}")
    for label_spec, spec in (("test", test.get("speculative")), ("rep", rep.get("speculative"))):
        validate_speculative(spec, expected_spec, stem, label_spec)
    return identity(path), validate_output_tokens(stem, lanes[0])


def validate_comparator(label: str, left: str, right: str, expected: dict) -> dict:
    output = RESULTS / f"{label}-comparison.json"
    stdout = RESULTS / f"{label}-comparator.stdout"
    stderr = RESULTS / f"{label}-comparator.stderr"
    process_path = RESULTS / f"{label}-comparator.process.json"
    command_line = ["/usr/bin/python3", str(ROOT / "tools/bench/compare_qwen3_layer_boundary_trace.py"),
                    "--left", str(RESULTS / f"{left}.trace.json"), "--right",
                    str(RESULTS / f"{right}.trace.json"), "--diagnostic", label, "--out", str(output)]
    process = load(process_path)
    if (set(process) != {"command", "exit_code", "stdout", "stderr"} or
            process.get("command") != command_line or process.get("exit_code") != 0 or
            process.get("stdout") != identity(stdout) or
            process.get("stderr") != identity(stderr) or stdout.read_bytes() or
            stderr.read_bytes() or load(output) != expected):
        fail(f"retained comparator differs: {label}")
    return identity(process_path)


def analyze(summary_path: Path) -> dict:
    if summary_path != RESULTS / "summary.json" or summary_path.exists() or summary_path.is_symlink():
        fail("summary output must be the exact fresh path")
    analysis_repair = validate_analysis_repair()
    reports = {}
    processes = {}
    tokens = {}
    manifests = {}
    sidecars = {}
    for stem in ("target-ordinary", "target-dflash", "text-fresh", "text-append"):
        manifest = RESULTS / f"{stem}.trace.json"
        sidecar = RESULTS / f"{stem}.trace.bin"
        report = RESULTS / f"{stem}.json"
        processes[stem] = validate_process(stem, command(stem, report), manifest, sidecar)
        reports[stem], tokens[stem] = validate_report(stem)
        manifests[stem], sidecars[stem] = identity(manifest), identity(sidecar)
    target = compare(RESULTS / "target-ordinary.trace.json",
                     RESULTS / "target-dflash.trace.json", "target")
    text = compare(RESULTS / "text-fresh.trace.json", RESULTS / "text-append.trace.json", "text")
    for stem in manifests:
        validate_trace_input_token(stem, load(RESULTS / f"{stem}.trace.json").get("token"))
    comparator_processes = {
        "target": validate_comparator("target", "target-ordinary", "target-dflash", target),
        "text": validate_comparator("text", "text-fresh", "text-append", text),
    }
    return {
        "artifact_type": "ninfer_qwen3_layer_boundary_trace_evidence",
        "schema_version": 1,
        "status": "valid_functional_localization",
        "timing_evidence_eligible": False,
        "production_routing_authorized": False,
        "source_commit": "43e5e4ccb3212aa43717104aa5f45ef717ad97b8",
        "analysis_repair": analysis_repair,
        "reports": reports,
        "processes": processes,
        "manifests": manifests,
        "sidecars": sidecars,
        "comparator_processes": comparator_processes,
        "target_comparison": target,
        "text_comparison": text,
        "limitations": [
            "synchronous trace transfer invalidates all timings",
            "first differing represented-BF16 boundary localizes a layer half, not a primitive root cause",
            "does not authorize production routing or performance claims",
        ],
    }


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--summary":
        fail("usage differs")
    output = Path(sys.argv[2])
    value = analyze(output)
    exclusive_json(output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
