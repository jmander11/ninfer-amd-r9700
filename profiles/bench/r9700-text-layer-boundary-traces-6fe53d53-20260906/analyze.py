#!/usr/bin/env python3
"""Fail-closed analyzer for the exact two-arm Text append-versus-fresh layer-boundary diagnostic."""

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
EXE = ROOT / "build-r9700-attention-whole-parity-6fe53d53-fresh-20260906/bench/ninfer_bench"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
HISTORY = ROOT / "profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
COMPARATOR = ROOT / "tools/bench/compare_qwen3_layer_boundary_trace.py"
sys.path.insert(0, str(ROOT / "tools/bench"))
from compare_qwen3_layer_boundary_trace import compare  # noqa: E402

ARMS = ("text-fresh", "text-append")
TOKEN_DOMAIN = 248077
RETAINED_TOKENS = [96558, 96917]
DEVICE_STAGING_BYTES = 1320972
CONFIG_KEYS = {
    "max_context", "prefill_chunk", "kv_cache_format", "kv_value_group", "kv_plane_layouts",
    "q4_activation_bits", "q4_prefill_cta_profile", "dflash_small_t_candidate",
    "dflash_mlp_down_t5_candidate", "dflash_rmsnorm_rows56_candidate",
    "attention_parity_candidate", "w8_activation_bits", "fp8_qk_wmma_enabled",
    "fp8_qk_wmma_profile", "fp8_qk_wmma_t1_min_context", "fp8_qk_wmma_t2_min_context",
    "xattention_qualification", "concurrency", "pending_timeout_ms", "pending_deadline",
    "spec", "draft_tokens", "speculative_execution", "dflash_verify_width_requested",
    "dflash_verify_width", "proposal_head", "use_device_graph", "retain_token_ids",
    "isolate_prompt_decode", "decode_path", "decode_graph_prime", "repetitions", "warmup",
    "corpus_path", "corpus_tokens",
}


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


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest(resolved)}


def exact(record: dict) -> bool:
    actual = identity(Path(record["path"]))
    return (actual["path"] == record.get("path") and
            actual["bytes"] == record.get("bytes") and
            actual["sha256"] == record.get("sha256"))


def validate_closure(path: Path) -> None:
    path = path.resolve(strict=True)
    seen = set()
    count = 0
    for line in path.read_text().splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64 or \
                any(c not in "0123456789abcdef" for c in fields[0]):
            fail(f"malformed closure: {path}")
        member = Path(fields[1])
        if not member.is_absolute():
            rooted = ROOT / member
            member = rooted if (rooted.exists() or rooted.is_symlink()) else path.parent / member
        member = member.resolve(strict=True)
        if member in seen or digest(member) != fields[0]:
            fail(f"closure member mismatch: {member}")
        seen.add(member)
        count += 1
    if count == 0:
        fail(f"empty closure: {path}")


def command(stem: str, report: Path) -> list[str]:
    corpus = CORPUS if stem == "text-append" else HISTORY
    result = [str(EXE), "--weights", str(ARTIFACT), "--corpus", str(corpus), "--device", "0",
              "--concurrency", "1"]
    result += (["-pg", "128,1"] if stem == "text-append" else ["--whole-pg", "129,1"])
    result += ["--prefill-chunk", "4096", "--kv-capacity", "workload", "--draft-tokens", "0",
               "--retain-token-ids"]
    if stem == "text-append":
        result += ["--isolate-prompt-decode"]
    result += ["--no-device-graph", "--output", "json", "--output-file", str(report),
               "-r", "1", "--warmup", "0"]
    return result


def trace_environment(stem: str) -> dict:
    role = {"text-fresh": "text-fresh-frontier129-column128",
            "text-append": "text-append-frontier129-column0"}[stem]
    return {
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": role,
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST": str(RESULTS / f"{stem}.trace.json"),
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR": str(RESULTS / f"{stem}.trace.bin"),
    }


def validate_plan() -> dict:
    plan = load(PACKAGE / "plan.json")
    build = load(PACKAGE / "build-provenance.json")
    if (plan.get("artifact_type") != "ninfer_qwen3_text_layer_boundary_trace_plan" or
            plan.get("schema_version") != 1 or
            plan.get("status") != "prepared_cpu_only_no_gpu_execution" or
            plan.get("claim") !=
            "eager-only functional localization of the first current Text append-versus-fresh "
            "divergence on the retained combined-selector build; all captured timings are "
            "ineligible" or
            plan.get("production_routing_authorized") is not False or
            plan.get("build_receipt") != "build-provenance.json" or
            plan.get("common") != {"concurrency": 1, "maximum_product_concurrency": 4,
                                   "prefill_chunk": 4096, "kv_capacity": "workload",
                                   "repetitions": 1, "warmup": 0, "device_graph": False,
                                   "retain_token_ids": True} or
            plan.get("text_pair") != {
                "absolute_frontier": 129,
                "selected_input_token": 24178,
                "fresh": {"role": "text-fresh-frontier129-column128", "width": 129,
                          "selected_column": 128, "shape": "base0/T129"},
                "append": {"role": "text-append-frontier129-column0", "width": 1,
                           "selected_column": 0, "shape": "base128/T1 isolated"}} or
            plan.get("trace") != {"hidden": 5120, "layers": 64, "snapshots": 129,
                                  "bytes_per_snapshot": 10240, "bytes_per_sidecar": 1320960,
                                  "device_staging_bytes": DEVICE_STAGING_BYTES,
                                  "prior_gate_available_after_startup_bytes": 8992587776,
                                  "prior_gate_headroom_multiple": 6807,
                                  "runtime_requirement":
                                      "reported planned slack and available-after-startup each "
                                      "cover device staging bytes",
                                  "one_staging_slab_per_process": True,
                                  "layout": "input then each layer post_mixer/post_mlp",
                                  "one_final_d2h": True} or
            plan.get("limitations") != [
                "functional localization only; synchronous final D2H invalidates timing",
                "first differing represented-BF16 boundary localizes a layer half, not a "
                "primitive root cause",
                "no production routing or performance claim"] or
            plan.get("hardware") != {"device": 0, "name": "AMD Radeon AI PRO R9700",
                                     "architecture": "gfx1201", "wavefront_width": 32,
                                     "pci_bus": "0000:13:00.0",
                                     "pci_vendor_device": "1002:7551",
                                     "power_profile": "auto"} or
            plan.get("results_directory") != "results"):
        fail("plan contract differs")
    if plan.get("source") != {"commit": "6fe53d531072dfd6069b951b64006603ef976ad9",
                              "tree": "b711ff3e93f6cf3249456427401152084c66b939",
                              "worktree":
                                  "/ssdpool2nvme/local_llm/ninfer-amd-r9700-combined-selector-src-6fe53d53"}:
        fail("plan source authority differs")
    if not exact(plan["artifact"]):
        fail("plan artifact identity differs")
    for key in ("source_corpus", "p129_fixture"):
        if not exact(plan["history"][key]):
            fail(f"history identity differs: {key}")
    if (set(plan["history"]) != {"source_corpus", "p129_fixture", "seed_token", "rule"} or
            plan["history"]["seed_token"] != 24178 or
            plan["history"]["rule"] !=
            "source corpus tokens [0,128) followed by retained fresh-P128 ordinary greedy seed"):
        fail("plan history authority differs")
    for key, record in plan["prior_authorities"].items():
        if not exact(record):
            fail(f"prior authority identity differs: {key}")
        if key == "combined_gate_capture":
            validate_closure(Path(record["path"]))
    source_tokens = [int(t) for t in Path(plan["history"]["source_corpus"]["path"]).read_text().split()]
    history = [int(t) for t in Path(plan["history"]["p129_fixture"]["path"]).read_text().split()]
    if len(history) != 129 or history[:128] != source_tokens[:128] or history[128] != 24178:
        fail("P129 history/seed derivation differs")
    return plan


def validate_build() -> dict:
    build = load(PACKAGE / "build-provenance.json")
    if (build.get("artifact_type") != "ninfer_retained_exact_committed_source_build_receipt" or
            build.get("schema_version") != 1):
        fail("build receipt type differs")
    if not exact(build["retained_receipt"]):
        fail("retained build receipt identity differs")
    if build.get("source") != {"commit": "6fe53d531072dfd6069b951b64006603ef976ad9",
                               "tree": "b711ff3e93f6cf3249456427401152084c66b939",
                               "worktree":
                                   "/ssdpool2nvme/local_llm/ninfer-amd-r9700-combined-selector-src-6fe53d53"}:
        fail("build source differs")
    if build.get("build_directory") != str(
            ROOT / "build-r9700-attention-whole-parity-6fe53d53-fresh-20260906"):
        fail("build directory differs")
    expected_profile = {"CMAKE_BUILD_TYPE": "Release", "GPU_BUILD_TARGETS": "gfx1201",
        "NINFER_R9700_KV_VALUE_GROUP": "16", "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
        "NINFER_R9700_W8_ACTIVATION_BITS": "8", "NINFER_R9700_FP8_QK_WMMA": "1",
        "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE": "0",
        "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE": "1",
        "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE": "1",
        "NINFER_R9700_ATTENTION_PARITY_CANDIDATE": "1",
        "NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE": "1",
        "NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE": "1",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF"}
    if build.get("profile") != expected_profile:
        fail("build profile differs")
    for group in ("outputs", "toolchain"):
        for key, record in build[group].items():
            if not exact(record):
                fail(f"build {group} identity differs: {key}")
    return build


def validate_report(stem: str) -> dict:
    path = RESULTS / f"{stem}.json"
    report = load(path)
    if set(report) != {"schema_version", "artifact_type", "tool", "command", "environment",
                       "artifact", "load", "memory", "config", "tests"}:
        fail(f"benchmark report keys differ: {stem}")
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or
            report.get("command") != " ".join(command(stem, path))):
        fail(f"benchmark report identity differs: {stem}")
    environment = report.get("environment")
    if environment != {"gpu_name": "AMD Radeon AI PRO R9700", "architecture_name": "gfx1201",
                       "hip_runtime_version": "7.15.26333", "hip_driver_version": "7.15.26333",
                       "device_id": 0}:
        fail(f"benchmark hardware differs: {stem}")
    if report.get("artifact") != {"path": str(ARTIFACT), "file_size_bytes": 22763026944}:
        fail(f"benchmark artifact differs: {stem}")
    config = report.get("config")
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        fail(f"benchmark config keys differ: {stem}")
    append = stem == "text-append"
    expected_config = {
        "max_context": 130, "prefill_chunk": 4096, "kv_cache_format": "fp8-k-int4-v",
        "kv_value_group": 16,
        "kv_plane_layouts": {"key": "token-fastest-head-major",
                             "value": "feature-fastest-page-major",
                             "value_scale": "feature-fastest-page-major"},
        "q4_activation_bits": 8,
        "q4_prefill_cta_profile": "m64n128-pingpong-n16-k16-scalar-base-production",
        "dflash_small_t_candidate": False, "dflash_mlp_down_t5_candidate": True,
        "dflash_rmsnorm_rows56_candidate": True, "attention_parity_candidate": True,
        "w8_activation_bits": 8, "fp8_qk_wmma_enabled": True,
        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
        "fp8_qk_wmma_t1_min_context": 64, "fp8_qk_wmma_t2_min_context": 320,
        "xattention_qualification": False, "concurrency": 1,
        "pending_timeout_ms": 4294967295, "pending_deadline": "unbounded", "spec": "none",
        "draft_tokens": 0, "speculative_execution": False,
        "dflash_verify_width_requested": 0, "dflash_verify_width": 0,
        "proposal_head": "full", "use_device_graph": False, "retain_token_ids": True,
        "isolate_prompt_decode": append, "decode_path": "eager",
        "decode_graph_prime": {"primed": False, "output_tokens": 0}, "repetitions": 1,
        "warmup": 0, "corpus_path": str(CORPUS if append else HISTORY),
        "corpus_tokens": 65536 if append else 129,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            fail(f"benchmark config differs: {stem}/{key}")
    memory = report.get("memory")
    if (not isinstance(memory, dict) or memory.get("max_context") != 130 or
            memory.get("device") != 0 or memory.get("kv_cache_format") != "fp8-k-int4-v" or
            isinstance(memory.get("planned_slack_bytes"), bool) or
            not isinstance(memory.get("planned_slack_bytes"), int) or
            memory["planned_slack_bytes"] < DEVICE_STAGING_BYTES or
            isinstance(memory.get("available_after_startup_bytes"), bool) or
            not isinstance(memory.get("available_after_startup_bytes"), int) or
            memory["available_after_startup_bytes"] < DEVICE_STAGING_BYTES):
        fail(f"trace device-staging headroom differs: {stem}")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        fail(f"benchmark test count differs: {stem}")
    test = tests[0]
    label = "pp128+tg1" if append else "whole-pp129+tg1"
    if (test.get("label") != label or
            test.get("n_prompt") != (128 if append else 129) or test.get("n_gen") != 1 or
            test.get("requested_output_tokens") != 2):
        fail(f"benchmark test geometry differs: {stem}")
    reps = test.get("reps")
    rep = reps[0] if isinstance(reps, list) and len(reps) == 1 else None
    if not isinstance(rep, dict):
        fail(f"benchmark repetition differs: {stem}")
    lanes = rep.get("generated_token_ids_by_lane")
    if not isinstance(lanes, list) or len(lanes) != 1 or not isinstance(lanes[0], list) \
            or len(lanes[0]) != 2:
        fail(f"retained token sequence shape differs: {stem}")
    for value in lanes[0]:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < TOKEN_DOMAIN:
            fail(f"retained token domain differs: {stem}")
    # The fresh arm is the reference: it must match the retained P129 ordinary control.
    # The append arm is the challenger and may legitimately diverge (that is the diagnostic).
    if not append and lanes[0] != RETAINED_TOKENS:
        fail(f"fresh arm diverges from retained P129 control: {stem}")
    spec = rep.get("speculative")
    if spec != {"enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
                "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
                "acceptance_length": None, "accepted_per_position": []}:
        fail(f"ordinary speculative report differs: {stem}")
    if test.get("speculative") != spec:
        fail(f"aggregate speculative report differs: {stem}")
    return identity(path)


def validate_process(stem: str) -> dict:
    record = load(RESULTS / f"{stem}.process.json")
    if set(record) != {"command", "trace_environment", "exit_code", "started_unix_ns",
                       "finished_unix_ns", "power_before", "power_after",
                       "instrumentation_environment_present", "stdout", "stderr"}:
        fail(f"process keys differ: {stem}")
    if (record.get("command") != command(stem, RESULTS / f"{stem}.json") or
            record.get("trace_environment") != trace_environment(stem) or
            record.get("exit_code") != 0 or record.get("power_before") != "auto" or
            record.get("power_after") != "auto" or
            record.get("instrumentation_environment_present") != []):
        fail(f"process contract differs: {stem}")
    for stream in ("stdout", "stderr"):
        if record.get(stream) != identity(RESULTS / f"{stem}.{stream}"):
            fail(f"process stream identity differs: {stem}")
    if (RESULTS / f"{stem}.stdout").read_text() != f"wrote {RESULTS / f'{stem}.json'}\n":
        fail(f"benchmark stdout differs: {stem}")
    label = "pp128+tg1" if stem == "text-append" else "whole-pp129+tg1"
    expected_stderr = (
        f"[ninfer_bench] loading {ARTIFACT} (max_context=130, concurrency=1, "
        f"kv_format=fp8-k-int4-v)\n[ninfer_bench] test 1/1 {label}: warmup=0 reps=1\n")
    if (RESULTS / f"{stem}.stderr").read_text() != expected_stderr:
        fail(f"benchmark stderr differs: {stem}")
    if (not isinstance(record.get("started_unix_ns"), int) or
            isinstance(record["started_unix_ns"], bool) or
            not isinstance(record.get("finished_unix_ns"), int) or
            isinstance(record["finished_unix_ns"], bool) or
            record["finished_unix_ns"] <= record["started_unix_ns"]):
        fail(f"process interval differs: {stem}")
    return identity(RESULTS / f"{stem}.process.json")


def validate_trace(stem: str) -> tuple[dict, dict]:
    manifest = RESULTS / f"{stem}.trace.json"
    sidecar = RESULTS / f"{stem}.trace.bin"
    value = load(manifest)
    expected_role = {"text-fresh": "text-fresh-frontier129-column128",
                     "text-append": "text-append-frontier129-column0"}[stem]
    expected_width = 129 if stem == "text-fresh" else 1
    expected_column = 128 if stem == "text-fresh" else 0
    if (value.get("artifact_type") != "ninfer_qwen3_layer_boundary_trace" or
            value.get("schema_version") != 1 or value.get("role") != expected_role or
            value.get("width") != expected_width or value.get("selected_column") != expected_column or
            value.get("absolute_frontier") != 129 or value.get("token") != 24178 or
            value.get("cache_position") != 128 or value.get("rope_position") != 128 or
            value.get("hidden") != 5120 or value.get("layers") != 64 or
            value.get("snapshot_count") != 129 or value.get("snapshot_bytes") != 10240 or
            value.get("sidecar_bytes") != 1320960 or
            value.get("sidecar_path") != str(sidecar) or
            value.get("layout") != "little-endian-u16: input, then layer0..63 post_mixer,post_mlp"):
        fail(f"trace manifest differs: {stem}")
    return identity(manifest), identity(sidecar)


def validate_comparator(expected: dict) -> dict:
    output = RESULTS / "text-comparison.json"
    stdout = RESULTS / "text-comparator.stdout"
    stderr = RESULTS / "text-comparator.stderr"
    process_path = RESULTS / "text-comparator.process.json"
    command_line = ["/usr/bin/python3", str(COMPARATOR), "--left",
                    str(RESULTS / "text-fresh.trace.json"), "--right",
                    str(RESULTS / "text-append.trace.json"), "--diagnostic", "text",
                    "--out", str(output)]
    process = load(process_path)
    if (set(process) != {"command", "exit_code", "stdout", "stderr"} or
            process.get("command") != command_line or process.get("exit_code") != 0 or
            process.get("stdout") != identity(stdout) or
            process.get("stderr") != identity(stderr) or stdout.read_bytes() or
            stderr.read_bytes() or load(output) != expected):
        fail("retained comparator differs")
    return identity(process_path)


def token_sequence(stem: str) -> list:
    report = load(RESULTS / f"{stem}.json")
    return report["tests"][0]["reps"][0]["generated_token_ids_by_lane"][0]


def analyze(summary_path: Path) -> dict:
    if summary_path != RESULTS / "summary.json" or summary_path.exists() or \
            summary_path.is_symlink():
        fail("summary output must be the exact fresh path")
    validate_plan()
    validate_build()
    reports = {}
    processes = {}
    manifests = {}
    sidecars = {}
    for stem in ARMS:
        processes[stem] = validate_process(stem)
        reports[stem] = validate_report(stem)
        manifests[stem], sidecars[stem] = validate_trace(stem)
    fresh_tokens = token_sequence("text-fresh")
    append_tokens = token_sequence("text-append")
    text = compare(RESULTS / "text-fresh.trace.json", RESULTS / "text-append.trace.json", "text")
    comparator_process = validate_comparator(text)
    parity_closed = fresh_tokens == append_tokens
    return {
        "artifact_type": "ninfer_qwen3_text_layer_boundary_trace_evidence",
        "schema_version": 1,
        "status": "valid_functional_localization",
        "timing_evidence_eligible": False,
        "production_routing_authorized": False,
        "source_commit": "6fe53d531072dfd6069b951b64006603ef976ad9",
        "retained_p129_ordinary_control": RETAINED_TOKENS,
        "fresh_tokens": fresh_tokens,
        "append_tokens": append_tokens,
        "parity_closed": parity_closed,
        "reports": reports,
        "processes": processes,
        "manifests": manifests,
        "sidecars": sidecars,
        "comparator_process": comparator_process,
        "text_comparison": text,
        "limitations": [
            "synchronous trace transfer invalidates all timings",
            "first differing represented-BF16 boundary localizes a layer half, not a primitive "
            "root cause",
            "does not authorize production routing or performance claims",
        ],
    }


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--summary":
        fail("usage differs")
    output = Path(sys.argv[2])
    value = analyze(output)
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
