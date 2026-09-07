#!/usr/bin/env python3
"""Fail-closed analyzer for the C1/P129+G27 DFlash K4/W5 after-parity screen."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
RESULTS = PACKAGE / "results"
EXE = ROOT / "build-r9700-attention-whole-parity-6fe53d53-fresh-20260906/bench/ninfer_bench"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
ARMS = ("base-decode", "dflash-k4w5")
TOKEN_DOMAIN = 248077
BASE_DECODE_TOK_S = 27.05729956
BREAK_EVEN_MS_PER_ROUND = 57.6915
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


def command(stem: str, report: Path) -> list[str]:
    result = [str(EXE), "--weights", str(ARTIFACT), "--corpus", str(CORPUS),
              "--device", "0", "--concurrency", "1", "--whole-pg", "129,27",
              "--prefill-chunk", "4096", "--kv-capacity", "workload",
              "--retain-token-ids", "--no-device-graph", "--output", "json",
              "--output-file", str(report), "-r", "3", "--warmup", "1"]
    if stem == "base-decode":
        result += ["--spec", "mtp", "--draft-tokens", "0"]
    else:
        result += ["--spec", "dflash", "--draft-tokens", "4", "--dflash-verify-width", "5"]
    return result


def validate_plan() -> dict:
    plan = load(PACKAGE / "plan.json")
    build = load(PACKAGE / "build-provenance.json")
    if (plan.get("artifact_type") != "ninfer_r9700_dflash_k4w5_after_parity_screen_plan" or
            plan.get("schema_version") != 1 or
            plan.get("status") != "prepared_cpu_only_no_gpu_execution" or
            plan.get("production_routing_authorized") is not False or
            plan.get("production_recipe_selected") is not False or
            plan.get("build_receipt") != "build-provenance.json"):
        fail("plan contract differs")
    if plan.get("source") != {"commit": "6fe53d531072dfd6069b951b64006603ef976ad9",
                              "tree": "b711ff3e93f6cf3249456427401152084c66b939",
                              "worktree":
                                  "/ssdpool2nvme/local_llm/ninfer-amd-r9700-combined-selector-src-6fe53d53"}:
        fail("plan source authority differs")
    if not exact(plan["artifact"]):
        fail("plan artifact identity differs")
    if not exact(plan["corpus"]):
        fail("plan corpus identity differs")
    if plan.get("workload") != {"prompt_tokens": 129, "generation_tokens": 27,
                                 "whole_pg": "129,27",
                                 "base_decode": {"spec": "mtp", "draft_tokens": 0,
                                                  "role": "base-decode-c1-p129-g27"},
                                 "dflash_k4w5": {"spec": "dflash", "draft_tokens": 4,
                                                  "verify_width": 5,
                                                  "role": "dflash-k4w5-c1-p129-g27"}}:
        fail("plan workload differs")
    if plan.get("timing", {}).get("base_decode_tok_s") != BASE_DECODE_TOK_S or \
            plan.get("timing", {}).get("break_even_bound_ms_per_round") != BREAK_EVEN_MS_PER_ROUND:
        fail("plan timing bounds differ")
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
    config = report.get("config")
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        fail(f"benchmark config keys differ: {stem}")
    dflash = stem == "dflash-k4w5"
    expected_config = {
        "max_context": 166 if dflash else 156, "prefill_chunk": 4096, "kv_cache_format": "fp8-k-int4-v",
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
        "pending_timeout_ms": 4294967295, "pending_deadline": "unbounded",
        "spec": "dflash" if dflash else "none",
        "draft_tokens": 4 if dflash else 0,
        "speculative_execution": dflash,
        "dflash_verify_width_requested": 5 if dflash else 0,
        "dflash_verify_width": 5 if dflash else 0,
        "proposal_head": "full", "use_device_graph": False, "retain_token_ids": True,
        "isolate_prompt_decode": False, "decode_path": "dflash_eager" if dflash else "eager",
        "decode_graph_prime": {"primed": False, "output_tokens": 0}, "repetitions": 3,
        "warmup": 1, "corpus_path": str(CORPUS), "corpus_tokens": 65536,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            fail(f"benchmark config differs: {stem}/{key}")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        fail(f"benchmark test count differs: {stem}")
    test = tests[0]
    if (test.get("label") != "whole-pp129+tg27" or test.get("n_prompt") != 129 or
            test.get("n_gen") != 27 or test.get("requested_output_tokens") != 28):
        fail(f"benchmark test geometry differs: {stem}")
    reps = test.get("reps")
    if not isinstance(reps, list) or len(reps) != 3:
        fail(f"benchmark repetition count differs: {stem}")
    for i, rep in enumerate(reps):
        lanes = rep.get("generated_token_ids_by_lane")
        if not isinstance(lanes, list) or len(lanes) != 1 or not isinstance(lanes[0], list) \
                or len(lanes[0]) != 28:
            fail(f"retained token sequence shape differs: {stem}/{i}")
        for value in lanes[0]:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < TOKEN_DOMAIN:
                fail(f"retained token domain differs: {stem}/{i}")
        spec = rep.get("speculative")
        if not isinstance(spec, dict):
            fail(f"speculative report missing: {stem}/{i}")
        if dflash:
            if spec.get("enabled") is not True or spec.get("draft_window") != 4:
                fail(f"DFlash speculative config differs: {stem}/{i}")
            for field in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps",
                          "accepted_per_position"):
                if field not in spec:
                    fail(f"DFlash speculative field missing: {stem}/{i}/{field}")
        else:
            if spec.get("enabled") is not False or spec.get("draft_window") != 0:
                fail(f"base decode speculative config differs: {stem}/{i}")
    # Cross-repetition token stability
    all_tokens = [rep["generated_token_ids_by_lane"][0] for rep in reps]
    if len(set(tuple(t) for t in all_tokens)) != 1:
        fail(f"tokens differ across repetitions: {stem}")
    return identity(path)


def validate_process(stem: str) -> dict:
    record = load(RESULTS / f"{stem}.process.json")
    if set(record) != {"command", "exit_code", "started_unix_ns", "finished_unix_ns",
                       "power_before", "power_after", "instrumentation_environment_present",
                       "stdout", "stderr"}:
        fail(f"process keys differ: {stem}")
    if (record.get("command") != command(stem, RESULTS / f"{stem}.json") or
            record.get("exit_code") != 0 or record.get("power_before") != "auto" or
            record.get("power_after") != "auto" or
            record.get("instrumentation_environment_present") != []):
        fail(f"process contract differs: {stem}")
    for stream in ("stdout", "stderr"):
        if record.get(stream) != identity(RESULTS / f"{stem}.{stream}"):
            fail(f"process stream identity differs: {stem}")
    if (RESULTS / f"{stem}.stdout").read_text() != f"wrote {RESULTS / f'{stem}.json'}\n":
        fail(f"benchmark stdout differs: {stem}")
    if (not isinstance(record.get("started_unix_ns"), int) or
            isinstance(record["started_unix_ns"], bool) or
            not isinstance(record.get("finished_unix_ns"), int) or
            isinstance(record["finished_unix_ns"], bool) or
            record["finished_unix_ns"] <= record["started_unix_ns"]):
        fail(f"process interval differs: {stem}")
    return identity(RESULTS / f"{stem}.process.json")


def timing_summary(stem: str) -> dict:
    report = load(RESULTS / f"{stem}.json")
    test = report["tests"][0]
    return {
        "total_seconds_mean": test.get("total_seconds_mean"),
        "total_seconds_stddev": test.get("total_seconds_stddev"),
        "whole_output_tok_s_mean": test.get("whole_output_tok_s_mean"),
        "whole_output_tok_s_stddev": test.get("whole_output_tok_s_stddev"),
        "decode_output_tok_s_mean": test.get("decode_output_tok_s_mean"),
        "prefill_tok_s_mean": test.get("prefill_tok_s_mean"),
    }


def analyze(summary_path: Path) -> dict:
    if summary_path != RESULTS / "summary.json" or summary_path.exists() or \
            summary_path.is_symlink():
        fail("summary output must be the exact fresh path")
    validate_plan()
    validate_build()
    reports = {}
    processes = {}
    for stem in ARMS:
        processes[stem] = validate_process(stem)
        reports[stem] = validate_report(stem)
    base = timing_summary("base-decode")
    dflash = timing_summary("dflash-k4w5")
    base_total = base["total_seconds_mean"]
    dflash_total = dflash["total_seconds_mean"]
    if base_total is None or dflash_total is None or base_total <= 0 or dflash_total <= 0:
        fail("timing means missing or non-positive")
    ratio = dflash_total / base_total
    win = dflash_total < base_total
    material = (base_total - dflash_total) / base_total >= 0.10
    return {
        "artifact_type": "ninfer_r9700_dflash_k4w5_after_parity_screen_evidence",
        "schema_version": 1,
        "status": "valid_directional_screen",
        "timing_evidence_eligible": True,
        "production_routing_authorized": False,
        "production_recipe_selected": False,
        "source_commit": "6fe53d531072dfd6069b951b64006603ef976ad9",
        "base_decode": base,
        "dflash_k4w5": dflash,
        "base_decode_total_seconds_mean": base_total,
        "dflash_k4w5_total_seconds_mean": dflash_total,
        "dflash_over_base_ratio": ratio,
        "dflash_wins": win,
        "material_win": material,
        "base_decode_tok_s_reference": BASE_DECODE_TOK_S,
        "break_even_ms_per_round": BREAK_EVEN_MS_PER_ROUND,
        "reports": reports,
        "processes": processes,
        "limitations": [
            "directional screen only; not production admission",
            "C1/P129+G27 only; no C2..4, longer generation, capacity, chunk, or profiling",
            "eager decode (no device graph); order-stability required for the win claim",
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
