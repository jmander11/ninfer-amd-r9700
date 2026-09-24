#!/usr/bin/env python3
"""Fail-closed analysis of the retained layer-0 recurrent-state capture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
RESULTS = PACKAGE / "results"
SOURCE_COMMIT = "fda2d17ad115bd9857ea848f5f068b0d67ddc100"
SOURCE_TREE = "b1e8b47ba90f8a3b1d476e163df97a3d5dc225ef"
EXE_SHA = "f36b75bf00daad9dcf3228f3ae4df553b4809ba123f1e33e5656c2472a0f315f"
STATE_SHA = "0590bf20498136400478b69595c0703fa47c25a1fac77d61647fe60b1756b0bc"
TOKENS = [96558, 96917]


def fail(message: str) -> None:
    raise RuntimeError(message)


def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            fail(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load(path: Path) -> dict:
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
    return {"path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": digest(resolved)}


def validate_plan() -> tuple[dict, dict, dict]:
    plan = load(PACKAGE / "plan.json")
    build = load(PACKAGE / "build-provenance.json")
    capture = load(PACKAGE / "capture-provenance.json")
    if (plan.get("artifact_type") != "ninfer_qwen3_layer0_recurrent_state_text_pair_plan" or
            plan.get("schema_version") != 1 or
            plan.get("status") != "completed_imported_exact_capture" or
            plan.get("production_routing_authorized") is not False or
            plan.get("timing_evidence_eligible") is not False or
            plan.get("source") != {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE} or
            plan.get("trace") != {"schema_version": 2, "text_layer": 0, "gdn_index": 0,
                "state_frontier": 128, "linear_state_slot": 0, "shape": [128,128,48],
                "dtype": "fp32", "sidecar_bytes": 3145728,
                "expected_classification": "layer0_recurrent_prefix_state_exact"}):
        fail("plan contract differs")
    if (build.get("artifact_type") != "ninfer_retained_fda2d17a_diagnostic_build_receipt" or
            build.get("schema_version") != 1 or build.get("source") != plan["source"] or
            build.get("benchmark") != plan["benchmark"]):
        fail("build provenance differs")
    if capture != {"artifact_type": "ninfer_imported_manual_capture_receipt",
            "schema_version": 1,
            "original_directory": "/tmp/ninfer-gdn-state-layer0-fda2d17a.MOBqmy",
            "benchmark_sha256": EXE_SHA, "source_commit": SOURCE_COMMIT,
            "capture_complete": True, "process_receipts_retained": False,
            "power_endpoint_retained": False,
            "interpretation": "functional exact-state evidence only; timing and power behavior are inadmissible"}:
        fail("capture provenance differs")
    if digest(PACKAGE / plan["benchmark"]["path"]) != EXE_SHA:
        fail("retained benchmark differs")
    if subprocess.check_output(["git", "rev-parse", f"{SOURCE_COMMIT}^{{tree}}"],
                               cwd=ROOT, text=True).strip() != SOURCE_TREE:
        fail("source commit/tree differs")
    for relative, expected in build["source_files"].items():
        blob = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)
        if len(blob) != expected["bytes"] or hashlib.sha256(blob).hexdigest() != expected["sha256"]:
            fail(f"source identity differs: {relative}")
    for expected in build["toolchain_runtime"].values():
        if digest(Path(expected["path"])) != expected["sha256"]:
            fail("toolchain/runtime identity differs")
    for key in ("artifact",):
        expected = plan[key]
        if identity(Path(expected["path"])) != {name: expected[name]
                for name in ("path", "bytes", "sha256")}:
            fail(f"{key} identity differs")
    for key in ("p129_fixture", "source_corpus"):
        expected = plan["history"][key]
        if identity(Path(expected["path"])) != expected:
            fail(f"history identity differs: {key}")
    return plan, build, capture


def validate_report(stem: str) -> dict:
    report = load(RESULTS / f"{stem}.report.json")
    append = stem == "append"
    expected_command = (
        "build-r9700/bench/ninfer_bench --weights "
        "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer "
        f"--corpus {'bench/fixtures/bench_corpus.ids' if append else 'profiles/bench/r9700-dflash-p129-isolation-discriminator-20260906/history-p129.ids'} "
        "--device 0 --concurrency 1 " + ("-pg 128,1 " if append else "--whole-pg 129,1 ") +
        "--prefill-chunk 4096 --kv-capacity workload --draft-tokens 0 --retain-token-ids " +
        ("--isolate-prompt-decode " if append else "") +
        "--no-device-graph --output json --output-file "
        f"/tmp/ninfer-gdn-state-layer0-fda2d17a.MOBqmy/{stem}.report.json -r 1 --warmup 0")
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or report.get("command") != expected_command or
            report.get("environment") != {"gpu_name": "AMD Radeon AI PRO R9700",
                "architecture_name": "gfx1201", "hip_runtime_version": "7.15.26333",
                "hip_driver_version": "7.15.26333", "device_id": 0}):
        fail(f"report identity differs: {stem}")
    config = report.get("config", {})
    expected_config = {"max_context":130, "prefill_chunk":4096,
        "kv_cache_format":"fp8-k-int4-v", "kv_value_group":16,
        "q4_activation_bits":8, "w8_activation_bits":8, "fp8_qk_wmma_enabled":True,
        "concurrency":1, "spec":"none", "draft_tokens":0,
        "speculative_execution":False, "dflash_verify_width_requested":0,
        "dflash_verify_width":0, "proposal_head":"full", "use_device_graph":False,
        "retain_token_ids":True, "isolate_prompt_decode":append,
        "decode_path":"eager", "repetitions":1, "warmup":0,
        "corpus_tokens":65536 if append else 129}
    for key, expected in expected_config.items():
        if config.get(key) != expected or isinstance(config.get(key), bool) != isinstance(expected, bool):
            fail(f"report config differs: {stem}/{key}")
    if any(config.get(key) is not False for key in
           ("dflash_small_t_candidate", "dflash_mlp_down_t5_candidate",
            "dflash_rmsnorm_rows56_candidate", "xattention_qualification")):
        fail(f"selector config differs: {stem}")
    tests = report.get("tests")
    test = tests[0] if isinstance(tests, list) and len(tests) == 1 else None
    reps = test.get("reps") if isinstance(test, dict) else None
    rep = reps[0] if isinstance(reps, list) and len(reps) == 1 else None
    if (not isinstance(rep, dict) or test.get("n_prompt") != (128 if append else 129) or
            test.get("n_gen") != 1 or rep.get("generated_token_ids_by_lane") != [TOKENS] or
            rep.get("decode_output_tokens") != 1 or rep.get("decode_engine_tokens") != 1):
        fail(f"report output differs: {stem}")
    expected_spec = {"enabled":False,"draft_window":0,"rounds":0,"drafted_tokens":0,
        "accepted_tokens":0,"fallback_steps":0,"acceptance_rate":None,
        "acceptance_length":None,"accepted_per_position":[]}
    if test.get("speculative") != expected_spec or rep.get("speculative") != expected_spec:
        fail(f"speculative accounting differs: {stem}")
    return report


def validate_manifest(stem: str) -> tuple[dict, bytes]:
    manifest = load(RESULTS / f"{stem}.state.json")
    append = stem == "append"
    expected = {
        "artifact_type":"ninfer_qwen3_gdn_recurrent_state_trace", "schema_version":2,
        "diagnostic_only":True, "timing_evidence_eligible":False,
        "production_routing_authorized":False, "execution":"eager",
        "role":"text-append-frontier129-column0" if append else "text-fresh-frontier129-column128",
        "capture_point":"restored-append-state-before-selected-column-0" if append else
            "wide-prefill-state-after-prefix-128-before-selected-column-128",
        "selected_token":24178, "selected_cache_position":128, "selected_rope_position":128,
        "state_frontier":128, "linear_state_slot":0, "text_layer":0, "gdn_index":0,
        "dtype":"fp32", "shape":[128,128,48], "elements":786432,
        "sidecar_path":f"/tmp/ninfer-gdn-state-layer0-fda2d17a.MOBqmy/{stem}.state.bin",
        "sidecar_bytes":3145728, "sidecar_fnv1a64":"3e63547bee007c21",
        "layout":"little-endian-fp32:key,value,value_head"}
    if manifest != expected:
        fail(f"state manifest differs: {stem}")
    data = (RESULTS / f"{stem}.state.bin").read_bytes()
    if len(data) != 3145728 or digest(RESULTS / f"{stem}.state.bin") != STATE_SHA:
        fail(f"state sidecar differs: {stem}")
    values = struct.unpack("<786432f", data)
    if not all(math.isfinite(value) for value in values):
        fail(f"nonfinite state: {stem}")
    return manifest, data


def compare_state(left: bytes, right: bytes) -> dict:
    if len(left) != 3145728 or len(right) != 3145728:
        fail("state byte count differs")
    left_values = struct.unpack("<786432f", left)
    right_values = struct.unpack("<786432f", right)
    if not all(math.isfinite(value) for value in left_values + right_values):
        fail("nonfinite recurrent state")
    mismatch = next((index for index, pair in enumerate(zip(left_values, right_values))
                     if struct.pack("<f", pair[0]) != struct.pack("<f", pair[1])), None)
    if mismatch is not None:
        fail(f"layer0 recurrent prefix state differs at {mismatch}")
    return {"elements": 786432, "mismatch_count": 0, "sha256": hashlib.sha256(left).hexdigest()}


def analyze() -> dict:
    plan, build, capture = validate_plan()
    reports = {stem: validate_report(stem) for stem in ("fresh", "append")}
    states = {stem: validate_manifest(stem) for stem in ("fresh", "append")}
    state_result = compare_state(states["fresh"][1], states["append"][1])
    retained = load(RESULTS / "state-comparison.json")
    if retained != {"artifact_type":"ninfer_qwen3_gdn_recurrent_state_comparison",
            "schema_version":2, "diagnostic_only":True, "timing_evidence_eligible":False,
            "production_routing_authorized":False, "diagnostic":"text-prefix-state",
            "text_layer":0, "gdn_index":0,
            "classification":"layer0_recurrent_prefix_state_exact", "first_difference":None,
            "limitations":["compares one exact GDN FP32 state frontier and does not localize an earlier update",
                "does not authorize production routing or a performance claim"]}:
        fail("retained comparator output differs")
    return {"artifact_type":"ninfer_qwen3_layer0_recurrent_state_text_pair_summary",
        "schema_version":1, "status":"pass", "classification":"layer0_recurrent_prefix_state_exact",
        "source":plan["source"], "benchmark":plan["benchmark"],
        "state":{**state_result,
            "text_layer":0,"gdn_index":0,"state_frontier":128},
        "reports":{stem:identity(RESULTS / f"{stem}.report.json") for stem in reports},
        "manifests":{stem:identity(RESULTS / f"{stem}.state.json") for stem in states},
        "capture_provenance":identity(PACKAGE / "capture-provenance.json"),
        "conclusion":"layer0 state is exact while retained layer1 state differs; next discriminate layer0 FP8 gate/up T128/T129 from layer1 FP8 query/key T128/T129",
        "limitations":plan["limitations"]}


def write_exclusive(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                fail("summary write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    write_exclusive(RESULTS / "summary.json", analyze())
