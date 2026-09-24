#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PLAN = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"
CONTROL = ROOT / "build-r9700-gdn-q4-pair-t1-control"
CANDIDATE = ROOT / "build-r9700-gdn-q4-pair-t1"
DIRECT = ROOT / "profiles/bench/r9700-a8q4-gdn-pair-t1-qualification-20260919.json"
BASELINE_PLAN = ROOT / "profiles/bench/r9700-base-decode-bandwidth-baseline-20260919/bound-plan.json"


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def cache(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        lhs, value = line.split("=", 1)
        result[lhs.split(":", 1)[0]] = value
    return result


def symbol_receipt(build: Path, expected_selector: str) -> dict:
    obj = build / "src/CMakeFiles/ninfer_r9700_target_bindings.dir/targets/qwen3_8_27b/impl/variant.cpp.o"
    output = subprocess.run(["nm", "-C", str(obj)], check=True, capture_output=True, text=True).stdout
    reference = " U ninfer::ops::r9700::linear::a8q4g64_gdn_pair_t1("
    present = reference in output
    if present is not (expected_selector == "1"):
        fail("selector object-symbol receipt differs")
    return {"object": identity(obj), "paired_op_undefined_reference": present}


def main() -> int:
    if PLAN.exists() or PLAN.is_symlink() or RESULTS.exists() or RESULTS.is_symlink():
        fail("plan/results already exists; preparation never overwrites")
    baseline = json.loads(BASELINE_PLAN.read_text())
    direct = json.loads(DIRECT.read_text())
    if direct.get("status") != "qualified_for_whole_ab_only" or direct.get("production_routing_authorized") is not False:
        fail("direct qualification is not the reviewed pre-whole state")
    cache_keys = [
        "CMAKE_BUILD_TYPE", "CMAKE_HIP_ARCHITECTURES", "NINFER_BUILD_BENCHMARKS",
        "NINFER_R9700_KV_VALUE_GROUP", "NINFER_R9700_Q4_ACTIVATION_BITS",
        "NINFER_R9700_W8_ACTIVATION_BITS", "NINFER_R9700_FP8_QK_WMMA",
        "NINFER_R9700_XATTENTION_QUALIFICATION", "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE",
        "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE",
        "NINFER_R9700_DFLASH_DOWN_SPLITK_CANDIDATE",
        "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE",
        "NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE",
        "NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE",
        "NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE",
        "NINFER_R9700_GDN_Q4_PAIR_T1_CANDIDATE",
    ]
    caches = {}
    for role, build, selector in (("control", CONTROL, "0"), ("candidate", CANDIDATE, "1")):
        values = cache(build / "CMakeCache.txt")
        selected = {key: values.get(key, "") for key in cache_keys}
        if selected["NINFER_BUILD_BENCHMARKS"] != "ON" or selected["CMAKE_HIP_ARCHITECTURES"] != "gfx1201" or selected["NINFER_R9700_GDN_Q4_PAIR_T1_CANDIDATE"] != selector:
            fail(f"{role} build cache differs")
        caches[role] = selected
    for key in cache_keys:
        if key != "NINFER_R9700_GDN_Q4_PAIR_T1_CANDIDATE" and caches["control"][key] != caches["candidate"][key]:
            fail(f"source-matched caches differ at {key}")
    sources = [
        "CMakeLists.txt", "src/ops/r9700/linear/r9700_linear.h",
        "src/ops/r9700/linear/r9700_linear.hip",
        "src/targets/qwen3_8_27b/impl/variant.h",
        "src/targets/qwen3_8_27b/impl/variant.cpp",
        "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
        "tools/r9700/a8q4_gdn_pair_t1_qual.hip",
        "tools/r9700/check_a8q4_gdn_pair_t1_static.py",
        "profiles/bench/r9700-gdn-q4-pair-t1-whole-ab-20260919/commands.sh",
        "profiles/bench/r9700-gdn-q4-pair-t1-whole-ab-20260919/prepare.py",
        "profiles/bench/r9700-gdn-q4-pair-t1-whole-ab-20260919/validate.py",
        "profiles/bench/r9700-gdn-q4-pair-t1-whole-ab-20260919/run.py",
    ]
    plan = {
        "schema": "ninfer.r9700.gdn-q4-pair-t1-whole-ab-plan.v1",
        "status": "prepared_awaiting_independent_review",
        "production_routing_authorized": False,
        "claim": "Whole ordinary-decode A/B for exact T1 paired GDN Q4 projections after direct qualification.",
        "workload": {"device": 0, "concurrency": 1, "whole_pg": "8192,256",
                     "prefill_chunk": 4096, "kv_capacity": "workload", "spec": "none",
                     "draft_tokens": 0, "device_graph": True, "retain_token_ids": True,
                     "independent_pairs": 3, "repetitions_per_process": 1,
                     "warmup_per_process": 1, "required_power": "auto"},
        "order": ["control", "candidate", "candidate", "control", "control", "candidate"],
        "builds": {
            "control": {"selector": 0, "directory": str(CONTROL),
                        "cache": identity(CONTROL / "CMakeCache.txt"), "cache_values": caches["control"],
                        "executable": identity(CONTROL / "bench/ninfer_bench"),
                        "symbol_receipt": symbol_receipt(CONTROL, "0")},
            "candidate": {"selector": 1, "directory": str(CANDIDATE),
                          "cache": identity(CANDIDATE / "CMakeCache.txt"), "cache_values": caches["candidate"],
                          "executable": identity(CANDIDATE / "bench/ninfer_bench"),
                          "symbol_receipt": symbol_receipt(CANDIDATE, "1")},
        },
        "sources": [identity(ROOT / path) for path in sources],
        "direct_qualification": identity(DIRECT),
        "artifact": baseline["artifact"], "corpus": baseline["corpus"],
        "retained_token_authority": baseline["retained_token_authority"],
        "expected_tokens": baseline["expected_tokens"],
        "admission": {"exact_public_tokens": True, "every_paired_ratio_below_one": True,
                      "paired_mean_upper_2se_below_one": True,
                      "median_candidate_over_control_at_most": 0.99},
        "limitations": ["This is an unprofiled whole-model timing gate; it does not prove physical HBM bandwidth."]
    }
    descriptor = os.open(PLAN, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print("r9700_gdn_q4_pair_t1_whole_ab_prepare: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
