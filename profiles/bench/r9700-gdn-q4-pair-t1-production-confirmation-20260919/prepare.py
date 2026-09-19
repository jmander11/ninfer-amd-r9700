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
BUILD = ROOT / "build-r9700-gdn-q4-pair-production-20260919"
DIRECT = ROOT / "profiles/bench/r9700-a8q4-gdn-pair-t1-qualification-20260919.json"
WHOLE = ROOT / "profiles/bench/r9700-gdn-q4-pair-t1-whole-ab-20260919/results"
BASELINE = ROOT / "profiles/bench/r9700-base-decode-bandwidth-baseline-20260919/bound-plan.json"


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


def main() -> int:
    if PLAN.exists() or PLAN.is_symlink() or RESULTS.exists() or RESULTS.is_symlink():
        fail("plan/results already exists; preparation never overwrites")
    values = cache(BUILD / "CMakeCache.txt")
    if values.get("NINFER_BUILD_BENCHMARKS") != "ON" or values.get("CMAKE_HIP_ARCHITECTURES") != "gfx1201":
        fail("selector-free production build cache differs")
    if "NINFER_R9700_GDN_Q4_PAIR_T1_CANDIDATE" in values:
        fail("removed qualification selector remains in production cache")

    variant_object = BUILD / "src/CMakeFiles/ninfer_r9700_target_bindings.dir/targets/qwen3_8_27b/impl/variant.cpp.o"
    symbols = subprocess.run(["nm", "-C", str(variant_object)], check=True,
                             capture_output=True, text=True).stdout
    route_symbol = " U ninfer::ops::r9700::linear::a8q4g64_gdn_pair_t1("
    if route_symbol not in symbols:
        fail("production Variant object does not bind the paired Op")

    assembly = ROOT / "tools/r9700/build/a8q4_gdn_pair_t1.s"
    static_output = subprocess.run(
        ["python3", str(ROOT / "tools/r9700/check_a8q4_gdn_pair_t1_static.py"), str(assembly)],
        check=True, capture_output=True, text=True).stdout.strip()
    if "PASS" not in static_output or "wave32=true lds=0 scratch=0" not in static_output:
        fail("paired Op static receipt differs")

    direct = json.loads(DIRECT.read_text())
    whole = json.loads((WHOLE / "summary.json").read_text())
    baseline = json.loads(BASELINE.read_text())
    if (direct.get("status") != "qualified_for_whole_ab_only" or
            direct.get("correctness", {}).get("maximum_bf16_steps") != 0):
        fail("retained direct qualification differs")
    if (whole.get("status") != "passed" or whole.get("production_routing_authorized") is not True or
            whole.get("exact_public_token_parity") is not True):
        fail("retained whole admission differs")

    sources = [
        "CMakeLists.txt",
        "src/ops/r9700/linear/r9700_linear.h",
        "src/ops/r9700/linear/r9700_linear.hip",
        "src/targets/qwen3_8_27b/impl/variant.h",
        "src/targets/qwen3_8_27b/impl/variant.cpp",
        "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
        "tools/r9700/a8q4_gdn_pair_t1_qual.hip",
        "tools/r9700/check_a8q4_gdn_pair_t1_static.py",
        "tools/r9700/target_variant_gdn_qual.cpp",
        "profiles/bench/r9700-gdn-q4-pair-t1-production-confirmation-20260919/commands.sh",
        "profiles/bench/r9700-gdn-q4-pair-t1-production-confirmation-20260919/prepare.py",
        "profiles/bench/r9700-gdn-q4-pair-t1-production-confirmation-20260919/validate.py",
        "profiles/bench/r9700-gdn-q4-pair-t1-production-confirmation-20260919/run.py",
    ]
    retained_seconds = sorted(
        pair["candidate_seconds"] for pair in whole["pairs"])
    retained_median = retained_seconds[len(retained_seconds) // 2]
    plan = {
        "schema": "ninfer.r9700.gdn-q4-pair-t1-production-confirmation-plan.v1",
        "status": "prepared_awaiting_independent_review",
        "production_routing_authorized": True,
        "claim": "Selector-free production confirmation of the exact all-Q4 T1 GDN pair and unchanged C2..4 fallback.",
        "workload": {"device": 0, "concurrency": [1, 2, 3, 4], "whole_pg": "8192,256",
                     "prefill_chunk": 4096, "kv_capacity": "workload", "spec": "none",
                     "draft_tokens": 0, "device_graph": True, "retain_token_ids": True,
                     "required_power": "auto", "repetitions_per_process": 1,
                     "warmup_per_process": 1,
                     "process_order": [1, 2, 3, 4, 4, 3, 2, 1, 1]},
        "route_contract": {
            "selected": "tokens=1 and both projections exact Q4G64_F16S shapes",
            "rejected_tokens": [2, 3, 4],
            "rejected_mixed_weights": True,
            "fallback": "unchanged selected_linear(query_key) plus serialized_linear(value_z)",
            "static_assert_receipt": identity(BUILD / "src/ninfer_r9700_target_variant_gdn_qual"),
            "production_symbol_receipt": {"object": identity(variant_object),
                                           "paired_op_undefined_reference": True},
        },
        "build": {"directory": str(BUILD), "cache": identity(BUILD / "CMakeCache.txt"),
                  "cache_values": {key: values.get(key, "") for key in (
                      "CMAKE_BUILD_TYPE", "CMAKE_HIP_ARCHITECTURES", "NINFER_BUILD_BENCHMARKS",
                      "NINFER_R9700_KV_VALUE_GROUP", "NINFER_R9700_Q4_ACTIVATION_BITS",
                      "NINFER_R9700_W8_ACTIVATION_BITS", "NINFER_R9700_FP8_QK_WMMA")},
                  "executable": identity(BUILD / "bench/ninfer_bench")},
        "sources": [identity(ROOT / path) for path in sources],
        "static_receipt": {"assembly": identity(assembly), "output": static_output},
        "direct_qualification": identity(DIRECT),
        "retained_whole": {"summary": identity(WHOLE / "summary.json"),
                           "closure": identity(WHOLE / "result.sha256")},
        "artifact": baseline["artifact"], "corpus": baseline["corpus"],
        "retained_token_authority": baseline["retained_token_authority"],
        "expected_c1_tokens": baseline["expected_tokens"],
        "admission": {"c1_exact_authority_tokens": True,
                      "c2_c3_c4_exact_repeat_tokens": True,
                      "retained_candidate_median_decode_seconds": retained_median,
                      "production_c1_median_over_retained_at_most": 1.01},
        "limitations": [
            "C2..4 exercise and exactly repeat the unchanged fallback; the paired route is deliberately T1-only.",
            "This confirms retained whole speed, not physical HBM bandwidth or stall freedom.",
        ],
    }
    descriptor = os.open(PLAN, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print("r9700_gdn_q4_pair_t1_production_confirmation_prepare: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
