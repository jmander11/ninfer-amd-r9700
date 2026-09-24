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
CONTROL = ROOT / "build-r9700-bf16-gdn-control-t1-control"
CANDIDATE = ROOT / "build-r9700-bf16-gdn-control-t1-candidate"
DIRECT = ROOT / "profiles/bench/r9700-bf16-gdn-control-t1-production-qualification-20260919/report.json"
BASELINE = ROOT / "profiles/bench/r9700-base-decode-bandwidth-baseline-20260919/bound-plan.json"


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))
    if not isinstance(value, dict):
        fail(f"not an object: {path}")
    return value


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def cache(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        lhs, value = line.split("=", 1)
        result[lhs.split(":", 1)[0]] = value
    return result


def symbol_receipt(build: Path, candidate: bool) -> dict:
    obj = build / "src/CMakeFiles/ninfer_r9700_target_bindings.dir/targets/qwen3_8_27b/impl/variant.cpp.o"
    output = subprocess.run(["nm", "-C", str(obj)], check=True, capture_output=True, text=True).stdout
    reference = " U ninfer::ops::bf16_gdn_projected_gating_t1("
    present = reference in output
    if present is not candidate:
        fail("selector object-symbol receipt differs")
    return {"object": identity(obj), "projected_control_undefined_reference": present}


def main() -> int:
    if PLAN.exists() or PLAN.is_symlink():
        fail("plan already exists; preparation never overwrites")
    direct = load(DIRECT)
    if direct.get("status") != "qualified_for_whole_ab_only" or direct.get("production_routing_authorized") is not False:
        fail("direct production-symbol qualification is not admissible")
    baseline = load(BASELINE)
    raw_caches = {"control": cache(CONTROL / "CMakeCache.txt"),
                  "candidate": cache(CANDIDATE / "CMakeCache.txt")}
    keys = sorted({"CMAKE_BUILD_TYPE", "CMAKE_HIP_ARCHITECTURES", "CMAKE_HIP_COMPILER",
                   "CMAKE_HIP_FLAGS", "CMAKE_HIP_FLAGS_RELEASE", "CMAKE_CXX_COMPILER",
                   "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_EXE_LINKER_FLAGS",
                   "CMAKE_EXE_LINKER_FLAGS_RELEASE", "NINFER_BUILD_APPS",
                   "NINFER_BUILD_BENCHMARKS"} |
                  {key for values in raw_caches.values() for key in values
                   if key.startswith("NINFER_R9700_")})
    builds = {}
    caches = {}
    for role, build, selector in (("control", CONTROL, "0"), ("candidate", CANDIDATE, "1")):
        values = raw_caches[role]
        selected = {key: values.get(key, "") for key in keys}
        expected = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
                    "NINFER_BUILD_BENCHMARKS": "ON",
                    "NINFER_BUILD_APPS": "ON",
                    "NINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE": selector,
                    "NINFER_R9700_DFLASH_DOWN_SPLITK_FACTOR": "8",
                    "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
                    "NINFER_R9700_XATTENTION_STRIDE": "16",
                    "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000"}
        if any(selected.get(key) != value for key, value in expected.items()):
            fail(f"{role} build cache differs")
        caches[role] = selected
        builds[role] = {"selector": int(selector), "directory": str(build),
                        "cache": identity(build / "CMakeCache.txt"), "cache_values": selected,
                        "executable": identity(build / "bench/ninfer_bench"),
                        "symbol_receipt": symbol_receipt(build, role == "candidate")}
    for key in keys:
        if key != "NINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE" and caches["control"][key] != caches["candidate"][key]:
            fail(f"source-matched caches differ at {key}")
    sources = ["CMakeLists.txt", "include/ninfer/ops/gdn_gating.h",
               "src/ops/r9700/gdn/gdn_gating.cpp", "src/ops/r9700/gdn/gdn_ops.h",
               "src/ops/r9700/gdn/gdn_ops.hip", "src/targets/qwen3_8_27b/impl/variant.cpp",
               "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
               "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/package-plan.json",
               "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/commands.sh",
               "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/prepare.py",
               "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/run.py"]
    plan = {"schema": "ninfer.r9700.bf16-gdn-control-t1-whole-ab-plan.v1",
            "status": "prepared", "production_routing_authorized": False,
            "workload": {"device": 0, "concurrency": 1, "whole_pg": "8192,256",
                         "prefill_chunk": 4096, "spec": "none", "draft_tokens": 0,
                         "device_graph": True, "retain_token_ids": True, "required_power": "auto"},
            "order": ["control", "candidate", "candidate", "control", "control", "candidate"],
            "builds": builds, "sources": [identity(ROOT / source) for source in sources],
            "direct_qualification": identity(DIRECT), "artifact": baseline["artifact"],
            "corpus": baseline["corpus"], "retained_token_authority": baseline["retained_token_authority"],
            "expected_tokens": baseline["expected_tokens"],
            "admission": {"exact_public_tokens": True, "every_paired_ratio_below_one": True,
                          "paired_mean_upper_2se_below_one": True,
                          "median_candidate_over_control_at_most": 0.99}}
    descriptor = os.open(PLAN, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(plan, stream, indent=2); stream.write("\n")
    print("r9700_bf16_gdn_control_t1_whole_ab_prepare: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
