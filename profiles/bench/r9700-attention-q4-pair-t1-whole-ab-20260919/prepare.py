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
DIRECT = ROOT / "profiles/bench/r9700-attention-projection-t1-production-qualification-20260919/qualification.json"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
TOKENS = ROOT / "profiles/bench/r9700-rmsnorm-production-final-p8192-g256-c1-20260906.json"
BUILDS = {
    "control": ROOT / "build-r9700-attention-q4-pair-t1-control",
    "candidate": ROOT / "build-r9700-attention-q4-pair-t1-candidate",
}
COMMON_CACHE = {
    "CMAKE_HIP_ARCHITECTURES": "gfx1201", "NINFER_BUILD_BENCHMARKS": "ON",
    "NINFER_R9700_KV_VALUE_GROUP": "16", "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
    "NINFER_R9700_W8_ACTIVATION_BITS": "8", "NINFER_R9700_FP8_QK_WMMA": "1",
    "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
    "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE": "0",
    "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE": "0",
    "NINFER_R9700_DFLASH_DOWN_SPLITK_CANDIDATE": "0",
    "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE": "0",
    "NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE": "0",
    "NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE": "0",
    "NINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE": "0",
    "NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE": "0",
}
SOURCES = [
    ROOT / "CMakeLists.txt", ROOT / "src/CMakeLists.txt",
    ROOT / "include/ninfer/ops/attention_projection.h",
    ROOT / "src/ops/r9700/attention_projection/attention_projection.h",
    ROOT / "src/ops/r9700/attention_projection/attention_projection.hip",
    ROOT / "src/targets/qwen3_8_27b/impl/variant.h",
    ROOT / "src/targets/qwen3_8_27b/impl/variant.cpp",
    ROOT / "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
    ROOT / "tools/r9700/target_variant_attention_projection_qual.cpp",
    PACKAGE / "commands.sh", PACKAGE / "prepare.py", PACKAGE / "validate.py", PACKAGE / "run.py",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_exclusive(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def retained_tokens() -> tuple[int, str]:
    authority = json.loads(TOKENS.read_text())
    tests = authority.get("tests", [])
    if (authority.get("schema_version") != 20 or len(tests) != 1 or
            tests[0].get("label") != "whole-pp8192+tg256"):
        fail("retained token authority geometry differs")
    reps = tests[0].get("reps", [])
    if len(reps) != 1:
        fail("retained token authority repetition differs")
    lanes = reps[0].get("generated_token_ids_by_lane")
    if not isinstance(lanes, list) or len(lanes) != 1 or not isinstance(lanes[0], list):
        fail("retained token authority lanes differ")
    payload = json.dumps(lanes[0], separators=(",", ":")).encode()
    return len(lanes[0]), hashlib.sha256(payload).hexdigest()


def configure(role: str, selector: int) -> dict:
    directory = BUILDS[role]
    subprocess.run([
        "cmake", "-S", str(ROOT), "-B", str(directory), "-GNinja",
        "-DNINFER_BUILD_BENCHMARKS=ON", "-DCMAKE_HIP_ARCHITECTURES=gfx1201",
        "-DNINFER_R9700_KV_VALUE_GROUP=16", "-DNINFER_R9700_Q4_ACTIVATION_BITS=8",
        "-DNINFER_R9700_W8_ACTIVATION_BITS=8", "-DNINFER_R9700_FP8_QK_WMMA=1",
        f"-DNINFER_R9700_ATTENTION_Q4_PAIR_T1_CANDIDATE={selector}",
        "-DNINFER_R9700_XATTENTION_QUALIFICATION=OFF",
        "-DNINFER_R9700_DFLASH_SMALL_T_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_DOWN_SPLITK_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=0",
        "-DNINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE=0",
        "-DNINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE=0",
        "-DNINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE=0",
        "-DNINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE=0",
    ], cwd=ROOT, check=True)
    subprocess.run(["cmake", "--build", str(directory), "-j2", "--target", "ninfer_bench"],
                   cwd=ROOT, check=True)
    executable = directory / "bench/ninfer_bench"
    obj = directory / "src/CMakeFiles/ninfer_r9700_target_bindings.dir/targets/qwen3_8_27b/impl/variant.cpp.o"
    symbols = subprocess.run(["nm", "-C", str(obj)], check=True,
                             capture_output=True, text=True).stdout
    present = " U ninfer::ops::full_attention_projection_t1(" in symbols
    if present != bool(selector):
        fail(f"selector symbol receipt differs: {role}")
    expected_cache = dict(COMMON_CACHE)
    expected_cache["NINFER_R9700_ATTENTION_Q4_PAIR_T1_CANDIDATE"] = str(selector)
    cache_text = (directory / "CMakeCache.txt").read_text()
    for key, value in expected_cache.items():
        if not any(line.startswith(key + ":") and line.endswith("=" + value)
                   for line in cache_text.splitlines()):
            fail(f"cache value differs for {role}: {key}")
    return {"selector": selector, "directory": str(directory),
            "cache_values": expected_cache,
            "cache": identity(directory / "CMakeCache.txt"),
            "executable": identity(executable),
            "symbol_receipt": {"object": identity(obj),
                               "attention_op_undefined_reference": present}}


def main() -> int:
    if Path.cwd() != ROOT:
        fail(f"prepare must run from {ROOT}")
    if PLAN.exists() or PLAN.is_symlink():
        fail("plan already exists; never overwrite")
    direct = json.loads(DIRECT.read_text())
    if (direct.get("schema") != "ninfer.r9700.attention-projection-t1-production-qualification.v1" or
            direct.get("status") != "qualified_for_whole_inference_ab" or
            direct.get("whole_bound", {}).get("passed") is not True):
        fail("direct production qualification does not authorize whole A/B")
    device = direct.get("device", {})
    if (device.get("index") != 0 or device.get("name") != "AMD Radeon AI PRO R9700" or
            device.get("architecture") != "gfx1201" or device.get("wave_size") != 32):
        fail("direct qualification device identity differs")
    pci = device.get("pci_bus_id")
    if not isinstance(pci, str) or not pci:
        fail("direct qualification lacks device-0 PCI identity")
    power_path = Path("/sys/bus/pci/devices") / pci / "power_dpm_force_performance_level"
    if not power_path.is_file() or power_path.is_symlink():
        fail("device-0-derived power profile path is absent or unsafe")
    token_count, token_hash = retained_tokens()
    builds = {"control": configure("control", 0), "candidate": configure("candidate", 1)}
    plan = {
        "schema": "ninfer.r9700.attention-q4-pair-t1-whole-ab-plan.v1",
        "status": "prepared_awaiting_independent_review", "production_routing_authorized": False,
        "claim": "Whole ordinary-decode A/B for the exact all-Q4 T1 attention projection candidate.",
        "workload": {"device": 0, "concurrency": 1, "whole_pg": "8192,256",
                     "prefill_chunk": 4096, "kv_capacity": "workload", "spec": "none",
                     "draft_tokens": 0, "device_graph": True, "retain_token_ids": True,
                     "independent_pairs": 3, "repetitions_per_process": 1,
                     "warmup_per_process": 1, "required_power": "auto"},
        "order": ["control", "candidate", "candidate", "control", "control", "candidate"],
        "hardware": {"device": 0, "name": device["name"],
                     "architecture": device["architecture"], "wave_size": device["wave_size"],
                     "pci": pci, "power_path": str(power_path), "required_power": "auto"},
        "builds": builds, "sources": [identity(path) for path in SOURCES],
        "direct_qualification": identity(DIRECT), "artifact": identity(ARTIFACT),
        "corpus": identity(CORPUS), "retained_token_authority": identity(TOKENS),
        "expected_tokens": {"count": token_count, "sha256": token_hash,
                            "derived_from": str(TOKENS)},
        "admission": {"exact_public_tokens": True, "every_paired_ratio_below_one": True,
                      "paired_mean_upper_2se_below_one": True,
                      "median_candidate_over_control_at_most": 0.99},
        "limitations": ["Unprofiled whole-model timing does not prove physical HBM bandwidth."],
    }
    write_exclusive(PLAN, plan)
    print("r9700_attention_q4_pair_t1_whole_ab_prepare: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
