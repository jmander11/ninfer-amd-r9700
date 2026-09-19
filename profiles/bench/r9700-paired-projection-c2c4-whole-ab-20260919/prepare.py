#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PACKAGE_PLAN = PACKAGE / "package-plan.json"
PLAN = PACKAGE / "plan.json"
DIRECT = ROOT / "profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919/qualification.json"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
AUTHORITY_DIR = ROOT / "profiles/bench/r9700-gdn-q4-pair-t1-production-confirmation-20260919/results"
AUTHORITIES = {2: AUTHORITY_DIR / "run-2-c2.json", 3: AUTHORITY_DIR / "run-3-c3.json",
               4: AUTHORITY_DIR / "run-4-c4.json"}
BUILDS = {role: ROOT / f"build-r9700-paired-projection-c2c4-{role}"
          for role in ("control", "candidate")}
ORDER = [
    {"concurrency": 2, "role": "control"}, {"concurrency": 2, "role": "candidate"},
    {"concurrency": 3, "role": "control"}, {"concurrency": 3, "role": "candidate"},
    {"concurrency": 4, "role": "control"}, {"concurrency": 4, "role": "candidate"},
    {"concurrency": 4, "role": "candidate"}, {"concurrency": 4, "role": "control"},
    {"concurrency": 3, "role": "candidate"}, {"concurrency": 3, "role": "control"},
    {"concurrency": 2, "role": "candidate"}, {"concurrency": 2, "role": "control"},
    {"concurrency": 2, "role": "control"}, {"concurrency": 2, "role": "candidate"},
    {"concurrency": 3, "role": "control"}, {"concurrency": 3, "role": "candidate"},
    {"concurrency": 4, "role": "control"}, {"concurrency": 4, "role": "candidate"},
]
EXACT_INVOCATION = {
    "prepare": "bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --prepare",
    "preflight": "bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --preflight",
    "measure": "bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --measure",
}
COMMON_CACHE = {
    "CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
    "NINFER_BUILD_APPS": "ON", "NINFER_BUILD_BENCHMARKS": "ON",
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
    "NINFER_R9700_ATTENTION_Q4_PAIR_T1_CANDIDATE": "0",
    "NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE": "0",
    "NINFER_R9700_DFLASH_DOWN_SPLITK_FACTOR": "8",
    "NINFER_R9700_XATTENTION_STRIDE": "16",
    "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000",
}
SOURCES = [ROOT / path for path in (
    "CMakeLists.txt", "src/CMakeLists.txt", "include/ninfer/ops/attention_projection.h",
    "include/ninfer/ops/gdn_projection.h",
    "src/ops/r9700/paired_projection/paired_projection.h",
    "src/ops/r9700/paired_projection/paired_projection.hip",
    "src/targets/qwen3_8_27b/impl/variant.h", "src/targets/qwen3_8_27b/impl/variant.cpp",
    "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
    "tools/r9700/target_variant_pair_c2c4_qual.cpp",
)] + [PACKAGE / name for name in
      ("README.md", "package-plan.json", "commands.sh", "prepare.py", "validate.py", "run.py")]


def fail(message: str) -> None: raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def package_authority() -> dict:
    authority = json.loads(PACKAGE_PLAN.read_text())
    if (authority.get("schema") !=
            "ninfer.r9700.paired-projection-c2c4-whole-ab-package.v1" or
            authority.get("status") != "reviewed_ready" or
            authority.get("production_routing_authorized") is not False or
            authority.get("exact_invocation") != EXACT_INVOCATION):
        fail("package invocation authority differs")
    return authority


def require_identity(item: dict, label: str) -> Path:
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
        fail(f"{label} identity shape differs")
    path = Path(item["path"])
    if item != identity(path): fail(f"{label} identity differs")
    return path


def cache(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if (not line or line.startswith(("#", "//")) or "=" not in line or
                ":" not in line.split("=", 1)[0]):
            continue
        lhs, value = line.split("=", 1)
        result[lhs.split(":", 1)[0]] = value
    return result


def exclusive(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2); stream.write("\n")


def direct_ok(report: dict) -> None:
    if (report.get("schema") != "ninfer.r9700.paired-projection-c2c4-production-qualification.v1" or
            report.get("status") != "qualified_for_whole_ab_only" or
            report.get("production_routing_authorized") is not False or
            report.get("correctness", {}).get("three_arm_complete_bf16_bit_exact") is not True or
            report.get("static", {}).get("native_iu4_wmma") is not True or
            report.get("decision") != "prepare_selector_off_on_whole_ab_c2c4"):
        fail("direct production qualification does not admit whole A/B")
    cases = {(case.get("role"), case.get("tokens")) for case in report.get("cases", [])
             if case.get("every_combined_copy_nonregressing") is True}
    if cases != {(role, c) for role in ("gdn", "attention") for c in (2, 3, 4)}:
        fail("direct production case inventory differs")
    bounds = report.get("whole_bounds", [])
    if ({item.get("concurrency") for item in bounds} != {2, 3, 4} or
            any(item.get("credible_gain_percent", 0.0) < 1.0 for item in bounds)):
        fail("direct whole bound differs")
    executable = report.get("executable", {})
    require_identity(executable, "direct executable")
    sources = report.get("sources")
    expected_roles = {"qualifier_entry", "qualifier_body", "shared_qualifier",
                      "static_checker", "attention_contract", "gdn_contract",
                      "raw_contract", "op_implementation", "linear_primitive"}
    if (not isinstance(sources, list) or
            {item.get("role") for item in sources} != expected_roles):
        fail("direct source inventory differs")
    for item in sources:
        require_identity({key: item[key] for key in ("path", "bytes", "sha256")},
                         f"direct source {item['role']}")
    baseline = report.get("whole_baselines", {})
    baseline_path = Path(baseline.get("path", ""))
    if (not baseline_path.is_file() or baseline_path.is_symlink() or
            digest(baseline_path) != baseline.get("sha256")):
        fail("direct whole-baseline identity differs")
    static = report.get("static", {})
    for label in ("assembly", "receipt"):
        item = static.get(label, {})
        path = Path(item.get("path", ""))
        if (not path.is_file() or path.is_symlink() or digest(path) != item.get("sha256")):
            fail(f"direct static {label} identity differs")
    checker = next(item for item in sources if item["role"] == "static_checker")
    if static.get("checker_sha256") != checker["sha256"]:
        fail("direct checker identity differs")
    hardware = report.get("hardware", {})
    pci = hardware.get("pci")
    if (hardware.get("device") != 0 or hardware.get("name") != "AMD Radeon AI PRO R9700" or
            hardware.get("architecture") != "gfx1201" or hardware.get("wave_size") != 32 or
            not isinstance(pci, str)):
        fail("direct hardware identity differs")
    power = (Path("/sys/bus/pci/devices") / pci /
             "power_dpm_force_performance_level").resolve(strict=True)
    if report.get("power_profile") != {"path": str(power), "required": "auto",
                                        "before": "auto", "after": "auto"}:
        fail("direct PCI/power identity differs")


def retained_tokens(path: Path, concurrency: int) -> dict:
    report = json.loads(path.read_text())
    tests = report.get("tests", [])
    if (report.get("schema_version") != 20 or len(tests) != 1 or
            tests[0].get("label") != "whole-pp8192+tg256"):
        fail(f"C{concurrency} token authority differs")
    reps = tests[0].get("reps", []); lanes = reps[0].get("generated_token_ids_by_lane") if len(reps) == 1 else None
    if (not isinstance(lanes, list) or len(lanes) != concurrency or
            any(not isinstance(lane, list) or len(lane) != 257 for lane in lanes)):
        fail(f"C{concurrency} token authority lanes differ")
    payload = json.dumps(lanes, separators=(",", ":")).encode()
    return {"authority": identity(path), "lane_count": concurrency, "tokens_per_lane": 257,
            "sha256": hashlib.sha256(payload).hexdigest()}


def configure(role: str, selector: int) -> dict:
    directory = BUILDS[role]
    args = ["cmake", "-S", str(ROOT), "-B", str(directory), "-GNinja"] + [
        f"-D{key}={value}" for key, value in COMMON_CACHE.items()] + [
        f"-DNINFER_R9700_Q4_PAIR_WMMA_C2C4_CANDIDATE={selector}"]
    subprocess.run(args, cwd=ROOT, check=True)
    subprocess.run(["cmake", "--build", str(directory), "-j2", "--target", "ninfer_bench",
                    "ninfer_r9700_target_variant_pair_c2c4_qual"], cwd=ROOT, check=True)
    raw_cache = cache(directory / "CMakeCache.txt")
    required = dict(COMMON_CACHE)
    required["NINFER_R9700_Q4_PAIR_WMMA_C2C4_CANDIDATE"] = str(selector)
    for key, value in required.items():
        if raw_cache.get(key) != value:
            fail(f"cache differs for {role}: {key}")
    cache_keys = sorted({"CMAKE_BUILD_TYPE", "CMAKE_HIP_ARCHITECTURES",
                         "CMAKE_HIP_COMPILER", "CMAKE_HIP_FLAGS", "CMAKE_HIP_FLAGS_RELEASE",
                         "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
                         "CMAKE_EXE_LINKER_FLAGS", "CMAKE_EXE_LINKER_FLAGS_RELEASE",
                         "NINFER_BUILD_APPS", "NINFER_BUILD_BENCHMARKS"} |
                        {key for key in raw_cache if key.startswith("NINFER_R9700_")})
    cache_values = {key: raw_cache.get(key, "") for key in cache_keys}
    obj = directory / "src/CMakeFiles/ninfer_r9700_target_bindings.dir/targets/qwen3_8_27b/impl/variant.cpp.o"
    symbols = subprocess.run(["nm", "-C", str(obj)], check=True, capture_output=True, text=True).stdout
    attention = " U ninfer::ops::full_attention_projection_decode(" in symbols
    gdn = " U ninfer::ops::gdn_input_projection_decode(" in symbols
    if attention != bool(selector) or gdn != bool(selector): fail(f"route symbols differ: {role}")
    selector_assert = directory / "src/ninfer_r9700_target_variant_pair_c2c4_qual"
    subprocess.run([str(selector_assert)], cwd=ROOT, check=True)
    return {"selector": selector, "directory": str(directory), "cache_values": cache_values,
            "cache": identity(directory / "CMakeCache.txt"),
            "executable": identity(directory / "bench/ninfer_bench"),
            "selector_assert": identity(selector_assert),
            "symbol_receipt": {"object": identity(obj), "attention": attention, "gdn": gdn}}


def main() -> int:
    if Path.cwd() != ROOT: fail(f"prepare must run from {ROOT}")
    authority = package_authority()
    if PLAN.exists() or PLAN.is_symlink(): fail("plan already exists; never overwrite")
    for role, directory in BUILDS.items():
        if directory.exists() or directory.is_symlink():
            fail(f"{role} build directory is not fresh; never reuse or overwrite")
    direct = json.loads(DIRECT.read_text()); direct_ok(direct)
    device = direct.get("hardware", {})
    if (device.get("device") != 0 or device.get("name") != "AMD Radeon AI PRO R9700" or
            device.get("architecture") != "gfx1201" or device.get("wave_size") != 32):
        fail("direct device identity differs")
    pci = device.get("pci"); power = Path("/sys/bus/pci/devices") / str(pci) / "power_dpm_force_performance_level"
    if not isinstance(pci, str) or not power.is_file() or power.is_symlink(): fail("device-0 power binding differs")
    builds = {"control": configure("control", 0), "candidate": configure("candidate", 1)}
    control_cache = builds["control"]["cache_values"]
    candidate_cache = builds["candidate"]["cache_values"]
    if set(control_cache) != set(candidate_cache): fail("build cache inventories differ")
    for key in control_cache:
        if (key != "NINFER_R9700_Q4_PAIR_WMMA_C2C4_CANDIDATE" and
                control_cache[key] != candidate_cache[key]):
            fail(f"source-matched build caches differ at {key}")
    plan = {
        "schema": "ninfer.r9700.paired-projection-c2c4-whole-ab-plan.v1",
        "status": "prepared_awaiting_independent_review", "production_routing_authorized": False,
        "claim": "Matched selector-off/on whole ordinary decode A/B at C2, C3, and C4.",
        "package_authority": identity(PACKAGE_PLAN),
        "exact_invocation": authority["exact_invocation"],
        "workload": {"device": 0, "concurrency": [2, 3, 4], "whole_pg": "8192,256",
                     "prefill_chunk": 4096, "kv_capacity": "workload", "spec": "none",
                     "draft_tokens": 0, "device_graph": True, "retain_token_ids": True,
                     "pairs_per_concurrency": 3, "repetitions": 1, "warmup": 1,
                     "required_power": "auto"},
        "order": ORDER, "hardware": {"device": 0, "name": device["name"],
            "architecture": device["architecture"], "wave_size": 32, "pci": pci,
            "power_path": str(power), "required_power": "auto"},
        "builds": builds,
        "sources": [identity(path) for path in SOURCES], "direct_qualification": identity(DIRECT),
        "artifact": identity(ARTIFACT), "corpus": identity(CORPUS),
        "expected_tokens": {str(c): retained_tokens(path, c) for c, path in AUTHORITIES.items()},
        "admission": {"exact_public_tokens": True, "every_paired_ratio_below_one": True,
            "paired_mean_upper_2se_below_one": True,
            "median_candidate_over_control_at_most": 0.99},
        "limitations": ["Unprofiled whole-model timing does not prove physical HBM bandwidth."],
    }
    exclusive(PLAN, plan); print("r9700_paired_projection_c2c4_whole_ab_prepare: PASS"); return 0


if __name__ == "__main__": raise SystemExit(main())
