#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PLAN = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"
EXPECTED_ORDER = [
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
EXPECTED_WORKLOAD = {"device": 0, "concurrency": [2, 3, 4], "whole_pg": "8192,256",
    "prefill_chunk": 4096, "kv_capacity": "workload", "spec": "none", "draft_tokens": 0,
    "device_graph": True, "retain_token_ids": True, "pairs_per_concurrency": 3,
    "repetitions": 1, "warmup": 1, "required_power": "auto"}
EXPECTED_ADMISSION = {"exact_public_tokens": True, "every_paired_ratio_below_one": True,
    "paired_mean_upper_2se_below_one": True, "median_candidate_over_control_at_most": 0.99}
EXPECTED_LIMITATIONS = ["Unprofiled whole-model timing does not prove physical HBM bandwidth."]
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


def fail(message: str) -> None: raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def check(item: dict) -> None:
    path = Path(item["path"])
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or
            digest(path) != item["sha256"]): fail(f"bound identity changed: {path}")


def token_identity(item: dict, concurrency: int) -> tuple[int, int, str]:
    check(item["authority"]); report = json.loads(Path(item["authority"]["path"]).read_text())
    tests = report.get("tests", []); reps = tests[0].get("reps", []) if len(tests) == 1 else []
    lanes = reps[0].get("generated_token_ids_by_lane") if len(reps) == 1 else None
    if (report.get("schema_version") != 20 or tests[0].get("label") != "whole-pp8192+tg256" or
            not isinstance(lanes, list) or len(lanes) != concurrency or
            any(not isinstance(lane, list) or len(lane) != 257 for lane in lanes)):
        fail(f"C{concurrency} token authority differs")
    value = hashlib.sha256(json.dumps(lanes, separators=(",", ":")).encode()).hexdigest()
    return len(lanes), len(lanes[0]), value


def direct_ok(report: dict) -> None:
    if (report.get("schema") != "ninfer.r9700.paired-projection-c2c4-production-qualification.v1" or
            report.get("status") != "qualified_for_whole_ab_only" or
            report.get("production_routing_authorized") is not False or
            report.get("correctness", {}).get("three_arm_complete_bf16_bit_exact") is not True or
            report.get("static", {}).get("native_iu4_wmma") is not True or
            report.get("decision") != "prepare_selector_off_on_whole_ab_c2c4"):
        fail("direct receipt differs")
    if {(x.get("role"), x.get("tokens")) for x in report.get("cases", [])
            if x.get("every_combined_copy_nonregressing") is True} != {
            (role, c) for role in ("gdn", "attention") for c in (2, 3, 4)}:
        fail("direct case inventory differs")
    if ({x.get("concurrency") for x in report.get("whole_bounds", [])} != {2, 3, 4} or
            any(x.get("credible_gain_percent", 0.0) < 1.0 for x in report["whole_bounds"])):
        fail("direct bound differs")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--require-fresh", action="store_true")
    args = parser.parse_args(); plan = json.loads(PLAN.read_text())
    if (plan.get("schema") != "ninfer.r9700.paired-projection-c2c4-whole-ab-plan.v1" or
            plan.get("status") != "prepared_awaiting_independent_review" or
            plan.get("production_routing_authorized") is not False or
            plan.get("claim") != "Matched selector-off/on whole ordinary decode A/B at C2, C3, and C4."):
        fail("plan disposition differs")
    if (plan.get("workload") != EXPECTED_WORKLOAD or plan.get("order") != EXPECTED_ORDER or
            plan.get("admission") != EXPECTED_ADMISSION or
            plan.get("limitations") != EXPECTED_LIMITATIONS): fail("plan contract differs")
    fixed = {"direct_qualification": ROOT / "profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919/qualification.json",
             "artifact": ROOT / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer",
             "corpus": ROOT / "bench/fixtures/bench_corpus.ids"}
    for key, path in fixed.items():
        if plan.get(key, {}).get("path") != str(path): fail(f"{key} path differs")
        check(plan[key])
    direct = json.loads(Path(plan["direct_qualification"]["path"]).read_text()); direct_ok(direct)
    expected_sources = {ROOT / path for path in (
        "CMakeLists.txt", "src/CMakeLists.txt", "include/ninfer/ops/attention_projection.h",
        "include/ninfer/ops/gdn_projection.h", "src/ops/r9700/paired_projection/paired_projection.h",
        "src/ops/r9700/paired_projection/paired_projection.hip",
        "src/targets/qwen3_8_27b/impl/variant.h", "src/targets/qwen3_8_27b/impl/variant.cpp",
        "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
        "tools/r9700/target_variant_pair_c2c4_qual.cpp")}
    expected_sources |= {PACKAGE / name for name in
                         ("README.md", "commands.sh", "prepare.py", "validate.py", "run.py")}
    if len(plan.get("sources", [])) != len(expected_sources) or {Path(x["path"]) for x in plan["sources"]} != expected_sources:
        fail("source inventory differs")
    for item in plan["sources"]: check(item)
    device = direct.get("hardware", {}); pci = device.get("pci")
    expected_hardware = {"device": 0, "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201",
        "wave_size": 32, "pci": pci,
        "power_path": f"/sys/bus/pci/devices/{pci}/power_dpm_force_performance_level",
        "required_power": "auto"}
    if plan.get("hardware") != expected_hardware or device.get("device") != 0: fail("hardware differs")
    power = Path(expected_hardware["power_path"])
    if not power.is_file() or power.is_symlink() or power.read_text().strip() != "auto": fail("power differs")
    if set(plan.get("builds", {})) != {"control", "candidate"}: fail("build inventory differs")
    live_cache_values = {}
    for role, selected in (("control", False), ("candidate", True)):
        build = plan["builds"][role]; expected = dict(COMMON_CACHE)
        expected["NINFER_R9700_Q4_PAIR_WMMA_C2C4_CANDIDATE"] = str(int(selected))
        values = build.get("cache_values")
        if (build.get("selector") != int(selected) or not isinstance(values, dict) or
                any(values.get(key) != value for key, value in expected.items()) or
                build.get("directory") != str(ROOT / f"build-r9700-paired-projection-c2c4-{role}") or
                build.get("symbol_receipt", {}).get("attention") is not selected or
                build.get("symbol_receipt", {}).get("gdn") is not selected): fail(f"build differs: {role}")
        for item in (build["cache"], build["executable"], build["selector_assert"], build["symbol_receipt"]["object"]): check(item)
        cache = Path(build["cache"]["path"]).read_text()
        for key, value in expected.items():
            if not any(line.startswith(key + ":") and line.endswith("=" + value) for line in cache.splitlines()):
                fail(f"live cache differs: {role}/{key}")
        for key, value in values.items():
            if not any(line.startswith(key + ":") and line.endswith("=" + value)
                       for line in cache.splitlines()):
                fail(f"bound live cache differs: {role}/{key}")
        live_cache_values[role] = values
        symbols = subprocess.run(["nm", "-C", build["symbol_receipt"]["object"]["path"]], check=True, capture_output=True, text=True).stdout
        if ((" U ninfer::ops::full_attention_projection_decode(" in symbols) is not selected or
                (" U ninfer::ops::gdn_input_projection_decode(" in symbols) is not selected):
            fail(f"live symbols differ: {role}")
    if set(live_cache_values["control"]) != set(live_cache_values["candidate"]):
        fail("build cache inventories differ")
    for key, value in live_cache_values["control"].items():
        if (key != "NINFER_R9700_Q4_PAIR_WMMA_C2C4_CANDIDATE" and
                live_cache_values["candidate"].get(key) != value):
            fail(f"source-matched build caches differ at {key}")
    if set(plan.get("expected_tokens", {})) != {"2", "3", "4"}: fail("token inventory differs")
    for c in (2, 3, 4):
        item = plan["expected_tokens"][str(c)]; lanes, count, token_hash = token_identity(item, c)
        if item.get("lane_count") != lanes or item.get("tokens_per_lane") != count or item.get("sha256") != token_hash:
            fail(f"C{c} token identity differs")
    if args.require_fresh and (RESULTS.exists() or RESULTS.is_symlink()): fail("results path is not fresh")
    print("r9700_paired_projection_c2c4_whole_ab_preflight: PASS"); return 0


if __name__ == "__main__": raise SystemExit(main())
