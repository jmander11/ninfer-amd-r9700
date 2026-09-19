#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PACKAGE = Path(__file__).resolve().parent
PLAN = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
EXPECTED_ORDER = ["control", "candidate", "candidate", "control", "control", "candidate"]
EXPECTED_WORKLOAD = {
    "device": 0, "concurrency": 1, "whole_pg": "8192,256", "prefill_chunk": 4096,
    "kv_capacity": "workload", "spec": "none", "draft_tokens": 0,
    "device_graph": True, "retain_token_ids": True, "independent_pairs": 3,
    "repetitions_per_process": 1, "warmup_per_process": 1, "required_power": "auto",
}
EXPECTED_ADMISSION = {
    "exact_public_tokens": True, "every_paired_ratio_below_one": True,
    "paired_mean_upper_2se_below_one": True,
    "median_candidate_over_control_at_most": 0.99,
}
EXPECTED_LIMITATIONS = ["Unprofiled whole-model timing does not prove physical HBM bandwidth."]
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


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(item: dict) -> None:
    path = Path(item["path"])
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or
            digest(path) != item["sha256"]):
        fail(f"bound identity changed: {path}")


def authority_tokens(item: dict) -> tuple[int, str]:
    authority = json.loads(Path(item["path"]).read_text())
    tests = authority.get("tests", [])
    if (authority.get("schema_version") != 20 or len(tests) != 1 or
            tests[0].get("label") != "whole-pp8192+tg256" or
            tests[0].get("n_prompt") != 8192 or tests[0].get("n_gen") != 256 or
            tests[0].get("requested_output_tokens") != 257):
        fail("retained token authority geometry differs")
    reps = tests[0].get("reps", [])
    if (len(reps) != 1 or reps[0].get("generated_output_tokens") != 257 or
            reps[0].get("decode_output_tokens") != 256):
        fail("retained token authority repetition differs")
    lanes = reps[0].get("generated_token_ids_by_lane")
    if not isinstance(lanes, list) or len(lanes) != 1 or not isinstance(lanes[0], list):
        fail("retained token authority lanes differ")
    payload = json.dumps(lanes[0], separators=(",", ":")).encode()
    return len(lanes[0]), hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--require-fresh", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan.get("schema") != "ninfer.r9700.attention-q4-pair-t1-whole-ab-plan.v1" or
            plan.get("status") != "prepared_awaiting_independent_review" or
            plan.get("production_routing_authorized") is not False):
        fail("plan disposition differs")
    if plan.get("claim") != "Whole ordinary-decode A/B for the exact all-Q4 T1 attention projection candidate.":
        fail("claim differs")
    if plan.get("workload") != EXPECTED_WORKLOAD or plan.get("order") != EXPECTED_ORDER:
        fail("workload or six-role order differs")
    if plan.get("admission") != EXPECTED_ADMISSION or plan.get("limitations") != EXPECTED_LIMITATIONS:
        fail("admission or limitations differ")
    expected_paths = {
        "artifact": ROOT / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer",
        "corpus": ROOT / "bench/fixtures/bench_corpus.ids",
        "retained_token_authority": ROOT / "profiles/bench/r9700-rmsnorm-production-final-p8192-g256-c1-20260906.json",
        "direct_qualification": ROOT / "profiles/bench/r9700-attention-projection-t1-production-qualification-20260919/qualification.json",
    }
    for key, path in expected_paths.items():
        if plan.get(key, {}).get("path") != str(path):
            fail(f"{key} path differs")
    expected_sources = {
        ROOT / "CMakeLists.txt", ROOT / "src/CMakeLists.txt",
        ROOT / "include/ninfer/ops/attention_projection.h",
        ROOT / "src/ops/r9700/attention_projection/attention_projection.h",
        ROOT / "src/ops/r9700/attention_projection/attention_projection.hip",
        ROOT / "src/targets/qwen3_8_27b/impl/variant.h",
        ROOT / "src/targets/qwen3_8_27b/impl/variant.cpp",
        ROOT / "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
        ROOT / "tools/r9700/target_variant_attention_projection_qual.cpp",
        PACKAGE / "commands.sh", PACKAGE / "prepare.py", PACKAGE / "validate.py", PACKAGE / "run.py",
    }
    if {Path(item["path"]) for item in plan.get("sources", [])} != expected_sources:
        fail("bound source inventory differs")
    for item in plan["sources"] + [plan["direct_qualification"], plan["artifact"],
                                    plan["corpus"], plan["retained_token_authority"]]:
        check(item)
    if set(plan.get("builds", {})) != {"control", "candidate"}:
        fail("build role inventory differs")
    for role, expected in (("control", False), ("candidate", True)):
        build = plan["builds"][role]
        expected_cache = dict(COMMON_CACHE)
        expected_cache["NINFER_R9700_ATTENTION_Q4_PAIR_T1_CANDIDATE"] = str(int(expected))
        if (build.get("selector") != int(expected) or
                build.get("directory") != str(ROOT / f"build-r9700-attention-q4-pair-t1-{role}") or
                build.get("cache_values") != expected_cache or
                build.get("symbol_receipt", {}).get("attention_op_undefined_reference") is not expected):
            fail(f"bound selector differs: {role}")
        check(build["cache"]); check(build["executable"]); check(build["symbol_receipt"]["object"])
        cache_text = Path(build["cache"]["path"]).read_text()
        for key, value in expected_cache.items():
            if not any(line.startswith(key + ":") and line.endswith("=" + value)
                       for line in cache_text.splitlines()):
                fail(f"live cache value differs for {role}: {key}")
        output = subprocess.run(["nm", "-C", build["symbol_receipt"]["object"]["path"]],
                                check=True, capture_output=True, text=True).stdout
        if (" U ninfer::ops::full_attention_projection_t1(" in output) is not expected:
            fail(f"live selector symbol differs: {role}")
    direct = json.loads(Path(plan["direct_qualification"]["path"]).read_text())
    if (direct.get("status") != "qualified_for_whole_inference_ab" or
            direct.get("correctness", {}).get("complete_outputs_bit_exact_to_serial_plus_four_extracts") is not True or
            direct.get("whole_bound", {}).get("passed") is not True or
            direct.get("static", {}).get("native_iu4_dot8") is not True):
        fail("direct qualification receipt differs")
    device = direct.get("device", {})
    hardware = plan.get("hardware")
    if hardware != {"device": 0, "name": "AMD Radeon AI PRO R9700",
                    "architecture": "gfx1201", "wave_size": 32,
                    "pci": device.get("pci_bus_id"),
                    "power_path": f"/sys/bus/pci/devices/{device.get('pci_bus_id')}/power_dpm_force_performance_level",
                    "required_power": "auto"} or device.get("index") != 0:
        fail("device-0 PCI/power binding differs")
    power = Path(hardware["power_path"])
    if not power.is_file() or power.is_symlink():
        fail("derived power path is absent or unsafe")
    count, token_hash = authority_tokens(plan["retained_token_authority"])
    if plan.get("expected_tokens") != {
            "count": count, "sha256": token_hash,
            "derived_from": plan["retained_token_authority"]["path"]}:
        fail("expected tokens are not derived from retained authority")
    if args.require_fresh and (RESULTS.exists() or RESULTS.is_symlink()):
        fail("results path is not fresh")
    if power.read_text().strip() != "auto":
        fail("R9700 power profile is not auto")
    print("r9700_attention_q4_pair_t1_whole_ab_preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
