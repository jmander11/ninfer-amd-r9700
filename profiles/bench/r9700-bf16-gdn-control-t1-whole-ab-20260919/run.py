#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PLAN_PATH = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))
    if not isinstance(value, dict): fail(f"not an object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def require_identity(record: dict, path: Path, label: str) -> None:
    if record != identity(path): fail(f"{label} identity differs")


def cache(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        lhs, value = line.split("=", 1); values[lhs.split(":", 1)[0]] = value
    return values


def power_profile(plan: dict) -> Path:
    direct = load(Path(plan["direct_qualification"]["path"]))
    pci = direct.get("device", {}).get("pci_bus_id")
    if not isinstance(pci, str): fail("direct qualification lacks PCI identity")
    profile = (Path("/sys/bus/pci/devices") / pci /
               "power_dpm_force_performance_level").resolve(strict=True)
    if direct["device"].get("power_profile_path") != str(profile):
        fail("direct qualification power profile is not linked to its PCI identity")
    return profile


def validate_plan() -> tuple[dict, Path]:
    plan = load(PLAN_PATH)
    if (plan.get("schema") != "ninfer.r9700.bf16-gdn-control-t1-whole-ab-plan.v1" or
            plan.get("status") != "prepared" or plan.get("production_routing_authorized") is not False):
        fail("whole A/B plan state differs")
    expected_sources = {str((ROOT / path).resolve()) for path in (
        "CMakeLists.txt", "include/ninfer/ops/gdn_gating.h",
        "src/ops/r9700/gdn/gdn_gating.cpp", "src/ops/r9700/gdn/gdn_ops.h",
        "src/ops/r9700/gdn/gdn_ops.hip", "src/targets/qwen3_8_27b/impl/variant.cpp",
        "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/package-plan.json",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/commands.sh",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/prepare.py",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/run.py")}
    sources = plan.get("sources")
    if not isinstance(sources, list) or {record.get("path") for record in sources} != expected_sources:
        fail("plan-bound source inventory differs")
    if plan.get("order") != ["control", "candidate", "candidate", "control", "control", "candidate"]:
        fail("A/B execution order differs")
    expected_workload = {"device": 0, "concurrency": 1, "whole_pg": "8192,256",
                         "prefill_chunk": 4096, "spec": "none", "draft_tokens": 0,
                         "device_graph": True, "retain_token_ids": True,
                         "required_power": "auto"}
    if plan.get("workload") != expected_workload:
        fail("whole A/B workload differs")
    for record in sources:
        require_identity(record, Path(record["path"]), "plan-bound source")
    direct_path = Path(plan["direct_qualification"]["path"])
    require_identity(plan["direct_qualification"], direct_path, "direct qualification")
    direct = load(direct_path)
    if (direct.get("schema") != "ninfer.r9700.bf16-gdn-control-t1-production-qualification.v1" or
            direct.get("status") != "qualified_for_whole_ab_only" or
            direct.get("production_routing_authorized") is not False):
        fail("direct qualification status differs")
    direct_identities = direct.get("identities")
    required_direct = {"executable", "assembly", "static_receipt", "stdout", "stderr",
                       "device_stdout", "device_stderr", "device_process", "device_identity",
                       "qualification_process", "kernel_source", "semantic_wrapper",
                       "qualifier", "checker", "runner", "commands", "writer", "plan"}
    if not isinstance(direct_identities, dict) or set(direct_identities) != required_direct:
        fail("direct qualification identity inventory differs")
    for label, record in direct_identities.items():
        require_identity(record, Path(record["path"]), f"direct qualification {label}")
    for label in ("artifact", "corpus", "retained_token_authority"):
        record = plan[label]; require_identity(record, Path(record["path"]), label)
    expected_tokens = plan.get("expected_tokens", {})
    if expected_tokens.get("count") != 257 or not isinstance(expected_tokens.get("sha256"), str):
        fail("token authority expectation differs")
    caches = {}
    for role in ("control", "candidate"):
        build = plan["builds"][role]
        directory = Path(build["directory"])
        cache_path = Path(build["cache"]["path"])
        executable = Path(build["executable"]["path"])
        object_record = build["symbol_receipt"]["object"]
        require_identity(build["cache"], cache_path, f"{role} cache")
        require_identity(build["executable"], executable, f"{role} executable")
        require_identity(object_record, Path(object_record["path"]), f"{role} object")
        values = cache(cache_path)
        selected = {key: values.get(key, "") for key in build["cache_values"]}
        if selected != build["cache_values"]:
            fail(f"{role} cache values differ")
        if selected.get("CMAKE_BUILD_TYPE") != "Release": fail(f"{role} is not Release")
        output = subprocess.run(["nm", "-C", str(Path(object_record["path"]))], check=True,
                                capture_output=True, text=True).stdout
        present = " U ninfer::ops::bf16_gdn_projected_gating_t1(" in output
        if present is not build["symbol_receipt"]["projected_control_undefined_reference"] or present is not (role == "candidate"):
            fail(f"{role} selector symbol receipt differs")
        if executable.parent.parent != directory:
            fail(f"{role} executable is outside its build directory")
        caches[role] = selected
    for key in caches["control"]:
        if key != "NINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE" and caches["control"][key] != caches["candidate"].get(key):
            fail(f"execution-affecting caches differ at {key}")
    required_cache = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
                      "NINFER_BUILD_APPS": "ON", "NINFER_BUILD_BENCHMARKS": "ON",
                      "NINFER_R9700_DFLASH_DOWN_SPLITK_FACTOR": "8",
                      "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
                      "NINFER_R9700_XATTENTION_STRIDE": "16",
                      "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000"}
    for role, values in caches.items():
        if any(values.get(key) != value for key, value in required_cache.items()):
            fail(f"{role} required execution cache differs")
    profile = power_profile(plan)
    if profile.read_text().strip() != "auto": fail("selected R9700 power profile is not auto")
    return plan, profile


def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream: stream.write(text)


def command(plan: dict, stem: str, role: str) -> list[str]:
    return [plan["builds"][role]["executable"]["path"], "--weights", plan["artifact"]["path"],
            "--corpus", plan["corpus"]["path"], "--device", "0", "--concurrency", "1",
            "--whole-pg", "8192,256", "--prefill-chunk", "4096", "--kv-capacity", "workload",
            "--spec", "mtp", "--draft-tokens", "0", "--retain-token-ids", "--output", "json",
            "--output-file", str(RESULTS / f"{stem}.json"), "-r", "1", "--warmup", "1"]


def run_one(plan: dict, profile: Path, stem: str, role: str) -> None:
    if profile.read_text().strip() != "auto": fail(f"power is not auto before {stem}")
    require_identity(plan["builds"][role]["executable"],
                     Path(plan["builds"][role]["executable"]["path"]),
                     f"{stem} executable")
    cmd = command(plan, stem, role); started = time.time_ns()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = time.time_ns(); after = profile.read_text().strip()
    stdout = RESULTS / f"{stem}.stdout"; stderr = RESULTS / f"{stem}.stderr"
    write_exclusive(stdout, completed.stdout); write_exclusive(stderr, completed.stderr)
    benchmark_report = RESULTS / f"{stem}.json"
    report_identity = identity(benchmark_report) if benchmark_report.is_file() else None
    write_exclusive(RESULTS / f"{stem}.process.json", json.dumps(
        {"stem": stem, "role": role, "command": cmd, "cwd": str(ROOT),
         "exit_code": completed.returncode,
         "started_unix_ns": started, "finished_unix_ns": finished,
         "power_profile_path": str(profile), "power_before": "auto", "power_after": after,
         "executable": identity(Path(cmd[0])), "stdout": identity(stdout),
         "stderr": identity(stderr), "benchmark_report": report_identity}, indent=2) + "\n")
    if completed.returncode != 0 or after != "auto": fail(f"benchmark failed: {stem}")


def validate(plan: dict, profile: Path, stem: str, role: str) -> tuple[tuple[int, ...], float, float]:
    report = load(RESULTS / f"{stem}.json")
    process = load(RESULTS / f"{stem}.process.json")
    if (process.get("stem") != stem or process.get("role") != role or
            process.get("command") != command(plan, stem, role) or
            process.get("cwd") != str(ROOT) or process.get("exit_code") != 0 or
            process.get("power_profile_path") != str(profile) or
            process.get("power_before") != "auto" or process.get("power_after") != "auto" or
            not isinstance(process.get("started_unix_ns"), int) or
            not isinstance(process.get("finished_unix_ns"), int) or
            process["started_unix_ns"] >= process["finished_unix_ns"]):
        fail(f"process receipt differs: {stem}")
    require_identity(process.get("executable"), Path(process["command"][0]), f"{stem} executable")
    require_identity(process.get("stdout"), RESULTS / f"{stem}.stdout", f"{stem} stdout")
    require_identity(process.get("stderr"), RESULTS / f"{stem}.stderr", f"{stem} stderr")
    require_identity(process.get("benchmark_report"), RESULTS / f"{stem}.json",
                     f"{stem} benchmark report")
    if report.get("schema_version") != 20 or report.get("tool") != "ninfer_bench":
        fail(f"report schema differs: {stem}")
    environment = report.get("environment", {})
    if (environment.get("gpu_name") != "AMD Radeon AI PRO R9700" or
            environment.get("architecture_name") != "gfx1201" or
            environment.get("device_id") != plan["workload"]["device"]):
        fail(f"device differs: {stem}")
    if report.get("artifact") != {"path": plan["artifact"]["path"],
                                  "file_size_bytes": plan["artifact"]["bytes"]}:
        fail(f"benchmark artifact differs: {stem}")
    config = report.get("config", {})
    expected = {"concurrency": 1, "prefill_chunk": 4096, "kv_cache_format": "fp8-k-int4-v",
                "kv_value_group": 16, "q4_activation_bits": 8, "w8_activation_bits": 8,
                "fp8_qk_wmma_enabled": True, "xattention_qualification": False,
                "bf16_gdn_control_t1_candidate": role == "candidate", "spec": "none",
                "draft_tokens": 0, "use_device_graph": True, "retain_token_ids": True,
                "decode_path": "device_graph", "repetitions": 1, "warmup": 1}
    for key, value in expected.items():
        if config.get(key) != value: fail(f"config differs {stem}: {key}={config.get(key)!r}")
    tests = report.get("tests", [])
    if (len(tests) != 1 or tests[0].get("label") != "whole-pp8192+tg256" or
            tests[0].get("requested_output_tokens") != plan["expected_tokens"]["count"]):
        fail(f"geometry differs: {stem}")
    reps = tests[0].get("reps", [])
    if len(reps) != 1: fail(f"repetitions differ: {stem}")
    lanes = reps[0].get("generated_token_ids_by_lane")
    seconds = reps[0].get("timings", {}).get("decode_seconds")
    if not isinstance(lanes, list) or len(lanes) != 1 or len(lanes[0]) != 257:
        fail(f"token shape differs: {stem}")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        fail(f"timing differs: {stem}")
    token_hash = hashlib.sha256(json.dumps(lanes[0], separators=(",", ":")).encode()).hexdigest()
    if len(lanes[0]) != plan["expected_tokens"]["count"] or token_hash != plan["expected_tokens"]["sha256"]:
        fail(f"tokens differ: {stem}")
    return tuple(lanes[0]), float(seconds), 256.0 / float(seconds)


def main() -> int:
    if RESULTS.exists() or RESULTS.is_symlink(): fail("results already exist; never overwrite")
    RESULTS.mkdir()
    try:
        plan, profile = validate_plan(); records = []
        for index, role in enumerate(plan["order"], 1):
            stem = f"run-{index}-{role}"; run_one(plan, profile, stem, role)
            tokens, seconds, rate = validate(plan, profile, stem, role)
            records.append({"stem": stem, "role": role, "tokens": tokens,
                            "decode_seconds": seconds, "decode_output_tok_s": rate})
        if len({record["tokens"] for record in records}) != 1: fail("public tokens differ")
        pairs = []; ratios = []
        for index in range(3):
            left, right = records[2 * index:2 * index + 2]
            control = left if left["role"] == "control" else right
            candidate = left if left["role"] == "candidate" else right
            ratio = candidate["decode_seconds"] / control["decode_seconds"]
            ratios.append(ratio); pairs.append({"order": [left["role"], right["role"]],
                "control_seconds": control["decode_seconds"], "candidate_seconds": candidate["decode_seconds"],
                "candidate_over_control": ratio})
        upper = statistics.mean(ratios) + 2.0 * statistics.stdev(ratios) / math.sqrt(3)
        if any(ratio >= 1.0 for ratio in ratios) or upper >= 1.0 or statistics.median(ratios) > 0.99:
            fail("whole-model performance admission failed")
        summary = {"schema": "ninfer.r9700.bf16-gdn-control-t1-whole-ab-summary.v1",
                   "status": "passed", "production_routing_authorized": True,
                   "exact_public_token_parity": True, "pairs": pairs,
                   "paired_ratio_mean": statistics.mean(ratios), "paired_ratio_upper_2se": upper,
                   "paired_ratio_median": statistics.median(ratios),
                   "control_decode_output_tok_s": [r["decode_output_tok_s"] for r in records if r["role"] == "control"],
                   "candidate_decode_output_tok_s": [r["decode_output_tok_s"] for r in records if r["role"] == "candidate"]}
        write_exclusive(RESULTS / "summary.json", json.dumps(summary, indent=2) + "\n")
        files = sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "result.sha256")
        write_exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path.name}\n" for path in files))
        print("r9700_bf16_gdn_control_t1_whole_ab: PASS")
        return 0
    except Exception:
        files = sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "result.sha256")
        if not (RESULTS / "result.sha256").exists():
            write_exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path.name}\n" for path in files))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
