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
PACKAGE_PLAN = PACKAGE / "package-plan.json"
PLAN_PATH = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"
EXPECTED_INVOCATION = {
    "prepare": "bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --prepare",
    "preflight": "bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --preflight",
    "measure": "bash profiles/bench/r9700-paired-projection-c2c4-whole-ab-20260919/commands.sh --measure",
}


def fail(message: str) -> None: raise RuntimeError(message)
def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))
    if not isinstance(value, dict): fail(f"not an object: {path}")
    return value
def digest(path: Path) -> str:
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
def identity(path: Path) -> dict:
    path = path.resolve(strict=True); return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
def require_identity(item: dict, path: Path, label: str) -> None:
    if item != identity(path): fail(f"{label} identity differs")
def exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream: stream.write(text)


def validate_invocation_authority(plan: dict) -> None:
    authority = load(PACKAGE_PLAN)
    if (authority.get("schema") !=
            "ninfer.r9700.paired-projection-c2c4-whole-ab-package.v1" or
            authority.get("status") != "reviewed_ready" or
            authority.get("production_routing_authorized") is not False or
            authority.get("exact_invocation") != EXPECTED_INVOCATION or
            plan.get("exact_invocation") != EXPECTED_INVOCATION or
            plan.get("package_authority") != identity(PACKAGE_PLAN)):
        fail("package invocation authority differs")


def command(stem: str, role: str, concurrency: int) -> list[str]:
    plan = load(PLAN_PATH)
    return [plan["builds"][role]["executable"]["path"], "--weights", plan["artifact"]["path"],
        "--corpus", plan["corpus"]["path"], "--device", "0", "--concurrency", str(concurrency),
        "--whole-pg", "8192,256", "--prefill-chunk", "4096", "--kv-capacity", "workload",
        "--spec", "mtp", "--draft-tokens", "0", "--retain-token-ids", "--output", "json",
        "--output-file", str(RESULTS / f"{stem}.json"), "-r", "1", "--warmup", "1"]


def run_one(stem: str, role: str, concurrency: int) -> None:
    plan = load(PLAN_PATH); power = Path(plan["hardware"]["power_path"])
    before = power.read_text().strip()
    if before != "auto": fail(f"power differs before {stem}")
    cmd = command(stem, role, concurrency); started = time.time_ns()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); finished = time.time_ns()
    after = power.read_text().strip(); stdout = RESULTS / f"{stem}.stdout"; stderr = RESULTS / f"{stem}.stderr"
    exclusive(stdout, completed.stdout); exclusive(stderr, completed.stderr)
    benchmark_report = RESULTS / f"{stem}.json"
    report_identity = identity(benchmark_report) if benchmark_report.is_file() else None
    exclusive(RESULTS / f"{stem}.process.json", json.dumps({"stem": stem, "role": role,
        "concurrency": concurrency, "command": cmd, "cwd": str(ROOT),
        "exit_code": completed.returncode, "started_unix_ns": started,
        "finished_unix_ns": finished, "device": plan["hardware"], "power_before": before,
        "power_after": after, "executable": identity(Path(cmd[0])),
        "stdout": identity(stdout), "stderr": identity(stderr),
        "benchmark_report": report_identity}, indent=2) + "\n")
    if completed.returncode != 0 or after != "auto": fail(f"benchmark failed: {stem}")


def validate_report(stem: str, role: str, concurrency: int) -> tuple[tuple[tuple[int, ...], ...], float]:
    plan = load(PLAN_PATH); report = load(RESULTS / f"{stem}.json"); process = load(RESULTS / f"{stem}.process.json")
    if (process.get("stem") != stem or process.get("role") != role or
            process.get("concurrency") != concurrency or
            process.get("command") != command(stem, role, concurrency) or
            process.get("cwd") != str(ROOT) or process.get("exit_code") != 0 or
            process.get("device") != plan["hardware"] or process.get("power_before") != "auto" or
            process.get("power_after") != "auto" or
            not isinstance(process.get("started_unix_ns"), int) or
            not isinstance(process.get("finished_unix_ns"), int) or
            process["started_unix_ns"] >= process["finished_unix_ns"]):
        fail(f"process provenance differs: {stem}")
    require_identity(process.get("executable"), Path(process["command"][0]),
                     f"{stem} executable")
    require_identity(process.get("stdout"), RESULTS / f"{stem}.stdout", f"{stem} stdout")
    require_identity(process.get("stderr"), RESULTS / f"{stem}.stderr", f"{stem} stderr")
    require_identity(process.get("benchmark_report"), RESULTS / f"{stem}.json",
                     f"{stem} benchmark report")
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or report.get("environment", {}).get("device_id") != 0 or
            report.get("environment", {}).get("gpu_name") != "AMD Radeon AI PRO R9700" or
            report.get("environment", {}).get("architecture_name") != "gfx1201" or
            report.get("artifact") != {"path": plan["artifact"]["path"], "file_size_bytes": plan["artifact"]["bytes"]}):
        fail(f"report provenance differs: {stem}")
    config = report.get("config", {}); expected = {"concurrency": concurrency, "prefill_chunk": 4096,
        "kv_cache_format": "fp8-k-int4-v", "q4_activation_bits": 8, "xattention_qualification": False,
        "q4_pair_wmma_c2c4_candidate": role == "candidate", "attention_q4_pair_t1_candidate": False,
        "bf16_gdn_control_t1_candidate": False, "spec": "none", "draft_tokens": 0,
        "use_device_graph": True, "retain_token_ids": True, "decode_path": "device_graph",
        "repetitions": 1, "warmup": 1}
    for key, value in expected.items():
        if config.get(key) != value: fail(f"config differs {stem}: {key}")
    tests = report.get("tests", [])
    if (len(tests) != 1 or tests[0].get("label") != "whole-pp8192+tg256" or
            tests[0].get("kind") != "whole" or tests[0].get("n_prompt") != 8192 or
            tests[0].get("n_gen") != 256 or tests[0].get("requested_output_tokens") != 257):
        fail(f"geometry differs: {stem}")
    reps = tests[0].get("reps", []); lanes = reps[0].get("generated_token_ids_by_lane") if len(reps) == 1 else None
    seconds = reps[0].get("timings", {}).get("decode_seconds") if len(reps) == 1 else None
    if (not isinstance(lanes, list) or len(lanes) != concurrency or any(len(lane) != 257 for lane in lanes) or
            reps[0].get("generated_output_tokens") != concurrency * 257 or
            reps[0].get("decode_output_tokens") != 256 or reps[0].get("decode_engine_tokens") != 256 or
            isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0):
        fail(f"tokens/timing differ: {stem}")
    token_hash = hashlib.sha256(json.dumps(lanes, separators=(",", ":")).encode()).hexdigest()
    if token_hash != plan["expected_tokens"][str(concurrency)]["sha256"]: fail(f"tokens differ: {stem}")
    return tuple(tuple(lane) for lane in lanes), float(seconds)


def analyze() -> dict:
    plan = load(PLAN_PATH); records = []
    for index, item in enumerate(plan["order"], 1):
        stem = f"run-{index}-c{item['concurrency']}-{item['role']}"
        tokens, seconds = validate_report(stem, item["role"], item["concurrency"])
        records.append({**item, "stem": stem, "tokens": tokens, "decode_seconds": seconds})
    pairs = {str(c): [] for c in (2, 3, 4)}
    for index in range(0, len(records), 2):
        left, right = records[index:index + 2]
        if left["concurrency"] != right["concurrency"] or left["role"] == right["role"]:
            fail("pair schedule differs")
        control = left if left["role"] == "control" else right; candidate = right if left["role"] == "control" else left
        if control["tokens"] != candidate["tokens"]: fail("paired public tokens differ")
        ratio = candidate["decode_seconds"] / control["decode_seconds"]
        pairs[str(left["concurrency"])].append({"order": [left["role"], right["role"]],
            "control_seconds": control["decode_seconds"], "candidate_seconds": candidate["decode_seconds"],
            "candidate_over_control": ratio})
    summaries = {}
    for c, values in pairs.items():
        ratios = [item["candidate_over_control"] for item in values]
        if len(ratios) != 3: fail(f"C{c} pair count differs")
        upper = statistics.mean(ratios) + 2.0 * statistics.stdev(ratios) / math.sqrt(3)
        median = statistics.median(ratios)
        if any(value >= 1.0 for value in ratios) or upper >= 1.0 or median > 0.99:
            fail(f"C{c} whole admission failed")
        summaries[c] = {"pairs": values, "mean": statistics.mean(ratios),
                        "upper_2se": upper, "median": median}
    return {"schema": "ninfer.r9700.paired-projection-c2c4-whole-ab-summary.v1",
        "status": "passed", "production_routing_authorized": True,
        "exact_public_token_parity": True, "by_concurrency": summaries,
        "limitations": plan["limitations"]}


def close_results() -> None:
    files = sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "result.sha256")
    exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path.name}\n" for path in files))


def main() -> int:
    validate_invocation_authority(load(PLAN_PATH))
    if RESULTS.exists() or RESULTS.is_symlink(): fail("results already exists; never overwrite")
    RESULTS.mkdir()
    try:
        for index, item in enumerate(load(PLAN_PATH)["order"], 1):
            run_one(f"run-{index}-c{item['concurrency']}-{item['role']}", item["role"], item["concurrency"])
        exclusive(RESULTS / "summary.json", json.dumps(analyze(), indent=2) + "\n")
        close_results(); print("r9700_paired_projection_c2c4_whole_ab: PASS"); return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists(): close_results()
        raise


if __name__ == "__main__": raise SystemExit(main())
