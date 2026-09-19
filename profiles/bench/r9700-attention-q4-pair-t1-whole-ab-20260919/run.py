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


def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream: stream.write(text)


def command(stem: str, role: str) -> list[str]:
    plan = load(PLAN_PATH)
    return [plan["builds"][role]["executable"]["path"], "--weights", plan["artifact"]["path"],
            "--corpus", plan["corpus"]["path"], "--device", "0", "--concurrency", "1",
            "--whole-pg", "8192,256", "--prefill-chunk", "4096", "--kv-capacity", "workload",
            "--spec", "mtp", "--draft-tokens", "0", "--retain-token-ids", "--output", "json",
            "--output-file", str(RESULTS / f"{stem}.json"), "-r", "1", "--warmup", "1"]


def run_one(stem: str, role: str) -> None:
    plan = load(PLAN_PATH)
    power = Path(plan["hardware"]["power_path"])
    power_before = power.read_text().strip()
    if power_before != plan["hardware"]["required_power"]:
        fail(f"power is not auto before {stem}: {power_before}")
    cmd = command(stem, role); started = time.time_ns()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = time.time_ns()
    power_after = power.read_text().strip()
    stdout, stderr = RESULTS / f"{stem}.stdout", RESULTS / f"{stem}.stderr"
    write_exclusive(stdout, completed.stdout); write_exclusive(stderr, completed.stderr)
    write_exclusive(RESULTS / f"{stem}.process.json", json.dumps({
        "role": role, "command": cmd, "exit_code": completed.returncode,
        "started_unix_ns": started, "finished_unix_ns": finished,
        "device": plan["hardware"], "power_before": power_before,
        "power_after": power_after,
        "stdout": identity(stdout), "stderr": identity(stderr)}, indent=2) + "\n")
    if completed.returncode != 0 or power_after != plan["hardware"]["required_power"]:
        fail(f"benchmark failed: {stem}")


def validate_report(stem: str, role: str) -> tuple[tuple[int, ...], float, float]:
    plan = load(PLAN_PATH); report = load(RESULTS / f"{stem}.json")
    process = load(RESULTS / f"{stem}.process.json")
    if process.get("role") != role or process.get("command") != command(stem, role) or process.get("exit_code") != 0:
        fail(f"process provenance differs: {stem}")
    if (process.get("device") != plan["hardware"] or
            process.get("power_before") != plan["hardware"]["required_power"] or
            process.get("power_after") != plan["hardware"]["required_power"]):
        fail(f"device/power provenance differs: {stem}")
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or
            report.get("environment", {}).get("gpu_name") != "AMD Radeon AI PRO R9700" or
            report.get("environment", {}).get("architecture_name") != "gfx1201" or
            report.get("environment", {}).get("device_id") != 0 or
            report.get("artifact") != {"path": plan["artifact"]["path"],
                                      "file_size_bytes": plan["artifact"]["bytes"]}):
        fail(f"benchmark provenance differs: {stem}")
    config = report.get("config", {})
    expected = {"concurrency": 1, "prefill_chunk": 4096, "kv_cache_format": "fp8-k-int4-v",
                "q4_activation_bits": 8, "xattention_qualification": False,
                "attention_q4_pair_t1_candidate": role == "candidate", "spec": "none",
                "draft_tokens": 0, "use_device_graph": True, "retain_token_ids": True,
                "decode_path": "device_graph", "repetitions": 1, "warmup": 1}
    for key, value in expected.items():
        if config.get(key) != value: fail(f"config differs {stem}: {key}={config.get(key)!r}")
    tests = report.get("tests", [])
    if (len(tests) != 1 or tests[0].get("label") != "whole-pp8192+tg256" or
            tests[0].get("kind") != "whole" or tests[0].get("n_prompt") != 8192 or
            tests[0].get("n_gen") != 256 or tests[0].get("requested_output_tokens") != 257):
        fail(f"geometry differs: {stem}")
    reps = tests[0].get("reps", [])
    if len(reps) != 1: fail(f"repetition count differs: {stem}")
    lanes = reps[0].get("generated_token_ids_by_lane")
    seconds = reps[0].get("timings", {}).get("decode_seconds")
    if (not isinstance(lanes, list) or len(lanes) != 1 or
            len(lanes[0]) != plan["expected_tokens"]["count"] or
            reps[0].get("generated_output_tokens") != 257 or
            reps[0].get("decode_output_tokens") != 256 or
            reps[0].get("decode_engine_tokens") != 256):
        fail(f"token shape differs: {stem}")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        fail(f"decode timing differs: {stem}")
    token_hash = hashlib.sha256(json.dumps(lanes[0], separators=(",", ":")).encode()).hexdigest()
    if token_hash != plan["expected_tokens"]["sha256"]: fail(f"tokens differ: {stem}")
    return tuple(lanes[0]), float(seconds), 256.0 / float(seconds)


def analyze() -> dict:
    plan = load(PLAN_PATH); records = []
    for index, role in enumerate(plan["order"], 1):
        stem = f"run-{index}-{role}"; tokens, seconds, rate = validate_report(stem, role)
        records.append({"stem": stem, "role": role, "tokens": tokens,
                        "decode_seconds": seconds, "decode_output_tok_s": rate})
    if len({r["tokens"] for r in records}) != 1: fail("public tokens differ")
    ratios, pairs = [], []
    for index in range(plan["workload"]["independent_pairs"]):
        left, right = records[2 * index:2 * index + 2]
        control = left if left["role"] == "control" else right
        candidate = left if left["role"] == "candidate" else right
        ratio = candidate["decode_seconds"] / control["decode_seconds"]
        ratios.append(ratio); pairs.append({"order": [left["role"], right["role"]],
            "control_seconds": control["decode_seconds"], "candidate_seconds": candidate["decode_seconds"],
            "candidate_over_control": ratio})
    upper = (statistics.mean(ratios) + 2.0 * statistics.stdev(ratios) /
             math.sqrt(plan["workload"]["independent_pairs"]))
    admission = plan["admission"]
    if (any(value >= 1.0 for value in ratios) or upper >= 1.0 or
            statistics.median(ratios) > admission["median_candidate_over_control_at_most"]):
        fail("whole-model performance admission failed")
    return {"schema": "ninfer.r9700.attention-q4-pair-t1-whole-ab-summary.v1", "status": "passed",
            "production_routing_authorized": True, "exact_public_token_parity": True,
            "pairs": pairs, "paired_ratio_mean": statistics.mean(ratios),
            "paired_ratio_upper_2se": upper, "paired_ratio_median": statistics.median(ratios),
            "limitations": plan["limitations"]}


def close_results() -> None:
    files = sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path.name}\n" for path in files))


def main() -> int:
    if RESULTS.exists() or RESULTS.is_symlink(): fail("results already exists; never overwrite")
    RESULTS.mkdir()
    try:
        for index, role in enumerate(load(PLAN_PATH)["order"], 1): run_one(f"run-{index}-{role}", role)
        write_exclusive(RESULTS / "summary.json", json.dumps(analyze(), indent=2) + "\n")
        close_results(); print("r9700_attention_q4_pair_t1_whole_ab: PASS"); return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists(): close_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
