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
POWER = Path("/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level")


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))
    if not isinstance(value, dict):
        fail(f"not a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)


def command(stem: str, concurrency: int) -> list[str]:
    plan = load(PLAN_PATH)
    return [plan["build"]["executable"]["path"], "--weights", plan["artifact"]["path"],
            "--corpus", plan["corpus"]["path"], "--device", "0", "--concurrency", str(concurrency),
            "--whole-pg", "8192,256", "--prefill-chunk", "4096", "--kv-capacity", "workload",
            "--spec", "mtp", "--draft-tokens", "0", "--retain-token-ids", "--output", "json",
            "--output-file", str(RESULTS / f"{stem}.json"), "-r", "1", "--warmup", "1"]


def run_one(stem: str, concurrency: int) -> None:
    if POWER.read_text().strip() != "auto":
        fail(f"power is not auto before {stem}")
    cmd = command(stem, concurrency)
    started = time.time_ns()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = time.time_ns()
    stdout, stderr = RESULTS / f"{stem}.stdout", RESULTS / f"{stem}.stderr"
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(RESULTS / f"{stem}.process.json", json.dumps({
        "concurrency": concurrency, "command": cmd, "exit_code": completed.returncode,
        "started_unix_ns": started, "finished_unix_ns": finished,
        "power_before": "auto", "power_after": POWER.read_text().strip(),
        "stdout": identity(stdout), "stderr": identity(stderr),
    }, indent=2) + "\n")
    if completed.returncode != 0 or POWER.read_text().strip() != "auto":
        fail(f"benchmark process failed: {stem}")


def validate_report(stem: str, concurrency: int) -> tuple[tuple[tuple[int, ...], ...], float]:
    plan = load(PLAN_PATH)
    report = load(RESULTS / f"{stem}.json")
    process = load(RESULTS / f"{stem}.process.json")
    if (process.get("concurrency") != concurrency or process.get("command") != command(stem, concurrency) or
            process.get("exit_code") != 0):
        fail(f"process provenance differs: {stem}")
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or
            report.get("environment", {}).get("gpu_name") != "AMD Radeon AI PRO R9700" or
            report.get("environment", {}).get("architecture_name") != "gfx1201" or
            report.get("artifact") != {"path": plan["artifact"]["path"],
                                      "file_size_bytes": plan["artifact"]["bytes"]}):
        fail(f"benchmark provenance differs: {stem}")
    config = report.get("config", {})
    expected = {"concurrency": concurrency, "prefill_chunk": 4096,
                "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 16,
                "q4_activation_bits": 8, "w8_activation_bits": 8,
                "fp8_qk_wmma_enabled": True, "xattention_qualification": False,
                "dflash_small_t_candidate": False, "dflash_mlp_down_t5_candidate": False,
                "dflash_down_splitk_candidate": False, "dflash_rmsnorm_rows56_candidate": False,
                "spec": "none", "draft_tokens": 0, "use_device_graph": True,
                "retain_token_ids": True, "decode_path": "device_graph",
                "repetitions": 1, "warmup": 1}
    for key, value in expected.items():
        if config.get(key) != value:
            fail(f"config differs {stem}: {key}={config.get(key)!r}")
    if "gdn_q4_pair_t1_candidate" in config:
        fail(f"removed selector leaked into report: {stem}")
    if config.get("decode_graph_prime") != {"primed": True, "output_tokens": 3}:
        fail(f"decode graph prime differs: {stem}")
    tests = report.get("tests", [])
    if (len(tests) != 1 or tests[0].get("label") != "whole-pp8192+tg256" or
            tests[0].get("requested_output_tokens") != 257):
        fail(f"geometry differs: {stem}")
    reps = tests[0].get("reps", [])
    if len(reps) != 1:
        fail(f"repetition count differs: {stem}")
    lanes = reps[0].get("generated_token_ids_by_lane")
    seconds = reps[0].get("timings", {}).get("decode_seconds")
    if (not isinstance(lanes, list) or len(lanes) != concurrency or
            any(not isinstance(lane, list) or len(lane) != 257 for lane in lanes)):
        fail(f"token shape differs: {stem}")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        fail(f"decode timing differs: {stem}")
    result = tuple(tuple(int(token) for token in lane) for lane in lanes)
    if concurrency == 1:
        token_bytes = json.dumps(lanes[0], separators=(",", ":")).encode()
        expected_tokens = plan["expected_c1_tokens"]
        if hashlib.sha256(token_bytes).hexdigest() != expected_tokens["sha256"]:
            fail(f"C1 tokens differ from retained authority: {stem}")
    return result, float(seconds)


def analyze(records: list[dict]) -> dict:
    plan = load(PLAN_PATH)
    by_concurrency = {concurrency: [record for record in records if record["concurrency"] == concurrency]
                      for concurrency in range(1, 5)}
    for concurrency, selected in by_concurrency.items():
        if len({record["tokens"] for record in selected}) != 1:
            fail(f"C{concurrency} exact public tokens differ between independent processes")
    c1_seconds = [record["decode_seconds"] for record in by_concurrency[1]]
    production_median = statistics.median(c1_seconds)
    retained_median = plan["admission"]["retained_candidate_median_decode_seconds"]
    ratio = production_median / retained_median
    if ratio > plan["admission"]["production_c1_median_over_retained_at_most"]:
        fail("selector-free production did not retain admitted C1 speed")
    return {
        "schema": "ninfer.r9700.gdn-q4-pair-t1-production-confirmation-summary.v1",
        "status": "passed", "production_route_confirmed": True,
        "c1_exact_retained_authority_tokens": True,
        "c2_c3_c4_exact_repeat_tokens": True,
        "route_selection": {"c1": "paired_q4_t1", "c2_c3_c4": "unchanged_fallback"},
        "c1_decode_seconds": c1_seconds,
        "c1_median_decode_seconds": production_median,
        "retained_candidate_median_decode_seconds": retained_median,
        "production_over_retained": ratio,
        "decode_seconds_by_concurrency": {
            str(concurrency): [record["decode_seconds"] for record in selected]
            for concurrency, selected in by_concurrency.items()},
        "limitations": plan["limitations"],
    }


def close_results() -> None:
    files = sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path.name}\n" for path in files))


def main() -> int:
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("results already exists; never overwrite")
    RESULTS.mkdir()
    try:
        plan = load(PLAN_PATH)
        records = []
        for index, concurrency in enumerate(plan["workload"]["process_order"], 1):
            stem = f"run-{index}-c{concurrency}"
            run_one(stem, concurrency)
            tokens, seconds = validate_report(stem, concurrency)
            records.append({"stem": stem, "concurrency": concurrency,
                            "tokens": tokens, "decode_seconds": seconds})
        write_exclusive(RESULTS / "summary.json", json.dumps(analyze(records), indent=2) + "\n")
        close_results()
        print("r9700_gdn_q4_pair_t1_production_confirmation: PASS")
        return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists():
            close_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
