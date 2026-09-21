#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
RESULTS = PACKAGE / "results"
POWER = Path("/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level")
PROFILES = {"k4w5": (4, 5), "k5w6": (5, 6)}
ORDERS = {"k4w5": ("control", "candidate", "candidate", "control"),
          "k5w6": ("candidate", "control", "control", "candidate")}


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
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        data = text.encode()
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                fail(f"write made no progress: {path}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def command(stem: str, role: str, draft: int = 0, width: int = 0) -> list[str]:
    plan = load(PACKAGE / "plan.json")
    exe = plan["builds"][role]["executable"]
    report = RESULTS / f"{stem}.json"
    cmd = [exe, "--weights", plan["artifact"]["path"], "--corpus", plan["corpus"]["path"],
           "--device", "0", "--concurrency", "1", "--whole-pg", "128,64",
           "--prefill-chunk", "4096", "--kv-capacity", "workload", "--retain-token-ids",
           "--output", "json", "--output-file", str(report), "-r", "3", "--warmup", "1"]
    if draft:
        cmd += ["--spec", "dflash", "--draft-tokens", str(draft),
                "--dflash-verify-width", str(width), "--lm-head-draft"]
    else:
        cmd += ["--spec", "mtp", "--draft-tokens", "0"]
    return cmd


def run_one(stem: str, role: str, draft: int = 0, width: int = 0) -> None:
    cmd = command(stem, role, draft, width)
    stdout, stderr = RESULTS / f"{stem}.stdout", RESULTS / f"{stem}.stderr"
    before = POWER.read_text().strip()
    if before != "auto":
        fail("power profile changed before benchmark")
    started = time.time_ns()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = time.time_ns()
    after = POWER.read_text().strip()
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(RESULTS / f"{stem}.process.json", json.dumps({
        "command": cmd, "exit_code": completed.returncode, "started_unix_ns": started,
        "finished_unix_ns": finished, "power_before": before, "power_after": after,
        "stdout": identity(stdout), "stderr": identity(stderr)
    }, indent=2) + "\n")
    if completed.returncode or after != "auto" or not (RESULTS / f"{stem}.json").is_file():
        fail(f"benchmark failed: {stem}")


def median(values: list[float]) -> float:
    return statistics.median(values)


def validate_report(stem: str, role: str, draft: int, width: int) -> tuple[list[int], list[float], tuple]:
    report = load(RESULTS / f"{stem}.json")
    process = load(RESULTS / f"{stem}.process.json")
    if process.get("command") != command(stem, role, draft, width) or process.get("exit_code") != 0:
        fail(f"process provenance differs: {stem}")
    plan = load(PACKAGE / "plan.json")
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or
            report.get("command") != " ".join(command(stem, role, draft, width)) or
            report.get("environment", {}).get("gpu_name") != "AMD Radeon AI PRO R9700" or
            report.get("environment", {}).get("architecture_name") != "gfx1201" or
            report.get("artifact") != {"path": plan["artifact"]["path"],
                                      "file_size_bytes": plan["artifact"]["bytes"]}):
        fail(f"report provenance differs: {stem}")
    config = report.get("config", {})
    expected = {
        "kv_cache_format": "fp8-k-int4-v", "kv_value_group": 16,
        "q4_activation_bits": 8, "w8_activation_bits": 8, "fp8_qk_wmma_enabled": True,
        "xattention_qualification": False, "dflash_small_t_candidate": False,
        "dflash_mlp_down_t5_candidate": False, "dflash_rmsnorm_rows56_candidate": False,
        "dflash_down_splitk_candidate": False,
        "dflash_down_scale_gather_candidate": role == "candidate", "concurrency": 1,
        "text_p129_wmma_tail_candidate": False,
        "dflash_verify_width_requested": width,
        "prefill_chunk": 4096, "spec": "dflash" if draft else "none",
        "draft_tokens": draft, "dflash_verify_width": width, "use_device_graph": True,
        "proposal_head": "optimized" if draft else "full",
        "retain_token_ids": True, "isolate_prompt_decode": False,
        "decode_path": "dflash_device_graph" if draft else "device_graph",
        "repetitions": 3, "warmup": 1
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(f"config differs for {stem}: {key}={config.get(key)!r}")
    prime = config.get("decode_graph_prime", {})
    expected_prime_tokens = 3 if draft == 0 else 2 * draft + 3
    if prime != {"primed": True, "output_tokens": expected_prime_tokens}:
        fail(f"wrong graph family was primed: {stem}")
    memory = report.get("memory", {})
    workspace_capacity = memory.get("workspace", {}).get("capacity_bytes")
    if not isinstance(workspace_capacity, int) or memory.get("device_graph_observed_bytes", 1) > memory.get("device_graph_allowance_bytes", 0):
        fail(f"graph capacity differs: {stem}")
    tests = report.get("tests", [])
    if (len(tests) != 1 or tests[0].get("label") != "whole-pp128+tg64" or
            tests[0].get("kind") != "whole" or tests[0].get("n_prompt") != 128 or
            tests[0].get("n_gen") != 64 or tests[0].get("requested_output_tokens") != 65):
        fail(f"test geometry differs: {stem}")
    test = tests[0]
    if test.get("workspace_peak_bytes", workspace_capacity + 1) > workspace_capacity:
        fail(f"workspace capacity exceeded: {stem}")
    reps = test.get("reps", [])
    if len(reps) != 3:
        fail(f"repetition count differs: {stem}")
    sequences, times, accounting = [], {"total_seconds": [], "decode_seconds": [], "prefill_seconds": []}, []
    for rep in reps:
        if (rep.get("generated_output_tokens") != 65 or rep.get("decode_output_tokens") != 64 or
                rep.get("decode_engine_tokens") != 64):
            fail(f"public output accounting differs: {stem}")
        lanes = rep.get("generated_token_ids_by_lane")
        if not isinstance(lanes, list) or len(lanes) != 1 or len(lanes[0]) != 65:
            fail(f"public token shape differs: {stem}")
        sequences.append(tuple(lanes[0]))
        total = rep.get("timings", {}).get("total_seconds")
        if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
            fail(f"timing differs: {stem}")
        for metric in times:
            value = rep.get("timings", {}).get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                fail(f"invalid {metric}: {stem}")
            times[metric].append(float(value))
        spec = rep.get("speculative")
        if not isinstance(spec, dict) or spec.get("enabled") is not (draft > 0) or spec.get("draft_window") != draft:
            fail(f"speculative route identity differs: {stem}")
        counts = [spec.get(key) for key in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            fail(f"speculative counts differ: {stem}")
        positions = spec.get("accepted_per_position")
        if (not isinstance(positions, list) or len(positions) != draft or
                any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in positions) or
                sum(positions) != counts[2] or counts[2] > counts[1]):
            fail(f"speculative per-position accounting differs: {stem}")
        if draft > 0 and (counts[0] <= 0 or counts[1] <= 0):
            fail(f"DFlash was not exercised: {stem}")
        if draft == 0 and (any(counts) or positions):
            fail(f"ordinary control has speculative activity: {stem}")
        fields = (*counts, tuple(positions))
        accounting.append(fields)
    if len(set(sequences)) != 1 or len(set(accounting)) != 1:
        fail(f"repetitions are not deterministic: {stem}")
    return list(sequences[0]), times, accounting[0], workspace_capacity



def speed_gate(records, metric):
    ratios, orders = [], []
    for left, right in ((records[0], records[1]), (records[2], records[3])):
        control, candidate = (left, right) if left["role"] == "control" else (right, left)
        c, b = candidate["times"][metric], control["times"][metric]
        orders.append(median(c) / median(b))
        ratios.extend(x / y for x, y in zip(c, b))
    upper = statistics.mean(ratios) + 2 * statistics.stdev(ratios) / math.sqrt(len(ratios))
    material = median(ratios) <= 0.99
    admitted = all(r < 1 for r in orders) and upper < 1 and material and abs(orders[0] - orders[1]) <= 0.02
    control = [v for r in records if r["role"] == "control" for v in r["times"][metric]]
    candidate = [v for r in records if r["role"] == "candidate" for v in r["times"][metric]]
    count = 64 if metric == "decode_seconds" else 65
    return {"admitted": admitted, "launch_order_median_ratios": orders,
            "paired_ratio_median": median(ratios), "paired_ratio_upper_2se": upper,
            "control_seconds": control, "candidate_seconds": candidate,
            "control_tokens_per_second": count / median(control),
            "candidate_tokens_per_second": count / median(candidate)}


def analyze():
    ordinary, _, _, capacity = validate_report("ordinary-control", "control", 0, 0)
    candidate, _, _, candidate_capacity = validate_report("ordinary-candidate", "candidate", 0, 0)
    if candidate != ordinary or capacity != candidate_capacity:
        fail("ordinary exact tokens or workspace capacity changed")
    cells = {}
    for profile, (draft, width) in PROFILES.items():
        records = []
        for index, role in enumerate(ORDERS[profile], 1):
            tokens, times, accounting, capacity = validate_report(f"{profile}-{index}-{role}", role, draft, width)
            if tokens != ordinary:
                fail(f"public token parity failed: {profile}")
            records.append({"role": role, "times": times, "accounting": accounting, "capacity": capacity})
        if len({r["accounting"] for r in records}) != 1 or len({r["capacity"] for r in records}) != 1:
            fail(f"accounting or workspace capacity changed: {profile}")
        gates = {metric: speed_gate(records, metric) for metric in ("decode_seconds", "total_seconds")}
        cells[profile] = {"draft_tokens": draft, "verify_width": width, "gates": gates,
                          "admitted": all(g["admitted"] for g in gates.values()),
                          "speculative_accounting": records[0]["accounting"],
                          "workspace_capacity_bytes": records[0]["capacity"],
                          "prefill_seconds": {role: [v for r in records if r["role"] == role
                              for v in r["times"]["prefill_seconds"]] for role in ("control", "candidate")}}
    return {"status": "completed", "disposition": "admitted" if all(c["admitted"] for c in cells.values()) else "rejected",
            "cells": cells, "exact_public_token_parity": True, "exact_speculative_accounting": True,
            "ordinary_tokens": ordinary, "production_routing_authorized": False,
            "passing_authorizes": "promotion review of C1 K4/W5 and K5/W6 only"}


def json_write(path, value):
    write_exclusive(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def measure():
    import preflight
    preflight.verify()
    RESULTS.mkdir()
    started = time.time_ns()
    status, disposition, error = "failed", "not_admitted", None
    json_write(RESULTS / "plan-identity.json", identity(PACKAGE / "plan.json"))
    try:
        ordinary = None
        for role in ("control", "candidate"):
            stem = f"ordinary-{role}"
            run_one(stem, role)
            tokens, _, _, _ = validate_report(stem, role, 0, 0)
            if ordinary is not None and tokens != ordinary:
                fail("ordinary public tokens changed")
            ordinary = tokens
        for profile, (draft, width) in PROFILES.items():
            expected_accounting, expected_capacity = None, None
            for index, role in enumerate(ORDERS[profile], 1):
                stem = f"{profile}-{index}-{role}"
                run_one(stem, role, draft, width)
                tokens, _, accounting, capacity = validate_report(stem, role, draft, width)
                if tokens != ordinary:
                    fail(f"public token parity failed: {stem}")
                if expected_accounting is not None and (accounting != expected_accounting or capacity != expected_capacity):
                    fail(f"accounting or workspace capacity changed: {stem}")
                expected_accounting, expected_capacity = accounting, capacity
        summary = analyze()
        json_write(RESULTS / "summary.json", summary)
        status, disposition = summary["status"], summary["disposition"]
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        artifacts = [identity(p) for p in sorted(RESULTS.rglob("*")) if p.is_file()]
        json_write(RESULTS / "closure.json", {
            "status": status, "disposition": disposition, "error": error,
            "started_unix_ns": started, "finished_unix_ns": time.time_ns(),
            "plan": identity(PACKAGE / "plan.json"), "artifacts": artifacts,
            "production_routing_authorized": False})
        files = [p for p in sorted(RESULTS.rglob("*")) if p.is_file()]
        write_exclusive(RESULTS / "result.sha256", "".join(
            f"{digest(p)}  {p.relative_to(RESULTS)}\n" for p in files))
    print(json.dumps({"status": status, "disposition": disposition}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "measure"))
    args = parser.parse_args()
    if args.action == "measure":
        measure()
    else:
        import preflight
        if args.action == "prepare":
            preflight.prepare()
        else:
            preflight.verify()
