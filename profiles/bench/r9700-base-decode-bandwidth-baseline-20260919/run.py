#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
RESULTS = PACKAGE / "results-unprofiled"
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


def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)


def identity(path: Path) -> dict:
    return {"path": str(path.resolve(strict=True)), "bytes": path.stat().st_size, "sha256": digest(path)}


def benchmark_command(index: int) -> list[str]:
    plan = load(PACKAGE / "bound-plan.json")
    return [plan["benchmark"]["path"], "--weights", plan["artifact"]["path"],
            "--corpus", plan["corpus"]["path"], "--device", "0", "--concurrency", "1",
            "--whole-pg", "8192,256", "--prefill-chunk", "4096", "--kv-capacity", "workload",
            "--spec", "mtp", "--draft-tokens", "0", "--retain-token-ids", "--output", "json",
            "--output-file", str(RESULTS / f"baseline-{index}.json"), "-r", "1", "--warmup", "1"]


def run_capture(stem: str, command: list[str]) -> None:
    if POWER.read_text().strip() != "auto":
        fail(f"power is not auto before {stem}")
    started = time.time_ns()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    finished = time.time_ns()
    stdout, stderr = RESULTS / f"{stem}.stdout", RESULTS / f"{stem}.stderr"
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(RESULTS / f"{stem}.process.json", json.dumps({
        "command": command, "exit_code": completed.returncode,
        "started_unix_ns": started, "finished_unix_ns": finished,
        "power_before": "auto", "power_after": POWER.read_text().strip(),
        "stdout": identity(stdout), "stderr": identity(stderr)
    }, indent=2) + "\n")
    if completed.returncode or POWER.read_text().strip() != "auto":
        fail(f"process failed: {stem}")


def validate_baseline(index: int) -> tuple[tuple[int, ...], float, float]:
    stem = f"baseline-{index}"
    report = load(RESULTS / f"{stem}.json")
    process = load(RESULTS / f"{stem}.process.json")
    command = benchmark_command(index)
    if process.get("command") != command or process.get("exit_code") != 0:
        fail(f"process provenance differs: {stem}")
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("environment", {}).get("gpu_name") != "AMD Radeon AI PRO R9700" or
            report.get("environment", {}).get("architecture_name") != "gfx1201"):
        fail(f"benchmark provenance differs: {stem}")
    config = report.get("config", {})
    expected = {"concurrency": 1, "prefill_chunk": 4096, "kv_cache_format": "fp8-k-int4-v",
                "kv_value_group": 16, "q4_activation_bits": 8, "w8_activation_bits": 8,
                "fp8_qk_wmma_enabled": True, "xattention_qualification": False,
                "spec": "none", "draft_tokens": 0, "use_device_graph": True,
                "retain_token_ids": True, "decode_path": "device_graph", "repetitions": 1,
                "warmup": 1}
    for key, value in expected.items():
        if config.get(key) != value:
            fail(f"config differs {stem}: {key}={config.get(key)!r}")
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
    if not isinstance(lanes, list) or len(lanes) != 1 or len(lanes[0]) != 257:
        fail(f"token shape differs: {stem}")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        fail(f"decode timing differs: {stem}")
    plan = load(PACKAGE / "bound-plan.json")
    token_bytes = json.dumps(lanes[0], separators=(",", ":")).encode()
    if (len(lanes[0]) != plan["expected_tokens"]["count"] or
            hashlib.sha256(token_bytes).hexdigest() != plan["expected_tokens"]["sha256"]):
        fail(f"public tokens differ from retained selector-free authority: {stem}")
    return tuple(lanes[0]), float(seconds), 256.0 / float(seconds)


def validate_cold_sweep(path: Path) -> dict:
    report = load(path)
    if (report.get("schema") != "ninfer.r9700.decode-dot8-cold-block-sweep.v1" or
            report.get("status") != "qualified" or
            report.get("power_profile_before") != "auto" or
            report.get("power_profile_after") != "auto" or
            report.get("block_threads") != [32, 64, 128, 256]):
        fail("cold block sweep provenance differs")
    device = report.get("device", {})
    executable = report.get("executable", {})
    if (device.get("name") != "AMD Radeon AI PRO R9700" or
            device.get("architecture") != "gfx1201" or device.get("wave_size") != 32 or
            not isinstance(executable.get("sha256"), str) or len(executable["sha256"]) != 64 or
            not isinstance(report.get("command"), str) or
            "--decode-dot8-t1-cold-block-sweep --out-json" not in report["command"] or
            len(report.get("sources", [])) != 4):
        fail("cold block sweep source/device identity differs")
    expected = [(4096, 5120), (5120, 6144), (5120, 17408), (7168, 5120),
                (12288, 5120), (34816, 5120), (248320, 5120)]
    shapes = report.get("shapes", [])
    if [(row.get("rows"), row.get("columns")) for row in shapes] != expected:
        fail("cold block sweep shape inventory differs")
    for row in shapes:
        if (row.get("maximum_bf16_steps", 3) > 2 or row.get("weight_copy_count", 0) < 2 or
                row.get("intervening_weight_bytes", 0) <= 64 * 1024 * 1024):
            fail("cold block sweep oracle or disjoint working-set gate failed")
        routes = row.get("routes", [])
        if [route.get("threads") for route in routes] != [32, 64, 128, 256]:
            fail("cold block sweep route inventory differs")
        for route in routes:
            samples = route.get("samples_ms", [])
            if (len(samples) < 5 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                                        not math.isfinite(value) or value <= 0 for value in samples)):
                fail("cold block sweep samples differ")
            if abs(float(route.get("median_ms", -1.0)) - statistics.median(samples)) > 1e-8:
                fail("cold block sweep median differs from samples")
        pipeline = row.get("pipeline", {})
        pipeline_samples = pipeline.get("samples_ms", [])
        if (len(pipeline_samples) < 5 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0 for value in pipeline_samples) or
                abs(float(pipeline.get("median_ms", -1.0)) -
                    statistics.median(pipeline_samples)) > 1e-8):
            fail("cold pipeline sweep samples differ")
    return report


def close_results() -> None:
    files = sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path.name}\n" for path in files))


def main() -> int:
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("results-unprofiled already exists; never overwrite")
    RESULTS.mkdir()
    try:
        plan = load(PACKAGE / "bound-plan.json")
        for index in range(1, 4):
            run_capture(f"baseline-{index}", benchmark_command(index))
        probe = [plan["stream_probe"]["path"], "--size-gib", "4", "--trials", "5"]
        run_capture("stream-probe", probe)
        cold_sweep = validate_cold_sweep(Path(plan["retained_cold_block_sweep"]["path"]))
        validated = [validate_baseline(index) for index in range(1, 4)]
        if len({row[0] for row in validated}) != 1:
            fail("independent baseline processes produced different public tokens")
        rates = [row[2] for row in validated]
        useful = plan["useful_byte_contract"]["bytes_per_round"]
        probe_text = (RESULTS / "stream-probe.stdout").read_text()
        probe_match = re.search(
            r"Best sustained bus rate:\s+([0-9]+(?:\.[0-9]+)?) GB/s \(([0-9]+(?:\.[0-9]+)?)%",
            probe_text)
        if probe_match is None:
            fail("stream probe did not report its sustained bus ceiling")
        stream_gbps = float(probe_match.group(1))
        stream_peak_fraction = float(probe_match.group(2)) / 100.0
        logical_gbps = statistics.median(rates) * useful / 1e9
        summary = {
            "schema": "ninfer.r9700.base-decode-bandwidth-baseline-summary.v1",
            "status": "passed", "source_commit": plan["source_commit"],
            "exact_public_tokens_across_processes": True,
            "decode_seconds": [row[1] for row in validated],
            "decode_output_tok_s": rates,
            "median_decode_output_tok_s": statistics.median(rates),
            "logical_useful_bytes_per_round": useful,
            "median_logical_useful_byte_rate_gbps": logical_gbps,
            "same_session_stream_ceiling_gbps": stream_gbps,
            "same_session_stream_peak_fraction": stream_peak_fraction,
            "logical_useful_rate_fraction_of_stream_ceiling": logical_gbps / stream_gbps,
            "physical_hbm_bandwidth_gbps": None, "stall_freedom": None,
            "stream_probe_stdout": identity(RESULTS / "stream-probe.stdout"),
            "retained_cold_block_sweep": plan["retained_cold_block_sweep"],
            "retained_cold_block_sweep_status": cold_sweep["status"],
            "limitations": [
                "logical useful-byte rate is not physical HBM traffic",
                "the stream probe is an isolated same-session ceiling, not inference traffic",
                "the retained disjoint-rotation Op timing is not physical HBM traffic and cannot prove whole-inference saturation"
            ]
        }
        write_exclusive(RESULTS / "summary.json", json.dumps(summary, indent=2) + "\n")
        close_results()
        print("r9700_base_decode_bandwidth_baseline: PASS")
        return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists():
            close_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
