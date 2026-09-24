#!/usr/bin/env python3
"""One C1/P8192/chunk1024 selected-region trace; attribution, never speed admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.bench.analyze_whole_profile import analyze
from tools.bench.run_ninfer_bench_matrix import (
    R9700_POWER_PROFILE, require_auto_power_profile, require_hip_pci_device,
)
from tools.ppl.run import file_sha256, inspect_candidate_artifact

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE / "run"
FROZEN = REPO / "profiles/bench/r9700-chunk-selection-receipt-bound-n16k16-20260921/inputs.json"
BENCH = REPO / "build-r9700-selection-dense-g16-20260921/bench/ninfer_bench"
WEIGHTS_ID = "r9700-q4g64-n16k16-eval"
WEIGHTS = REPO / f"out/qwen3.8-27b-{WEIGHTS_ID}.ninfer"
CORPUS = REPO / "bench/fixtures/bench_corpus.ids"
PROFILER = Path("/opt/rocm/core-10.0/bin/rocprofv3")
PREFIX = "chunked-prefill"


def commands():
    benchmark = [str(BENCH), "--weights", str(WEIGHTS), "--corpus", str(CORPUS),
                 "--device", "0", "--concurrency", "1", "-p", "8192",
                 "--prefill-chunk", "1024", "--draft-tokens", "0",
                 "--output", "json", "--output-file", str(OUTPUT / "benchmark-report.json"),
                 "-r", "1", "--warmup", "0", "--profile-measured"]
    profiler = [str(PROFILER), "--selected-regions", "-f", "rocpd", "-d",
                str(OUTPUT / "raw"), "-o", PREFIX, "--marker-trace", "--kernel-trace",
                "--memory-copy-trace", "--", *benchmark]
    return benchmark, profiler


def preflight():
    if os.path.lexists(OUTPUT):
        raise ValueError(f"trace output already exists; preserve the previous attempt: {OUTPUT}")
    frozen = json.loads(FROZEN.read_text())
    if (frozen.get("artifact_type") != "ninfer_r9700_chunk_campaign_inputs"
            or frozen.get("schema_version") != 1):
        raise ValueError("unsupported frozen chunk input receipt")
    for path in (BENCH, CORPUS):
        if file_sha256(path) != frozen["files"].get(str(path.relative_to(REPO))):
            raise ValueError(f"trace input differs from frozen chunk campaign: {path}")
    artifact = inspect_candidate_artifact(WEIGHTS)
    expected = [item for item in frozen["artifacts"] if item["weights_id"] == WEIGHTS_ID]
    if expected != [artifact]:
        raise ValueError("trace artifact/receipt differs from frozen chunk campaign")
    for executable in (BENCH, PROFILER):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"executable unavailable: {executable}")
    require_auto_power_profile()
    return {"frozen_chunk_inputs": {"path": str(FROZEN), "sha256": file_sha256(FROZEN)},
            "artifact": artifact, "benchmark_sha256": file_sha256(BENCH),
            "corpus_sha256": file_sha256(CORPUS),
            "profiler": {"path": str(PROFILER), "sha256": file_sha256(PROFILER)}}


def idle():
    require_auto_power_profile()
    device = R9700_POWER_PROFILE.parent
    used = int((device / "mem_info_vram_used").read_text())
    busy = int((device / "gpu_busy_percent").read_text())
    if used > 1024 ** 3 or busy != 0:
        raise ValueError(f"R9700 must be idle before profiling: resident={used} bytes, busy={busy}%")


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def close_trace():
    path = OUTPUT / "closure.json"
    if os.path.lexists(path):
        raise ValueError(f"closure already exists: {path}")
    execution = json.loads((OUTPUT / "execution.json").read_text())
    benchmark, profiler = commands()
    if (execution["benchmark_command"] != benchmark or execution["profiler_command"] != profiler
            or execution["power_before"] != "auto" or execution["power_after"] != "auto"
            or execution["returncode"] != 0):
        raise ValueError("trace execution receipt does not match the completed workload")
    database = OUTPUT / "raw" / f"{PREFIX}_results.db"
    report = OUTPUT / "benchmark-report.json"
    result = analyze(database, report)
    workload = result["workload"]
    if (workload["kind"] != "pp" or workload["prompt_tokens"] != 8192
            or workload["concurrency"] != 1 or workload["prefill_chunk"] != 1024
            or result["ranges"]["text_prefill_chunks"] != 8
            or result["ranges"]["mtp_prefill_ranges"] != 0):
        raise ValueError("trace is not C1 P8192 chunk1024 ordinary prefill with eight chunks")
    write_new(path, {"artifact_type": "ninfer_chunked_prefill_attribution", "schema_version": 1,
                    "purpose": "attribute appended-chunk attention cost; profiler timings are not speed admission",
                    "execution": execution, "benchmark_report_sha256": file_sha256(report),
                    "database_sha256": file_sha256(database), "attribution": result})


def execute(inputs):
    idle()
    require_hip_pci_device(0)
    # The ordinal probe itself briefly activates the GPU; its utilization sample
    # can outlive the exited probe process. The exclusive-use check precedes it.
    require_auto_power_profile()
    benchmark, profiler = commands()
    OUTPUT.mkdir(exist_ok=False)
    execution = {"inputs": inputs, "benchmark_command": benchmark, "profiler_command": profiler,
                 "pci": "0000:13:00.0", "power_before": require_auto_power_profile()}
    write_new(OUTPUT / "launch.json", execution)
    completed = subprocess.run(profiler, check=False)
    execution.update(returncode=completed.returncode,
                     power_after=R9700_POWER_PROFILE.read_text().strip())
    write_new(OUTPUT / "execution.json", execution)
    if completed.returncode:
        raise ValueError(f"profiler failed ({completed.returncode}); partial outputs retained")
    require_auto_power_profile()
    close_trace()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "analyze"))
    args = parser.parse_args()
    if args.stage == "analyze":
        close_trace()
    else:
        inputs = preflight()
        if args.stage == "run":
            execute(inputs)
        else:
            print("trace inputs and PCI auto power verified; no outputs or GPU work launched")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
