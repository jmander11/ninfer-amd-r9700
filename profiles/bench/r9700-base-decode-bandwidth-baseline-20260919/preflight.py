#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
POWER = Path("/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level")
EXACT_ENV = {"LD_PRELOAD", "LD_AUDIT", "HSA_TOOLS_LIB", "ROCPROFILER_TOOL_LIBRARIES",
             "ROCP_TOOL_LIBRARIES", "HIP_FORCE_QUEUE_PROFILING", "AMD_SERIALIZE_KERNEL",
             "AMD_SERIALIZE_COPY", "ROC_SERIALIZE_KERNEL"}
PREFIX_ENV = ("ROCPROF", "ROCP_", "ROCTRACER_", "ROCTX_", "HSA_TOOLS_", "HIP_TRACE_",
              "AQLPROFILE_", "ATT_PROFILE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact(record: dict) -> None:
    path = Path(record["path"]).resolve(strict=True)
    if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        fail(f"identity differs: {path}")


def cache_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("//", "#")):
            values[line.split(":", 1)[0]] = line.split("=", 1)[1]
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "measure", "profile"), default="all")
    args = parser.parse_args()
    plan_path = PACKAGE / "bound-plan.json"
    if not plan_path.is_file():
        fail("prepare.py has not created bound-plan.json")
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "prepared_awaiting_independent_review_and_gpu_execution" or "UNBOUND" in json.dumps(plan):
        fail("bound plan is not runnable")
    outputs = ([PACKAGE / "results-unprofiled"] if args.mode == "measure" else
               [PACKAGE / "results-profile"] if args.mode == "profile" else
               [PACKAGE / "results-unprofiled", PACKAGE / "results-profile"])
    for output in outputs:
        if output.exists() or output.is_symlink():
            fail(f"output already exists; never overwrite: {output}")
    for key in ("build_cache", "benchmark", "stream_probe", "artifact",
                "corpus", "profiler", "counter_preflight", "retained_cold_block_sweep",
                "retained_token_authority"):
        exact(plan[key])
    source = Path(plan["source_worktree"]).resolve(strict=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != plan["source_commit"]:
        fail("source commit differs")
    if subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=source, text=True).strip() != plan["source_tree"]:
        fail("source tree differs")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        fail("bound source worktree is dirty")
    values = cache_values(Path(plan["build_cache"]["path"]))
    for key, value in plan["selector_free_cache"].items():
        if values.get(key) != value:
            fail(f"selector-free cache differs: {key}")
    injected = sorted(key for key in os.environ if key in EXACT_ENV or key.startswith(PREFIX_ENV))
    if injected:
        fail(f"profiling or interception environment is present: {injected}")
    if POWER.read_text().strip() != "auto":
        fail("R9700 power profile is not auto")
    pci = POWER.parent
    if (pci / "vendor").read_text().strip() != "0x1002" or (pci / "device").read_text().strip() != "0x7551":
        fail("GPU0 PCI identity differs")
    print(f"r9700_base_decode_bandwidth_preflight: PASS mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
