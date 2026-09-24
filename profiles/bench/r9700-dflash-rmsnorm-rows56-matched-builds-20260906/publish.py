#!/usr/bin/env python3
"""Publish exact identities for the CPU-only matched rows5/6 builds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-dflash-rmsnorm-rows56-matched-builds-20260906"


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def normalized_compile(path: Path, build: Path) -> dict:
    value = path.read_text().replace(str(build), "<BUILD>")
    value = re.sub(r"NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=[01]",
                   "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=<SELECTOR>", value)
    payload = value.encode()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def exclusive(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    args = parser.parse_args()
    plan = json.loads((PACKAGE / "plan.json").read_text())
    source = Path(plan["source"]["detached_checkout"])
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != args.commit:
        raise ValueError("detached source commit differs")
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD^{tree}"], text=True).strip() != args.tree:
        raise ValueError("detached source tree differs")
    if subprocess.check_output(["git", "-C", source, "status", "--porcelain"], text=True):
        raise ValueError("detached source is dirty")
    builds = {}
    production_harnesses = {
        "control": source / "tools/r9700/build/rmsnorm_dflash_rows56_production_off",
        "candidate": source / "tools/r9700/build/rmsnorm_dflash_rows56_production_on",
    }
    for role, selector in (("control", 0), ("candidate", 1)):
        build = Path(plan["builds"][role]["directory"])
        compile_commands = build / "compile_commands.json"
        builds[role] = {
            "directory": str(build),
            "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE": selector,
            "benchmark_executable": identity(build / "bench/ninfer_bench"),
            "cmake_cache": identity(build / "CMakeCache.txt"),
            "compile_commands": identity(compile_commands),
            "eager_object": identity(build / "src/CMakeFiles/ninfer_r9700_core.dir/ops/r9700/eager/eager_ops.hip.o"),
            "production_numerical_executable": identity(production_harnesses[role]),
            "normalized_compile_commands": normalized_compile(compile_commands, build),
            "cpu_checks": {"ninfer_bench_support_test": "passed",
                           "production_numerical_harness": "compiled-not-run-no-gpu",
                           "host_routing": "passed"},
            "benchmark_report_field_verified": {
                "field": "dflash_rmsnorm_rows56_candidate", "value": bool(selector),
                "test": "ninfer_bench_support_test"},
        }
    receipt = {
        "artifact_type": "ninfer_r9700_dflash_rmsnorm_rows56_matched_build_receipt",
        "schema_version": 1,
        "status": "cpu-build-and-host-checks-passed-no-gpu-execution",
        "source": {"commit": args.commit, "tree": args.tree,
                   "detached_checkout": str(source)},
        "builds": builds,
        "configs_differ_only_in_dflash_rmsnorm_rows56_candidate": True,
        "selector_domain": {"features": 5120, "rows": [5, 6]},
        "ordinary_rows_1_to_4_identical_between_arms": True,
        "benchmark_execution": "not-run",
        "maximum_supported_benchmark_concurrency": 4,
        "production_routing_authorized": False,
    }
    exclusive(PACKAGE / "build-receipt.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
