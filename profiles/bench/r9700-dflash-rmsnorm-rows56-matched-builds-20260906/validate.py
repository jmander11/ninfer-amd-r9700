#!/usr/bin/env python3
"""Fail-closed validation for matched rows5/6 RMSNorm CPU builds."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-dflash-rmsnorm-rows56-matched-builds-20260906"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(record: dict) -> Path:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise ValueError(f"identity differs: {path}")
    return path


def cache(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def normalized(path: Path, build: Path) -> bytes:
    value = path.read_text().replace(str(build), "<BUILD>")
    return re.sub(r"NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=[01]",
                  "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=<SELECTOR>", value).encode()


def main() -> int:
    plan = json.loads((PACKAGE / "plan.json").read_text())
    receipt = json.loads((PACKAGE / "build-receipt.json").read_text())
    if plan.get("artifact_type") != "ninfer_r9700_dflash_rmsnorm_rows56_matched_build_plan" or \
            plan.get("schema_version") != 1 or \
            plan.get("status") != "prepared-awaiting-committed-source" or \
            plan.get("production_routing_authorized") is not False or \
            plan.get("benchmark_execution") != "not-run":
        raise ValueError("plan boundary differs")
    if receipt.get("artifact_type") != "ninfer_r9700_dflash_rmsnorm_rows56_matched_build_receipt" or \
            receipt.get("schema_version") != 1 or \
            receipt.get("status") != "cpu-build-and-host-checks-passed-no-gpu-execution" or \
            receipt.get("production_routing_authorized") is not False or \
            receipt.get("benchmark_execution") != "not-run" or \
            receipt.get("configs_differ_only_in_dflash_rmsnorm_rows56_candidate") is not True or \
            receipt.get("selector_domain") != {"features": 5120, "rows": [5, 6]}:
        raise ValueError("receipt boundary differs")
    if plan.get("production_numerical_harness") != {
            "both_selector_states": "rows1..6 eager through public rmsnorm_bf16",
            "graph_rows": [5, 6], "unit_offset": [False, True],
            "epsilon": [1e-8, 1e-6, 1e-4],
            "execution": "compiled now; GPU execution remains a separate prerequisite"}:
        raise ValueError("production numerical harness scope differs")
    source = Path(receipt["source"]["detached_checkout"])
    for expression, expected in (("HEAD", receipt["source"]["commit"]),
                                 ("HEAD^{tree}", receipt["source"]["tree"])):
        if subprocess.check_output(["git", "-C", source, "rev-parse", expression], text=True).strip() != expected:
            raise ValueError("source identity differs")
    if subprocess.check_output(["git", "-C", source, "status", "--porcelain"], text=True):
        raise ValueError("source checkout is dirty")
    shared = plan["shared_configuration"]
    expected_cache = {key: ("ON" if value is True else "OFF" if value is False else str(value))
                      for key, value in shared.items() if key != "CMAKE_GENERATOR"}
    normalized_commands = []
    eager_hashes = []
    for role, selector in (("control", 0), ("candidate", 1)):
        record = receipt["builds"][role]
        build = Path(record["directory"])
        if plan["builds"][role] != {
                "directory": str(build),
                "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE": selector} or \
                record["NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE"] != selector or \
                record.get("cpu_checks") != {"ninfer_bench_support_test": "passed",
                                              "production_numerical_harness": "compiled-not-run-no-gpu",
                                              "host_routing": "passed"} or \
                record.get("benchmark_report_field_verified") != {
                    "field": "dflash_rmsnorm_rows56_candidate", "value": bool(selector),
                    "test": "ninfer_bench_support_test"}:
            raise ValueError(f"{role} selector/report contract differs")
        for key in ("benchmark_executable", "cmake_cache", "compile_commands", "eager_object"):
            identity(record[key])
        expected_harness = source / "tools/r9700/build" / (
            "rmsnorm_dflash_rows56_production_off" if role == "control" else
            "rmsnorm_dflash_rows56_production_on")
        if identity(record["production_numerical_executable"]) != expected_harness:
            raise ValueError(f"{role} production harness identity differs")
        values = cache(Path(record["cmake_cache"]["path"]))
        if values.get("CMAKE_GENERATOR") != shared["CMAKE_GENERATOR"] or \
                any(values.get(key) != value for key, value in expected_cache.items()) or \
                values.get("NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE") != str(selector):
            raise ValueError(f"{role} CMake profile differs")
        compiled = normalized(Path(record["compile_commands"]["path"]), build)
        if {"bytes": len(compiled), "sha256": hashlib.sha256(compiled).hexdigest()} != \
                record["normalized_compile_commands"]:
            raise ValueError(f"{role} normalized compile identity differs")
        normalized_commands.append(compiled)
        eager_hashes.append(record["eager_object"]["sha256"])
        environment = os.environ.copy()
        environment["HIP_VISIBLE_DEVICES"] = "-1"
        subprocess.run([str(build / "tests/ninfer_bench_support_test")], check=True,
                       env=environment, stdout=subprocess.DEVNULL)
    if normalized_commands[0] != normalized_commands[1]:
        raise ValueError("compile commands differ beyond build path and RMSNorm selector")
    if eager_hashes[0] == eager_hashes[1]:
        raise ValueError("RMSNorm selector did not change the eager object")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
