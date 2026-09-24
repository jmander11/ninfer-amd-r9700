#!/usr/bin/env python3
"""Validate the exact d1a9b6fb MLP-down T5 matched CPU builds."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-dflash-mlp-down-t5-matched-builds-20260906"
COMMIT = "d1a9b6fb33a843363fba3ad46d6569071df1ab66"
TREE = "557bbda04f2926319bf7c25f26eb9ec193d74824"


def fail(message: str) -> None:
    raise ValueError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(value, dict):
        fail(f"expected object: {path}")
    return value


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(record: dict) -> Path:
    path = Path(record["path"])
    if (not path.is_file() or path.stat().st_size != record["bytes"] or
            sha(path) != record["sha256"]):
        fail(f"identity differs: {path}")
    return path


def cache(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        name_type, value = line.split("=", 1)
        values[name_type.split(":", 1)[0]] = value
    return values


def normalized_compile(path: Path, build: Path) -> bytes:
    value = path.read_text(encoding="utf-8").replace(str(build), "<BUILD>")
    value = re.sub(r"NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE=[01]",
                   "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE=<SELECTOR>", value)
    return value.encode()


def relocation_count(tool: Path, obj: Path, symbol: str) -> int:
    output = subprocess.check_output([str(tool), "-dr", "--demangle", str(obj)], text=True)
    return sum("R_X86_64_PLT32" in line and symbol in line for line in output.splitlines())


def main() -> int:
    plan = load(PACKAGE / "plan.json")
    receipt = load(PACKAGE / "build-receipt.json")
    if (plan.get("artifact_type") != "ninfer_r9700_dflash_mlp_down_t5_matched_build_plan" or
            plan.get("schema_version") != 1 or
            plan.get("production_routing_authorized") is not False or
            plan.get("benchmark_execution") != "not-run"):
        fail("plan identity or no-production boundary differs")
    if (receipt.get("artifact_type") !=
            "ninfer_r9700_dflash_mlp_down_t5_matched_build_receipt" or
            receipt.get("schema_version") != 1 or
            receipt.get("status") != "cpu-build-and-host-checks-passed-no-gpu-execution" or
            receipt.get("production_routing_authorized") is not False or
            receipt.get("benchmark_execution") != "not-run" or
            receipt.get("configs_differ_only_in_dflash_mlp_down_t5_candidate") is not True or
            receipt.get("shared_gate_up_small_t_selector") != 0 or
            receipt.get("selector_domain") !=
            {"tokens": 5, "rows": 5120, "columns": 17408, "padded_columns": 17408} or
            receipt.get("T6_candidate") != "rejected-static-incumbent-WMMA"):
        fail("receipt boundary differs")
    source = receipt.get("source", {})
    if source != plan.get("source"):
        fail("plan and receipt source differ")
    checkout = Path(source.get("detached_checkout", ""))
    if (source.get("commit") != COMMIT or source.get("tree") != TREE or
            subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"],
                                    text=True).strip() != COMMIT or
            subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD^{tree}"],
                                    text=True).strip() != TREE or
            subprocess.check_output(["git", "-C", checkout, "status", "--porcelain"],
                                    text=True)):
        fail("detached source is not exact and clean")
    shared = plan["shared_configuration"]
    expected_cache = {key: ("ON" if value is True else "OFF" if value is False else str(value))
                      for key, value in shared.items() if key != "CMAKE_GENERATOR"}
    normalized = []
    objects = {}
    for role, selector in (("control", 0), ("candidate", 1)):
        record = receipt["builds"][role]
        build = Path(record["directory"])
        if plan.get("builds", {}).get(role) != {
                "directory": str(build),
                "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE": selector}:
            fail(f"{role} plan and receipt build differ")
        if (record.get("NINFER_R9700_DFLASH_SMALL_T_CANDIDATE") != 0 or
                record.get("NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE") != selector or
                record.get("cpu_checks") != {
                    "ninfer_bench_support_test": "passed",
                    "ninfer_r9700_linear_prefill_dispatch_test": "passed"} or
                record.get("benchmark_report_field_verified") != {
                    "field": "dflash_mlp_down_t5_candidate", "value": bool(selector),
                    "test": "ninfer_bench_support_test"}):
            fail(f"{role} selector/report contract differs")
        executable = identity(record["benchmark_executable"])
        cache_path = identity(record["cmake_cache"])
        compile_path = identity(record["compile_commands"])
        objects[role] = identity(record["linear_object"])
        values = cache(cache_path)
        if values.get("CMAKE_GENERATOR") != shared["CMAKE_GENERATOR"] or \
                any(values.get(key) != value for key, value in expected_cache.items()) or \
                values.get("NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE") != str(selector):
            fail(f"{role} CMake profile differs")
        compiled = normalized_compile(compile_path, build)
        expected_normalized = record["normalized_compile_commands"]
        if (len(compiled) != expected_normalized["bytes"] or
                hashlib.sha256(compiled).hexdigest() != expected_normalized["sha256"]):
            fail(f"{role} normalized compile identity differs")
        normalized.append(compiled)
        environment = os.environ.copy()
        environment["HIP_VISIBLE_DEVICES"] = "-1"
        for test in ("ninfer_bench_support_test", "ninfer_r9700_linear_prefill_dispatch_test"):
            subprocess.run([str(build / "tests" / test)], check=True, env=environment,
                           stdout=subprocess.DEVNULL)
        if not executable.stat().st_mode & 0o111:
            fail(f"{role} benchmark is not executable")
    if normalized[0] != normalized[1]:
        fail("compile commands differ beyond build path and MLP-down selector")
    proof = receipt["dispatch_relocation_proof"]
    tool = identity(proof["tool"])
    mlp_symbol = "a8q4g64_linear_dflash_mlp_down_t5_qualification"
    gate_symbol = "a8q4g64_linear_dflash_small_t_qualification"
    observed_mlp = {role: relocation_count(tool, obj, mlp_symbol) for role, obj in objects.items()}
    observed_gate = {role: relocation_count(tool, obj, gate_symbol) for role, obj in objects.items()}
    if (observed_mlp != proof["MLP_down_T5_qualification_calls"] or
            observed_gate != proof["gate_up_small_T_qualification_calls"]):
        fail("dispatch relocation proof differs")
    for record in receipt.get("toolchain", {}).values():
        identity(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
