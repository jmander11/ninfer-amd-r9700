#!/usr/bin/env python3
"""Fail-closed validation of the exact matched N16 DFlash A/B builds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-dflash-small-t-n16-matched-builds-20260906"
COMMIT = "cd966d72ed18e1b5b7b57b664572b3c8aa1e02ce"
TREE = "a8cf2ffcef1646d2c39128eab72c5808c889cbcd"


def load(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_identity(identity: dict) -> Path:
    path = Path(identity["path"])
    if (not path.is_file() or path.stat().st_size != identity["bytes"] or
            sha256(path) != identity["sha256"]):
        raise ValueError(f"identity differs: {path}")
    return path


def cache_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or ":" not in line or "=" not in line:
            continue
        name_type, value = line.split("=", 1)
        name = name_type.split(":", 1)[0]
        values[name] = value
    return values


def normalized_compile(path: Path, build: Path) -> bytes:
    value = path.read_text(encoding="utf-8").replace(str(build), "<BUILD>")
    value = re.sub(
        r"NINFER_R9700_DFLASH_SMALL_T_CANDIDATE=[01]",
        "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE=<SELECTOR>",
        value,
    )
    return value.encode()


def main() -> int:
    plan = load(PACKAGE / "plan.json")
    receipt = load(PACKAGE / "build-receipt.json")
    if (receipt.get("artifact_type") != "ninfer_r9700_dflash_small_t_matched_build_receipt" or
            receipt.get("schema_version") != 1 or
            receipt.get("status") != "cpu-build-and-host-checks-passed-no-gpu-execution" or
            receipt.get("source", {}).get("commit") != COMMIT or
            receipt.get("source", {}).get("tree") != TREE or
            receipt.get("configs_differ_only_in_dflash_small_t_candidate") is not True or
            receipt.get("benchmark_execution") != "not-run" or
            receipt.get("maximum_supported_benchmark_concurrency") != 4):
        raise ValueError("receipt authority differs")
    if (receipt.get("shared_configuration") != plan.get("shared_configuration") or
            receipt.get("expected_benchmark_profile") != plan.get("expected_benchmark_profile")):
        raise ValueError("receipt shared production profile differs from plan")
    checkout = Path(receipt["source"]["detached_checkout"])
    if (subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
            != COMMIT or subprocess.check_output(
                ["git", "-C", checkout, "rev-parse", "HEAD^{tree}"], text=True
            ).strip() != TREE or subprocess.check_output(
                ["git", "-C", checkout, "status", "--porcelain"], text=True
            )):
        raise ValueError("detached source checkout differs")
    expected_cache = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
        "CMAKE_HIP_COMPILER": "/opt/rocm/llvm/bin/clang++",
        "NINFER_BUILD_APPS": "ON", "NINFER_BUILD_BENCHMARKS": "ON",
        "NINFER_BUILD_R9700_CORE_QUALIFIER": "ON",
        "NINFER_R9700_KV_VALUE_GROUP": "16", "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
        "NINFER_R9700_W8_ACTIVATION_BITS": "8", "NINFER_R9700_FP8_QK_WMMA": "1",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
        "NINFER_R9700_XATTENTION_STRIDE": "16",
        "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000",
    }
    normalized = []
    for role, selector in (("control", 0), ("candidate", 1)):
        build = receipt["builds"][role]
        if (build.get("directory") != plan["builds"][role]["directory"] or
                build.get("NINFER_R9700_DFLASH_SMALL_T_CANDIDATE") != selector or
                build.get("cpu_checks") != {
                    "ninfer_bench_support_test": "passed",
                    "ninfer_r9700_linear_prefill_dispatch_test": "passed",
                }):
            raise ValueError(f"{role} build contract differs")
        executable = validate_identity(build["benchmark_executable"])
        cache = validate_identity(build["cmake_cache"])
        compile_commands = validate_identity(build["compile_commands"])
        values = cache_values(cache)
        if any(values.get(key) != value for key, value in expected_cache.items()):
            raise ValueError(f"{role} shared cache profile differs")
        if values.get("NINFER_R9700_DFLASH_SMALL_T_CANDIDATE") != str(selector):
            raise ValueError(f"{role} candidate selector differs")
        compiled = normalized_compile(compile_commands, Path(build["directory"]))
        normalized_identity = build["normalized_compile_commands"]
        if (len(compiled) != normalized_identity["bytes"] or
                hashlib.sha256(compiled).hexdigest() != normalized_identity["sha256"]):
            raise ValueError(f"{role} normalized compile identity differs")
        normalized.append(compiled)
        env = {"HIP_VISIBLE_DEVICES": "-1"}
        for check in ("ninfer_bench_support_test", "ninfer_r9700_linear_prefill_dispatch_test"):
            subprocess.run(
                [str(Path(build["directory"]) / "tests" / check)],
                check=True, env=env, stdout=subprocess.DEVNULL,
            )
        if not executable.stat().st_mode & 0o111:
            raise ValueError(f"{role} benchmark is not executable")
    if normalized[0] != normalized[1]:
        raise ValueError("normalized compile configurations differ beyond candidate selector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
