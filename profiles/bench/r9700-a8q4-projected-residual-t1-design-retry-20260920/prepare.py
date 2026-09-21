#!/usr/bin/env python3
"""Bind this harness-only retry without changing its original evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
ORIGINAL = ROOT / "profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919"
PLAN = PACKAGE / "plan.json"
ATTEMPT = PACKAGE / "attempt-1"
HIPCC = Path("/opt/rocm/bin/hipcc")
QUALIFIER = ROOT / "tools/r9700/a8q4_projected_residual_t1_qual.hip"
CHECKER = ROOT / "tools/r9700/check_a8q4_projected_residual_t1_static.py"
LINEAR = ROOT / "src/ops/r9700/linear/r9700_linear.hip"
EAGER = ROOT / "src/ops/r9700/eager/eager_ops.hip"
SOURCE_INPUTS = [QUALIFIER, ROOT / "tools/r9700/a8q4_shape_sweep_qual.hip", CHECKER,
                 LINEAR.with_suffix(".h"), LINEAR, EAGER.with_suffix(".h"), EAGER,
                 ROOT / "profiles/rocprof/ordinary-decode-c1-one-round-trace-20260905/analysis-v2.json"]
SCRIPT_INPUTS = [PACKAGE / name for name in
                 ("commands.sh", "prepare.py", "preflight.py", "measure.py", "README.md")]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular retained file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def open_exclusive(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def write_json(path: Path, value: dict) -> None:
    with open_exclusive(path) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def require_environment() -> None:
    if Path.cwd() != ROOT or sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"run from {ROOT} using Python 3.11")


def compile_commands(build: Path) -> tuple[list[str], list[str], list[str]]:
    common = [str(HIPCC), "-O3", "-std=c++20", "--offload-arch=gfx1201",
              f"-I{ROOT / 'src'}", "-isystem", "/opt/rocm/include",
              f'-DNINFER_SOURCE_DIR="{ROOT}"']
    host = [*common, "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
            str(QUALIFIER), str(LINEAR), str(EAGER), "-L/opt/rocm/lib",
            "-Wl,-rpath,/opt/rocm/lib", "-o", str(build / "qual")]
    device = [*common, "-Wno-unused-command-line-argument", "--offload-device-only",
              "-S", str(QUALIFIER), "-o", str(build / "qual.s")]
    checker = [sys.executable, str(CHECKER), str(build / "qual.s")]
    return host, device, checker


def main() -> int:
    require_environment()
    if PLAN.exists() or PLAN.is_symlink() or ATTEMPT.exists() or ATTEMPT.is_symlink():
        raise RuntimeError("retry package plan and attempt must be fresh")
    plan = json.loads((ORIGINAL / "plan.json").read_text(encoding="utf-8"))
    plan["schema"] = "ninfer.r9700.a8q4-projected-residual-t1-retry-plan.v1"
    plan["status"] = "prepared_for_review"
    plan["claim"] = "Retry the unchanged direct candidate with an explicit stream and retained process receipts."
    plan["repair"] = {
        "scope": "qualifier stream ownership and retained harness evidence only",
        "unchanged": ["production Ops", "kernel mechanism", "oracle", "shapes", "sample counts", "thresholds"],
        "stream": "one explicit nonblocking HIP stream for both arms, transfers, scrub, warmup and events",
    }
    plan["bound_inputs"] = [identity(path) for path in SOURCE_INPUTS + SCRIPT_INPUTS]
    plan["original_evidence"] = [identity(path) for path in sorted(ORIGINAL.rglob("*")) if path.is_file()]
    if not any(Path(item["path"]).parent == ORIGINAL / "attempt-1" for item in plan["original_evidence"]):
        raise RuntimeError("original attempt evidence is missing")
    relative = PACKAGE.relative_to(ROOT)
    plan["exact_invocation"] = {
        name: f"bash {relative}/commands.sh --{name}" for name in ("preflight", "measure")}
    plan["output"] = {"path": str(ATTEMPT / "qualification.json"), "create_only": True}
    plan["attempt_closure"] = str(ATTEMPT / "closure.json")
    write_json(PLAN, plan)
    print(f"projected_residual_retry_prepare: PASS {PLAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
