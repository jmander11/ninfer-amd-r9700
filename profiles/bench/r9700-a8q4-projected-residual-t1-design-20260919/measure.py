#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919"
ATTEMPT = PACKAGE / "attempt-1"
QUALIFIER = ROOT / "tools/r9700/a8q4_projected_residual_t1_qual.hip"
CHECKER = ROOT / "tools/r9700/check_a8q4_projected_residual_t1_static.py"
LINEAR = ROOT / "src/ops/r9700/linear/r9700_linear.hip"
EAGER = ROOT / "src/ops/r9700/eager/eager_ops.hip"
HIPCC = Path("/opt/rocm/bin/hipcc")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def open_exclusive(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    return os.fdopen(descriptor, "w")


def retained_step(name: str, command: list[str], stdout_path: Path | None = None) -> dict:
    stdout_path = stdout_path or ATTEMPT / f"{name}.stdout"
    stderr_path = ATTEMPT / f"{name}.stderr"
    started = time.time_ns()
    with open_exclusive(stdout_path) as stdout, open_exclusive(stderr_path) as stderr:
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    finished = time.time_ns()
    record = {"name": name, "argv": command, "exit_code": completed.returncode,
              "started_unix_ns": started, "finished_unix_ns": finished,
              "stdout": identity(stdout_path), "stderr": identity(stderr_path)}
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed; immutable attempt retained at {ATTEMPT}")
    return record


def main() -> int:
    if Path.cwd() != ROOT:
        raise RuntimeError(f"measure must run from {ROOT}")
    ATTEMPT.mkdir(mode=0o755)
    binary = ATTEMPT / "qual"
    assembly = ATTEMPT / "qual.s"
    static_receipt = ATTEMPT / "static-receipt.txt"
    compile_receipt = ATTEMPT / "compile-receipt.json"
    report = ATTEMPT / "qualification.json"
    common = ["-O3", "-std=c++20", "--offload-arch=gfx1201", f"-I{ROOT / 'src'}",
              "-isystem", "/opt/rocm/include"]
    define = f'-DNINFER_SOURCE_DIR="{ROOT}"'
    host = [str(HIPCC), *common, "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
            define, str(QUALIFIER), str(LINEAR), str(EAGER), "-L/opt/rocm/lib",
            "-Wl,-rpath,/opt/rocm/lib", "-o", str(binary)]
    device = [str(HIPCC), *common, "-Wno-unused-command-line-argument", define,
              "--offload-device-only", "-S", str(QUALIFIER), "-o", str(assembly)]
    checker = ["python3", str(CHECKER), str(assembly)]
    steps = [retained_step("host-compile", host), retained_step("device-compile", device),
             retained_step("static-check", checker, static_receipt)]
    qualifier = [str(binary), "--out-json", str(report), "--assembly", str(assembly),
                 "--static-receipt", str(static_receipt),
                 "--compile-receipt", str(compile_receipt)]
    receipt = {"schema": "ninfer.r9700.immutable-compile-process-receipt.v1",
               "attempt": str(ATTEMPT), "create_only": True, "steps": steps,
               "outputs": {"executable": identity(binary), "assembly": identity(assembly),
                           "static_receipt": identity(static_receipt)},
               "measurement_process": {"argv": qualifier}}
    with open_exclusive(compile_receipt) as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    completed = subprocess.run(qualifier, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"qualification failed; immutable attempt retained at {ATTEMPT}")
    print("r9700_a8q4_projected_residual_t1_measure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
