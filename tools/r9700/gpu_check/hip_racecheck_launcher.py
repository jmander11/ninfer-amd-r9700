#!/usr/bin/env python3
"""CMake HIP compiler launcher for the LDS race-check build.

Runs the HIP driver's own job list (``clang++ -###``) unchanged except for the gfx1201 device
compile: that job emits optimized bitcode instead, which is instrumented by
``ninfer_lds_race_instrument``, linked with the race-check device runtime, and compiled to the
object the driver expected. Bundling and the host compile run as the driver planned them.
Non-HIP compiles pass through.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(os.environ.get("NINFER_GPU_CHECK_TOOLS", Path(__file__).resolve().parent / "build"))


def driver_jobs(argv: list[str], scratch: str) -> list[list[str]]:
    env = dict(os.environ, TMPDIR=scratch)
    result = subprocess.run([*argv, "-###"], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return [shlex.split(line) for line in result.stderr.splitlines() if line.startswith(' "')]


def is_device_cc1(job: list[str]) -> bool:
    return (
        "-cc1" in job and "-triple" in job and job[job.index("-triple") + 1].startswith("amdgcn-")
    )


def option_value(job: list[str], option: str) -> str:
    return job[job.index(option) + 1]


def run(job: list[str]) -> None:
    completed = subprocess.run(job, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def device_job(job: list[str], scratch: str) -> None:
    target_object = option_value(job, "-o")
    stem = Path(scratch) / Path(target_object).name
    bitcode = f"{stem}.bc"
    emit = [bitcode if arg == target_object else arg for arg in job]
    emit = ["-emit-llvm-bc" if arg == "-emit-obj" else arg for arg in emit]
    emit += ["-debug-info-kind=line-tables-only", "-dwarf-version=5"]
    run(emit)
    instrumented = f"{stem}.rc.bc"
    run([str(TOOLS / "ninfer_lds_race_instrument"), bitcode, instrumented])
    linked = f"{stem}.linked.bc"
    run(
        [
            str(Path(job[0]).parent / "llvm-link"),
            instrumented,
            "--only-needed",
            str(TOOLS / "lds_race_rt.bc"),
            "-o",
            linked,
        ]
    )
    cpu = option_value(job, "-target-cpu")
    # The device job's backend options (e.g. -amdgpu-internalize-symbols) shape the code object.
    backend = []
    for i, arg in enumerate(job[:-1]):
        if arg == "-mllvm":
            backend += ["-mllvm", job[i + 1]]
    run(
        [
            str(Path(job[0]).parent / "clang"),
            "--target=amdgcn-amd-amdhsa",
            f"-mcpu={cpu}",
            "-O3",
            "-Wno-override-module",
            *backend,
            "-c",
            linked,
            "-o",
            target_object,
        ]
    )


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("usage: hip_racecheck_launcher.py <compiler> <args...>", file=sys.stderr)
        return 2
    is_hip = "-c" in argv and any(
        arg == "hip" and i > 0 and argv[i - 1] == "-x" for i, arg in enumerate(argv)
    )
    if not is_hip:
        os.execvp(argv[0], argv)
    scratch = tempfile.mkdtemp(prefix="ninfer-rc-")
    try:
        for job in driver_jobs(argv, scratch):
            if is_device_cc1(job):
                device_job(job, scratch)
            else:
                run(job)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
