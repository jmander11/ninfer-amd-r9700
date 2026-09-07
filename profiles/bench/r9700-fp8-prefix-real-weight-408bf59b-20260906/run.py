#!/usr/bin/env python3
"""Preflight and create-only exact FP8 prefix discriminator rerun."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import analyze

PACKAGE = Path(__file__).resolve().parent
RESULTS = PACKAGE / "results"
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
EXACT = {"LD_PRELOAD","LD_AUDIT","HIP_FORCE_QUEUE_PROFILING","AMD_SERIALIZE_KERNEL",
         "AMD_SERIALIZE_COPY","ROC_SERIALIZE_KERNEL","GPU_DUMP_CODE_OBJECT"}
PREFIX = ("ROCPROF","ROCP_","ROCTRACER_","ROCTX_","HSA_TOOLS_","HIP_TRACE_",
          "AQLPROFILE_","ATT_PROFILE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def write_exclusive(path: Path, data: bytes | str) -> None:
    payload = data.encode() if isinstance(data,str) else data
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o644)
    try:
        offset=0
        while offset < len(payload):
            count=os.write(fd,payload[offset:])
            if count <= 0: fail("write made no progress")
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)


def power() -> str:
    value=POWER.read_text().strip()
    if value != "auto": fail(f"power profile differs: {value}")
    return value


def preflight() -> None:
    injected=sorted(key for key in os.environ if key in EXACT or key.startswith(PREFIX))
    if injected: fail(f"profiling/injection environment is not clean: {injected}")
    analyze.validate_prepared()
    if (Path("/sys/class/drm/card2/device/vendor").read_text().strip() != "0x1002" or
            Path("/sys/class/drm/card2/device/device").read_text().strip() != "0x7551" or
            Path("/sys/class/drm/card2/device").resolve().name != "0000:13:00.0"):
        fail("R9700 PCI identity differs")
    power()
    ldd=subprocess.check_output(["ldd",str(analyze.EXE)],
        env={**os.environ,"LD_LIBRARY_PATH":"/opt/rocm/lib"},text=True)
    if ("not found" in ldd or
            "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7" not in ldd or
            "libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1" not in ldd):
        fail("runtime DSO resolution differs")
    subprocess.run(["/usr/bin/python3",str(ROOT_TEST),str(analyze.EXE)],check=True,
                   env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})


ROOT_TEST = PACKAGE / "retained-build/test_fp8_linear_prefix_discriminator.py"


def close_results() -> None:
    files=sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS/"result.sha256",
        "".join(f"{analyze.digest(path)}  {path}\n" for path in files))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    preflight()
    if args.preflight_only: return
    if RESULTS.exists() or RESULTS.is_symlink(): fail("results path already exists; refusing overwrite")
    RESULTS.mkdir()
    stdout_path=RESULTS/"stdout"
    stderr_path=RESULTS/"stderr"
    output=RESULTS/"result.json"
    argv=[str(analyze.EXE),"--artifact",str(analyze.ARTIFACT),"--output",str(output)]
    before=power()
    environment={**os.environ,"LD_LIBRARY_PATH":"/opt/rocm/lib"}
    completed=subprocess.run(argv,capture_output=True,env=environment)
    after=power()
    write_exclusive(stdout_path,completed.stdout)
    write_exclusive(stderr_path,completed.stderr)
    process={"artifact_type":"ninfer_fp8_prefix_process_receipt","schema_version":1,
        "argv":argv,"executable":analyze.identity(analyze.EXE),"exit_code":completed.returncode,
        "power_before":before,"power_after":after,"stdout":analyze.identity(stdout_path),
        "stderr":analyze.identity(stderr_path)}
    write_exclusive(RESULTS/"process.json",
        json.dumps(process,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if completed.returncode != 0: fail("discriminator failed; retained partial attempt")
    analyze.write_exclusive(RESULTS/"summary.json",analyze.analyze())
    close_results()


if __name__ == "__main__":
    main()
