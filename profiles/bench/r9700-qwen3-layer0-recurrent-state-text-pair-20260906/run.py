#!/usr/bin/env python3
"""Create-only importer and preflight for the retained manual capture."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import analyze

PACKAGE = Path(__file__).resolve().parent
RESULTS = PACKAGE / "results"
SOURCE = Path("/tmp/ninfer-gdn-state-layer0-fda2d17a.MOBqmy")
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
NAMES = ("fresh.report.json", "append.report.json", "fresh.layer.bin", "append.layer.bin",
         "fresh.layer.json", "append.layer.json", "fresh.state.bin", "append.state.bin",
         "fresh.state.json", "append.state.json", "state-comparison.json")
EXACT = {"LD_PRELOAD", "LD_AUDIT", "HIP_FORCE_QUEUE_PROFILING", "AMD_SERIALIZE_KERNEL",
         "AMD_SERIALIZE_COPY", "ROC_SERIALIZE_KERNEL", "GPU_DUMP_CODE_OBJECT"}
PREFIXES = ("ROCPROF", "ROCP_", "ROCTRACER_", "ROCTX_", "HSA_TOOLS_", "HIP_TRACE_",
            "AQLPROFILE_", "ATT_PROFILE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        offset = 0
        while offset < len(data):
            count = os.write(descriptor, data[offset:])
            if count <= 0:
                fail("write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def preflight() -> None:
    injected = sorted(key for key in os.environ if key in EXACT or key.startswith(PREFIXES))
    if injected:
        fail(f"profiling/injection environment is not clean: {injected}")
    analyze.validate_plan()
    ldd = subprocess.check_output(["ldd", str(PACKAGE / "retained-bin/ninfer_bench")],
                                  env={**os.environ, "LD_LIBRARY_PATH":"/opt/rocm/lib"}, text=True)
    if ("not found" in ldd or
            "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7" not in ldd or
            "libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1" not in ldd):
        fail("runtime DSO resolution differs")
    if (Path("/sys/class/drm/card2/device/vendor").read_text().strip() != "0x1002" or
            Path("/sys/class/drm/card2/device/device").read_text().strip() != "0x7551" or
            Path("/sys/class/drm/card2/device").resolve().name != "0000:13:00.0"):
        fail("R9700 PCI identity differs")
    if POWER.read_text().strip() != "auto":
        fail("power profile must be auto for any reproduction/import operation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    preflight()
    if args.preflight_only:
        return
    if RESULTS.exists() and any(RESULTS.iterdir()):
        fail("results already exist; refusing overwrite")
    RESULTS.mkdir(parents=False, exist_ok=True)
    for name in NAMES:
        source = SOURCE / name
        if not source.is_file() or source.is_symlink():
            fail(f"capture source missing or unsafe: {name}")
        write_exclusive(RESULTS / name, source.read_bytes())
    analyze.write_exclusive(RESULTS / "summary.json", analyze.analyze())


if __name__ == "__main__":
    main()
