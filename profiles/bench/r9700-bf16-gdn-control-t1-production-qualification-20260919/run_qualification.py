#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
ARTIFACTS = PACKAGE / "artifacts"
EXECUTABLE = ROOT / "build-r9700-bf16-gdn-control-t1-direct/src/ninfer_r9700_gdn_qual"


def fail(message: str) -> None:
    raise RuntimeError(message)


def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> int:
    device_command = [str(EXECUTABLE), "--device-info"]
    device_started = time.time_ns()
    device_process = subprocess.run(device_command, cwd=ROOT, capture_output=True, text=True)
    device_finished = time.time_ns()
    write_exclusive(ARTIFACTS / "device.stdout", device_process.stdout)
    write_exclusive(ARTIFACTS / "device.stderr", device_process.stderr)
    write_exclusive(ARTIFACTS / "device-process.json", json.dumps({
        "command": device_command, "cwd": str(ROOT), "started_unix_ns": device_started,
        "finished_unix_ns": device_finished, "exit_code": device_process.returncode,
        "executable": identity(EXECUTABLE),
        "stdout": identity(ARTIFACTS / "device.stdout"),
        "stderr": identity(ARTIFACTS / "device.stderr")
    }, indent=2) + "\n")
    if device_process.returncode != 0 or device_process.stderr:
        fail("device identity query failed")
    device = json.loads(device_process.stdout)
    if (device.get("ordinal") != 0 or device.get("name") != "AMD Radeon AI PRO R9700" or
            device.get("architecture") != "gfx1201" or device.get("wavefront") != 32 or
            not isinstance(device.get("runtime_version"), int) or device["runtime_version"] <= 0 or
            not isinstance(device.get("driver_version"), int) or device["driver_version"] <= 0):
        fail("selected HIP device identity differs")
    pci = device.get("pci_bus_id")
    if not isinstance(pci, str) or re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci) is None:
        fail("selected HIP PCI identity is malformed")
    profile = Path("/sys/bus/pci/devices") / pci / "power_dpm_force_performance_level"
    if not profile.is_file():
        fail("selected HIP device has no readable power profile")
    power_before = profile.read_text().strip()
    if power_before != "auto":
        fail("selected HIP device power profile is not auto before qualification")
    write_exclusive(ARTIFACTS / "device.json", json.dumps({
        **device, "power_profile_path": str(profile.resolve(strict=True))
    }, indent=2) + "\n")

    command = [str(EXECUTABLE)]
    started = time.time_ns()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    finished = time.time_ns()
    power_after = profile.read_text().strip()
    write_exclusive(ARTIFACTS / "qualification.stdout", completed.stdout)
    write_exclusive(ARTIFACTS / "qualification.stderr", completed.stderr)
    write_exclusive(ARTIFACTS / "qualification-process.json", json.dumps({
        "command": command, "cwd": str(ROOT), "started_unix_ns": started,
        "finished_unix_ns": finished, "exit_code": completed.returncode,
        "power_profile_path": str(profile.resolve(strict=True)),
        "power_before": power_before, "power_after": power_after,
        "device_identity": identity(ARTIFACTS / "device.json"),
        "executable": identity(EXECUTABLE),
        "stdout": identity(ARTIFACTS / "qualification.stdout"),
        "stderr": identity(ARTIFACTS / "qualification.stderr")
    }, indent=2) + "\n")
    if completed.returncode != 0 or completed.stderr or power_after != "auto":
        fail("production-symbol qualification process failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
