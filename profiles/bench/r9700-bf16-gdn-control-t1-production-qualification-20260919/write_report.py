#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
ARTIFACTS = PACKAGE / "artifacts"
REPORT = PACKAGE / "report.json"
BUILD = ROOT / "build-r9700-bf16-gdn-control-t1-direct"


def fail(message: str) -> None:
    raise RuntimeError(message)


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def cache(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))
    if not isinstance(value, dict):
        fail(f"not an object: {path}")
    return value


def require_identity(record: dict, path: Path, label: str) -> None:
    if record != identity(path):
        fail(f"{label} identity differs")


def main() -> int:
    if REPORT.exists() or REPORT.is_symlink():
        fail("report already exists; qualification never overwrites")
    stdout = (ARTIFACTS / "qualification.stdout").read_text()
    stderr = (ARTIFACTS / "qualification.stderr").read_text()
    receipt = (ARTIFACTS / "static-receipt.txt").read_text().strip()
    device = load(ARTIFACTS / "device.json")
    queried_device = load(ARTIFACTS / "device.stdout")
    if {key: value for key, value in device.items() if key != "power_profile_path"} != queried_device:
        fail("retained device identity differs from the HIP query")
    pci = device.get("pci_bus_id")
    profile = (Path("/sys/bus/pci/devices") / str(pci) /
               "power_dpm_force_performance_level").resolve(strict=True)
    if device.get("power_profile_path") != str(profile):
        fail("device identity is not linked to its exact sysfs power profile")
    if (device.get("ordinal") != 0 or device.get("name") != "AMD Radeon AI PRO R9700" or
            device.get("architecture") != "gfx1201" or device.get("wavefront") != 32 or
            not isinstance(device.get("runtime_version"), int) or device["runtime_version"] <= 0 or
            not isinstance(device.get("driver_version"), int) or device["driver_version"] <= 0):
        fail("selected device provenance differs")
    device_line = (f"device: name={device['name']} arch={device['architecture']} "
                   f"wave={device['wavefront']} pci={pci} runtime={device['runtime_version']} "
                   f"driver={device['driver_version']}")
    if device_line not in stdout:
        fail("qualification did not bind the selected R9700")
    if "r9700_gdn_op_qualification: PASS" not in stdout or stderr:
        fail("production-symbol qualification did not pass cleanly")
    if not receipt.startswith("PASS kernel=bf16_projected_control_t1_kernel"):
        fail("static receipt did not pass")
    executable = BUILD / "src/ninfer_r9700_gdn_qual"
    device_process = load(ARTIFACTS / "device-process.json")
    expected_device_command = [str(executable), "--device-info"]
    if (device_process.get("command") != expected_device_command or
            device_process.get("cwd") != str(ROOT) or device_process.get("exit_code") != 0 or
            not isinstance(device_process.get("started_unix_ns"), int) or
            not isinstance(device_process.get("finished_unix_ns"), int) or
            device_process["started_unix_ns"] >= device_process["finished_unix_ns"]):
        fail("device query process receipt differs")
    require_identity(device_process.get("executable"), executable,
                     "device query executable")
    require_identity(device_process.get("stdout"), ARTIFACTS / "device.stdout",
                     "device query stdout")
    require_identity(device_process.get("stderr"), ARTIFACTS / "device.stderr",
                     "device query stderr")
    process = load(ARTIFACTS / "qualification-process.json")
    if (process.get("command") != [str(executable)] or process.get("cwd") != str(ROOT) or
            process.get("exit_code") != 0 or process.get("power_profile_path") != str(profile) or
            process.get("power_before") != "auto" or process.get("power_after") != "auto" or
            not isinstance(process.get("started_unix_ns"), int) or
            not isinstance(process.get("finished_unix_ns"), int) or
            process["started_unix_ns"] >= process["finished_unix_ns"] or
            process["started_unix_ns"] < device_process["finished_unix_ns"]):
        fail("qualification process receipt differs")
    require_identity(process.get("device_identity"), ARTIFACTS / "device.json",
                     "qualification device")
    require_identity(process.get("executable"), executable,
                     "qualification executable")
    require_identity(process.get("stdout"), ARTIFACTS / "qualification.stdout",
                     "qualification stdout")
    require_identity(process.get("stderr"), ARTIFACTS / "qualification.stderr",
                     "qualification stderr")
    values = cache(BUILD / "CMakeCache.txt")
    expected = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
                "NINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE": "1"}
    for key, value in expected.items():
        if values.get(key) != value:
            fail(f"build cache differs at {key}")
    report = {
        "schema": "ninfer.r9700.bf16-gdn-control-t1-production-qualification.v1",
        "status": "qualified_for_whole_ab_only",
        "production_routing_authorized": False,
        "device": device,
        "shape": {"tokens": 1, "columns": 5120, "heads": 48},
        "numeric": {"serial_fp32_outputs_bit_exact": True, "independent_fp64_oracle_passed": True,
                    "explicit_bf16_boundary": True, "malformed_input_checks_passed": True},
        "static": {"passed": True, "receipt": receipt},
        "build_cache": {key: values[key] for key in expected},
        "identities": {
            "executable": identity(BUILD / "src/ninfer_r9700_gdn_qual"),
            "assembly": identity(ARTIFACTS / "gdn_ops.s"),
            "static_receipt": identity(ARTIFACTS / "static-receipt.txt"),
            "stdout": identity(ARTIFACTS / "qualification.stdout"),
            "stderr": identity(ARTIFACTS / "qualification.stderr"),
            "device_stdout": identity(ARTIFACTS / "device.stdout"),
            "device_stderr": identity(ARTIFACTS / "device.stderr"),
            "device_process": identity(ARTIFACTS / "device-process.json"),
            "device_identity": identity(ARTIFACTS / "device.json"),
            "qualification_process": identity(ARTIFACTS / "qualification-process.json"),
            "kernel_source": identity(ROOT / "src/ops/r9700/gdn/gdn_ops.hip"),
            "semantic_wrapper": identity(ROOT / "src/ops/r9700/gdn/gdn_gating.cpp"),
            "qualifier": identity(ROOT / "tools/r9700/gdn_op_qual.hip"),
            "checker": identity(ROOT / "tools/r9700/check_bf16_gdn_projected_control_t1_static.py"),
            "runner": identity(PACKAGE / "run_qualification.py"),
            "commands": identity(PACKAGE / "commands.sh"),
            "writer": identity(PACKAGE / "write_report.py"),
            "plan": identity(PACKAGE / "plan.json")
        },
        "next_gate": "matched whole ordinary-decode A/B; no production promotion is authorized by this report"
    }
    descriptor = os.open(REPORT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("r9700_bf16_gdn_control_t1_production_qualification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
