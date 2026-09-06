#!/usr/bin/env python3
"""Run the create-only product-source scalar-base operator gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/ssdpool2nvme/local_llm/.venv-ninfer-r9700/bin/python")
REPORT = ROOT / "profiles/bench/r9700-a8q4-n16k16-scalar-base-product-p2048-ab-20260905.json"
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
BUILD = ROOT / "build-r9700-scalar-base-qualification"
PRODUCT_QUALIFIER = BUILD / "tests/ninfer_r9700_a8q4_scalar_base_product_qual"
BOUNDS_TEST = BUILD / "tests/ninfer_r9700_a8q4_scalar_base_bounds_test"
SHAPE_QUALIFIER = ROOT / "tools/r9700/build/a8q4_scalar_base_shape_qual"
SOURCE = ROOT / "src/ops/r9700/linear/r9700_linear.hip"
HEADER = ROOT / "src/ops/r9700/linear/r9700_linear.h"
PROFILE = ROOT / "src/ops/r9700/linear/r9700_q4_activation_profile.h"
HARNESS = ROOT / "tools/r9700/a8q4_n16k16_scalar_base_harness.hip"
SHAPE_SOURCE = ROOT / "tools/r9700/a8q4_shape_sweep_qual.hip"
BOUNDS_SOURCE = ROOT / "tools/r9700/test_a8q4_scalar_base_bounds.cpp"
ASSEMBLY = ROOT / "tools/r9700/build/r9700_linear_scalar_base_product.s"
CHECKER = ROOT / "tools/r9700/check_a8q4_n16k16_scalar_base_static.py"
PRODUCT_OBJECT = BUILD / "src/CMakeFiles/ninfer_r9700_core.dir/ops/r9700/linear/r9700_linear.hip.o"
CMAKE_CACHE = BUILD / "CMakeCache.txt"
COMPILE_COMMANDS = BUILD / "compile_commands.json"

SCHEMA = "ninfer.r9700.a8q4-n16k16-scalar-base-product-gate.v1"
MEASUREMENT_SCHEMA = "ninfer.r9700.a8q4-n16k16-scalar-base-measurement.v3"
PROFILE_NAME = "m64n128-pingpong-n16-k16-scalar-base-qualification"
ROBUST_FACTOR = 4.4478
SHAPES = (
    ("value_z", 12288, 5120, 48),
    ("gdn_output", 5120, 6144, 64),
    ("mlp_down", 5120, 17408, 64),
)
INPUTS = (
    PRODUCT_QUALIFIER, BOUNDS_TEST, SHAPE_QUALIFIER, SOURCE, HEADER, PROFILE,
    HARNESS, SHAPE_SOURCE, BOUNDS_SOURCE, ASSEMBLY, CHECKER, PRODUCT_OBJECT,
    CMAKE_CACHE, COMPILE_COMMANDS, Path(__file__).resolve(),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_inputs() -> dict[str, str]:
    result = {}
    for path in INPUTS:
        absolute = Path(os.path.abspath(path))
        metadata = os.lstat(absolute)
        if not stat.S_ISREG(metadata.st_mode) or absolute.is_symlink():
            raise RuntimeError(f"input must be a real regular file: {absolute}")
        if metadata.st_uid != os.getuid():
            raise RuntimeError(f"input owner differs from the repository owner: {absolute}")
        if path in (PRODUCT_QUALIFIER, BOUNDS_TEST, SHAPE_QUALIFIER) and not os.access(path, os.X_OK):
            raise RuntimeError(f"qualification executable is not executable: {absolute}")
        result[str(absolute.relative_to(ROOT))] = sha256(absolute)
    return result


def validate_build_contract() -> None:
    cache = CMAKE_CACHE.read_text(encoding="utf-8")
    expected = {
        "NINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION:BOOL": "ON",
        "NINFER_R9700_Q4_ACTIVATION_BITS:STRING": "8",
        "NINFER_R9700_KV_VALUE_GROUP:STRING": "16",
        "NINFER_R9700_W8_ACTIVATION_BITS:STRING": "8",
        "CMAKE_HIP_COMPILER:STRING": "/opt/rocm/llvm/bin/clang++",
    }
    observed = {}
    for key, wanted in expected.items():
        match = re.search(rf"(?m)^{re.escape(key)}=(.+)$", cache)
        if match is None or match.group(1) != wanted:
            raise RuntimeError(f"candidate build does not bind {key}={wanted}")
        observed[key] = match.group(1)
    commands = json.loads(COMPILE_COMMANDS.read_text(encoding="utf-8"))
    selected = [entry for entry in commands if Path(entry.get("file", "")).resolve() == SOURCE]
    if len(selected) != 1:
        raise RuntimeError("candidate compile database lacks one product linear source entry")
    command = selected[0].get("command", "")
    if (
        command.count("-DNINFER_R9700_Q4_SCALAR_BASE_QUALIFICATION=1") != 1
        or "--offload-arch=gfx1201" not in command
        or "-DNINFER_R9700_Q4_ACTIVATION_BITS=8" not in command
        or Path(selected[0].get("output", "")).name != "r9700_linear.hip.o"
    ):
        raise RuntimeError("candidate compile command is not the exact gfx1201 scalar-base product build")
    if PROFILE_NAME not in PROFILE.read_text(encoding="utf-8"):
        raise RuntimeError("candidate report profile string is absent from the bound product profile")


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=ROOT, text=True, capture_output=True, check=False)


def power() -> dict[str, object]:
    command = [
        "/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname", "--showprofile",
        "--showperflevel", "--json",
    ]
    result = run(command)
    if result.returncode != 0 or result.stderr:
        raise RuntimeError("rocm-smi identity query failed")
    payload = json.loads(result.stdout)
    level = POWER.read_text(encoding="utf-8").strip()
    card = payload.get("card0") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict) or set(payload) != {"card0"} or not isinstance(card, dict)
        or card.get("Card Series") != "AMD Radeon AI PRO R9700"
        or card.get("GFX Version") != "gfx1201"
        or card.get("Performance Level") != "auto" or level != "auto"
    ):
        raise RuntimeError("device0 must be the exact R9700/gfx1201 auto-power target")
    return {
        "command": command,
        "device": {"ordinal": 0, "name": card["Card Series"], "architecture": card["GFX Version"]},
        "rocm_smi_performance_level": "auto",
        "sysfs_path": str(POWER),
        "sysfs_value": "auto",
    }


def validate_power(value: object) -> None:
    expected_command = [
        "/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname", "--showprofile",
        "--showperflevel", "--json",
    ]
    expected = {
        "command": expected_command,
        "device": {"ordinal": 0, "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201"},
        "rocm_smi_performance_level": "auto", "sysfs_path": str(POWER), "sysfs_value": "auto",
    }
    if value != expected:
        raise RuntimeError("power evidence differs from the exact auto R9700 contract")


def mad(values: Sequence[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def decision(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema", "device", "actual_product_wrappers", "warmup_order", "pair_order", "cells"
    }:
        raise RuntimeError("measurement field inventory")
    if raw.get("schema") != MEASUREMENT_SCHEMA or raw.get("actual_product_wrappers") is not True:
        raise RuntimeError("measurement is not the product-wrapper v3 authority")
    if raw.get("device") != {"ordinal": 0, "name": "AMD Radeon AI PRO R9700", "architecture": "gfx1201"}:
        raise RuntimeError("measurement device")
    if raw.get("warmup_order") != ["A", "B", "B", "A"] or raw.get("pair_order") != ["AB", "BA"] * 4:
        raise RuntimeError("measurement balanced order")
    cells = raw.get("cells")
    if not isinstance(cells, list) or len(cells) != 3:
        raise RuntimeError("measurement cell count")
    summaries = []
    stable = True
    robust_uppers = []
    weighted_point = 0.0
    weighted_uncertainty = 0.0
    for cell, (name, rows, columns, calls) in zip(cells, SHAPES, strict=True):
        if not isinstance(cell, dict) or set(cell) != {
            "name", "tokens", "rows", "columns", "calls", "iterations_per_sample",
            "control_interval_samples_ms", "candidate_interval_samples_ms",
        }:
            raise RuntimeError("measurement cell field inventory")
        if {key: cell.get(key) for key in ("name", "tokens", "rows", "columns", "calls", "iterations_per_sample")} != {
            "name": name, "tokens": 2048, "rows": rows, "columns": columns,
            "calls": calls, "iterations_per_sample": 128,
        }:
            raise RuntimeError("measurement cell identity")
        control = cell.get("control_interval_samples_ms")
        candidate = cell.get("candidate_interval_samples_ms")
        if (
            not isinstance(control, list) or not isinstance(candidate, list)
            or len(control) != 8 or len(candidate) != 8
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 100.0 for value in control + candidate)
        ):
            raise RuntimeError("measurement samples")
        control = [float(value) / 128.0 for value in control]
        candidate = [float(value) / 128.0 for value in candidate]
        control_median = statistics.median(control)
        candidate_median = statistics.median(candidate)
        control_mad = mad(control)
        candidate_mad = mad(candidate)
        cell_stable = (
            control_mad / control_median <= 0.02
            and candidate_mad / candidate_median <= 0.02
            and (max(control) - min(control)) / control_median <= 0.06
            and (max(candidate) - min(candidate)) / candidate_median <= 0.06
        )
        savings = [a - b for a, b in zip(control, candidate, strict=True)]
        ratios = [b / a for a, b in zip(control, candidate, strict=True)]
        saving_median = statistics.median(savings)
        saving_mad = mad(savings)
        ratio_median = statistics.median(ratios)
        ratio_mad = mad(ratios)
        robust_upper = ratio_median + ROBUST_FACTOR * ratio_mad
        stable &= cell_stable
        robust_uppers.append(robust_upper)
        weighted_point += calls * saving_median
        weighted_uncertainty += ROBUST_FACTOR * calls * saving_mad
        summaries.append({
            "name": name, "control_median_ms": control_median,
            "candidate_median_ms": candidate_median, "control_mad_ms": control_mad,
            "candidate_mad_ms": candidate_mad, "paired_saving_median_ms": saving_median,
            "paired_saving_mad_ms": saving_mad, "paired_ratio_median": ratio_median,
            "paired_ratio_mad": ratio_mad, "robust_ratio_upper": robust_upper,
            "stability_pass": cell_stable,
        })
    robust_lower = weighted_point - weighted_uncertainty
    admitted = stable and max(robust_uppers) <= 1.01 and robust_lower >= 5.0
    return {
        "classification": "admitted" if admitted else ("inconclusive" if not stable else "rejected"),
        "admission_pass": admitted, "stability_pass": stable,
        "robust_mad_factor": ROBUST_FACTOR, "cells": summaries,
        "weighted_paired_saving_point_ms": weighted_point,
        "weighted_saving_uncertainty_ms": weighted_uncertainty,
        "robust_weighted_saving_lower_ms": robust_lower,
        "required_robust_weighted_saving_lower_ms": 5.0,
        "maximum_every_cell_robust_ratio_upper": 1.01,
    }


def parse_static(stdout: str) -> dict[str, object]:
    if not stdout.startswith("PASS ") or not stdout.endswith("\n"):
        raise RuntimeError("static checker did not emit one PASS record")
    fields = {}
    for piece in stdout[5:].split():
        if piece.count("=") != 1:
            raise RuntimeError("static checker field is malformed")
        key, value = piece.split("=", 1)
        fields[key] = int(value) if value.isdigit() and not key.endswith("_sha256") else value
    expected = {
        "logical_vgpr": 88, "architectural_vgpr": 88, "lds_bytes": 17152,
        "compiler_occupancy_waves": 16, "iu4": 8, "qualified_tuples": 32,
        "activation_code_max_voffset": 71_303_164,
        "activation_code_max_final_byte": 71_303_167,
        "weight_code_max_voffset": 89_128_952,
        "weight_code_max_final_byte": 89_128_959,
        "activation_scale_max_voffset": 4_456_446,
        "activation_scale_max_final_byte": 4_456_447,
        "weight_scale_max_voffset": 5_570_558,
        "weight_scale_max_final_byte": 5_570_559,
        "source_sha256": sha256(SOURCE), "profile_sha256": sha256(PROFILE),
        "assembly_sha256": sha256(ASSEMBLY),
        "product_executable_sha256": sha256(PRODUCT_QUALIFIER),
        "scratch": 0, "spills": 0, "scalar_bases": 5,
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(f"static checker field {key} differs")
    for key in ("loaded_code_object_sha256", "loaded_disassembly_sha256", "loaded_notes_sha256"):
        value = fields.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"static checker lacks {key}")
    if set(fields) != set(expected) | {
        "loaded_code_object_sha256", "loaded_disassembly_sha256", "loaded_notes_sha256"
    }:
        raise RuntimeError("static checker field inventory differs")
    return fields


def checked(command: Sequence[str], stdout_pattern: str) -> dict[str, object]:
    result = run(command)
    if result.returncode != 0 or result.stderr or re.fullmatch(stdout_pattern, result.stdout) is None:
        raise RuntimeError(f"qualification command failed: {' '.join(command)}\n{result.stdout}{result.stderr}")
    return {"command": list(command), "returncode": 0, "stdout": result.stdout, "stderr": ""}


def validate_report(value: object, expected_hashes: dict[str, str] | None = None) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema", "command", "input_sha256", "build_contract", "static", "loaded_object",
        "bounds_regression", "shape_sweep_regression", "product_wrapper_regression",
        "measurement_command", "power_before", "power_after", "measurement", "decision",
    } or value.get("schema") != SCHEMA:
        raise RuntimeError("product gate report schema")
    command = [str(PYTHON), "-m", "tools.r9700.run_a8q4_n16k16_scalar_base_gate",
               "--benchmark", "--output", str(REPORT)]
    if value.get("command") != command or value.get("build_contract") != {
        "build_directory": str(BUILD.relative_to(ROOT)), "qualification_flag": "ON",
        "target_architecture": "gfx1201", "q4_activation_bits": 8,
        "q4_prefill_cta_profile": PROFILE_NAME,
    }:
        raise RuntimeError("product gate command/build contract")
    hashes = snapshot_inputs() if expected_hashes is None else expected_hashes
    if value.get("input_sha256") != hashes:
        raise RuntimeError("product gate input hashes")
    static = value.get("static")
    static_command = [
        str(PYTHON), "-m", "tools.r9700.check_a8q4_n16k16_scalar_base_static",
        "--source", str(SOURCE), "--profile", str(PROFILE), "--assembly", str(ASSEMBLY),
        "--product-executable", str(PRODUCT_QUALIFIER),
    ]
    if not isinstance(static, dict) or static.get("command") != static_command or static.get(
        "returncode"
    ) != 0 or static.get("stderr") != "" or static.get("fields") != parse_static(static.get("stdout", "")):
        raise RuntimeError("product gate static evidence")
    if value.get("loaded_object") != {
        key: static["fields"][key] for key in (
            "loaded_code_object_sha256", "loaded_disassembly_sha256", "loaded_notes_sha256",
            "product_executable_sha256",
        )
    }:
        raise RuntimeError("product gate loaded-object evidence")
    expected_commands = (
        ("bounds_regression", [str(BOUNDS_TEST)], r"PASS tuples=32 activation_code_max=71303167 weight_code_max=89128959 activation_scale_max=4456447 weight_scale_max=5570559 uint32_boundary=true\n"),
        ("shape_sweep_regression", [str(SHAPE_QUALIFIER), "--prefill-cta-scalar-base-regression"], r"A8Q4 scalar-base product-wrapper regression: PASS \(exact tail case and six Text plus two MTP shapes at four token extents\)\n"),
        ("product_wrapper_regression", [str(PRODUCT_QUALIFIER), "--regression"], r"PASS actual_product_wrappers=true exact_shapes=3 full_bit_parity=true complete_k_fp64_samples=60 graph_replay=true status_poison_reset=true guards=true\n"),
    )
    for key, expected_command, pattern in expected_commands:
        record = value.get(key)
        if not isinstance(record, dict) or record.get("command") != expected_command or record.get(
            "returncode"
        ) != 0 or record.get("stderr") != "" or re.fullmatch(pattern, record.get("stdout", "")) is None:
            raise RuntimeError(f"product gate {key}")
    if value.get("measurement_command") != {
        "command": [str(PRODUCT_QUALIFIER), "--benchmark"],
        "returncode": 0,
        "stderr": "",
    }:
        raise RuntimeError("product gate measurement command")
    validate_power(value.get("power_before")); validate_power(value.get("power_after"))
    if value.get("decision") != decision(value.get("measurement")):
        raise RuntimeError("product gate decision")


def _inode(path: Path) -> tuple[int, int, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("owned publication is not regular")
    return metadata.st_dev, metadata.st_ino, metadata.st_uid


def _unlink_if_owned(path: Path, owner: tuple[int, int, int]) -> bool:
    try:
        if _inode(path) != owner:
            return False
    except FileNotFoundError:
        return False
    path.unlink(); return True


def publish(path: Path, value: dict, expected_hashes: dict[str, str]) -> None:
    parent = os.lstat(path.parent)
    if (
        os.path.lexists(path) or not stat.S_ISDIR(parent.st_mode) or path.parent.is_symlink()
        or parent.st_uid != os.getuid()
    ):
        raise RuntimeError("product gate output must be fresh under a real directory")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    pending = Path(name); owner = None; linked = False
    try:
        metadata = os.fstat(descriptor); owner = (metadata.st_dev, metadata.st_ino, metadata.st_uid)
        if metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
            raise RuntimeError("pending publication owner")
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0: raise RuntimeError("short publication write")
            view = view[count:]
        os.fsync(descriptor); os.fchmod(descriptor, 0o444)
        if _inode(pending) != owner: raise RuntimeError("pending publication replaced")
        validate_report(json.loads(data), expected_hashes)
        os.link(pending, path); linked = True
        if _inode(path) != owner or path.read_bytes() != data: raise RuntimeError("publication readback")
        validate_report(json.loads(path.read_text()), expected_hashes)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        if linked and owner is not None: _unlink_if_owned(path, owner)
        raise
    finally:
        os.close(descriptor)
        if owner is not None: _unlink_if_owned(pending, owner)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path(os.path.abspath(args.output))
    if Path.cwd().resolve() != ROOT or output != REPORT or os.path.lexists(output):
        raise RuntimeError("invoke from the repository root with the exact fresh product report path")
    before = snapshot_inputs(); validate_build_contract()
    static_command = [
        str(PYTHON), "-m", "tools.r9700.check_a8q4_n16k16_scalar_base_static",
        "--source", str(SOURCE), "--profile", str(PROFILE), "--assembly", str(ASSEMBLY),
        "--product-executable", str(PRODUCT_QUALIFIER),
    ]
    static_result = run(static_command)
    if static_result.returncode != 0 or static_result.stderr:
        raise RuntimeError(static_result.stdout + static_result.stderr)
    static_fields = parse_static(static_result.stdout)
    bounds = checked([str(BOUNDS_TEST)], r"PASS tuples=32 activation_code_max=71303167 weight_code_max=89128959 activation_scale_max=4456447 weight_scale_max=5570559 uint32_boundary=true\n")
    power_before = power()
    shape = checked(
        [str(SHAPE_QUALIFIER), "--prefill-cta-scalar-base-regression"],
        r"A8Q4 scalar-base product-wrapper regression: PASS \(exact tail case and six Text plus two MTP shapes at four token extents\)\n",
    )
    regression = checked(
        [str(PRODUCT_QUALIFIER), "--regression"],
        r"PASS actual_product_wrappers=true exact_shapes=3 full_bit_parity=true complete_k_fp64_samples=60 graph_replay=true status_poison_reset=true guards=true\n",
    )
    measured = run([str(PRODUCT_QUALIFIER), "--benchmark"])
    power_after = power()
    if measured.returncode != 0 or measured.stderr:
        raise RuntimeError(measured.stdout + measured.stderr)
    measurement = json.loads(measured.stdout); result = decision(measurement)
    if snapshot_inputs() != before:
        raise RuntimeError("product gate input changed during qualification")
    static_record = {
        "command": static_command, "returncode": 0, "stdout": static_result.stdout,
        "stderr": "", "fields": static_fields,
    }
    report = {
        "schema": SCHEMA,
        "command": [str(PYTHON), "-m", "tools.r9700.run_a8q4_n16k16_scalar_base_gate",
                    "--benchmark", "--output", str(output)],
        "input_sha256": before,
        "build_contract": {"build_directory": str(BUILD.relative_to(ROOT)),
                           "qualification_flag": "ON", "target_architecture": "gfx1201",
                           "q4_activation_bits": 8, "q4_prefill_cta_profile": PROFILE_NAME},
        "static": static_record,
        "loaded_object": {key: static_fields[key] for key in (
            "loaded_code_object_sha256", "loaded_disassembly_sha256", "loaded_notes_sha256",
            "product_executable_sha256",
        )},
        "bounds_regression": bounds, "shape_sweep_regression": shape,
        "product_wrapper_regression": regression,
        "measurement_command": {"command": [str(PRODUCT_QUALIFIER), "--benchmark"],
                                "returncode": 0, "stderr": ""},
        "power_before": power_before,
        "power_after": power_after, "measurement": measurement, "decision": result,
    }
    publish(output, report, before)
    return 0 if result["admission_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
