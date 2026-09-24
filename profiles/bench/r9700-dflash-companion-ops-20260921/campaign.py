#!/usr/bin/env python3
"""Create-only public-Op qualification, not DFlash recipe or speed selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
BUILD = REPO / "build-r9700-dflash-companion-ops-20260921"
OUTPUT = PACKAGE / "physical"
sys.path.insert(0, str(REPO))
from tools.bench.run_ninfer_bench_matrix import (
    R9700_POWER_PROFILE, require_auto_power_profile, require_hip_pci_device,
)

TARGETS = {
    "linear": "ninfer_r9700_linear_op_qual",
    "selector": "ninfer_r9700_dflash2_select_qual",
    "conv": "ninfer_r9700_grouped_dynamic_conv_qual",
}
WIDTHS = (5, 6, 10, 12, 15, 18, 20, 24)
SHAPES = ((5120, 25600), (6144, 5120), (5120, 4096), (34816, 5120), (5120, 17408))
SOURCES = (
    "tools/r9700/linear_op_qual.hip", "tools/r9700/dflash2_select_qual.hip",
    "tools/r9700/grouped_dynamic_conv_qual.hip", "src/CMakeLists.txt", "CMakeLists.txt",
    "include/ninfer/ops/linear.h", "include/ninfer/ops/dflash2_path_select.h",
    "include/ninfer/ops/grouped_dynamic_conv.h", "src/ops/r9700/linear/linear_tensor_op.cpp",
    "src/ops/r9700/linear/r9700_linear.hip", "src/ops/r9700/linear/r9700_linear.h",
    "src/ops/r9700/linear/r9700_w8_activation_profile.h",
    "src/ops/r9700/dflash/dflash2_path_select.cpp", "src/ops/r9700/dflash/dflash2_path_select.hip",
    "src/ops/r9700/dflash/grouped_dynamic_conv.cpp", "src/ops/r9700/dflash/grouped_dynamic_conv.hip",
    "src/targets/qwen3/impl/runtime/dflash_impl.h",
)


def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def commands():
    yield "linear", [str(BUILD / "src" / TARGETS["linear"]), "--dflash-companion"]
    for k in (4, 5):
        for concurrency in (1, 2, 3, 4):
            yield f"conv-w{k+1}-c{concurrency}", [str(BUILD / "src" / TARGETS["conv"]),
                "--dflash-companion", str(k + 1), str(concurrency)]
            yield f"selector-k{k}-c{concurrency}", [str(BUILD / "src" / TARGETS["selector"]),
                "--dflash-companion", str(k), str(concurrency)]


def preflight():
    cache = {}
    for line in (BUILD / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    expected = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
        "NINFER_R9700_Q4_ACTIVATION_BITS": "8", "NINFER_R9700_W8_ACTIVATION_BITS": "8",
        "NINFER_R9700_KV_VALUE_GROUP": "16", "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF"}
    if any(cache.get(key) != value for key, value in expected.items()):
        raise ValueError("fresh qualifier build differs from declared gfx1201 arithmetic profile")
    binaries = [BUILD / "src" / target for target in TARGETS.values()]
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in binaries):
        raise ValueError("fresh qualification binaries are missing; build explicit targets first")
    # Dry-run only: reject stale objects without compiling or launching any qualifier.
    pending = subprocess.run(["cmake", "--build", str(BUILD), "--target", *TARGETS.values(),
                              "--", "-n"], check=True, capture_output=True, text=True)
    if pending.stdout.strip() != "ninja: no work to do.":
        raise ValueError("qualifier build is stale; complete its explicit compile-only targets")
    paths = [*binaries, BUILD / "CMakeCache.txt", *(REPO / name for name in SOURCES),
             PACKAGE / "campaign.py", PACKAGE / "commands.sh"]
    return {"artifact_type": "ninfer_r9700_dflash_companion_op_inputs", "schema_version": 1,
        "configuration": expected, "files": [identity(path) for path in paths],
        "commands": [{"name": name, "command": command} for name, command in commands()],
        "scope": "represented W8 public Ops only; not recipe quality, speed, or production admission"}


def idle():
    deadline = time.monotonic() + 10.
    while True:
        require_auto_power_profile()
        device = R9700_POWER_PROFILE.parent
        used = int((device / "mem_info_vram_used").read_text())
        busy = int((device / "gpu_busy_percent").read_text())
        if used < 1024**3 and busy == 0:
            return
        if time.monotonic() >= deadline:
            raise ValueError(f"R9700 is not idle: {used} resident bytes, {busy}% busy")
        time.sleep(.1)


def fields(line, prefix):
    if not line.startswith(prefix + ": PASS "):
        return None
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


def validate_output(name, text):
    if name == "linear":
        parsed = [row for line in text.splitlines()
                  if (row := fields(line, "dflash-companion-linear")) is not None]
        expected = {(n, k, t) for n, k in SHAPES for t in WIDTHS}
        expected |= {(5120, 25600, 2048), (6144, 5120, 2048)}
        observed = {(int(row["N"]), int(row["K"]), int(row["T"])) for row in parsed}
        if observed != expected or len(parsed) != len(expected):
            raise ValueError("linear qualification lacks exact declared shape/row coverage")
        for row in parsed:
            n, k, t = (int(row[key]) for key in ("N", "K", "T"))
            if (int(row["graph_replays"]) != 2 or int(row["full_K"]) != k
                    or int(row["fp64_samples"]) != 9 * (t if t <= 24 else 7)
                    or row["public_profile"] != ("A8W8" if (n, k) == (5120, 17408) else "BF16xW8")
                    or any(not math.isfinite(float(row[key])) for key in ("max_abs", "max_normalized"))
                    or not 0. <= float(row["max_normalized"]) <= 1.):
                raise ValueError("linear oracle/graph evidence is invalid")
        if "dflash-companion-linear: ALL PASS represented-BF16-input FP64 sampled oracle" not in text.splitlines():
            raise ValueError("linear qualification did not complete")
        return len(parsed)
    kind, width, concurrency = name.split("-")
    prefix = "dflash-companion-selector" if kind == "selector" else "dflash-companion-conv"
    parsed = [row for line in text.splitlines() if (row := fields(line, prefix)) is not None]
    required = 2 if kind == "selector" else 1
    if len(parsed) != required:
        raise ValueError("selector/conv qualification did not complete")
    for row in parsed:
        if (int(row["K" if kind == "selector" else "W"]) != int(width[1:])
                or int(row["C"]) != int(concurrency[1:]) or int(row["graph_replays"]) != 2):
            raise ValueError("selector/conv geometry or replay count differs")
        if kind == "selector":
            if int(row["logits_rows"]) != 131072 or int(row["bf16_codebook_rows"]) != 248320:
                raise ValueError("selector lost optimized head/full BF16 codebook geometry")
        elif (any(not math.isfinite(float(row[key])) for key in ("max_abs", "relative_l2"))
              or float(row["max_abs"]) > .04 or float(row["relative_l2"]) > .02):
            raise ValueError("convolution oracle criterion failed")
    if kind == "selector" and {row["mapped"] for row in parsed} != {"0", "1"}:
        raise ValueError("selector lacks identity and mapped-token checks")
    return len(parsed)


def run():
    if os.path.lexists(OUTPUT):
        raise ValueError("qualification output already exists; preserve it and use a separate reviewed retry")
    inputs = preflight()
    idle()
    require_hip_pci_device(0)
    OUTPUT.mkdir()
    with (OUTPUT / "inputs.json").open("x") as stream:
        json.dump(inputs, stream, indent=2)
    results = []
    for name, command in commands():
        idle()
        with (OUTPUT / f"{name}.stdout.txt").open("x") as stdout, \
             (OUTPUT / f"{name}.stderr.txt").open("x") as stderr:
            result = subprocess.run(command, cwd=REPO, stdout=stdout, stderr=stderr)
        if result.returncode:
            raise ValueError(f"{name} failed; retained raw outputs, no admission")
        count = validate_output(name, (OUTPUT / f"{name}.stdout.txt").read_text())
        require_auto_power_profile()
        results.append({"name": name, "qualified_cases": count})
    if preflight() != inputs:
        raise ValueError("qualification inputs changed during execution")
    with (OUTPUT / "result.json").open("x") as stream:
        json.dump({"artifact_type": "ninfer_r9700_dflash_companion_op_qualification", "schema_version": 1,
            "status": "passed_public_op_qualification", "production_selected": False,
            "inputs": identity(OUTPUT / "inputs.json"), "results": results}, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run"))
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise ValueError("use the explicit Python 3.11 interpreter in commands.sh")
    if args.stage == "preflight":
        preflight()
        print("DFlash companion Op preflight PASS; no GPU initialization or output writes")
    else:
        run()


if __name__ == "__main__":
    main()
