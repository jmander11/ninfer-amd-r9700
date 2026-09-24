#!/usr/bin/env python3
"""Run the disconnected N16/K16 gate and publish one immutable atomic JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "tools/r9700/build/a8q4_n16k16_layout_harness"
DEFAULT_ASSEMBLY = ROOT / "tools/r9700/build/a8q4_n16k16_layout_qual.s"
CHECKER = ROOT / "tools/r9700/check_a8q4_n16k16_layout_static.py"
RUNNER = Path(__file__).resolve()
SOURCES = (
    ROOT / "tools/r9700/a8q4_n16k16_layout_qual.hip",
    ROOT / "tools/r9700/a8q4_n16k16_layout_harness.hip",
    ROOT / "src/ops/r9700/linear/r9700_linear.hip",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        run = subprocess.run(command, text=True, capture_output=True, check=False)
        return {"command": command, "returncode": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr}
    except OSError as error:
        return {"command": command, "error": str(error)}


def parse_harness(stdout: str) -> dict[str, Any]:
    trials, checks, medians, summary, complete, device = [], [], [], None, None, None
    for line in stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "DEVICE" and len(fields) == 4:
            device = {"ordinal": int(fields[1]), "name": fields[2], "architecture": fields[3]}
        elif fields[0] == "TRIAL" and len(fields) == 10:
            trials.append({"phase": fields[1], "shape": fields[2],
                           "n": int(fields[3]), "k": int(fields[4]),
                           "production_calls": int(fields[5]), "trial": int(fields[6]),
                           "order": fields[7], "incumbent_ms": float(fields[8]),
                           "candidate_ms": float(fields[9])})
        elif fields[0] == "CHECK":
            checks.append({"shape": fields[1], "passed": fields[2:]})
        elif fields[0] == "MEDIAN" and len(fields) == 6:
            medians.append({"shape": fields[1], "prefill_incumbent_ms": float(fields[2]),
                            "prefill_candidate_ms": float(fields[3]),
                            "decode_incumbent_ms": float(fields[4]),
                            "decode_candidate_ms": float(fields[5])})
        elif fields[0] == "SUMMARY":
            summary = {fields[i]: fields[i + 1] for i in range(1, len(fields), 2)}
        elif fields[0] == "COMPLETE" and len(fields) == 3:
            complete = {fields[1]: fields[2]}
    return {"trials": trials, "checks": checks, "medians": medians, "summary": summary,
            "complete": complete, "device": device}


def validate_hardware(sample: dict[str, Any]) -> dict[str, Any]:
    if sample.get("returncode") != 0:
        raise RuntimeError("rocm-smi query failed")
    parsed = json.loads(sample["stdout"])
    if not isinstance(parsed, dict) or len(parsed) != 1:
        raise RuntimeError("rocm-smi must identify exactly GPU0")
    key, gpu = next(iter(parsed.items()))
    if str(key).lower().replace("_", "") not in {"card0", "gpu0", "0"}:
        raise RuntimeError(f"rocm-smi did not map the device as GPU0: {key}")
    flattened = json.dumps(gpu, sort_keys=True).lower()
    if "r9700" not in flattened:
        raise RuntimeError("rocm-smi GPU0 is not identified as Radeon AI PRO R9700")
    if "auto" not in flattened:
        raise RuntimeError("GPU0 performance/power profile is not auto")
    return parsed


def validate_result(parsed: dict[str, Any]) -> dict[str, Any]:
    shapes = {"gdn_value_z": (12288, 5120, 48),
              "attention_gdn_output": (5120, 6144, 64),
              "post_mixer_down": (5120, 17408, 64)}
    device = parsed.get("device")
    if not device or device["ordinal"] != 0 or "R9700" not in device["name"] or \
            device["architecture"] != "gfx1201":
        raise RuntimeError("HIP execution did not map device 0 to gfx1201 R9700")
    trials = parsed["trials"]
    if len(trials) != 54:
        raise RuntimeError(f"expected exactly 54 raw trials, got {len(trials)}")
    for phase in ("prefill", "decode"):
        for name, (n, k, calls) in shapes.items():
            selected = [x for x in trials if x["phase"] == phase and x["shape"] == name]
            if len(selected) != 9 or {x["trial"] for x in selected} != set(range(9)):
                raise RuntimeError(f"incomplete trials for {phase}/{name}")
            if any((x["n"], x["k"], x["production_calls"]) != (n, k, calls)
                   for x in selected):
                raise RuntimeError(f"wrong production tuple for {phase}/{name}")
            expected_orders = {i: ("candidate_incumbent" if i & 1 else
                                   "incumbent_candidate") for i in range(9)}
            if any(x["order"] != expected_orders[x["trial"]] for x in selected):
                raise RuntimeError(f"unbalanced order record for {phase}/{name}")
    required_checks = {"layout_exact", "full_prefill_bit_parity", "full_decode_bit_parity",
                       "20_sampled_complete_k_fp64_coordinates", "nonuniform_fp16_scales",
                       "output_canaries"}
    if len(parsed["checks"]) != 3 or {x["shape"] for x in parsed["checks"]} != set(shapes):
        raise RuntimeError("expected exactly three production-shape correctness checks")
    if any(set(x["passed"]) != required_checks for x in parsed["checks"]):
        raise RuntimeError("correctness check inventory mismatch")
    if len(parsed["medians"]) != 3 or {x["shape"] for x in parsed["medians"]} != set(shapes):
        raise RuntimeError("expected exactly three median records")
    medians: dict[str, dict[str, float]] = {}
    for name in shapes:
        entry: dict[str, float] = {}
        for phase in ("prefill", "decode"):
            selected = [x for x in trials if x["phase"] == phase and x["shape"] == name]
            entry[f"{phase}_incumbent_ms"] = statistics.median(x["incumbent_ms"] for x in selected)
            entry[f"{phase}_candidate_ms"] = statistics.median(x["candidate_ms"] for x in selected)
        medians[name] = entry
        emitted = next(x for x in parsed["medians"] if x["shape"] == name)
        if any(abs(emitted[key] - value) > 1e-6 for key, value in entry.items()):
            raise RuntimeError(f"median recomputation failed for {name}")
    saving = sum(shapes[name][2] * (m["prefill_incumbent_ms"] - m["prefill_candidate_ms"])
                 for name, m in medians.items())
    decode_i = sum(shapes[name][2] * m["decode_incumbent_ms"] for name, m in medians.items())
    decode_c = sum(shapes[name][2] * m["decode_candidate_ms"] for name, m in medians.items())
    ratio = decode_c / decode_i
    each = all(m["prefill_candidate_ms"] / m["prefill_incumbent_ms"] <= 1.01 and
               m["decode_candidate_ms"] / m["decode_incumbent_ms"] <= 1.01
               for m in medians.values())
    summary = parsed.get("summary")
    if not summary or abs(float(summary["prefill_weighted_saving_ms"]) - saving) > 0.001 or \
       abs(float(summary["decode_weighted_ratio"]) - ratio) > 1e-5:
        raise RuntimeError("summary recomputation failed")
    gates = {"prefill_gate_ge_30ms": saving >= 30.0, "decode_gate_le_0.99": ratio <= 0.99,
             "each_shape_le_1.01": each, "guard_no_clobber": True,
             "prefill_status": True, "decode_status": True}
    if any(summary[key] != ("1" if value else "0") for key, value in gates.items()):
        raise RuntimeError("emitted admission/check gate mismatch")
    admission_pass = gates["prefill_gate_ge_30ms"] and gates["decode_gate_le_0.99"] and each
    if parsed.get("complete") != {"admission_pass": "1" if admission_pass else "0"}:
        raise RuntimeError("missing or inconsistent completion record")
    return {"prefill_weighted_saving_ms": saving, "decode_weighted_ratio": ratio,
            "each_shape_le_1.01": each, "admission_pass": admission_pass}


def publish_atomic(output: Path, payload: dict[str, Any]) -> None:
    parent = output.parent
    if not os.path.lexists(parent):
        raise FileNotFoundError(f"report parent does not exist: {parent}")
    parent_stat = os.lstat(parent)
    if not stat.S_ISDIR(parent_stat.st_mode) or Path(os.path.realpath(parent)) != parent:
        raise RuntimeError(f"report parent must be a real existing directory: {parent}")
    if os.path.lexists(output):
        os.lstat(output)
        raise FileExistsError(f"refusing to replace immutable report: {output}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp",
                                          dir=output.parent)
    temporary = Path(temporary_name)
    pending_identity = None
    published_identity = None
    try:
        opened = os.fstat(fd)
        pending_identity = (opened.st_dev, opened.st_ino)
        if not os.path.lexists(temporary):
            raise RuntimeError("pending report leaf disappeared")
        pending = os.lstat(temporary)
        if pending_identity != (pending.st_dev, pending.st_ino):
            raise RuntimeError("pending report leaf does not name the opened inode")
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid() or opened.st_nlink != 1:
            raise RuntimeError("temporary report is not an owned regular single-link inode")
        stream = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1
        with stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, output)
        published_identity = pending_identity
        final_stat = os.lstat(output)
        if (final_stat.st_dev, final_stat.st_ino) != pending_identity:
            raise RuntimeError("atomic report link inode mismatch")
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
        if output.read_bytes() != encoded or hashlib.sha256(output.read_bytes()).digest() != \
                hashlib.sha256(encoded).digest():
            raise RuntimeError("published report readback/hash verification failed")
    except Exception:
        if published_identity is not None and os.path.lexists(output):
            observed = os.lstat(output)
            if (observed.st_dev, observed.st_ino) == published_identity and \
                    observed.st_uid == os.getuid():
                output.unlink()
        raise
    finally:
        if fd >= 0:
            os.close(fd)
        if pending_identity is not None and os.path.lexists(temporary):
            observed = os.lstat(temporary)
            if (observed.st_dev, observed.st_ino) == pending_identity and \
                    observed.st_uid == os.getuid():
                temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--assembly", type=Path, default=DEFAULT_ASSEMBLY)
    args = parser.parse_args()
    binary, assembly = args.binary.resolve(), args.assembly.resolve()
    inputs = {str(path.relative_to(ROOT)): path for path in (*SOURCES, CHECKER, RUNNER,
                                                              binary, assembly)}
    inputs["tool:python"] = Path(sys.executable).resolve()
    inputs["tool:rocm-smi"] = Path("/opt/rocm/bin/rocm-smi").resolve()
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    before_hashes = {name: sha256(path) for name, path in inputs.items()}
    static = command_output([sys.executable, str(CHECKER), str(assembly)])
    if static.get("returncode") != 0:
        raise RuntimeError("static gate must pass before GPU execution")
    power_command = ["/opt/rocm/bin/rocm-smi", "-d", "0", "--showproductname", "--showserial",
                     "--showuniqueid", "--showpower", "--showclocks", "--showprofile",
                     "--showperflevel", "--json"]
    before = command_output(power_command)
    before_device = validate_hardware(before)
    run = subprocess.run([str(binary)], cwd=ROOT, text=True, capture_output=True, check=False)
    after = command_output(power_command)
    after_device = validate_hardware(after)
    after_hashes = {name: sha256(path) for name, path in inputs.items()}
    if after_hashes != before_hashes:
        raise RuntimeError("an executed input changed during the run")
    if run.returncode != 0 or run.stderr:
        raise RuntimeError(f"harness did not exit cleanly: {run.returncode}: {run.stderr}")
    parsed = parse_harness(run.stdout)
    recomputed = validate_result(parsed)
    report = {
        "schema": "ninfer.r9700.a8q4-n16k16-layout-gate.v1",
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "cwd": str(ROOT),
        "device_and_power_before": {"raw": before, "validated_json": before_device},
        "device_and_power_after": {"raw": after, "validated_json": after_device},
        "static_gate": static,
        "hashes_sha256_pre_and_post_identical": before_hashes,
        "resources": {"prefill": {"vgpr": 94, "sgpr": 22, "lds_bytes": 17152,
                                      "occupancy": 16, "sgpr_spills": 0, "vgpr_spills": 0,
                                      "native_iu4_sites": 8, "successor_weight_b64_sites": 1},
                      "decode": {"vgpr": 43, "sgpr": 22, "lds_bytes": 0,
                                  "occupancy": 16, "sgpr_spills": 0, "vgpr_spills": 0,
                                  "native_iu4_sites": 4}},
        "admission": {"prefill_weighted_saving_ms_min": 30.0,
                       "each_shape_candidate_over_incumbent_max": 1.01,
                       "decode_weighted_candidate_over_incumbent_max": 0.99,
                       "weights": {"12288x5120": 48, "5120x6144": 64,
                                   "5120x17408": 64}},
        "harness": {"returncode": run.returncode, "stdout": run.stdout,
                    "stderr": run.stderr, **parsed, "recomputed": recomputed},
    }
    # abspath normalizes lexical `.`/`..` components without following the
    # final leaf (including a dangling symlink), unlike Path.resolve().
    output = Path(os.path.abspath(os.fspath(args.output)))
    publish_atomic(output, report)
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
