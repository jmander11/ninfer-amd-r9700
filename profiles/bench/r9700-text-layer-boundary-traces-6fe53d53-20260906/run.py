#!/usr/bin/env python3
"""Run the exact two-arm Text append-versus-fresh layer-boundary diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
RESULTS = PACKAGE / "results"
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
sys.path.insert(0, str(PACKAGE))
import analyze  # noqa: E402

EXACT = {
    "LD_PRELOAD", "LD_AUDIT", "HIP_FORCE_QUEUE_PROFILING", "AMD_SERIALIZE_KERNEL",
    "AMD_SERIALIZE_COPY", "ROC_SERIALIZE_KERNEL", "GPU_DUMP_CODE_OBJECT",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
    "NINFER_DFLASH_DECISION_TRACE_OUT", "NINFER_QWEN3_PREFILL_P129_TRACE",
    "NINFER_QWEN3_PREFILL_P129_TRACE_OUT", "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE",
    "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST", "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR",
    "NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST", "NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR",
    "NINFER_QWEN3_GDN_STATE_TRACE_MANIFEST", "NINFER_QWEN3_GDN_STATE_TRACE_SIDECAR",
    "NINFER_QWEN3_LAYER3_ATTENTION_TRACE_MANIFEST", "NINFER_QWEN3_LAYER3_ATTENTION_TRACE_SIDECAR",
}
PREFIX = ("ROCPROF", "ROCP_", "ROCTRACER_", "ROCTX_", "HSA_TOOLS_", "HIP_TRACE_",
          "AQLPROFILE_", "ATT_PROFILE")
ROLES = {
    "text-fresh": "text-fresh-frontier129-column128",
    "text-append": "text-append-frontier129-column0",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest(resolved)}


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"not object: {path}")
    return value


def injected() -> list[str]:
    return sorted(key for key in os.environ
                  if key in EXACT or key.startswith(PREFIX) or
                  (key.startswith("NINFER_") and "TRACE" in key))


def power() -> str:
    value = POWER.read_text().strip()
    if value != "auto":
        fail(f"power profile is {value!r}, expected auto")
    return value


def write_exclusive(path: Path, payload: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    data = payload.encode()
    try:
        offset = 0
        while offset != len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                fail(f"write made no progress: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_inputs() -> None:
    if injected():
        fail(f"profiling/trace injection environment must be clean before preflight: {injected()}")
    plan = analyze.validate_plan()
    build = analyze.validate_build()
    worktree = Path(build["source"]["worktree"])
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip() !=
            build["source"]["commit"] or
            subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=worktree,
                                    text=True).strip() != build["source"]["tree"] or
            subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree, text=True)):
        fail("detached source worktree differs")
    if subprocess.run(["git", "diff", "--quiet", build["source"]["commit"], "--",
                       "tools/bench/compare_qwen3_layer_boundary_trace.py",
                       "tools/bench/test_compare_qwen3_layer_boundary_trace.py"],
                      cwd=ROOT).returncode:
        fail("live comparator differs from bound commit")
    for key in ("artifact",):
        expected = plan[key]
        actual = identity(Path(expected["path"]))
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            fail(f"{key} identity differs")
    for key in ("source_corpus", "p129_fixture"):
        if identity(Path(plan["history"][key]["path"])) != plan["history"][key]:
            fail(f"history identity differs: {key}")
    env = {**os.environ, "LD_LIBRARY_PATH": "/opt/rocm/core-10.0/lib",
           "PYTHONDONTWRITEBYTECODE": "1"}
    ldd = subprocess.check_output(["ldd", str(analyze.EXE)], env=env, text=True)
    if ("not found" in ldd or
            "libhipblaslt.so.1 => /opt/rocm/core-10.0/lib/libhipblaslt.so.1" not in ldd or
            "libamdhip64.so.7 => /opt/rocm/core-10.0/lib/libamdhip64.so.7" not in ldd):
        fail("runtime DSO resolution differs")
    device = Path("/sys/class/drm/card2/device")
    if (device.resolve().name != "0000:13:00.0" or
            (device / "vendor").read_text().strip() != "0x1002" or
            (device / "device").read_text().strip() != "0x7551"):
        fail("R9700 PCI identity differs")
    power()
    for key in ("bench_support", "runtime_mechanisms", "decision_trace", "trace_path"):
        subprocess.run([build["outputs"][key]["path"]], cwd=worktree, env=env, check=True)
    subprocess.run([build["outputs"]["runtime_planner"]["path"],
                    "--host-attention-parity-routing"], cwd=worktree, env=env, check=True)
    subprocess.run(["/usr/bin/python3", str(ROOT / "tools/bench/test_compare_qwen3_layer_boundary_trace.py")],
                   cwd=ROOT, env=env, check=True)
    subprocess.run(["/usr/bin/python3", str(PACKAGE / "test_analyze.py")], cwd=ROOT, env=env,
                   check=True)


def close_results() -> None:
    files = sorted(path for path in RESULTS.rglob("*")
                   if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS / "result.sha256",
                    "".join(f"{digest(path)}  {path}\n" for path in files))


def run_one(stem: str) -> None:
    report = RESULTS / f"{stem}.json"
    manifest = RESULTS / f"{stem}.trace.json"
    sidecar = RESULTS / f"{stem}.trace.bin"
    stdout = RESULTS / f"{stem}.stdout"
    stderr = RESULTS / f"{stem}.stderr"
    process = RESULTS / f"{stem}.process.json"
    for path in (report, manifest, sidecar, stdout, stderr, process):
        if path.exists() or path.is_symlink():
            fail(f"refusing overwrite: {path}")
    trace_environment = {
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": ROLES[stem],
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST": str(manifest),
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR": str(sidecar),
    }
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/core-10.0/lib"
    environment.update(trace_environment)
    command = analyze.command(stem, report)
    before = power()
    started = time.time_ns()
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    finished = time.time_ns()
    after = power()
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(process, json.dumps({
        "command": command, "trace_environment": trace_environment,
        "exit_code": completed.returncode, "started_unix_ns": started,
        "finished_unix_ns": finished, "power_before": before, "power_after": after,
        "instrumentation_environment_present": [], "stdout": identity(stdout),
        "stderr": identity(stderr),
    }, indent=2) + "\n")
    if completed.returncode or not report.is_file() or not manifest.is_file() or \
            not sidecar.is_file():
        close_results()
        fail(f"{stem} failed; partial evidence retained")


def run_comparator() -> None:
    output = RESULTS / "text-comparison.json"
    stdout = RESULTS / "text-comparator.stdout"
    stderr = RESULTS / "text-comparator.stderr"
    process = RESULTS / "text-comparator.process.json"
    command = ["/usr/bin/python3", str(analyze.COMPARATOR), "--left",
               str(RESULTS / "text-fresh.trace.json"), "--right",
               str(RESULTS / "text-append.trace.json"), "--diagnostic", "text",
               "--out", str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(process, json.dumps({"command": command, "exit_code": completed.returncode,
                                         "stdout": identity(stdout),
                                         "stderr": identity(stderr)}, indent=2) + "\n")
    if completed.returncode or not output.is_file():
        close_results()
        fail("text comparator failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    validate_inputs()
    if args.preflight_only:
        print("text layer-boundary trace package preflight PASS")
        return 0
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("results path must be fresh")
    RESULTS.mkdir()
    try:
        for stem in ROLES:
            run_one(stem)
        run_comparator()
        old = sys.argv
        sys.argv = ["analyze.py", "--summary", str(RESULTS / "summary.json")]
        try:
            analyze.main()
        finally:
            sys.argv = old
        close_results()
        print("text layer-boundary trace capture and comparison PASS")
        return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists():
            close_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
