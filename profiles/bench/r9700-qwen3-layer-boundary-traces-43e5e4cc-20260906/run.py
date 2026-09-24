#!/usr/bin/env python3
"""Run the exact four-arm layer-boundary functional diagnostic."""

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
    "NINFER_DFLASH_DECISION_TRACE_OUT", "NINFER_QWEN3_PREFILL_P129_TRACE",
    "NINFER_QWEN3_PREFILL_P129_TRACE_OUT", "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE",
    "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST", "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR",
}
PREFIX = ("ROCPROF", "ROCP_", "ROCTRACER_", "ROCTX_", "HSA_TOOLS_", "HIP_TRACE_",
          "AQLPROFILE_", "ATT_PROFILE")
ROLES = {
    "target-ordinary": "target-ordinary-frontier130",
    "target-dflash": "target-dflash-frontier130-column0",
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
    return sorted(key for key in os.environ if key in EXACT or key.startswith(PREFIX))


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
        fail("profiling/trace injection environment must be clean before preflight")
    plan = load(PACKAGE / "plan.json")
    build = load(PACKAGE / "build-provenance.json")
    source = plan.get("source")
    if source != {"commit": "43e5e4ccb3212aa43717104aa5f45ef717ad97b8",
                  "tree": "d489924cdde71cf7162cd83b718fc837534e191c"}:
        fail("source authority differs")
    if (plan.get("artifact_type") != "ninfer_qwen3_layer_boundary_trace_plan" or
            plan.get("schema_version") != 1 or
            plan.get("status") != "prepared_cpu_only_no_gpu_execution" or
            plan.get("claim") != "eager-only functional localization; all captured timings are ineligible" or
            plan.get("production_routing_authorized") is not False or
            plan.get("common") != {"concurrency": 1, "maximum_product_concurrency": 4,
                                   "prefill_chunk": 4096, "kv_capacity": "workload",
                                   "repetitions": 1, "warmup": 0, "device_graph": False,
                                   "retain_token_ids": True} or
            plan.get("trace") != {"hidden": 5120, "layers": 64, "snapshots": 129,
                                  "bytes_per_snapshot": 10240, "bytes_per_sidecar": 1320960,
                                  "device_staging_bytes": 1320972,
                                  "prior_seed_available_after_startup_bytes": 8948547584,
                                  "prior_seed_headroom_multiple": 6774,
                                  "runtime_requirement": "reported planned slack and available-after-startup each cover device staging bytes",
                                  "one_staging_slab_per_process": True,
                                  "layout": "input then each layer post_mixer/post_mlp",
                                  "one_final_d2h": True} or
            plan.get("limitations") != [
                "functional localization only; synchronous final D2H invalidates timing",
                "first differing represented-BF16 boundary localizes a layer half, not a primitive root cause",
                "no production routing or performance claim"]):
        fail("plan contract differs")
    if (plan.get("hardware") != {"device": 0, "name": "AMD Radeon AI PRO R9700",
            "architecture": "gfx1201", "wavefront_width": 32, "pci_bus": "0000:13:00.0",
            "pci_vendor_device": "1002:7551", "power_profile": "auto"} or
            plan.get("target_pair") != {
                "absolute_frontier": 130,
                "selected_input_token": 96558,
                "ordinary": {"role": "target-ordinary-frontier130", "width": 1,
                             "selected_column": 0, "shape": "whole-pg129,1 ordinary"},
                "dflash": {"role": "target-dflash-frontier130-column0", "width": 5,
                           "selected_column": 0, "shape": "whole-pg129,1 DFlash K4/W5"}} or
            plan.get("text_pair") != {
                "absolute_frontier": 129,
                "selected_input_token": 24178,
                "fresh": {"role": "text-fresh-frontier129-column128", "width": 129,
                          "selected_column": 128, "shape": "base0/T129"},
                "append": {"role": "text-append-frontier129-column0", "width": 1,
                           "selected_column": 0, "shape": "base128/T1 isolated"}}):
        fail("plan diagnostic/hardware scope differs")
    if (set(plan.get("history", {})) != {"source_corpus", "p129_fixture", "seed_report",
            "seed_token", "rule"} or plan["history"].get("seed_token") != 24178 or
            plan["history"].get("rule") !=
            "source corpus tokens [0,128) followed by retained fresh-P128 ordinary greedy seed" or
            plan.get("results_directory") != "results" or
            plan.get("artifact", {}).get("weights_id") !=
            "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval"):
        fail("plan artifact/history authority differs")
    worktree = Path(build["source"]["worktree"])
    expected_profile = {"GPU_BUILD_TARGETS": "gfx1201", "NINFER_R9700_KV_VALUE_GROUP": "16",
        "NINFER_R9700_Q4_ACTIVATION_BITS": "8", "NINFER_R9700_W8_ACTIVATION_BITS": "8",
        "NINFER_R9700_FP8_QK_WMMA": "1", "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE": "0",
        "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE": "0",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
        "NINFER_R9700_XATTENTION_STRIDE": "16", "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000"}
    expected_configure = ["cmake", "-S", str(worktree), "-B",
        "build-r9700-layer-boundary-43e5e4cc-20260906", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++",
        "-DNINFER_BUILD_BENCHMARKS=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"] + [
            f"-D{key}={value}" for key, value in expected_profile.items()]
    if (build.get("artifact_type") != "ninfer_qwen3_layer_boundary_trace_build" or
            build.get("schema_version") != 1 or build.get("source") != {
                "commit": source["commit"], "tree": source["tree"], "worktree": str(worktree)} or
            build.get("build_directory") != str(ROOT / "build-r9700-layer-boundary-43e5e4cc-20260906") or
            build.get("configure") != expected_configure or build.get("profile") != expected_profile or
            build.get("build") != ["cmake", "--build", "build-r9700-layer-boundary-43e5e4cc-20260906",
                "--target", "ninfer_bench", "ninfer_qwen3_runtime_mechanisms_test",
                "ninfer_qwen3_decision_trace_test", "-j", "16"]):
        fail("build recipe/profile differs")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip() != source["commit"] or \
            subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=worktree, text=True).strip() != source["tree"] or \
            subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree, text=True):
        fail("detached source worktree differs")
    if subprocess.run(["git", "diff", "--quiet", source["commit"], "--",
                       "tools/bench/compare_qwen3_layer_boundary_trace.py",
                       "tools/bench/test_compare_qwen3_layer_boundary_trace.py"], cwd=ROOT).returncode:
        fail("live comparator differs from bound commit")
    for group, keys in (("outputs", ("benchmark", "cmake_cache", "compile_commands", "runtime_test",
                                     "decision_trace_test")),
                        ("toolchain_runtime", ("cxx", "hip_compiler", "hipcc", "python",
                                               "hipblaslt", "amdhip"))):
        for key in keys:
            if identity(Path(build[group][key]["path"])) != build[group][key]:
                fail(f"build/tool identity differs: {key}")
    for key in ("artifact",):
        expected = plan[key]
        actual = identity(Path(expected["path"]))
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            fail(f"{key} identity differs")
    for key in ("source_corpus", "p129_fixture", "seed_report"):
        if identity(Path(plan["history"][key]["path"])) != plan["history"][key]:
            fail(f"history identity differs: {key}")
    source_tokens = [int(token) for token in Path(plan["history"]["source_corpus"]["path"]).read_text().split()]
    history = [int(token) for token in Path(plan["history"]["p129_fixture"]["path"]).read_text().split()]
    if len(history) != 129 or history[:128] != source_tokens[:128] or history[128] != 24178:
        fail("P129 history/seed derivation differs")
    seed = load(Path(plan["history"]["seed_report"]["path"]))
    tests = seed.get("tests")
    authority = next((test for test in tests if test.get("label") == "whole-pp128+tg64"), None) \
        if isinstance(tests, list) else None
    lanes = authority.get("reps", [{}])[0].get("generated_token_ids_by_lane") \
        if isinstance(authority, dict) and authority.get("reps") else None
    if not isinstance(lanes, list) or len(lanes) != 1 or not lanes[0] or lanes[0][0] != 24178:
        fail("seed report does not prove token 24178")
    ldd = subprocess.check_output(["ldd", str(analyze.EXE)],
                                  env={**os.environ, "LD_LIBRARY_PATH": "/opt/rocm/lib"}, text=True)
    if ("not found" in ldd or "libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1" not in ldd or
            "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7" not in ldd):
        fail("runtime DSO resolution differs")
    if (Path("/sys/class/drm/card2/device/vendor").read_text().strip() != "0x1002" or
            Path("/sys/class/drm/card2/device/device").read_text().strip() != "0x7551" or
            Path("/sys/class/drm/card2/device").resolve().name != "0000:13:00.0"):
        fail("R9700 PCI identity differs")
    power()
    for command in ([build["outputs"]["runtime_test"]["path"]],
                    [build["outputs"]["decision_trace_test"]["path"]],
                    ["/usr/bin/python3", str(ROOT / "tools/bench/test_compare_qwen3_layer_boundary_trace.py")],
                    ["/usr/bin/python3", str(PACKAGE / "test_analyze.py")]):
        subprocess.run(command, cwd=ROOT, check=True)


def close_results() -> None:
    files = sorted(path for path in RESULTS.rglob("*") if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path}\n" for path in files))


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
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    environment.update(trace_environment)
    command = analyze.command(stem, report)
    before = power()
    started = time.time_ns()
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    finished = time.time_ns()
    after = power()
    write_exclusive(report, completed.stdout)
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(process, json.dumps({
        "command": command, "trace_environment": trace_environment,
        "exit_code": completed.returncode, "started_unix_ns": started,
        "finished_unix_ns": finished, "power_before": before, "power_after": after,
        "instrumentation_environment_present": [], "stdout": identity(stdout),
        "stderr": identity(stderr),
    }, indent=2) + "\n")
    if completed.returncode or not report.is_file() or not manifest.is_file() or not sidecar.is_file():
        close_results()
        fail(f"{stem} failed; partial evidence retained")


def run_comparator(label: str, left: str, right: str) -> None:
    output = RESULTS / f"{label}-comparison.json"
    stdout = RESULTS / f"{label}-comparator.stdout"
    stderr = RESULTS / f"{label}-comparator.stderr"
    process = RESULTS / f"{label}-comparator.process.json"
    command = ["/usr/bin/python3", str(ROOT / "tools/bench/compare_qwen3_layer_boundary_trace.py"),
               "--left", str(RESULTS / f"{left}.trace.json"), "--right",
               str(RESULTS / f"{right}.trace.json"), "--diagnostic", label, "--out", str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(process, json.dumps({"command": command, "exit_code": completed.returncode,
                                         "stdout": identity(stdout), "stderr": identity(stderr)},
                                        indent=2) + "\n")
    if completed.returncode or not output.is_file():
        close_results()
        fail(f"{label} comparator failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    validate_inputs()
    if args.preflight_only:
        print("layer-boundary trace package preflight PASS")
        return 0
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("results path must be fresh")
    RESULTS.mkdir()
    try:
        for stem in ROLES:
            run_one(stem)
        run_comparator("target", "target-ordinary", "target-dflash")
        run_comparator("text", "text-fresh", "text-append")
        old = sys.argv
        sys.argv = ["analyze.py", "--summary", str(RESULTS / "summary.json")]
        try:
            analyze.main()
        finally:
            sys.argv = old
        close_results()
        print("layer-boundary trace capture and comparison PASS")
        return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists():
            close_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
