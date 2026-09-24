#!/usr/bin/env python3
"""Preflight and run the immutable two-arm recurrent-state diagnostic."""

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
    "NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST", "NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR",
    "NINFER_QWEN3_GDN_STATE_TRACE_MANIFEST", "NINFER_QWEN3_GDN_STATE_TRACE_SIDECAR",
}
PREFIX = ("ROCPROF", "ROCP_", "ROCTRACER_", "ROCTX_", "HSA_TOOLS_", "HIP_TRACE_",
          "AQLPROFILE_", "ATT_PROFILE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=analyze.pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict):
        fail(f"not object: {path}")
    return value


def identity(path: Path) -> dict:
    return analyze.identity(path)


def injected() -> list[str]:
    return sorted(key for key in os.environ if key in EXACT or key.startswith(PREFIX))


def power() -> str:
    value = POWER.read_text().strip()
    if value != "auto":
        fail(f"power profile is {value!r}, expected auto")
    return value


def write_exclusive(path: Path, payload: bytes | str) -> None:
    data = payload.encode() if isinstance(payload, str) else payload
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
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
    if (set(plan) != {"artifact_type", "schema_version", "status", "claim",
            "production_routing_authorized", "timing_evidence_eligible", "source", "artifact",
            "history", "hardware", "workload", "roles", "trace", "comparisons", "limitations",
            "expected_recurrent_state_classification", "expected_gdn_output_classification",
            "results_directory"} or
            plan.get("artifact_type") != "ninfer_qwen3_layer1_recurrent_state_text_pair_plan" or
            plan.get("schema_version") != 1 or plan.get("status") != "prepared_cpu_only_no_gpu_execution" or
            plan.get("claim") !=
            "functional-only comparison of the exact layer1 FP32 recurrent state at frontier 128 and the existing selected-column GDN boundaries" or
            plan.get("production_routing_authorized") is not False or
            plan.get("timing_evidence_eligible") is not False or plan.get("results_directory") != "results"):
        fail("plan identity differs")
    source = {"commit": "c3d44daaa41448c8a899ea8b6eee478eb1f5bdc8",
              "tree": "7dba466a7d48ae8fe54b00497f01f1582ee4223d"}
    if plan.get("source") != {**source,
            "authority": "exact clean detached committed source and fresh exact gfx1201 diagnostic build; live branch-tip equality is not required"}:
        fail("source authority differs")
    if plan.get("workload") != {"concurrency": 1, "maximum_product_concurrency": 4,
            "prefill_chunk": 4096, "warmup": 0, "repetitions": 1, "device_graph": False,
            "draft_tokens": 0, "retain_token_ids": True}:
        fail("workload differs")
    if plan.get("roles") != {
            "text-fresh": {"trace_role": "text-fresh-frontier129-column128",
                "command_shape": "whole-pg129,1", "selected_column": 128,
                "selected_token": 24178, "state_capture": "after prefix 128 before selected column 128"},
            "text-append": {"trace_role": "text-append-frontier129-column0",
                "command_shape": "isolated pg128,1", "selected_column": 0,
                "selected_token": 24178, "state_capture": "restored state before selected column 0"}}:
        fail("role identity differs")
    if plan.get("trace") != {"legacy_layer_sidecar_bytes": 1320960,
            "gdn_detail_sidecar_bytes": 78208, "recurrent_state_sidecar_bytes": 3145728,
            "recurrent_state_shape": [128,128,48], "recurrent_state_dtype": "fp32",
            "state_frontier": 128, "linear_state_slot": 0, "text_layer": 1, "gdn_index": 1,
            "combined_device_staging_bytes": 4544908, "one_combined_final_d2h": True}:
        fail("trace contract differs")
    if (plan.get("comparisons") != [
            "exact FP32 recurrent state: fresh prefix-128 versus append restored frontier-128",
            "existing selected-column layer1 GDN boundaries: fresh column128 versus append column0"] or
            plan.get("limitations") != [
            "diagnostic eager execution and final synchronous D2H make all timings ineligible",
            "state equality or divergence localizes the recurrence frontier but does not by itself identify an earlier transition",
            "no production routing or performance claim"]):
        fail("claim boundaries differ")
    if plan.get("expected_recurrent_state_classification") != {
            "classification": "first_difference_layer1_recurrent_prefix_state",
            "first_element_index": 599, "mismatch_count": 402925,
            "fresh_bits": 950422399, "append_bits": 950422400,
            "maximum_absolute_difference": 0.000046528875827789307}:
        fail("expected recurrent-state classification differs")
    if plan.get("expected_gdn_output_classification") != {
            "classification": "first_difference_gdn_recurrence", "field": "o",
            "first_element_index": 154, "mismatch_count": 106,
            "fresh_bits": 14405, "append_bits": 14406,
            "maximum_absolute_difference": 0.000003814697265625}:
        fail("expected GDN output classification differs")
    if plan.get("hardware") != {"device":0, "name":"AMD Radeon AI PRO R9700",
            "architecture":"gfx1201", "wavefront_width":32, "pci_bus":"0000:13:00.0",
            "pci_vendor_device":"1002:7551", "power_profile":"auto"}:
        fail("hardware plan differs")
    artifact = plan.get("artifact", {})
    if (artifact.get("weights_id") != "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval" or
            identity(Path(artifact.get("path", ""))) != {key: artifact[key]
            for key in ("path", "bytes", "sha256")}):
        fail("artifact identity differs")
    history = plan.get("history", {})
    if (set(history) != {"source_corpus", "p129_fixture", "seed_report", "seed_token", "rule"} or
            history.get("seed_token") != 24178 or history.get("rule") !=
            "source corpus tokens [0,128) followed by retained fresh-P128 ordinary greedy seed"):
        fail("history contract differs")
    for key in ("source_corpus", "p129_fixture", "seed_report"):
        if identity(Path(history[key]["path"])) != history[key]:
            fail(f"history identity differs: {key}")
    corpus_tokens = [int(item) for item in Path(history["source_corpus"]["path"]).read_text().split()]
    p129 = [int(item) for item in Path(history["p129_fixture"]["path"]).read_text().split()]
    if len(p129) != 129 or p129[:128] != corpus_tokens[:128] or p129[128] != 24178:
        fail("P129 fixture derivation differs")
    seed = load(Path(history["seed_report"]["path"]))
    test = next((item for item in seed.get("tests", [])
                 if item.get("label") == "whole-pp128+tg64"), None)
    lanes = test.get("reps", [{}])[0].get("generated_token_ids_by_lane") \
        if isinstance(test, dict) and test.get("reps") else None
    if not isinstance(lanes, list) or len(lanes) != 1 or not lanes[0] or lanes[0][0] != 24178:
        fail("seed report does not prove token 24178")

    source_worktree = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700-recurrent-state-src-c3d44daa")
    build_dir = ROOT / "build-r9700-recurrent-state-c3d44daa-20260906"
    expected_source = {**source, "worktree": str(source_worktree),
                       "relationship": "clean detached exact commit"}
    if (set(build) != {"artifact_type", "schema_version", "source", "build_directory",
            "configure_command", "build_command", "source_files", "outputs", "profile",
            "toolchain_runtime"} or
            build.get("artifact_type") != "ninfer_exact_committed_source_diagnostic_build_receipt" or
            build.get("schema_version") != 1 or build.get("build_directory") != str(build_dir) or
            build.get("source") != expected_source):
        fail("build receipt identity differs")
    if (subprocess.check_output(["git", "rev-parse", f"{source['commit']}^{{tree}}"],
                                cwd=ROOT, text=True).strip() != source["tree"]):
        fail("bound source commit/tree differs")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_worktree,
                                text=True).strip() != source["commit"] or
            subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=source_worktree,
                                    text=True).strip() != source["tree"] or
            subprocess.check_output(["git", "status", "--porcelain"], cwd=source_worktree,
                                    text=True)):
        fail("detached source worktree differs")
    for relative, expected in build.get("source_files", {}).items():
        blob = subprocess.check_output(["git", "show", f"{source['commit']}:{relative}"], cwd=ROOT)
        if len(blob) != expected.get("bytes") or hashlib.sha256(blob).hexdigest() != expected.get("sha256"):
            fail(f"source byte identity differs: {relative}")
    for group in ("outputs", "toolchain_runtime"):
        for key, expected in build.get(group, {}).items():
            if identity(Path(expected["path"])) != expected:
                fail(f"build/tool identity differs: {group}/{key}")
    profile = {"GPU_BUILD_TARGETS":"gfx1201", "NINFER_R9700_KV_VALUE_GROUP":"16",
        "NINFER_R9700_Q4_ACTIVATION_BITS":"8", "NINFER_R9700_W8_ACTIVATION_BITS":"8",
        "NINFER_R9700_FP8_QK_WMMA":"1", "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE":"0",
        "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE":"0",
        "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE":"0",
        "NINFER_R9700_XATTENTION_QUALIFICATION":"OFF", "NINFER_R9700_XATTENTION_STRIDE":"16",
        "NINFER_R9700_XATTENTION_TAU_PERMILLE":"1000"}
    expected_configure = ["cmake", "-S", str(source_worktree), "-B", str(build_dir),
        "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++", "-DNINFER_BUILD_BENCHMARKS=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"] + [f"-D{key}={value}"
                                                   for key, value in profile.items()]
    expected_build = ["cmake", "--build", str(build_dir), "--target", "ninfer_bench",
        "ninfer_qwen3_runtime_mechanisms_test", "ninfer_qwen3_prefill_tail_trace_path_test",
        "-j", "16"]
    if (build.get("profile") != profile or build.get("configure_command") != expected_configure or
            build.get("build_command") != expected_build):
        fail("build profile differs")
    cache = Path(build["outputs"]["cmake_cache"]["path"]).read_text()
    compile_commands = Path(build["outputs"]["compile_commands"]["path"]).read_text()
    for key, value in profile.items():
        cache_type = "BOOL" if key == "NINFER_R9700_XATTENTION_QUALIFICATION" else "STRING"
        if f"{key}:{cache_type}={value}" not in cache:
            fail(f"CMake cache profile differs: {key}")
    for key in ("NINFER_R9700_DFLASH_SMALL_T_CANDIDATE",
                "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE",
                "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE"):
        if f"-D{key}=0" not in compile_commands:
            fail(f"compiled selector differs: {key}")
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
                    [build["outputs"]["path_test"]["path"]],
                    ["/usr/bin/python3", str(source_worktree / "tools/bench/test_compare_qwen3_layer_boundary_trace.py")],
                    ["/usr/bin/python3", str(PACKAGE / "test_analyze.py")]):
        subprocess.run(command, cwd=ROOT, check=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def close_results() -> None:
    files = sorted(path for path in RESULTS.iterdir()
                   if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS / "result.sha256",
                    "".join(f"{analyze.digest(path)}  {path}\n" for path in files))


def run_one(stem: str) -> None:
    paths = {name: RESULTS / f"{stem}.{suffix}" for name, suffix in (
        ("report", "json"), ("layer_manifest", "trace.json"), ("layer_sidecar", "trace.bin"),
        ("gdn_manifest", "gdn.json"), ("gdn_sidecar", "gdn.bin"),
        ("state_manifest", "state.json"), ("state_sidecar", "state.bin"),
        ("stdout", "stdout"), ("stderr", "stderr"), ("process", "process.json"))}
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        fail(f"refusing overwrite for arm: {stem}")
    trace_environment = analyze.expected_trace_environment(stem)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    environment.update(trace_environment)
    command = analyze.command(stem)
    before = power()
    started = time.time_ns()
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    finished = time.time_ns()
    after = power()
    write_exclusive(paths["report"], completed.stdout)
    write_exclusive(paths["stdout"], completed.stdout)
    write_exclusive(paths["stderr"], completed.stderr)
    write_exclusive(paths["process"], json.dumps({
        "command": command, "trace_environment": trace_environment,
        "exit_code": completed.returncode, "started_unix_ns": started,
        "finished_unix_ns": finished, "power_before": before, "power_after": after,
        "instrumentation_environment_present": [], "stdout": identity(paths["stdout"]),
        "stderr": identity(paths["stderr"]),
    }, indent=2) + "\n")
    required = ("report", "layer_manifest", "layer_sidecar", "gdn_manifest", "gdn_sidecar",
                "state_manifest", "state_sidecar")
    if completed.returncode or any(not paths[key].is_file() for key in required):
        close_results()
        fail(f"{stem} failed; partial evidence retained")


def run_comparator(kind: str) -> None:
    output = RESULTS / f"text-{kind}-comparison.json"
    stdout = RESULTS / f"text-{kind}-comparator.stdout"
    stderr = RESULTS / f"text-{kind}-comparator.stderr"
    process = RESULTS / f"text-{kind}-comparator.process.json"
    if any(path.exists() or path.is_symlink() for path in (output, stdout, stderr, process)):
        fail(f"refusing comparator overwrite: {kind}")
    command = analyze.comparator_command(kind, output)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(process, json.dumps({"command": command, "exit_code": completed.returncode,
                                         "stdout": identity(stdout), "stderr": identity(stderr)},
                                        indent=2) + "\n")
    if completed.returncode or not output.is_file():
        close_results()
        fail(f"{kind} comparator failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    validate_inputs()
    if args.preflight_only:
        print("layer1 recurrent-state Text pair preflight PASS")
        return 0
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("results path must be fresh")
    RESULTS.mkdir()
    try:
        for stem in analyze.ROLES:
            run_one(stem)
        run_comparator("gdn")
        run_comparator("state")
        analyze.write_summary(RESULTS / "summary.json")
        close_results()
        print("layer1 recurrent-state Text pair capture PASS")
        return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists():
            close_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
