#!/usr/bin/env python3
"""Preflight and run the immutable four-arm layer-1 GDN detail diagnostic."""

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


def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            fail(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict):
        fail(f"not object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest(resolved)}


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
    source = {"commit": "e915a5e45a51c07784b36aede8650352bfe5e374",
              "tree": "8f447301e37f1263ab6431164276731ded6a12f1"}
    required_plan_keys = {"artifact_type", "schema_version", "status", "source", "claim",
                          "production_routing_authorized", "artifact", "history", "hardware",
                          "common", "selectors", "target_pair", "text_pair", "trace", "comparisons",
                          "prior_selector_on_parity", "limitations", "results_directory"}
    if (set(plan) != required_plan_keys or
            plan.get("artifact_type") != "ninfer_qwen3_layer1_gdn_detail_trace_plan" or
            plan.get("schema_version") != 1 or plan.get("source") != source or
            plan.get("status") != "prepared_cpu_only_no_gpu_execution" or
            plan.get("claim") !=
            "eager-only functional layer1 GDN localization; all captured timings are ineligible" or
            plan.get("production_routing_authorized") is not False or
            plan.get("results_directory") != "results"):
        fail("plan identity differs")
    if plan.get("common") != {"concurrency": 1, "maximum_product_concurrency": 4,
            "prefill_chunk": 4096, "kv_capacity": "workload", "repetitions": 1,
            "warmup": 0, "device_graph": False, "retain_token_ids": True}:
        fail("plan execution scope differs")
    if (plan.get("target_pair") != {"absolute_frontier":130, "selected_input_token":96558,
            "ordinary":{"role":"target-ordinary-frontier130", "width":1,
                        "selected_column":0, "shape":"whole-pg129,1 ordinary"},
            "dflash":{"role":"target-dflash-frontier130-column0", "width":5,
                       "selected_column":0, "shape":"whole-pg129,1 DFlash K4/W5"}} or
            plan.get("text_pair") != {"absolute_frontier":129, "selected_input_token":24178,
            "fresh":{"role":"text-fresh-frontier129-column128", "width":129,
                     "selected_column":128, "shape":"base0/T129"},
            "append":{"role":"text-append-frontier129-column0", "width":1,
                      "selected_column":0, "shape":"base128/T1 isolated"}} or
            plan.get("comparisons") != [
                "target ordinary W1 column0 versus DFlash W5 column0",
                "text fresh W129 column128 versus append W1 column0"] or
            plan.get("limitations") != [
                "functional localization only; synchronous final D2H invalidates timing",
                "captures selected columns and no persistent convolution or recurrence state",
                "no production routing or performance claim"]):
        fail("plan diagnostic contract differs")
    if plan.get("selectors") != {"dflash_small_t_candidate": False,
            "dflash_mlp_down_t5_candidate": True,
            "dflash_rmsnorm_rows56_candidate": True,
            "ordinary_w1_uses_rows56_or_t5_candidates": False}:
        fail("plan selector scope differs")
    parity = plan.get("prior_selector_on_parity", {})
    if (parity.get("summary") != identity(Path(
            "/ssdpool2nvme/local_llm/ninfer-amd-r9700/profiles/bench/r9700-dflash-rmsnorm-rows56-token-parity-20260906/results/summary.json")) or
            {key: parity.get(key) for key in ("matches_ordinary_through_index",
             "first_difference_index", "ordinary_index21", "selector_on_index21",
             "ordinary_index27", "selector_on_index27")} != {
                "matches_ordinary_through_index":20, "first_difference_index":21,
                "ordinary_index21":96723, "selector_on_index21":128415,
                "ordinary_index27":95946, "selector_on_index27":96843} or
            parity.get("purpose") !=
            "bypass the known W5 generic RMSNorm divergence and localize the next primitive"):
        fail("prior selector-on parity authority differs")
    trace = plan.get("trace", {})
    if (trace.get("legacy_layer_sidecar_bytes") != 1320960 or
            trace.get("gdn_detail_sidecar_bytes") != 78208 or
            trace.get("metadata_bytes") != 12 or
            trace.get("combined_device_staging_bytes") != 1399180 or
            trace.get("one_combined_final_d2h") is not True or
            trace.get("persistent_state_captured") is not False or
            trace.get("gdn_fields") != analyze.GDN_FIELDS or
            trace.get("runtime_requirement") !=
            "reported planned slack and available-after-startup each cover 1399180 bytes"):
        fail("plan trace layout differs")
    if plan.get("hardware") != {"device": 0, "name": "AMD Radeon AI PRO R9700",
            "architecture": "gfx1201", "wavefront_width": 32, "pci_bus": "0000:13:00.0",
            "pci_vendor_device": "1002:7551", "power_profile": "auto"}:
        fail("hardware plan differs")
    for key in ("artifact",):
        expected = plan[key]
        if expected.get("weights_id") != "r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval":
            fail("artifact weights identity differs")
        actual = identity(Path(expected["path"]))
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            fail(f"{key} identity differs")
    for key in ("source_corpus", "p129_fixture", "seed_report"):
        expected = plan["history"][key]
        if identity(Path(expected["path"])) != expected:
            fail(f"history identity differs: {key}")
    if (set(plan.get("history", {})) != {"source_corpus", "p129_fixture", "seed_report",
            "seed_token", "rule"} or plan["history"].get("seed_token") != 24178 or
            plan["history"].get("rule") !=
            "source corpus tokens [0,128) followed by retained fresh-P128 ordinary greedy seed"):
        fail("history contract differs")
    source_tokens = [int(token) for token in Path(plan["history"]["source_corpus"]["path"]).read_text().split()]
    history = [int(token) for token in Path(plan["history"]["p129_fixture"]["path"]).read_text().split()]
    if len(history) != 129 or history[:128] != source_tokens[:128] or history[128] != 24178:
        fail("P129 history derivation differs")
    seed = load(Path(plan["history"]["seed_report"]["path"]))
    authority = next((item for item in seed.get("tests", [])
                      if item.get("label") == "whole-pp128+tg64"), None)
    lanes = authority.get("reps", [{}])[0].get("generated_token_ids_by_lane") \
        if isinstance(authority, dict) and authority.get("reps") else None
    if not isinstance(lanes, list) or len(lanes) != 1 or not lanes[0] or lanes[0][0] != 24178:
        fail("seed report does not prove token 24178")

    worktree = Path(build["source"]["worktree"])
    expected_profile = {"GPU_BUILD_TARGETS":"gfx1201", "NINFER_R9700_KV_VALUE_GROUP":"16",
        "NINFER_R9700_Q4_ACTIVATION_BITS":"8", "NINFER_R9700_W8_ACTIVATION_BITS":"8",
        "NINFER_R9700_FP8_QK_WMMA":"1", "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE":"0",
        "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE":"1",
        "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE":"1",
        "NINFER_R9700_XATTENTION_QUALIFICATION":"OFF",
        "NINFER_R9700_XATTENTION_STRIDE":"16", "NINFER_R9700_XATTENTION_TAU_PERMILLE":"1000"}
    expected_configure = ["cmake", "-S", str(worktree), "-B",
        "build-r9700-layer1-gdn-e915a5e4-20260906", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++",
        "-DNINFER_BUILD_BENCHMARKS=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"] + [
            f"-D{key}={value}" for key, value in expected_profile.items()]
    expected_build = ["cmake", "--build", "build-r9700-layer1-gdn-e915a5e4-20260906",
        "--target", "ninfer_bench", "ninfer_qwen3_runtime_mechanisms_test",
        "ninfer_qwen3_decision_trace_test", "ninfer_qwen3_prefill_tail_trace_path_test",
        "-j", "16"]
    if (set(build) != {"artifact_type", "schema_version", "source", "build_directory",
                       "configure", "build", "profile", "outputs", "toolchain_runtime"} or
            build.get("artifact_type") != "ninfer_qwen3_layer1_gdn_detail_trace_build" or
            build.get("schema_version") != 1 or build.get("source") != {
                **source, "worktree": str(worktree)} or
            build.get("build_directory") != str(ROOT / "build-r9700-layer1-gdn-e915a5e4-20260906") or
            build.get("configure") != expected_configure or build.get("build") != expected_build or
            build.get("profile") != expected_profile):
        fail("build provenance differs")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip() != source["commit"] or
            subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=worktree, text=True).strip() != source["tree"] or
            subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree, text=True)):
        fail("detached source worktree differs")
    for group, keys in (("outputs", ("benchmark", "cmake_cache", "compile_commands",
                                     "runtime_test", "decision_trace_test", "path_test")),
                        ("toolchain_runtime", ("cxx", "hip_compiler", "hipcc", "python",
                                               "hipblaslt", "amdhip"))):
        for key in keys:
            if identity(Path(build[group][key]["path"])) != build[group][key]:
                fail(f"build/tool identity differs: {key}")
    cache = Path(build["outputs"]["cmake_cache"]["path"]).read_text()
    compile_commands = Path(build["outputs"]["compile_commands"]["path"]).read_text()
    for name, value in (("NINFER_R9700_DFLASH_SMALL_T_CANDIDATE", "0"),
                        ("NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE", "1"),
                        ("NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE", "1")):
        if (f"{name}:STRING={value}" not in cache or
                f"-D{name}={value}" not in compile_commands):
            fail(f"compiled selector differs: {name}")
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
                    [build["outputs"]["path_test"]["path"]],
                    ["/usr/bin/python3", str(worktree / "tools/bench/test_compare_qwen3_layer_boundary_trace.py")],
                    ["/usr/bin/python3", str(PACKAGE / "test_analyze.py")]):
        subprocess.run(command, cwd=worktree, check=True)


def close_results() -> None:
    files = sorted(path for path in RESULTS.rglob("*") if path.is_file() and path.name != "result.sha256")
    write_exclusive(RESULTS / "result.sha256", "".join(f"{digest(path)}  {path}\n" for path in files))


def run_one(stem: str) -> None:
    paths = {
        "report": RESULTS / f"{stem}.json", "layer_manifest": RESULTS / f"{stem}.trace.json",
        "layer_sidecar": RESULTS / f"{stem}.trace.bin",
        "gdn_manifest": RESULTS / f"{stem}.gdn.json", "gdn_sidecar": RESULTS / f"{stem}.gdn.bin",
        "stdout": RESULTS / f"{stem}.stdout", "stderr": RESULTS / f"{stem}.stderr",
        "process": RESULTS / f"{stem}.process.json",
    }
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        fail(f"refusing overwrite for arm: {stem}")
    trace_environment = {
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_ROLE": ROLES[stem],
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_MANIFEST": str(paths["layer_manifest"]),
        "NINFER_QWEN3_LAYER_BOUNDARY_TRACE_SIDECAR": str(paths["layer_sidecar"]),
        "NINFER_QWEN3_GDN_DETAIL_TRACE_MANIFEST": str(paths["gdn_manifest"]),
        "NINFER_QWEN3_GDN_DETAIL_TRACE_SIDECAR": str(paths["gdn_sidecar"]),
    }
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    environment.update(trace_environment)
    command = analyze.command(stem, paths["report"])
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
    if completed.returncode or any(not paths[key].is_file() for key in
                                   ("report", "layer_manifest", "layer_sidecar",
                                    "gdn_manifest", "gdn_sidecar")):
        close_results()
        fail(f"{stem} failed; partial evidence retained")


def run_comparator(label: str, left: str, right: str) -> None:
    output = RESULTS / f"{label}-gdn-comparison.json"
    stdout = RESULTS / f"{label}-gdn-comparator.stdout"
    stderr = RESULTS / f"{label}-gdn-comparator.stderr"
    process = RESULTS / f"{label}-gdn-comparator.process.json"
    command = analyze.comparator_command(label, left, right, output)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    write_exclusive(process, json.dumps({"command": command, "exit_code": completed.returncode,
                                         "stdout": identity(stdout), "stderr": identity(stderr)},
                                        indent=2) + "\n")
    if completed.returncode or not output.is_file():
        close_results()
        fail(f"{label} GDN comparator failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    validate_inputs()
    if args.preflight_only:
        print("layer1 GDN detail trace package preflight PASS")
        return 0
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("results path must be fresh")
    RESULTS.mkdir()
    try:
        for stem in ROLES:
            run_one(stem)
        run_comparator("target", "target-ordinary", "target-dflash")
        run_comparator("text", "text-fresh", "text-append")
        analyze.write_summary(RESULTS / "summary.json")
        close_results()
        print("layer1 GDN detail trace capture and comparison PASS")
        return 0
    except Exception:
        if RESULTS.is_dir() and not (RESULTS / "result.sha256").exists():
            close_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
