#!/usr/bin/env python3
"""Numerical qualification before unprofiled appended-chunk attention measurement."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.bench.run_ninfer_bench_matrix import (
    R9700_POWER_PROFILE, load_bench_report, require_auto_power_profile, require_hip_pci_device,
)
from tools.ppl.run import file_sha256, inspect_candidate_artifact
from tools.r9700.check_dense_prefill_attention_static import check as check_static

PACKAGE = Path(__file__).resolve().parent
PYTHON = Path("/home/battlefront/.local/bin/python3.11")
FROZEN = REPO / "profiles/bench/r9700-chunk-selection-receipt-bound-n16k16-20260921/inputs.json"
BASELINE = REPO / "build-r9700-selection-dense-g16-20260921/bench/ninfer_bench"
CANDIDATE = REPO / "build-r9700-chunked-attention-candidate-g16-20260921/bench/ninfer_bench"
BUILD = CANDIDATE.parent.parent
QUALIFIER = BUILD / "qualification/dense_prefill_attention_qual"
PLANNER = BUILD / "src/ninfer_r9700_runtime_planner_qual"
LEAF = BUILD / "src/ninfer_r9700_full_attention_qual"
ASSEMBLY = BUILD / "qualification/dense_prefill_attention.s"
SOURCE_PATHS = (
    "src/ops/r9700/kv/fp8_int4_kv_attention.hip",
    "src/ops/r9700/kv/fp8_int4_kv_attention.h",
    "src/ops/r9700/kv/r9700_attention_profile.h",
    "src/targets/qwen3_8_27b/impl/r9700_full_attention.hip",
    "tools/r9700/dense_prefill_attention_qual.hip",
    "tools/r9700/check_dense_prefill_attention_static.py",
    "tools/r9700/runtime_planner_qual.cpp",
    "tools/r9700/full_attention_leaf_qual.hip",
)
WEIGHTS = REPO / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
CORPUS = REPO / "bench/fixtures/bench_corpus.ids"
RETAINED = REPO / ("profiles/bench/prefill-chunk-screen-all-q4-g16-dense-receipt-bound-n16k16-20260921/"
                   "json/production_prefill_chunk/c1/prefill_p8192_chunk1024_ordinary.json")


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def identity(path):
    return {"path": str(path.resolve(strict=True)), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def idle():
    require_auto_power_profile()
    device = R9700_POWER_PROFILE.parent
    used = int((device / "mem_info_vram_used").read_text())
    busy = int((device / "gpu_busy_percent").read_text())
    if used > 1024 ** 3 or busy:
        raise ValueError(f"R9700 must be idle: resident={used} bytes busy={busy}%")


def previous_job_released():
    # Under the root's exclusive GPU lease, preceding children are ours and have
    # exited. The sampled busy counter can lag their completion; check resident
    # memory and auto power instead of mistaking that sample for a new GPU job.
    require_auto_power_profile()
    used = int((R9700_POWER_PROFILE.parent / "mem_info_vram_used").read_text())
    if used > 1024 ** 3:
        raise ValueError(f"previous job has not released R9700 residency: {used} bytes")


def bench_command(binary, prompt, chunk, output):
    return [str(binary), "--weights", str(WEIGHTS), "--corpus", str(CORPUS),
            "--device", "0", "--concurrency", "1", "-p", str(prompt),
            "--prefill-chunk", str(chunk), "--draft-tokens", "0", "--output", "json",
            "--output-file", str(output), "-r", "3", "--warmup", "1"]


def report(path, artifact, command, prompt, chunk):
    expected = dict(artifact)
    expected["file_size_bytes"] = expected.pop("bytes")
    result = load_bench_report(
        path, expected_kv_value_group=16, expected_q4_activation_bits=8,
        expected_w8_activation_bits=8, expected_fp8_qk_wmma=True,
        expected_concurrency=1, expected_artifact=expected, expected_command=command,
        expected_xattention_profile="dense")
    config = result["config"]
    for key, wanted in {"prefill_chunk": chunk, "spec": "none", "draft_tokens": 0,
                        "speculative_execution": False, "repetitions": 3, "warmup": 1}.items():
        if config.get(key) != wanted:
            raise ValueError(f"benchmark {key} differs from experiment: {path}")
    tests = result.get("tests")
    if (not isinstance(tests, list) or len(tests) != 1 or tests[0].get("kind") != "pp"
            or tests[0].get("n_prompt") != prompt or tests[0].get("n_gen") != 0):
        raise ValueError(f"benchmark workload differs from experiment: {path}")
    if not math.isfinite(tests[0]["prefill_tok_s_mean"]) or tests[0]["prefill_tok_s_mean"] <= 0:
        raise ValueError("benchmark lacks finite positive throughput")
    return result


def preflight():
    frozen = json.loads(FROZEN.read_text())
    if (frozen.get("artifact_type") != "ninfer_r9700_chunk_campaign_inputs"
            or frozen.get("schema_version") != 1):
        raise ValueError("unsupported frozen baseline input receipt")
    for path in (BASELINE, CORPUS):
        if file_sha256(path) != frozen["files"].get(str(path.relative_to(REPO))):
            raise ValueError(f"baseline input changed: {path}")
    artifact = inspect_candidate_artifact(WEIGHTS)
    if artifact not in frozen["artifacts"]:
        raise ValueError("experiment artifact differs from frozen baseline")
    baseline = report(RETAINED, artifact, bench_command(BASELINE, 8192, 1024, RETAINED), 8192, 1024)
    if baseline["tests"][0]["prefill_tok_s_mean"] != 184.8640144:
        raise ValueError("retained baseline differs from the measured 184.8640144 tok/s authority")
    for executable in (BASELINE, CANDIDATE, QUALIFIER, PLANNER, LEAF):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"required executable unavailable: {executable}")
    require_auto_power_profile()
    return {"artifact": artifact, "baseline": identity(BASELINE), "candidate": identity(CANDIDATE),
            "corpus": identity(CORPUS), "retained_baseline": identity(RETAINED),
            "qualifier": identity(QUALIFIER), "assembly": identity(ASSEMBLY),
            "planner": identity(PLANNER), "public_leaf": identity(LEAF),
            "sources": {name: identity(REPO / name) for name in SOURCE_PATHS},
            "build": {name: identity(BUILD / name) for name in ("CMakeCache.txt", "compile_commands.json")}}


def static_checks():
    text = ASSEMBLY.read_text()
    checks = []
    for stage, group, specialization in (
        ("qk_bk16", 16, "26dense_full_score_qk_kernelILb0EE"),
        ("qk_bk32", 16, "31dense_full_score_qk_bk32_kernelILb0EE"),
        ("maximum", 16, "31dense_full_score_maximum_kernelILb0EE"),
        ("pv", 16, "26dense_full_score_pv_kernelILj16ELb0EE"),
        ("pv", 32, "26dense_full_score_pv_kernelILj32ELb0EE"),
    ):
        symbols = re.findall(r"^(_Z\w*" + specialization + r"\w*):", text, re.MULTILINE)
        if len(symbols) != 1:
            raise ValueError(f"ISA lacks one exact {stage}/G{group} production specialization")
        checks.append(check_static(assembly=ASSEMBLY, metadata=ASSEMBLY, symbol=symbols[0],
                                   value_group=group, full_score_stage=stage))
    return checks


def qualification(inputs):
    output = PACKAGE / "qualification"
    if os.path.lexists(output):
        raise ValueError(f"output already exists: {output}")
    static = static_checks()
    idle()
    output.mkdir()
    write_new(output / "inputs.json", inputs)
    write_new(output / "static.json", static)
    run_logged([str(PLANNER), "--host-split512-routing"], output, "host-routing", check_idle=False)
    require_hip_pci_device(0)
    # The qualifier has a no-argument interface and covers both G16 and G32,
    # including appended 8K/32K panels, panel tails, graph replay and active rows.
    run_logged([str(QUALIFIER)], output, "dense-prefill-attention", check_idle=False)
    run_logged([str(LEAF)], output, "full-attention-leaf")
    if preflight() != inputs:
        raise ValueError("qualification inputs changed during execution")
    write_new(output / "result.json", {"pass": True, "inputs": inputs,
                                      "static_complete": True, "numerical_complete": True,
                                      "command": [str(QUALIFIER)], "power_profile": "auto"})


def run_logged(command, directory, name, *, check_idle=True):
    if check_idle:
        previous_job_released()
    else:
        # The stage already established idle before the HIP PCI probe. Its own
        # initialization can transiently raise gpu_busy_percent after it exits.
        require_auto_power_profile()
    with (directory / f"{name}.stdout.txt").open("x") as stdout, (directory / f"{name}.stderr.txt").open("x") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    require_auto_power_profile()
    if completed.returncode:
        raise ValueError(f"{name} failed ({completed.returncode}); outputs preserved in {directory}")


def whole(inputs):
    qualification = json.loads((PACKAGE / "qualification/result.json").read_text())
    if (qualification.get("pass") is not True or qualification.get("inputs") != inputs
            or qualification.get("static_complete") is not True
            or qualification.get("numerical_complete") is not True):
        raise ValueError("whole measurement requires passing qualification of these exact inputs")
    output = PACKAGE / "whole"
    if os.path.lexists(output):
        raise ValueError(f"output already exists: {output}")
    idle()
    require_hip_pci_device(0)
    output.mkdir()
    write_new(output / "inputs.json", inputs)
    results = {}
    for label, binary, prompt, chunk in (
        ("candidate-p8192", CANDIDATE, 8192, 1024),
        ("baseline-p2048", BASELINE, 2048, 4096),
        ("candidate-p2048", CANDIDATE, 2048, 4096),
    ):
        path = output / f"{label}.json"
        command = bench_command(binary, prompt, chunk, path)
        if identity(binary) != inputs["baseline" if binary == BASELINE else "candidate"]:
            raise ValueError("benchmark executable changed after qualification")
        run_logged(command, output, label, check_idle=label != "candidate-p8192")
        results[label] = report(path, inputs["artifact"], command, prompt, chunk)["tests"][0]
    if preflight() != inputs:
        raise ValueError("experiment inputs changed during measurement")
    baseline8 = report(RETAINED, inputs["artifact"], bench_command(BASELINE, 8192, 1024, RETAINED), 8192, 1024)["tests"][0]
    results["retained-baseline-p8192"] = baseline8
    decision = admission(results)
    write_new(output / "result.json", {
        "artifact_type": "ninfer_chunked_attention_whole_prefill_comparison", "schema_version": 1,
        **decision,
        "inputs": inputs, "power_profile": "auto", "pci": "0000:13:00.0",
        "qualification": identity(PACKAGE / "qualification/result.json"),
        "retained_baseline_p8192_tok_s": 184.8640144,
        "measurements": results,
    })
    if decision["status"] != "admitted":
        raise ValueError("whole-inference admission failed; result retained; no automatic retry")


def admission(results):
    medians = {}
    for name, test in results.items():
        times = [rep["timings"]["prefill_seconds"] for rep in test["reps"]]
        if len(times) != 3 or any(not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError(f"{name} lacks three finite positive prefill measurements")
        medians[name] = statistics.median(times)
    ratio8 = medians["candidate-p8192"] / medians["retained-baseline-p8192"]
    ratio2 = medians["candidate-p2048"] / medians["baseline-p2048"]
    return {"status": "admitted" if ratio8 <= 0.90 and ratio2 <= 1.02 else "not_admitted",
            "median_prefill_seconds": medians,
            "candidate_over_baseline_duration_p8192": ratio8,
            "candidate_over_baseline_duration_p2048": ratio2,
            "maximum_duration_ratio_p8192": 0.90, "maximum_duration_ratio_p2048": 1.02}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "qualification", "whole"))
    args = parser.parse_args()
    inputs = preflight()
    if args.stage == "qualification":
        qualification(inputs)
    elif args.stage == "whole":
        whole(inputs)
    else:
        print("experiment prerequisites verified; no GPU work or output creation")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
