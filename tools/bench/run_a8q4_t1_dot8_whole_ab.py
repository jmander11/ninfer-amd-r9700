#!/usr/bin/env python3
"""Build and run the temporary source-matched ordinary-T1 dot8 whole A/B gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFINE = "NINFER_R9700_A8Q4_T1_DOT8_QUALIFICATION"
FACTOR = 4.4478
DECODE_RATIO_LIMIT = 1.0
PREFILL_RATIO_LIMIT = 1.01
DECODE_SAVING_FLOOR_MS_PER_TOKEN = 5.0
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"
CORPUS = ROOT / "bench/fixtures/bench_corpus.ids"
OPERATOR_REPORT = ROOT / "profiles/bench/r9700-a8q4-t1-native-dot8-all-text-20260905.json"
OPERATOR_REPORT_SHA256 = "300626f0d45b5b9bb8f6b420f44b7e5652e3a848738c636959f02f3448d2b5b0"
ARTIFACT_SHA256 = "60719c5d5bfe978376c7de94db460bcd82d965ec447870cc47f0a02bf8d59953"
ARTIFACT_SIZE = 15172833280
CORPUS_SHA256 = "27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002"


def _run(command: Sequence[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _run_benchmark(command: Sequence[str], report_path: Path) -> tuple[str, str]:
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expected_stdout = f"wrote {report_path}\n"
    if completed.stdout != expected_stdout:
        raise ValueError("benchmark emitted non-canonical report notice")
    return completed.stdout, completed.stderr


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_digest() -> str:
    files = subprocess.run(
        ["git", "ls-files", "CMakeLists.txt", "src", "bench"], cwd=ROOT,
        check=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()
    digest = hashlib.sha256()
    for relative in sorted(files):
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_operator_report(report: dict[str, Any]) -> dict[str, Any]:
    cells = [(row.get("rows"), row.get("columns"), row.get("exact_decode_calls"))
             for row in report.get("shapes", [])]
    expected = [(4096, 5120, 48), (5120, 6144, 64), (5120, 17408, 64),
                (7168, 5120, 32), (12288, 5120, 48), (34816, 5120, 64),
                (248320, 5120, 1)]
    if (report.get("status") != "passed" or
            report.get("correctness", {}).get("passed") is not True or
            report.get("static_and_resources", {}).get("runtime_resources_passed") is not True or
            report.get("static_and_resources", {}).get("required_opcode") !=
                "v_dot8_i32_iu4" or
            report.get("static_and_resources", {}).get("required_opcode_count") != 16 or
            report.get("static_and_resources", {}).get("wmma_forbidden") is not True or
            report.get("performance_decision", {}).get("accepted") is not True or
            report.get("performance_decision", {}).get("every_cell_passed") is not True or
            report.get("performance_decision", {}).get("aggregate_passed") is not True or
            cells != expected):
        raise ValueError("operator report does not bind the exact accepted seven-cell decision")
    return report


def bind_operator_report() -> dict[str, Any]:
    path = OPERATOR_REPORT.resolve(strict=True)
    if _sha256(path) != OPERATOR_REPORT_SHA256:
        raise ValueError("qualified seven-cell operator report identity differs")
    report = validate_operator_report(json.loads(path.read_text(encoding="utf-8")))
    cells = [(row["rows"], row["columns"], row["exact_decode_calls"])
             for row in report["shapes"]]
    return {"path": str(path), "sha256": OPERATOR_REPORT_SHA256,
            "status": "passed", "cells": cells,
            "weighted_saving_lower_ms_per_token": report["performance_decision"][
                "exact_call_weighted_saving_lower_ms_per_token"]}


def configure_and_build(build: Path, enabled: bool) -> None:
    if build.exists():
        raise ValueError(f"build directory must be fresh: {build}")
    command = [
        "cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DNINFER_BUILD_APPS=OFF", "-DNINFER_BUILD_BENCHMARKS=ON",
        "-DNINFER_BUILD_R9700_CORE_QUALIFIER=OFF",
        f"-D{DEFINE}={'ON' if enabled else 'OFF'}",
    ]
    _run(command)
    _run(["cmake", "--build", str(build), "--target", "ninfer_bench", "-j", "2"])


def cache_identity(build: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, _ = key_type.split(":", 1)
        if key in ("CMAKE_BUILD_TYPE", "CMAKE_HIP_COMPILER", "NINFER_BUILD_APPS",
                   "NINFER_BUILD_BENCHMARKS", "NINFER_BUILD_R9700_CORE_QUALIFIER",
                   "NINFER_R9700_XATTENTION_QUALIFICATION",
                   "NINFER_R9700_A8Q4_T1_DOT8_QUALIFICATION"):
            values[key] = value
    required = {"CMAKE_BUILD_TYPE", "CMAKE_HIP_COMPILER", "NINFER_BUILD_APPS",
                "NINFER_BUILD_BENCHMARKS", "NINFER_BUILD_R9700_CORE_QUALIFIER",
                "NINFER_R9700_XATTENTION_QUALIFICATION",
                "NINFER_R9700_A8Q4_T1_DOT8_QUALIFICATION"}
    if set(values) != required:
        raise ValueError("CMake cache lacks the exact whole A/B identity fields")
    return values


def normalized_compile_commands(build: Path) -> list[dict[str, str]]:
    rows = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    normalized = []
    for row in rows:
        item = {key: str(value).replace(str(build), "<BUILD>")
                for key, value in row.items() if key in ("file", "command", "output")}
        command = item.get("command", "")
        command = command.replace(f"-D{DEFINE}=1 ", "").replace(f"-D{DEFINE}=1", "")
        item["command"] = " ".join(command.split())
        normalized.append(item)
    return sorted(normalized, key=lambda item: (item.get("file", ""), item.get("output", "")))


def robust_bounds(samples: list[float]) -> tuple[float, float, float]:
    center = statistics.median(samples)
    uncertainty = FACTOR * statistics.median(abs(value - center) for value in samples)
    return center, center - uncertainty, center + uncertainty


def extract(report: dict[str, Any], decode_tokens: int) -> dict[str, Any]:
    config = report.get("config", {})
    memory = report.get("memory", {})
    tests = report.get("tests", [])
    environment = report.get("environment", {})
    if (report.get("schema_version") != 20 or
            report.get("artifact_type") != "ninfer_bench_report" or len(tests) != 1 or
            environment.get("gpu_name") != "AMD Radeon AI PRO R9700" or
            environment.get("architecture_name") != "gfx1201" or
            environment.get("device_id") != 0 or
            not all(isinstance(environment.get(key), str) and environment[key]
                    for key in ("hip_runtime_version", "hip_driver_version")) or
            report.get("artifact", {}).get("file_size_bytes") != ARTIFACT_SIZE or
            Path(report.get("artifact", {}).get("path", "")).resolve() !=
                ARTIFACT.resolve(strict=True) or
            config.get("concurrency") != 1 or config.get("spec") != "none" or
            config.get("draft_tokens") != 0 or config.get("use_device_graph") is not True or
            config.get("retain_token_ids") is not True or config.get("prefill_chunk") != 4096 or
            config.get("max_context") != 8192 + decode_tokens or
            config.get("kv_cache_format") != "fp8-k-int4-v" or
            config.get("kv_value_group") != 16 or
            config.get("kv_plane_layouts") != {
                "key": "token-fastest-head-major",
                "value": "feature-fastest-page-major",
                "value_scale": "feature-fastest-page-major"} or
            memory.get("kv_capacity_mode") != "explicit" or
            config.get("q4_activation_bits") != 8 or
            config.get("q4_prefill_cta_profile") !=
                "m64n128-pingpong-n16-k16-scalar-base-production" or
            config.get("w8_activation_bits") != 8 or
            config.get("xattention_qualification") is not False or
            config.get("fp8_qk_wmma_enabled") is not True or
            config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1" or
            config.get("fp8_qk_wmma_t1_min_context") != 64 or
            config.get("fp8_qk_wmma_t2_min_context") != 320 or
            config.get("speculative_execution") is not False or
            config.get("dflash_verify_width_requested") != 0 or
            config.get("dflash_verify_width") != 0 or
            config.get("proposal_head") != "full" or
            config.get("decode_path") != "device_graph" or
            config.get("repetitions") != 1 or config.get("warmup") != 1 or
            config.get("corpus_tokens") != 65536 or
            Path(config.get("corpus_path", "")).resolve() != CORPUS.resolve(strict=True)):
        raise ValueError("benchmark report does not describe the fixed C1 ordinary graph case")
    test = tests[0]
    reps = test.get("reps", [])
    if (test.get("kind") != "whole" or test.get("n_prompt") != 8192 or
            test.get("n_gen") != decode_tokens or len(reps) != 1 or
            reps[0].get("generated_output_tokens") != decode_tokens + 1 or
            reps[0].get("decode_output_tokens") != decode_tokens):
        raise ValueError("benchmark report does not contain the requested exact P8192 decode evidence")
    token_ids = reps[0].get("generated_token_ids_by_lane")
    if (not isinstance(token_ids, list) or len(token_ids) != 1 or
            not isinstance(token_ids[0], list) or len(token_ids[0]) != decode_tokens + 1):
        raise ValueError("C1 benchmark report lacks one retained generated-token lane")
    token_digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
    return {
        "prefill_seconds": float(test["prefill_seconds_mean"]),
        "decode_seconds": float(test["decode_seconds_mean"]),
        "token_ids": token_ids,
        "generated_token_digest": token_digest,
        "generated_token_count": len(token_ids[0]),
        "environment": environment,
        "artifact": report["artifact"],
        "workspace": memory["workspace"],
        "config": config,
    }


def decision(control: list[dict[str, Any]], candidate: list[dict[str, Any]], *,
             pairs: int = 4, decode_tokens: int = 256) -> dict[str, Any]:
    if len(control) != 2 * pairs or len(candidate) != 2 * pairs:
        raise ValueError("whole A/B run count differs from its AB/BA pair count")
    reference = control[0]
    for run in control + candidate:
        if any(run[field] != reference[field] for field in
               ("token_ids", "environment", "artifact", "workspace", "config")):
            raise ValueError("whole A/B token, environment, artifact, workspace, or config parity failed")
    # Each adjacent pair is one order-balanced AB/BA repetition.
    control_decode = [(control[i]["decode_seconds"] + control[i + 1]["decode_seconds"]) / 2
                      for i in range(0, 2 * pairs, 2)]
    candidate_decode = [(candidate[i]["decode_seconds"] + candidate[i + 1]["decode_seconds"]) / 2
                        for i in range(0, 2 * pairs, 2)]
    control_prefill = [(control[i]["prefill_seconds"] + control[i + 1]["prefill_seconds"]) / 2
                       for i in range(0, 2 * pairs, 2)]
    candidate_prefill = [(candidate[i]["prefill_seconds"] + candidate[i + 1]["prefill_seconds"]) / 2
                         for i in range(0, 2 * pairs, 2)]
    cd_median, cd_lower, _ = robust_bounds(control_decode)
    nd_median, _, nd_upper = robust_bounds(candidate_decode)
    cp_median, cp_lower, _ = robust_bounds(control_prefill)
    np_median, _, np_upper = robust_bounds(candidate_prefill)
    decode_upper_ratio = nd_upper / cd_lower if cd_lower > 0 else math.inf
    prefill_upper_ratio = np_upper / cp_lower if cp_lower > 0 else math.inf
    saving_lower_ms_per_token = (cd_lower - nd_upper) * 1000.0 / decode_tokens
    performance_pass = (decode_upper_ratio <= DECODE_RATIO_LIMIT and
                saving_lower_ms_per_token >= DECODE_SAVING_FLOOR_MS_PER_TOKEN and
                prefill_upper_ratio <= PREFILL_RATIO_LIMIT)
    promotion_eligible = pairs == 4 and decode_tokens == 256
    accepted = promotion_eligible and performance_pass
    return {
        "uncertainty": f"4.4478*MAD over {pairs} order-balanced AB/BA pairs",
        "control_decode_seconds": control_decode,
        "candidate_decode_seconds": candidate_decode,
        "control_prefill_seconds": control_prefill,
        "candidate_prefill_seconds": candidate_prefill,
        "control_decode_median_seconds": cd_median,
        "candidate_decode_median_seconds": nd_median,
        "candidate_over_control_decode_robust_upper": decode_upper_ratio,
        "decode_robust_upper_limit": DECODE_RATIO_LIMIT,
        "decode_saving_lower_ms_per_token": saving_lower_ms_per_token,
        "decode_saving_required_ms_per_token": DECODE_SAVING_FLOOR_MS_PER_TOKEN,
        "control_prefill_median_seconds": cp_median,
        "candidate_prefill_median_seconds": np_median,
        "candidate_over_control_prefill_robust_upper": prefill_upper_ratio,
        "prefill_robust_upper_limit": PREFILL_RATIO_LIMIT,
        "performance_passed": performance_pass,
        "promotion_eligible_full_gate": promotion_eligible,
        "accepted": accepted,
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write while publishing whole A/B report")
            offset += written
    except Exception:
        os.unlink(path)
        raise
    finally:
        os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--control-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.pairs <= 4:
        raise ValueError("--pairs must be in [1,4]")
    if not 1 <= args.decode_tokens <= 256:
        raise ValueError("--decode-tokens must be in [1,256]")
    artifact = args.artifact.resolve(strict=True)
    corpus = args.corpus.resolve(strict=True)
    if artifact != ARTIFACT.resolve(strict=True) or corpus != CORPUS.resolve(strict=True):
        raise ValueError("whole A/B is pinned to the selected N16K16 artifact and corpus fixture")
    output = args.out.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("whole A/B output must be fresh with an existing parent")
    if POWER.read_text(encoding="utf-8").strip() != "auto":
        raise ValueError("whole A/B requires auto power before building")
    source_before = source_digest()
    artifact_before = _sha256(artifact)
    corpus_before = _sha256(corpus)
    if (artifact.stat().st_size != ARTIFACT_SIZE or artifact_before != ARTIFACT_SHA256 or
            corpus_before != CORPUS_SHA256):
        raise ValueError("pinned artifact or corpus content identity differs")
    operator_gate = bind_operator_report()
    configure_and_build(args.control_build.resolve(), False)
    configure_and_build(args.candidate_build.resolve(), True)
    control_cache = cache_identity(args.control_build.resolve())
    candidate_cache = cache_identity(args.candidate_build.resolve())
    cache_without_option = lambda value: {
        key: item for key, item in value.items()
        if key != "NINFER_R9700_A8Q4_T1_DOT8_QUALIFICATION"}
    if (control_cache["NINFER_R9700_A8Q4_T1_DOT8_QUALIFICATION"] != "OFF" or
            candidate_cache["NINFER_R9700_A8Q4_T1_DOT8_QUALIFICATION"] != "ON" or
            cache_without_option(control_cache) != cache_without_option(candidate_cache)):
        raise ValueError("control/candidate CMake cache differs beyond the private option")
    control_compile = normalized_compile_commands(args.control_build.resolve())
    candidate_compile = normalized_compile_commands(args.candidate_build.resolve())
    if control_compile != candidate_compile:
        raise ValueError("emitted compile commands differ by more than the private dot8 define")
    binaries = {"control": args.control_build.resolve() / "bench/ninfer_bench",
                "candidate": args.candidate_build.resolve() / "bench/ninfer_bench"}
    binary_before = {name: _sha256(path) for name, path in binaries.items()}
    compile_before = {"control": _sha256(args.control_build.resolve() / "compile_commands.json"),
                      "candidate": _sha256(args.candidate_build.resolve() / "compile_commands.json")}
    observations: dict[str, list[dict[str, Any]]] = {"control": [], "candidate": []}
    orders = [("control", "candidate"), ("candidate", "control")] * args.pairs
    ordered_processes: list[dict[str, Any]] = []
    canonical_stderr: str | None = None
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".dot8-whole-ab-") as temporary:
        temporary_path = Path(temporary)
        for index, order in enumerate(orders):
            for position, name in enumerate(order):
                report_path = temporary_path / f"{index:02d}-{name}.json"
                command = [str(binaries[name]), "--weights", str(artifact), "--corpus",
                           str(corpus), "--device", "0", "--concurrency", "1",
                           "--whole-pg", f"8192,{args.decode_tokens}", "--prefill-chunk", "4096",
                           "--spec", "mtp", "--draft-tokens", "0", "--retain-token-ids",
                           "--output", "json", "--output-file", str(report_path),
                           "-r", "1", "--warmup", "1"]
                if POWER.read_text(encoding="utf-8").strip() != "auto":
                    raise ValueError("whole A/B requires auto power before every process")
                stdout, stderr = _run_benchmark(command, report_path)
                if POWER.read_text(encoding="utf-8").strip() != "auto":
                    raise ValueError("whole A/B requires auto power after every process")
                expected_stderr = (
                    f"[ninfer_bench] loading {artifact} (max_context={8192 + args.decode_tokens}, "
                    "concurrency=1, kv_format=fp8-k-int4-v)\n"
                    f"[ninfer_bench] test 1/1 whole-pp8192+tg{args.decode_tokens}: "
                    "warmup=1 reps=1\n")
                if stderr != expected_stderr:
                    raise ValueError("benchmark emitted non-canonical progress stderr")
                if canonical_stderr is None:
                    canonical_stderr = stderr
                elif stderr != canonical_stderr:
                    raise ValueError("benchmark stderr differs from the canonical first process")
                extracted = extract(json.loads(report_path.read_text()), args.decode_tokens)
                observations[name].append(extracted)
                ordered_processes.append({
                    "ordinal": len(ordered_processes), "pair": index // 2,
                    "direction": "AB" if index % 2 == 0 else "BA",
                    "position": position, "role": name,
                    "decode_seconds": extracted["decode_seconds"],
                    "prefill_seconds": extracted["prefill_seconds"],
                    "generated_token_digest": extracted["generated_token_digest"],
                    "generated_token_count": extracted["generated_token_count"],
                    "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                })
    result = decision(observations["control"], observations["candidate"],
                      pairs=args.pairs, decode_tokens=args.decode_tokens)
    if POWER.read_text(encoding="utf-8").strip() != "auto":
        raise ValueError("whole A/B did not retain auto power")
    source_after = source_digest()
    if source_before != source_after:
        raise ValueError("source changed during whole A/B")
    if (artifact_before != _sha256(artifact) or corpus_before != _sha256(corpus) or
            _sha256(OPERATOR_REPORT.resolve(strict=True)) != OPERATOR_REPORT_SHA256):
        raise ValueError("artifact, corpus, or operator authority changed during whole A/B")
    if binary_before != {name: _sha256(path) for name, path in binaries.items()}:
        raise ValueError("control or candidate binary changed during whole A/B")
    if compile_before != {
            "control": _sha256(args.control_build.resolve() / "compile_commands.json"),
            "candidate": _sha256(args.candidate_build.resolve() / "compile_commands.json")}:
        raise ValueError("emitted compile commands changed during whole A/B")
    report = {
        "schema": "ninfer.r9700.a8q4_t1_dot8_whole_ab.v1",
        "artifact_type": "ninfer_r9700_a8q4_t1_dot8_whole_ab",
        "schema_version": 1,
        "status": ("passed" if result["accepted"] else
                   "screened" if not result["promotion_eligible_full_gate"] else "rejected"),
        "temporary_qualification_route": True,
        "removal_required_after_promotion": True,
        "workload": {"concurrency": 1, "prompt_tokens": 8192,
                     "decode_tokens": args.decode_tokens, "spec": "none", "device_graph": True,
                     "balanced_ab_ba_pairs": args.pairs},
        "provenance": {"source_digest": source_before,
                       "artifact_path": str(artifact), "artifact_size": artifact.stat().st_size,
                       "artifact_sha256": artifact_before,
                       "corpus_path": str(corpus), "corpus_sha256": corpus_before,
                       "control_binary": str(binaries["control"]),
                       "control_binary_sha256": _sha256(binaries["control"]),
                       "candidate_binary": str(binaries["candidate"]),
                       "candidate_binary_sha256": _sha256(binaries["candidate"]),
                       "compile_commands_equal_except_define": True,
                       "control_cmake_cache": control_cache,
                       "candidate_cmake_cache": candidate_cache},
        "operator_gate": operator_gate,
        "parity": {"exact_generated_tokens": True, "environment": True,
                   "artifact": True, "workspace": True, "config": True,
                   "generated_token_ids_sha256": hashlib.sha256(json.dumps(
                       observations["control"][0]["token_ids"], separators=(",", ":")
                   ).encode()).hexdigest(),
                   "matched_environment": observations["control"][0]["environment"],
                   "matched_artifact": observations["control"][0]["artifact"],
                   "matched_workspace": observations["control"][0]["workspace"],
                   "matched_config": observations["control"][0]["config"]},
        "routing_audit": {
            "candidate_branch": (
                "a8q4g64_linear_candidate T==1, columns==padded_columns, "
                "and exact seven qualified Text (N,K) shapes"),
            "ordinary_qwen3_8_27b_q4_calls_per_token": 321,
            "off_inventory_t1_fallback": "a8q4g64_linear_wmma32",
            "tail_fallback": "a8q4g64_linear_wmma32",
        },
        "ordered_processes": ordered_processes,
        "canonical_stderr": {"sha256": hashlib.sha256(
            (canonical_stderr or "").encode()).hexdigest(),
            "byte_count": len((canonical_stderr or "").encode()),
            "identical_across_processes": True},
        "decision": result,
    }
    write_exclusive(output, report)
    print(json.dumps({"report": str(output), "status": report["status"]}, sort_keys=True))
    return 0 if result["accepted"] or not result["promotion_eligible_full_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
