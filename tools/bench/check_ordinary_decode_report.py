#!/usr/bin/env python3
"""Validate the exact C1 8K ordinary non-speculative diagnostic matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.run_ninfer_bench_matrix import (
    FP8_QK_WMMA_PROFILE,
    FP8_QK_WMMA_T1_MIN_CONTEXT,
    FP8_QK_WMMA_T2_MIN_CONTEXT,
    MATRIX_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
    R9700_POWER_PROFILE,
    add_repetition_args,
    build_cases,
    file_sha256,
    inspect_artifact,
    inspect_executable,
    load_bench_report,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_power(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def positive_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def require_statistic(test: dict, name: str, values: list[float]) -> None:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stddev = math.sqrt(variance)
    reported_mean = test.get(f"{name}_mean")
    reported_stddev = test.get(f"{name}_stddev")
    require(type(reported_mean) in (int, float) and math.isfinite(reported_mean)
            and type(reported_stddev) in (int, float) and math.isfinite(reported_stddev)
            # The reporter computes aggregates from full-precision event values, then emits
            # both aggregates and raw repetitions with ten significant decimal digits. Rebuilt
            # statistics can therefore differ by the raw-sample serialization error.
            and math.isclose(reported_mean, mean, rel_tol=2e-6, abs_tol=1e-9)
            and math.isclose(reported_stddev, stddev, rel_tol=2e-6, abs_tol=1e-9),
            f"ordinary report {name} aggregate differs from repetitions")


def validate_matrix(
    root: Path,
    expected_executable: Path,
    expected_artifact: Path,
    power_reader: Callable[[Path], str] = read_power,
) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict)
            and manifest.get("artifact_type") == "ninfer_bench_matrix_run"
            and manifest.get("schema_version") == MATRIX_SCHEMA_VERSION
            and manifest.get("preset") == "ordinary-diagnostic"
            and manifest.get("dry_run") is False
            and type(manifest.get("case_count")) is int and manifest["case_count"] == 1
            and type(manifest.get("point_count")) is int and manifest["point_count"] == 1
            and isinstance(manifest.get("concurrency"), list)
            and len(manifest["concurrency"]) == 1
            and type(manifest["concurrency"][0]) is int
            and manifest["concurrency"][0] == 1
            and manifest.get("expected_kv_value_group") == 16
            and manifest.get("expected_kv_plane_layouts") == R9700_KV_PLANE_LAYOUTS
            and manifest.get("expected_q4_activation_bits") == 8
            and manifest.get("expected_w8_activation_bits") == 8
            and manifest.get("expected_fp8_qk_wmma_enabled") is True
            and manifest.get("expected_fp8_qk_wmma_profile") == FP8_QK_WMMA_PROFILE
            and manifest.get("expected_fp8_qk_wmma_t1_min_context") == FP8_QK_WMMA_T1_MIN_CONTEXT
            and manifest.get("expected_fp8_qk_wmma_t2_min_context") == FP8_QK_WMMA_T2_MIN_CONTEXT
            and manifest.get("expected_xattention_profile") == "b128-s16-tau900",
            "ordinary diagnostic manifest contract differs")
    require(not (root / "failures.json").exists(),
            "ordinary diagnostic retains terminal failures")
    require(manifest.get("power_profile") == {
        "required": "auto", "sysfs_path": str(R9700_POWER_PROFILE),
        "observed": "auto", "rechecked_after": "auto",
    }, "ordinary diagnostic power evidence differs")
    try:
        live_power = power_reader(R9700_POWER_PROFILE).strip()
    except OSError as error:
        raise ValueError(f"cannot read live power profile: {error}") from error
    require(live_power == "auto", "live power profile is not auto")

    executable = expected_executable.resolve(strict=True)
    artifact = expected_artifact.resolve(strict=True)
    require(manifest.get("bench") == inspect_executable(executable),
            "benchmark executable identity differs")
    require(manifest.get("artifact") == inspect_artifact(artifact),
            "artifact identity differs")
    corpus = Path(manifest.get("corpus", ""))
    require(corpus.is_absolute() and corpus.is_file()
            and manifest.get("corpus_sha256") == file_sha256(corpus),
            "corpus identity differs")
    commands = manifest.get("commands")
    require(isinstance(commands, list) and len(commands) == 1,
            "ordinary diagnostic command inventory differs")
    command = commands[0]
    case = build_cases("ordinary-diagnostic")[0]
    require(command.get("suite") == case.suite and command.get("case") == case.name
            and type(command.get("concurrency")) is int and command["concurrency"] == 1,
            "ordinary diagnostic command geometry differs")
    report_path = Path(command.get("report", ""))
    require(report_path == root.resolve(strict=True) / "json" / case.suite / "c1" /
            f"{case.name}.json", "ordinary diagnostic report path differs")
    expected_command = add_repetition_args([
        str(executable), "--weights", str(artifact), "--corpus", str(corpus),
        "--device", "0", "--concurrency", "1", *case.args,
        "--output", "json", "--output-file", str(report_path),
    ], case, None, None)
    require(command.get("command") == expected_command,
            "ordinary diagnostic command differs from the fixed protocol")
    before = file_sha256(report_path)
    report = load_bench_report(
        report_path, 16, 8, 8, True, 1, manifest["artifact"], expected_command, case,
        "b128-s16-tau900",
    )
    require(file_sha256(report_path) == before, "ordinary diagnostic report changed during validation")
    config = report.get("config", {})
    require(config.get("spec") == "none"
            and type(config.get("draft_tokens")) is int and config["draft_tokens"] == 0
            and config.get("speculative_execution") is False
            and config.get("use_device_graph") is True
            and config.get("decode_path") == "device_graph",
            "ordinary decode is not explicitly non-speculative")
    require(type(config.get("concurrency")) is int and config["concurrency"] == 1
            and type(config.get("repetitions")) is int and config["repetitions"] == 3
            and type(config.get("warmup")) is int and config["warmup"] == 1
            and type(config.get("prefill_chunk")) is int and config["prefill_chunk"] == 4096,
            "ordinary decode schedule fields differ")
    tests = report.get("tests")
    require(isinstance(tests, list) and len(tests) == 1,
            "ordinary report test inventory differs")
    test = tests[0]
    require(report.get("command") == " ".join(expected_command),
            "ordinary report command differs from the fixed protocol")
    disabled = {
        "enabled": False, "draft_window": 0, "rounds": 0, "drafted_tokens": 0,
        "accepted_tokens": 0, "fallback_steps": 0, "acceptance_rate": None,
        "acceptance_length": None, "accepted_per_position": [],
    }
    require(test.get("kind") == "whole" and test.get("n_prompt") == 8192
            and test.get("n_gen") == 256
            and positive_number(test.get("prefill_tok_s_mean"))
            and positive_number(test.get("decode_output_tok_s_mean"))
            and positive_number(test.get("prefill_seconds_mean"))
            and positive_number(test.get("decode_seconds_mean"))
            and test.get("speculative") == disabled
            and isinstance(test.get("reps"), list) and len(test["reps"]) == 3
            and all(isinstance(rep, dict) and rep.get("speculative") == disabled
                    for rep in test["reps"]),
            "ordinary report lacks separate prefill/decode evidence")
    prepare_times: list[float] = []
    prefill_times: list[float] = []
    decode_times: list[float] = []
    total_times: list[float] = []
    for index, rep in enumerate(test["reps"]):
        require(type(rep.get("generated_output_tokens")) is int
                and rep["generated_output_tokens"] == 257
                and type(rep.get("decode_output_tokens")) is int
                and rep["decode_output_tokens"] == 256
                and type(rep.get("decode_engine_tokens")) is int
                and rep["decode_engine_tokens"] == 256,
                f"ordinary repetition {index} token counts differ")
        timings = rep.get("timings")
        require(isinstance(timings, dict)
                and type(timings.get("prepare_seconds")) in (int, float)
                and math.isfinite(timings["prepare_seconds"])
                and timings["prepare_seconds"] >= 0
                and type(timings.get("vision_seconds")) in (int, float)
                and math.isfinite(timings["vision_seconds"])
                and timings["vision_seconds"] == 0
                and positive_number(timings.get("prefill_seconds"))
                and positive_number(timings.get("decode_seconds"))
                and positive_number(timings.get("total_seconds")),
                f"ordinary repetition {index} timings differ")
        prepare_times.append(float(timings["prepare_seconds"]))
        prefill_times.append(float(timings["prefill_seconds"]))
        decode_times.append(float(timings["decode_seconds"]))
        total_times.append(float(timings["total_seconds"]))
    require_statistic(test, "prepare_seconds", prepare_times)
    require_statistic(test, "prefill_seconds", prefill_times)
    require_statistic(test, "decode_seconds", decode_times)
    require_statistic(test, "total_seconds", total_times)
    require_statistic(test, "prefill_tok_s", [8192.0 / value for value in prefill_times])
    require_statistic(test, "decode_output_tok_s", [256.0 / value for value in decode_times])
    require_statistic(test, "decode_engine_tok_s", [256.0 / value for value in decode_times])
    require_statistic(test, "whole_output_tok_s", [257.0 / value for value in total_times])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    validate_matrix(args.matrix.resolve(strict=True), args.executable, args.artifact)
    print("ordinary C1 8K non-speculative diagnostic passed")


if __name__ == "__main__":
    main()
