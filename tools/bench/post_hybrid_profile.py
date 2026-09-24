#!/usr/bin/env python3
"""Prepare or validate the one bounded post-relocation hybrid P2048 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bench.analyze_whole_profile import analyze
from tools.bench.validate_profile_trace import _parse_database


ROCPROFV3 = Path("/opt/rocm/bin/rocprofv3")
POWER_PROFILE = Path(
    "/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level"
)
WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
OUTPUT_STEM = "post_mlp_fusion_hybrid_p2048"
STAGES = ("post_mixer", "attention", "gdn")
FP8_CALLS = {"post_mixer": 64, "attention": 32, "gdn": 48}
Q4_CALLS = {"post_mixer": 64, "attention": 16, "gdn": 96}
Q4_ACTIVATION_SYMBOLS = {
    "post_mixer": "fused_silu_a8g64_quantize_kernel",
    "attention": "a8g64_quantize_activation_kernel",
    "gdn": "a8g64_quantize_activation_kernel",
}
PROFILE_POWER = "auto"
BENCHMARK_ADMISSION_POWER = "auto"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return {
        "path": str(resolved),
        "file_size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _option(command: Sequence[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"benchmark command must contain one {option}")
    return command[positions[0] + 1]


def _replace(command: list[str], option: str, value: str) -> None:
    _option(command, option)
    command[command.index(option) + 1] = value


def _stage(region: str | None) -> str | None:
    if not region:
        return None
    if region.startswith("ninfer.post-mixer.prefill."):
        return "post_mixer"
    if region.startswith("ninfer.attention.prefill."):
        return "attention"
    if region.startswith("ninfer.gdn.prefill."):
        return "gdn"
    return None


def _sum_ms(rows: list[dict[str, Any]]) -> float:
    return sum(int(row["duration_ns"]) for row in rows) / 1e6


def bucket(dispatches: list[dict[str, Any]], attribution: dict[str, Any]) -> dict[str, Any]:
    staged = {stage: [row for row in dispatches if _stage(row["roctx_region"]) == stage]
              for stage in STAGES}
    fp8_parts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    q4_parts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for stage, rows in staged.items():
        fp8_parts[stage] = {
            "activation": [row for row in rows if "fp8_quantize_activation_kernel" in row["symbol"]],
            "matrix": [row for row in rows if row["symbol"].startswith("Cijk_") and "F8BS" in row["symbol"]],
            "poison": [row for row in rows if "poison_nonfinite_output" in row["symbol"]],
        }
        q4_parts[stage] = {
            "activation": [row for row in rows
                           if Q4_ACTIVATION_SYMBOLS[stage] in row["symbol"]],
            "matrix": [row for row in rows if "a8q4g64_linear_prefill_cta_kernel" in row["symbol"]],
        }
        for part, values in fp8_parts[stage].items():
            if len(values) != FP8_CALLS[stage]:
                raise ValueError(
                    f"{stage} FP8 {part} count {len(values)} != {FP8_CALLS[stage]}"
                )
        for part, values in q4_parts[stage].items():
            if len(values) != Q4_CALLS[stage]:
                raise ValueError(
                    f"{stage} Q4 {part} count {len(values)} != {Q4_CALLS[stage]}"
                )

    operators = attribution["operator_attribution"]
    dense_rows = [row for row in operators if row.get("execution_category") == "base_text_prefill"
                  and row.get("stage") == "attention"
                  and row.get("operator_family") == "dense_attention"]
    recurrence_rows = [row for row in operators
                       if row.get("execution_category") == "base_text_prefill"
                       and row.get("stage") == "gdn"
                       and row.get("operator_family") == "gdn_recurrence"]
    if len(dense_rows) != 1 or len(recurrence_rows) != 1:
        raise ValueError("trace lacks one dense-attention or GDN-recurrence attribution")
    recurrence = recurrence_rows[0]
    if recurrence.get("calls") != 48:
        raise ValueError("GDN recurrence count is not 48")
    dense_qk = [row for row in staged["attention"] if "dense_full_score_qk" in row["symbol"]]
    dense_pv = [row for row in staged["attention"] if "dense_full_score_pv_kernel" in row["symbol"]]
    dense_maximum = [row for row in staged["attention"]
                     if "dense_full_score_maximum_kernel" in row["symbol"]]
    if len(dense_qk) != 16 or len(dense_maximum) != 16 or len(dense_pv) != 16:
        raise ValueError(
            "dense full-score trace lacks 16 QK, maximum, or PV layer dispatches"
        )

    fp8 = []
    q4 = []
    for stage in STAGES:
        fp8_rows = [row for values in fp8_parts[stage].values() for row in values]
        q4_rows = [row for values in q4_parts[stage].values() for row in values]
        fp8.append({
            "stage": stage, "semantic_calls": FP8_CALLS[stage],
            "dispatches": len(fp8_rows), "summed_duration_ms": _sum_ms(fp8_rows),
        })
        q4.append({
            "stage": stage, "semantic_calls": Q4_CALLS[stage],
            "dispatches": len(q4_rows), "summed_duration_ms": _sum_ms(q4_rows),
        })

    base = next(row for row in attribution["kernel_execution_categories"]
                if row["category"] == "base_text_prefill")
    fp8_ms = sum(row["summed_duration_ms"] for row in fp8)
    q4_ms = sum(row["summed_duration_ms"] for row in q4)
    dense_ms = float(dense_rows[0]["summed_duration_ms"])
    recurrence_ms = float(recurrence["summed_duration_ms"])
    other_ms = float(base["independent_summed_duration_ms"]) - fp8_ms - q4_ms - dense_ms - recurrence_ms
    if other_ms < 0.0:
        raise ValueError("decision buckets exceed base Text kernel service")
    return {
        "selected_fp8": {"semantic_calls": 144, "stages": fp8,
                         "summed_duration_ms": fp8_ms},
        "remaining_q4": {"semantic_calls": 176, "stages": q4,
                         "summed_duration_ms": q4_ms},
        "dense_full_score_attention": {
            "semantic_layer_calls": 16,
            "kernel_calls": 48,
            "qk_calls": 16,
            "maximum_calls": 16,
            "pv_calls": 16,
            "summed_duration_ms": dense_ms,
        },
        "gdn_recurrence": {"semantic_calls": 48, "kernel_calls": 48,
                           "summed_duration_ms": recurrence_ms},
        "other_base_text_kernel_service": {"summed_duration_ms": other_ms},
        "kernel_inactive_text_prefill_wall": {
            "summed_duration_ms": float(attribution["prefill_no_kernel_wall_ms"])
        },
    }


def prepare(source_report: Path, out_dir: Path) -> dict[str, Any]:
    source_report = source_report.resolve(strict=True)
    out_dir = out_dir.resolve()
    if os.path.lexists(out_dir):
        raise ValueError(f"refusing to reuse output directory: {out_dir}")
    report = _load(source_report, "source hybrid report")
    tests = report.get("tests")
    config = report.get("config", {})
    if (report.get("artifact_type") != "ninfer_bench_report"
            or report.get("schema_version") != 20
            or report.get("load", {}).get("weights_id") != WEIGHTS_ID
            or config.get("concurrency") != 1 or config.get("prefill_chunk") != 4096
            or config.get("kv_value_group") != 16 or config.get("spec") != "none"
            or config.get("draft_tokens") != 0 or config.get("xattention_qualification") is not False
            or config.get("repetitions") != 3 or config.get("warmup") != 1
            or not isinstance(tests, list) or len(tests) != 1
            or (tests[0].get("kind"), tests[0].get("n_prompt"), tests[0].get("n_gen"))
            != ("pp", 2048, 0)):
        raise ValueError("source is not the exact shared-workspace hybrid C1/P2048 report")
    command_text = report.get("command")
    if not isinstance(command_text, str):
        raise ValueError("source report lacks its benchmark command")
    source_command = shlex.split(command_text)
    executable = Path(source_command[0]).resolve(strict=True)
    artifact = Path(_option(source_command, "--weights")).resolve(strict=True)
    corpus = Path(_option(source_command, "--corpus")).resolve(strict=True)
    if (artifact.stat().st_size != report.get("artifact", {}).get("file_size_bytes")
            or Path(str(report["artifact"].get("path", ""))).resolve() != artifact
            or Path(str(config.get("corpus_path", ""))).resolve() != corpus
            or _option(source_command, "--concurrency") != "1"
            or _option(source_command, "-p") != "2048"
            or _option(source_command, "--prefill-chunk") != "4096"
            or _option(source_command, "--draft-tokens") != "0"
            or _option(source_command, "-r") != "3"
            or _option(source_command, "--warmup") != "1"
            or "--profile-measured" in source_command):
        raise ValueError("source command differs from the exact report contract")

    out_dir.mkdir(parents=True)
    trace_dir = out_dir / "rocprof-selected"
    benchmark_report = out_dir / "benchmark-report.json"
    database = trace_dir / f"{OUTPUT_STEM}_results.db"
    before = out_dir / "power-profile-before.txt"
    after = out_dir / "power-profile-after.txt"
    attribution = out_dir / "post-hybrid-buckets.json"
    command = list(source_command)
    command[0] = str(executable)
    _replace(command, "--weights", str(artifact))
    _replace(command, "--corpus", str(corpus))
    _replace(command, "--output-file", str(benchmark_report))
    _replace(command, "-r", "1")
    command.append("--profile-measured")
    profiler = [
        str(ROCPROFV3.resolve(strict=True)), "--selected-regions", "-f", "rocpd",
        "-d", str(trace_dir), "-o", OUTPUT_STEM, "--marker-trace", "--kernel-trace",
        "--memory-copy-trace", "--stats", "--summary", "--", *command,
    ]
    tool = Path(__file__).resolve(strict=True)
    plan_path = out_dir / "plan.json"
    plan = {
        "schema": "ninfer.r9700.post_hybrid_trace_plan.v1",
        "status": "command_only_not_executed",
        "source_report": _snapshot(source_report, "source report"),
        "benchmark_executable": _snapshot(executable, "benchmark executable"),
        "artifact": {**_snapshot(artifact, "artifact"), "weights_id": WEIGHTS_ID},
        "corpus": {**_snapshot(corpus, "corpus"), "tokens": config.get("corpus_tokens")},
        "rocprofv3": _snapshot(ROCPROFV3, "rocprofv3"),
        "tool": _snapshot(tool, "post-hybrid profile tool"),
        "workload": {"kind": "pp", "prompt_tokens": 2048, "generated_tokens": 0,
                     "concurrency": 1, "prefill_chunk": 4096, "kv_value_group": 16,
                     "xattention_profile": "dense", "spec": "none"},
        "expected_inventory": {"selected_fp8_calls": 144, "remaining_q4_calls": 176,
                               "fused_mlp_down_calls": 64,
                               "dense_layer_calls": 16, "gdn_recurrence_calls": 48},
        "timing_contract": {
            "source_benchmark_admission_power_profile": BENCHMARK_ADMISSION_POWER,
            "profile_capture_power_profile": PROFILE_POWER,
            "separate_pmc_capture_power_profile": "profile_standard",
            "profile_timing_admissible": False,
            "purpose": "selected-region attribution only",
        },
        "required_power_profile": {"path": str(POWER_PROFILE), "value": PROFILE_POWER,
                                   "before": str(before), "after": str(after)},
        "outputs": {"trace_directory": str(trace_dir), "database": str(database),
                    "benchmark_report": str(benchmark_report),
                    "attribution": str(attribution)},
        "benchmark_command": command,
        "profiler_command": profiler,
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    quoted_outputs = " ".join(shlex.quote(str(path)) for path in
                              (trace_dir, benchmark_report, before, after, attribution))
    validate_command = [str(Path(sys.executable).resolve()), str(tool), "validate",
                        "--plan", str(plan_path), "--database", str(database),
                        "--benchmark-report", str(benchmark_report), "--power-before", str(before),
                        "--power-after", str(after), "--out", str(attribution)]
    preflight_command = [str(Path(sys.executable).resolve()), str(tool), "preflight",
                         "--plan", str(plan_path)]
    script = (
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + shlex.join(preflight_command) + "\n"
        f"for output in {quoted_outputs}; do test ! -e \"$output\"; done\n"
        f"POWER_PROFILE_VALUE=\"$(cat {shlex.quote(str(POWER_PROFILE))})\"\n"
        f"test \"$POWER_PROFILE_VALUE\" = {PROFILE_POWER}\n"
        f"(set -C; printf '%s\\n' \"$POWER_PROFILE_VALUE\" > {shlex.quote(str(before))})\n"
        "set +e\n" + shlex.join(profiler) + "\nPROFILE_RC=$?\nset -e\n"
        f"POWER_PROFILE_VALUE=\"$(cat {shlex.quote(str(POWER_PROFILE))})\"\n"
        f"(set -C; printf '%s\\n' \"$POWER_PROFILE_VALUE\" > {shlex.quote(str(after))})\n"
        "if (( PROFILE_RC != 0 )); then exit \"$PROFILE_RC\"; fi\n"
        f"test \"$POWER_PROFILE_VALUE\" = {PROFILE_POWER}\n"
        f"test -f {shlex.quote(str(database))}\n"
        + shlex.join(validate_command) + "\n"
    )
    (out_dir / "commands.sh").write_text(script, encoding="utf-8")
    return plan


def _matches(snapshot: dict[str, Any], label: str) -> None:
    actual = _snapshot(Path(snapshot["path"]), label)
    if actual != {key: snapshot[key] for key in actual}:
        raise ValueError(f"{label} identity differs from plan")


def preflight(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = _load(plan_path, "plan")
    if (plan.get("schema") != "ninfer.r9700.post_hybrid_trace_plan.v1"
            or plan.get("status") != "command_only_not_executed"
            or plan.get("expected_inventory") != {
                "selected_fp8_calls": 144, "remaining_q4_calls": 176,
                "fused_mlp_down_calls": 64,
                "dense_layer_calls": 16, "gdn_recurrence_calls": 48,
            }):
        raise ValueError("not the exact post-hybrid trace plan")
    for key in ("source_report", "benchmark_executable", "artifact", "corpus",
                "rocprofv3", "tool"):
        _matches(plan[key], key)
    if plan["artifact"].get("weights_id") != WEIGHTS_ID:
        raise ValueError("planned artifact has the wrong weights identity")
    expected_workload = {"kind": "pp", "prompt_tokens": 2048, "generated_tokens": 0,
                         "concurrency": 1, "prefill_chunk": 4096, "kv_value_group": 16,
                         "xattention_profile": "dense", "spec": "none"}
    expected_timing_contract = {
        "source_benchmark_admission_power_profile": BENCHMARK_ADMISSION_POWER,
        "profile_capture_power_profile": PROFILE_POWER,
        "separate_pmc_capture_power_profile": "profile_standard",
        "profile_timing_admissible": False,
        "purpose": "selected-region attribution only",
    }
    if plan.get("workload") != expected_workload:
        raise ValueError("planned workload differs from the bounded P2048 trace")
    if plan.get("timing_contract") != expected_timing_contract:
        raise ValueError("planned attribution/admission timing contract differs")
    outputs = plan.get("outputs", {})
    out_dir = plan_path.parent
    expected_outputs = {
        "trace_directory": str(out_dir / "rocprof-selected"),
        "database": str(out_dir / "rocprof-selected" / f"{OUTPUT_STEM}_results.db"),
        "benchmark_report": str(out_dir / "benchmark-report.json"),
        "attribution": str(out_dir / "post-hybrid-buckets.json"),
    }
    power = plan.get("required_power_profile", {})
    if (outputs != expected_outputs or power != {
            "path": str(POWER_PROFILE), "value": PROFILE_POWER,
            "before": str(out_dir / "power-profile-before.txt"),
            "after": str(out_dir / "power-profile-after.txt"),
    }):
        raise ValueError("planned output or power paths differ from the fresh trace directory")
    command = plan.get("benchmark_command")
    profiler = plan.get("profiler_command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("planned benchmark command is invalid")
    if (command[0] != plan["benchmark_executable"]["path"]
            or Path(_option(command, "--weights")).resolve()
            != Path(plan["artifact"]["path"])
            or Path(_option(command, "--corpus")).resolve() != Path(plan["corpus"]["path"])
            or Path(_option(command, "--output-file")).resolve()
            != Path(outputs["benchmark_report"])
            or _option(command, "--concurrency") != "1" or _option(command, "-p") != "2048"
            or _option(command, "--prefill-chunk") != "4096"
            or _option(command, "--draft-tokens") != "0" or _option(command, "-r") != "1"
            or _option(command, "--warmup") != "1" or command.count("--profile-measured") != 1):
        raise ValueError("planned benchmark command differs from the bounded workload")
    expected_profiler = [
        plan["rocprofv3"]["path"], "--selected-regions", "-f", "rocpd", "-d",
        outputs["trace_directory"], "-o", OUTPUT_STEM, "--marker-trace", "--kernel-trace",
        "--memory-copy-trace", "--stats", "--summary", "--", *command,
    ]
    if profiler != expected_profiler:
        raise ValueError("profiler command differs from the exact selected-region trace")
    fresh_paths = [Path(value) for key, value in outputs.items() if key != "database"]
    fresh_paths += [Path(power["before"]), Path(power["after"])]
    existing = [str(path) for path in fresh_paths if os.path.lexists(path)]
    if existing:
        raise ValueError(f"refusing to reuse trace output: {existing[0]}")
    return plan


def validate(plan_path: Path, database: Path, benchmark_report: Path,
             power_before: Path, power_after: Path) -> dict[str, Any]:
    plan = _load(plan_path, "plan")
    if plan.get("schema") != "ninfer.r9700.post_hybrid_trace_plan.v1":
        raise ValueError("not a post-hybrid trace plan")
    for key in ("source_report", "benchmark_executable", "artifact", "corpus",
                "rocprofv3", "tool"):
        _matches(plan[key], key)
    outputs = plan["outputs"]
    actual_paths = {"database": database, "benchmark_report": benchmark_report,
                    "before": power_before, "after": power_after}
    planned_paths = {"database": outputs["database"],
                     "benchmark_report": outputs["benchmark_report"],
                     "before": plan["required_power_profile"]["before"],
                     "after": plan["required_power_profile"]["after"]}
    if any(path.resolve() != Path(planned_paths[key]).resolve()
           for key, path in actual_paths.items()):
        raise ValueError("validation paths differ from plan")
    if (power_before.read_text(encoding="utf-8").strip() != PROFILE_POWER
            or power_after.read_text(encoding="utf-8").strip() != PROFILE_POWER):
        raise ValueError("dispatch-trace power profile endpoints are not auto")
    report = _load(benchmark_report, "benchmark report")
    command = plan["benchmark_command"]
    tests = report.get("tests")
    config = report.get("config", {})
    if (report.get("artifact_type") != "ninfer_bench_report"
            or report.get("schema_version") != 20 or report.get("command") != " ".join(command)
            or report.get("load", {}).get("weights_id") != WEIGHTS_ID
            or config.get("concurrency") != 1 or config.get("prefill_chunk") != 4096
            or config.get("kv_value_group") != 16 or config.get("spec") != "none"
            or config.get("draft_tokens") != 0 or config.get("xattention_qualification") is not False
            or config.get("repetitions") != 1 or config.get("warmup") != 1
            or not isinstance(tests, list) or len(tests) != 1
            or (tests[0].get("kind"), tests[0].get("n_prompt"), tests[0].get("n_gen"))
            != ("pp", 2048, 0)):
        raise ValueError("captured report differs from planned hybrid workload")
    dispatches, aggregates = _parse_database(database, " ".join(command))
    attribution = analyze(database, benchmark_report)
    buckets = bucket(dispatches, attribution)
    return {
        "schema": "ninfer.r9700.post_hybrid_trace_attribution.v1",
        "status": "valid_attribution_only",
        "profile_timing_admissible": False,
        "timing_contract": plan["timing_contract"],
        "power_profile": {"required": PROFILE_POWER,
                          "before": PROFILE_POWER, "after": PROFILE_POWER},
        "plan": _snapshot(plan_path, "plan"),
        "benchmark_report": _snapshot(benchmark_report, "benchmark report"),
        "database": _snapshot(database, "database"),
        "workload": plan["workload"],
        "aggregates": aggregates,
        "decision_buckets": buckets,
        "analyzer": attribution,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-report", required=True, type=Path)
    prepare_parser.add_argument("--out-dir", required=True, type=Path)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--plan", required=True, type=Path)
    validate_parser = subparsers.add_parser("validate")
    for name in ("plan", "database", "benchmark-report", "power-before", "power-after", "out"):
        validate_parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.source_report, args.out_dir)
            print(shlex.join(result["profiler_command"]))
        elif args.action == "preflight":
            preflight(args.plan)
            print("post-hybrid trace plan preflight passed")
        else:
            if os.path.lexists(args.out):
                raise ValueError(f"refusing to overwrite output: {args.out}")
            result = validate(args.plan, args.database, args.benchmark_report,
                              args.power_before, args.power_after)
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, sqlite3.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
