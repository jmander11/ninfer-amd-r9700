#!/usr/bin/env python3
"""Validate the companion and run the matched fast C1 whole-DFlash mechanism A/B."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

REPO = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PLAN_PATH = PACKAGE / "plan.json"
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
PROFILE_ENV = {
    "HSA_TOOLS_LIB", "ROCPROFILER_TOOL_LIBRARIES", "ROCP_TOOL_LIBRARIES",
    "ROCPROFILER_OUTPUT_PATH", "ROCPROFILER_OUTPUT_FILE_NAME",
}
PROMPT_TOKENS = 128
DECODE_STEPS = 64
REQUESTED_OUTPUT_TOKENS = 65


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        fail(f"{label} is not a valid finite value")
    return result


def close(actual: object, expected: float, label: str) -> None:
    if not math.isclose(finite(actual, label), expected, rel_tol=1e-9, abs_tol=1e-12):
        fail(f"{label} is inconsistent")


def inspect(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        fail(f"required file is not regular: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha(resolved)}


def require_power() -> str:
    try:
        value = POWER.read_text(encoding="utf-8").strip()
    except OSError as error:
        fail(f"cannot read R9700 power profile: {error}")
    if value != "auto":
        fail(f"R9700 power profile must be auto, observed {value!r}")
    return value


def validate_plan() -> dict:
    plan = load(PLAN_PATH)
    if (plan.get("artifact_type") != "ninfer_r9700_dflash_small_t_whole_ab_plan" or
            plan.get("schema_version") != 2 or
            plan.get("production_routing_authorized") is not False or
            plan.get("production_recipe_selected") is not False):
        fail("A/B plan would make a production claim")
    work = plan.get("workload", {})
    if (work.get("concurrency") != [1] or work.get("maximum_product_concurrency") != 4 or
            work.get("profiles") != [
                {"draft_tokens": 4, "verify_width": 5},
                {"draft_tokens": 5, "verify_width": 6},
            ] or work.get("prompt_tokens") != PROMPT_TOKENS or
            work.get("decode_steps") != DECODE_STEPS or
            work.get("requested_output_tokens") != REQUESTED_OUTPUT_TOKENS or
            work.get("prefill_chunk") != 4096 or
            work.get("repetitions_per_process") != 3 or
            work.get("warmup_per_process") != 1 or
            work.get("launch_orders") != {
                "K4_W5": ["control", "candidate", "candidate", "control"],
                "K5_W6": ["candidate", "control", "control", "candidate"],
            }):
        fail("A/B workload is not the exact bounded C1 K4/W5 and K5/W6 contract")
    failed = plan.get("failed_attempt_provenance", {})
    expected_failure = (
        "Device Graph preparation consumed 71303168 bytes, exceeding the planned allowance "
        "of 69206016 bytes"
    )
    if (failed.get("status") != "retained_functional_failure_before_matched_ab" or
            failed.get("source_commit") !=
            "ca2b58c5728de0fc52916796a2ac5ef6cee5e5f9" or
            failed.get("failure") != expected_failure or
            failed.get("timing_or_routing_conclusion") != "none" or
            plan.get("later_required_evidence") !=
            "selected-recipe matched 8K/32K C1..4 capacity, acceptance, decode, and fresh-prompt whole inference"):
        fail("A/B plan does not retain the exact failed-attempt boundary")
    archive = Path(failed.get("archived_directory", ""))
    if not archive.is_dir():
        fail("failed-attempt archive is absent")
    for label in ("failing_process", "stderr"):
        identity = failed.get(label)
        if not isinstance(identity, dict) or identity != inspect(Path(identity.get("path", ""))):
            fail(f"failed-attempt {label} identity changed")
    return plan


def validate_companion(plan: dict) -> dict:
    sys.path.insert(0, str(REPO))
    from tools.artifact.container import Artifact
    from tools.convert.qwen3_8_27b_r9700.convert_dflash2_q4 import preflight, preflight_summary

    prereq = plan["prerequisites"]
    conversion_plan_path = Path(prereq["conversion_plan"])
    if sha(conversion_plan_path) != prereq["conversion_plan_sha256"]:
        fail("DFlash conversion plan identity changed")
    conversion_plan = load(conversion_plan_path)
    base = Path(prereq["base_artifact"])
    source = Path(prereq["dflash_source"])
    artifact = Path(prereq["companion_artifact"])
    report_path = Path(prereq["conversion_report"])
    pending = Path(str(artifact) + ".conversion.pending.json")
    if pending.exists() or pending.is_symlink():
        fail("DFlash conversion retains a pending receipt")
    checked = preflight(base, source)
    checked_summary = preflight_summary(checked)
    expected = plan["expected_artifact"]
    if (checked.output_identity.model_id != expected["model_id"] or
            checked.output_identity.weights_id != expected["weights_id"] or
            checked.projected_file_bytes != expected["bytes"] or
            len(checked.objects) != expected["objects"] or
            checked_summary.get("combined_plan", {}).get("sha256") !=
            expected["combined_object_plan_sha256"]):
        fail("live DFlash conversion plan differs from the A/B artifact contract")
    artifact_identity = inspect(artifact)
    if artifact_identity["bytes"] != expected["bytes"]:
        fail("DFlash companion size differs from its exact projection")
    if artifact_identity["sha256"] != expected["sha256"]:
        fail("DFlash companion SHA-256 differs from the converted authority")
    with Artifact.open(artifact) as opened:
        if opened.identity != checked.output_identity or opened.objects != checked.objects:
            fail("DFlash companion identity or object plan differs from preflight")
    report = load(report_path)
    if sha(report_path) != expected["conversion_report_sha256"]:
        fail("DFlash conversion report SHA-256 differs from the converted authority")
    expected_base = conversion_plan["base"]
    source_record = checked.source
    if (report.get("identity") != {
            "model_id": expected["model_id"], "weights_id": expected["weights_id"]} or
            report.get("status") != "registered-evaluation-only" or
            report.get("weight_recipe_selected") is not False or
            report.get("base", {}).get("sha256") != expected_base["sha256"] or
            report.get("base", {}).get("authority") != expected_base["authority"] or
            report.get("dflash_source") != source_record or
            report.get("dflash_recipe", {}).get("key") != expected["dflash_recipe"] or
            report.get("dflash_recipe", {}).get("selector_codebook_format") != "BF16" or
            report.get("artifact", {}).get("path") != str(artifact) or
            report.get("artifact", {}).get("bytes") != expected["bytes"] or
            report.get("artifact", {}).get("sha256") != artifact_identity["sha256"] or
            report.get("artifact", {}).get("projected_bytes") != expected["bytes"] or
            report.get("converter", {}).get("mode") != "convert"):
        fail("DFlash conversion report does not bind the exact evaluation-only companion")
    return {
        **artifact_identity,
        "model_id": expected["model_id"],
        "weights_id": expected["weights_id"],
        "objects": expected["objects"],
        "conversion_report": inspect(report_path),
        "conversion_plan": inspect(Path(prereq["conversion_plan"])),
        "production_recipe_selected": False,
    }


def validate_builds(plan: dict) -> tuple[dict, dict[str, dict]]:
    receipt_path = Path(plan["prerequisites"]["matched_build_receipt"])
    receipt = load(receipt_path)
    expected = plan["matched_builds"]
    if (receipt.get("artifact_type") != expected["receipt_artifact_type"] or
            receipt.get("schema_version") != expected["receipt_schema_version"] or
            receipt.get("status") != "cpu-build-and-host-checks-passed-no-gpu-execution" or
            receipt.get("source", {}).get("commit") != expected["source_commit"] or
            receipt.get("benchmark_execution") != "not-run" or
            receipt.get("maximum_supported_benchmark_concurrency") != 4):
        fail("matched-build receipt identity or source commit differs")
    builds = receipt.get("builds")
    if not isinstance(builds, dict) or set(builds) != {"control", "candidate"}:
        fail("matched-build receipt lacks the exact control/candidate pair")
    validated: dict[str, dict] = {}
    for role, selector in (("control", False), ("candidate", True)):
        record = builds[role]
        expected_path = Path(expected[f"{role}_executable"]).resolve()
        if Path(record.get("directory", "")).resolve() != expected_path.parents[1]:
            fail(f"matched-build receipt has the wrong {role} build directory")
        executable_record = record.get("benchmark_executable")
        if not isinstance(executable_record, dict):
            fail(f"matched-build receipt lacks {role} executable identity")
        actual = inspect(expected_path)
        if executable_record != actual:
            fail(f"{role} executable differs from matched-build receipt")
        selected = record.get("NINFER_R9700_DFLASH_SMALL_T_CANDIDATE")
        if selected not in (int(selector), selector):
            fail(f"{role} build has the wrong small-T selector")
        for identity_name in ("cmake_cache", "compile_commands"):
            identity = record.get(identity_name)
            if not isinstance(identity, dict) or identity != inspect(Path(identity.get("path", ""))):
                fail(f"{role} {identity_name} differs from matched-build receipt")
        if record.get("cpu_checks") != {
                "ninfer_bench_support_test": "passed",
                "ninfer_r9700_linear_prefill_dispatch_test": "passed"}:
            fail(f"{role} build lacks its focused host checks")
        validated[role] = actual
    if receipt.get("configs_differ_only_in_dflash_small_t_candidate") is not True:
        fail("build receipt does not prove the matched configuration boundary")
    control_normalized = builds["control"].get("normalized_compile_commands")
    candidate_normalized = builds["candidate"].get("normalized_compile_commands")
    if (control_normalized != candidate_normalized or
            not isinstance(control_normalized, dict) or
            set(control_normalized) != {"bytes", "sha256"}):
        fail("normalized control/candidate compile commands are not identical")
    shared = receipt.get("shared_configuration", {})
    if (shared.get("CMAKE_BUILD_TYPE") != "Release" or
            shared.get("CMAKE_HIP_ARCHITECTURES") != "gfx1201" or
            shared.get("NINFER_R9700_KV_VALUE_GROUP") != 16 or
            shared.get("NINFER_R9700_Q4_ACTIVATION_BITS") != 8 or
            shared.get("NINFER_R9700_W8_ACTIVATION_BITS") != 8 or
            shared.get("NINFER_R9700_FP8_QK_WMMA") != 1 or
            shared.get("NINFER_R9700_XATTENTION_QUALIFICATION") is not False):
        fail("matched builds do not carry the required common R9700 execution profile")
    return {"path": str(receipt_path), "sha256": sha(receipt_path), "report": receipt}, validated


def command_for(executable: Path, artifact: Path, corpus: Path, output: Path,
                draft: int, width: int, *, ordinary: bool = False) -> list[str]:
    command = [
        str(executable), "--weights", str(artifact), "--corpus", str(corpus),
        "--device", "0", "--concurrency", "1", "-pg", "128,64",
        "--whole-pg", "128,64", "--prefill-chunk", "4096",
        "--kv-capacity", "workload",
    ]
    if ordinary:
        command += ["--draft-tokens", "0"]
    else:
        command += ["--spec", "dflash", "--draft-tokens", str(draft),
                    "--dflash-verify-width", str(width), "--lm-head-draft"]
    return command + [
        "--retain-token-ids", "--isolate-prompt-decode", "--output", "json",
        "--output-file", str(output), "-r", "3", "--warmup", "1",
    ]


def run_one(command: list[str], stem: Path) -> dict:
    report = Path(command[command.index("--output-file") + 1])
    paths = (report, stem.with_suffix(".stdout.txt"), stem.with_suffix(".stderr.txt"),
             stem.with_suffix(".process.json"))
    if any(path.exists() or path.is_symlink() for path in paths):
        fail(f"refusing to overwrite A/B output for {stem.name}")
    before = require_power()
    environment = os.environ.copy()
    for key in PROFILE_ENV:
        environment.pop(key, None)
    started = time.time_ns()
    process = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True)
    finished = time.time_ns()
    after = require_power()
    stem.with_suffix(".stdout.txt").write_text(process.stdout, encoding="utf-8")
    stem.with_suffix(".stderr.txt").write_text(process.stderr, encoding="utf-8")
    evidence = {
        "command": command, "exit_code": process.returncode,
        "started_unix_ns": started, "finished_unix_ns": finished,
        "power_profile_before": before, "power_profile_after": after,
        "profiling_environment_removed": sorted(PROFILE_ENV),
        "stdout": inspect(stem.with_suffix(".stdout.txt")),
        "stderr": inspect(stem.with_suffix(".stderr.txt")),
    }
    stem.with_suffix(".process.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if process.returncode != 0 or not report.is_file():
        fail(f"benchmark failed; retained process evidence at {stem}.process.json")
    return evidence


def validate_process(stem: Path, command: list[str]) -> dict:
    record_path = stem.with_suffix(".process.json")
    record = load(record_path)
    if (record.get("command") != command or record.get("exit_code") != 0 or
            record.get("power_profile_before") != "auto" or
            record.get("power_profile_after") != "auto" or
            record.get("profiling_environment_removed") != sorted(PROFILE_ENV) or
            not isinstance(record.get("started_unix_ns"), int) or
            not isinstance(record.get("finished_unix_ns"), int) or
            record["finished_unix_ns"] <= record["started_unix_ns"] or
            record.get("stdout") != inspect(stem.with_suffix(".stdout.txt")) or
            record.get("stderr") != inspect(stem.with_suffix(".stderr.txt"))):
        fail(f"benchmark process evidence differs: {record_path}")
    return {"path": str(record_path), "sha256": sha(record_path)}


def validate_spec(spec: object, draft: int, label: str) -> tuple:
    if not isinstance(spec, dict) or spec.get("enabled") is not (draft > 0) or \
            spec.get("draft_window") != draft:
        fail(f"{label} has invalid speculative identity")
    fields = ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps")
    values = []
    for field in fields:
        value = spec.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(f"{label} has invalid speculative {field}")
        values.append(value)
    rounds, drafted, accepted, fallback = values
    positions = spec.get("accepted_per_position")
    if (not isinstance(positions, list) or len(positions) != draft or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in positions) or sum(positions) != accepted or accepted > drafted):
        fail(f"{label} has incoherent accepted-token accounting")
    if draft > 0:
        if rounds <= 0 or drafted <= 0:
            fail(f"{label} did not exercise speculative rounds")
        close(spec.get("acceptance_rate"), accepted / drafted, f"{label} acceptance_rate")
        close(spec.get("acceptance_length"), 1.0 + accepted / rounds,
              f"{label} acceptance_length")
    elif any(values) or any(positions) or spec.get("acceptance_rate") is not None or \
            spec.get("acceptance_length") is not None:
        fail(f"{label} ordinary control has nonzero speculative accounting")
    return (*values, tuple(positions))


def validate_report(path: Path, command: list[str], artifact: dict, role: str,
                    draft: int, width: int, expected_profile: dict) -> dict:
    report = load(path)
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or shlex.split(report.get("command", "")) != command or
            report.get("artifact") != {"path": str(Path(artifact["path"])),
                                      "file_size_bytes": artifact["bytes"]} or
            report.get("load", {}).get("target") != "qwen3_8_27b_r9700" or
            report.get("load", {}).get("weights_id") != artifact["weights_id"]):
        fail(f"benchmark report provenance differs: {path}")
    environment = report.get("environment", {})
    if (environment.get("gpu_name") != "AMD Radeon AI PRO R9700" or
            environment.get("architecture_name") != "gfx1201" or
            environment.get("device_id") != 0):
        fail(f"benchmark did not run on exact GPU0 R9700/gfx1201: {path}")
    config = report.get("config", {})
    ordinary = draft == 0
    expected_config = {
        **expected_profile,
        "kv_plane_layouts": {"key": "token-fastest-head-major",
                             "value": "feature-fastest-page-major",
                             "value_scale": "feature-fastest-page-major"},
        "concurrency": 1, "prefill_chunk": 4096,
        "spec": "none" if ordinary else "dflash",
        "draft_tokens": draft, "speculative_execution": not ordinary,
        "dflash_verify_width_requested": 0 if ordinary else width,
        "dflash_verify_width": 0 if ordinary else width,
        "proposal_head": "full" if ordinary else "optimized",
        "use_device_graph": True, "retain_token_ids": True,
        "isolate_prompt_decode": True,
        "decode_path": "device_graph" if ordinary else "dflash_device_graph",
        "repetitions": 3, "warmup": 1,
    }
    expected_config["dflash_small_t_candidate"] = role == "candidate"
    for key, value in expected_config.items():
        if config.get(key) != value:
            fail(f"benchmark config {key} differs for {path}")
    tests = report.get("tests")
    if not isinstance(tests, list) or [test.get("label") for test in tests] != [
            "pp128+tg64", "whole-pp128+tg64"]:
        fail(f"benchmark lacks exact decode and whole tests: {path}")
    normalized = {}
    for test, kind, report_kind in zip(tests, ("decode", "whole"), ("pp+tg", "whole"),
                                       strict=True):
        if (test.get("n_prompt") != PROMPT_TOKENS or
                test.get("n_gen") != DECODE_STEPS or
                test.get("requested_output_tokens") != REQUESTED_OUTPUT_TOKENS or
                test.get("kind") != report_kind):
            fail(f"benchmark test geometry differs: {path}")
        reps = test.get("reps")
        if not isinstance(reps, list) or len(reps) != 3:
            fail(f"benchmark does not retain three measured samples: {path}")
        samples = []
        tokens = []
        accounting = []
        for index, rep in enumerate(reps):
            if (rep.get("generated_output_tokens") != REQUESTED_OUTPUT_TOKENS or
                    rep.get("decode_output_tokens") != DECODE_STEPS):
                fail(f"benchmark output/spec work count is invalid: {path}")
            lane_tokens = rep.get("generated_token_ids_by_lane")
            if (not isinstance(lane_tokens, list) or len(lane_tokens) != 1 or
                    not isinstance(lane_tokens[0], list) or
                    len(lane_tokens[0]) != REQUESTED_OUTPUT_TOKENS or
                    any(isinstance(token, bool) or not isinstance(token, int) or token < 0
                        for token in lane_tokens[0])):
                fail(f"benchmark retained output tokens are invalid: {path}")
            timing_key = "decode_seconds" if kind == "decode" else "total_seconds"
            samples.append(finite(rep.get("timings", {}).get(timing_key),
                                  f"{path} {kind} sample", positive=True))
            tokens.append(tuple(lane_tokens[0]))
            rep_accounting = validate_spec(rep.get("speculative"), draft,
                                           f"{path} {kind} rep {index}")
            expected_engine_tokens = (DECODE_STEPS if draft == 0 else
                                      rep_accounting[0] + rep_accounting[2] +
                                      rep_accounting[3])
            if rep.get("decode_engine_tokens") != expected_engine_tokens:
                fail(f"benchmark decode_engine_tokens disagrees with spec accounting: {path}")
            accounting.append(rep_accounting)
        if len(set(tokens)) != 1 or len(set(accounting)) != 1:
            fail(f"benchmark is not deterministic within report: {path}")
        aggregate = validate_spec(test.get("speculative"), draft, f"{path} {kind} aggregate")
        fields = list(zip(*accounting, strict=True))
        expected_aggregate = tuple(sum(field) if i < 4 else tuple(
            sum(values) for values in zip(*field, strict=True))
            for i, field in enumerate(fields))
        if aggregate != expected_aggregate:
            fail(f"benchmark aggregate speculative accounting differs: {path}")
        normalized[kind] = {"samples_seconds": samples, "tokens": tokens[0],
                            "speculative": accounting[0]}
    return normalized


def cell_summary(records: list[dict], kind: str) -> dict:
    by_order: dict[str, dict[str, list[float]]] = {
        "control_first": {"control": [], "candidate": []},
        "candidate_first": {"control": [], "candidate": []},
    }
    for record in records:
        by_order[record["order"]][record["role"]].extend(
            record["validated"][kind]["samples_seconds"])
    ratios = []
    order_ratios = {}
    for order, routes in by_order.items():
        if len(routes["control"]) != 3 or len(routes["candidate"]) != 3:
            fail(f"{kind} does not have exact control-first/candidate-first sample balance")
        paired = [candidate / control for candidate, control in zip(
            routes["candidate"], routes["control"], strict=True)]
        ratios.extend(paired)
        order_ratios[order] = statistics.median(paired)
    mean = statistics.mean(ratios)
    sd = statistics.stdev(ratios)
    upper = mean + 2.0 * sd / math.sqrt(len(ratios))
    order_delta = abs(order_ratios["control_first"] - order_ratios["candidate_first"])
    order_wins = {
        order: statistics.median(routes["candidate"]) < statistics.median(routes["control"])
        for order, routes in by_order.items()
    }
    passed = all(order_wins.values()) and upper < 1.0 and order_delta <= 0.02
    return {
        "control_samples_seconds": (by_order["control_first"]["control"] +
                                    by_order["candidate_first"]["control"]),
        "candidate_samples_seconds": (by_order["control_first"]["candidate"] +
                                      by_order["candidate_first"]["candidate"]),
        "paired_ratios": ratios, "paired_ratio_mean": mean,
        "paired_ratio_stddev": sd, "paired_ratio_upper_2se": upper,
        "launch_order_median_ratios": order_ratios,
        "launch_order_ratio_delta": order_delta, "launch_order_wins": order_wins,
        "passed_initial_screen": passed,
    }


def validate_results(plan: dict, artifact: dict, receipt: dict,
                     builds: dict[str, dict], results: Path) -> dict:
    expected_profile = receipt["report"].get("expected_benchmark_profile")
    if not isinstance(expected_profile, dict):
        fail("matched-build receipt lacks expected benchmark profile")
    corpus = Path(plan["workload"]["corpus"])
    if sha(corpus) != plan["workload"]["corpus_sha256"]:
        fail("benchmark corpus identity changed")
    ordinary_path = results / "ordinary-control.json"
    ordinary_command = command_for(Path(builds["control"]["path"]), Path(artifact["path"]),
                                   corpus, ordinary_path, 0, 0, ordinary=True)
    ordinary_process = validate_process(results / "ordinary-control", ordinary_command)
    ordinary = validate_report(ordinary_path, ordinary_command, artifact, "control", 0, 0,
                               expected_profile)
    cells = []
    launch_orders = plan["workload"]["launch_orders"]
    for draft, width in ((4, 5), (5, 6)):
        label = f"K{draft}_W{width}"
        roles = launch_orders[label]
        records = []
        for index, role in enumerate(roles):
            pair = roles[(index // 2) * 2:(index // 2) * 2 + 2]
            if pair == ["control", "candidate"]:
                order = "control_first"
            elif pair == ["candidate", "control"]:
                order = "candidate_first"
            else:
                fail(f"{label} launch-order pair is not matched")
            report_path = results / f"{label.lower()}-{index + 1}-{role}.json"
            command = command_for(Path(builds[role]["path"]), Path(artifact["path"]), corpus,
                                  report_path, draft, width)
            process = validate_process(results / f"{label.lower()}-{index + 1}-{role}", command)
            validated = validate_report(report_path, command, artifact, role, draft, width,
                                        expected_profile)
            records.append({"role": role, "order": order, "report": inspect(report_path),
                            "process": process, "validated": validated})
        for kind in ("decode", "whole"):
            token_sets = {record["validated"][kind]["tokens"] for record in records}
            token_sets.add(ordinary[kind]["tokens"])
            if len(token_sets) != 1:
                fail(f"{label} {kind} output differs from ordinary or between A/B routes")
            spec_sets = {record["validated"][kind]["speculative"] for record in records}
            if len(spec_sets) != 1:
                fail(f"{label} {kind} speculative accounting differs between A/B routes")
        decode = cell_summary(records, "decode")
        whole = cell_summary(records, "whole")
        cells.append({
            "draft_tokens": draft, "verify_width": width,
            "output_and_spec_accounting_exact": True,
            "decode": decode, "whole_inference": whole,
            "eligible_for_later_production_evaluation": (
                decode["passed_initial_screen"] and whole["passed_initial_screen"]),
        })
    return {
        "artifact_type": "ninfer_r9700_dflash_small_t_whole_ab_result",
        "schema_version": 2,
        "status": "complete_fast_c1_mechanism_screen",
        "production_routing_authorized": False,
        "production_recipe_selected": False,
        "scope": {"concurrency": [1], "maximum_product_concurrency": 4,
                  "prompt_tokens": PROMPT_TOKENS, "decode_steps": DECODE_STEPS,
                  "profiles": [{"draft_tokens": 4, "verify_width": 5},
                               {"draft_tokens": 5, "verify_width": 6}]},
        "artifact": artifact, "matched_build_receipt": {
            "path": receipt["path"], "sha256": receipt["sha256"]},
        "ordinary_control": {"report": inspect(ordinary_path), "process": ordinary_process},
        "cells": cells,
        "failed_attempt_provenance": plan["failed_attempt_provenance"],
        "limitation": "P128/G64 C1 canonical-Q4 mechanism screen only; a pass only retains the route for later selected-recipe matched 8K/32K C1..4 evidence and is not production admission",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    plan = validate_plan()
    results = Path(plan["results_directory"])
    if not args.validate_existing and (results.exists() or results.is_symlink()):
        fail(f"refusing to overwrite A/B results directory: {results}")
    artifact = validate_companion(plan)
    receipt, builds = validate_builds(plan)
    corpus = Path(plan["workload"]["corpus"])
    if inspect(corpus)["sha256"] != plan["workload"]["corpus_sha256"]:
        fail("benchmark corpus identity changed")
    require_power()
    if args.preflight_only:
        print(json.dumps({"status": "passed", "artifact": artifact,
                          "matched_build_receipt": {"path": receipt["path"],
                                                    "sha256": receipt["sha256"]},
                          "builds": builds, "power_profile": "auto",
                          "production_routing_authorized": False}, indent=2))
        return 0
    if args.validate_existing:
        if not results.is_dir():
            fail("A/B results directory does not exist")
        summary = validate_results(plan, artifact, receipt, builds, results)
        retained = load(results / "summary.json")
        if retained != summary:
            fail("retained A/B summary differs from recomputed raw evidence")
        print(json.dumps({"status": "passed", "summary": str(results / "summary.json"),
                          "sha256": sha(results / "summary.json")}, indent=2))
        return 0
    results.mkdir(parents=False)
    preflight = {
        "artifact_type": "ninfer_r9700_dflash_small_t_whole_ab_preflight",
        "schema_version": 2, "artifact": artifact,
        "matched_build_receipt": {"path": receipt["path"], "sha256": receipt["sha256"]},
        "builds": builds, "corpus": inspect(corpus), "power_profile": "auto",
        "production_routing_authorized": False, "production_recipe_selected": False,
    }
    (results / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    ordinary = results / "ordinary-control.json"
    run_one(command_for(Path(builds["control"]["path"]), Path(artifact["path"]), corpus,
                        ordinary, 0, 0, ordinary=True), results / "ordinary-control")
    for draft, width in ((4, 5), (5, 6)):
        label = f"K{draft}_W{width}"
        for index, role in enumerate(plan["workload"]["launch_orders"][label]):
            stem = results / f"{label.lower()}-{index + 1}-{role}"
            run_one(command_for(Path(builds[role]["path"]), Path(artifact["path"]), corpus,
                                stem.with_suffix(".json"), draft, width), stem)
    require_power()
    summary = validate_results(plan, artifact, receipt, builds, results)
    summary_path = results / "summary.json"
    if summary_path.exists() or summary_path.is_symlink():
        fail("refusing to overwrite A/B summary")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(summary_path),
                      "sha256": sha(summary_path), "cells": summary["cells"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
