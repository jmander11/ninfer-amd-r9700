#!/usr/bin/env python3
"""Run the exact fresh-P129 versus isolated-prefix-reuse functional discriminator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

REPO = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PLAN_PATH = PACKAGE / "plan.json"
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
PROFILE_EXACT = {"LD_PRELOAD", "LD_AUDIT", "HIP_FORCE_QUEUE_PROFILING",
                 "AMD_SERIALIZE_KERNEL", "AMD_SERIALIZE_COPY", "GPU_DUMP_CODE_OBJECT"}
PROFILE_PREFIXES = ("ROCPROF", "ROCP_", "ROCTRACER_", "ROCTX_", "HSA_TOOLS_",
                    "HIP_TRACE_", "AQLPROFILE_", "ATT_PROFILE")
PROFILE_POLICY = {"exact": sorted(PROFILE_EXACT), "prefixes": list(PROFILE_PREFIXES)}
PROMPT = 128
HISTORY = 129
GEN = 64
OUTPUTS = 65


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        fail(f"required file is not regular: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha(resolved)}


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        fail(f"{label} is not finite/nonnegative")
    return result


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
    if (plan.get("artifact_type") != "ninfer_r9700_dflash_p129_isolation_discriminator_plan" or
            plan.get("schema_version") != 1 or
            plan.get("production_routing_authorized") is not False or
            plan.get("production_recipe_selected") is not False):
        fail("plan identity or no-production boundary changed")
    work = plan.get("workload", {})
    if work != {
            "device": 0, "concurrency": 1, "maximum_product_concurrency": 4,
            "prompt_tokens_before_seed": 128, "effective_history_tokens": 129,
            "decode_steps": 64, "requested_output_tokens": 65,
            "prefill_chunk": 4096, "kv_capacity": "workload",
            "dflash": {"draft_tokens": 4, "verify_width": 5,
                       "proposal_head": "optimized"},
            "repetitions": 1, "warmup": 0, "phases": ["eager", "graph"],
            "seed_authority_arms_per_phase": ["ordinary_seed_p128", "dflash_seed_p128"],
            "discriminator_arms_per_phase": ["ordinary_isolated", "ordinary_fresh_p129",
                                             "dflash_isolated", "dflash_fresh_p129"],
            "total_single_test_processes": 12}:
        fail("workload is not the exact bounded twelve-process discriminator")
    decision = plan.get("decision", {})
    if (decision.get("classification_only") is not True or
            decision.get("timing_evidence_eligible") is not False or
            decision.get("all_arms_single_test_processes") is not True or
            decision.get("each_phase_and_spec_mode_has_a_fresh_p128_seed_authority") is not True or
            decision.get("fresh_and_isolated_effective_histories_must_be_exactly_identical")
            is not True or decision.get("candidate_small_t_route_used") is not False or
            decision.get("no_routing_or_recipe_claim") is not True):
        fail("decision boundary changed")
    return plan


def validate_inputs(plan: dict) -> dict:
    artifact_expected = plan["artifact"]
    artifact = inspect(Path(artifact_expected["path"]))
    if artifact["bytes"] != artifact_expected["bytes"] or \
            artifact["sha256"] != artifact_expected["sha256"]:
        fail("DFlash evaluation artifact identity changed")

    build = plan["build"]
    receipt_path = Path(build["receipt"])
    if sha(receipt_path) != build["receipt_sha256"]:
        fail("matched-build receipt identity changed")
    receipt = load(receipt_path)
    control = inspect(Path(build["control_executable"]))
    if (control["bytes"] != build["control_executable_bytes"] or
            control["sha256"] != build["control_executable_sha256"] or
            receipt.get("source", {}).get("commit") != build["source_commit"] or
            receipt.get("builds", {}).get("control", {}).get("benchmark_executable") != control or
            receipt.get("builds", {}).get("control", {}).get(
                "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE") != 0):
        fail("control build is not the exact selector-off cd966d72 build")

    authority = plan["history_authority"]
    source = inspect(Path(authority["source_corpus"]))
    if (source["bytes"] != authority["source_corpus_bytes"] or
            source["sha256"] != authority["source_corpus_sha256"]):
        fail("source corpus identity changed")
    seed_report = inspect(Path(authority["seed_report"]))
    if (seed_report["bytes"] != authority["seed_report_bytes"] or
            seed_report["sha256"] != authority["seed_report_sha256"]):
        fail("seed authority report identity changed")
    report = load(Path(authority["seed_report"]))
    tests = report.get("tests")
    if not isinstance(tests, list):
        fail("seed authority has no tests")
    whole = next((test for test in tests if test.get("label") == "whole-pp128+tg64"), None)
    if not isinstance(whole, dict) or whole.get("kind") != "whole" or \
            whole.get("n_prompt") != PROMPT or whole.get("n_gen") != GEN:
        fail("seed authority lacks exact fresh-P128 ordinary test")
    seed_sequences = []
    for rep in whole.get("reps", []):
        lanes = rep.get("generated_token_ids_by_lane")
        if not isinstance(lanes, list) or len(lanes) != 1 or len(lanes[0]) != OUTPUTS:
            fail("seed authority lacks exact retained C1 outputs")
        seed_sequences.append(tuple(lanes[0]))
    if not seed_sequences or len(set(seed_sequences)) != 1 or \
            seed_sequences[0][0] != authority["seed_token"]:
        fail("seed authority does not prove the exact deterministic seed")

    history_path = PACKAGE / authority["history_corpus"]
    history = inspect(history_path)
    if (history["bytes"] != authority["history_corpus_bytes"] or
            history["sha256"] != authority["history_corpus_sha256"]):
        fail("P129 history fixture identity changed")
    source_ids = [int(token) for token in Path(source["path"]).read_text().split()]
    history_ids = [int(token) for token in history_path.read_text().split()]
    if len(history_ids) != HISTORY or history_ids[:PROMPT] != source_ids[:PROMPT] or \
            history_ids[-1] != authority["seed_token"]:
        fail("P129 history is not exact source[0:128] plus retained greedy seed")
    return {"artifact": artifact, "control": control, "receipt": inspect(receipt_path),
            "source_corpus": source, "seed_report": seed_report, "history_corpus": history}


def command_for(plan: dict, inputs: dict, phase: str, arm: str, report: Path) -> list[str]:
    isolated = arm.endswith("_isolated")
    dflash = arm.startswith("dflash_")
    seed = arm.endswith("_seed_p128")
    corpus = (inputs["source_corpus"]["path"] if isolated or seed
              else inputs["history_corpus"]["path"])
    command = [inputs["control"]["path"], "--weights", inputs["artifact"]["path"],
               "--corpus", corpus, "--device", "0", "--concurrency", "1"]
    if isolated:
        command += ["-pg", "128,64"]
    elif seed:
        command += ["--whole-pg", "128,1"]
    else:
        command += ["--whole-pg", "129,64"]
    command += ["--prefill-chunk", "4096", "--kv-capacity", "workload"]
    if dflash:
        command += ["--spec", "dflash", "--draft-tokens", "4",
                    "--dflash-verify-width", "5", "--lm-head-draft"]
    else:
        command += ["--draft-tokens", "0"]
    command += ["--retain-token-ids"]
    if isolated:
        command += ["--isolate-prompt-decode"]
    if phase == "eager":
        command += ["--no-device-graph"]
    command += ["--output", "json", "--output-file", str(report),
                "-r", "1", "--warmup", "0"]
    return command


def expected_corpus_token_count(isolated: bool, seed: bool) -> int:
    return 65536 if isolated or seed else HISTORY


def run_one(command: list[str], stem: Path) -> None:
    report = Path(command[command.index("--output-file") + 1])
    stdout = stem.with_suffix(".stdout.txt")
    stderr = stem.with_suffix(".stderr.txt")
    process_path = stem.with_suffix(".process.json")
    if any(path.exists() or path.is_symlink() for path in (report, stdout, stderr, process_path)):
        fail(f"refusing to overwrite output for {stem.name}")
    before = require_power()
    environment = os.environ.copy()
    removed = sorted(key for key in environment if instrumentation_key(key))
    for key in removed:
        environment.pop(key)
    started = time.time_ns()
    process = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True)
    finished = time.time_ns()
    after = require_power()
    stdout.write_text(process.stdout, encoding="utf-8")
    stderr.write_text(process.stderr, encoding="utf-8")
    record = {
        "command": command, "exit_code": process.returncode,
        "started_unix_ns": started, "finished_unix_ns": finished,
        "power_profile_before": before, "power_profile_after": after,
        "profiling_environment_policy": PROFILE_POLICY,
        "profiling_environment_removed": removed,
        "stdout": inspect(stdout), "stderr": inspect(stderr),
    }
    process_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if process.returncode != 0 or not report.is_file():
        fail(f"arm failed; retained process evidence at {process_path}")


def instrumentation_key(key: str) -> bool:
    return key in PROFILE_EXACT or key.startswith(PROFILE_PREFIXES)


def validate_spec(spec: object, dflash: bool, label: str, decode_steps: int) -> tuple:
    if not isinstance(spec, dict) or spec.get("enabled") is not dflash or \
            spec.get("draft_window") != (4 if dflash else 0):
        fail(f"{label} speculative identity differs")
    values = []
    for key in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps"):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(f"{label} {key} is invalid")
        values.append(value)
    rounds, drafted, accepted, fallback = values
    positions = spec.get("accepted_per_position")
    if not isinstance(positions, list) or len(positions) != (4 if dflash else 0) or \
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in positions) or sum(positions) != accepted or accepted > drafted:
        fail(f"{label} speculative accounting is invalid")
    if dflash:
        if rounds + accepted + fallback != decode_steps:
            fail(f"{label} DFlash engine work does not account for exactly {decode_steps} transitions")
        if drafted > 0:
            if rounds <= 0:
                fail(f"{label} drafted work has no rounds")
            expected_rate = accepted / drafted
            expected_length = 1.0 + accepted / rounds
            if (not math.isclose(finite(spec.get("acceptance_rate"), f"{label} acceptance_rate"),
                                 expected_rate, rel_tol=1e-9, abs_tol=1e-12) or
                    not math.isclose(finite(spec.get("acceptance_length"),
                                            f"{label} acceptance_length"),
                                     expected_length, rel_tol=1e-9, abs_tol=1e-12)):
                fail(f"{label} acceptance summaries are inconsistent")
        elif rounds != 0 or accepted != 0 or spec.get("acceptance_rate") is not None or \
                spec.get("acceptance_length") is not None:
            fail(f"{label} zero-draft fallback has speculative summaries")
    elif any(values):
        fail(f"{label} ordinary arm has speculative work")
    elif spec.get("acceptance_rate") is not None or spec.get("acceptance_length") is not None:
        fail(f"{label} ordinary arm has speculative summaries")
    return rounds, drafted, accepted, fallback, tuple(positions)


def validate_process(stem: Path, command: list[str]) -> dict:
    record_path = stem.with_suffix(".process.json")
    record = load(record_path)
    if (record.get("command") != command or record.get("exit_code") != 0 or
            record.get("power_profile_before") != "auto" or
            record.get("power_profile_after") != "auto" or
            record.get("profiling_environment_policy") != PROFILE_POLICY or
            not isinstance(record.get("profiling_environment_removed"), list) or
            record.get("profiling_environment_removed") !=
            sorted(set(record["profiling_environment_removed"])) or
            any(not isinstance(key, str) or not instrumentation_key(key)
                for key in record["profiling_environment_removed"]) or
            isinstance(record.get("started_unix_ns"), bool) or
            not isinstance(record.get("started_unix_ns"), int) or
            isinstance(record.get("finished_unix_ns"), bool) or
            not isinstance(record.get("finished_unix_ns"), int) or
            record["finished_unix_ns"] <= record["started_unix_ns"] or
            record.get("stdout") != inspect(stem.with_suffix(".stdout.txt")) or
            record.get("stderr") != inspect(stem.with_suffix(".stderr.txt"))):
        fail(f"process evidence differs for {stem.name}")
    return inspect(record_path)


def validate_report(plan: dict, inputs: dict, phase: str, arm: str,
                    path: Path, command: list[str]) -> dict:
    report = load(path)
    isolated = arm.endswith("_isolated")
    dflash = arm.startswith("dflash_")
    seed = arm.endswith("_seed_p128")
    decode_steps = 1 if seed else GEN
    outputs = decode_steps + 1
    expected_prompt = PROMPT if isolated or seed else HISTORY
    expected_label = ("pp128+tg64" if isolated else
                      "whole-pp128+tg1" if seed else "whole-pp129+tg64")
    expected_kind = "pp+tg" if isolated else "whole"
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report" or
            report.get("tool") != "ninfer_bench" or
            shlex.split(report.get("command", "")) != command or
            report.get("artifact") != {"path": inputs["artifact"]["path"],
                                      "file_size_bytes": inputs["artifact"]["bytes"]} or
            report.get("load", {}).get("target") != "qwen3_8_27b_r9700" or
            report.get("load", {}).get("weights_id") != plan["artifact"]["weights_id"]):
        fail(f"report provenance differs for {arm}/{phase}")
    environment = report.get("environment", {})
    if (environment.get("gpu_name") != "AMD Radeon AI PRO R9700" or
            environment.get("architecture_name") != "gfx1201" or
            environment.get("device_id") != 0):
        fail(f"report did not run on exact GPU0 R9700/gfx1201 for {arm}/{phase}")
    config = report.get("config", {})
    expected_context = ((139 if dflash else 129) if seed else 203 if dflash else 193)
    expected_corpus = inputs["source_corpus"] if isolated or seed else inputs["history_corpus"]
    checks = {
        "max_context": expected_context, "concurrency": 1, "prefill_chunk": 4096,
        "spec": "dflash" if dflash else "none", "draft_tokens": 4 if dflash else 0,
        "speculative_execution": dflash,
        "dflash_verify_width_requested": 5 if dflash else 0,
        "dflash_verify_width": 5 if dflash else 0,
        "proposal_head": "optimized" if dflash else "full",
        "use_device_graph": phase == "graph", "retain_token_ids": True,
        "isolate_prompt_decode": isolated,
        "decode_path": ("dflash_" if dflash else "") +
                       ("device_graph" if phase == "graph" else "eager"),
        "decode_graph_prime": {"primed": phase == "graph",
                               "output_tokens": (11 if dflash else 3) if phase == "graph" else 0},
        "repetitions": 1, "warmup": 0,
        "corpus_path": expected_corpus["path"],
        "corpus_tokens": expected_corpus_token_count(isolated, seed),
        "dflash_small_t_candidate": False,
    }
    receipt = load(Path(plan["build"]["receipt"]))
    checks.update(receipt["expected_benchmark_profile"])
    for key, value in checks.items():
        if config.get(key) != value:
            fail(f"report config {key} differs for {arm}/{phase}: {config.get(key)!r}")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) != 1:
        fail(f"{arm}/{phase} was not a single-test process")
    test = tests[0]
    if (test.get("label") != expected_label or test.get("kind") != expected_kind or
            test.get("n_prompt") != expected_prompt or test.get("n_gen") != decode_steps or
            test.get("requested_output_tokens") != outputs):
        fail(f"test geometry differs for {arm}/{phase}")
    reps = test.get("reps")
    if not isinstance(reps, list) or len(reps) != 1:
        fail(f"{arm}/{phase} lacks its one measured repetition")
    rep = reps[0]
    lanes = rep.get("generated_token_ids_by_lane")
    if (rep.get("generated_output_tokens") != outputs or
            rep.get("decode_output_tokens") != decode_steps or
            not isinstance(lanes, list) or len(lanes) != 1 or
            not isinstance(lanes[0], list) or len(lanes[0]) != outputs or
            any(isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in lanes[0])):
        fail(f"retained outputs differ for {arm}/{phase}")
    accounting = validate_spec(rep.get("speculative"), dflash, f"{arm}/{phase} rep", decode_steps)
    expected_engine = decode_steps if not dflash else accounting[0] + accounting[2] + accounting[3]
    if rep.get("decode_engine_tokens") != expected_engine:
        fail(f"decode_engine_tokens differs for {arm}/{phase}")
    aggregate = validate_spec(test.get("speculative"), dflash,
                              f"{arm}/{phase} aggregate", decode_steps)
    if aggregate != accounting:
        fail(f"aggregate speculative accounting differs from rep for {arm}/{phase}")
    for key in ("prepare_seconds", "prefill_seconds", "decode_seconds", "total_seconds"):
        finite(rep.get("timings", {}).get(key), f"{arm}/{phase} {key}")
    return {"tokens": tuple(lanes[0]), "speculative": accounting,
            "report": inspect(path)}


def mismatch(left: tuple, right: tuple) -> dict | None:
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            return {"generated_index": index, "left": a, "right": b}
    return None


def classify(validated: dict) -> dict:
    modes = {}
    for phase in ("eager", "graph"):
        ordinary_pair = mismatch(validated[(phase, "ordinary_isolated")]["tokens"],
                                 validated[(phase, "ordinary_fresh_p129")]["tokens"])
        dflash_pair = mismatch(validated[(phase, "dflash_isolated")]["tokens"],
                               validated[(phase, "dflash_fresh_p129")]["tokens"])
        fresh_cross = mismatch(validated[(phase, "ordinary_fresh_p129")]["tokens"],
                               validated[(phase, "dflash_fresh_p129")]["tokens"])
        isolated_cross = mismatch(validated[(phase, "ordinary_isolated")]["tokens"],
                                  validated[(phase, "dflash_isolated")]["tokens"])
        if ordinary_pair is not None:
            diagnosis = "ordinary_append_vs_fresh_execution_path_dependence"
        elif dflash_pair is not None:
            diagnosis = "dflash_append_vs_fresh_execution_path_dependence"
        elif fresh_cross is not None or isolated_cross is not None:
            diagnosis = "dflash_vs_ordinary_execution_path_dependence"
        else:
            diagnosis = "divergence_not_reproduced"
        modes[phase] = {
            "diagnosis": diagnosis,
            "ordinary_isolated_vs_fresh_p129": ordinary_pair,
            "dflash_isolated_vs_fresh_p129": dflash_pair,
            "ordinary_vs_dflash_fresh_p129": fresh_cross,
            "ordinary_vs_dflash_isolated": isolated_cross,
        }
    graph_differences = {
        arm: mismatch(validated[("eager", arm)]["tokens"],
                      validated[("graph", arm)]["tokens"])
        for arm in ("ordinary_isolated", "ordinary_fresh_p129",
                    "dflash_isolated", "dflash_fresh_p129")
    }
    graph_specific = any(value is not None for value in graph_differences.values())
    return {
        "artifact_type": "ninfer_r9700_dflash_p129_isolation_discriminator_result",
        "schema_version": 1, "status": "functional_classification_complete",
        "classification_only": True, "timing_evidence_eligible": False,
        "production_routing_authorized": False, "production_recipe_selected": False,
        "phase_results": modes, "graph_specific_difference": graph_specific,
        "eager_vs_graph_by_arm": graph_differences,
    }


def analyze_results(plan: dict, inputs: dict, results: Path) -> dict:
    if not results.is_dir():
        fail(f"results directory is absent: {results}")
    preflight_path = results / "preflight.json"
    validated = {}
    for phase in ("eager", "graph"):
        for arm in ("ordinary_seed_p128", "dflash_seed_p128",
                    "ordinary_isolated", "ordinary_fresh_p129",
                    "dflash_isolated", "dflash_fresh_p129"):
            stem = results / f"{phase}-{arm}"
            report = stem.with_suffix(".json")
            command = command_for(plan, inputs, phase, arm, report)
            process = validate_process(stem, command)
            value = validate_report(plan, inputs, phase, arm, report, command)
            value["process"] = process
            validated[(phase, arm)] = value
    expected_seed = plan["history_authority"]["seed_token"]
    seed_authorities = {}
    for phase in ("eager", "graph"):
        for arm in ("ordinary_seed_p128", "dflash_seed_p128"):
            observed = validated[(phase, arm)]["tokens"][0]
            if observed != expected_seed:
                fail(f"{phase}/{arm} seed {observed} differs from P129 history {expected_seed}")
            seed_authorities[f"{phase}-{arm}"] = observed
    summary = classify(validated)
    summary["seed_authorities"] = seed_authorities
    summary["plan"] = inspect(PLAN_PATH)
    summary["preflight"] = inspect(preflight_path)
    summary["arms"] = {
        f"{phase}-{arm}": {"report": value["report"], "process": value["process"],
                           "speculative": list(value["speculative"][:-1]) +
                                          [list(value["speculative"][-1])]}
        for (phase, arm), value in validated.items()
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.analyze_only:
        fail("choose only one of --preflight-only and --analyze-only")
    plan = validate_plan()
    inputs = validate_inputs(plan)
    preflight = {"status": "passed", "plan": inspect(PLAN_PATH), **inputs,
                 "power_profile": require_power()}
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0
    results = Path(plan["results_directory"])
    if args.analyze_only:
        summary = analyze_results(plan, inputs, results)
        if load(results / "summary.json") != summary:
            fail("stored summary differs from recomputed raw evidence")
        print(json.dumps(summary, indent=2))
        return 0
    if results.exists() or results.is_symlink():
        fail(f"refusing to overwrite results directory: {results}")
    results.mkdir(parents=False)
    (results / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n",
                                             encoding="utf-8")
    for phase in ("eager", "graph"):
        for arm in ("ordinary_seed_p128", "dflash_seed_p128",
                    "ordinary_isolated", "ordinary_fresh_p129",
                    "dflash_isolated", "dflash_fresh_p129"):
            stem = results / f"{phase}-{arm}"
            report = stem.with_suffix(".json")
            command = command_for(plan, inputs, phase, arm, report)
            run_one(command, stem)
    summary = analyze_results(plan, inputs, results)
    summary_path = results / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(1)
