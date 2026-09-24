#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import prepare


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PLAN_PATH = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))
    if not isinstance(value, dict):
        fail(f"not an object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def require_identity(record: dict, path: Path, label: str) -> None:
    if record != identity(path):
        fail(f"{label} identity differs")


def prior_module(plan: dict):
    require_identity(plan["imported_helpers"]["original_whole_runner"],
                     prepare.PRIOR_RUN, "original whole runner before import")
    spec = importlib.util.spec_from_file_location("ninfer_reviewed_bf16_gdn_whole_ab_run",
                                                  prepare.PRIOR_RUN)
    if spec is None or spec.loader is None:
        fail("cannot load reviewed whole-A/B implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_plan() -> tuple[dict, dict, Path]:
    package, reviewed, direct, current_vram = prepare.validate_authorities(False, True)
    plan = load(PLAN_PATH)
    expected_sources = {str(path.resolve()) for path in
                        (prepare.PACKAGE_PLAN, PACKAGE / "commands.sh",
                         PACKAGE / "prepare.py", PACKAGE / "run.py")}
    sources = plan.get("sources")
    if (plan.get("schema") != "ninfer.r9700.bf16-gdn-control-t1-whole-ab-retry2-plan.v1" or
            plan.get("status") != "prepared" or
            plan.get("production_routing_authorized") is not False or
            not isinstance(sources, list) or
            {item.get("path") for item in sources} != expected_sources or
            plan.get("reviewed_plan") != package["reviewed_plan"] or
            plan.get("retained_failure") != package["retained_failure"] or
            plan.get("retained_retry_failure") != package["retained_retry_failure"] or
            plan.get("imported_helpers") != package["imported_helpers"] or
            plan.get("post_exit_drain") != package["post_exit_drain"] or
            plan.get("measurement_limitations") != package["measurement_limitations"] or
            plan.get("workload") != reviewed["workload"] or
            plan.get("order") != reviewed["order"] or
            plan.get("builds") != reviewed["builds"] or
            plan.get("direct_qualification") != reviewed["direct_qualification"] or
            plan.get("artifact") != reviewed["artifact"] or
            plan.get("corpus") != reviewed["corpus"] or
            plan.get("retained_token_authority") != reviewed["retained_token_authority"] or
            plan.get("expected_tokens") != reviewed["expected_tokens"] or
            plan.get("admission") != reviewed["admission"] or
            plan.get("exact_invocation") != package["exact_invocation"] or
            plan.get("vram_gate") != package["vram_gate"] or
            plan.get("campaign") != {"fresh_complete_six_role_campaign": True,
                                     "partial_pair_resume_allowed": False}):
        fail("retry plan differs from reviewed authorities")
    for record in sources:
        require_identity(record, Path(record["path"]), "retry source")
    preflight = plan.get("preflight_vram", {})
    if (preflight.get("passed") is not True or
            preflight.get("pci_bus_id") != current_vram["pci_bus_id"] or
            preflight.get("total_bytes") != current_vram["total_bytes"]):
        fail("retry plan VRAM preflight receipt differs")
    profile = Path(direct["device"]["power_profile_path"]).resolve(strict=True)
    if profile.read_text().strip() != "auto":
        fail("device-0 power profile is not auto")
    return plan, direct, profile


def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)


def command(plan: dict, stem: str, role: str) -> list[str]:
    return [plan["builds"][role]["executable"]["path"], "--weights", plan["artifact"]["path"],
            "--corpus", plan["corpus"]["path"], "--device", "0", "--concurrency", "1",
            "--whole-pg", "8192,256", "--prefill-chunk", "4096", "--kv-capacity", "workload",
            "--spec", "mtp", "--draft-tokens", "0", "--retain-token-ids", "--output", "json",
            "--output-file", str(RESULTS / f"{stem}.json"), "-r", "1", "--warmup", "1"]



def drain_vram(direct: dict, contract: dict, drain: dict, profile: Path,
               expected_total: int, start: float) -> dict:
    samples = []
    while True:
        sample = {"started_seconds": time.monotonic() - start,
                  "vram": None, "vram_error": None, "power": None, "power_error": None}
        try:
            sample["vram"] = prepare.read_vram(direct, contract, False)
            if sample["vram"]["total_bytes"] != expected_total:
                raise RuntimeError("VRAM capacity changed since preflight")
        except Exception as error:
            sample["vram_error"] = f"{type(error).__name__}: {error}"
        try:
            sample["power"] = profile.read_text().strip()
        except Exception as error:
            sample["power_error"] = f"{type(error).__name__}: {error}"
        elapsed = sample["finished_seconds"] = time.monotonic() - start
        samples.append(sample)
        if sample["vram_error"] is not None or sample["power_error"] is not None:
            status = "telemetry_error"
            break
        if sample["power"] != "auto":
            status = "power_changed"
            break
        if elapsed > drain["maximum_seconds"]:
            status = "timeout"
            break
        if sample["vram"]["passed"]:
            status = "drained"
            break
        remaining = drain["maximum_seconds"] - elapsed
        if remaining <= 0:
            status = "timeout"
            break
        time.sleep(min(drain["poll_seconds"], remaining))
        elapsed = time.monotonic() - start
        if elapsed > drain["maximum_seconds"]:
            status = "timeout"
            break
    return {"status": status, "samples": samples,
            "initial": samples[0]["vram"], "final": samples[-1]["vram"],
            "elapsed_seconds": elapsed, "sample_count": len(samples),
            "poll_count": len(samples) - 1,
            "power_samples_auto": all(s["power"] == "auto" and
                                      s["power_error"] is None for s in samples),
            "maximum_seconds": drain["maximum_seconds"],
            "poll_seconds": drain["poll_seconds"]}


def validate_vram(receipt: dict, direct: dict, contract: dict, require_low: bool) -> None:
    used_path, total_path, pci = prepare.vram_paths(direct)
    used, total = receipt.get("used_bytes"), receipt.get("total_bytes")
    if (type(used) is not int or type(total) is not int or used < 0 or
            total < contract["minimum_total_bytes"] or used > total):
        fail("invalid VRAM receipt amounts")
    fraction = (1000 * used + total - 1) // total
    passed = (used <= contract["maximum_used_bytes"] and
              fraction <= contract["maximum_used_fraction_permille"])
    if (receipt.get("pci_bus_id") != pci or receipt.get("used_path") != str(used_path) or
            receipt.get("total_path") != str(total_path) or
            receipt.get("used_fraction_permille_ceil") != fraction or
            receipt.get("maximum_used_bytes") != contract["maximum_used_bytes"] or
            receipt.get("maximum_used_fraction_permille") !=
            contract["maximum_used_fraction_permille"] or
            receipt.get("passed") is not passed or (require_low and not passed)):
        fail("VRAM receipt contract differs")


def run_one(plan: dict, direct: dict, profile: Path, stem: str, role: str) -> None:
    before_vram = prepare.read_vram(direct, plan["vram_gate"], True)
    expected_total = plan["preflight_vram"]["total_bytes"]
    if before_vram["total_bytes"] != expected_total:
        fail(f"VRAM capacity changed before {stem}")
    if profile.read_text().strip() != "auto":
        fail(f"power is not auto before {stem}")
    executable = Path(plan["builds"][role]["executable"]["path"])
    require_identity(plan["builds"][role]["executable"], executable, f"{stem} executable")
    cmd = command(plan, stem, role)
    started = time.time_ns()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    drain_started = time.monotonic()
    finished = time.time_ns()
    drain = drain_vram(direct, plan["vram_gate"], plan["post_exit_drain"], profile,
                       expected_total, drain_started)
    stdout = RESULTS / f"{stem}.stdout"
    stderr = RESULTS / f"{stem}.stderr"
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    power_after = drain["samples"][-1]["power"]
    after_vram = drain["final"]
    benchmark = RESULTS / f"{stem}.json"
    report_identity = identity(benchmark) if benchmark.is_file() else None
    receipt = {"stem": stem, "role": role, "command": cmd, "cwd": str(ROOT),
               "exit_code": completed.returncode, "started_unix_ns": started,
               "finished_unix_ns": finished, "power_profile_path": str(profile),
               "power_before": "auto", "power_after": power_after,
               "vram_before": before_vram, "vram_after": after_vram,
               "post_exit_drain": drain,
               "post_exit_drain_started_monotonic_seconds": drain_started,
               "executable": identity(executable), "stdout": identity(stdout),
               "stderr": identity(stderr), "benchmark_report": report_identity}
    write_exclusive(RESULTS / f"{stem}.process.json", json.dumps(receipt, indent=2) + "\n")
    if completed.returncode != 0 or drain["status"] != "drained":
        fail(f"benchmark failed or left dirty device state: {stem}")


def validate_process(plan: dict, direct: dict, profile: Path, stem: str, role: str) -> None:
    process = load(RESULTS / f"{stem}.process.json")
    drain_started = process.get("post_exit_drain_started_monotonic_seconds")
    if (process.get("stem") != stem or process.get("role") != role or
            process.get("command") != command(plan, stem, role) or
            process.get("cwd") != str(ROOT) or process.get("exit_code") != 0 or
            process.get("power_profile_path") != str(profile) or
            process.get("power_before") != "auto" or process.get("power_after") != "auto" or
            not isinstance(process.get("started_unix_ns"), int) or
            not isinstance(process.get("finished_unix_ns"), int) or
            not isinstance(drain_started, (int, float)) or not math.isfinite(drain_started) or
            drain_started < 0 or
            process["started_unix_ns"] >= process["finished_unix_ns"]):
        fail(f"retry process receipt differs: {stem}")
    for label in ("vram_before", "vram_after"):
        validate_vram(process.get(label, {}), direct, plan["vram_gate"], True)
        if process[label]["total_bytes"] != plan["preflight_vram"]["total_bytes"]:
            fail(f"{stem} capacity differs from preflight")
    drain = process.get("post_exit_drain", {})
    validate_vram(drain.get("initial", {}), direct, plan["vram_gate"], False)
    validate_vram(drain.get("final", {}), direct, plan["vram_gate"], True)
    elapsed = drain.get("elapsed_seconds")
    samples = drain.get("samples")
    if (not isinstance(samples, list) or not samples or
            drain.get("status") != "drained" or
            drain.get("final") != process["vram_after"] or
            drain.get("power_samples_auto") is not True or
            drain.get("maximum_seconds") != plan["post_exit_drain"]["maximum_seconds"] or
            drain.get("poll_seconds") != plan["post_exit_drain"]["poll_seconds"] or
            type(drain.get("sample_count")) is not int or drain["sample_count"] < 1 or
            drain.get("poll_count") != drain["sample_count"] - 1 or
            drain["sample_count"] != len(samples) or
            not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or
            not 0 <= elapsed <= plan["post_exit_drain"]["maximum_seconds"]):
        fail(f"{stem} post-exit drain receipt differs")
    previous = 0.0
    for index, sample in enumerate(samples):
        begin, end = sample.get("started_seconds"), sample.get("finished_seconds")
        if (not isinstance(begin, (float, int)) or not isinstance(end, (float, int)) or
                not math.isfinite(begin) or not math.isfinite(end) or
                not previous <= begin <= end <= elapsed or
                sample.get("vram_error") is not None or sample.get("power_error") is not None or
                sample.get("power") != "auto"):
            fail(f"{stem} invalid drain telemetry sample")
        validate_vram(sample.get("vram", {}), direct, plan["vram_gate"], index == len(samples) - 1)
        if sample["vram"]["total_bytes"] != plan["preflight_vram"]["total_bytes"]:
            fail(f"{stem} sampled capacity differs from preflight")
        if index < len(samples) - 1 and sample["vram"]["passed"]:
            fail(f"{stem} polling continued after a clean sample")
        previous = end
    if (drain["initial"] != samples[0]["vram"] or drain["final"] != samples[-1]["vram"] or
            elapsed != samples[-1]["finished_seconds"] or process["power_after"] != samples[-1]["power"]):
        fail(f"{stem} drain summary differs from samples")
    require_identity(process["executable"], Path(process["command"][0]), f"{stem} executable")
    require_identity(process["stdout"], RESULTS / f"{stem}.stdout", f"{stem} stdout")
    require_identity(process["stderr"], RESULTS / f"{stem}.stderr", f"{stem} stderr")
    require_identity(process["benchmark_report"], RESULTS / f"{stem}.json", f"{stem} report")


def main() -> int:
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("retry results already exist; never resume or overwrite")
    plan, direct, profile = validate_plan()
    prior = prior_module(plan)
    prior.RESULTS = RESULTS
    RESULTS.mkdir()
    try:
        records = []
        for index, role in enumerate(plan["order"], 1):
            stem = f"run-{index}-{role}"
            run_one(plan, direct, profile, stem, role)
            validate_process(plan, direct, profile, stem, role)
            tokens, seconds, rate = prior.validate(plan, profile, stem, role)
            records.append({"stem": stem, "role": role, "tokens": tokens,
                            "decode_seconds": seconds, "decode_output_tok_s": rate})
        if len(records) != 6 or len({record["tokens"] for record in records}) != 1:
            fail("complete campaign token parity differs")
        pairs = []
        ratios = []
        for index in range(3):
            left, right = records[2 * index:2 * index + 2]
            control = left if left["role"] == "control" else right
            candidate = left if left["role"] == "candidate" else right
            ratio = candidate["decode_seconds"] / control["decode_seconds"]
            ratios.append(ratio)
            pairs.append({"order": [left["role"], right["role"]],
                          "control_seconds": control["decode_seconds"],
                          "candidate_seconds": candidate["decode_seconds"],
                          "candidate_over_control": ratio})
        upper = statistics.mean(ratios) + 2.0 * statistics.stdev(ratios) / math.sqrt(3)
        if any(ratio >= 1.0 for ratio in ratios) or upper >= 1.0 or statistics.median(ratios) > 0.99:
            fail("whole-model performance admission failed")
        summary = {"schema": "ninfer.r9700.bf16-gdn-control-t1-whole-ab-retry2-summary.v1",
                   "status": "passed", "production_routing_authorized": True,
                   "fresh_complete_six_role_campaign": True,
                   "measurement_limitations": plan["measurement_limitations"],
                   "exact_public_token_parity": True, "pairs": pairs,
                   "paired_ratio_mean": statistics.mean(ratios),
                   "paired_ratio_upper_2se": upper,
                   "paired_ratio_median": statistics.median(ratios),
                   "control_decode_output_tok_s":
                       [record["decode_output_tok_s"] for record in records
                        if record["role"] == "control"],
                   "candidate_decode_output_tok_s":
                       [record["decode_output_tok_s"] for record in records
                        if record["role"] == "candidate"]}
        write_exclusive(RESULTS / "summary.json", json.dumps(summary, indent=2) + "\n")
        files = sorted(path for path in RESULTS.iterdir()
                       if path.is_file() and path.name != "result.sha256")
        write_exclusive(RESULTS / "result.sha256",
                        "".join(f"{digest(path)}  {path.name}\n" for path in files))
        print("r9700_bf16_gdn_control_t1_whole_ab_retry2: PASS")
        return 0
    except Exception:
        files = sorted(path for path in RESULTS.iterdir()
                       if path.is_file() and path.name != "result.sha256")
        if not (RESULTS / "result.sha256").exists():
            write_exclusive(RESULTS / "result.sha256",
                            "".join(f"{digest(path)}  {path.name}\n" for path in files))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
