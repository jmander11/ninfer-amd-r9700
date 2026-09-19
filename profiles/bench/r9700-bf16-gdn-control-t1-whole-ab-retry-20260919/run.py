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


def prior_module():
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
    if (plan.get("schema") != "ninfer.r9700.bf16-gdn-control-t1-whole-ab-retry-plan.v1" or
            plan.get("status") != "prepared" or
            plan.get("production_routing_authorized") is not False or
            not isinstance(sources, list) or
            {item.get("path") for item in sources} != expected_sources or
            plan.get("reviewed_plan") != package["reviewed_plan"] or
            plan.get("retained_failure") != package["retained_failure"] or
            plan.get("workload") != reviewed["workload"] or
            plan.get("order") != reviewed["order"] or
            plan.get("builds") != reviewed["builds"] or
            plan.get("direct_qualification") != reviewed["direct_qualification"] or
            plan.get("artifact") != reviewed["artifact"] or
            plan.get("corpus") != reviewed["corpus"] or
            plan.get("retained_token_authority") != reviewed["retained_token_authority"] or
            plan.get("expected_tokens") != reviewed["expected_tokens"] or
            plan.get("admission") != reviewed["admission"] or
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


def run_one(plan: dict, direct: dict, profile: Path, stem: str, role: str) -> None:
    before_vram = prepare.read_vram(direct, plan["vram_gate"], True)
    if profile.read_text().strip() != "auto":
        fail(f"power is not auto before {stem}")
    executable = Path(plan["builds"][role]["executable"]["path"])
    require_identity(plan["builds"][role]["executable"], executable, f"{stem} executable")
    cmd = command(plan, stem, role)
    started = time.time_ns()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = time.time_ns()
    power_after = profile.read_text().strip()
    after_vram = prepare.read_vram(direct, plan["vram_gate"], False)
    stdout = RESULTS / f"{stem}.stdout"
    stderr = RESULTS / f"{stem}.stderr"
    write_exclusive(stdout, completed.stdout)
    write_exclusive(stderr, completed.stderr)
    benchmark = RESULTS / f"{stem}.json"
    report_identity = identity(benchmark) if benchmark.is_file() else None
    receipt = {"stem": stem, "role": role, "command": cmd, "cwd": str(ROOT),
               "exit_code": completed.returncode, "started_unix_ns": started,
               "finished_unix_ns": finished, "power_profile_path": str(profile),
               "power_before": "auto", "power_after": power_after,
               "vram_before": before_vram, "vram_after": after_vram,
               "executable": identity(executable), "stdout": identity(stdout),
               "stderr": identity(stderr), "benchmark_report": report_identity}
    write_exclusive(RESULTS / f"{stem}.process.json", json.dumps(receipt, indent=2) + "\n")
    if completed.returncode != 0 or power_after != "auto" or not after_vram["passed"]:
        fail(f"benchmark failed or left dirty device state: {stem}")


def validate_process(plan: dict, direct: dict, profile: Path, stem: str, role: str) -> None:
    process = load(RESULTS / f"{stem}.process.json")
    if (process.get("stem") != stem or process.get("role") != role or
            process.get("command") != command(plan, stem, role) or
            process.get("cwd") != str(ROOT) or process.get("exit_code") != 0 or
            process.get("power_profile_path") != str(profile) or
            process.get("power_before") != "auto" or process.get("power_after") != "auto" or
            not isinstance(process.get("started_unix_ns"), int) or
            not isinstance(process.get("finished_unix_ns"), int) or
            process["started_unix_ns"] >= process["finished_unix_ns"]):
        fail(f"retry process receipt differs: {stem}")
    for label in ("vram_before", "vram_after"):
        receipt = process.get(label, {})
        if (receipt.get("passed") is not True or
                receipt.get("pci_bus_id") != direct["device"]["pci_bus_id"] or
                receipt.get("maximum_used_bytes") != plan["vram_gate"]["maximum_used_bytes"] or
                receipt.get("maximum_used_fraction_permille") !=
                plan["vram_gate"]["maximum_used_fraction_permille"]):
            fail(f"{stem} {label} receipt differs")
    require_identity(process["executable"], Path(process["command"][0]), f"{stem} executable")
    require_identity(process["stdout"], RESULTS / f"{stem}.stdout", f"{stem} stdout")
    require_identity(process["stderr"], RESULTS / f"{stem}.stderr", f"{stem} stderr")
    require_identity(process["benchmark_report"], RESULTS / f"{stem}.json", f"{stem} report")


def main() -> int:
    if RESULTS.exists() or RESULTS.is_symlink():
        fail("retry results already exist; never resume or overwrite")
    plan, direct, profile = validate_plan()
    prior = prior_module()
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
        summary = {"schema": "ninfer.r9700.bf16-gdn-control-t1-whole-ab-retry-summary.v1",
                   "status": "passed", "production_routing_authorized": True,
                   "fresh_complete_six_role_campaign": True,
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
        print("r9700_bf16_gdn_control_t1_whole_ab_retry: PASS")
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
