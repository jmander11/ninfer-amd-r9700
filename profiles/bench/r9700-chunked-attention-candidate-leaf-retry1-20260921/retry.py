#!/usr/bin/env python3
"""Reuse passing bounded-panel evidence and retry only the corrected public leaf."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

PACKAGE = Path(__file__).resolve().parent
ORIGINAL = PACKAGE.parent / "r9700-chunked-attention-candidate-20260921"
RETAINED = ORIGINAL / "qualification"
spec = importlib.util.spec_from_file_location("original_attention_experiment", ORIGINAL / "experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.PACKAGE = PACKAGE
original_preflight = base.preflight
LEAF_SOURCE = "tools/r9700/full_attention_leaf_qual.hip"


def validate_retained(inputs):
    previous = json.loads((RETAINED / "inputs.json").read_text())
    current = copy.deepcopy(inputs)
    # Only the reproduced fixture allocation bug was corrected. Every raw Op,
    # static, host-routing, benchmark, artifact and build input must be unchanged.
    current["public_leaf"] = previous["public_leaf"]
    current["sources"][LEAF_SOURCE] = previous["sources"][LEAF_SOURCE]
    if current != previous:
        raise ValueError("retained qualification inputs changed beyond the public-leaf fixture")
    raw = (RETAINED / "dense-prefill-attention.stdout.txt").read_text().splitlines()
    final = "production Bk16/Bk32 dense G16/G32 FP64 qualification: PASS"
    cases = {(group, rows, context) for group in (16, 32) for rows, context in (
        (128, 128), (129, 129), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096),
        (1024, 8192), (1537, 8192), (8192, 8192), (1025, 32768), (8192, 32768), (128, 262144))}
    parsed = []
    for line in raw[:-1]:
        match = re.fullmatch(r"rows=(\d+) context=(\d+) group=(16|32) panel=(\d+) "
                             r"oracle=FP64 complete-sampled-vectors PASS", line)
        if match is None or int(match[4]) <= 0:
            raise ValueError("retained raw qualification lacks complete PASS cases")
        parsed.append((int(match[3]), int(match[1]), int(match[2])))
    if not raw or raw[-1] != final or set(parsed) != cases or len(parsed) != len(cases):
        raise ValueError("retained raw qualification lacks complete PASS cases")
    host = (RETAINED / "host-routing.stdout.txt").read_text().strip()
    if host != "r9700_runtime_planner: PASS host split-512 production routing":
        raise ValueError("retained host-routing qualification did not pass")
    static = json.loads((RETAINED / "static.json").read_text())
    expected = {("qk_bk16", 16), ("qk_bk32", 16), ("maximum", 16), ("pv", 16), ("pv", 32)}
    if (len(static) != 5 or {(row["full_score_stage"], row["value_group"]) for row in static} != expected
            or any(row[key] != 0 for row in static for key in
                   ("private_bytes", "scratch_bytes", "flat_scratch", "vgpr_spills", "sgpr_spills"))):
        raise ValueError("retained static qualification is incomplete")
    return {name: base.identity(RETAINED / name) for name in (
        "inputs.json", "static.json", "host-routing.stdout.txt", "host-routing.stderr.txt",
        "dense-prefill-attention.stdout.txt", "dense-prefill-attention.stderr.txt")}


def preflight():
    inputs = original_preflight()
    inputs["retained_qualification"] = validate_retained(inputs)
    inputs["experiment_driver"] = base.identity(ORIGINAL / "experiment.py")
    return inputs


base.preflight = preflight


def wait_for_release():
    """Allow an exited owned child up to ten seconds for deferred VRAM teardown."""
    deadline = time.monotonic() + 10.0
    while True:
        base.require_auto_power_profile()
        used = int((base.R9700_POWER_PROFILE.parent / "mem_info_vram_used").read_text())
        if used <= 1024 ** 3:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"owned job VRAM release timed out after 10 seconds: {used} bytes")
        time.sleep(min(0.1, remaining))


base.previous_job_released = wait_for_release


def wait_for_idle():
    """Require an idle sample, allowing transient busy-counter samples to settle."""
    deadline = time.monotonic() + 10.0
    while True:
        base.require_auto_power_profile()
        device = base.R9700_POWER_PROFILE.parent
        used = int((device / "mem_info_vram_used").read_text())
        busy = int((device / "gpu_busy_percent").read_text())
        if used <= 1024 ** 3 and busy == 0:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"R9700 idle wait timed out: resident={used} bytes busy={busy}%")
        time.sleep(min(0.1, remaining))


base.idle = wait_for_idle


def qualification(inputs):
    output = PACKAGE / "qualification"
    if os.path.lexists(output):
        raise ValueError(f"output already exists: {output}")
    base.idle()
    base.require_hip_pci_device(0)
    output.mkdir()
    base.write_new(output / "inputs.json", inputs)
    base.run_logged([str(base.LEAF)], output, "full-attention-leaf", check_idle=False)
    if "r9700_full_attention_leaf: PASS" not in (output / "full-attention-leaf.stdout.txt").read_text():
        raise ValueError("public-leaf qualification did not emit its complete PASS result")
    if preflight() != inputs:
        raise ValueError("qualification inputs changed during retry")
    base.write_new(output / "result.json", {
        "pass": True, "inputs": inputs, "static_complete": True, "numerical_complete": True,
        "command": [str(base.LEAF)], "power_profile": "auto",
        "reused_static_host_raw": inputs["retained_qualification"],
    })


def finish_whole(inputs):
    whole = PACKAGE / "whole"
    completion = PACKAGE / "whole-completion"
    if any(os.path.lexists(path) for path in (
        completion, whole / "result.json", whole / "baseline-p2048.json", whole / "candidate-p2048.json"
    )):
        raise ValueError("whole completion requires an absent completion/result and missing P2048 reports")
    if json.loads((whole / "inputs.json").read_text()) != inputs:
        raise ValueError("retained whole inputs changed before completion")
    qualification_path = PACKAGE / "qualification/result.json"
    qualified = json.loads(qualification_path.read_text())
    if (qualified.get("inputs") != inputs or qualified.get("pass") is not True
            or qualified.get("static_complete") is not True or qualified.get("numerical_complete") is not True):
        raise ValueError("whole completion requires passing qualification of these exact inputs")
    candidate8 = whole / "candidate-p8192.json"
    candidate8_identity = base.identity(candidate8)
    qualification_identity = base.identity(qualification_path)
    results = {"candidate-p8192": base.report(
        candidate8, inputs["artifact"], base.bench_command(base.CANDIDATE, 8192, 1024, candidate8),
        8192, 1024)["tests"][0]}
    results["retained-baseline-p8192"] = base.report(
        base.RETAINED, inputs["artifact"], base.bench_command(base.BASELINE, 8192, 1024, base.RETAINED),
        8192, 1024)["tests"][0]
    base.idle()
    base.require_hip_pci_device(0)
    completion.mkdir()
    reason = "completed candidate P8192 retained; prior stage stopped on deferred owned-child VRAM teardown"
    base.write_new(completion / "inputs.json", {
        "inputs": inputs, "retained_candidate_p8192": candidate8_identity,
        "qualification": qualification_identity, "reason": reason,
    })
    reports = {"candidate-p8192": candidate8_identity}
    for label, binary in (("baseline-p2048", base.BASELINE), ("candidate-p2048", base.CANDIDATE)):
        path = completion / f"{label}.json"
        command = base.bench_command(binary, 2048, 4096, path)
        if base.identity(binary) != inputs["baseline" if binary == base.BASELINE else "candidate"]:
            raise ValueError("benchmark changed before remaining P2048 measurement")
        base.run_logged(command, completion, label, check_idle=label != "baseline-p2048")
        results[label] = base.report(path, inputs["artifact"], command, 2048, 4096)["tests"][0]
        reports[label] = base.identity(path)
    if (preflight() != inputs or base.identity(candidate8) != candidate8_identity
            or base.identity(qualification_path) != qualification_identity):
        raise ValueError("retained experiment evidence changed during completion")
    decision = base.admission(results)
    base.write_new(whole / "result.json", {
        "artifact_type": "ninfer_chunked_attention_whole_prefill_comparison", "schema_version": 1,
        **decision, "inputs": inputs, "power_profile": "auto", "pci": "0000:13:00.0",
        "qualification": qualification_identity, "retained_baseline_p8192_tok_s": 184.8640144,
        "measurements": results, "continuation_reason": reason, "reports": reports,
        "completion_inputs": base.identity(completion / "inputs.json"),
    })
    if decision["status"] != "admitted":
        raise ValueError("whole-inference admission failed; result retained; no automatic retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "qualification", "whole", "finish-whole"))
    args = parser.parse_args()
    inputs = preflight()
    if args.stage == "qualification":
        qualification(inputs)
    elif args.stage == "whole":
        base.whole(inputs)
    elif args.stage == "finish-whole":
        finish_whole(inputs)
    else:
        print("retained raw/static/host PASS evidence and fixture-only retry bindings verified")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
