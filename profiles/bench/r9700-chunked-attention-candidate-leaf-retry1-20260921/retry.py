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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "qualification", "whole"))
    args = parser.parse_args()
    inputs = preflight()
    if args.stage == "qualification":
        qualification(inputs)
    elif args.stage == "whole":
        base.whole(inputs)
    else:
        print("retained raw/static/host PASS evidence and fixture-only retry bindings verified")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
