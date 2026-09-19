#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PACKAGE = Path(__file__).resolve().parent
PLAN = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"
POWER = Path("/sys/bus/pci/devices/0000:13:00.0/power_dpm_force_performance_level")


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_identity(item: dict) -> None:
    path = Path(item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
        fail(f"bound identity changed: {path}")


def check_closure(directory: Path, closure: Path) -> None:
    for line in closure.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            fail(f"retained whole closure differs: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-fresh", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan.get("schema") != "ninfer.r9700.gdn-q4-pair-t1-production-confirmation-plan.v1" or
            plan.get("status") != "prepared_awaiting_independent_review" or
            plan.get("production_routing_authorized") is not True):
        fail("plan disposition differs")
    for item in (plan["sources"] + [plan["build"]["cache"], plan["build"]["executable"],
                 plan["route_contract"]["static_assert_receipt"],
                 plan["route_contract"]["production_symbol_receipt"]["object"],
                 plan["static_receipt"]["assembly"], plan["direct_qualification"],
                 plan["retained_whole"]["summary"], plan["retained_whole"]["closure"],
                 plan["artifact"], plan["corpus"], plan["retained_token_authority"]]):
        check_identity(item)
    check_closure(Path(plan["retained_whole"]["summary"]["path"]).parent,
                  Path(plan["retained_whole"]["closure"]["path"]))
    if "NINFER_R9700_GDN_Q4_PAIR_T1_CANDIDATE" in Path(plan["build"]["cache"]["path"]).read_text():
        fail("removed selector remains in production cache")
    obj = plan["route_contract"]["production_symbol_receipt"]["object"]["path"]
    symbols = subprocess.run(["nm", "-C", obj], check=True, capture_output=True, text=True).stdout
    if " U ninfer::ops::r9700::linear::a8q4g64_gdn_pair_t1(" not in symbols:
        fail("live production route symbol differs")
    if "wave32=true lds=0 scratch=0" not in plan["static_receipt"]["output"]:
        fail("static resource receipt differs")
    direct = json.loads(Path(plan["direct_qualification"]["path"]).read_text())
    if (direct.get("correctness", {}).get("maximum_bf16_steps") != 0 or
            direct.get("timing", {}).get("allocation_balanced") is not True or
            direct.get("static_receipt", {}).get("status") != "passed"):
        fail("direct qualification receipt differs")
    whole = json.loads(Path(plan["retained_whole"]["summary"]["path"]).read_text())
    if (whole.get("status") != "passed" or whole.get("production_routing_authorized") is not True or
            whole.get("exact_public_token_parity") is not True):
        fail("whole admission receipt differs")
    if plan["route_contract"].get("rejected_tokens") != [2, 3, 4]:
        fail("C2..4 rejection contract differs")
    if args.require_fresh and (RESULTS.exists() or RESULTS.is_symlink()):
        fail("results path is not fresh")
    if POWER.read_text().strip() != "auto":
        fail("R9700 power profile is not auto")
    print("r9700_gdn_q4_pair_t1_production_confirmation_preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
