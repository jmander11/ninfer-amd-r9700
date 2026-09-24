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
    if (not path.is_file() or path.stat().st_size != item["bytes"] or
            digest(path) != item["sha256"]):
        fail(f"bound identity changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-fresh", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if plan.get("schema") != "ninfer.r9700.gdn-q4-pair-t1-whole-ab-plan.v1" or plan.get("status") != "prepared_awaiting_independent_review" or plan.get("production_routing_authorized") is not False:
        fail("plan disposition differs")
    for item in plan["sources"] + [plan["direct_qualification"], plan["artifact"],
                                    plan["corpus"], plan["retained_token_authority"]]:
        check_identity(item)
    for role, expected in (("control", False), ("candidate", True)):
        build = plan["builds"][role]
        check_identity(build["cache"]); check_identity(build["executable"])
        check_identity(build["symbol_receipt"]["object"])
        if build["symbol_receipt"]["paired_op_undefined_reference"] is not expected:
            fail(f"selector receipt differs: {role}")
        output = subprocess.run(["nm", "-C", build["symbol_receipt"]["object"]["path"]],
                                check=True, capture_output=True, text=True).stdout
        present = " U ninfer::ops::r9700::linear::a8q4g64_gdn_pair_t1(" in output
        if present is not expected:
            fail(f"live selector symbol differs: {role}")
    direct = json.loads(Path(plan["direct_qualification"]["path"]).read_text())
    timing = direct.get("timing", {})
    static = direct.get("static_receipt", {})
    if (direct.get("status") != "qualified_for_whole_ab_only" or
            direct.get("production_routing_authorized") is not False or
            direct.get("correctness", {}).get("maximum_bf16_steps") != 0 or
            timing.get("allocation_balanced") is not True or timing.get("arm_order_balanced") is not True or
            timing.get("all_copy_medians_clear_one_percent") is not True or
            timing.get("intervening_weight_bytes", 0) <= 64 * 1024 * 1024 or
            static.get("status") != "passed" or static.get("wave_size") != 32 or
            static.get("lds_bytes") != 0 or static.get("scratch_bytes") != 0):
        fail("direct qualification receipt differs")
    if args.require_fresh and (RESULTS.exists() or RESULTS.is_symlink()):
        fail("results path is not fresh")
    if POWER.read_text().strip() != "auto":
        fail("R9700 power profile is not auto")
    print("r9700_gdn_q4_pair_t1_whole_ab_preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
