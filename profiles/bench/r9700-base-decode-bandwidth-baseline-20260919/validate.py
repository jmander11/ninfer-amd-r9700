#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    plan = json.loads((PACKAGE / "plan.json").read_text())
    inventory = plan.get("dot8_round_inventory", [])
    if sum(row.get("calls", 0) for row in inventory) != 321:
        fail("dot8 inventory is not 321 calls")
    useful = 0
    weight = 0
    for row in inventory:
        n, k, calls = row["rows"], row["columns"], row["calls"]
        kpad = (k + 127) // 128 * 128
        groups = kpad // 64
        weight_call = n * kpad // 2 + n * groups * 2
        useful_call = weight_call + kpad + groups * 2 + 4 + n * 2
        weight += calls * weight_call
        useful += calls * useful_call
    contract = plan.get("useful_byte_contract", {})
    if useful != contract.get("bytes_per_round") or weight != contract.get(
            "weight_code_and_scale_bytes_per_round"):
        fail("useful-byte contract differs from inventory")
    required = {"plan.json", "prepare.py", "preflight.py", "run.py", "profile.sh",
                "commands.sh", "validate.py"}
    if not all((PACKAGE / name).is_file() for name in required):
        fail("package file is missing")
    bound = PACKAGE / "bound-plan.json"
    if bound.is_file():
        value = json.loads(bound.read_text())
        if value.get("status") != "prepared_awaiting_independent_review_and_gpu_execution":
            fail("bound plan status differs")
        if "benchmark" not in value or "selector_free_cache" not in value:
            fail("bound plan lacks selector-free build")
    print("r9700_base_decode_bandwidth_package_validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
