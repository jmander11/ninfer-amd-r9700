#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from prepare import (ATTEMPT, ORIGINAL, PACKAGE, PLAN, ROOT, SCRIPT_INPUTS,
                     SOURCE_INPUTS, compile_commands, identity, require_environment)


def verify() -> dict:
    require_environment()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    original = json.loads((ORIGINAL / "plan.json").read_text(encoding="utf-8"))
    if (plan.get("schema") != "ninfer.r9700.a8q4-projected-residual-t1-retry-plan.v1" or
            plan.get("status") != "prepared_for_review" or
            plan.get("production_routing_authorized") is not False):
        raise RuntimeError("retry disposition differs")
    for field in ("compile_contract", "workload", "timing_design", "semantic_boundary",
                  "admission", "static_contract"):
        if plan.get(field) != original[field]:
            raise RuntimeError(f"retry changed original contract: {field}")
    expected = [str(path) for path in SOURCE_INPUTS + SCRIPT_INPUTS]
    if [item["path"] for item in plan.get("bound_inputs", [])] != expected:
        raise RuntimeError("retry source/script inventory differs")
    prior = [str(path) for path in sorted(ORIGINAL.rglob("*")) if path.is_file()]
    if [item["path"] for item in plan.get("original_evidence", [])] != prior:
        raise RuntimeError("original evidence inventory changed")
    for item in plan["bound_inputs"] + plan["original_evidence"]:
        if identity(Path(item["path"])) != item:
            raise RuntimeError(f"bound identity changed: {item['path']}")
    relative = PACKAGE.relative_to(ROOT)
    if plan.get("exact_invocation") != {
            name: f"bash {relative}/commands.sh --{name}" for name in ("preflight", "measure")}:
        raise RuntimeError("retry invocation differs")
    if (plan.get("output") != {"path": str(ATTEMPT / "qualification.json"), "create_only": True} or
            plan.get("attempt_closure") != str(ATTEMPT / "closure.json")):
        raise RuntimeError("retry output differs")
    if ATTEMPT.exists() or ATTEMPT.is_symlink():
        raise RuntimeError("immutable retry attempt is not fresh")
    return plan


def main() -> int:
    verify()
    with tempfile.TemporaryDirectory(prefix="ninfer-projected-residual-retry-preflight-") as temporary:
        host, device, checker = compile_commands(Path(temporary))
        subprocess.run(host, cwd=ROOT, check=True)
        subprocess.run(device, cwd=ROOT, check=True)
        receipt = subprocess.run(checker, cwd=ROOT, check=True, capture_output=True,
                                 text=True).stdout.strip()
        for required in ("a8q4_projected_residual_t1_static: PASS", "wave32=true",
                         "lds=0", "scratch=0", "spills=0"):
            if required not in receipt:
                raise RuntimeError("retry static receipt differs")
    print("projected_residual_retry_preflight: PASS " + receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
