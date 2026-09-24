#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919"
PLAN = PACKAGE / "plan.json"
ATTEMPT = PACKAGE / "attempt-1"
REPORT = ATTEMPT / "qualification.json"
HIPCC = Path("/opt/rocm/bin/hipcc")
QUALIFIER = ROOT / "tools/r9700/a8q4_projected_residual_t1_qual.hip"
CHECKER = ROOT / "tools/r9700/check_a8q4_projected_residual_t1_static.py"
LINEAR = ROOT / "src/ops/r9700/linear/r9700_linear.hip"
EAGER = ROOT / "src/ops/r9700/eager/eager_ops.hip"


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_identity(item: dict) -> None:
    path = Path(item["path"])
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or
            digest(path) != item["sha256"]):
        fail(f"bound identity changed: {path}")


def main() -> int:
    if Path.cwd() != ROOT:
        fail(f"preflight must run from {ROOT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (plan.get("schema") != "ninfer.r9700.a8q4-projected-residual-t1-plan.v1" or
            plan.get("status") != "reviewed_ready" or
            plan.get("production_routing_authorized") is not False):
        fail("plan disposition differs")
    if plan.get("compile_contract") != {
        "architecture": "gfx1201", "optimization": "O3", "language": "c++20",
        "hipcc": str(HIPCC), "source_root": str(ROOT),
    }:
        fail("compile contract differs")
    if plan.get("workload") != {
        "device": 0, "tokens": 1, "rows": 5120, "columns": [6144, 17408],
        "calls_per_shape_per_token": 64, "disjoint_weight_copies": 3,
        "cache_scrub_bytes_before_each_arm": 83886080, "trials_per_arm": 12,
        "required_power": "auto",
    }:
        fail("workload contract differs")
    if plan.get("timing_design") != {
        "joint_allocation_order_balanced": True,
        "samples_per_arm_per_copy": 4,
        "first_position_samples_per_arm_per_copy": 2,
    }:
        fail("timing design differs")
    if plan.get("semantic_boundary") != \
            "delta=BF16(dot); residual=BF16(FP32(residual)+FP32(delta))":
        fail("semantic boundary differs")
    if plan.get("admission") != {
        "complete_residual_bit_exact": True,
        "independent_fp64_represented_weight_oracle_max_bf16_steps": 2,
        "maximum_candidate_over_incumbent_ratio_each_allocation": 1.01,
        "minimum_aggregate_saved_ms_per_token": 0.2,
        "aggregate_formula": "64*sum(incumbent_shape_median_ms-candidate_shape_median_ms)",
        "passing_authorizes": "selector-off/on whole C1 A/B only",
    }:
        fail("admission contract differs")
    if plan.get("static_contract") != {
        "architecture": "gfx1201", "native_iu4_dot8": True, "wave_size": 32,
        "maximum_vgpr": 32, "maximum_sgpr": 64, "lds_bytes": 0,
        "scratch_bytes": 0, "spill_count": 0,
    }:
        fail("static contract differs")
    if plan.get("exact_invocation") != {
        "preflight": "bash profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/commands.sh --preflight",
        "measure": "bash profiles/bench/r9700-a8q4-projected-residual-t1-design-20260919/commands.sh --measure",
    }:
        fail("exact invocation differs")
    if plan.get("output") != {"path": str(REPORT), "create_only": True}:
        fail("output contract differs")
    if len(plan.get("bound_inputs", [])) != 11:
        fail("bound input inventory differs")
    for item in plan["bound_inputs"]:
        check_identity(item)
    if ATTEMPT.exists() or ATTEMPT.is_symlink():
        fail("immutable attempt path is not fresh")

    common = ["-O3", "-std=c++20", "--offload-arch=gfx1201", f"-I{ROOT / 'src'}",
              "-isystem", "/opt/rocm/include"]
    define = f'-DNINFER_SOURCE_DIR="{ROOT}"'
    with tempfile.TemporaryDirectory(prefix="ninfer-projected-residual-t1-preflight-") as temporary:
        build = Path(temporary)
        binary = build / "qual"
        assembly = build / "qual.s"
        subprocess.run([
            str(HIPCC), *common, "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
            define, str(QUALIFIER), str(LINEAR), str(EAGER), "-L/opt/rocm/lib",
            "-Wl,-rpath,/opt/rocm/lib", "-o", str(binary),
        ], check=True)
        subprocess.run([
            str(HIPCC), *common, "-Wno-unused-command-line-argument", define,
            "--offload-device-only", "-S", str(QUALIFIER), "-o", str(assembly),
        ], check=True)
        receipt = subprocess.run(
            ["python3", str(CHECKER), str(assembly)], check=True,
            capture_output=True, text=True).stdout.strip()
        for required in ("a8q4_projected_residual_t1_static: PASS", "wave32=true",
                         "lds=0", "scratch=0", "spills=0"):
            if required not in receipt:
                fail("static receipt differs")
    print("r9700_a8q4_projected_residual_t1_preflight: PASS " + receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
