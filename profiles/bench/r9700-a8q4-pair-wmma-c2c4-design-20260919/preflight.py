#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-a8q4-pair-wmma-c2c4-design-20260919"
PLAN = PACKAGE / "plan.json"
BASELINES = PACKAGE / "whole-baselines.json"
REPORT = PACKAGE / "qualification.json"
HIPCC = Path("/opt/rocm/bin/hipcc")
QUALIFIER = ROOT / "tools/r9700/a8q4_pair_wmma_c2c4_qual.hip"
CHECKER = ROOT / "tools/r9700/check_a8q4_pair_wmma_c2c4_static.py"
LINEAR = ROOT / "src/ops/r9700/linear/r9700_linear.hip"


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
    if (plan.get("schema") != "ninfer.r9700.a8q4-pair-wmma-c2c4-plan.v1" or
            plan.get("status") != "reviewed_ready" or
            plan.get("production_routing_authorized") is not False or
            len(plan.get("bound_inputs", [])) != 6):
        fail("bound plan differs")
    if plan.get("workload") != {
        "device": 0, "tokens": [2, 3, 4], "gdn_pair": [4096, 12288, 5120],
        "attention_pair": [7168, 7168, 5120], "gdn_calls_per_round": 48,
        "attention_calls_per_round": 16, "disjoint_weight_copies": 3,
        "trials_per_arm": 12, "required_power": "auto",
    }:
        fail("workload/layer accounting differs")
    if plan.get("arms") != [
        "complete_serial_two_quantize_wmma_plus_attention_extracts",
        "shared_quantize_two_production_wmma_diagnostic",
        "shared_quantize_combined_wmma_direct_candidate",
    ]:
        fail("three-arm contract differs")
    if plan.get("admission") != {
        "complete_three_arm_bf16_bit_exact": True,
        "independent_fp64_represented_oracle_max_bf16_steps": 2,
        "every_combined_cold_allocation_nonregressing": True,
        "minimum_credible_whole_gain_percent_each_width": 1.0,
        "passing_authorizes": "selector-off/on whole C2-C4 A/B only",
    }:
        fail("admission contract differs")
    if plan.get("static_contract") != {
        "architecture": "gfx1201", "native_iu4_wmma": True, "wave_size": 32,
        "maximum_vgpr": 64, "maximum_sgpr": 64, "lds_bytes": 0,
        "scratch_bytes": 0, "spill_count": 0,
    }:
        fail("static contract differs")
    if plan.get("exact_invocation") != {
        "preflight": "bash profiles/bench/r9700-a8q4-pair-wmma-c2c4-design-20260919/commands.sh --preflight",
        "measure": "bash profiles/bench/r9700-a8q4-pair-wmma-c2c4-design-20260919/commands.sh --measure",
    }:
        fail("exact invocation differs")
    if plan.get("output") != {
        "path": str(REPORT), "create_only": True,
    }:
        fail("output contract differs")
    expected_inputs = {
        QUALIFIER,
        ROOT / "tools/r9700/a8q4_shape_sweep_qual.hip",
        CHECKER,
        ROOT / "src/ops/r9700/linear/r9700_linear.h",
        LINEAR,
        ROOT / "profiles/bench/r9700-gdn-q4-pair-t1-production-confirmation-20260919/results/summary.json",
    }
    if {Path(item["path"]) for item in plan["bound_inputs"]} != expected_inputs:
        fail("bound input inventory differs")
    for item in plan["bound_inputs"]:
        check_identity(item)
    baseline = json.loads(BASELINES.read_text(encoding="utf-8"))
    if (baseline.get("schema") != "ninfer.r9700.a8q4-pair-wmma-c2c4-whole-baselines.v1" or
            baseline.get("status") != "bound_from_selector_free_production_confirmation" or
            baseline.get("tokens_per_request") != 256):
        fail("whole baseline binding differs")
    check_identity(baseline["source"])
    source = json.loads(Path(baseline["source"]["path"]).read_text(encoding="utf-8"))
    if (source.get("status") != "passed" or
            source.get("route_selection", {}).get("c2_c3_c4") != "unchanged_fallback" or
            source.get("c2_c3_c4_exact_repeat_tokens") is not True):
        fail("selector-free confirmation authority differs")
    for concurrency in (2, 3, 4):
        values = source["decode_seconds_by_concurrency"][str(concurrency)]
        if values != baseline[f"c{concurrency}_decode_seconds"]:
            fail(f"C{concurrency} decode samples differ")
        derived = statistics.median(values) * 1000.0 / baseline["tokens_per_request"]
        if derived != baseline[f"c{concurrency}_round_median_ms"]:
            fail(f"C{concurrency} round baseline differs")
    if REPORT.exists() or REPORT.is_symlink():
        fail("qualification output is not fresh")
    flags = ["-O3", "-std=c++20", "--offload-arch=gfx1201", f"-I{ROOT / 'src'}",
             "-isystem", "/opt/rocm/include"]
    define = f'-DNINFER_SOURCE_DIR="{ROOT}"'
    with tempfile.TemporaryDirectory(prefix="ninfer-pair-wmma-c2c4-preflight-") as temporary:
        build = Path(temporary)
        binary = build / "qual"
        assembly = build / "qual.s"
        subprocess.run([str(HIPCC), *flags, "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-function", define, str(QUALIFIER), str(LINEAR),
                        "-L/opt/rocm/lib", "-Wl,-rpath,/opt/rocm/lib", "-o", str(binary)],
                       check=True)
        subprocess.run([str(HIPCC), *flags, "-Wno-unused-command-line-argument", define,
                        "--offload-device-only", "-S", str(QUALIFIER), "-o", str(assembly)],
                       check=True)
        receipt = subprocess.run(["python3", str(CHECKER), str(assembly)], check=True,
                                 capture_output=True, text=True).stdout.strip()
        for required in ("a8q4_pair_wmma_c2c4_static: PASS", "wave32=true", "lds=0",
                         "scratch=0", "spills=0"):
            if required not in receipt:
                fail("static receipt differs")
    print("r9700_a8q4_pair_wmma_c2c4_preflight: PASS " + receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
