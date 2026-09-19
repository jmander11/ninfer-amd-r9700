#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919"
PLAN = PACKAGE / "plan.json"
REPORT = PACKAGE / "qualification.json"
BUILD = PACKAGE / "build"
HIPCC = Path("/opt/rocm/bin/hipcc")


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(item: dict) -> None:
    path = Path(item["path"])
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or
            digest(path) != item["sha256"]):
        fail(f"bound input changed: {path}")


def main() -> int:
    if Path.cwd() != ROOT:
        fail(f"preflight must run from {ROOT}")
    plan = json.loads(PLAN.read_text())
    if (plan.get("schema") != "ninfer.r9700.paired-projection-c2c4-production-plan.v1" or
            plan.get("status") != "reviewed_ready" or
            plan.get("production_routing_authorized") is not False):
        fail("plan disposition differs")
    if plan.get("exact_invocation") != {
        "preflight": "bash profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919/commands.sh --preflight",
        "measure": "bash profiles/bench/r9700-paired-projection-c2c4-production-qualification-20260919/commands.sh --measure",
    }:
        fail("exact invocation differs")
    if plan.get("domain") != {
        "tokens": [2, 3, 4], "columns": 5120,
        "gdn_rows": [4096, 12288], "attention_rows": [7168, 7168],
        "qtype": "Q4G64_F16S", "layout": "Q4N16K16", "group": 64,
        "scale_dtype": "FP16", "gdn_calls": 48, "attention_calls": 16,
    }:
        fail("production domain differs")
    if plan.get("admission") != {
        "complete_three_arm_bf16_bit_exact": True,
        "independent_fp64_represented_oracle_max_bf16_steps": 2,
        "every_combined_cold_allocation_nonregressing": True,
        "minimum_credible_whole_gain_percent_each_width": 1.0,
        "passing_authorizes": "selector-off/on whole C2-C4 A/B only",
    }:
        fail("admission differs")
    if len(plan.get("bound_inputs", [])) != 12:
        fail("bound input inventory differs")
    for item in plan["bound_inputs"]:
        check(item)
    if REPORT.exists() or REPORT.is_symlink():
        fail("qualification output is not fresh")
    if BUILD.exists() or BUILD.is_symlink():
        fail("qualification build/attempt is not fresh")
    common = ["-O3", "-std=c++20", "--offload-arch=gfx1201", f"-I{ROOT / 'include'}",
              f"-I{ROOT / 'src'}", f"-I{ROOT / 'tools/r9700'}",
              "-isystem", "/opt/rocm/include"]
    define = f'-DNINFER_SOURCE_DIR="{ROOT}"'
    with tempfile.TemporaryDirectory(prefix="ninfer-paired-projection-c2c4-preflight-") as name:
        build = Path(name); binary = build / "qual"; assembly = build / "op.s"
        subprocess.run([str(HIPCC), *common, "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-function", define,
                        str(ROOT / "tools/r9700/paired_projection_c2c4_op_qual.hip"),
                        str(ROOT / "src/ops/r9700/paired_projection/paired_projection.hip"),
                        str(ROOT / "src/ops/r9700/linear/r9700_linear.hip"),
                        str(ROOT / "src/core/device.hip"), str(ROOT / "src/core/tensor.cpp"),
                        str(ROOT / "src/core/dtype.cpp"), "-L/opt/rocm/lib",
                        "-Wl,-rpath,/opt/rocm/lib", "-o", str(binary)], check=True)
        subprocess.run([str(HIPCC), *common[:-2], "-isystem", "/opt/rocm/include",
                        "-Wno-unused-command-line-argument", "--offload-device-only", "-S",
                        str(ROOT / "src/ops/r9700/paired_projection/paired_projection.hip"),
                        "-o", str(assembly)], check=True)
        receipt = subprocess.run([
            "python3", str(ROOT / "tools/r9700/check_paired_projection_c2c4_op_static.py"),
            str(assembly)], check=True, capture_output=True, text=True).stdout.strip()
        for fragment in ("paired_projection_c2c4_op_static: PASS", "vgpr=57", "sgpr=32",
                         "wave32=true", "lds=0", "scratch=0", "spills=0",
                         "native_iu4_wmma=true"):
            if fragment not in receipt:
                fail("production static receipt differs")
    print("r9700_paired_projection_c2c4_production_preflight: PASS " + receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
