#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-attention-projection-t1-production-qualification-20260919"
PLAN = PACKAGE / "plan.json"
REPORT = PACKAGE / "qualification.json"
HIPCC = Path("/opt/rocm/bin/hipcc")


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    if Path.cwd() != ROOT:
        fail(f"preflight must run from {ROOT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (plan.get("schema") != "ninfer.r9700.attention-projection-t1-production-plan.v1" or
            plan.get("status") != "prepared_awaiting_independent_review" or
            plan.get("production_routing_authorized") is not False):
        fail("plan disposition differs")
    if plan.get("exact_invocation") != {
        "preflight": "bash profiles/bench/r9700-attention-projection-t1-production-qualification-20260919/commands.sh --preflight",
        "measure": "bash profiles/bench/r9700-attention-projection-t1-production-qualification-20260919/commands.sh --measure",
    }:
        fail("exact invocation differs")
    if plan.get("domain") != {
        "tokens": 1, "columns": 5120, "matrix_rows": [7168, 7168],
        "outputs": [6144, 1024, 6144, 1024], "qtype": "Q4G64_F16S",
        "layout": "Q4N16K16", "group": 64, "scale_dtype": "FP16",
    }:
        fail("exact domain differs")
    for item in plan.get("bound_inputs", []):
        path = Path(item["path"])
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or
                digest(path) != item["sha256"]):
            fail(f"bound input changed: {path}")
    if len(plan.get("bound_inputs", [])) != 8:
        fail("bound input inventory differs")
    if REPORT.exists() or REPORT.is_symlink():
        fail("qualification output is not fresh")
    with tempfile.TemporaryDirectory(prefix="ninfer-attention-projection-t1-preflight-") as name:
        build = Path(name)
        binary = build / "qual"
        assembly = build / "op.s"
        common = ["-O3", "-std=c++20", "--offload-arch=gfx1201",
                  f"-I{ROOT / 'include'}", f"-I{ROOT / 'src'}",
                  f"-I{ROOT / 'tools/r9700'}", "-isystem", "/opt/rocm/include"]
        subprocess.run([
            str(HIPCC), *common, "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
            f'-DNINFER_SOURCE_DIR="{ROOT}"',
            str(ROOT / "tools/r9700/attention_projection_t1_op_qual.hip"),
            str(ROOT / "src/ops/r9700/attention_projection/attention_projection.hip"),
            str(ROOT / "src/ops/r9700/linear/r9700_linear.hip"),
            str(ROOT / "src/core/device.hip"), str(ROOT / "src/core/tensor.cpp"),
            str(ROOT / "src/core/dtype.cpp"), "-L/opt/rocm/lib",
            "-Wl,-rpath,/opt/rocm/lib", "-o", str(binary),
        ], check=True)
        subprocess.run([
            str(HIPCC), *common[:-2], "-isystem", "/opt/rocm/include",
            "-Wno-unused-command-line-argument", "--offload-device-only", "-S",
            str(ROOT / "src/ops/r9700/attention_projection/attention_projection.hip"),
            "-o", str(assembly),
        ], check=True)
        receipt = subprocess.run([
            "python3", str(ROOT / "tools/r9700/check_attention_projection_t1_op_static.py"),
            str(assembly),
        ], check=True, capture_output=True, text=True).stdout.strip()
        for fragment in ("attention_projection_t1_op_static: PASS", "vgpr=20", "sgpr=24",
                         "wave32=true", "lds=0", "scratch=0", "spills=0",
                         "native_mixed_iu4_dot8=true"):
            if fragment not in receipt:
                fail("static receipt differs")
    print("r9700_attention_projection_t1_production_preflight: PASS " + receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
