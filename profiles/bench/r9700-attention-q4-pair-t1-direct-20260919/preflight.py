#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = ROOT / "profiles/bench/r9700-attention-q4-pair-t1-direct-20260919"
PLAN = PACKAGE / "plan.json"
REPORT = PACKAGE / "qualification.json"
HIPCC = Path("/opt/rocm/bin/hipcc")
QUALIFIER = ROOT / "tools/r9700/a8q4_attention_pair_t1_qual.hip"
CHECKER = ROOT / "tools/r9700/check_a8q4_attention_pair_t1_static.py"
LINEAR_IMPL = ROOT / "src/ops/r9700/linear/r9700_linear.hip"

HOST_FLAGS = [
    "-O3", "-std=c++20", "--offload-arch=gfx1201", f"-I{ROOT / 'src'}",
    "-isystem", "/opt/rocm/include", "-Wall", "-Wextra", "-Werror",
    "-Wno-unused-function", f'-DNINFER_SOURCE_DIR="{ROOT}"',
]
DEVICE_FLAGS = [
    "-O3", "-std=c++20", "--offload-arch=gfx1201", f"-I{ROOT / 'src'}",
    "-isystem", "/opt/rocm/include", "-Wno-unused-command-line-argument",
    f'-DNINFER_SOURCE_DIR="{ROOT}"', "--offload-device-only", "-S",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_identity(item: dict) -> None:
    path = Path(item["path"])
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or
            digest(path) != item["sha256"]):
        fail(f"bound input changed: {path}")


def main() -> int:
    if Path.cwd() != ROOT:
        fail(f"preflight must run from {ROOT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (plan.get("schema") != "ninfer.r9700.a8q4-attention-pair-t1-direct-plan.v1" or
            plan.get("status") != "prepared_awaiting_independent_review" or
            plan.get("production_routing_authorized") is not False):
        fail("plan disposition differs")
    if plan.get("exact_invocation") != {
        "preflight": "bash profiles/bench/r9700-attention-q4-pair-t1-direct-20260919/commands.sh --preflight",
        "measure": "bash profiles/bench/r9700-attention-q4-pair-t1-direct-20260919/commands.sh --measure",
    }:
        fail("exact invocation differs")
    if plan.get("compile_contract") != {
        "architecture": "gfx1201", "optimization": "O3", "language": "c++20",
        "hipcc": str(HIPCC), "source_root": str(ROOT),
    }:
        fail("compile contract differs")
    bound = plan.get("bound_inputs", [])
    if len(bound) != 5:
        fail("bound input inventory differs")
    for item in bound:
        check_identity(item)
    if REPORT.exists() or REPORT.is_symlink():
        fail("qualification output is not fresh")

    # Disposable compilation performs no HIP device call and leaves no package output.
    with tempfile.TemporaryDirectory(prefix="ninfer-attention-pair-preflight-") as temporary:
        build = Path(temporary)
        binary = build / "a8q4_attention_pair_t1_qual"
        assembly = build / "a8q4_attention_pair_t1_qual.s"
        subprocess.run([
            str(HIPCC), *HOST_FLAGS, str(QUALIFIER), str(LINEAR_IMPL),
            "-L/opt/rocm/lib", "-Wl,-rpath,/opt/rocm/lib", "-o", str(binary),
        ], check=True)
        subprocess.run([
            str(HIPCC), *DEVICE_FLAGS, str(QUALIFIER), "-o", str(assembly),
        ], check=True)
        static = subprocess.run(
            ["python3", str(CHECKER), str(assembly)], check=True,
            capture_output=True, text=True).stdout.strip()
        required = ("a8q4_attention_pair_t1_static: PASS", "wave32=true", "lds=0",
                    "scratch=0", "spills=0")
        if not all(fragment in static for fragment in required):
            fail("static receipt differs")
    print("r9700_attention_q4_pair_t1_direct_preflight: PASS " + static)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
