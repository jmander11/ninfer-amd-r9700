#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def cache_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("//", "#")):
            values[line.split(":", 1)[0]] = line.split("=", 1)[1]
    return values


def write_exclusive(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main() -> int:
    bound = PACKAGE / "bound-plan.json"
    if bound.exists() or bound.is_symlink():
        fail("bound-plan.json already exists; never overwrite a prepared package")
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        fail("tracked source tree is dirty; bind a committed source identity")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    short = commit[:8]
    source = Path(f"/ssdpool2nvme/local_llm/ninfer-amd-r9700-base-decode-source-{short}")
    build = ROOT / f"build-r9700-base-decode-bandwidth-{short}-20260919"
    tool_build = ROOT / f"build-r9700-base-decode-tools-{short}-20260919"
    for path in (source, build, tool_build):
        if path.exists() or path.is_symlink():
            fail(f"fresh build path required: {path}")
    run(["git", "worktree", "add", "--detach", str(source), commit], cwd=ROOT)
    common = [
        "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_HIP_ARCHITECTURES=gfx1201",
        "-DGPU_BUILD_TARGETS=gfx1201", "-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++",
        "-DNINFER_BUILD_APPS=OFF", "-DNINFER_BUILD_BENCHMARKS=ON",
        "-DNINFER_BUILD_R9700_CORE_QUALIFIER=ON", "-DNINFER_R9700_KV_VALUE_GROUP=16",
        "-DNINFER_R9700_Q4_ACTIVATION_BITS=8", "-DNINFER_R9700_W8_ACTIVATION_BITS=8",
        "-DNINFER_R9700_FP8_QK_WMMA=1", "-DNINFER_R9700_XATTENTION_QUALIFICATION=OFF",
        "-DNINFER_R9700_DFLASH_SMALL_T_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_DOWN_SPLITK_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=0",
        "-DNINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE=0",
        "-DNINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE=0",
        "-DNINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE=0",
    ]
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "-1"
    run(["cmake", "-S", str(source), "-B", str(build), *common], cwd=ROOT)
    run(["cmake", "--build", str(build), "--target", "ninfer_bench",
         "ninfer_bench_support_test", "-j2"], cwd=ROOT)
    run([str(build / "tests/ninfer_bench_support_test")], cwd=ROOT, env=env)
    targets = [str(tool_build / "hbm_bandwidth_probe")]
    run(["make", "-C", str(source / "tools/r9700"), f"BUILD_DIR={tool_build}", "-j2", *targets], cwd=ROOT)
    expected = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
        "GPU_BUILD_TARGETS": "gfx1201", "NINFER_BUILD_BENCHMARKS": "ON",
        "NINFER_R9700_Q4_ACTIVATION_BITS": "8", "NINFER_R9700_W8_ACTIVATION_BITS": "8",
        "NINFER_R9700_KV_VALUE_GROUP": "16", "NINFER_R9700_FP8_QK_WMMA": "1",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
        "NINFER_R9700_DFLASH_SMALL_T_CANDIDATE": "0",
        "NINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE": "0",
        "NINFER_R9700_DFLASH_DOWN_SPLITK_CANDIDATE": "0",
        "NINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE": "0",
        "NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE": "0",
        "NINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE": "0",
        "NINFER_R9700_FP8_PREFIX_COMMON_ALGO_CANDIDATE": "0",
    }
    values = cache_values(build / "CMakeCache.txt")
    for key, value in expected.items():
        if values.get(key) != value:
            fail(f"selector-free build differs: {key}={values.get(key)!r}")
    retained_report_path = ROOT / "profiles/bench/r9700-base-decode-dot8-cold-block-sweep-20260919.json"
    token_authority_path = ROOT / "profiles/bench/r9700-rmsnorm-production-final-p8192-g256-c1-20260906.json"
    token_authority = json.loads(token_authority_path.read_text())
    expected_tokens = token_authority["tests"][0]["reps"][0]["generated_token_ids_by_lane"][0]
    if len(expected_tokens) != 257:
        fail("retained selector-free token authority has the wrong geometry")
    token_bytes = json.dumps(expected_tokens, separators=(",", ":")).encode()
    plan = json.loads((PACKAGE / "plan.json").read_text())
    plan.update({
        "status": "prepared_awaiting_independent_review_and_gpu_execution",
        "source_commit": commit, "source_tree": tree,
        "source_worktree": str(source), "build_directory": str(build),
        "build_cache": identity(build / "CMakeCache.txt"),
        "benchmark": identity(build / "bench/ninfer_bench"),
        "stream_probe": identity(tool_build / "hbm_bandwidth_probe"),
        "artifact": identity(ROOT / "out/qwen3.8-27b-r9700-q4g64-n16k16-eval.ninfer"),
        "corpus": identity(ROOT / "bench/fixtures/bench_corpus.ids"),
        "profiler": identity(Path("/opt/rocm/core-10.0/bin/rocprofv3")),
        "counter_preflight": identity(Path("/opt/rocm/core-10.0/bin/rocprofv3-avail")),
        "retained_cold_block_sweep": identity(retained_report_path),
        "retained_token_authority": identity(token_authority_path),
        "expected_tokens": {"count": len(expected_tokens),
                            "sha256": hashlib.sha256(token_bytes).hexdigest()},
        "selector_free_cache": expected,
    })
    write_exclusive(bound, plan)
    print(f"base decode package bound to {commit}; independent review is required before GPU execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
