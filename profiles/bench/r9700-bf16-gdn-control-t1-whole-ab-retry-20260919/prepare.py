#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
PACKAGE = Path(__file__).resolve().parent
PACKAGE_PLAN = PACKAGE / "package-plan.json"
PLAN = PACKAGE / "plan.json"
RESULTS = PACKAGE / "results"
PRIOR_PACKAGE = ROOT / "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919"
PRIOR_RUN = PRIOR_PACKAGE / "run.py"


def fail(message: str) -> None:
    raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))
    if not isinstance(value, dict):
        fail(f"not an object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def require_identity(record: dict, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        fail(f"{label} identity shape differs")
    path = Path(record["path"])
    if record != identity(path):
        fail(f"{label} identity differs")
    return path


def cache(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if (not line or line.startswith(("#", "//")) or "=" not in line or
                ":" not in line.split("=", 1)[0]):
            continue
        lhs, value = line.split("=", 1)
        values[lhs.split(":", 1)[0]] = value
    return values


def validate_reviewed_plan(path: Path) -> tuple[dict, Path]:
    plan = load(path)
    if (plan.get("schema") != "ninfer.r9700.bf16-gdn-control-t1-whole-ab-plan.v1" or
            plan.get("status") != "prepared" or
            plan.get("production_routing_authorized") is not False or
            plan.get("order") !=
            ["control", "candidate", "candidate", "control", "control", "candidate"] or
            plan.get("workload") != {
                "device": 0, "concurrency": 1, "whole_pg": "8192,256",
                "prefill_chunk": 4096, "spec": "none", "draft_tokens": 0,
                "device_graph": True, "retain_token_ids": True, "required_power": "auto"} or
            plan.get("admission") != {
                "exact_public_tokens": True, "every_paired_ratio_below_one": True,
                "paired_mean_upper_2se_below_one": True,
                "median_candidate_over_control_at_most": 0.99}):
        fail("reviewed plan contract differs")
    sources = plan.get("sources")
    expected_source_paths = {str((ROOT / relative).resolve()) for relative in (
        "CMakeLists.txt", "include/ninfer/ops/gdn_gating.h",
        "src/ops/r9700/gdn/gdn_gating.cpp", "src/ops/r9700/gdn/gdn_ops.h",
        "src/ops/r9700/gdn/gdn_ops.hip", "src/targets/qwen3_8_27b/impl/variant.cpp",
        "bench/targets/qwen3_8_27b/ninfer_bench_support.cpp",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/package-plan.json",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/commands.sh",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/prepare.py",
        "profiles/bench/r9700-bf16-gdn-control-t1-whole-ab-20260919/run.py")}
    if (not isinstance(sources, list) or
            {record.get("path") for record in sources} != expected_source_paths or
            any(set(record) != {"path", "bytes", "sha256"} for record in sources)):
        fail("reviewed historical source inventory differs")
    direct_path = require_identity(plan.get("direct_qualification"), "direct qualification")
    direct = load(direct_path)
    if (direct.get("schema") !=
            "ninfer.r9700.bf16-gdn-control-t1-production-qualification.v1" or
            direct.get("status") != "qualified_for_whole_ab_only" or
            direct.get("production_routing_authorized") is not False):
        fail("direct qualification status differs")
    required_direct = {"executable", "assembly", "static_receipt", "stdout", "stderr",
                       "device_stdout", "device_stderr", "device_process", "device_identity",
                       "qualification_process", "kernel_source", "semantic_wrapper",
                       "qualifier", "checker", "runner", "commands", "writer", "plan"}
    identities = direct.get("identities")
    if not isinstance(identities, dict) or set(identities) != required_direct:
        fail("direct qualification identity inventory differs")
    for label, record in identities.items():
        require_identity(record, f"direct qualification {label}")
    for label in ("artifact", "corpus", "retained_token_authority"):
        require_identity(plan.get(label), label)
    expected_tokens = plan.get("expected_tokens", {})
    if expected_tokens.get("count") != 257 or not isinstance(expected_tokens.get("sha256"), str):
        fail("retained token authority differs")
    caches = {}
    for role in ("control", "candidate"):
        build = plan.get("builds", {}).get(role, {})
        cache_path = require_identity(build.get("cache"), f"{role} cache")
        executable = require_identity(build.get("executable"), f"{role} executable")
        symbol = build.get("symbol_receipt", {})
        obj = require_identity(symbol.get("object"), f"{role} selector object")
        directory = Path(build.get("directory", ""))
        if executable.parent.parent != directory:
            fail(f"{role} executable is outside reviewed build")
        values = cache(cache_path)
        selected = {key: values.get(key, "") for key in build.get("cache_values", {})}
        if selected != build.get("cache_values"):
            fail(f"{role} execution cache changed")
        output = subprocess.run(["nm", "-C", str(obj)], check=True,
                                capture_output=True, text=True).stdout
        present = " U ninfer::ops::bf16_gdn_projected_gating_t1(" in output
        if (present is not symbol.get("projected_control_undefined_reference") or
                present is not (role == "candidate")):
            fail(f"{role} selector symbol differs")
        caches[role] = selected
    for key in caches["control"]:
        if (key != "NINFER_R9700_BF16_GDN_CONTROL_T1_CANDIDATE" and
                caches["control"][key] != caches["candidate"].get(key)):
            fail(f"reviewed builds differ at {key}")
    required_cache = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
        "NINFER_BUILD_APPS": "ON", "NINFER_BUILD_BENCHMARKS": "ON",
        "NINFER_R9700_DFLASH_DOWN_SPLITK_FACTOR": "8",
        "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
        "NINFER_R9700_XATTENTION_STRIDE": "16",
        "NINFER_R9700_XATTENTION_TAU_PERMILLE": "1000"}
    for role, values in caches.items():
        if any(values.get(key) != value for key, value in required_cache.items()):
            fail(f"{role} required execution cache differs")
    _, _, pci = vram_paths(direct)
    profile = (Path("/sys/bus/pci/devices") / pci /
               "power_dpm_force_performance_level").resolve(strict=True)
    if profile.read_text().strip() != "auto":
        fail("qualified device power profile is not auto")
    return plan, profile


def validate_manifest(path: Path) -> None:
    records = {}
    for line in path.read_text().splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            fail("retained result manifest syntax differs")
        checksum, name = fields
        if (not name or name in records or Path(name).name != name or
                any(character not in "0123456789abcdef" for character in checksum)):
            fail("retained result manifest entry differs")
        records[name] = checksum
    expected = {
        "run-1-control.json", "run-1-control.process.json", "run-1-control.stderr",
        "run-1-control.stdout", "run-2-candidate.json", "run-2-candidate.process.json",
        "run-2-candidate.stderr", "run-2-candidate.stdout",
        "run-3-candidate.process.json", "run-3-candidate.stderr",
        "run-3-candidate.stdout",
    }
    if set(records) != expected:
        fail("retained result manifest inventory differs")
    for name, checksum in records.items():
        if digest(path.parent / name) != checksum:
            fail(f"retained failed result changed: {name}")


def vram_paths(direct: dict) -> tuple[Path, Path, str]:
    device = direct.get("device", {})
    if (device.get("ordinal") != 0 or device.get("name") != "AMD Radeon AI PRO R9700" or
            device.get("architecture") != "gfx1201"):
        fail("direct qualification device identity differs")
    pci = device.get("pci_bus_id")
    if not isinstance(pci, str) or pci != pci.lower() or len(pci) != 12:
        fail("direct qualification PCI identity differs")
    pci_root = (Path("/sys/bus/pci/devices") / pci).resolve(strict=True)
    expected_power = (pci_root / "power_dpm_force_performance_level").resolve(strict=True)
    if device.get("power_profile_path") != str(expected_power):
        fail("power path is not derived from the qualified device PCI")
    return ((pci_root / "mem_info_vram_used").resolve(strict=True),
            (pci_root / "mem_info_vram_total").resolve(strict=True), pci)


def read_vram(direct: dict, contract: dict, require_low: bool = True) -> dict:
    expected = {"source", "maximum_used_bytes", "maximum_used_fraction_permille",
                "minimum_total_bytes"}
    if set(contract) != expected or contract.get("source") != \
            "PCI-derived amdgpu sysfs mem_info_vram_used/mem_info_vram_total":
        fail("VRAM gate contract differs")
    used_path, total_path, pci = vram_paths(direct)
    try:
        used = int(used_path.read_text().strip())
        total = int(total_path.read_text().strip())
    except ValueError as error:
        raise RuntimeError("amdgpu VRAM readout is not an integer") from error
    if used < 0 or total < contract["minimum_total_bytes"] or used > total:
        fail("amdgpu VRAM readout is implausible")
    fraction_permille = (1000 * used + total - 1) // total
    passed = (used <= contract["maximum_used_bytes"] and
              fraction_permille <= contract["maximum_used_fraction_permille"])
    receipt = {"pci_bus_id": pci, "used_path": str(used_path),
               "total_path": str(total_path), "used_bytes": used,
               "total_bytes": total, "used_fraction_permille_ceil": fraction_permille,
               "maximum_used_bytes": contract["maximum_used_bytes"],
               "maximum_used_fraction_permille": contract["maximum_used_fraction_permille"],
               "passed": passed}
    if require_low and not passed:
        fail(f"device-0 VRAM is not reset-clean: used={used} total={total}; "
             f"require <= {contract['maximum_used_bytes']} bytes and "
             f"<= {contract['maximum_used_fraction_permille']} permille")
    return receipt


def validate_authorities(require_fresh: bool, require_low_vram: bool = True) -> tuple[dict, dict, dict, dict]:
    if Path.cwd() != ROOT:
        fail(f"run from {ROOT}")
    package = load(PACKAGE_PLAN)
    if (package.get("schema") !=
            "ninfer.r9700.bf16-gdn-control-t1-whole-ab-retry-package.v1" or
            package.get("status") != "reviewed_ready_after_gpu_reset" or
            package.get("production_routing_authorized") is not False or
            package.get("order") !=
            ["control", "candidate", "candidate", "control", "control", "candidate"]):
        fail("retry package disposition differs")
    reviewed_plan_path = require_identity(package.get("reviewed_plan"), "reviewed plan")
    failure = package.get("retained_failure")
    if not isinstance(failure, dict) or set(failure) != {
            "result_manifest", "failure_process", "failure_stderr"}:
        fail("retained failure inventory differs")
    manifest_path = require_identity(failure["result_manifest"], "retained result manifest")
    process_path = require_identity(failure["failure_process"], "retained failure process")
    stderr_path = require_identity(failure["failure_stderr"], "retained failure stderr")
    validate_manifest(manifest_path)
    process = load(process_path)
    if (process.get("stem") != "run-3-candidate" or process.get("role") != "candidate" or
            process.get("exit_code") != 1 or process.get("benchmark_report") is not None or
            process.get("stderr") != identity(stderr_path)):
        fail("retained OOM process receipt differs")
    if "hipMalloc arena: hipErrorOutOfMemory" not in stderr_path.read_text():
        fail("retained failure is not the reviewed arena OOM")
    reviewed_plan, profile = validate_reviewed_plan(reviewed_plan_path)
    direct_path = Path(reviewed_plan["direct_qualification"]["path"])
    direct = load(direct_path)
    vram = read_vram(direct, package.get("vram_gate", {}), require_low_vram)
    if profile.read_text().strip() != "auto":
        fail("qualified device power profile is not auto")
    if require_fresh and (PLAN.exists() or PLAN.is_symlink() or RESULTS.exists() or RESULTS.is_symlink()):
        fail("retry plan/results are not fresh; never resume or overwrite")
    return package, reviewed_plan, direct, vram


def write_plan() -> None:
    package, reviewed, _, vram = validate_authorities(True, True)
    sources = [PACKAGE_PLAN, PACKAGE / "commands.sh", PACKAGE / "prepare.py", PACKAGE / "run.py"]
    plan = {"schema": "ninfer.r9700.bf16-gdn-control-t1-whole-ab-retry-plan.v1",
            "status": "prepared", "production_routing_authorized": False,
            "reviewed_plan": package["reviewed_plan"],
            "retained_failure": package["retained_failure"],
            "sources": [identity(path) for path in sources],
            "workload": reviewed["workload"], "order": reviewed["order"],
            "builds": reviewed["builds"],
            "direct_qualification": reviewed["direct_qualification"],
            "artifact": reviewed["artifact"], "corpus": reviewed["corpus"],
            "retained_token_authority": reviewed["retained_token_authority"],
            "expected_tokens": reviewed["expected_tokens"],
            "admission": reviewed["admission"],
            "vram_gate": package["vram_gate"], "preflight_vram": vram,
            "campaign": {"fresh_complete_six_role_campaign": True,
                         "partial_pair_resume_allowed": False}}
    descriptor = os.open(PLAN, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print("r9700_bf16_gdn_control_t1_whole_ab_retry_prepare: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--write-plan", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _, _, _, vram = validate_authorities(True, True)
        print("r9700_bf16_gdn_control_t1_whole_ab_retry_preflight: PASS "
              f"used={vram['used_bytes']} total={vram['total_bytes']}")
    else:
        write_plan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
