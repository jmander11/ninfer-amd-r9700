#!/usr/bin/env python3
"""Fail-closed analyzer for the two-cell real-weight FP8 prefix discriminator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("/ssdpool2nvme/local_llm/ninfer-amd-r9700")
RESULTS = PACKAGE / "results"
EXE = PACKAGE / "retained-bin/fp8_linear_prefix_discriminator"
ARTIFACT = ROOT / "out/qwen3.8-27b-r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval.ninfer"
COMMIT = "408bf59bcedda20d1294e613714fe81c43545f02"
TREE = "b286ec6984f4b10e42a551a26b1e8eba1fc0ff15"


def fail(message: str) -> None:
    raise RuntimeError(message)


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    if not isinstance(value, dict):
        fail(f"not object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path":str(resolved),"bytes":resolved.stat().st_size,"sha256":digest(resolved)}


def expected_cell(name: str) -> dict:
    common_oracle = {"probe_count":28,"maximum_bf16_steps":0,
        "criterion":"abs_error <= BF16_rounding_error + gamma_5120*sum_abs; ratio <= 1.05"}
    common_profiles = {
        128:{"tokens":128,"activation_workspace_bytes":655876,
            "selected_matmul_workspace_bytes":0,"algorithm_max_workspace_bytes":536870912,
            "selected_heuristic_rank":0,"heuristic_result_count":8,"waves_count":0},
        129:{"tokens":129,"activation_workspace_bytes":661252,
            "selected_matmul_workspace_bytes":0,"algorithm_max_workspace_bytes":536870912,
            "selected_heuristic_rank":0,"heuristic_result_count":8,"waves_count":0}}
    if name == "text/layers/0/mlp/gate_up":
        profiles = [{**common_profiles[128],"algorithm_fingerprint":"e2e00100000000000000000000000000"},
                    {**common_profiles[129],"algorithm_fingerprint":"dde00100000000000000000000000000"}]
        return {"weight_name":name,"rows":34816,"columns":5120,
            "activation_code_prefix_exact":True,"activation_scale_prefix_exact":True,
            "bf16_output_prefix_exact":False,"bf16_output_prefix_mismatch_count":291,
            "first_output_mismatch":{"flat_index":2231847,"token":64,"row":3623,
                "p128_bits":14790,"p129_bits":14789},
            "represented_fp64_oracle":{**common_oracle,
                "maximum_error_bound_ratio":0.21287897269736736},"profiles":profiles}
    if name == "text/layers/0/gdn/query_key":
        profiles = [{**common_profiles[token],"algorithm_fingerprint":"dde00100000000000000000000000000"}
                    for token in (128,129)]
        return {"weight_name":name,"rows":4096,"columns":5120,
            "activation_code_prefix_exact":True,"activation_scale_prefix_exact":True,
            "bf16_output_prefix_exact":True,"bf16_output_prefix_mismatch_count":0,
            "first_output_mismatch":None,
            "represented_fp64_oracle":{**common_oracle,
                "maximum_error_bound_ratio":0.22709865698681717},"profiles":profiles}
    fail("unknown cell")


def validate_result(value: dict) -> None:
    expected = {"artifact_type":"ninfer_r9700_fp8_linear_prefix_discriminator_result",
        "schema_version":1,"diagnostic_only":True,"timing_evidence_eligible":False,
        "production_routing_authorized":False,
        "artifact":{"path":str(ARTIFACT),"bytes":22763026944,
            "sha256":"d8fc77c36cf17c92e96d67b9a6b5a1826a1ade4f59d59c003b2368fe981fc512",
            "model_id":"qwen3.8-27b",
            "weights_id":"r9700-q4g64-f8e4m3-four-role-n16k16-dflash2-q4-eval"},
        "device":{"ordinal":0,"name":"AMD Radeon AI PRO R9700",
            "architecture":"gfx1201","wave_size":32},
        "input_contract":"one identical represented-BF16 128-column prefix plus one appended column; independent per-token E4M3 outer scale",
        "cells":[expected_cell("text/layers/0/mlp/gate_up"),
                 expected_cell("text/layers/0/gdn/query_key")],
        "conclusion":"activation prefixes are exact but at least one hipBLASLt BF16 output prefix differs"}
    if value != expected:
        fail("result contract differs")


def validate_prepared() -> tuple[dict, dict]:
    plan = load(PACKAGE / "plan.json")
    build = load(PACKAGE / "build-provenance.json")
    if (plan.get("artifact_type") != "ninfer_r9700_fp8_prefix_real_weight_plan" or
            plan.get("schema_version") != 1 or
            plan.get("status") != "prepared_exact_rerun_required_for_process_and_auto_provenance" or
            plan.get("source") != {"commit":COMMIT,"tree":TREE} or
            plan.get("production_routing_authorized") is not False or
            plan.get("timing_evidence_eligible") is not False or
            plan.get("results_directory") != "results"):
        fail("plan identity differs")
    if (build.get("artifact_type") != "ninfer_r9700_fp8_prefix_retained_build_receipt" or
            build.get("schema_version") != 1 or build.get("source") != plan["source"] or
            build.get("outputs",{}).get("executable") != plan["executable"]):
        fail("build receipt differs")
    if subprocess.check_output(["git","rev-parse",f"{COMMIT}^{{tree}}"],cwd=ROOT,
                               text=True).strip() != TREE:
        fail("source tree differs")
    for relative, expected in build["source_files"].items():
        blob = subprocess.check_output(["git","show",f"{COMMIT}:{relative}"],cwd=ROOT)
        if len(blob) != expected["bytes"] or hashlib.sha256(blob).hexdigest() != expected["sha256"]:
            fail(f"source differs: {relative}")
    for expected in build["outputs"].values():
        path = PACKAGE / expected["path"]
        if {"path":expected["path"],"bytes":path.stat().st_size,"sha256":digest(path)} != expected:
            fail("retained build output differs")
    for expected in build["build_records"].values():
        path = PACKAGE / expected["path"]
        if {"path":expected["path"],"bytes":path.stat().st_size,"sha256":digest(path)} != expected:
            fail("retained build record differs")
    compile_commands=(PACKAGE/build["build_records"]["compile_commands"]["path"]).read_text()
    required=("--offload-arch=gfx1201", "-O3 -DNDEBUG",
        "-DNINFER_R9700_DFLASH_SMALL_T_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_MLP_DOWN_T5_CANDIDATE=0",
        "-DNINFER_R9700_DFLASH_RMSNORM_ROWS56_CANDIDATE=0",
        "-DNINFER_R9700_FP8_QK_WMMA=1",
        "-DNINFER_R9700_GDN_VERIFY_WAVE_QK_CANDIDATE=0",
        "-DNINFER_R9700_KV_VALUE_GROUP=16",
        "-DNINFER_R9700_Q4_ACTIVATION_BITS=8",
        "-DNINFER_R9700_W8_ACTIVATION_BITS=8")
    if not all(item in compile_commands for item in required):
        fail("compiled profile differs")
    for expected in build["toolchain_runtime"].values():
        if identity(Path(expected["path"])) != expected:
            fail("toolchain/runtime differs")
    artifact = plan["artifact"]
    if identity(Path(artifact["path"])) != {key:artifact[key] for key in ("path","bytes","sha256")}:
        fail("artifact identity differs")
    preliminary = plan["expected_preliminary_result"]
    path = PACKAGE / preliminary["path"]
    if path.stat().st_size != preliminary["bytes"] or digest(path) != preliminary["sha256"]:
        fail("preliminary result identity differs")
    validate_result(load(path))
    return plan, build


def analyze() -> dict:
    plan, build = validate_prepared()
    result = load(RESULTS / "result.json")
    validate_result(result)
    process = load(RESULTS / "process.json")
    argv = [str(EXE),"--artifact",str(ARTIFACT),"--output",str(RESULTS / "result.json")]
    if (set(process) != {"artifact_type","schema_version","argv","executable","exit_code",
            "power_before","power_after","stdout","stderr"} or
            process.get("artifact_type") != "ninfer_fp8_prefix_process_receipt" or
            process.get("schema_version") != 1 or process.get("argv") != argv or
            process.get("executable") != identity(EXE) or process.get("exit_code") != 0 or
            process.get("power_before") != "auto" or process.get("power_after") != "auto" or
            process.get("stdout") != identity(RESULTS / "stdout") or
            process.get("stderr") != identity(RESULTS / "stderr")):
        fail("process receipt differs")
    if (RESULTS / "stdout").read_bytes() or (RESULTS / "stderr").read_bytes():
        fail("unexpected discriminator output")
    return {"artifact_type":"ninfer_r9700_fp8_prefix_real_weight_summary",
        "schema_version":1,"status":"pass","source":plan["source"],
        "executable":identity(EXE),"artifact":identity(ARTIFACT),
        "classification":"gate_up_width_dependent_bf16_prefix_only",
        "gate_up":{"prefix_mismatch_count":291,"first_output_mismatch":result["cells"][0]["first_output_mismatch"],
            "t128_fingerprint":"e2e00100000000000000000000000000",
            "t129_fingerprint":"dde00100000000000000000000000000"},
        "query_key":{"prefix_mismatch_count":0,"t128_fingerprint":"dde00100000000000000000000000000",
            "t129_fingerprint":"dde00100000000000000000000000000"},
        "conclusion":"layer0 gate_up changes hipBLASLt algorithm and 291 BF16 prefix outputs; layer1 query_key retains one algorithm and exact prefix outputs",
        "limitations":plan["limitations"]}


def write_exclusive(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC, 0o644)
    try:
        offset = 0
        while offset < len(data):
            count = os.write(fd,data[offset:])
            if count <= 0: fail("write made no progress")
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    write_exclusive(RESULTS / "summary.json", analyze())
