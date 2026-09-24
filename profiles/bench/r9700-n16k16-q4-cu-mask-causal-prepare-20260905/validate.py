#!/usr/bin/env python3
"""Validate and classify the prepared R9700 N16/K16 CU-mask experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "ninfer.r9700.n16k16_q4_cu_mask_causal_plan.v1"
RESULT_SCHEMA = "ninfer.r9700.n16k16_q4_cu_mask_causal_result.v1"
ARM_IDS = ("full64", "half32", "quarter16")
MASKS = {"full64": None, "half32": "0:0-31", "quarter16": "0:0-15"}
FRACTIONS = {"full64": 1.0, "half32": 0.5, "quarter16": 0.25}
GPU = "AMD Radeon AI PRO R9700"
ARCH = "gfx1201"
KERNEL = "a8q4g64_linear_prefill_cta_kernel"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def validate_plan(plan: dict[str, Any], *, verify_inputs: bool) -> None:
    _require(plan.get("schema") == SCHEMA and plan.get("status") == "prepared_not_executed",
             "wrong CU-mask plan schema/status")
    hw = plan.get("hardware", {})
    _require(hw.get("device") == GPU and hw.get("architecture") == ARCH and
             hw.get("architectural_cus") == 64 and hw.get("wave_size") == 32,
             "wrong R9700 hardware contract")
    arms = plan.get("arms")
    _require(isinstance(arms, list) and [x.get("id") for x in arms] == list(ARM_IDS),
             "plan must contain full64/half32/quarter16 in order")
    for arm in arms:
        arm_id = arm["id"]
        _require(arm.get("hsa_cu_mask") == MASKS[arm_id] and
                 arm.get("active_cus") == int(64 * FRACTIONS[arm_id]) and
                 arm.get("nominal_fraction") == FRACTIONS[arm_id],
                 f"wrong {arm_id} mask contract")
    workload = plan.get("workload", {})
    exact = {"prompt_tokens": 2048, "generated_tokens": 0, "concurrency": 1,
             "prefill_chunk": 4096, "draft_tokens": 0, "spec": "none",
             "repetitions": 1, "warmup": 1, "profile_measured": True,
             "artifact_weights_id": "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
             "q4_kernel_substring": KERNEL, "expected_q4_dispatches": 176}
    _require(all(workload.get(k) == v for k, v in exact.items()), "wrong P2048 workload")
    _require(plan.get("power", {}).get("required_before_and_after_each_arm") == "auto" and
             plan.get("power", {}).get("mutation", "").startswith("none"),
             "experiment must remain auto without power mutation")
    _require(plan.get("peak_probe", {}).get("minimum_working_set_over_l2") == 8.0 and
             plan.get("classification", {}).get("relative_curve_tolerance") == 0.10,
             "wrong calibration/classification threshold")
    inputs = plan.get("inputs")
    _require(isinstance(inputs, dict) and set(inputs) ==
             {"benchmark", "artifact", "corpus", "peak_probe", "python", "rocprofv3"},
             "wrong bound input inventory")
    if verify_inputs:
        for label, identity in inputs.items():
            path = Path(identity["path"])
            st = path.stat()
            _require(stat.S_ISREG(st.st_mode) and st.st_size == identity["bytes"],
                     f"{label} size/type changed")
            _require(_sha256(path) == identity["sha256"], f"{label} hash changed")


def _one_table(connection: sqlite3.Connection, prefix: str) -> str:
    rows = [r[0] for r in connection.execute(
        "select name from sqlite_master where type in ('table','view') and name like ?",
        (prefix + "%",))]
    _require(len(rows) == 1, f"trace needs one {prefix} table")
    return rows[0]


def _expected_command(plan: dict[str, Any], package: Path, arm: str) -> list[str]:
    inputs = plan["inputs"]
    return [inputs["benchmark"]["path"], "--weights", inputs["artifact"]["path"],
            "--corpus", inputs["corpus"]["path"], "--device", "0", "--concurrency", "1",
            "-p", "2048", "--prefill-chunk", "4096", "--draft-tokens", "0",
            "--output", "json", "--output-file", str(package / f"benchmark-{arm}.json"),
            "-r", "1", "--warmup", "1", "--profile-measured"]


def _validate_benchmark(report: dict[str, Any], plan: dict[str, Any], package: Path,
                        arm: str) -> None:
    _require(report.get("artifact_type") == "ninfer_bench_report" and
             report.get("schema_version") == 20, "wrong benchmark report schema")
    _require(shlex.split(report.get("command", "")) == _expected_command(plan, package, arm),
             f"{arm} benchmark command mismatch")
    config = report.get("config", {})
    expected = {"max_context": 2048, "prefill_chunk": 4096, "concurrency": 1,
                "spec": "none", "draft_tokens": 0, "speculative_execution": False,
                "repetitions": 1, "warmup": 1,
                "q4_prefill_cta_profile": "m64n128-pingpong-n16-k16-production"}
    _require(all(config.get(k) == v for k, v in expected.items()),
             f"{arm} benchmark config mismatch")
    tests = report.get("tests")
    _require(isinstance(tests, list) and len(tests) == 1 and tests[0].get("kind") == "pp" and
             tests[0].get("n_prompt") == 2048 and tests[0].get("n_gen") == 0 and
             tests[0].get("prefill_seconds_mean", 0) > 0,
             f"{arm} benchmark payload mismatch")


def _validate_trace(path: Path, expected_command: list[str], mask: str | None,
                    plan: dict[str, Any]) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        process_table = _one_table(connection, "rocpd_info_process_")
        agent_table = _one_table(connection, "rocpd_info_agent_")
        processes = list(connection.execute(
            f'select command, environment from "{process_table}" order by id'))
        _require(len(processes) == 1 and shlex.split(processes[0]["command"]) == expected_command,
                 "trace process command mismatch")
        environment = json.loads(processes[0]["environment"])
        _require(isinstance(environment, dict), "trace process environment is invalid")
        if mask is None:
            _require("HSA_CU_MASK" not in environment, "full64 trace inherited HSA_CU_MASK")
        else:
            _require(environment.get("HSA_CU_MASK") == mask, "trace HSA_CU_MASK mismatch")
        agents = list(connection.execute(
            f'select name, product_name from "{agent_table}" where type = ?', ("GPU",)))
        _require(sum(r["name"] == ARCH and r["product_name"] == GPU for r in agents) == 1,
                 "trace lacks exactly one gfx1201 R9700")
        columns = {r[1] for r in connection.execute('pragma table_info("kernels")')}
        required = {"name", "duration", "grid_x", "grid_y", "grid_z", "workgroup_x",
                    "workgroup_y", "workgroup_z", "vgpr_count", "lds_size", "scratch_size"}
        _require(required <= columns, "trace kernel view lacks resource/launch fields")
        rows = list(connection.execute(
            "select name,duration,grid_x,grid_y,grid_z,workgroup_x,workgroup_y,workgroup_z,"
            "vgpr_count,lds_size,scratch_size from kernels where name like ? order by start",
            (f"%{KERNEL}%",)))
    finally:
        connection.close()
    _require(len(rows) == 176 and all(KERNEL in r["name"] for r in rows),
             "trace must contain exactly 176 selected Q4 dispatches")
    launches: dict[tuple[int, ...], int] = {}
    for row in rows:
        _require(type(row["duration"]) is int and row["duration"] > 0,
                 "trace contains invalid duration")
        launch = tuple(int(row[k]) for k in
                       ("grid_x", "grid_y", "grid_z", "workgroup_x", "workgroup_y", "workgroup_z"))
        launches[launch] = launches.get(launch, 0) + 1
        resources = plan["workload"]["required_resources"]
        _require(row["vgpr_count"] == resources["vgpr"] and
                 row["lds_size"] == resources["lds_bytes"] and
                 row["scratch_size"] == resources["scratch_bytes"],
                 "trace Q4 resources changed")
    _require(launches == {(49152, 32, 1, 512, 1, 1): 48,
                          (20480, 32, 1, 512, 1, 1): 128},
             "trace Q4 launch inventory changed")
    return {"dispatches": len(rows), "summed_duration_ms":
            sum(int(r["duration"]) for r in rows) / 1e6,
            "database_sha256": _sha256(path)}


def _validate_peak(report: dict[str, Any], plan: dict[str, Any]) -> dict[str, float]:
    _require(report.get("artifact_type") == "ninfer_r9700_q4_hardware_peak_probe" and
             report.get("schema_version") == 2 and report.get("status") == "passed",
             "wrong peak-probe schema/status")
    hardware = report.get("hardware", {})
    _require(hardware.get("device") == GPU and hardware.get("architecture") == ARCH and
             hardware.get("wave_size") == 32 and hardware.get("l2_bytes", 0) > 0,
             "wrong peak-probe device")
    power = report.get("power_profile", {})
    _require(power.get("before") == power.get("after") == power.get("required") == "auto",
             "peak probe did not remain auto")
    issue = report.get("iu4_issue", {})
    stream = report.get("q4g64_stream", {})
    trials = plan["peak_probe"]["trials"]
    issue_samples = issue.get("samples_ms", [])
    stream_samples = stream.get("samples_ms", [])
    _require(len(issue_samples) == trials and len(stream_samples) == trials and
             issue.get("median_useful_w4a8_tops", 0) > 0 and
             stream.get("median_represented_gbps", 0) > 0,
             "peak probe lacks complete positive samples")
    _require(issue.get("opcode") == "v_wmma_i32_16x16x32_iu4" and
             issue.get("independent_accumulator_chains") == 8,
             "peak probe is not the exact IU4 issue control")
    _require(stream.get("group_size") == 64 and
             stream.get("code_bytes_per_group_row") == 32 and
             stream.get("scale_bytes_per_group_row") == 2 and
             stream.get("code_to_scale_byte_ratio") == 16,
             "peak probe is not the exact Q4G64 stream control")
    _require(all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                 for x in [*issue_samples, *stream_samples]),
             "peak probe samples must be positive and finite")
    _require(math.isclose(float(issue["median_ms"]),
                          float(sorted(issue_samples)[trials // 2]), rel_tol=0, abs_tol=1e-12) and
             math.isclose(float(stream["median_ms"]),
                          float(sorted(stream_samples)[trials // 2]), rel_tol=0, abs_tol=1e-12),
             "peak probe medians do not recompute from raw samples")
    _require(stream.get("working_set_over_l2", 0) >=
             plan["peak_probe"]["minimum_working_set_over_l2"],
             "stream working set is not sufficiently larger than L2")
    return {"iu4": float(issue["median_useful_w4a8_tops"]),
            "stream": float(stream["median_represented_gbps"])}


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual / expected - 1.0)


def classify(arms: dict[str, dict[str, float]], tolerance: float = 0.10) -> dict[str, Any]:
    _require(set(arms) == set(ARM_IDS), "classification requires exactly all three arms")
    full = arms["full64"]
    _require(all(math.isfinite(full[k]) and full[k] > 0 for k in ("iu4", "stream", "q4_ms")),
             "full64 metrics must be positive and finite")
    curves: dict[str, dict[str, float]] = {}
    for arm in ARM_IDS:
        values = arms[arm]
        _require(all(math.isfinite(values[k]) and values[k] > 0
                     for k in ("iu4", "stream", "q4_ms")),
                 f"{arm} metrics must be positive and finite")
        curves[arm] = {"iu4": values["iu4"] / full["iu4"],
                       "stream": values["stream"] / full["stream"],
                       "production": full["q4_ms"] / values["q4_ms"]}
    for arm in ("half32", "quarter16"):
        _require(_relative_error(curves[arm]["iu4"], FRACTIONS[arm]) <= tolerance + 1.0e-12,
                 f"{arm} IU4 control does not validate the requested CU mask")
    compute_match = all(_relative_error(curves[a]["production"], curves[a]["iu4"])
                        <= tolerance + 1.0e-12 for a in ("half32", "quarter16"))
    stream_match = all(_relative_error(curves[a]["production"], curves[a]["stream"])
                       <= tolerance + 1.0e-12 for a in ("half32", "quarter16"))
    if compute_match and not stream_match:
        result = "cu_local_service"
    elif stream_match and not compute_match:
        result = "shared_memory_service"
    else:
        result = "mixed_or_inconclusive"
    return {"classification": result, "curves_relative_to_full64": curves,
            "within_ten_percent": {"iu4": compute_match, "stream": stream_match}}


def _publish(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    _require(not os.path.lexists(path), "final result already exists")
    _require(path.parent.is_dir() and not path.parent.is_symlink(), "invalid output parent")
    fd, temporary_name = tempfile.mkstemp(prefix=".cu-mask-result-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            _require(written > 0, "short final-result write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        st = os.fstat(fd)
        os.close(fd); fd = -1
        os.link(temporary, path)
        final_st = path.lstat()
        _require((final_st.st_dev, final_st.st_ino) == (st.st_dev, st.st_ino),
                 "final publication inode changed")
        with path.open("rb") as source:
            _require(source.read() == data, "final result readback mismatch")
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    except Exception:
        if os.path.lexists(path):
            current = path.lstat()
            if 'st' in locals() and (current.st_dev, current.st_ino) == (st.st_dev, st.st_ino):
                path.unlink()
        raise
    finally:
        if fd >= 0: os.close(fd)
        if os.path.lexists(temporary):
            current = temporary.lstat()
            if 'st' not in locals() or (current.st_dev, current.st_ino) == (st.st_dev, st.st_ino):
                temporary.unlink()


def validate(plan_path: Path, output: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    validate_plan(plan, verify_inputs=True)
    package = plan_path.resolve().parent
    arm_metrics: dict[str, dict[str, float]] = {}
    evidence: dict[str, Any] = {}
    for arm in ARM_IDS:
        for when in ("before", "after"):
            _require((package / f"power-{arm}-{when}.txt").read_text().strip() == "auto",
                     f"{arm} power {when} is not auto")
        peak_path = package / f"peak-{arm}.json"
        report_path = package / f"benchmark-{arm}.json"
        database_path = package / f"raw-{arm}/cu-mask-{arm}_results.db"
        peak = _validate_peak(_load(peak_path), plan)
        report = _load(report_path)
        _validate_benchmark(report, plan, package, arm)
        trace = _validate_trace(database_path, _expected_command(plan, package, arm),
                                MASKS[arm], plan)
        arm_metrics[arm] = {**peak, "q4_ms": trace["summed_duration_ms"]}
        evidence[arm] = {"hsa_cu_mask": MASKS[arm], "active_cus": int(64 * FRACTIONS[arm]),
                         "peak_probe_sha256": _sha256(peak_path),
                         "benchmark_report_sha256": _sha256(report_path), **peak, **trace}
    decision = classify(arm_metrics, plan["classification"]["relative_curve_tolerance"])
    payload = {"schema": RESULT_SCHEMA, "status": "valid", "decision": decision,
               "workload": plan["workload"], "arms": evidence,
               "plan": {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)},
               "interpretation": "Causal CU-count response classification; no production change or kernel admission."}
    _publish(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _require(args.output is None, "--preflight does not accept --output")
        validate_plan(_load(args.plan), verify_inputs=True)
        print("CU-mask causal package preflight: PASS")
        return 0
    _require(args.output is not None, "--output is required unless --preflight is used")
    result = validate(args.plan, args.output)
    print(f"CU-mask causal experiment: {result['decision']['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
