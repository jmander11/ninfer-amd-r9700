#!/usr/bin/env python3
"""Validate the selected FP8 gate/up two-pass PMC capture and classify its bottleneck."""

from __future__ import annotations

import argparse, csv, hashlib, json, math, os, tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tools.bench import analyze_q4_pmc as common

SCHEMA = "ninfer.r9700.selected_fp8_gate_up_p2048_pmc_plan.v1"
OUTPUT = "ninfer.r9700.selected_fp8_gate_up_p2048_pmc_attribution.v1"
PASS_A = {"SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_WAVE_CYCLES", "SQ_WAVES",
          "GL2C_HIT", "GL2C_MISS", "GL2C_MC_WRREQ_STALL", "GL2C_EA_RDREQ",
          "TCP_REQ", "TCP_REQ_MISS", "GRBM_GUI_ACTIVE"}
PASS_B = {"SQ_INST_CYCLES_VALU", "SQ_INST_CYCLES_VMEM", "SQ_INSTS_LDS",
          "SQ_INSTS_TEX_LOAD", "SQ_INSTS_TEX_STORE", "SQC_LDS_BANK_CONFLICT",
          "SQC_LDS_IDX_ACTIVE", "SQ_WAVES", "TA_TA_BUSY", "GRBM_GUI_ACTIVE"}
GATE_UP_ITERATIONS = (2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 20, 21, 23, 25, 27, 28,
    30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 48, 49, 51, 53, 55, 56, 58, 60,
    62, 63, 65, 67, 69, 70, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 90, 91,
    93, 95, 97, 98, 100, 102, 104, 105, 107, 109, 111, 112)


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def number(value: object, label: str) -> Decimal:
    try: result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error: raise ValueError(f"{label} is not numeric") from error
    if not result.is_finite() or result < 0: raise ValueError(f"{label} is invalid")
    return result


def validate_plan(path: Path) -> tuple[dict, str]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("schema") != SCHEMA or plan.get("status") != "command_only_not_executed"
            or plan.get("workload") != {"prompt_tokens": 2048, "generated_tokens": 0,
                "concurrency": 1, "prefill_chunk": 4096, "kv_value_group": 16,
                "xattention_profile": "dense", "spec": "none", "repetitions": 1,
                "warmup": 1}
            or plan.get("shape") != {"m_tokens": 2048, "n_rows": 34816,
                                     "k_columns": 5120, "calls": 64}
            or plan.get("resource_contract") != {
                "profiler_dispatch_metadata": {"grid_workitems": 557056,
                    "workgroup_size": 128, "waves_per_dispatch": 17408,
                    "vgpr_count": 188, "lds_bytes": 25088, "scratch_bytes": 0},
                "captured_elf_resources": {"sgpr_count": 128,
                    "architectural_vgpr_count": 192, "accumulator_vgpr_count": 0,
                    "group_segment_lds_bytes": 25088, "private_segment_bytes": 0,
                    "spill_counts_available": False}}
            or plan.get("dispatch_filter", {}).get("kernel_iteration_range")
               != "[" + ",".join(map(str, GATE_UP_ITERATIONS)) + "]"
            or plan.get("dispatch_filter", {}).get("expected_captured_dispatches_per_pass") != 64
            or plan.get("dispatch_filter", {}).get("expected_matching_symbol_occurrences") != 112
            or set(plan.get("passes", {}).get("cache_wait", {}).get("counters", [])) != PASS_A
            or set(plan.get("passes", {}).get("issue_lds", {}).get("counters", [])) != PASS_B):
        raise ValueError("FP8 gate/up PMC plan contract differs")
    for label in ("benchmark_executable", "artifact", "corpus", "linear_source",
                  "target_source", "rocprofv3", "rocprofv3_avail", "trace_attribution",
                  "whole_report", "algorithm_report", "loaded_elf_proof", "streaming_audit"):
        item = plan["inputs"][label]; source = Path(item["path"]).resolve(strict=True)
        if sha(source) != item["sha256"]: raise ValueError(f"{label} bytes changed")
    proof = json.loads(Path(plan["inputs"]["loaded_elf_proof"]["path"]).read_text())
    fp8 = proof.get("fp8", {})
    selected = plan["selected_algorithm"]
    if (proof.get("pass") is not True or fp8.get("native_fp8_matrix_opcode_count") != 112
            or selected.get("solution_index") != 123104
            or selected.get("fingerprint") != "e0e00100000000000000000000000000"):
        raise ValueError("loaded-ELF FP8 proof differs from selected algorithm")
    selected["loaded_elf_kernel_name"] = fp8["kernel_name"]
    selected["trace_kernel_name"] = fp8["kernel_name"].removesuffix(".kd")
    return plan, selected["trace_kernel_name"]


def validate_benchmark_report(path: Path, plan: dict, label: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    config = report.get("config", {})
    tests = report.get("tests", [])
    if (report.get("schema_version") != 20 or report.get("artifact_type") != "ninfer_bench_report"
            or report.get("tool") != "ninfer_bench"
            or report.get("command") != plan["benchmark_commands"][label]
            or report.get("environment", {}).get("gpu_name") != "AMD Radeon AI PRO R9700"
            or report.get("environment", {}).get("architecture_name") != "gfx1201"
            or report.get("artifact", {}).get("path") != plan["inputs"]["artifact"]["path"]
            or report.get("load", {}).get("target") != "qwen3_8_27b_r9700"
            or report.get("load", {}).get("weights_id") != "r9700-q4g64-f8e4m3-four-role-n16k16-eval"
            or config.get("max_context") != 2048 or config.get("prefill_chunk") != 4096
            or config.get("kv_value_group") != 16 or config.get("concurrency") != 1
            or config.get("q4_activation_bits") != 8 or config.get("w8_activation_bits") != 8
            or config.get("q4_prefill_cta_profile") != "m64n128-pingpong-production"
            or config.get("fp8_qk_wmma_enabled") is not True
            or config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1"
            or config.get("fp8_qk_wmma_t1_min_context") != 64
            or config.get("fp8_qk_wmma_t2_min_context") != 320
            or config.get("xattention_qualification") is not False
            or config.get("spec") != "none" or config.get("draft_tokens") != 0
            or config.get("speculative_execution") is not False
            or config.get("proposal_head") != "full"
            or config.get("repetitions") != 1 or config.get("warmup") != 1
            or Path(config.get("corpus_path", "")).resolve()
               != Path(plan["inputs"]["corpus"]["path"])
            or len(tests) != 1 or tests[0].get("label") != "pp2048"
            or tests[0].get("kind") != "pp" or tests[0].get("n_prompt") != 2048
            or tests[0].get("n_gen") != 0
            or not isinstance(tests[0].get("prefill_seconds_mean"), (int, float))
            or tests[0]["prefill_seconds_mean"] <= 0):
        raise ValueError(f"{label} benchmark report differs from the exact workload")
    return {"path": str(path.resolve()), "sha256": sha(path),
            "prefill_seconds": tests[0]["prefill_seconds_mean"]}


def read_csv(path: Path, counters: set[str], kernel: str):
    required = {"Dispatch_Id", "Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Block_Size",
                "Scratch_Size", "VGPR_Count", "Counter_Name", "Counter_Value",
                "Start_Timestamp", "End_Timestamp"}
    metadata, values = {}, {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("counter CSV lacks required rocprof columns")
        for row in reader:
            dispatch = int(row["Dispatch_Id"]); counter = row["Counter_Name"]
            record = {"symbol": row["Kernel_Name"], "grid": int(row["Grid_Size"]),
                      "workgroup": int(row["Workgroup_Size"]), "lds": int(row["LDS_Block_Size"]),
                      "scratch": int(row["Scratch_Size"]), "vgpr": int(row["VGPR_Count"]),
                      "start": int(row["Start_Timestamp"]), "end": int(row["End_Timestamp"])}
            if record["symbol"] != kernel or counter not in counters or record["end"] < record["start"]:
                raise ValueError("counter CSV contains a foreign or invalid dispatch")
            if metadata.setdefault(dispatch, record) != record or (dispatch, counter) in values:
                raise ValueError("counter CSV contains inconsistent or duplicate rows")
            values[dispatch, counter] = number(row["Counter_Value"], counter)
    if not metadata: raise ValueError("counter CSV is empty")
    for dispatch in metadata:
        if {counter for item, counter in values if item == dispatch} != counters:
            raise ValueError("counter CSV dispatch lacks the exact counter set")
    return metadata, values


def reduce(ids, values, counters):
    return {counter: sum((values[item, counter] for item in ids), Decimal(0))
            for counter in counters}


def pct(a: Decimal, b: Decimal, label: str) -> float:
    if b <= 0: raise ValueError(f"{label} denominator is not positive")
    value = float(a * 100 / b)
    if not math.isfinite(value) or value > 100: raise ValueError(f"{label} exceeds a bounded ratio")
    return value


def analyze(plan_path: Path, report_a: Path, csv_a: Path, db_a: Path, agent_a: Path,
            before_a: Path, after_a: Path, report_b: Path, csv_b: Path, db_b: Path,
            agent_b: Path, before_b: Path, after_b: Path) -> dict:
    plan, kernel = validate_plan(plan_path)
    benchmark_a = validate_benchmark_report(report_a, plan, "cache_wait")
    benchmark_b = validate_benchmark_report(report_b, plan, "issue_lds")
    if any(path.read_text().strip() != expected for path, expected in
           ((before_a, "profile_standard"), (after_a, "auto"),
            (before_b, "profile_standard"), (after_b, "auto"))):
        raise ValueError("PMC power-profile evidence differs")
    cu, simd, gpu = common._read_agent(agent_a)
    if common._read_agent(agent_b) != (cu, simd, gpu): raise ValueError("pass GPU identities differ")
    summaries = []
    for label, csv_path, db_path, counters in (("cache_wait", csv_a, db_a, PASS_A),
                                                ("issue_lds", csv_b, db_b, PASS_B)):
        metadata, values = read_csv(csv_path, counters, kernel)
        regions, db_gpu, process = common._read_regions(db_path)
        if db_gpu != gpu or not set(metadata) <= set(regions): raise ValueError("CSV/DB identity differs")
        ids = []
        for dispatch, record in metadata.items():
            symbol, region = regions[dispatch]
            if symbol != kernel: raise ValueError("CSV/DB symbol differs")
            if record["grid"] == 557056 and region and region.startswith("ninfer.post-mixer.prefill."):
                ids.append(dispatch)
            else:
                raise ValueError(f"{label} contains an unrecognized selected-symbol dispatch")
        if len(ids) != 64: raise ValueError(f"{label} lacks exact 64 gate/up dispatches")
        if set(ids) != set(metadata):
            raise ValueError(f"{label} selected-symbol inventory is not exact")
        if process["command"] != plan["benchmark_commands"][label]:
            raise ValueError(f"{label} process command differs from the exact workload")
        if any(metadata[item][key] != value for item in ids for key, value in
               (("workgroup", 128), ("lds", 25088), ("scratch", 0), ("vgpr", 188))):
            raise ValueError(f"{label} gate/up resources differ")
        if any(values[item, "SQ_WAVES"] != Decimal(17408) for item in ids):
            raise ValueError(f"{label} SQ_WAVES differs from grid/wave geometry")
        summaries.append((reduce(ids, values, counters), process))
    a, process_a = summaries[0]; b, process_b = summaries[1]
    for key in ("SQ_WAVES", "SQ_WAVE_CYCLES", "TCP_REQ", "GL2C_EA_RDREQ"):
        if a[key] <= 0: raise ValueError(f"nonpositive activity control {key}")
    if (b["SQ_WAVES"] <= 0 or a["TCP_REQ_MISS"] > a["TCP_REQ"]
            or a["GRBM_GUI_ACTIVE"] <= 0 or b["GRBM_GUI_ACTIVE"] <= 0
            or a["GL2C_HIT"] + a["GL2C_MISS"] <= 0):
        raise ValueError("invalid activity control")
    metrics = {
        "dependency_wait_percent": pct(a["SQ_WAIT_ANY"], a["SQ_WAVE_CYCLES"], "dependency wait"),
        "issue_wait_percent": pct(a["SQ_WAIT_INST_ANY"], a["SQ_WAVE_CYCLES"], "issue wait"),
        "gl2_hit_percent": pct(a["GL2C_HIT"], a["GL2C_HIT"] + a["GL2C_MISS"], "GL2 hit"),
        "l0_vector_cache_hit_percent": pct(a["TCP_REQ"] - a["TCP_REQ_MISS"], a["TCP_REQ"], "L0 hit"),
        "occupancy_percent": pct(a["SQ_WAVE_CYCLES"],
            a["GRBM_GUI_ACTIVE"] * Decimal(cu) * Decimal(32), "occupancy"),
    }
    metrics["lds_bank_conflict_percent"] = (None if b["SQC_LDS_IDX_ACTIVE"] == 0 else
        pct(b["SQC_LDS_BANK_CONFLICT"], b["SQC_LDS_IDX_ACTIVE"], "LDS conflict"))
    metrics["lds_utilization_percent"] = pct(
        b["SQC_LDS_IDX_ACTIVE"], b["GRBM_GUI_ACTIVE"] * Decimal(32), "LDS utilization")
    logical_flop, median_ms, ceiling, closure_ms = 730144440320, 4.061550617, 400.835, 1.646909
    achieved = logical_flop / (median_ms * 1e9)
    ceiling_ms = logical_flop / (ceiling * 1e9)
    return {"schema": OUTPUT, "status": "valid_attribution_only",
            "profile_timing_admissible": False, "power_profile": {"capture": "profile_standard", "restored": "auto"},
            "plan": {"path": str(plan_path.resolve()), "sha256": sha(plan_path)}, "gpu": gpu,
            "shape": plan["shape"], "selected_algorithm": plan["selected_algorithm"],
            "resources": plan["resource_contract"],
            "metrics_percent": metrics,
            "normalized_activity": {"wave_cycles_per_wave": float(a["SQ_WAVE_CYCLES"] / a["SQ_WAVES"]),
                "gl2_read_requests_per_wave": float(a["GL2C_EA_RDREQ"] / a["SQ_WAVES"]),
                "valu_cycles_per_wave": float(b["SQ_INST_CYCLES_VALU"] / b["SQ_WAVES"]),
                "vmem_cycles_per_wave": float(b["SQ_INST_CYCLES_VMEM"] / b["SQ_WAVES"]),
                "lds_instructions_per_wave": float(b["SQ_INSTS_LDS"] / b["SQ_WAVES"]),
                "tex_load_instructions_per_wave": float(b["SQ_INSTS_TEX_LOAD"] / b["SQ_WAVES"]),
                "tex_store_instructions_per_wave": float(b["SQ_INSTS_TEX_STORE"] / b["SQ_WAVES"]),
                "gl2_write_stall_instance_sum_per_grbm_active_cycle":
                    float(a["GL2C_MC_WRREQ_STALL"] / a["GRBM_GUI_ACTIVE"]),
                "ta_busy_instance_sum_per_grbm_active_cycle":
                    float(b["TA_TA_BUSY"] / b["GRBM_GUI_ACTIVE"])},
            "roofline": {"useful_flop_per_call": logical_flop, "unprofiled_median_ms": median_ms,
                "achieved_tflops": achieved, "achieved_useful_tmac_per_second": achieved / 2,
                "architectural_ceiling_tflops": ceiling,
                "architectural_ceiling_useful_tmac_per_second": ceiling / 2,
                "fraction_of_architectural_ceiling": achieved / ceiling,
                "ceiling_time_ms": ceiling_ms, "whole_closure_inferred_time_ms": closure_ms,
                "whole_closure_inferred_tflops": logical_flop / (closure_ms * 1e9),
                "whole_closure_inferred_useful_tmac_per_second": logical_flop / (closure_ms * 2e9),
                "physical_read_bandwidth": None, "wmma_utilization": None},
            "decision": {
                "fundamentally_different_architecture_warranted_by_this_capture_alone": False,
                "reason": "broad overlapping SQ waits and cache hit ratios diagnose where to inspect but cannot causally admit a rewrite; moreover the inferred standalone whole-closure rate exceeds the gfx1201 architectural FP8 ceiling",
                "next_use": "use the decision-ready cache/wait/occupancy signature to choose a bounded ATT region or retain the selected library kernel; require direct correctness and timing before any custom architecture",
                "standalone_whole_closure_possible_at_architectural_ceiling": ceiling_ms <= closure_ms},
            "passes": {"cache_wait": {"benchmark": benchmark_a, "csv": sha(csv_a),
                                        "database": sha(db_a), "process": process_a},
                       "issue_lds": {"benchmark": benchmark_b, "csv": sha(csv_b),
                                     "database": sha(db_b), "process": process_b}}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    for suffix in ("a", "b"):
        parser.add_argument(f"--counter-csv-{suffix}", type=Path, required=True)
        parser.add_argument(f"--database-{suffix}", type=Path, required=True)
        parser.add_argument(f"--agent-csv-{suffix}", type=Path, required=True)
        parser.add_argument(f"--benchmark-report-{suffix}", type=Path, required=True)
        parser.add_argument(f"--power-before-{suffix}", type=Path, required=True)
        parser.add_argument(f"--power-after-{suffix}", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = analyze(args.plan, args.benchmark_report_a, args.counter_csv_a, args.database_a,
            args.agent_csv_a, args.power_before_a, args.power_after_a, args.benchmark_report_b,
            args.counter_csv_b, args.database_b, args.agent_csv_b, args.power_before_b,
            args.power_after_b)
        if os.path.lexists(args.out): raise ValueError(f"refusing to overwrite {args.out}")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{args.out.name}.", dir=args.out.parent)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(value, output, indent=2); output.write("\n")
            os.link(temporary, args.out)
        finally:
            Path(temporary).unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error: raise SystemExit(str(error)) from error


if __name__ == "__main__": main()
