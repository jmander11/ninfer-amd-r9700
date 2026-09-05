#!/usr/bin/env python3
"""Validate and publish the conditional selected mixed-route MTP-bulk W8 proof."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Callable

from tools.bench.run_ninfer_bench_matrix import REPORT_SCHEMA_VERSION
from tools.bench.validate_profile_trace import _parse_database

REPO = Path(__file__).resolve().parents[2]
RESOLVER = REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
MIXED_ID = "r9700-q4-w8-mse-n16k16-eval"
POWER = Path("/sys/class/drm/card2/device/power_dpm_force_performance_level")
QUALIFIER_SOURCE = REPO / "tools/r9700/w8a8_wmma_linear_qual.hip"
KERNEL_SOURCE = REPO / "src/ops/r9700/linear/r9700_linear.hip"
PROFILE_SOURCE = REPO / "src/ops/r9700/linear/r9700_w8_activation_profile.h"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def snapshot(path: Path) -> dict:
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ValueError(f"symbolic-link authority is forbidden: {path}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    return {"path": str(path), "file_size_bytes": path.stat().st_size, "sha256": sha(path)}


def resolve(selection: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ninfer_terminal_route", RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load terminal route resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve(selection)


def static_iu8(assembly: Path, metadata: Path) -> dict:
    text = assembly.read_text(encoding="utf-8")
    labels = list(re.finditer(
        r"^[0-9a-f]+ <[^>]*a8w8g32_linear_prefill_cta_kernel[^>]*>:$", text, re.MULTILINE))
    if len(labels) != 1:
        raise ValueError("production W8 prefill CTA symbol is absent from gfx1201 assembly")
    marker = labels[0]
    following = re.search(r"^[0-9a-f]+ <[^>]+>:$", text[marker.end():], re.MULTILINE)
    end = marker.end() + following.start() if following is not None else len(text)
    interval = text[marker.start():end]
    opcodes = re.findall(
        r"^\s*v_wmma_i32_16x16x16_iu8\b[^\n]*\bneg_lo:\[(\d),(\d),(\d)\]",
        interval, re.MULTILINE,
    )
    opcode_count = len(opcodes)
    note = metadata.read_text(encoding="utf-8")
    blocks = [block for block in note.split("\n  - .args:")
              if "a8w8g32_linear_prefill_cta_kernel" in block]
    if len(blocks) != 1:
        raise ValueError("production W8 prefill CTA lacks unique ELF metadata")
    metadata_block = blocks[0]
    def field(name: str) -> int:
        match = re.search(rf"^\s*\.{name}:\s+(\d+)$", metadata_block, re.MULTILINE)
        if match is None:
            raise ValueError(f"production W8 prefill CTA lacks {name} metadata")
        return int(match.group(1))
    proof = {"symbol": "a8w8g32_linear_prefill_cta_kernel",
             "opcode": "v_wmma_i32_16x16x16_iu8", "opcode_sites": opcode_count,
             "signedness": "signed_activation_signed_weight",
             "neg_lo": [list(map(int, values)) for values in opcodes],
             "vgpr": field("vgpr_count"), "lds_bytes": field("group_segment_fixed_size"),
             "scratch_bytes": field("private_segment_fixed_size")}
    if proof != {"symbol": "a8w8g32_linear_prefill_cta_kernel",
                 "opcode": "v_wmma_i32_16x16x16_iu8", "opcode_sites": 2,
                 "signedness": "signed_activation_signed_weight",
                 "neg_lo": [[1, 1, 0], [1, 1, 0]],
                 "vgpr": 50, "lds_bytes": 4352, "scratch_bytes": 0}:
        raise ValueError("production W8 prefill CTA ISA/resources differ")
    return proof


def _owned(path: Path, owner: tuple[int, int]) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == owner


def _remove_owned(path: Path, owner: tuple[int, int]) -> None:
    if _owned(path, owner):
        path.unlink()


def _validate_plan(plan: dict, route: dict, plan_path: Path, raw: Path,
                   assembly: Path, metadata: Path, code_object: Path, benchmark_report: Path,
                   trace_database: Path, power_before: Path, power_after: Path) -> list[str]:
    expected_outputs = {
        "operator": str(raw), "evidence": str(Path(plan["outputs"]["evidence"])),
        "trace_database": str(trace_database), "benchmark_report": str(benchmark_report),
        "power_before": str(power_before), "power_after": str(power_after),
        "production_code_object": str(code_object), "production_assembly": str(assembly),
        "production_metadata": str(metadata),
    }
    command = plan.get("benchmark_command")
    whole_path = Path(route["source_matrices"]["pareto-whole"]["path"]).resolve(strict=True)
    whole = json.loads(whole_path.read_text(encoding="utf-8"))
    corpus = Path(str(whole.get("corpus", ""))).resolve(strict=True)
    corpus_identity = {"path": str(corpus), "sha256": sha(corpus)}
    expected_command = [
        route["benchmark"]["path"], "--weights", route["artifact"]["path"],
        "--corpus", str(corpus), "--device", "0", "--concurrency", "1",
        "-p", str(route["selected_prefill_chunk"]), "--prefill-chunk",
        str(route["selected_prefill_chunk"]), "--spec", "mtp", "--draft-tokens", "3",
        "--lm-head-draft", "--output", "json", "--output-file", str(benchmark_report),
        "-r", "1", "--warmup", "1", "--profile-measured",
    ]
    if (
        plan.get("artifact_type") != "ninfer_r9700_selected_mtp_bulk_w8_plan"
        or plan.get("schema_version") != 1
        or plan.get("status") != "prepared_post_admission_trace"
        or plan.get("conditional_recipe") != MIXED_ID
        or plan.get("selected_route") != route
        or plan.get("concurrency") != 1
        or plan.get("selected_prefill_chunk") != route["selected_prefill_chunk"]
        or plan.get("outputs") != expected_outputs
        or plan.get("corpus") != corpus_identity
        or whole.get("corpus_sha256") != corpus_identity["sha256"]
        or command != expected_command
    ):
        raise ValueError("MTP-bulk plan differs from the selected route or output namespace")
    return command


def _option(command: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"benchmark command lacks exactly one {name}")
    return command[positions[0] + 1]


def _validate_engine_trace(route: dict, command: list[str], report_path: Path,
                           database_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config, artifact, load = report.get("config", {}), report.get("artifact", {}), report.get("load", {})
    tests = report.get("tests")
    expected_sparse = route["execution_profile"]["xattention_profile"] != "dense"
    expected_command = " ".join(command)
    if (
        report.get("artifact_type") != "ninfer_bench_report"
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("command") != expected_command
        or Path(str(artifact.get("path", ""))).resolve() != Path(route["artifact"]["path"])
        or artifact.get("file_size_bytes") != route["artifact"]["file_size_bytes"]
        or load.get("target") != "qwen3_8_27b_r9700" or load.get("weights_id") != MIXED_ID
        or config.get("concurrency") != 1
        or config.get("prefill_chunk") != route["selected_prefill_chunk"]
        or config.get("kv_value_group") != route["cache_profile"]["value_group"]
        or config.get("q4_activation_bits") != 8 or config.get("w8_activation_bits") != 8
        or config.get("fp8_qk_wmma_enabled") is not True
        or config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1"
        or config.get("xattention_qualification") is not expected_sparse
        or config.get("spec") != "mtp" or config.get("draft_tokens") != 3
        or config.get("proposal_head") != "optimized"
        or config.get("repetitions") != 1 or config.get("warmup") != 1
        or not isinstance(tests, list) or len(tests) != 1
        or tests[0].get("kind") != "pp"
        or tests[0].get("n_prompt") != route["selected_prefill_chunk"]
        or tests[0].get("n_gen") != 0 or tests[0].get("requested_output_tokens") != 1
    ):
        raise ValueError("selected Engine report does not bind the mixed MTP C1 route")
    if expected_sparse:
        profile = route["execution_profile"]["xattention_profile"]
        if (config.get("xattention_profile") != profile
                or config.get("xattention_stride") != 16
                or config.get("xattention_tau_permille") != 900):
            raise ValueError("selected Engine report lacks its XAttention profile")
    elif any(name in config for name in ("xattention_profile", "xattention_stride",
                                          "xattention_tau_permille")):
        raise ValueError("dense selected Engine report unexpectedly enables XAttention")
    dispatches, aggregate = _parse_database(database_path, expected_command)
    region = f"ninfer.mtp.prefill.mtp_chunk payload={route['selected_prefill_chunk']}"
    in_region = [row for row in dispatches if row["roctx_region"] == region]
    w8_dispatches = [row for row in in_region
                     if "a8w8g32_linear_prefill_cta_kernel" in row["symbol"]]
    if len(w8_dispatches) != 3:
        raise ValueError("selected Engine did not execute exactly three native W8 CTA dispatches")
    token_blocks = route["selected_prefill_chunk"] // 64
    expected = {(40960, token_blocks, 1): 1, (8192, token_blocks, 1): 2}
    observed: dict[tuple[int, int, int], int] = {}
    for row in w8_dispatches:
        grid = row["grid"]
        key = (grid["x"], grid["y"], grid["z"])
        observed[key] = observed.get(key, 0) + 1
        if row["workgroup"] != {"x": 512, "y": 1, "z": 1}:
            raise ValueError("selected Engine W8 CTA workgroup differs from production")
    if observed != expected:
        raise ValueError("selected Engine W8 CTA dispatch shapes/counts differ")
    fallback_grids = {(10240, route["selected_prefill_chunk"] // 16, 1),
                      (2048, route["selected_prefill_chunk"] // 16, 1)}
    if any("a8w8g32_linear_wmma32_kernel" in row["symbol"]
           and tuple(row["grid"][axis] for axis in ("x", "y", "z")) in fallback_grids
           and row["workgroup"] == {"x": 32, "y": 1, "z": 1}
           for row in in_region):
        raise ValueError("selected Engine retained an MTP-bulk one-wave fallback dispatch")
    return {"region": region, "dispatch_count": 3, "grid_inventory": [
        {"grid": list(grid), "calls": calls} for grid, calls in sorted(observed.items())],
        "trace_dispatch_count": aggregate["dispatch_count"]}


def _assemble(selection: Path, plan_path: Path, raw: Path, assembly: Path, metadata: Path,
              code_object: Path, benchmark_report: Path, trace_database: Path,
              power_before: Path, power_after: Path, qualifier: Path,
              planner: Path, *, power_reader: Callable[[Path], str]) -> dict:
    route = resolve(selection.resolve(strict=True))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    command = _validate_plan(plan, route, plan_path, raw, assembly, metadata, code_object,
                             benchmark_report, trace_database, power_before, power_after)
    if route["artifact"].get("weights_id") != MIXED_ID:
        raise ValueError("terminal winner is not the mixed W8 recipe")
    build = Path(route["build_directory"]).resolve(strict=True)
    expected_planner = (build / "src/ninfer_r9700_runtime_planner_qual").resolve(strict=True)
    if planner.resolve(strict=True) != expected_planner:
        raise ValueError("planner is not from the selected benchmark build")
    raw_value = json.loads(raw.read_text(encoding="utf-8"))
    shapes = raw_value.get("shapes")
    operator_executable = raw_value.get("executable", {})
    operator_sources = raw_value.get("sources")
    expected_source_paths = {
        "qualifier": QUALIFIER_SOURCE.resolve(),
        "contract_header": (REPO / "src/ops/r9700/linear/r9700_linear.h").resolve(),
        "kernel": KERNEL_SOURCE.resolve(), "dispatch_profile": PROFILE_SOURCE.resolve(),
    }
    source_by_role = {row.get("role"): row for row in operator_sources
                      if isinstance(row, dict)} if isinstance(operator_sources, list) else {}
    expected = {(5120, 10240): 1, (1024, 5120): 2}
    observed = {(row.get("rows"), row.get("columns")):
                row.get("dispatches_per_full_chunk") for row in shapes} \
        if isinstance(shapes, list) else {}
    if (raw_value.get("artifact_type") != "ninfer_r9700_mtp_bulk_w8_operator_qualification"
            or raw_value.get("schema_version") != 1 or raw_value.get("status") != "passed"
            or raw_value.get("tokens") != route["selected_prefill_chunk"]
            or raw_value.get("concurrency") != 1
            or not isinstance(shapes, list) or len(shapes) != 2 or observed != expected
            or raw_value.get("power_profile") != {"path": str(POWER), "value": "auto"}
            or raw_value.get("hardware", {}).get("architecture") != "gfx1201"
            or raw_value.get("hardware", {}).get("wave_size") != 32
            or raw_value.get("candidate", {}).get("opcode") != "v_wmma_i32_16x16x16_iu8"
            or raw_value.get("candidate", {}).get("activation") != "signed A8G32"
            or raw_value.get("candidate", {}).get("weight") != "W8G32_F16S"
            or raw_value.get("baseline", {}).get("activation") != "signed A8G32"
            or raw_value.get("baseline", {}).get("route") != "a8w8g32_linear_wmma32"
            or raw_value.get("resources") != {"vgpr": 50, "lds_bytes": 4352,
                                               "scratch_bytes": 0}
            or Path(str(operator_executable.get("path", ""))).resolve() != qualifier.resolve()
            or operator_executable.get("sha256") != sha(qualifier)
            or not isinstance(operator_sources, list) or len(operator_sources) != 4
            or set(source_by_role) != set(expected_source_paths)
            or any(Path(str(source_by_role[role].get("path", ""))).resolve() != path
                   for role, path in expected_source_paths.items())
            or any(source_by_role[role].get("sha256") != sha(path)
                   for role, path in expected_source_paths.items() if role != "dispatch_profile")
            or source_by_role.get("dispatch_profile", {}).get("sha256") == sha(PROFILE_SOURCE)
            or any(row.get("oracle_max_bf16_steps", 3) > 2
                   or row.get("oracle")
                   != "independent FP64 represented A8G32-times-W8G32 formula"
                   or row.get("oracle_sample_count") != 64
                   or row.get("candidate_faster") is not True
                   or not (row.get("candidate_median_ms", float("inf"))
                           < row.get("baseline_median_ms", float("-inf")))
                   for row in shapes)):
        raise ValueError("MTP-bulk operator qualification is incomplete")
    if power_before.read_text(encoding="utf-8").strip() != "auto" or power_after.read_text(
            encoding="utf-8").strip() != "auto" or power_reader(POWER).strip() != "auto":
        raise ValueError("selected trace lacks auto power before/after/live evidence")
    isa = static_iu8(assembly, metadata)
    code_bytes = code_object.read_bytes()
    selected_executable = Path(route["benchmark"]["path"])
    if not code_bytes or selected_executable.read_bytes().count(code_bytes) != 1:
        raise ValueError("production W8 code object is not uniquely embedded in selected Engine")
    engine = _validate_engine_trace(route, command, benchmark_report, trace_database)
    selected_route = {
        "terminal_selection": route["terminal_selection"],
        "winner": route["winner"], "weights_id": MIXED_ID,
        "artifact": route["artifact"], "executable": route["benchmark"],
        "artifact_sha256": route["artifact"]["sha256"],
        "executable_sha256": route["benchmark"]["sha256"],
        "kv_value_group": route["cache_profile"]["value_group"],
        "xattention_profile": route["execution_profile"]["xattention_profile"],
        "prefill_chunk": route["selected_prefill_chunk"],
        "planner": snapshot(planner), "concurrency": 1,
    }
    return {
        "artifact_type": "ninfer_r9700_mixed_mtp_bulk_w8_evidence",
        "schema_version": 1, "status": "passed", "selected_route": selected_route,
        "quantization": {"weights": "W8G32_F16S", "activation": "signed A8G32",
                         "output": "BF16", "baseline_activation": "signed A8G32"},
        "native_hardware": isa, "selected_engine_execution": engine,
        "shapes": [{**row, "opcode": isa["opcode"]} for row in shapes],
        "authorities": {"plan": snapshot(plan_path), "operator_report": snapshot(raw),
                        "selected_engine_report": snapshot(benchmark_report),
                        "selected_engine_trace": snapshot(trace_database),
                        "production_code_object": snapshot(code_object),
                        "production_gfx1201_assembly": snapshot(assembly),
                        "production_elf_metadata": snapshot(metadata),
                        "qualifier": snapshot(qualifier), "planner": snapshot(planner),
                        "qualifier_source": snapshot(QUALIFIER_SOURCE),
                        "production_kernel_source": snapshot(KERNEL_SOURCE),
                        "dispatch_profile_source": snapshot(PROFILE_SOURCE)},
    }


def finalize(selection: Path, plan: Path, raw: Path, assembly: Path, metadata: Path,
             code_object: Path,
             benchmark_report: Path, trace_database: Path, power_before: Path,
             power_after: Path, qualifier: Path, planner: Path, out: Path, *,
             power_reader: Callable[[Path], str] = lambda path: path.read_text(encoding="utf-8"),
             ) -> dict:
    if os.path.lexists(out):
        raise ValueError(f"refusing to overwrite {out}")
    plan_value = json.loads(plan.read_text(encoding="utf-8"))
    if Path(str(plan_value.get("outputs", {}).get("evidence", ""))).resolve() != out.resolve():
        raise ValueError("evidence output path differs from the prepared plan")
    value = _assemble(selection, plan, raw, assembly, metadata, code_object, benchmark_report,
                      trace_database, power_before, power_after, qualifier, planner,
                      power_reader=power_reader)
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    temporary_path = Path(temporary)
    owner: tuple[int, int] | None = None
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2); output.write("\n")
            output.flush(); os.fsync(output.fileno())
            pending_stat = os.fstat(output.fileno())
            owner = (pending_stat.st_dev, pending_stat.st_ino)
        if not _owned(temporary_path, owner):
            raise ValueError("pending evidence inode changed before publication")
        os.link(temporary_path, out)
        published = True
        if not _owned(out, owner):
            raise ValueError("published evidence is not the owned pending inode")
        if json.loads(out.read_text(encoding="utf-8")) != value:
            raise ValueError("published evidence readback differs")
        fresh = _assemble(selection, plan, raw, assembly, metadata, code_object, benchmark_report,
                          trace_database, power_before, power_after, qualifier, planner,
                          power_reader=power_reader)
        if fresh != value:
            raise ValueError("MTP-bulk authorities changed during publication")
        directory = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _remove_owned(temporary_path, owner)
        return value
    except Exception:
        if owner is not None:
            if published:
                _remove_owned(out, owner)
            _remove_owned(temporary_path, owner)
        raise
    finally:
        if owner is None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--assembly", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--code-object", required=True, type=Path)
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument("--trace-database", required=True, type=Path)
    parser.add_argument("--power-before", required=True, type=Path)
    parser.add_argument("--power-after", required=True, type=Path)
    parser.add_argument("--qualifier", required=True, type=Path)
    parser.add_argument("--planner", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        finalize(args.selection, args.plan, args.raw, args.assembly, args.metadata, args.code_object,
                 args.benchmark_report, args.trace_database, args.power_before,
                 args.power_after, args.qualifier, args.planner, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
