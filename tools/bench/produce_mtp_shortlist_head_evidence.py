#!/usr/bin/env python3
"""Bind one executed whole-MTP3 shortlist-head dispatch to its exact gfx1201 code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from tools.artifact.container import Artifact, TensorObject
from tools.bench.produce_operation_static_evidence import (
    _embedded_code_object,
)
from tools.bench.extract_embedded_code_object import _snapshot as _filesystem_snapshot
from tools.bench.run_ninfer_bench_matrix import (
    MATRIX_SCHEMA_VERSION,
    PRODUCT_CONCURRENCIES,
    REPORT_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
    _validate_speculative,
    build_cases,
    load_bench_report,
)
from tools.bench.validate_profile_trace import _parse_database
from tools.ppl.assemble_pareto import _validate_mtp_target_parity
from tools.ppl.pareto import validate_terminal_production_authority


ARTIFACT_TYPE = "ninfer_r9700_mtp_shortlist_head_evidence"
SCHEMA_VERSION = 1
MODEL_ID = "qwen3.8-27b"
HEAD_NAME = "text/draft_head"
HEAD_SHAPE = (131072, 5120)
HEAD_FORMAT = "Q4G64_F16S"
HEAD_LAYOUT = "row-split-k128-v1"
HEAD_GRID_X = HEAD_SHAPE[0] // 16
HEAD_DISPATCH_COUNT = 3 * math.ceil(256 / 4)
IU4_OPCODE = "v_wmma_i32_16x16x32_iu4"
DEFAULT_OBJDUMP = Path("/opt/rocm/llvm/bin/llvm-objdump")
DEFAULT_READELF = Path("/opt/rocm/llvm/bin/llvm-readelf")
CODE_SYMBOL = (
    "_ZN6ninfer3ops5r97006linear12_GLOBAL__N_128a8q4g64_linear_wmma32_kernel"
    "EPKhS5_PKtPKjS5_S7_P12hip_bfloat16jjj"
)
DISPLAY_SYMBOL = (
    "ninfer::ops::r9700::linear::(anonymous namespace)::"
    "a8q4g64_linear_wmma32_kernel(unsigned char const*, unsigned char const*, "
    "unsigned short const*, unsigned int const*, unsigned char const*, unsigned short const*, "
    "hip_bfloat16*, unsigned int, unsigned int, unsigned int)"
)
SOURCE_AUTHORITIES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[2] / "src/ops/r9700/linear/r9700_linear.hip",
    Path(__file__).resolve().parents[2] / "src/ops/r9700/linear/r9700_q4_activation_profile.h",
    Path(__file__).resolve().parents[2] / "src/ops/r9700/linear/linear_tensor_op.cpp",
    Path(__file__).resolve().parents[2] / "src/artifact/typed_binding.cpp",
    Path(__file__).resolve().parents[2] / "src/targets/qwen3/impl/runtime/text_context_impl.h",
    Path(__file__).resolve().parents[2] / "src/targets/qwen3_8_27b/impl/load/bindings.cpp",
    Path(__file__).resolve().parents[2]
    / "tools/convert/qwen3_8_27b_r9700/source_inventory.py",
    Path(__file__).resolve().parents[2]
    / "tools/convert/qwen3_8_27b_r9700/source_recipe.py",
)


def _publish(path: Path, value: dict[str, Any]) -> None:
    """Durably publish one immutable JSON report or remove only our inode on failure."""

    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    published = False
    durable = False
    created_inode: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary_stat = os.stat(temporary, follow_symlinks=False)
        created_inode = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.link(temporary, path)
        published = True
        output_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(output_stat.st_mode)
            or (output_stat.st_dev, output_stat.st_ino) != created_inode
            or path.read_bytes() != payload
        ):
            raise ValueError("published shortlist-head evidence changed before durability")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        durable = True
    finally:
        Path(temporary).unlink(missing_ok=True)
        if published and not durable and created_inode is not None:
            try:
                current = os.stat(path, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == created_inode:
                    path.unlink()
            except FileNotFoundError:
                pass


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _snapshot(path: Path, label: str) -> dict[str, Any]:
    resolved, size, digest, device, inode, mode, mtime_ns = _filesystem_snapshot(path, label)
    return {
        "path": str(resolved), "file_size_bytes": size, "sha256": digest,
        "device": device, "inode": inode, "mode": mode, "mtime_ns": mtime_ns,
    }


def _identity_matches(identity: object, snapshot: dict[str, Any], label: str) -> None:
    if not isinstance(identity, dict) or (
        Path(str(identity.get("path", ""))).resolve() != Path(snapshot["path"])
        or identity.get("file_size_bytes") != snapshot["file_size_bytes"]
        or identity.get("sha256") != snapshot["sha256"]
    ):
        raise ValueError(f"{label} identity differs")


def _one_option(command: Sequence[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"whole-MTP3 command requires exactly one {option}")
    return command[positions[0] + 1]


def _replace_option(command: list[str], option: str, value: str) -> None:
    _one_option(command, option)
    command[command.index(option) + 1] = value


def _validate_artifact(path: Path, expected_weights_id: str) -> dict[str, Any]:
    with Artifact.open(path) as artifact:
        if artifact.identity.model_id != MODEL_ID or artifact.identity.weights_id != expected_weights_id:
            raise ValueError("selected artifact identity differs from the terminal winner")
        try:
            head = artifact.find(HEAD_NAME)
        except KeyError as error:
            raise ValueError(f"selected artifact lacks {HEAD_NAME}") from error
        if not isinstance(head, TensorObject) or (
            head.shape != HEAD_SHAPE or head.format != HEAD_FORMAT or head.layout != HEAD_LAYOUT
        ):
            raise ValueError(
                f"selected artifact must contain {HEAD_NAME} {HEAD_FORMAT}{list(HEAD_SHAPE)} "
                f"in {HEAD_LAYOUT}"
            )
        return {
            "name": head.name, "shape": list(head.shape), "format": head.format,
            "layout": head.layout, "offset": head.offset, "bytes": head.bytes,
        }


def _tool_output(
    tool: Path, arguments: list[str], code_object: Path, label: str,
    runner: Callable[..., subprocess.CompletedProcess],
) -> tuple[str, dict[str, Any]]:
    tool_snapshot = _snapshot(tool, label)
    command = [tool_snapshot["path"], *arguments, str(code_object)]
    try:
        result = runner(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{label} failed on the selected code object: {detail}") from error
    if not isinstance(result.stdout, bytes):
        raise ValueError(f"{label} did not return byte output")
    output = result.stdout.decode("utf-8", errors="strict")
    if _snapshot(Path(tool_snapshot["path"]), f"{label} after use") != tool_snapshot:
        raise ValueError(f"{label} changed while deriving code-object evidence")
    return output, tool_snapshot


def _metadata_resources(body: str) -> dict[str, int]:
    blocks = re.findall(r"(?ms)^  - \.args:.*?(?=^  - \.args:|\Z)", body)
    selected = [block for block in blocks if re.search(
        rf"^    \.name:\s+{re.escape(CODE_SYMBOL)}\s*$", block, flags=re.MULTILINE,
    )]
    if len(selected) != 1:
        raise ValueError("readelf metadata does not contain one exact shortlist-head kernel")
    block = selected[0]

    def integer(name: str) -> int:
        matches = re.findall(rf"^    \.{re.escape(name)}:\s+(\d+)\s*$", block, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"readelf metadata lacks one exact {name}")
        return int(matches[0])

    dynamic = re.findall(r"^    \.uses_dynamic_stack:\s+(true|false)\s*$", block, re.MULTILINE)
    if dynamic != ["false"]:
        raise ValueError("shortlist-head code object uses or omits dynamic-stack metadata")
    return {
        "lds_bytes": integer("group_segment_fixed_size"),
        "private_bytes": integer("private_segment_fixed_size"),
        "vgpr_count": integer("vgpr_count"),
        "sgpr_spill_count": integer("sgpr_spill_count"),
        "vgpr_spill_count": integer("vgpr_spill_count"),
        "wavefront_size": integer("wavefront_size"),
        "flat_scratch": 0,
        "scratch_bytes": integer("private_segment_fixed_size"),
    }


def _validate_iu4(
    code_object: Path,
    *,
    objdump: Path = DEFAULT_OBJDUMP,
    readelf: Path = DEFAULT_READELF,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    assembly_body, objdump_snapshot = _tool_output(
        objdump, [f"--disassemble-symbols={CODE_SYMBOL}"], code_object,
        "llvm-objdump", runner,
    )
    metadata_body, readelf_snapshot = _tool_output(
        readelf, ["--notes"], code_object, "llvm-readelf", runner,
    )
    headings = re.findall(r"^[0-9a-fA-F]+\s+<([^>]+)>:\s*$", assembly_body, re.MULTILINE)
    if headings != [CODE_SYMBOL]:
        raise ValueError("objdump did not isolate one exact shortlist-head function")
    wmma_lines = re.findall(r"^\s*(v_wmma_\S+.*)$", assembly_body, flags=re.MULTILINE)
    iu4_lines = [line for line in wmma_lines if line.split()[0] == IU4_OPCODE]
    if len(wmma_lines) != 4 or len(iu4_lines) != 4:
        raise ValueError("shortlist-head specialization must contain exactly four native IU4 WMMAs")
    signedness = {"unsigned_low_signed_weight": 0, "signed_high_signed_weight": 0}
    for line in iu4_lines:
        match = re.search(r"\bneg_lo:\[(\d),(\d),(\d)\](?:\s*//.*)?\s*$", line)
        if match is None:
            raise ValueError("native IU4 instruction lacks an explicit neg_lo signedness tuple")
        value = tuple(int(item) for item in match.groups())
        if value == (0, 1, 0):
            signedness["unsigned_low_signed_weight"] += 1
        elif value == (1, 1, 0):
            signedness["signed_high_signed_weight"] += 1
        else:
            raise ValueError(f"unexpected native IU4 signedness tuple {value}")
    if signedness != {"unsigned_low_signed_weight": 2, "signed_high_signed_weight": 2}:
        raise ValueError(f"shortlist-head IU4 low/high paths differ: {signedness}")
    resources = _metadata_resources(metadata_body)
    if resources["lds_bytes"] != 0 or not 1 <= resources["vgpr_count"] <= 64:
        raise ValueError("shortlist-head specialization requires zero LDS and at most 64 VGPRs")
    if any(resources[name] != 0 for name in ("private_bytes", "scratch_bytes", "flat_scratch")):
        raise ValueError("shortlist-head specialization must have zero private/scratch/flat-scratch")
    if (
        resources["sgpr_spill_count"] != 0
        or resources["vgpr_spill_count"] != 0
        or resources["wavefront_size"] != 32
    ):
        raise ValueError("shortlist-head specialization must be spill-free wave32")
    return signedness, resources, {
        "objdump": objdump_snapshot, "readelf": readelf_snapshot,
        "disassembly_sha256": hashlib.sha256(assembly_body.encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_body.encode()).hexdigest(),
        "derivation": "direct tool output from the exact uniquely embedded gfx1201 ELF",
    }


def _validate_report(
    report: dict[str, Any], command: list[str], workload: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, int]:
    config = report.get("config")
    tests = report.get("tests")
    expected_xattention = workload["xattention_profile"]
    if (
        report.get("artifact_type") != "ninfer_bench_report"
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("tool") != "ninfer_bench"
        or report.get("command") != " ".join(command)
        or not isinstance(config, dict)
        or not isinstance(tests, list) or len(tests) != 1
        or config.get("concurrency") != workload["concurrency"]
        or config.get("prefill_chunk") != workload["prefill_chunk"]
        or config.get("kv_value_group") != workload["kv_value_group"]
        or config.get("kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or config.get("spec") != "mtp" or config.get("draft_tokens") != 3
        or config.get("speculative_execution") is not True
        or config.get("proposal_head") != "optimized"
        or config.get("q4_activation_bits") != 8
        or config.get("w8_activation_bits") != 8
        or config.get("q4_prefill_cta_profile") != "m64n128-pingpong-production"
        or config.get("fp8_qk_wmma_enabled") is not True
        or config.get("fp8_qk_wmma_profile") != "t1-ge64-t2-ge320-t3plus-stream-v1"
        or config.get("fp8_qk_wmma_t1_min_context") != 64
        or config.get("fp8_qk_wmma_t2_min_context") != 320
        or config.get("dflash_verify_width_requested") != 0
        or config.get("dflash_verify_width") != 0
        or config.get("repetitions") != 1 or config.get("warmup") != 1
        or config.get("retain_token_ids") is not True
        or config.get("use_device_graph") is not True
        or report.get("load", {}).get("target") != "qwen3_8_27b_r9700"
        or report.get("load", {}).get("weights_id") != artifact.get("weights_id")
    ):
        raise ValueError("benchmark report is not the executed optimized whole-MTP3 trace row")
    if expected_xattention == "dense":
        if config.get("xattention_qualification") is not False or any(
            key in config for key in (
                "xattention_profile", "xattention_find_block", "xattention_stride",
                "xattention_tau_permille",
            )
        ):
            raise ValueError("dense trace report has ambiguous sparse-attention configuration")
    elif expected_xattention == "b128-s16-tau900":
        if any(config.get(key) != value for key, value in (
            ("xattention_qualification", True), ("xattention_profile", expected_xattention),
            ("xattention_find_block", 128), ("xattention_stride", 16),
            ("xattention_tau_permille", 900),
        )):
            raise ValueError("sparse trace report differs from the selected XAttention profile")
    else:
        raise ValueError("terminal winner has an unsupported XAttention profile")
    report_artifact = report.get("artifact")
    if not isinstance(report_artifact, dict) or (
        Path(str(report_artifact.get("path", ""))).resolve() != Path(artifact["path"])
        or report_artifact.get("file_size_bytes") != artifact["file_size_bytes"]
    ):
        raise ValueError("benchmark-report artifact identity differs")
    row = tests[0]
    if any(row.get(key) != value for key, value in (
        ("kind", "whole"), ("n_prompt", workload["prompt_tokens"]),
        ("n_gen", workload["generated_tokens"]),
    )):
        raise ValueError("benchmark report geometry differs from the selected whole trace")
    spec = row.get("speculative")
    try:
        _validate_speculative(
            spec, enabled=True, draft_window=3, require_sample=True,
            require_accepted=True, label="selected whole-MTP3 trace",
        )
    except ValueError as error:
        raise ValueError("whole-MTP3 trace has no coherent executed, accepted proposal sample") from error
    expected_rounds = workload["concurrency"] * math.ceil(workload["generated_tokens"] / 4)
    if spec["rounds"] != expected_rounds:
        raise ValueError(
            "whole-MTP3 trace requires theoretical-minimum lane-summed rounds: "
            f"observed {spec['rounds']}, expected {expected_rounds}"
        )
    reps = row.get("reps")
    if not isinstance(reps, list) or len(reps) != 1 or not isinstance(reps[0], dict):
        raise ValueError("whole-MTP3 trace must retain exactly one repetition")
    repetition_spec = reps[0].get("speculative")
    try:
        _validate_speculative(
            repetition_spec, enabled=True, draft_window=3, require_sample=True,
            require_accepted=True, label="selected whole-MTP3 trace repetition",
        )
    except ValueError as error:
        raise ValueError("whole-MTP3 trace repetition has incoherent speculative counters") from error
    if repetition_spec != spec or repetition_spec["rounds"] != expected_rounds:
        raise ValueError("whole-MTP3 one-repetition counters differ from the row aggregate")
    expected_drafts = expected_rounds * 3
    if spec["drafted_tokens"] != expected_drafts:
        raise ValueError(
            "theoretical-minimum whole-MTP3 trace requires every proposal position "
            f"drafted: observed {spec['drafted_tokens']}, expected {expected_drafts}"
        )
    return {
        **{key: spec[key] for key in ("rounds", "drafted_tokens", "accepted_tokens")},
        "expected_rounds": expected_rounds, "repetition_count": 1,
    }


def _validate_source_matrix(
    plan: dict[str, Any], workload: dict[str, Any], command: list[str],
    profile_report: Path, artifact: dict[str, Any], executable: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = plan.get("source_matrix")
    if not isinstance(source, dict) or source.get("preset") != "pareto-whole":
        raise ValueError("profile plan lacks its selected pareto-whole matrix authority")
    manifest_snapshot = _snapshot(Path(str(source.get("path", ""))), "source matrix")
    report_identity = source.get("report")
    source_report_snapshot = _snapshot(
        Path(str(report_identity.get("path", ""))) if isinstance(report_identity, dict) else Path(""),
        "source whole-MTP3 report",
    )
    if (
        source.get("sha256") != manifest_snapshot["sha256"]
        or not isinstance(report_identity, dict)
        or report_identity.get("sha256") != source_report_snapshot["sha256"]
    ):
        raise ValueError("profile plan source matrix/report authority bytes differ")
    manifest_path = Path(manifest_snapshot["path"])
    if os.path.lexists(manifest_path.parent / "failures.json"):
        raise ValueError("source pareto-whole matrix retains failures")
    manifest = _load(manifest_path, "source pareto-whole matrix")
    if (
        manifest.get("artifact_type") != "ninfer_bench_matrix_run"
        or manifest.get("schema_version") != MATRIX_SCHEMA_VERSION
        or manifest.get("preset") != "pareto-whole"
        or sorted(manifest.get("concurrency", [])) != list(PRODUCT_CONCURRENCIES)
        or manifest.get("expected_kv_value_group") != workload["kv_value_group"]
        or manifest.get("expected_kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or manifest.get("expected_q4_activation_bits") != 8
        or manifest.get("expected_w8_activation_bits") != 8
        or manifest.get("expected_fp8_qk_wmma_enabled") is not True
        or manifest.get("expected_fp8_qk_wmma_profile")
        != "t1-ge64-t2-ge320-t3plus-stream-v1"
        or manifest.get("expected_fp8_qk_wmma_t1_min_context") != 64
        or manifest.get("expected_fp8_qk_wmma_t2_min_context") != 320
        or manifest.get("expected_xattention_profile") != workload["xattention_profile"]
        or manifest.get("selected_prefill_chunk") != workload["prefill_chunk"]
        or manifest.get("artifact") != artifact or manifest.get("bench") != executable
    ):
        raise ValueError("source matrix is not the complete selected C1..4 static profile")
    all_records = manifest.get("commands")
    expected_points = {
        (suite, concurrency)
        for suite in ("pareto_whole_inference", "pareto_whole_control")
        for concurrency in PRODUCT_CONCURRENCIES
    }
    actual_points = [
        (row.get("suite"), row.get("concurrency"))
        for row in all_records if isinstance(row, dict)
    ] if isinstance(all_records, list) else []
    if len(actual_points) != 8 or len(set(actual_points)) != 8 or set(actual_points) != expected_points:
        raise ValueError("source matrix lacks the exact MTP3/control C1..4 point set")
    records = [
        row for row in all_records
        if isinstance(row, dict) and row.get("suite") == "pareto_whole_inference"
        and row.get("concurrency") == workload["concurrency"]
    ]
    if len(records) != 1 or Path(str(records[0].get("report", ""))).resolve() != Path(
        source_report_snapshot["path"]
    ):
        raise ValueError("source matrix lacks one exact selected whole-MTP3 row")
    source_command = records[0].get("command")
    if not isinstance(source_command, list) or any(not isinstance(item, str) for item in source_command):
        raise ValueError("source whole-MTP3 command is invalid")
    expected = list(source_command)
    _replace_option(expected, "--whole-pg", f"{workload['prompt_tokens']},{workload['generated_tokens']}")
    _replace_option(expected, "-r", "1")
    _replace_option(expected, "--output-file", str(profile_report.resolve()))
    if "--profile-measured" not in expected:
        expected.append("--profile-measured")
    if command != expected:
        raise ValueError("profile command is not derived from the selected matrix command")
    cases = {
        case.suite: case for case in build_cases(
            "pareto-whole", production_prefill_chunk=workload["prefill_chunk"]
        )
    }
    validated: dict[tuple[str, int], dict[str, Any]] = {}
    report_snapshots: dict[str, dict[str, Any]] = {}
    report_paths: set[Path] = set()
    for record in all_records:
        suite, concurrency = record["suite"], record["concurrency"]
        report_path = Path(str(record.get("report", ""))).resolve()
        record_command = record.get("command")
        if report_path in report_paths:
            raise ValueError("source matrix report paths must be unique per C1..4 point")
        report_paths.add(report_path)
        snapshot = _snapshot(report_path, f"source {suite} C{concurrency} report")
        report_snapshots[f"{suite}.c{concurrency}"] = snapshot
        if not isinstance(record_command, list) or any(
            not isinstance(item, str) for item in record_command
        ):
            raise ValueError("source matrix contains an invalid benchmark command")
        validated[(suite, concurrency)] = load_bench_report(
            report_path,
            expected_kv_value_group=workload["kv_value_group"],
            expected_q4_activation_bits=8,
            expected_concurrency=concurrency,
            expected_artifact=artifact,
            expected_command=record_command,
            expected_case=cases[suite],
            expected_xattention_profile=workload["xattention_profile"],
        )
    selected_key = f"pareto_whole_inference.c{workload['concurrency']}"
    if report_snapshots[selected_key] != source_report_snapshot:
        raise ValueError("profile plan selected report differs from its matrix record")
    round_cells: dict[str, Any] = {}
    for concurrency in PRODUCT_CONCURRENCIES:
        _, parity = _validate_mtp_target_parity([
            validated[("pareto_whole_inference", concurrency)],
            validated[("pareto_whole_control", concurrency)],
        ], concurrency)
        round_cells.update({
            f"{label}_c{concurrency}": detail
            for label, detail in parity["round_gate"]["cells"].items()
        })
    return {"manifest": manifest_snapshot, **report_snapshots}, round_cells


def produce(
    *, plan_path: Path, benchmark_report: Path, trace_database: Path,
    power_before: Path, power_after: Path, terminal_selection: Path,
    artifact: Path, executable: Path, code_object: Path,
    dispatch_symbol: str, stages: list[str],
    tool_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    if dispatch_symbol != DISPLAY_SYMBOL:
        raise ValueError("dispatch symbol must be the exact traced A8Q4 WMMA32 display symbol")
    if not stages or any(not stage for stage in stages) or len(set(stages)) != len(stages):
        raise ValueError("one or more unique exact trace stages are required")
    paths = {
        "plan": plan_path, "benchmark_report": benchmark_report,
        "trace_database": trace_database, "power_before": power_before,
        "power_after": power_after, "terminal_selection": terminal_selection,
        "artifact": artifact, "executable": executable, "code_object": code_object,
    }
    snapshots = {name: _snapshot(path, name) for name, path in paths.items()}
    sources = [_snapshot(path, f"source {path.name}") for path in SOURCE_AUTHORITIES]
    plan = _load(Path(snapshots["plan"]["path"]), "profile plan")
    workload = plan.get("workload")
    command = plan.get("benchmark_command")
    terminal_identity = plan.get("terminal_selection")
    if (
        plan.get("artifact_type") != "ninfer_whole_profile_plan"
        or plan.get("schema_version") != 2 or plan.get("status") != "command_only_not_executed"
        or plan.get("profile_kind") != "trace" or plan.get("measured_region") != "ninfer_bench_measured"
        or plan.get("kernel_include_regex") is not None or plan.get("counters") != []
        or not isinstance(workload, dict) or workload.get("prompt_tokens") != 8192
        or workload.get("generated_tokens") != 256
        or workload.get("concurrency") != 1
        or workload.get("spec") != "mtp" or workload.get("draft_tokens") != 3
        or workload.get("dflash_verify_width") != 0
        or workload.get("kv_value_group") not in (16, 32)
        or workload.get("kv_plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or type(workload.get("prefill_chunk")) is not int
        or not isinstance(command, list) or any(not isinstance(item, str) for item in command)
    ):
        raise ValueError("profile plan is not the exact selected C1/8K+256 whole-MTP3 trace")
    if (
        not isinstance(terminal_identity, dict)
        or Path(str(terminal_identity.get("path", ""))).resolve()
        != Path(snapshots["terminal_selection"]["path"])
        or terminal_identity.get("sha256") != snapshots["terminal_selection"]["sha256"]
    ):
        raise ValueError("profile plan terminal-selection identity differs")
    if Path(str(plan.get("expected_trace_database", ""))).resolve() != Path(
        snapshots["trace_database"]["path"]
    ):
        raise ValueError("profile plan fixed trace database differs")
    artifact_identity = plan.get("artifact")
    executable_identity = plan.get("benchmark_executable")
    _identity_matches(artifact_identity, snapshots["artifact"], "plan artifact")
    _identity_matches(executable_identity, snapshots["executable"], "plan executable")
    if (
        command[0] != snapshots["executable"]["path"]
        or Path(_one_option(command, "--weights")).resolve() != Path(snapshots["artifact"]["path"])
        or _one_option(command, "--spec") != "mtp"
        or _one_option(command, "--draft-tokens") != "3"
        or command.count("--lm-head-draft") != 1
        or _one_option(command, "--whole-pg")
        != f"{workload['prompt_tokens']},{workload['generated_tokens']}"
        or _one_option(command, "--concurrency") != str(workload["concurrency"])
        or _one_option(command, "--prefill-chunk") != str(workload["prefill_chunk"])
        or _one_option(command, "-r") != "1"
        or command.count("--profile-measured") != 1
        or Path(_one_option(command, "--output-file")).resolve()
        != Path(snapshots["benchmark_report"]["path"])
    ):
        raise ValueError("profile command is not the exact optimized whole-MTP3 derivation")
    source_matrix, source_round_cells = _validate_source_matrix(
        plan, workload, command, Path(snapshots["benchmark_report"]["path"]),
        artifact_identity, executable_identity,
    )

    terminal_value = _load(Path(snapshots["terminal_selection"]["path"]), "terminal selection")
    terminal, selected = validate_terminal_production_authority(terminal_value)
    winner_artifact = terminal.get("winner_artifact")
    winner_cache = terminal.get("winner_cache_profile")
    winner_execution = terminal.get("winner_execution_profile")
    if (
        terminal.get("production_status")
        != "selected_route_pending_shortlist_head_trace_and_niah"
        or terminal.get("shortlist_head_precision_status")
        != "q4_round_gate_pass_pending_trace"
    ):
        raise ValueError(
            "terminal selection requires conditional head precision; Q4 trace cannot finalize it"
        )
    if (
        not isinstance(winner_artifact, dict)
        or winner_artifact.get("weights_id") != artifact_identity.get("weights_id")
        or winner_artifact.get("sha256") != snapshots["artifact"]["sha256"]
        or not isinstance(winner_cache, dict)
        or winner_cache.get("value_group") != workload["kv_value_group"]
        or winner_cache.get("plane_layouts") != R9700_KV_PLANE_LAYOUTS
        or not isinstance(winner_execution, dict)
        or winner_execution.get("xattention_profile") != workload.get("xattention_profile")
        or winner_execution.get("q4_activation_bits") != 8
        or winner_execution.get("w8_activation_bits") != 8
        or winner_execution.get("fp8_qk_wmma_profile")
        != "t1-ge64-t2-ge320-t3plus-stream-v1"
        or selected.get("prefill_chunk") != workload["prefill_chunk"]
    ):
        raise ValueError("trace route differs from the recomputed terminal production winner")
    terminal_head_gate = terminal.get("shortlist_head_precision_gate")
    if not isinstance(terminal_head_gate, dict) or terminal_head_gate.get("cells") != source_round_cells:
        raise ValueError("terminal shortlist-head round gate differs from the reopened source matrix")
    head = _validate_artifact(Path(snapshots["artifact"]["path"]), winner_artifact["weights_id"])
    report = _load(Path(snapshots["benchmark_report"]["path"]), "benchmark report")
    acceptance = _validate_report(report, command, workload, artifact_identity)
    if Path(snapshots["power_before"]["path"]).read_text().strip() != "auto" or Path(
        snapshots["power_after"]["path"]
    ).read_text().strip() != "auto":
        raise ValueError("whole-MTP3 trace lacks auto power-profile endpoint evidence")

    dispatches, aggregates = _parse_database(
        Path(snapshots["trace_database"]["path"]), " ".join(command)
    )
    shape_rows = [
        row for row in dispatches
        if row["grid"] == {"x": HEAD_GRID_X, "y": 1, "z": 1}
        and row["symbol"] == DISPLAY_SYMBOL
    ]
    if not shape_rows:
        raise ValueError("trace contains no executed [131072,5120] A8Q4 shortlist-head dispatch")
    symbols = {row["symbol"] for row in shape_rows}
    observed_stages = {row["roctx_region"] for row in shape_rows}
    if symbols != {dispatch_symbol} or observed_stages != set(stages):
        raise ValueError(
            "shortlist-head trace is ambiguous across display symbols or sparse/dense stages: "
            f"symbols={sorted(symbols)!r} stages={sorted(observed_stages, key=str)!r}"
        )
    for row in shape_rows:
        if row["workgroup"] != {"x": 32, "y": 1, "z": 1}:
            raise ValueError("shortlist-head trace dispatch is not the wave32 specialization")
        resources = row["resources"]
        if any(resources.get(name) not in (None, 0) for name in (
            "lds_bytes", "scratch_bytes", "static_lds_bytes", "static_scratch_bytes",
        )):
            raise ValueError("traced shortlist-head dispatch reports LDS or scratch use")
    drafted = acceptance["drafted_tokens"]
    concurrency = workload["concurrency"]
    if (
        drafted % concurrency != 0
        or drafted // concurrency != HEAD_DISPATCH_COUNT
        or len(shape_rows) != HEAD_DISPATCH_COUNT
    ):
        raise ValueError(
            "shortlist-head trace dispatch multiplicity differs from the exact MTP3 g256 count: "
            f"observed {len(shape_rows)}, drafted/concurrency={drafted // concurrency}, "
            f"expected {HEAD_DISPATCH_COUNT}"
        )

    embedding = _embedded_code_object(
        Path(snapshots["executable"]["path"]), Path(snapshots["code_object"]["path"])
    )
    signedness, resources, derivation = _validate_iu4(
        Path(snapshots["code_object"]["path"]), runner=tool_runner,
    )
    for row in shape_rows:
        traced_vgprs = row["resources"].get("vgpr_count")
        if traced_vgprs is not None and traced_vgprs != resources["vgpr_count"]:
            raise ValueError("trace and selected code-object VGPR counts differ")

    for name, snapshot in snapshots.items():
        if _snapshot(Path(snapshot["path"]), name) != snapshot:
            raise ValueError(f"{name} changed while producing shortlist-head evidence")
    for snapshot in sources:
        if _snapshot(Path(snapshot["path"]), "source") != snapshot:
            raise ValueError("source changed while producing shortlist-head evidence")
    for name, snapshot in source_matrix.items():
        if _snapshot(Path(snapshot["path"]), f"source matrix {name}") != snapshot:
            raise ValueError(f"source matrix {name} changed while producing shortlist-head evidence")
    if os.path.lexists(Path(source_matrix["manifest"]["path"]).parent / "failures.json"):
        raise ValueError("source pareto-whole failures appeared while producing evidence")
    return {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "status": "passed", "operation_family": "mtp_optimized_shortlist_head",
        "selected_route": {
            "terminal_selection": snapshots["terminal_selection"],
            "winner": terminal["winner"], "artifact": artifact_identity,
            "benchmark_executable": executable_identity, **workload,
        },
        "tensor": head,
        "executed_trace": {
            "plan": snapshots["plan"], "benchmark_report": snapshots["benchmark_report"],
            "source_matrix": source_matrix,
            "database": snapshots["trace_database"], "power_before": snapshots["power_before"],
            "power_after": snapshots["power_after"], "acceptance": acceptance,
            "display_symbol": dispatch_symbol, "stages": stages,
            "dispatch_count": len(shape_rows), "dispatch_ids": [row["dispatch_id"] for row in shape_rows],
            "expected_dispatch_count": HEAD_DISPATCH_COUNT,
            "grid": {"x": HEAD_GRID_X, "y": 1, "z": 1},
            "workgroup": {"x": 32, "y": 1, "z": 1}, "trace_aggregates": aggregates,
        },
        "static_proof": {
            "code_symbol": CODE_SYMBOL, "code_object": snapshots["code_object"],
            "executable_embedding": embedding, "opcode": IU4_OPCODE,
            "opcode_count": 4, "signedness_paths": signedness,
            "resources": resources, "zero_scratch": True, "sources": sources,
            "binary_derivation": derivation,
            "quantization": {
                "weights": {
                    "format": HEAD_FORMAT, "codes": "packed signed int4",
                    "group_size": 64, "scales": "FP16 per output-row/G64 group",
                    "resident_binding": "direct artifact object; no runtime weight repack",
                },
                "activations": {
                    "represented_input": "BF16", "codec": "signed A8G64",
                    "planes": "packed unsigned low int4 and signed high int4",
                    "reconstruction": "a8 = low + 16 * high",
                    "scales": "FP16 per token/G64 group",
                },
                "arithmetic": "four native IU4 WMMAs -> I32 low/high recombination -> FP32 scale/FMA accumulation -> BF16 output",
            },
        },
        "profile_binding": {
            "xattention_profile": workload["xattention_profile"],
            "source": "recomputed terminal tuple plus exact report configuration; never symbol inference",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "plan", "benchmark-report", "trace-database", "power-before", "power-after",
        "terminal-selection", "artifact", "executable", "code-object",
    ):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--dispatch-symbol", required=True)
    parser.add_argument("--stage", required=True, action="append")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    try:
        result = produce(
            plan_path=args.plan, benchmark_report=args.benchmark_report,
            trace_database=args.trace_database, power_before=args.power_before,
            power_after=args.power_after, terminal_selection=args.terminal_selection,
            artifact=args.artifact, executable=args.executable, code_object=args.code_object,
            dispatch_symbol=args.dispatch_symbol, stages=args.stage,
        )
        _publish(args.out, result)
    except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote executed MTP shortlist-head evidence to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
