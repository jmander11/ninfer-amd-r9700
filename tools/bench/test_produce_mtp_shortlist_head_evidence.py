#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.artifact.container import (
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    PREFIX_BYTES,
    ArtifactIdentity,
    TensorObject,
    encode_directory,
)
from tools.artifact.layouts import align_up, encoded_size
from tools.bench.produce_mtp_shortlist_head_evidence import (
    CODE_SYMBOL,
    DISPLAY_SYMBOL,
    _publish,
    main,
    produce,
)
from tools.bench.finalize_selected_mtp_shortlist_head import exact_trace_signature, finalize
from tools.bench.run_ninfer_bench_matrix import (
    MATRIX_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    R9700_KV_PLANE_LAYOUTS,
)


class MtpShortlistHeadEvidenceTest(unittest.TestCase):
    def _identity(self, path: Path) -> dict:
        return {
            "path": str(path.resolve()), "file_size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _artifact(self, path: Path, *, format_name: str = "Q4G64_F16S") -> None:
        size = encoded_size("row-split-k128-v1", format_name, (131072, 5120))
        obj = TensorObject(
            "text/draft_head", (131072, 5120), format_name,
            "row-split-k128-v1", 0, size,
        )
        directory = encode_directory(ArtifactIdentity("qwen3.8-27b", "selected"), [obj])
        payload = align_up(PREFIX_BYTES + len(directory), PAYLOAD_ALIGNMENT)
        with path.open("wb") as output:
            output.write(PREFIX.pack(MAGIC, len(directory)))
            output.write(directory)
            output.seek(payload + size - 1)
            output.write(b"\0")

    def fixture(self, root: Path, *, sparse: bool = False) -> dict:
        artifact = root / "model.ninfer"
        executable = root / "ninfer_bench"
        code_object = root / "selected.co"
        report_path = root / "benchmark-report.json"
        plan_path = root / "plan.json"
        matrix = root / "matrix"
        matrix.mkdir()
        source_report = matrix / "whole.json"
        manifest_path = matrix / "manifest.json"
        database = root / "trace.db"
        before, after = root / "before.txt", root / "after.txt"
        terminal = root / "terminal.json"
        self._artifact(artifact)
        code_object.write_bytes(b"\x7fELF-unique-" + CODE_SYMBOL.encode())
        executable.write_bytes(b"host-prefix" + code_object.read_bytes() + b"host-suffix")
        wmma = "\n".join([
            f"\t{('v_wmma_i32_16x16x32_iu4')} v[0:7], v[8:9], v[10:11], 0 neg_lo:[0,1,0]",
            f"\t{('v_wmma_i32_16x16x32_iu4')} v[8:15], v[8:9], v[10:11], 0 neg_lo:[1,1,0]",
            f"\t{('v_wmma_i32_16x16x32_iu4')} v[0:7], v[8:9], v[10:11], v[0:7] neg_lo:[0,1,0]",
            f"\t{('v_wmma_i32_16x16x32_iu4')} v[8:15], v[8:9], v[10:11], v[8:15] neg_lo:[1,1,0]",
        ])
        objdump_output = f"""selected.co: file format elf64-amdgpu

0000000000000100 <{CODE_SYMBOL}>:
{wmma}
"""
        readelf_output = f"""Displaying notes found in: .note
    AMDGPU Metadata:
        ---
amdhsa.kernels:
  - .args:
      - .offset: 0
        .size: 8
    .group_segment_fixed_size: 0
    .name:           {CODE_SYMBOL}
    .private_segment_fixed_size: 0
    .sgpr_spill_count: 0
    .symbol:         {CODE_SYMBOL}.kd
    .uses_dynamic_stack: false
    .vgpr_count:     64
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .offset: 0
        .size: 8
    .group_segment_fixed_size: 0
    .name:           unrelated_kernel
    .private_segment_fixed_size: 0
    .sgpr_spill_count: 0
    .uses_dynamic_stack: false
    .vgpr_count: 1
    .vgpr_spill_count: 0
    .wavefront_size: 32
"""
        before.write_text("auto\n", encoding="utf-8")
        after.write_text("auto\n", encoding="utf-8")
        terminal.write_text("{}", encoding="utf-8")
        artifact_identity = {**self._identity(artifact), "weights_id": "selected"}
        executable_identity = self._identity(executable)
        profile = "b128-s16-tau900" if sparse else "dense"
        xattention = ({"xattention_qualification": False} if not sparse else {
            "xattention_qualification": True, "xattention_profile": profile,
            "xattention_find_block": 128, "xattention_stride": 16,
            "xattention_tau_permille": 900,
        })

        def speculative(enabled: bool, concurrency: int) -> dict:
            rounds = concurrency * 64
            if not enabled:
                return {
                    "enabled": False, "draft_window": 0, "rounds": 0,
                    "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0,
                    "acceptance_rate": None, "acceptance_length": None,
                    "accepted_per_position": [],
                }
            return {
                "enabled": True, "draft_window": 3, "rounds": rounds,
                "drafted_tokens": rounds * 3, "accepted_tokens": rounds * 3,
                "fallback_steps": 0, "acceptance_rate": 1.0,
                "acceptance_length": 4.0,
                "accepted_per_position": [rounds, rounds, rounds],
            }

        def whole_row(tokens: int, concurrency: int, enabled: bool) -> dict:
            rep_spec = speculative(enabled, concurrency)
            aggregate = {
                key: (
                    value * 3
                    if key in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps")
                    else [item * 3 for item in value]
                    if key == "accepted_per_position"
                    else value
                )
                for key, value in rep_spec.items()
            }
            return {
                "label": f"whole-pp{tokens}+tg256", "kind": "whole",
                "n_prompt": tokens, "n_gen": 256, "requested_output_tokens": 257,
                "workspace_peak_bytes": 1, "workspace_allocator_peak_bytes": 1,
                "prefill_seconds_mean": 1.0, "prefill_tok_s_mean": float(tokens),
                "decode_seconds_mean": 1.0, "decode_output_tok_s_mean": 256.0,
                "decode_engine_tok_s_mean": 256.0, "whole_output_tok_s_mean": 128.0,
                "total_seconds_mean": 2.0, "speculative": aggregate,
                "reps": [{
                    "generated_token_ids_by_lane": [[1] * 257 for _ in range(concurrency)],
                    "speculative": rep_spec,
                } for _ in range(3)],
            }

        def source_config(concurrency: int, enabled: bool) -> dict:
            return {
                "concurrency": concurrency, "prefill_chunk": 4096,
                "kv_value_group": 16, "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
                "q4_activation_bits": 8, "w8_activation_bits": 8,
                "q4_prefill_cta_profile": "m64n128-pingpong-production",
                "fp8_qk_wmma_enabled": True,
                "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                "fp8_qk_wmma_t1_min_context": 64,
                "fp8_qk_wmma_t2_min_context": 320,
                "pending_timeout_ms": 0xFFFFFFFF, "pending_deadline": "unbounded",
                "spec": "mtp" if enabled else "none", "draft_tokens": 3 if enabled else 0,
                "speculative_execution": enabled,
                "dflash_verify_width_requested": 0, "dflash_verify_width": 0,
                "proposal_head": "optimized" if enabled else "full",
                "use_device_graph": True, "retain_token_ids": True,
                "repetitions": 3, "warmup": 1, **xattention,
            }

        records = []
        source_command = None
        for suite, enabled in (
            ("pareto_whole_inference", True), ("pareto_whole_control", False),
        ):
            for concurrency in (1, 2, 3, 4):
                point_report = matrix / f"{suite}.c{concurrency}.json"
                case_args = [
                    "--whole-pg", "8192,256;32768,256", "--prefill-chunk", "4096",
                    "--spec", "mtp", "--draft-tokens", "3" if enabled else "0",
                ]
                if enabled:
                    case_args.append("--lm-head-draft")
                case_args.append("--retain-token-ids")
                point_command = [
                    str(executable.resolve()), "--weights", str(artifact.resolve()),
                    "--concurrency", str(concurrency), *case_args,
                    "--output-file", str(point_report.resolve()),
                    "-r", "3", "--warmup", "1",
                ]
                point_report.write_text(json.dumps({
                    "artifact_type": "ninfer_bench_report",
                    "schema_version": REPORT_SCHEMA_VERSION, "tool": "ninfer_bench",
                    "command": " ".join(point_command),
                    "artifact": {"path": str(artifact.resolve()),
                                 "file_size_bytes": artifact.stat().st_size},
                    "load": {"target": "qwen3_8_27b_r9700", "weights_id": "selected"},
                    "config": source_config(concurrency, enabled),
                    "memory": {"kv_capacity_mode": "explicit"},
                    "tests": [whole_row(tokens, concurrency, enabled)
                              for tokens in (8192, 32768)],
                }), encoding="utf-8")
                records.append({
                    "suite": suite, "concurrency": concurrency,
                    "report": str(point_report.resolve()), "command": point_command,
                })
                if (suite, concurrency) == ("pareto_whole_inference", 1):
                    source_command = point_command
                    source_report = point_report
        assert source_command is not None
        command = list(source_command)
        command[command.index("--whole-pg") + 1] = "8192,256"
        command[command.index("--output-file") + 1] = str(report_path.resolve())
        command[command.index("-r") + 1] = "1"
        command.append("--profile-measured")
        workload = {
            "concurrency": 1, "prompt_tokens": 8192, "generated_tokens": 256,
            "spec": "mtp", "draft_tokens": 3, "dflash_verify_width": 0,
            "kv_value_group": 16, "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "xattention_profile": profile, "prefill_chunk": 4096,
        }
        manifest_path.write_text(json.dumps({
            "artifact_type": "ninfer_bench_matrix_run", "schema_version": MATRIX_SCHEMA_VERSION,
            "preset": "pareto-whole", "concurrency": [1, 2, 3, 4],
            "expected_kv_value_group": 16,
            "expected_kv_plane_layouts": R9700_KV_PLANE_LAYOUTS,
            "expected_q4_activation_bits": 8, "expected_w8_activation_bits": 8,
            "expected_fp8_qk_wmma_enabled": True,
            "expected_fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
            "expected_fp8_qk_wmma_t1_min_context": 64,
            "expected_fp8_qk_wmma_t2_min_context": 320,
            "expected_xattention_profile": profile,
            "selected_prefill_chunk": 4096, "artifact": artifact_identity,
            "bench": executable_identity, "commands": records,
        }), encoding="utf-8")
        plan_path.write_text(json.dumps({
            "artifact_type": "ninfer_whole_profile_plan", "schema_version": 2,
            "status": "command_only_not_executed", "profile_kind": "trace",
            "measured_region": "ninfer_bench_measured", "kernel_include_regex": None,
            "counters": [], "artifact": artifact_identity,
            "benchmark_executable": executable_identity, "workload": workload,
            "benchmark_command": command,
            "terminal_selection": {
                "path": str(terminal.resolve()),
                "sha256": hashlib.sha256(terminal.read_bytes()).hexdigest(),
            },
            "expected_trace_database": str(database.resolve()),
            "source_matrix": {
                "path": str(manifest_path.resolve()),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "preset": "pareto-whole",
                "report": {"path": str(source_report.resolve()),
                           "sha256": hashlib.sha256(source_report.read_bytes()).hexdigest()},
            },
        }), encoding="utf-8")
        report_path.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": REPORT_SCHEMA_VERSION,
            "tool": "ninfer_bench",
            "command": " ".join(command),
            "artifact": {"path": str(artifact.resolve()),
                         "file_size_bytes": artifact.stat().st_size},
            "load": {"target": "qwen3_8_27b_r9700", "weights_id": "selected"},
            "config": {
                "concurrency": 1, "prefill_chunk": 4096, "kv_value_group": 16,
                "kv_plane_layouts": R9700_KV_PLANE_LAYOUTS, "spec": "mtp",
                "draft_tokens": 3, "speculative_execution": True,
                "proposal_head": "optimized", "q4_activation_bits": 8,
                "w8_activation_bits": 8,
                "q4_prefill_cta_profile": "m64n128-pingpong-production",
                "fp8_qk_wmma_enabled": True,
                "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                "fp8_qk_wmma_t1_min_context": 64,
                "fp8_qk_wmma_t2_min_context": 320,
                "dflash_verify_width_requested": 0, "dflash_verify_width": 0,
                "repetitions": 1, "warmup": 1, "retain_token_ids": True,
                "use_device_graph": True, **xattention,
            },
            "tests": [{
                "kind": "whole", "n_prompt": 8192, "n_gen": 256,
                "speculative": {"enabled": True, "draft_window": 3, "rounds": 64,
                                "drafted_tokens": 192, "accepted_tokens": 180,
                                "fallback_steps": 0, "acceptance_rate": 0.9375,
                                "acceptance_length": 3.8125,
                                "accepted_per_position": [64, 64, 52]},
                "reps": [{"speculative": {
                    "enabled": True, "draft_window": 3, "rounds": 64,
                    "drafted_tokens": 192, "accepted_tokens": 180,
                    "fallback_steps": 0, "acceptance_rate": 0.9375,
                    "acceptance_length": 3.8125,
                    "accepted_per_position": [64, 64, 52],
                }}],
            }],
        }), encoding="utf-8")
        display = DISPLAY_SYMBOL
        stage = "ninfer.mtp.decode payload=0"
        connection = sqlite3.connect(database)
        connection.executescript("""
            create table rocpd_info_process_x(id integer, command text);
            create table rocpd_info_agent_x(type text, name text, product_name text,
                                             absolute_index integer);
            create table regions(start integer, "end" integer, extdata text);
            create table memory_copies(start integer, "end" integer, duration integer,
                                       name text, region_name text, size integer);
            create table kernels(
                dispatch_id integer, name text, region text, start integer, "end" integer,
                duration integer, stream_id integer, agent_abs_index integer,
                grid_x integer, grid_y integer, grid_z integer,
                workgroup_x integer, workgroup_y integer, workgroup_z integer,
                vgpr_count integer, accum_vgpr_count integer, lds_size integer,
                scratch_size integer, static_lds_size integer, static_scratch_size integer);
        """)
        connection.execute("insert into rocpd_info_process_x values (?,?)", (1, " ".join(command)))
        connection.execute(
            "insert into rocpd_info_agent_x values (?,?,?,?)",
            ("GPU", "gfx1201", "AMD Radeon AI PRO R9700", 1),
        )
        connection.executemany(
            "insert into kernels values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(index, display, stage, index * 10, index * 10 + 5, 5, 3, 1,
              8192, 1, 1, 32, 1, 1, 64, 0, 0, 0, 0, 0)
             for index in range(1, 193)],
        )
        connection.commit()
        connection.close()
        return {
            "plan_path": plan_path, "benchmark_report": report_path,
            "trace_database": database, "power_before": before, "power_after": after,
            "terminal_selection": terminal, "artifact": artifact, "executable": executable,
            "code_object": code_object,
            "dispatch_symbol": display, "stages": [stage], "workload": workload,
            "artifact_identity": artifact_identity,
            "tool_outputs": {"llvm-objdump": objdump_output, "llvm-readelf": readelf_output},
        }

    def terminal(self, args: dict) -> tuple[dict, dict]:
        execution = {
            "xattention_profile": args["workload"]["xattention_profile"],
            "q4_activation_bits": 8,
            "w8_activation_bits": 8,
            "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
        }
        cells = {}
        for concurrency in (1, 2, 3, 4):
            rounds = concurrency * 64
            for tokens in (8192, 32768):
                cells[f"whole-pp{tokens}+tg256_c{concurrency}"] = {
                    "generated_tokens_per_lane": 256, "draft_window": 3,
                    "minimum_rounds_per_lane": 64, "concurrency": concurrency,
                    "expected_rounds_per_repetition": rounds,
                    "expected_rounds_all_repetitions": rounds * 3,
                    "observed_rounds_all_repetitions": rounds * 3,
                    "all_repetitions_minimum": True,
                    "repetitions": [{
                        "repetition": repetition, "observed_rounds": rounds,
                        "expected_minimum_rounds": rounds, "minimum_met": True,
                    } for repetition in range(3)],
                }
        terminal = {
            "winner": "selected-g16", "winner_artifact": {
                "kind": "artifact", "weights_id": "selected",
                "sha256": args["artifact_identity"]["sha256"],
            },
            "winner_cache_profile": {"value_group": 16,
                                     "plane_layouts": R9700_KV_PLANE_LAYOUTS},
            "winner_execution_profile": execution,
            "production_status": "selected_route_pending_shortlist_head_trace_and_niah",
            "shortlist_head_precision_status": "q4_round_gate_pass_pending_trace",
            "shortlist_head_precision_gate": {"cells": cells},
        }
        return terminal, {"prefill_chunk": 4096}

    def produce_result(self, args: dict) -> dict:
        passed = {key: value for key, value in args.items()
                  if key not in ("workload", "artifact_identity", "tool_outputs")}

        def runner(command, **kwargs):
            tool = "llvm-readelf" if "--notes" in command else "llvm-objdump"
            self.assertEqual(Path(command[-1]).resolve(), args["code_object"].resolve())
            return subprocess.CompletedProcess(
                command, 0, args["tool_outputs"][tool].encode(), b"",
            )

        passed["tool_runner"] = runner
        with patch(
            "tools.bench.produce_mtp_shortlist_head_evidence."
            "validate_terminal_production_authority",
            return_value=self.terminal(args),
        ):
            return produce(**passed)

    def test_binds_executed_q4_head_to_iu4_low_high_and_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), sparse=True)
            result = self.produce_result(args)
            self.assertEqual(result["tensor"]["shape"], [131072, 5120])
            self.assertEqual(result["tensor"]["format"], "Q4G64_F16S")
            self.assertEqual(result["executed_trace"]["dispatch_count"], 192)
            self.assertEqual(result["executed_trace"]["expected_dispatch_count"], 192)
            self.assertEqual(result["executed_trace"]["acceptance"]["accepted_tokens"], 180)
            self.assertEqual(result["static_proof"]["signedness_paths"], {
                "unsigned_low_signed_weight": 2, "signed_high_signed_weight": 2,
            })
            self.assertEqual(result["static_proof"]["resources"]["vgpr_count"], 64)
            self.assertEqual(result["static_proof"]["quantization"]["activations"]["codec"],
                             "signed A8G64")
            self.assertEqual(result["profile_binding"]["xattention_profile"],
                             "b128-s16-tau900")

            symbol, stages = exact_trace_signature(
                args["trace_database"], json.loads(args["plan_path"].read_text())["benchmark_command"]
            )
            self.assertEqual(symbol, args["dispatch_symbol"])
            self.assertEqual(stages, args["stages"])

    def test_rejects_load_only_or_zero_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = json.loads(args["benchmark_report"].read_text())
            for spec in (
                report["tests"][0]["speculative"],
                report["tests"][0]["reps"][0]["speculative"],
            ):
                spec["accepted_tokens"] = 0
                spec["acceptance_rate"] = 0.0
                spec["acceptance_length"] = 1.0
                spec["accepted_per_position"] = [0, 0, 0]
            args["benchmark_report"].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "no coherent executed, accepted"):
                self.produce_result(args)

    def test_rejects_profile_run_with_wrong_compiled_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = json.loads(args["benchmark_report"].read_text())
            report["config"]["w8_activation_bits"] = 16
            args["benchmark_report"].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "optimized whole-MTP3 trace row"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            connection = sqlite3.connect(args["trace_database"])
            connection.execute("delete from kernels")
            connection.execute(
                "insert into kernels values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (8, "load_kernel", "load", 1, 2, 1, 1, 1, 1, 1, 1, 32, 1, 1,
                 1, 0, 0, 0, 0, 0),
            )
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "no executed.*shortlist-head"):
                self.produce_result(args)

    def test_rejects_extra_rounds_or_conditional_terminal_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = json.loads(args["benchmark_report"].read_text())
            for spec in (
                report["tests"][0]["speculative"],
                report["tests"][0]["reps"][0]["speculative"],
            ):
                spec["rounds"] = 65
                spec["acceptance_length"] = 1.0 + 180 / 65
            args["benchmark_report"].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "theoretical-minimum lane-summed rounds"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            passed = {key: value for key, value in args.items()
                      if key not in ("workload", "artifact_identity", "tool_outputs")}
            passed["tool_runner"] = lambda command, **kwargs: subprocess.CompletedProcess(
                command, 0, args["tool_outputs"][
                    "llvm-readelf" if "--notes" in command else "llvm-objdump"
                ].encode(), b"",
            )
            terminal, selected = self.terminal(args)
            terminal["production_status"] = "conditional_head_precision_required"
            terminal["shortlist_head_precision_status"] = "conditional_head_precision_required"
            with patch(
                "tools.bench.produce_mtp_shortlist_head_evidence."
                "validate_terminal_production_authority",
                return_value=(terminal, selected),
            ), self.assertRaisesRegex(ValueError, "requires conditional head precision"):
                produce(**passed)

    def test_rejects_symbol_or_stage_ambiguity_and_wrong_iu4_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            connection = sqlite3.connect(args["trace_database"])
            connection.execute(
                "insert into kernels values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (999, DISPLAY_SYMBOL, "sparse-stage", 2000, 2010, 10,
                 3, 1, 8192, 1, 1, 32, 1, 1, 64, 0, 0, 0, 0, 0),
            )
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args["tool_outputs"]["llvm-objdump"] = args["tool_outputs"][
                "llvm-objdump"
            ].replace("neg_lo:[1,1,0]", "neg_lo:[0,1,0]")
            with self.assertRaisesRegex(ValueError, "low/high paths differ"):
                self.produce_result(args)

    def test_rejects_duplicate_matrix_point_or_any_control_parity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            plan = json.loads(args["plan_path"].read_text())
            manifest_path = Path(plan["source_matrix"]["path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["commands"].append(dict(manifest["commands"][0]))
            manifest_path.write_text(json.dumps(manifest))
            plan["source_matrix"]["sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            args["plan_path"].write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "exact MTP3/control C1..4 point set"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            plan = json.loads(args["plan_path"].read_text())
            manifest = json.loads(Path(plan["source_matrix"]["path"]).read_text())
            control = next(
                row for row in manifest["commands"]
                if row["suite"] == "pareto_whole_control" and row["concurrency"] == 1
            )
            report_path = Path(control["report"])
            report = json.loads(report_path.read_text())
            report["tests"][0]["reps"][0]["generated_token_ids_by_lane"][0][0] = 2
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "target-output parity failed"):
                self.produce_result(args)

    def test_rejects_optimized_head_or_speculation_in_ordinary_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            plan = json.loads(args["plan_path"].read_text())
            manifest = json.loads(Path(plan["source_matrix"]["path"]).read_text())
            control = next(
                row for row in manifest["commands"]
                if row["suite"] == "pareto_whole_control" and row["concurrency"] == 1
            )
            report_path = Path(control["report"])
            report = json.loads(report_path.read_text())
            report["config"]["proposal_head"] = "optimized"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "proposal_head"):
                self.produce_result(args)

    def test_rejects_repetition_or_dispatch_multiplicity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = json.loads(args["benchmark_report"].read_text())
            spec = report["tests"][0]["reps"][0]["speculative"]
            spec["accepted_tokens"] = 179
            spec["acceptance_rate"] = 179 / 192
            spec["acceptance_length"] = 1.0 + 179 / 64
            spec["accepted_per_position"] = [64, 64, 51]
            args["benchmark_report"].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "one-repetition counters differ"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            connection = sqlite3.connect(args["trace_database"])
            connection.execute("delete from kernels where dispatch_id=1")
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "dispatch multiplicity differs"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = json.loads(args["benchmark_report"].read_text())
            for spec in (
                report["tests"][0]["speculative"],
                report["tests"][0]["reps"][0]["speculative"],
            ):
                spec["drafted_tokens"] = 188
                spec["accepted_tokens"] = 176
                spec["acceptance_rate"] = 176 / 188
                spec["acceptance_length"] = 1.0 + 176 / 64
                spec["accepted_per_position"] = [64, 64, 48]
            args["benchmark_report"].write_text(json.dumps(report))
            connection = sqlite3.connect(args["trace_database"])
            connection.execute("delete from kernels where dispatch_id=192")
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError, "every proposal position drafted"):
                self.produce_result(args)

    def test_rejects_non_c1_or_non_8k_trace_scope(self) -> None:
        for field, value in (("concurrency", 2), ("prompt_tokens", 32768)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                plan = json.loads(args["plan_path"].read_text())
                plan["workload"][field] = value
                args["plan_path"].write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "exact selected C1/8K"):
                    self.produce_result(args)

    def test_rejects_terminal_selection_or_database_drift_from_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args["terminal_selection"].write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "terminal-selection identity differs"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            plan = json.loads(args["plan_path"].read_text())
            plan["expected_trace_database"] = str(Path(directory) / "other.db")
            args["plan_path"].write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "fixed trace database differs"):
                self.produce_result(args)

    def test_finalizer_removes_extracted_code_object_after_late_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            database = package / "trace.db"
            report = package / "benchmark-report.json"
            artifact = package / "model.ninfer"
            executable = package / "ninfer_bench"
            terminal = package / "terminal.json"
            before = package / "before.txt"
            after = package / "after.txt"
            for path in (database, report, artifact, executable, terminal, before, after):
                path.write_bytes(b"x")
            plan = {
                "expected_trace_database": str(database),
                "benchmark_command": [str(executable), "--output-file", str(report)],
                "benchmark_executable": {"path": str(executable)},
                "artifact": {"path": str(artifact)},
                "terminal_selection": {"path": str(terminal)},
                "required_power_profile": {
                    "before_evidence": str(before), "after_evidence": str(after),
                },
            }
            (package / "plan.json").write_text(json.dumps(plan))

            def fake_extract(_executable, output, **_kwargs):
                output.write_bytes(b"device-code")

            with patch(
                "tools.bench.finalize_selected_mtp_shortlist_head.exact_trace_signature",
                return_value=("a8q4g64_linear_wmma32_kernel", ["mtp"]),
            ), patch(
                "tools.bench.finalize_selected_mtp_shortlist_head.extract",
                side_effect=fake_extract,
            ), patch(
                "tools.bench.finalize_selected_mtp_shortlist_head.produce",
                side_effect=ValueError("late validation failure"),
            ), self.assertRaisesRegex(ValueError, "late validation failure"):
                finalize(package)
            self.assertFalse((package / "selected-mtp-shortlist-head.hsaco").exists())
            self.assertFalse((package / "shortlist-head-evidence.json").exists())

    def test_rejects_late_failures_file_and_resolves_input_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            plan = json.loads(args["plan_path"].read_text())
            failures = Path(plan["source_matrix"]["path"]).parent / "failures.json"
            from tools.bench.produce_mtp_shortlist_head_evidence import _validate_iu4

            def publish_failure(*values, **keywords):
                result = _validate_iu4(*values, **keywords)
                failures.write_text("[]")
                return result

            with patch(
                "tools.bench.produce_mtp_shortlist_head_evidence._validate_iu4",
                side_effect=publish_failure,
            ), self.assertRaisesRegex(ValueError, "failures appeared"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            plan = json.loads(args["plan_path"].read_text())
            failures = Path(plan["source_matrix"]["path"]).parent / "failures.json"
            failures.symlink_to(failures.parent / "missing-failure-target")
            with self.assertRaisesRegex(ValueError, "retains failures"):
                self.produce_result(args)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            link = root / "artifact-link.ninfer"
            link.symlink_to(args["artifact"])
            args["artifact"] = link
            result = self.produce_result(args)
            self.assertEqual(result["selected_route"]["artifact"]["path"],
                             str((root / "model.ninfer").resolve()))

    def test_rejects_identical_byte_code_object_replacement_during_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            from tools.bench.produce_mtp_shortlist_head_evidence import _validate_iu4

            def replace_after_derivation(*values, **keywords):
                result = _validate_iu4(*values, **keywords)
                replacement = root / "replacement.hsaco"
                replacement.write_bytes(args["code_object"].read_bytes())
                os.replace(replacement, args["code_object"])
                return result

            with patch(
                "tools.bench.produce_mtp_shortlist_head_evidence._validate_iu4",
                side_effect=replace_after_derivation,
            ), self.assertRaisesRegex(ValueError, "code_object changed"):
                self.produce_result(args)

    def test_owner_cli_rejects_dangling_output_symlink_without_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            out = root / "evidence.json"
            out.symlink_to(root / "missing-target")
            argv = []
            for key in (
                "plan_path", "benchmark_report", "trace_database", "power_before",
                "power_after", "terminal_selection", "artifact", "executable",
                "code_object",
            ):
                argv.extend(["--" + key.removesuffix("_path").replace("_", "-"), str(args[key])])
            argv.extend(["--dispatch-symbol", args["dispatch_symbol"],
                         "--stage", args["stages"][0], "--out", str(out)])
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                main(argv)

    def test_evidence_publication_removes_only_owned_inode_if_fsync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            with patch(
                "tools.bench.produce_mtp_shortlist_head_evidence.os.fsync",
                side_effect=[None, OSError("directory fsync failed")],
            ), self.assertRaisesRegex(OSError, "directory fsync failed"):
                _publish(output, {"status": "passed"})
            self.assertFalse(os.path.lexists(output))


if __name__ == "__main__":
    unittest.main()
