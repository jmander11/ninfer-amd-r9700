#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench.analyze_selected_decode_memory import (
    expected_round_regions, read_marker_counts, summarize, validate_benchmark_report,
)
from tools.bench.prepare_selected_decode_memory_profile import (
    COUNTERS, PROFILE_BUILD_PRODUCER, PROFILE_BUILD_RECEIPT_TYPE,
    PROFILE_ROUTE_RESOLVER, profile_build_commands, sha, validate_profile_build_receipt,
)
from tools.bench.run_ninfer_bench_matrix import inspect_executable
from tools.bench.run_ninfer_bench_matrix import REPORT_SCHEMA_VERSION


class SelectedDecodeMemoryAnalysisTest(unittest.TestCase):
    def test_prepared_module_entry_points_import_from_repo(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        prepare_script = (
            repo / "profiles/rocprof/selected-ordinary-decode-memory-prepare-20260905/prepare.sh"
        ).read_text(encoding="utf-8")
        build_script = (
            repo / "profiles/rocprof/selected-ordinary-decode-profile-build-20260905/build.sh"
        ).read_text(encoding="utf-8")
        producer = (repo / "tools/bench/prepare_selected_decode_memory_profile.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("python3 -m tools.bench.prepare_selected_decode_memory_profile", prepare_script)
        self.assertIn("python3 -m tools.bench.build_selected_decode_profile", build_script)
        self.assertIn("python3 -m tools.bench.analyze_selected_decode_memory", producer)
        for module in ("tools.bench.prepare_selected_decode_memory_profile",
                       "tools.bench.analyze_selected_decode_memory",
                       "tools.bench.build_selected_decode_profile"):
            result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=repo,
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, (module, result.stderr))

    def test_profile_build_receipt_binds_distinct_selected_compile_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            manifest = root / "matrix.json"
            marker = root / "program_impl.h"
            terminal = root / "timing-bench"
            build = root / "profile-build"
            binary = build / "bench/ninfer_bench"
            cache = build / "CMakeCache.txt"
            compile_database = build / "compile_commands.json"
            for path, value in ((selection, "selection"), (manifest, "matrix"),
                                (marker, "marker"), (terminal, "timing")):
                path.write_text(value, encoding="utf-8")
            binary.parent.mkdir(parents=True)
            binary.write_text("profile", encoding="utf-8")
            definitions = {
                "CMAKE_BUILD_TYPE": "Release", "CMAKE_HIP_ARCHITECTURES": "gfx1201",
                "NINFER_BUILD_BENCHMARKS": "ON", "NINFER_R9700_KV_VALUE_GROUP": "16",
                "NINFER_R9700_Q4_ACTIVATION_BITS": "8",
                "NINFER_R9700_Q4_PREFILL_PINGPONG_QUALIFICATION": "OFF",
                "NINFER_R9700_W8_ACTIVATION_BITS": "8",
                "NINFER_R9700_FP8_QK_WMMA": "1",
                "NINFER_R9700_XATTENTION_QUALIFICATION": "OFF",
                "NINFER_R9700_XATTENTION_STRIDE": "16",
                "NINFER_R9700_XATTENTION_TAU_PERMILLE": "900",
            }
            cache.write_text("\n".join(
                [*(f"{key}:STRING={value}" for key, value in definitions.items()),
                 "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(Path(__file__).resolve().parents[2]),
                 "CMAKE_PROJECT_NAME:STATIC=ninfer", "CMAKE_GENERATOR:INTERNAL=Ninja",
                 "CMAKE_HIP_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang++"]),
                encoding="utf-8")
            compile_database.write_text("[]", encoding="utf-8")
            artifact = {"path": "/artifact", "sha256": "a" * 64}
            timing = inspect_executable(terminal)
            marker_id = {"path": str(marker), "file_size_bytes": marker.stat().st_size,
                         "sha256": sha(marker)}
            selected = {"winner": "route", "artifact": artifact,
                        "cache_profile": {"value_group": 16},
                        "execution_profile": {"xattention_profile": "dense"},
                        "prefill_chunk": 2048}
            compiled = {"q4_activation_bits": 8, "w8_activation_bits": 8,
                        "fp8_qk_wmma_enabled": True}
            receipt = {
                "artifact_type": PROFILE_BUILD_RECEIPT_TYPE, "schema_version": 1,
                "status": "passed", "purpose": "profiler_attribution_only",
                "profile_timing_admissible": False,
                "producer": {"path": str(PROFILE_BUILD_PRODUCER),
                             "file_size_bytes": PROFILE_BUILD_PRODUCER.stat().st_size,
                             "sha256": sha(PROFILE_BUILD_PRODUCER)},
                "route_resolver": {"path": str(PROFILE_ROUTE_RESOLVER),
                                   "file_size_bytes": PROFILE_ROUTE_RESOLVER.stat().st_size,
                                   "sha256": sha(PROFILE_ROUTE_RESOLVER)},
                "terminal_selection": {"path": str(selection), "sha256": sha(selection)},
                "source_matrix": {"path": str(manifest), "sha256": sha(manifest)},
                "artifact": artifact, "terminal_timing_executable": timing,
                "ordinary_round_marker_source": marker_id,
                "selected_route": selected, "compiled_route": compiled,
                "hybrid_shared_workspace_authority": None,
                "instrumentation_executable": inspect_executable(binary),
                "build_directory": str(build),
                "cmake_cache": {"path": str(cache), "file_size_bytes": cache.stat().st_size,
                                "sha256": sha(cache)},
                "compile_database": {"path": str(compile_database),
                                     "file_size_bytes": compile_database.stat().st_size,
                                     "sha256": sha(compile_database)},
                "compile_definitions": definitions,
            }
            receipt_path = root / "receipt.json"
            with patch(
                "tools.bench.prepare_selected_decode_memory_profile.PROFILE_BUILD_DIR", build
            ):
                receipt["build_commands"] = profile_build_commands(definitions)
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                validated, resolved, validated_cache = validate_profile_build_receipt(
                    receipt_path, selection_path=selection, selection_sha256=sha(selection),
                    manifest_path=manifest, manifest_sha256=sha(manifest), artifact=artifact,
                    terminal_executable=timing, marker_source=marker_id,
                    selected_route=selected, compiled_route=compiled,
                    hybrid_workspace_authority=None,
                )
            self.assertEqual(validated, receipt)
            self.assertEqual(resolved, binary.resolve())
            self.assertEqual(validated_cache, cache)
            receipt["build_commands"]["build"][-1] = "-j8"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with (
                patch("tools.bench.prepare_selected_decode_memory_profile.PROFILE_BUILD_DIR", build),
                self.assertRaisesRegex(ValueError, "canonical build commands"),
            ):
                validate_profile_build_receipt(
                    receipt_path, selection_path=selection,
                    selection_sha256=sha(selection), manifest_path=manifest,
                    manifest_sha256=sha(manifest), artifact=artifact,
                    terminal_executable=timing, marker_source=marker_id,
                    selected_route=selected, compiled_route=compiled,
                    hybrid_workspace_authority=None,
                )
            with patch(
                "tools.bench.prepare_selected_decode_memory_profile.PROFILE_BUILD_DIR", build
            ):
                receipt["build_commands"] = profile_build_commands(definitions)
            binary.unlink()
            binary.hardlink_to(terminal)
            receipt["instrumentation_executable"] = inspect_executable(binary)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with (
                patch("tools.bench.prepare_selected_decode_memory_profile.PROFILE_BUILD_DIR", build),
                self.assertRaisesRegex(ValueError, "aliases timing authority"),
            ):
                validate_profile_build_receipt(
                    receipt_path, selection_path=selection,
                    selection_sha256=sha(selection), manifest_path=manifest,
                    manifest_sha256=sha(manifest), artifact=artifact,
                    terminal_executable=timing, marker_source=marker_id,
                    selected_route=selected, compiled_route=compiled,
                    hybrid_workspace_authority=None,
                )
            binary.unlink()
            binary.write_text("profile", encoding="utf-8")
            receipt["instrumentation_executable"] = inspect_executable(binary)
            receipt["compile_definitions"]["NINFER_R9700_KV_VALUE_GROUP"] = "32"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with (
                patch("tools.bench.prepare_selected_decode_memory_profile.PROFILE_BUILD_DIR", build),
                self.assertRaisesRegex(ValueError, "selected profiler route"),
            ):
                validate_profile_build_receipt(
                    receipt_path, selection_path=selection,
                    selection_sha256=sha(selection), manifest_path=manifest,
                    manifest_sha256=sha(manifest), artifact=artifact,
                    terminal_executable=timing, marker_source=marker_id,
                    selected_route=selected, compiled_route=compiled,
                    hybrid_workspace_authority=None,
                )

    def test_reads_semantic_marker_message_from_event_extdata(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            create table strings(id integer primary key, string text);
            create table events(id integer primary key, category_id integer, extdata text);
            create table regions(id integer primary key, name_id integer, event_id integer);
            insert into strings values(1, 'roctxThreadRangeA');
            insert into strings values(2, 'MARKER_CORE_RANGE_API');
            insert into strings values(3, 'roctxProfilerResume');
            insert into strings values(4, 'MARKER_CONTROL_API');
            insert into events values(1, 2, '{"message":"ninfer.gdn.verify.gdn payload=0"}');
            insert into events values(2, 4, '{}');
            insert into regions values(1, 1, 1);
            insert into regions values(2, 3, 2);
        """)
        self.assertEqual(read_marker_counts(
            connection, "regions", "events", "strings"),
            {"ninfer.gdn.verify.gdn payload=0": 1},
        )
    @staticmethod
    def database_result(regions: dict[int, str]) -> tuple[dict, dict, dict]:
        return ({dispatch: {"region": region, "symbol": "kernel", "grid_size": 64,
                            "workgroup_size": 64, "static_lds_bytes": 0,
                            "scratch_bytes": 0, "vgpr": 32}
                 for dispatch, region in regions.items()},
                {"cu_count": 64, "max_waves_per_cu": 32}, expected_round_regions())

    def write_csv(self, path: Path, *, zero: str | None = None, omit: str | None = None) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output, fieldnames=("Dispatch_Id", "Kernel_Name", "Grid_Size",
                                    "Workgroup_Size", "LDS_Block_Size", "Scratch_Size",
                                    "VGPR_Count", "Counter_Name", "Counter_Value")
            )
            writer.writeheader()
            for counter in COUNTERS:
                if counter == omit:
                    continue
                value = 0 if counter == zero else 10
                if counter in ("SQ_WAVE_CYCLES", "GRBM_GUI_ACTIVE"):
                    value = 100
                if counter == "TCP_REQ_MISS":
                    value = 2
                writer.writerow({"Dispatch_Id": 7, "Kernel_Name": "kernel", "Grid_Size": 64,
                                 "Workgroup_Size": 64, "LDS_Block_Size": 0,
                                 "Scratch_Size": 0, "VGPR_Count": 32, "Counter_Name": counter,
                                 "Counter_Value": value})

    def test_reports_only_defensible_proxy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "counters.csv"
            self.write_csv(csv_path)
            with patch(
                "tools.bench.analyze_selected_decode_memory.read_database",
                return_value=self.database_result(
                    {7: "ninfer.decode.decode.ordinary_round payload=8192"}),
            ):
                result = summarize(csv_path, root / "results.db", ["bench"])
            self.assertEqual(result["ordinary_decode_model_layer_dispatch_count"], 1)
            self.assertIn("relative_gl2_read_requests", result["proxy_metrics"])
            self.assertNotIn("physical_memory_bandwidth_bytes_per_second", result)

    def test_fails_closed_on_zero_or_missing_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "tools.bench.analyze_selected_decode_memory.read_database",
                return_value=self.database_result(
                    {7: "ninfer.decode.decode.ordinary_round payload=8192"}),
            ):
                zero = root / "zero.csv"
                self.write_csv(zero, zero="GL2C_EA_RDREQ")
                with self.assertRaisesRegex(ValueError, "nonpositive required metric"):
                    summarize(zero, root / "results.db", ["bench"])
                missing = root / "missing.csv"
                self.write_csv(missing, omit="SQ_WAVES")
                with self.assertRaisesRegex(ValueError, "exact counter inventory"):
                    summarize(missing, root / "results.db", ["bench"])

    def test_rejects_incomplete_measured_dispatch_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counters = root / "counters.csv"
            self.write_csv(counters)
            with (
                patch(
                    "tools.bench.analyze_selected_decode_memory.read_database",
                    return_value=self.database_result({
                        7: "ninfer.decode.decode.ordinary_round payload=8192",
                        8: "ninfer.attention.prefill.attention payload=3"}),
                ),
                self.assertRaisesRegex(ValueError, "dispatch inventories differ"),
            ):
                summarize(counters, root / "results.db", ["bench"])

    def test_rejects_incomplete_marker_or_resource_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counters = root / "counters.csv"
            self.write_csv(counters)
            database = self.database_result(
                {7: "ninfer.decode.decode.ordinary_round payload=8192"})
            incomplete = dict(database[2])
            incomplete["ninfer.decode.decode.ordinary_round payload=8192"] -= 1
            with (
                patch("tools.bench.analyze_selected_decode_memory.read_database",
                      return_value=(database[0], database[1], incomplete)),
                self.assertRaisesRegex(ValueError, "ordinary frontiers"),
            ):
                summarize(counters, root / "results.db", ["bench"])
            changed = {key: dict(value) for key, value in database[0].items()}
            changed[7]["vgpr"] = 31
            with (
                patch("tools.bench.analyze_selected_decode_memory.read_database",
                      return_value=(changed, database[1], database[2])),
                self.assertRaisesRegex(ValueError, "dispatch metadata differ"),
            ):
                summarize(counters, root / "results.db", ["bench"])

    def test_benchmark_report_is_exact_ordinary_selected_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "bench.json"
            command = ["/bench", "--profile-measured"]
            value = {
                "artifact_type": "ninfer_bench_report",
                "schema_version": REPORT_SCHEMA_VERSION,
                "command": " ".join(command),
                "environment": {"gpu_name": "AMD Radeon AI PRO R9700",
                                "architecture_name": "gfx1201"},
                "artifact": {"path": "/weights", "file_size_bytes": 123},
                "load": {"weights_id": "weights", "target": "qwen3_8_27b_r9700"},
                "config": {"concurrency": 2, "spec": "none", "draft_tokens": 0,
                           "speculative_execution": False, "proposal_head": "full",
                           "retain_token_ids": True, "use_device_graph": True,
                           "decode_path": "device_graph",
                           "prefill_chunk": 2048, "kv_value_group": 16,
                           "corpus_path": "/corpus", "corpus_tokens": 65536,
                           "q4_activation_bits": 8, "w8_activation_bits": 8,
                           "fp8_qk_wmma_enabled": True,
                           "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                           "fp8_qk_wmma_t1_min_context": 64,
                           "fp8_qk_wmma_t2_min_context": 320,
                           "q4_prefill_cta_profile":
                               "m64n128-pingpong-n16-k16-scalar-base-production",
                           "xattention_qualification": False,
                           "repetitions": 1, "warmup": 1},
                "tests": [{"kind": "whole", "n_prompt": 8192, "n_gen": 256,
                           "requested_output_tokens": 257,
                           "decode_output_tok_s_mean": 20.0,
                           "speculative": {"enabled": False},
                           "reps": [{"generated_output_tokens": 257}]}],
            }
            report.write_text(json.dumps(value), encoding="utf-8")
            plan = {"selected_route": {"prefill_chunk": 2048,
                                       "cache_profile": {"value_group": 16},
                                       "execution_profile": {"xattention_profile": "dense"}},
                    "artifact": {"path": "/weights", "sha256": "a" * 64,
                                 "file_size_bytes": 123,
                                 "weights_id": "weights"},
                    "corpus": {"path": "/corpus", "tokens": 65536},
                    "compiled_route": {
                        "q4_activation_bits": 8, "w8_activation_bits": 8,
                        "fp8_qk_wmma_enabled": True,
                        "fp8_qk_wmma_profile": "t1-ge64-t2-ge320-t3plus-stream-v1",
                        "fp8_qk_wmma_t1_min_context": 64,
                        "fp8_qk_wmma_t2_min_context": 320,
                        "q4_prefill_cta_profile":
                            "m64n128-pingpong-n16-k16-scalar-base-production"}}
            result = validate_benchmark_report(
                report, {"benchmark_command": command}, plan, 2
            )
            self.assertEqual(result["decode_output_tok_s"], 20.0)
            value["config"]["spec"] = "mtp"
            report.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordinary route"):
                validate_benchmark_report(report, {"benchmark_command": command}, plan, 2)

    def test_benchmark_report_rejects_stale_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "bench.json"
            command = ["/bench", "--profile-measured"]
            value = {
                "artifact_type": "ninfer_bench_report", "schema_version": 6,
                "command": " ".join(command),
                "environment": {"gpu_name": "AMD Radeon AI PRO R9700",
                                "architecture_name": "gfx1201"},
                "artifact": {"path": "/weights", "file_size_bytes": 123},
                "load": {"weights_id": "weights"},
                "config": {"concurrency": 1, "spec": "none", "draft_tokens": 0,
                           "speculative_execution": False, "proposal_head": "full",
                           "retain_token_ids": True, "use_device_graph": True,
                           "decode_path": "device_graph",
                           "prefill_chunk": 2048, "kv_value_group": 16,
                           "xattention_qualification": False,
                           "repetitions": 1, "warmup": 1},
                "tests": [{"kind": "whole", "n_prompt": 8192, "n_gen": 256,
                           "requested_output_tokens": 257,
                           "decode_output_tok_s_mean": 20.0,
                           "speculative": {"enabled": False},
                           "reps": [{"generated_output_tokens": 257}]}],
            }
            report.write_text(json.dumps(value), encoding="utf-8")
            plan = {"selected_route": {"prefill_chunk": 2048,
                                       "cache_profile": {"value_group": 16},
                                       "execution_profile": {"xattention_profile": "dense"}},
                    "artifact": {"path": "/weights", "file_size_bytes": 123,
                                 "weights_id": "weights"}}
            with self.assertRaisesRegex(ValueError, "ordinary route"):
                validate_benchmark_report(report, {"benchmark_command": command}, plan, 1)


if __name__ == "__main__":
    unittest.main()
