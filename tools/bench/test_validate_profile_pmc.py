#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.bench.prepare_whole_profile import DISPATCH_COUNTERS
from tools.bench.run_ninfer_bench_matrix import REPORT_SCHEMA_VERSION
from tools.bench.validate_profile_pmc import main, validate


class ValidateProfilePmcTest(unittest.TestCase):
    def identity(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def fixture(self, root: Path, *, markers: bool = True) -> dict[str, Path]:
        artifact = root / "selected.ninfer"
        executable = root / "ninfer_bench"
        corpus = root / "corpus.ids"
        terminal = root / "selection.json"
        source_manifest = root / "manifest.json"
        source_report = root / "source-p2048.json"
        evaluation = root / "evaluation.json"
        for path, data in (
            (artifact, b"artifact"), (executable, b"executable"), (corpus, b"1\n2\n"),
            (terminal, b'{"selection":true}\n'), (source_manifest, b'{"matrix":true}\n'),
            (source_report, b'{"source":true}\n'), (evaluation, b'{"evaluation":true}\n'),
        ):
            path.write_bytes(data)
        report = root / "benchmark-report.json"
        command = [
            str(executable.resolve()), "--weights", str(artifact.resolve()),
            "--corpus", str(corpus.resolve()), "--device", "0", "--concurrency", "1",
            "-p", "2048", "--prefill-chunk", "4096", "--spec", "mtp",
            "--draft-tokens", "0", "--output", "json", "--output-file",
            str(report.resolve()), "-r", "1", "--warmup", "1", "--profile-measured",
        ]
        source_command = list(command[:-1])
        source_command[source_command.index("--output-file") + 1] = str(source_report.resolve())
        source_command[source_command.index("-r") + 1] = "3"
        source_manifest.write_text(json.dumps({
            "commands": [{
                "suite": "low_context_prefill", "case": "prefill_p2048_dense_none",
                "concurrency": 1, "report": str(source_report.resolve()),
                "command": source_command,
            }],
        }), encoding="utf-8")
        artifact_identity = {**self.identity(artifact), "weights_id": "selected"}
        terminal_winner = {
            "winner": "selected-g16",
            "winner_artifact": {
                "weights_id": "selected", "sha256": artifact_identity["sha256"],
            },
            "winner_cache_profile": {"value_group": 16},
            "winner_execution_profile": {"xattention_profile": "dense"},
        }
        terminal.write_text(json.dumps({
            "schema_version": 7, "selected_prefill_chunk": 4096,
            "terminal_production_selection": terminal_winner,
        }), encoding="utf-8")
        report.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": REPORT_SCHEMA_VERSION,
            "tool": "ninfer_bench", "command": " ".join(command),
            "environment": {"gpu_name": "AMD Radeon AI PRO R9700",
                            "architecture_name": "gfx1201"},
            "artifact": {"path": str(artifact.resolve()),
                         "file_size_bytes": artifact.stat().st_size},
            "load": {"weights_id": "selected"},
            "config": {"concurrency": 1, "kv_value_group": 16,
                       "prefill_chunk": 4096, "spec": "none", "draft_tokens": 0,
                       "kv_plane_layouts": {
                           "key": "token-fastest-head-major",
                           "value": "feature-fastest-page-major",
                           "value_scale": "feature-fastest-page-major",
                       },
                       "dflash_verify_width_requested": 0, "dflash_verify_width": 0,
                       "xattention_qualification": False, "repetitions": 1, "warmup": 1},
            "tests": [{"kind": "pp", "n_prompt": 2048, "n_gen": 0}],
        }), encoding="utf-8")
        plan = root / "plan.json"
        terminal_identity = self.identity(terminal)
        terminal_provenance = {
            "path": terminal_identity["path"], "bytes": terminal_identity["file_size_bytes"],
            "sha256": terminal_identity["sha256"], **terminal_winner,
            "selected_prefill_chunk": 4096,
        }
        source_matrix_identity = {
            "path": str(source_manifest.resolve()), "sha256": self.identity(source_manifest)["sha256"],
            "preset": "low-context-prefill",
            "report": {"path": str(source_report.resolve()),
                       "sha256": self.identity(source_report)["sha256"]},
        }
        evaluation.write_text(json.dumps({
            "artifact_type": "ninfer_r9700_low_context_prefill_evaluation",
            "schema_version": 1, "manifest": {
                "path": source_matrix_identity["path"],
                "sha256": source_matrix_identity["sha256"],
            },
            "artifact": artifact_identity, "bench": self.identity(executable),
            "terminal_selection": terminal_provenance, "expected_kv_value_group": 16,
            "selected_prefill_chunk": 4096, "minimum_p2048_tok_s": 2000.0,
            "observed_p2048_tok_s": 1900.0, "passes_p2048_gate": False,
        }), encoding="utf-8")
        plan_value = {
            "artifact_type": "ninfer_whole_profile_plan", "schema_version": 2,
            "profile_kind": "dispatch-pmc", "counters": list(DISPATCH_COUNTERS),
            "required_power_profile": {
                "value": "profile_standard", "sysfs_path": "/sys/power",
                "before_evidence": str((root / "power-before.txt").resolve()),
                "after_evidence": str((root / "power-after.txt").resolve()),
            },
            "artifact": artifact_identity,
            "benchmark_executable": self.identity(executable),
            "corpus": {**self.identity(corpus), "tokens": 2},
            "terminal_selection": terminal_provenance,
            "source_matrix": source_matrix_identity,
            "low_context_evaluation": {
                "path": str(evaluation.resolve()), "sha256": self.identity(evaluation)["sha256"],
                "minimum_p2048_tok_s": 2000.0, "observed_p2048_tok_s": 1900.0,
                "passes_p2048_gate": False,
            },
            "workload": {
                "concurrency": 1, "prompt_tokens": 2048, "generated_tokens": 0,
                "spec": "none", "draft_tokens": 0, "dflash_verify_width": 0,
                "kv_value_group": 16, "xattention_profile": "dense", "prefill_chunk": 4096,
                "kv_plane_layouts": {
                    "key": "token-fastest-head-major",
                    "value": "feature-fastest-page-major",
                    "value_scale": "feature-fastest-page-major",
                },
            },
            "benchmark_command": command,
            "kernel_include_regex": "kernel_[ab]",
            "profiler_command": ["/opt/rocm/bin/rocprofv3", "--selected-regions",
                                 "-f", "rocpd", "-d",
                                 str((root / "rocprof-dispatch-pmc").resolve()),
                                 *( ["--marker-trace", "--kernel-trace"] if markers else []),
                                 "--kernel-include-regex", "kernel_[ab]",
                                 "--pmc", *DISPATCH_COUNTERS, "--", *command],
        }
        plan.write_text(json.dumps(plan_value), encoding="utf-8")

        counter_csv = root / "counter_collection.csv"
        rows = []
        base = {
            "GL2C_HIT": "8", "GL2C_MISS": "2", "L2CacheHit": "80",
            "TCP_REQ": "10", "TCP_REQ_MISS": "3", "GL2C_EA_RDREQ": "4",
            "GL2C_EA_WRREQ": "0", "SQ_WAVES": "1",
        }
        for dispatch_id, kernel, start, end in ((1, "kernel_a", 100, 200), (2, "kernel_b", 210, 300)):
            for counter in DISPATCH_COUNTERS:
                rows.append({
                    "Dispatch_Id": dispatch_id, "Kernel_Name": kernel,
                    "Counter_Name": counter, "Counter_Value": base[counter],
                    "Start_Timestamp": start, "End_Timestamp": end,
                })
        with counter_csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        database = root / "capture.db"
        connection = sqlite3.connect(database)
        connection.executescript("""
            create table rocpd_info_process_x(id integer, command text);
            create table rocpd_info_agent_x(id integer, type text, name text, product_name text);
            create table rocpd_info_pmc_x(id integer, symbol text);
            create table rocpd_pmc_event_x(id integer, event_id integer, pmc_id integer, value real);
            create table rocpd_kernel_dispatch_x(event_id integer, dispatch_id integer,
                start integer, "end" integer, kernel_id integer, region_name_id integer);
            create table rocpd_info_kernel_symbol_x(id integer, display_name text);
            create table rocpd_string_x(id integer, string text);
        """)
        connection.execute("insert into rocpd_info_process_x values (?,?)", (1, " ".join(command)))
        connection.execute("insert into rocpd_info_agent_x values (1,?,?,?)",
                           ("GPU", "gfx1201", "AMD Radeon AI PRO R9700"))
        connection.executemany("insert into rocpd_info_pmc_x values (?,?)",
                               list(enumerate(DISPATCH_COUNTERS, start=1)))
        connection.executemany("insert into rocpd_info_kernel_symbol_x values (?,?)",
                               ((1, "kernel_a"), (2, "kernel_b")))
        connection.executemany("insert into rocpd_string_x values (?,?)",
                               ((1, "ninfer.attention.prefill.attention"),
                                (2, "ninfer.gdn.prefill.gdn")))
        event_id = 10
        pmc_rows = []
        for dispatch_id, kernel_id, start, end in ((1, 1, 100, 200), (2, 2, 210, 300)):
            connection.execute("insert into rocpd_kernel_dispatch_x values (?,?,?,?,?,?)",
                               (event_id, dispatch_id, start, end, kernel_id,
                                dispatch_id if markers else None))
            for pmc_id, counter in enumerate(DISPATCH_COUNTERS, start=1):
                pmc_rows.append((len(pmc_rows) + 1, event_id, pmc_id, float(base[counter])))
            event_id += 1
        connection.executemany("insert into rocpd_pmc_event_x values (?,?,?,?)", pmc_rows)
        connection.commit()
        connection.close()
        power_before = root / "power-before.txt"
        power_after = root / "power-after.txt"
        power_before.write_text("profile_standard\n", encoding="utf-8")
        power_after.write_text("auto\n", encoding="utf-8")
        return {
            "plan": plan, "report": report, "csv": counter_csv, "database": database,
            "before": power_before, "after": power_after, "terminal": terminal,
            "artifact": artifact, "executable": executable, "corpus": corpus,
        }

    def run_validate(self, paths: dict[str, Path]) -> dict:
        return validate(
            paths["plan"], paths["report"], paths["csv"], paths["database"],
            paths["before"], paths["after"], paths["terminal"], paths["artifact"],
            paths["executable"], paths["corpus"],
        )

    def test_validates_ratios_zero_semantics_and_same_capture_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_validate(self.fixture(Path(directory)))
            self.assertEqual(result["metrics"]["gl2_hit_ratio"]["value"], 0.8)
            self.assertEqual(result["metrics"]["tcp_hit_ratio"]["value"], 0.7)
            self.assertEqual(result["counters"]["GL2C_EA_WRREQ"]["state"], "observed_zero")
            self.assertEqual(result["roctx_stage_join"]["state"], "exact_same_capture")
            self.assertEqual(
                {row["symbol"] for row in result["roctx_stage_join"]["dispatches"]},
                {"kernel_a", "kernel_b"})
            self.assertFalse(result["profile_timing_admissible"])

    def test_rejects_plan_without_same_capture_marker_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exact same-capture PMC contract"):
                self.run_validate(self.fixture(Path(directory), markers=False))

    def test_rejects_missing_counter_and_partial_stage_join(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            with paths["csv"].open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows = [row for row in rows if not (
                row["Dispatch_Id"] == "2" and row["Counter_Name"] == "SQ_WAVES"
            )]
            with paths["csv"].open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "incomplete counter set"):
                self.run_validate(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            connection = sqlite3.connect(paths["database"])
            connection.execute(
                "update rocpd_kernel_dispatch_x set region_name_id=null where dispatch_id=2")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "partial ROCTX"):
                self.run_validate(paths)

    def test_rejects_forged_profiler_prefix_or_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
            plan["profiler_command"][0] = "/tmp/rocprofv3"
            plan["profiler_command"].remove("rocpd")
            plan["profiler_command"].insert(3, "csv")
            paths["plan"].write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact same-capture PMC contract"):
                self.run_validate(paths)

    def test_rejects_self_consistent_command_not_derived_from_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
            warmup = plan["benchmark_command"].index("--warmup") + 1
            plan["benchmark_command"][warmup] = "0"
            separator = plan["profiler_command"].index("--") + 1
            plan["profiler_command"][separator:] = plan["benchmark_command"]
            paths["plan"].write_text(json.dumps(plan), encoding="utf-8")
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["command"] = " ".join(plan["benchmark_command"])
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            connection = sqlite3.connect(paths["database"])
            connection.execute(
                "update rocpd_info_process_x set command=?",
                (" ".join(plan["benchmark_command"]),),
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "exact source P2048 derivation"):
                self.run_validate(paths)

    def test_cli_refuses_to_clobber_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            output = Path(directory) / "evidence.json"
            output.write_text("retained", encoding="utf-8")
            arguments = [
                "--plan", str(paths["plan"]), "--benchmark-report", str(paths["report"]),
                "--counter-csv", str(paths["csv"]), "--database", str(paths["database"]),
                "--power-before", str(paths["before"]), "--power-after", str(paths["after"]),
                "--terminal-selection", str(paths["terminal"]), "--artifact", str(paths["artifact"]),
                "--executable", str(paths["executable"]), "--corpus", str(paths["corpus"]),
                "--out", str(output),
            ]
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                main(arguments)
            self.assertEqual(output.read_text(encoding="utf-8"), "retained")


if __name__ == "__main__":
    unittest.main()
