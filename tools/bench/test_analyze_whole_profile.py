#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.bench.analyze_whole_profile import analyze


class AnalyzeWholeProfileTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        database = root / "trace.db"
        connection = sqlite3.connect(database)
        connection.executescript("""
            create table regions(start integer, "end" integer, extdata text);
            create table kernels(start integer, "end" integer, duration integer,
                                 name text, region text);
            create table memory_copies(start integer, "end" integer, duration integer,
                                       name text, region_name text, size integer);
        """)
        regions = [
            (0, 1_000_000_000, {"message": "ninfer_bench_measured"}),
            (100_000_000, 900_000_000,
             {"message": "ninfer.prefill.prefill.chunk payload=8"}),
            (700_000_000, 800_000_000,
             {"message": "ninfer.mtp.prefill.mtp_chunk payload=8"}),
        ]
        connection.executemany("insert into regions values (?,?,?)", [
            (begin, end, json.dumps(extdata)) for begin, end, extdata in regions])
        connection.executemany("insert into kernels values (?,?,?,?,?)", [
            (110_000_000, 300_000_000, 190_000_000,
             "a8q4g64_linear_wmma32_kernel", "ninfer.attention.prefill.attention"),
            (250_000_000, 500_000_000, 250_000_000,
             "xattention_flash_consumer_kernel", "ninfer.attention.prefill.attention"),
            (710_000_000, 760_000_000, 50_000_000,
             "gdn_recurrence_kernel", "ninfer.mtp.prefill.mtp_chunk payload=8"),
            (820_000_000, 880_000_000, 60_000_000,
             "post_kernel", "ninfer.post-mixer.prefill.post_mixer"),
            (10_000_000, 20_000_000, 10_000_000, "sampling_kernel", ""),
        ])
        connection.executemany("insert into memory_copies values (?,?,?,?,?,?)", [
            (600_000_000, 650_000_000, 50_000_000, "H2D",
             "ninfer.attention.prefill.attention", 100),
            (720_000_000, 730_000_000, 10_000_000, "D2D",
             "ninfer.mtp.prefill.mtp_chunk payload=8", 20),
        ])
        connection.commit()
        connection.close()
        report = root / "benchmark-report.json"
        report.write_text(json.dumps({
            "artifact_type": "ninfer_bench_report", "schema_version": 20,
            "config": {"prefill_chunk": 8},
            "tests": [{"kind": "pp", "n_prompt": 8, "n_gen": 0,
                       "prefill_seconds_mean": 0.8}],
        }), encoding="utf-8")
        return database, report

    def test_exact_prefill_categories_union_gap_copies_and_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, report = self.fixture(Path(directory))
            result = analyze(database, report)
            categories = {row["category"]: row for row in result["kernel_execution_categories"]}
            self.assertEqual(categories["base_text_prefill"]["calls"], 3)
            self.assertEqual(categories["base_text_prefill"]["independent_summed_duration_ms"], 500)
            self.assertEqual(categories["base_text_prefill"]["active_union_ms"], 450)
            self.assertEqual(categories["mtp_prefill"]["independent_summed_duration_ms"], 50)
            self.assertEqual(result["prefill_no_kernel_wall_ms"], 300)
            copies = {row["category"]: row for row in result["memory_copy_categories"]}
            self.assertEqual(copies["base_text_prefill"]["bytes"], 100)
            self.assertEqual(copies["mtp_prefill"]["bytes"], 20)
            copy_kinds = {(row["execution_category"], row["name"]): row
                          for row in result["memory_copy_kinds"]}
            self.assertEqual(copy_kinds[("base_text_prefill", "H2D")]["bytes"], 100)
            self.assertEqual(copy_kinds[("mtp_prefill", "D2D")]["calls"], 1)
            families = {(row["execution_category"], row["family"]): row
                        for row in result["symbol_families"]}
            self.assertEqual(families[("base_text_prefill", "a8q4_matrix")]["calls"], 1)
            self.assertEqual(families[("base_text_prefill", "xattention_consumer")]["calls"], 1)
            self.assertEqual(families[("mtp_prefill", "gdn")]["calls"], 1)
            markers = {(row["execution_category"], row["family"]): row
                       for row in result["marker_families"]}
            self.assertEqual(markers[("base_text_prefill", "attention")]["calls"], 2)
            self.assertEqual(markers[("base_text_prefill", "post_mixer")]["calls"], 1)
            self.assertTrue(result["prefill_stage_attribution"]["kernel_complete"])
            self.assertTrue(result["prefill_stage_attribution"]["memory_copy_complete"])

    def test_region_association_overrides_temporal_mtp_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, report = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute(
                "update kernels set region=? where name='gdn_recurrence_kernel'",
                ("ninfer.gdn.prefill.gdn",))
            connection.commit()
            connection.close()
            result = analyze(database, report)
            categories = {row["category"]: row for row in result["kernel_execution_categories"]}
            self.assertEqual(categories["base_text_prefill"]["calls"], 4)
            self.assertNotIn("mtp_prefill", categories)

    def test_selected_region_association_admits_async_completion_after_host_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, report = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute("insert into kernels values (?,?,?,?,?)", (
                1_000_100_000, 1_000_200_000, 100_000, "async_tail",
                "ninfer.prefill.prefill.chunk payload=8",
            ))
            connection.commit()
            connection.close()
            result = analyze(database, report)
            categories = {row["category"]: row
                          for row in result["kernel_execution_categories"]}
            self.assertEqual(categories["prefill_orchestration"]["calls"], 1)

    def test_unmarked_prefill_work_is_exposed_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, report = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute(
                "update kernels set region='' where name='gdn_recurrence_kernel'")
            connection.execute("update memory_copies set region_name='' where name='D2D'")
            connection.commit()
            connection.close()
            result = analyze(database, report)
            attribution = result["prefill_stage_attribution"]
            self.assertFalse(attribution["kernel_complete"])
            self.assertEqual(attribution["unattributed_kernel_calls"], 1)
            self.assertFalse(attribution["memory_copy_complete"])
            self.assertEqual(attribution["unattributed_memory_copy_calls"], 1)
            operator_attribution = result["operator_stage_attribution"]
            self.assertFalse(operator_attribution["complete"])
            self.assertEqual(operator_attribution["ambiguous_kernel_calls"], 1)

    def test_p2048_operator_families_use_marker_stage_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, report = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            rows = [
                (120_000_000, 121_000_000, 1_000_000,
                 "a8q4g64_linear_prefill_cta_kernel", "ninfer.attention.prefill.attention"),
                (121_000_000, 122_000_000, 1_000_000,
                 "a8g64_quantize_activation_kernel", "ninfer.attention.prefill.attention"),
                (130_000_000, 131_000_000, 1_000_000,
                 "a8w8g32_linear_prefill_cta_kernel", "ninfer.post-mixer.prefill.post_mixer"),
                (131_000_000, 132_000_000, 1_000_000,
                 "a8g32_quantize_activation_kernel", "ninfer.post-mixer.prefill.post_mixer"),
                (132_000_000, 133_000_000, 1_000_000,
                 "fused_silu_a8g64_quantize_kernel",
                 "ninfer.post-mixer.prefill.post_mixer"),
                (140_000_000, 141_000_000, 1_000_000,
                 "fused_attention_causal_kernel", "ninfer.attention.prefill.attention"),
                (141_000_000, 142_000_000, 1_000_000,
                 "dense_full_score_qk_bk32_kernel", "ninfer.attention.prefill.attention"),
                (142_000_000, 143_000_000, 1_000_000,
                 "dense_full_score_maximum_kernel", "ninfer.attention.prefill.attention"),
                (143_000_000, 144_000_000, 1_000_000,
                 "dense_full_score_pv_kernel", "ninfer.attention.prefill.attention"),
                (150_000_000, 151_000_000, 1_000_000,
                 "void ninfer::ops::(anonymous namespace)::ordinary_kernel<true>",
                 "ninfer.gdn.prefill.gdn"),
                (160_000_000, 161_000_000, 1_000_000,
                 "control_gates_kernel", "ninfer.gdn.prefill.gdn"),
                (170_000_000, 171_000_000, 1_000_000,
                 "residual_rmsnorm_kernel", "ninfer.post-mixer.prefill.post_mixer"),
                (180_000_000, 181_000_000, 1_000_000,
                 "silu_mul_kernel", "ninfer.post-mixer.prefill.post_mixer"),
                (190_000_000, 191_000_000, 1_000_000,
                 "residual_add_kernel", "ninfer.post-mixer.prefill.post_mixer"),
                # This same shared quantizer symbol has no stage authority. Temporal
                # containment must not relabel it as attention or post-mixer work.
                (200_000_000, 201_000_000, 1_000_000,
                 "a8g64_quantize_activation_kernel", ""),
                (201_000_000, 202_000_000, 1_000_000,
                 "void ninfer::ops::(anonymous namespace)::ordinary_kernel<true>", ""),
            ]
            connection.executemany("insert into kernels values (?,?,?,?,?)", rows)
            connection.commit()
            connection.close()

            result = analyze(database, report)
            attribution = {
                (row["stage"], row["operator_family"]): row
                for row in result["operator_attribution"]
            }
            for expected in (
                ("attention", "a8q4_prefill_cta"),
                ("attention", "dense_attention"),
                ("gdn", "gdn_recurrence"),
                ("gdn", "gdn_control"),
                ("post_mixer", "a8w8_prefill_cta"),
                ("post_mixer", "fused_silu_a8q4_prepare"),
                ("post_mixer", "rmsnorm"),
                ("post_mixer", "silu_mul"),
                ("post_mixer", "residual_add"),
            ):
                self.assertIn(expected, attribution)
            self.assertEqual(attribution[("attention", "a8q4_activation_quantize")]["calls"], 1)
            self.assertEqual(attribution[("attention", "dense_attention")]["calls"], 4)
            self.assertEqual(attribution[("post_mixer", "a8w8_activation_quantize")]["calls"], 1)
            ambiguous = attribution[("ambiguous", "a8q4_activation_quantize")]
            self.assertTrue(ambiguous["ambiguous"])
            self.assertEqual(ambiguous["calls"], 1)
            self.assertEqual(attribution[("ambiguous", "other")]["calls"], 1)
            self.assertFalse(result["operator_stage_attribution"]["complete"])
            self.assertEqual(
                result["operator_stage_attribution"]["ambiguous_kernel_calls"], 2
            )

    def test_rejects_payload_or_timing_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, report = self.fixture(Path(directory))
            value = json.loads(report.read_text(encoding="utf-8"))
            value["tests"][0]["n_prompt"] = 9
            report.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload sum"):
                analyze(database, report)
            value["tests"][0]["n_prompt"] = 8
            value["tests"][0]["prefill_seconds_mean"] = 0.6
            report.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefill duration"):
                analyze(database, report)

    def test_rejects_activity_outside_selected_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, report = self.fixture(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute("insert into kernels values (?,?,?,?,?)",
                               (1_100_000_000, 1_110_000_000, 10_000_000, "bad", ""))
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "outside the measured range"):
                analyze(database, report)


if __name__ == "__main__":
    unittest.main()
