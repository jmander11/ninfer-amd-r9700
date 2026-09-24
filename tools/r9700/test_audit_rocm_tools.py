#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from audit_rocm_tools import _supported_compute_architectures, summarize_database


class RocprofEvidenceTest(unittest.TestCase):
    def test_summary_joins_random_suffix_tables_and_counts_zero_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "evidence.db"
            connection = sqlite3.connect(database)
            suffix = "_00000001_fixture"
            connection.executescript(f'''
                create table rocpd_info_process{suffix} (id integer, command text);
                insert into rocpd_info_process{suffix} values (1, 'fixture --context 66');
                create table rocpd_info_agent{suffix}
                    (id integer, type text, logical_index integer, name text,
                     model_name text, product_name text);
                insert into rocpd_info_agent{suffix}
                    values (1, 'GPU', 0, 'gfx1201', 'ip discovery',
                            'AMD Radeon AI PRO R9700');
                create table rocpd_metadata{suffix} (id integer, tag text, value text);
                insert into rocpd_metadata{suffix} values (1, 'schema_version', '3.0.3');
                create table rocpd_kernel_dispatch{suffix} (id integer);
                insert into rocpd_kernel_dispatch{suffix} values (1);
                create table rocpd_memory_copy{suffix} (id integer);
                insert into rocpd_memory_copy{suffix} values (1);
                create table rocpd_region{suffix} (id integer);
                insert into rocpd_region{suffix} values (1);
                create table rocpd_info_pmc{suffix} (id integer, symbol text);
                insert into rocpd_info_pmc{suffix} values (7, 'SQ_WAVES');
                create table rocpd_pmc_event{suffix} (id integer, pmc_id integer, value real);
                insert into rocpd_pmc_event{suffix} values (1, 7, 0), (2, 7, 4);
            ''')
            connection.commit()
            connection.close()

            summary = summarize_database(database)
            self.assertEqual(summary["commands"], ["fixture --context 66"])
            self.assertEqual(summary["row_counts"], {
                "kernel_dispatch": 1, "memory_copy": 1, "runtime_region": 1})
            self.assertEqual(summary["counters"]["SQ_WAVES"]["samples"], 2)
            self.assertEqual(summary["counters"]["SQ_WAVES"]["nonzero_samples"], 1)
            self.assertEqual(summary["counters"]["SQ_WAVES"]["sum"], 4)

    def test_compute_architecture_parser_is_exact(self) -> None:
        architectures = _supported_compute_architectures("gfx950 gfx1150 gfx1153")
        self.assertEqual(architectures, ["gfx1150", "gfx1153", "gfx950"])
        self.assertNotIn("gfx1201", architectures)


if __name__ == "__main__":
    unittest.main()
