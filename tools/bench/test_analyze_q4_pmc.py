import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.bench.analyze_q4_pmc import (
    BENCHMARK_COMMANDS, PASS_A_INSTANCE_SUM_COUNTERS, PASS_A_SUM_COUNTERS,
    PASS_B_INSTANCE_SUM_COUNTERS, PASS_B_SUM_COUNTERS, PLAN_SHAPES, PROFILER,
    WORKLOAD, analyze,
)


class AnalyzeQ4PmcTest(unittest.TestCase):
    def fixture(self, root: Path):
        pass_a_counters = sorted(PASS_A_SUM_COUNTERS | PASS_A_INSTANCE_SUM_COUNTERS)
        pass_b_counters = sorted(PASS_B_SUM_COUNTERS | PASS_B_INSTANCE_SUM_COUNTERS)
        counters = sorted(set(pass_a_counters) | set(pass_b_counters))
        plan = root / "plan.json"
        plan.write_text(json.dumps({
            "schema": "ninfer.r9700.production_q4_p2048_pmc_two_pass_plan.v1",
            "status": "command_only_not_executed", "passes": {
                "cache_wait": {"counters": pass_a_counters},
                "issue_lds": {"counters": pass_b_counters},
            },
            "workload": WORKLOAD, "expected_shapes": PLAN_SHAPES,
            "expected_total_dispatches": 176,
            "kernel_include_regex": "a8q4g64_linear_prefill_cta_kernel",
            "profiler": PROFILER, "benchmark_commands": BENCHMARK_COMMANDS,
        }))
        before, after = root / "before", root / "after"
        before.write_text("profile_standard\n")
        after.write_text("auto\n")
        agent = root / "agent.csv"
        with agent.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=[
                "Agent_Type", "Name", "Product_Name", "Cu_Count", "Simd_Count",
                "Wave_Front_Size"])
            writer.writeheader()
            writer.writerow({"Agent_Type": "GPU", "Name": "gfx1201",
                "Product_Name": "AMD Radeon AI PRO R9700", "Cu_Count": 64,
                "Simd_Count": 128, "Wave_Front_Size": 32})
        rows = (["post", 655360] for _ in range(64))
        entries = list(rows)
        entries += [["attention", 655360] for _ in range(16)]
        entries += [["gdn_output", 655360] for _ in range(48)]
        entries += [["gdn_value_z", 1572864] for _ in range(48)]
        regions = {
            "post": "ninfer.post-mixer.prefill.post_mixer payload=0",
            "attention": "ninfer.attention.prefill.attention payload=0",
            "gdn_output": "ninfer.gdn.prefill.gdn payload=0",
            "gdn_value_z": "ninfer.gdn.prefill.gdn payload=0",
        }
        counter_csv = root / "counter.csv"
        fields = ["Dispatch_Id", "Kernel_Name", "Grid_Size", "Workgroup_Size",
                  "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Counter_Name",
                  "Counter_Value", "Start_Timestamp", "End_Timestamp"]
        with counter_csv.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for dispatch, (_role, grid) in enumerate(entries, 1):
                for counter in counters:
                    value = 10
                    if counter == "GRBM_GUI_ACTIVE": value = 1000
                    if counter == "SQ_WAVE_CYCLES": value = 64000
                    if counter == "SQ_WAVES": value = grid // 32
                    if counter == "TCP_REQ": value = 100
                    if counter == "TCP_REQ_MISS": value = 20
                    writer.writerow({"Dispatch_Id": dispatch,
                        "Kernel_Name": "void a8q4g64_linear_prefill_cta_kernel()",
                        "Grid_Size": grid, "Workgroup_Size": 512,
                        "LDS_Block_Size": 17408, "Scratch_Size": 0, "VGPR_Count": 88,
                        "Counter_Name": counter, "Counter_Value": value,
                        "Start_Timestamp": dispatch * 1000, "End_Timestamp": dispatch * 1000 + 1})
        def make_database(path: Path, command: str):
            db = sqlite3.connect(path)
            db.executescript('''
              create table rocpd_info_agent_x(type text,name text,product_name text,extdata text);
              create table rocpd_info_process_x(command text,environment text);
              create table rocpd_kernel_dispatch_x(dispatch_id integer,kernel_id integer,region_name_id integer);
              create table rocpd_info_kernel_symbol_x(id integer,display_name text);
              create table rocpd_string_x(id integer,string text);
            ''')
            db.execute("insert into rocpd_info_agent_x values(?,?,?,?)", ("GPU", "gfx1201",
              "AMD Radeon AI PRO R9700", json.dumps({"cu_count":64,"simd_count":128,"wave_front_size":32})))
            db.execute("insert into rocpd_info_process_x values(?,?)", (command, json.dumps({
              "ROCPROFILER_REGISTER_LIBRARY": "/opt/rocm/core-10.0/lib/librocprofiler-sdk.so.1.3.5"})))
            db.execute("insert into rocpd_info_kernel_symbol_x values(1,?)",
                       ("a8q4g64_linear_prefill_cta_kernel()",))
            for dispatch, (role, _grid) in enumerate(entries, 1):
                db.execute("insert into rocpd_string_x values(?,?)", (dispatch, regions[role]))
                db.execute("insert into rocpd_kernel_dispatch_x values(?,?,?)", (dispatch, 1, dispatch))
            db.commit(); db.close()
        database = root / "capture.db"
        database_b = root / "capture-b.db"
        make_database(database, BENCHMARK_COMMANDS["cache_wait"])
        make_database(database_b, BENCHMARK_COMMANDS["issue_lds"])
        counter_csv_b = root / "counter-b.csv"
        with counter_csv.open(newline="") as source, counter_csv_b.open("w", newline="") as output:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["Counter_Name"] in pass_b_counters:
                    writer.writerow(row)
        with counter_csv.open(newline="") as source:
            reader = csv.DictReader(source)
            filtered = [row for row in reader if row["Counter_Name"] in pass_a_counters]
            fields = reader.fieldnames
        with counter_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader(); writer.writerows(filtered)
        return (plan, counter_csv, database, agent, before, after,
                counter_csv_b, database_b, agent, before, after)

    def test_exact_inventory_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            result = analyze(*paths)
            self.assertEqual(result["dispatch_count_by_pass"], {
                "cache_wait": 176, "issue_lds": 176})
            self.assertEqual(result["native_iu4_wmma_total"], 5368709120)
            self.assertEqual(result["shapes"]["post_mixer_down_n5120_k17408"]
                             ["dispatch_count_by_pass"], {"cache_wait": 64, "issue_lds": 64})
            self.assertAlmostEqual(result["shapes"]["gdn_value_z_n12288_k5120"]
                                   ["metrics_percent"]["l0_vector_cache_hit_percent"], 80.0)
            metrics = result["shapes"]["gdn_value_z_n12288_k5120"]["metrics_percent"]
            self.assertNotIn("valu_busy_percent", metrics)
            self.assertNotIn("memory_unit_busy_percent", metrics)
            self.assertNotIn("ta_busy_percent", metrics)
            self.assertNotIn("write_unit_stalled_percent", metrics)
            self.assertTrue(all(value is None or 0 <= value <= 100
                                for value in metrics.values()))
            normalized = result["shapes"]["gdn_value_z_n12288_k5120"][
                "normalized_activity"]
            self.assertIn("valu_cycles_per_wave", normalized)
            self.assertIn("wave_life_cycles_per_wave", normalized)
            self.assertIn("ta_busy_instance_sum_per_grbm_active_cycle", normalized)
            self.assertIn("gl2_write_stall_instance_sum_per_grbm_active_cycle", normalized)
            self.assertIsNone(result["shapes"]["gdn_output_n5120_k6144"]
                              ["physical_read_bandwidth"])

    def test_rejects_missing_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = list(self.fixture(Path(directory)))
            with paths[1].open(newline="") as source:
                rows = list(csv.reader(source))
            # Remove every counter row for the final dispatch.
            paths[1].write_text("\n".join(",".join(row) for row in rows if not row or row[0] != "176") + "\n")
            with self.assertRaisesRegex(ValueError, "shape inventory differs"):
                analyze(*paths)

    def test_rejects_replayed_dispatch_counter_row(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = list(self.fixture(Path(directory)))
            with paths[1].open(newline="") as source:
                rows = list(csv.reader(source))
            # A native counter CSV has one already instance-reduced value for each
            # dispatch/counter. Appending a replayed row must not silently double-count it.
            with paths[1].open("a", newline="") as output:
                csv.writer(output).writerow(rows[1])
            with self.assertRaisesRegex(ValueError, "possible profiler replay"):
                analyze(*paths)

    def test_rejects_wrong_wave_count_for_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = list(self.fixture(Path(directory)))
            with paths[1].open(newline="") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
                fields = reader.fieldnames
            changed = False
            for row in rows:
                if row["Dispatch_Id"] == "1" and row["Counter_Name"] == "SQ_WAVES":
                    row["Counter_Value"] = "1"
                    changed = True
                    break
            self.assertTrue(changed)
            with paths[1].open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "SQ_WAVES differs"):
                analyze(*paths)


if __name__ == "__main__":
    unittest.main()
