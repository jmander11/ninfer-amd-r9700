from __future__ import annotations

import csv
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from tools.bench import analyze_post_rmsnorm_decode_proxy as analyzer
from tools.bench.prepare_post_rmsnorm_decode_proxy import PASSES


class ReductionTests(unittest.TestCase):
    def test_each_pass_is_bounded(self) -> None:
        self.assertEqual(set(PASSES), {"cache-wait", "issue-lds"})
        self.assertEqual(sum(name.startswith(("SQ_", "SQC_"))
                             for name in PASSES["cache-wait"]), 4)
        self.assertEqual(sum(name.startswith(("SQ_", "SQC_"))
                             for name in PASSES["issue-lds"]), 8)
        self.assertTrue({"GRBM_GUI_ACTIVE", "TA_TA_BUSY", "GL2C_MC_WRREQ_STALL"}
                        .issubset(PASSES["cache-wait"]))
        self.assertTrue({"GRBM_GUI_ACTIVE", "TA_TA_BUSY"}.issubset(PASSES["issue-lds"]))

    def test_native_dispatch_values_are_summed_before_ratios(self) -> None:
        counters = PASSES["cache-wait"]
        fields = ["Dispatch_Id", "Kernel_Name", "Grid_Size", "Workgroup_Size",
                  "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Counter_Name",
                  "Counter_Value"]
        regions = {}
        markers = {f"ninfer.decode.decode.ordinary_round payload={frontier}": 1
                   for frontier in range(8192, 8448)}
        with tempfile.TemporaryDirectory() as raw:
            csv_path = Path(raw) / "counter.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
                for dispatch, multiplier in ((7, 1), (9, 10)):
                    regions[dispatch] = {
                        "region": f"ninfer.decode.decode.ordinary_round payload={8192 + dispatch - 7}",
                        "symbol": "kernel", "grid_size": 256, "workgroup_size": 256,
                        "static_lds_bytes": 4, "scratch_bytes": 0, "vgpr": 16,
                    }
                    for counter in counters:
                        writer.writerow({"Dispatch_Id": dispatch, "Kernel_Name": "kernel",
                                         "Grid_Size": 256, "Workgroup_Size": 256,
                                         "LDS_Block_Size": 512, "Scratch_Size": 0,
                                         "VGPR_Count": 16, "Counter_Name": counter,
                                         "Counter_Value": multiplier})
            with patch.object(analyzer, "read_database",
                              return_value=(regions, {"wave_front_size": 32}, markers)):
                result = analyzer.reduce_pass(csv_path, Path(raw) / "unused.db", [], counters)
                with csv_path.open(newline="", encoding="utf-8") as source:
                    duplicate = next(csv.DictReader(source))
                with csv_path.open("a", newline="", encoding="utf-8") as output:
                    csv.DictWriter(output, fieldnames=fields).writerow(duplicate)
                with self.assertRaisesRegex(ValueError, "duplicate counter"):
                    analyzer.reduce_pass(csv_path, Path(raw) / "unused.db", [], counters)
        self.assertEqual(result["ordinary_dispatch_count"], 2)
        self.assertTrue(all(value == "11" for value in result["counter_sums"].values()))
        self.assertEqual(result["dispatch_inventory"][0]["dispatch_count"], 2)

    def test_stream_probe_parser_requires_exact_large_working_set(self) -> None:
        rows = "\n".join(
            f"{name:<28}        600.0        610.0        605.0      95.3%      1.000"
            for name in ("hipMemsetAsync (write)", "kernel uint4 write", "kernel uint4 read",
                         "kernel uint4 copy", "kernel uint4x4 copy", "hipMemcpyAsync D2D"))
        text = ("GPU:               AMD Radeon AI PRO R9700\n"
                "Architecture:      gfx1201 (wave32)\n"
                "Working set:       4.00 GiB per buffer (32.0x L2)\n"
                "Trial policy:      5 trials, >= 0.25 s each, best + median\n" + rows + "\n"
                "Read checksum exact:     01234567:89abcdef:01234567:89abcdef\n")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "probe.txt"; path.write_text(text, encoding="utf-8")
            result = analyzer.parse_stream_probe(path)
        self.assertEqual(result["best_observed_bus_gbps"], 610.0)
        self.assertEqual(len(result["methods"]), 6)

    def test_percent_rejects_negative_numerator(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid operands"):
            analyzer.percent(Decimal(-1), Decimal(10), "negative hit count")

    def test_trace_inventory_requires_and_scales_exact_one_round(self) -> None:
        row = {
            "symbol": "kernel", "grid_size": 256, "workgroup_size": 256,
            "static_lds_bytes": 4, "scratch_bytes": 0, "vgpr": 16,
            "dispatch_count": analyzer.ONE_ROUND_DISPATCHES - 16,
        }
        qk = {
            "symbol": "void qk_wmma_kernel<true>(...)", "grid_size": 65664,
            "workgroup_size": 32, "static_lds_bytes": 0, "scratch_bytes": 0,
            "vgpr": 24, "dispatch_count": 16,
        }
        scaled = analyzer.trace_graph_inventory([row, qk], analyzer.QK_GRID_CONTRACT)
        self.assertEqual(scaled[0]["allocated_lds_bytes"], 512)
        self.assertEqual(sum(item["dispatch_count"] for item in scaled),
                         analyzer.ONE_ROUND_DISPATCHES)
        self.assertEqual(next(item for item in scaled if "qk_wmma" in item["symbol"])["grid_size"],
                         67584)
        with self.assertRaisesRegex(ValueError, "exact 1806-dispatch"):
            analyzer.trace_graph_inventory([{**row, "dispatch_count": 1}, qk],
                                           analyzer.QK_GRID_CONTRACT)
        with self.assertRaisesRegex(ValueError, "QK grid contract"):
            analyzer.trace_graph_inventory([row, qk],
                                           {**analyzer.QK_GRID_CONTRACT,
                                            "capture_grid_size": 1})


if __name__ == "__main__":
    unittest.main()
