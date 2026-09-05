import unittest

from tools.bench.produce_fp8_hybrid_capacity import assemble, parse_planner


CSV = """concurrency,minimum_groups,maximum_groups,minimum_sequence_bytes,workspace_bytes,graph_allowance_bytes,request_transient_bytes,minimum_reservation_bytes,kv_payload_bytes,kv_increment_bytes
1,4096,4096,100,20,10,0,130,80,1810432
2,4096,8192,200,20,20,0,240,160,1810432
3,4096,12288,300,20,30,0,350,240,1810432
4,4096,16384,400,20,40,0,460,320,1810432
"""


class HybridCapacityTest(unittest.TestCase):
    def test_exact_cells_and_budget(self):
        report = assemble(parse_planner(CSV), 1000, (1 << 30) + 2000)
        self.assertEqual(report["runtime_budget_after_weights_and_headroom_bytes"], 1000)
        self.assertEqual(report["cells"][0]["resolved_groups"], 4096)
        self.assertEqual(report["cells"][0]["remaining_slack_bytes"], 870)
        self.assertFalse(report["prior_c4_slack"]["confirmed_exact"])

    def test_missing_concurrency_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_planner("\n".join(CSV.splitlines()[:-1]))


if __name__ == "__main__":
    unittest.main()
