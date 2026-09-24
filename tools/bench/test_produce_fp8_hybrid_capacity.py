import unittest
import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.bench.produce_fp8_hybrid_capacity import (
    DENSE_FULL_HEAD_PAYLOAD_BYTES,
    DENSE_FULL_HEAD_TENSOR_COUNT,
    DENSE_FULL_HEAD_WEIGHT_BYTES,
    MTP3_OPTIMIZED_WEIGHT_BYTES,
    _durable_create,
    assemble,
    capacity_weights,
    current_report,
    parse_planner,
)


CSV = """target,weights_profile,capacity_tokens,page_tokens,prefill_chunk,kv_value_group,speculative_backend,draft_tokens,proposal_head,device_graph,concurrency,minimum_groups,maximum_groups,minimum_sequence_bytes,workspace_bytes,graph_allowance_bytes,request_transient_bytes,minimum_reservation_bytes,kv_payload_bytes,kv_increment_bytes
qwen3_8_27b_r9700,R9700Q4G64Fp8FourRoleN16K16Evaluation,262144,64,8192,16,mtp,3,optimized,1,1,4096,4096,100,20,10,0,130,80,1810432
qwen3_8_27b_r9700,R9700Q4G64Fp8FourRoleN16K16Evaluation,262144,64,8192,16,mtp,3,optimized,1,2,4096,8192,200,20,20,0,240,160,1810432
qwen3_8_27b_r9700,R9700Q4G64Fp8FourRoleN16K16Evaluation,262144,64,8192,16,mtp,3,optimized,1,3,4096,12288,300,20,30,0,350,240,1810432
qwen3_8_27b_r9700,R9700Q4G64Fp8FourRoleN16K16Evaluation,262144,64,8192,16,mtp,3,optimized,1,4,4096,16384,400,20,40,0,460,320,1810432
"""


def weights_source() -> dict[str, object]:
    return {
        "artifact_type": "ninfer_bench_report",
        "schema_version": 20,
        "tool": "ninfer_bench",
        "load": {
            "target": "qwen3_8_27b_r9700",
            "weights_id": "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
            "host_to_device_bytes": DENSE_FULL_HEAD_PAYLOAD_BYTES,
            "tensor_count": DENSE_FULL_HEAD_TENSOR_COUNT,
            "resource_count": 6,
        },
        "config": {
            "max_context": 2048,
            "concurrency": 1,
            "spec": "none",
            "draft_tokens": 0,
            "proposal_head": "full",
            "use_device_graph": True,
            "decode_path": "device_graph",
            "speculative_execution": False,
            "prefill_chunk": 4096,
            "kv_cache_format": "fp8-k-int4-v",
            "kv_value_group": 16,
            "kv_plane_layouts": {
                "key": "token-fastest-head-major",
                "value": "feature-fastest-page-major",
                "value_scale": "feature-fastest-page-major",
            },
            "q4_activation_bits": 8,
            "q4_prefill_cta_profile":
                "m64n128-pingpong-n16-k16-scalar-base-production",
            "xattention_qualification": False,
        },
        "environment": {
            "device_id": 0,
            "gpu_name": "AMD Radeon AI PRO R9700",
            "architecture_name": "gfx1201",
        },
        "memory": {
            "device": 0,
            "kv_cache_format": "fp8-k-int4-v",
            "weights": {
                "capacity_bytes": DENSE_FULL_HEAD_WEIGHT_BYTES,
                "used_bytes": DENSE_FULL_HEAD_WEIGHT_BYTES,
                "peak_used_bytes": DENSE_FULL_HEAD_WEIGHT_BYTES,
            },
            "available_after_weights_bytes": 13_270_777_856,
        },
    }


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

    def test_planner_profile_header_and_arithmetic_fail_closed(self):
        mutations = (
            ("proposal_head", "full", "proposal_head"),
            ("maximum_groups", "0", "maximum groups"),
            ("minimum_reservation_bytes", "129", "reservation arithmetic"),
            ("kv_payload_bytes", "101", "KV payload"),
            ("workspace_bytes", "20.0", "exact nonnegative integer"),
        )
        header, *lines = CSV.strip().splitlines()
        fields = header.split(",")
        for field, replacement, message in mutations:
            with self.subTest(field=field):
                values = lines[0].split(",")
                values[fields.index(field)] = replacement
                changed = "\n".join((header, ",".join(values), *lines[1:])) + "\n"
                with self.assertRaisesRegex(ValueError, message):
                    parse_planner(changed)
        with self.assertRaisesRegex(ValueError, "unexpected CSV header"):
            parse_planner(CSV.replace("target,weights_profile", "weights_profile,target", 1))

    def test_unresolved_minimum_is_not_capacity_preserved(self):
        report = assemble(parse_planner(CSV), 1000, (1 << 30) + 1100)
        self.assertIsNone(report["cells"][0]["resolved_groups"])
        self.assertFalse(report["cells"][0]["capacity_preserved"])

    def test_dense_observation_expands_to_mtp3_materialization(self):
        source = weights_source()
        weights, device, arithmetic = capacity_weights(source)
        self.assertEqual(DENSE_FULL_HEAD_WEIGHT_BYTES, 20_707_768_320)
        self.assertEqual(MTP3_OPTIMIZED_WEIGHT_BYTES, 21_290_468_352)
        self.assertEqual(weights, MTP3_OPTIMIZED_WEIGHT_BYTES)
        self.assertEqual(device, 33_978_546_176)
        self.assertEqual(arithmetic["mtp_and_optimized_draft_increment_bytes"], 582_700_032)

    def test_wrong_execution_profile_fails_closed(self):
        source = weights_source()
        source["config"]["spec"] = "mtp"
        with self.assertRaisesRegex(ValueError, "config.spec"):
            capacity_weights(source)

    def test_stale_dense_materialization_fails_closed(self):
        source = weights_source()
        source["memory"]["weights"]["capacity_bytes"] -= 256
        source["memory"]["weights"]["used_bytes"] -= 256
        source["memory"]["weights"]["peak_used_bytes"] -= 256
        with self.assertRaisesRegex(ValueError, "current hybrid inventory"):
            capacity_weights(source)

    def test_inconsistent_weight_arena_counters_fail_closed(self):
        source = weights_source()
        source["memory"]["weights"]["peak_used_bytes"] -= 256
        with self.assertRaisesRegex(ValueError, "capacity, use, and peak differ"):
            capacity_weights(source)

    def test_wrong_weights_identity_fails_closed(self):
        source = weights_source()
        source["load"]["weights_id"] = "r9700-q4g64-f8e4m3-four-role-eval"
        with self.assertRaisesRegex(ValueError, "load.weights_id"):
            capacity_weights(source)

    def test_string_and_float_byte_values_fail_closed(self):
        for path, replacement in (
            (("memory", "weights", "capacity_bytes"), str(DENSE_FULL_HEAD_WEIGHT_BYTES)),
            (("memory", "weights", "used_bytes"), float(DENSE_FULL_HEAD_WEIGHT_BYTES)),
            (("memory", "available_after_weights_bytes"), 13_270_777_856.0),
        ):
            with self.subTest(path=path):
                source = deepcopy(weights_source())
                target = source
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaisesRegex(ValueError, "exact integer"):
                    capacity_weights(source)

    def test_wrong_hardware_cache_or_dense_profile_fails_closed(self):
        for section, key, replacement in (
            ("environment", "gpu_name", "AMD Radeon RX 7900 XTX"),
            ("environment", "architecture_name", "gfx1100"),
            ("config", "kv_value_group", 32),
            ("config", "xattention_qualification", True),
            ("config", "q4_prefill_cta_profile", "other"),
            ("memory", "kv_cache_format", "other"),
        ):
            with self.subTest(section=section, key=key):
                source = deepcopy(weights_source())
                source[section][key] = replacement
                with self.assertRaisesRegex(ValueError, f"{section}.{key}"):
                    capacity_weights(source)

    def test_create_only_publication_does_not_overwrite(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "capacity.json"
            _durable_create(output, "first\n")
            with self.assertRaisesRegex(ValueError, "already exists"):
                _durable_create(output, "second\n")
            self.assertEqual(output.read_text(), "first\n")

    def test_current_report_binds_executed_planner_and_read_report_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            planner = root / "planner"
            planner.write_text(f"#!/usr/bin/env python3\nprint({CSV!r}, end='')\n")
            planner.chmod(0o755)
            weights = root / "weights.json"
            weights.write_text(json.dumps(weights_source()))
            report = current_report(planner, weights)
            self.assertEqual(report["provenance"]["planner_executable"]["path"], str(planner))
            self.assertEqual(report["provenance"]["weights_report"]["path"], str(weights))
            self.assertTrue(report["cells"][0]["capacity_preserved"])

    def test_symlink_inputs_fail_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            planner = root / "planner"
            planner.write_text(f"#!/usr/bin/env python3\nprint({CSV!r}, end='')\n")
            planner.chmod(0o755)
            report = root / "weights.json"
            report.write_text(json.dumps(weights_source()))
            linked = root / "linked.json"
            os.symlink(report, linked)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                current_report(planner, linked)


if __name__ == "__main__":
    unittest.main()
