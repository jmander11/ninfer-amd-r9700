#!/usr/bin/env python3

from __future__ import annotations

import unittest

from tools.bench.prepare_qwen3_8_27b_dispatch_schedule import (
    SOURCE_AUTHORITIES,
    _post_marker,
    _source_marker,
    build_schedule,
)
from tools.bench.model_qwen3_8_27b_roofline import FORMATS, _counts


def _row(index: int, marker: str | None, symbol: str, grid_x: int = 1,
         workgroup_x: int = 64, grid_y: int = 1) -> dict:
    return {
        "dispatch_id": f"trace:{index}", "rocprof_dispatch_id": index,
        "roctx_region": marker, "symbol": symbol, "start_ns": index * 10,
        "end_ns": index * 10 + 5, "duration_ns": 5, "stream_id": 1,
        "grid": {"x": grid_x, "y": grid_y, "z": 1},
        "workgroup": {"x": workgroup_x, "y": 1, "z": 1},
        "resources": {"sgpr_count": None, "vgpr_count": None, "accum_vgpr_count": None,
                      "lds_bytes": None, "scratch_bytes": None,
                      "static_lds_bytes": None, "static_scratch_bytes": None},
    }


class PrepareDispatchScheduleTest(unittest.TestCase):
    def fixture(self) -> tuple[dict, list[dict], dict]:
        workload = {"concurrency": 1, "prompt_tokens": 2048, "generated_tokens": 0,
                    "spec": "none", "draft_tokens": 0, "dflash_verify_width": 0,
                    "kv_value_group": 16, "xattention_profile": "dense",
                    "prefill_chunk": 2048}
        trace = {"workload": workload, "selected_route": {"fixture": True}}
        rows = []
        tensors = {}
        for layer in range(64):
            full = layer >= 3 and (layer - 3) % 4 == 0
            if full:
                roles = (("attention/query_key", 7168, 5120),
                         ("attention/gate_value", 7168, 5120),
                         ("attention/output", 5120, 6144))
            else:
                roles = (("gdn/query_key", 4096, 5120),
                         ("gdn/value_z", 12288, 5120),
                         ("gdn/output", 5120, 6144))
            marker = _source_marker(layer)
            rows.append(_row(len(rows), marker, "ops::rmsnorm_kernel(", 2048 * 256, 256))
            if not full:
                rows.append(_row(len(rows), marker, "control_gates_kernel", 48 * 2048, 256))

            def append_pair(spec: tuple[str, int, int]) -> None:
                suffix, r, c = spec
                tensors[f"text/layers/{layer}/{suffix}"] = ((r, c), "Q4G64_F16S")
                rows.append(_row(len(rows), marker, "a8g64_quantize_activation_kernel",
                                 (c // 64) * 32, 32, 2048))
                rows.append(_row(len(rows), marker, "a8q4g64_linear_prefill_cta_kernel",
                                 ((r + 63) // 64) * 512, 512, 32))

            append_pair(roles[0])
            append_pair(roles[1])
            if full:
                rows.append(_row(len(rows), marker,
                                 "fused_attention_causal_kernel<16u, true>",
                                 24 * 256, 256, 2048))
                append_pair(roles[2])
                rows.append(_row(len(rows), marker, "residual_add_kernel", 5120 * 2048, 256))
            else:
                rows.append(_row(len(rows), marker, "causal_conv1d_silu_kernel", 10240, 256))
                rows.append(_row(len(rows), marker, "ordinary_kernel<true>", 48 * 4 * 256, 256))
                rows.append(_row(len(rows), marker, "ops::gated_rmsnorm_kernel(",
                                 48 * 2048 * 256, 256))
                append_pair(roles[2])
                rows.append(_row(len(rows), marker, "residual_add_kernel", 5120 * 2048, 256))
            marker = _post_marker(layer)
            rows.append(_row(len(rows), marker, "ops::rmsnorm_kernel(", 2048 * 256, 256))
            post_roles = (("mlp/gate_up", 34816, 5120),
                          ("mlp/down", 5120, 17408))
            for suffix, r, c in post_roles:
                tensors[f"text/layers/{layer}/{suffix}"] = ((r, c), "Q4G64_F16S")
            suffix, r, c = post_roles[0]
            rows.append(_row(len(rows), marker, "a8g64_quantize_activation_kernel",
                             (c // 64) * 32, 32, 2048))
            rows.append(_row(len(rows), marker, "a8q4g64_linear_prefill_cta_kernel",
                             ((r + 63) // 64) * 512, 512, 32))
            rows.append(_row(len(rows), marker, "silu_mul_strided_kernel", 17408 * 2048, 256))
            suffix, r, c = post_roles[1]
            rows.append(_row(len(rows), marker, "a8g64_quantize_activation_kernel",
                             (c // 64) * 32, 32, 2048))
            rows.append(_row(len(rows), marker, "a8q4g64_linear_prefill_cta_kernel",
                             ((r + 63) // 64) * 512, 512, 32))
            rows.append(_row(len(rows), marker, "residual_add_kernel", 5120 * 2048, 256))
        return trace, rows, tensors

    def test_source_program_order_and_artifact_formats_drive_roles(self) -> None:
        trace, rows, tensors = self.fixture()
        tensors["text/layers/0/gdn/query_key"] = ((4096, 5120), "W8G32_F16S")
        rows[2].update(symbol="a8g32_quantize_activation_kernel",
                       grid={"x": (5120 // 32) * 32, "y": 2048, "z": 1})
        rows[3]["symbol"] = "a8w8g32_linear_prefill_cta_kernel"
        result = build_schedule(trace, rows, tensors, [{"source": "fixture"}])
        self.assertEqual(len(result["dispatches"]), len(rows))
        modeled = [row for row in result["dispatches"]
                   if row["classification"] == "modeled"]
        self.assertEqual(len(modeled), 64 * 13 + 48 * 5 + 16 * 3)
        attention = next(row for row in modeled if row["operation"] == "dense_attention")
        self.assertEqual(attention["layer"], {"kind": "text", "index": 3})
        self.assertEqual(attention["parameters"]["context_start"], 0)
        self.assertEqual(attention["parameters"]["represented_metadata_bytes"], 8320)
        self.assertEqual(result["dispatches"][2]["operation"], "w8_quantize")
        self.assertEqual(result["dispatches"][3]["operation"], "w8_linear")
        self.assertEqual(sum(row["operation"] == "gdn_recurrence" for row in modeled), 48)
        self.assertEqual(sum(row["operation"] == "gdn_conv" for row in modeled), 48)
        self.assertEqual(sum(row["operation"] == "gated_rmsnorm" for row in modeled), 48)
        self.assertEqual(sum(row["operation"] == "silu_mul" for row in modeled), 64)
        for row in modeled:
            self.assertEqual(row["format"], FORMATS[row["operation"]])
            self.assertGreaterEqual(
                _counts(row["operation"], row["parameters"], "fixture", {})[
                    "represented_minimum_bytes"], 0)
        uncovered = [row for row in result["dispatches"]
                     if row["classification"] == "unmodeled"]
        self.assertTrue(uncovered)
        self.assertTrue(all(row["role"] == "unassigned"
                            and row["operation"] == "unmodeled_dispatch"
                            and "reason" in row for row in uncovered))

    def test_rejects_marker_order_format_and_quantize_matrix_ambiguity(self) -> None:
        trace, rows, tensors = self.fixture()
        rows[0]["roctx_region"] = _source_marker(1)
        with self.assertRaisesRegex(ValueError, "marker order"):
            build_schedule(trace, rows, tensors, [])

        trace, rows, tensors = self.fixture()
        tensors["text/layers/0/gdn/query_key"] = ((4096, 5120), "W8G32_F16S")
        with self.assertRaisesRegex(ValueError, "kernel format differs"):
            build_schedule(trace, rows, tensors, [])

        trace, rows, tensors = self.fixture()
        rows[2], rows[3] = rows[3], rows[2]
        with self.assertRaisesRegex(ValueError, "kernel format differs"):
            build_schedule(trace, rows, tensors, [])

        trace, rows, tensors = self.fixture()
        conv = next(i for i, row in enumerate(rows)
                    if "causal_conv1d_silu_kernel" in row["symbol"])
        recurrence = next(i for i, row in enumerate(rows)
                          if "ordinary_kernel<true>" in row["symbol"])
        rows[conv], rows[recurrence] = rows[recurrence], rows[conv]
        with self.assertRaisesRegex(ValueError, "modeled dispatch order"):
            build_schedule(trace, rows, tensors, [])

        trace, rows, tensors = self.fixture()
        gated = next(i for i, row in enumerate(rows)
                     if "gated_rmsnorm_kernel" in row["symbol"])
        rows[gated]["grid"]["x"] -= 1
        with self.assertRaisesRegex(ValueError, "gdn_gated_rmsnorm launch geometry"):
            build_schedule(trace, rows, tensors, [])

    def test_unmarked_dispatch_is_retained_but_never_modeled(self) -> None:
        trace, rows, tensors = self.fixture()
        rows.append(_row(len(rows), None, "ambiguous_kernel"))
        result = build_schedule(trace, rows, tensors, [])
        self.assertIsNone(result["dispatches"][-1]["marker"])
        self.assertEqual(result["dispatches"][-1]["classification"], "unmodeled")
        self.assertEqual(result["dispatches"][-1]["role"], "unassigned")
        self.assertEqual(result["dispatches"][-1]["operation"], "unmodeled_dispatch")

    def test_sparse_hybrid_route_has_exact_attention_and_fp8_role_accounting(self) -> None:
        trace, rows, tensors = self.fixture()
        trace["workload"]["xattention_profile"] = "b128-s16-tau900"
        replacements = []
        for layer in range(64):
            full = layer >= 3 and (layer - 3) % 4 == 0
            marker = _source_marker(layer)
            block = [i for i, row in enumerate(rows) if row["roctx_region"] == marker]
            matrices = [i for i in block if "a8q4g64_linear_prefill_cta_kernel" in rows[i]["symbol"]]
            selected = matrices[:2] if full else matrices[:1]
            tensor_names = ([f"text/layers/{layer}/attention/query_key",
                             f"text/layers/{layer}/attention/gate_value"] if full else
                            [f"text/layers/{layer}/gdn/query_key"])
            for matrix, tensor in zip(selected, tensor_names, strict=True):
                replacements.append((matrix - 1, matrix, marker))
                tensors[tensor] = (tensors[tensor][0], "F8E4M3_ROW_F32S")
            post_marker = _post_marker(layer)
            post_block = [i for i, row in enumerate(rows) if row["roctx_region"] == post_marker]
            matrix = next(i for i in post_block
                          if "a8q4g64_linear_prefill_cta_kernel" in rows[i]["symbol"])
            replacements.append((matrix - 1, matrix, post_marker))
            name = f"text/layers/{layer}/mlp/gate_up"
            tensors[name] = (tensors[name][0], "F8E4M3_ROW_F32S")
        for quant, matrix, marker in sorted(replacements, reverse=True):
            rows[quant:matrix + 1] = [
                _row(0, marker, "fp8_quantize_activation_kernel", 2048, 256),
                _row(0, marker, "Cijk_Alik_Bljk_F8BS_fixture", 7, 256, 11),
                _row(0, marker, "poison_nonfinite_output", 1, 256),
            ]
        for layer in reversed(range(64)):
            if layer >= 3 and (layer - 3) % 4 == 0:
                marker = _source_marker(layer)
                dense = next(i for i, row in enumerate(rows)
                             if row["roctx_region"] == marker
                             and "fused_attention_causal_kernel" in row["symbol"])
                rows[dense:dense + 1] = [
                    _row(0, marker, "xattention_pack_keys_kernel<16u, true>", 4, 256, 32),
                    _row(0, marker, "xattention_rank_kernel<16u>", 24, 256, 16),
                    _row(0, marker, "xattention_flash_consumer_kernel<16u, false, false>",
                         24, 256, 128),
                ]
        for index, row in enumerate(rows):
            row["dispatch_id"] = f"trace:{index}"
            row["rocprof_dispatch_id"] = index
            row["start_ns"] = index * 10
            row["end_ns"] = index * 10 + 5
        result = build_schedule(trace, rows, tensors, [{"source": "fixture"}])
        modeled = [row for row in result["dispatches"]
                   if row["classification"] == "modeled"]
        self.assertEqual(sum(row["operation"] == "sparse_pack" for row in modeled), 16)
        self.assertEqual(sum(row["operation"] == "sparse_rank" for row in modeled), 16)
        self.assertEqual(sum(row["operation"] == "sparse_consumer" for row in modeled), 16)
        fp8_roles = [row["role"] for row in modeled if row["operation"] == "fp8_linear"]
        self.assertEqual(len(fp8_roles), 144)
        self.assertEqual(fp8_roles.count("mlp_gate_up"), 64)
        self.assertEqual(fp8_roles.count("gdn_query_key"), 48)
        self.assertEqual(fp8_roles.count("attention_query_key"), 16)
        self.assertEqual(fp8_roles.count("attention_gate_value"), 16)

    def test_source_authorities_include_recursive_kernel_dependencies(self) -> None:
        relative = {path.name for path in SOURCE_AUTHORITIES}
        self.assertIn("eager_ops.hip", relative)
        self.assertIn("gated_delta_net.hip", relative)
        self.assertIn("linear_execution.hip", relative)
        self.assertIn("fp8_activation.hip", relative)
        self.assertIn("fp8_int4_kv_xattention.hip", relative)
        self.assertIn("tensor.h", relative)
        self.assertGreater(len(SOURCE_AUTHORITIES), 40)


if __name__ == "__main__":
    unittest.main()
