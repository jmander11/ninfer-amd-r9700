import json
import struct
import tempfile
import unittest
from pathlib import Path

from analyze_qwen3_layer3_attention_trace import (
    CACHE_BYTES, FIELDS, FIELD_BY_NAME, ROLES, SIDECAR_BYTES, STAGE_BYTES, STAGE_FIELDS,
    analyze, fnv1a64,
)


class Layer3AttentionTraceTest(unittest.TestCase):
    def test_exact_layout_contract(self):
        self.assertEqual(len(STAGE_FIELDS), 12)
        self.assertEqual(STAGE_BYTES, 137216)
        self.assertEqual(FIELD_BY_NAME["value_scale_fp16"]["bytes"], 16640)
        self.assertEqual(CACHE_BYTES, 216320)
        self.assertEqual(SIDECAR_BYTES, 353536)

    def fixture(self, directory: Path, kind: str, suffix: str, mutate=None,
                visibility_override=None) -> Path:
        role, width, frontier = ROLES[kind]
        data = bytearray(SIDECAR_BYTES)
        if mutate is not None:
            field_name, byte_index, value = mutate
            field = FIELD_BY_NAME[field_name]
            data[field["offset"] + byte_index] = value
        sidecar = (directory / suffix).with_suffix(".bin")
        sidecar.write_bytes(data)
        visibility = ({"row_position": 129, "ancestor_masks": None,
                       "prefix_lengths": None, "prefix_length_stride": 1}
                      if kind == "ordinary" else
                      {"row_position": 129, "ancestor_masks": [1, 3, 7, 15, 31],
                       "prefix_lengths": [129], "prefix_length_stride": 0})
        if visibility_override is not None:
            visibility = visibility_override
        manifest = (directory / suffix).with_suffix(".json")
        manifest.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_layer3_attention_trace", "schema_version": 1,
            "diagnostic_only": True, "timing_evidence_eligible": False,
            "production_routing_authorized": False, "execution": "eager", "role": role,
            "call": {"phase": "verify", "batch": 1, "width": width,
                     "selected_column": 0, "absolute_frontier": 130, "token": 96558,
                     "cache_position": 129, "rope_position": 129, "text_layer": 3,
                     "full_attention_index": 0, "sequence_batch": 1,
                     "sequence_width": width, "transaction_position_count": width,
                     "live_width": width},
            "cache_read": {"valid_for_stream": True, "pending": True,
                           "visible_frontier": frontier, "mapped_pages": 1,
                           "head_dim": 256, "kv_heads": 4, "value_group": 16,
                           "key_layout": "token-fastest-head-major",
                           "value_layout": "feature-fastest-page-major",
                           "value_scale_layout": "feature-fastest-page-major",
                           "device_table_row_present": True,
                           "pool_table_row_stride": 256, "pool_table_row_count": 4,
                           "active_query_rows_present": False},
            "visibility": visibility,
            "provenance": {"source_commit": "1" * 40, "source_tree": "2" * 40,
                           "executable_sha256": "3" * 64, "artifact_sha256": "4" * 64,
                           "history_sha256": "5" * 64, "corpus_sha256": "6" * 64,
                           "device_index": 0, "device_name": "AMD Radeon AI PRO R9700",
                           "architecture": "gfx1201", "wave_size": 32,
                           "pci": "0000:13:00.0", "power_profile_source": "/sys/power",
                           "power_profile_value": "auto"},
            "fields": FIELDS, "sidecar_path": str(sidecar),
            "sidecar_bytes": SIDECAR_BYTES, "sidecar_fnv1a64": fnv1a64(data),
            "layout": "typed selected-column layer3 attention stages then canonical logical cache positions0..129",
        }), encoding="utf-8")
        return manifest

    def test_exact_and_stage_precedence(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            ordinary = self.fixture(directory, "ordinary", "ordinary")
            dflash = self.fixture(directory, "dflash", "dflash")
            self.assertEqual(analyze(ordinary, dflash)["classification"],
                             "layer3_attention_internal_exact")
        cases = (("norm_h", "input_rmsnorm_difference"),
                 ("projection_v", "attention_projection_difference"),
                 ("normalized_qk", "qk_rmsnorm_difference"),
                 ("rope_qk", "rope_or_position_difference"),
                 ("attention_fp32", "full_attention_route_arithmetic_difference"),
                 ("attention_bf16", "fp32_to_bf16_cast_difference"),
                 ("gated_attention", "sigmoid_gate_difference"),
                 ("residual_x", "attention_output_projection_or_residual_difference"))
        for field, expected in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                ordinary = self.fixture(directory, "ordinary", "ordinary")
                dflash = self.fixture(directory, "dflash", "dflash", (field, 0, 1))
                self.assertEqual(analyze(ordinary, dflash)["classification"], expected)

    def test_visibility_and_cache_classification(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            ordinary = self.fixture(directory, "ordinary", "ordinary")
            visibility = {"row_position": 129, "ancestor_masks": [3, 3, 7, 15, 31],
                          "prefix_lengths": [129], "prefix_length_stride": 0}
            dflash = self.fixture(directory, "dflash", "dflash",
                                  ("key_fp8", 0, 1), visibility_override=visibility)
            self.assertEqual(analyze(ordinary, dflash)["classification"],
                             "tree_visibility_difference")
        key = next(field for field in FIELDS if field["name"] == "key_fp8")
        token_stride = key["bytes"] // 130
        for byte_index, expected in ((0, "prior_cache_state_difference"),
                                     (129 * token_stride, "kv_append_codec_difference")):
            with self.subTest(byte_index=byte_index), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                ordinary = self.fixture(directory, "ordinary", "ordinary")
                dflash = self.fixture(directory, "dflash", "dflash",
                                      ("key_fp8", byte_index, 1))
                self.assertEqual(analyze(ordinary, dflash)["classification"], expected)

    def test_rejects_schema_provenance_layout_and_nonfinite(self):
        mutations = (
            ("schema", lambda value: value["call"].update({"width": 4}), "call width"),
            ("provenance", lambda value: value["provenance"].update({"architecture": "gfx1200"}), "provenance"),
            ("layout", lambda value: value["fields"][0].update({"offset": 4}), "field layout"),
        )
        for name, mutation, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                ordinary = self.fixture(directory, "ordinary", "ordinary")
                dflash = self.fixture(directory, "dflash", "dflash")
                value = json.loads(dflash.read_text())
                mutation(value)
                dflash.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError, message):
                    analyze(ordinary, dflash)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            ordinary = self.fixture(directory, "ordinary", "ordinary")
            dflash = self.fixture(directory, "dflash", "dflash")
            sidecar = dflash.with_suffix(".bin")
            data = bytearray(sidecar.read_bytes())
            field = FIELD_BY_NAME["attention_fp32"]
            struct.pack_into("<I", data, field["offset"], 0x7f800000)
            sidecar.write_bytes(data)
            value = json.loads(dflash.read_text())
            value["sidecar_fnv1a64"] = fnv1a64(data)
            dflash.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                analyze(ordinary, dflash)

    def test_rejects_malformed_visibility_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            ordinary = self.fixture(directory, "ordinary", "ordinary")
            dflash = self.fixture(directory, "dflash", "dflash")
            value = json.loads(dflash.read_text())
            value["visibility"]["ancestor_masks"] = [1]
            dflash.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "ancestor"):
                analyze(ordinary, dflash)
            value["visibility"]["ancestor_masks"] = [32, 3, 7, 15, 31]
            dflash.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "ancestor"):
                analyze(ordinary, dflash)
            dflash.write_text('{"role": 1, "role": 2}')
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                analyze(ordinary, dflash)


if __name__ == "__main__":
    unittest.main()
