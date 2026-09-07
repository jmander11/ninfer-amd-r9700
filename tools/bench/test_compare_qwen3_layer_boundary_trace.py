import json
import struct
import tempfile
import unittest
from pathlib import Path

from compare_qwen3_layer_boundary_trace import (
    GDN_FIELDS, GDN_PAYLOAD_BYTES, PAYLOAD_BYTES, RECURRENT_STATE_BYTES,
    RECURRENT_STATE_ELEMENTS, ROLES, compare, compare_gdn, compare_recurrent_state, fnv1a64,
)


class CompareTest(unittest.TestCase):
    def fixture(self, directory: Path, expected, mutate=None):
        role, width, column, frontier, position = expected
        values = [0] * (PAYLOAD_BYTES // 2)
        if mutate is not None:
            snapshot, hidden, bits = mutate
            values[snapshot * 5120 + hidden] = bits
        data = struct.pack(f"<{len(values)}H", *values)
        sidecar = directory / f"{role}.bin"
        sidecar.write_bytes(data)
        manifest = directory / f"{role}.json"
        manifest.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_layer_boundary_trace",
            "schema_version": 1, "diagnostic_only": True,
            "timing_evidence_eligible": False, "production_routing_authorized": False,
            "execution": "eager", "role": role, "width": width,
            "selected_column": column, "absolute_frontier": frontier, "token": 42,
            "cache_position": position, "rope_position": position, "hidden": 5120,
            "layers": 64, "snapshot_count": 129, "snapshot_bytes": 10240,
            "sidecar_path": str(sidecar), "sidecar_bytes": PAYLOAD_BYTES,
            "sidecar_fnv1a64": fnv1a64(data),
            "layout": "little-endian-u16: input, then layer0..63 post_mixer,post_mlp",
        }))
        return manifest

    def gdn_fixture(self, directory: Path, expected, suffix: str, mutate=None):
        role, width, column, frontier, position = expected
        data = bytearray(GDN_PAYLOAD_BYTES)
        if mutate is not None:
            field_name, element, bits = mutate
            field = next(value for value in GDN_FIELDS if value[0] == field_name)
            _, dtype, _, offset, _ = field
            struct.pack_into("<H" if dtype == "bf16" else "<I", data,
                             offset + element * (2 if dtype == "bf16" else 4), bits)
        sidecar = directory / f"{suffix}.bin"
        sidecar.write_bytes(data)
        manifest = directory / f"{suffix}.json"
        manifest.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_layer1_gdn_detail_trace",
            "schema_version": 1, "diagnostic_only": True,
            "timing_evidence_eligible": False, "production_routing_authorized": False,
            "execution": "eager", "role": role, "width": width,
            "selected_column": column, "absolute_frontier": frontier, "token": 42,
            "cache_position": position, "rope_position": position, "text_layer": 1,
            "gdn_index": 1, "sidecar_path": str(sidecar),
            "sidecar_bytes": GDN_PAYLOAD_BYTES, "sidecar_fnv1a64": fnv1a64(data),
            "fields": [
                {"name": name, "dtype": dtype, "elements": elements, "offset": offset,
                 "bytes": byte_count}
                for name, dtype, elements, offset, byte_count in GDN_FIELDS
            ],
            "layout": "typed selected-column layer1 GDN boundaries in field order",
        }))
        return manifest

    def recurrent_state_fixture(self, directory: Path, role: str, suffix: str, mutate=None,
                                layer=1):
        data = bytearray(RECURRENT_STATE_BYTES)
        if mutate is not None:
            element, bits = mutate
            struct.pack_into("<I", data, element * 4, bits)
        sidecar = directory / f"{suffix}.bin"
        sidecar.write_bytes(data)
        manifest = directory / f"{suffix}.json"
        point = ("wide-prefill-state-after-prefix-128-before-selected-column-128"
                 if role == ROLES["text"][0][0] else
                 "restored-append-state-before-selected-column-0")
        manifest.write_text(json.dumps({
            "artifact_type": "ninfer_qwen3_gdn_recurrent_state_trace",
            "schema_version": 2, "diagnostic_only": True,
            "timing_evidence_eligible": False, "production_routing_authorized": False,
            "execution": "eager", "role": role, "capture_point": point,
            "selected_token": 24178, "selected_cache_position": 128,
            "selected_rope_position": 128, "state_frontier": 128,
            "linear_state_slot": 0, "text_layer": layer, "gdn_index": layer, "dtype": "fp32",
            "shape": [128, 128, 48], "elements": RECURRENT_STATE_ELEMENTS,
            "sidecar_path": str(sidecar), "sidecar_bytes": RECURRENT_STATE_BYTES,
            "sidecar_fnv1a64": fnv1a64(data),
            "layout": "little-endian-fp32:key,value,value_head",
        }))
        return manifest

    def test_exact_and_mixer_localization(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.fixture(directory, ROLES["target"][0])
            right = self.fixture(directory, ROLES["target"][1])
            self.assertEqual(compare(left, right, "target")["classification"],
                             "transformer_residual_stack_exact")
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.fixture(directory, ROLES["target"][0])
            right = self.fixture(directory, ROLES["target"][1], (13, 17, 9))
            result = compare(left, right, "target")
            self.assertEqual(result["classification"], "first_visible_post_mixer_layer_6")
            self.assertEqual(result["first_difference"]["first_hidden_index"], 17)

    def test_input_and_mlp_localization(self):
        for snapshot, expected in ((0, "represented_input_differs"),
                                   (128, "first_visible_post_mlp_layer_63")):
            with self.subTest(snapshot=snapshot), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                left = self.fixture(directory, ROLES["text"][0])
                right = self.fixture(directory, ROLES["text"][1], (snapshot, 0, 1))
                self.assertEqual(compare(left, right, "text")["classification"], expected)

    def test_rejects_token_and_hash_mutations(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.fixture(directory, ROLES["target"][0])
            right = self.fixture(directory, ROLES["target"][1])
            value = json.loads(right.read_text())
            value["token"] = 248077
            right.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "token"):
                compare(left, right, "target")
            value["token"] = 42
            value["sidecar_fnv1a64"] = "0" * 16
            right.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "hash"):
                compare(left, right, "target")

    def test_rejects_duplicate_and_nonfinite_json(self):
        for payload, message in (("{\"role\":1,\"role\":2}", "duplicate"),
                                 ("{\"value\":NaN}", "nonfinite")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                left = self.fixture(directory, ROLES["target"][0])
                right = self.fixture(directory, ROLES["target"][1])
                right.write_text(payload)
                with self.assertRaisesRegex(RuntimeError, message):
                    compare(left, right, "target")

    def test_gdn_exact_and_each_stage_localization(self):
        cases = {
            "h": "first_difference_gdn_input_rmsnorm",
            "g": "first_difference_gdn_g_control",
            "beta": "first_difference_gdn_beta_control",
            "z": "first_difference_gdn_output_gate_projection",
            "q": "first_difference_gdn_query_projection_or_conv",
            "k": "first_difference_gdn_key_projection_or_conv",
            "v": "first_difference_gdn_value_projection_or_conv",
            "o": "first_difference_gdn_recurrence",
            "on": "first_difference_gdn_gated_rmsnorm",
            "x": "first_difference_gdn_output_projection_or_residual",
        }
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.gdn_fixture(directory, ROLES["target"][0], "left")
            right = self.gdn_fixture(directory, ROLES["target"][1], "right")
            self.assertEqual(compare_gdn(left, right, "target")["classification"],
                             "layer1_gdn_boundaries_exact")
        for field, expected_classification in cases.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                left = self.gdn_fixture(directory, ROLES["target"][0], "left")
                right = self.gdn_fixture(directory, ROLES["target"][1], "right",
                                         (field, 1, 0x3f80 if field not in {"g", "beta"}
                                          else 0x3f800000))
                result = compare_gdn(left, right, "target")
                self.assertEqual(result["classification"], expected_classification)
                self.assertEqual(result["first_difference"]["field"], field)
                self.assertEqual(result["first_difference"]["first_element_index"], 1)
                self.assertEqual(result["first_difference"]["right_value"], 1.0)
                self.assertEqual(result["first_difference"]["maximum_absolute_difference"], 1.0)

    def test_gdn_rejects_typed_layout_and_sidecar_identity_mutations(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.gdn_fixture(directory, ROLES["text"][0], "left")
            right = self.gdn_fixture(directory, ROLES["text"][1], "right")
            value = json.loads(right.read_text())
            value["fields"][4]["dtype"] = "fp32"
            right.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "field layout"):
                compare_gdn(left, right, "text")
            value["fields"][4]["dtype"] = "bf16"
            value["sidecar_path"] = str(directory / "left.bin")
            right.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "sidecar identity"):
                compare_gdn(left, right, "text")

    def test_gdn_rejects_nonfinite_bf16_and_fp32(self):
        for field, bits in (("q", 0x7f80), ("g", 0x7f800000)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                left = self.gdn_fixture(directory, ROLES["target"][0], "left")
                right = self.gdn_fixture(directory, ROLES["target"][1], "right",
                                         (field, 0, bits))
                with self.assertRaisesRegex(RuntimeError, "nonfinite represented"):
                    compare_gdn(left, right, "target")

    def test_recurrent_state_exact_and_first_difference(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.recurrent_state_fixture(directory, ROLES["text"][0][0], "left")
            right = self.recurrent_state_fixture(directory, ROLES["text"][1][0], "right")
            self.assertEqual(compare_recurrent_state(left, right)["classification"],
                             "layer1_recurrent_prefix_state_exact")
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.recurrent_state_fixture(directory, ROLES["text"][0][0], "left")
            right = self.recurrent_state_fixture(directory, ROLES["text"][1][0], "right",
                                                  (154, 0x3f800000))
            result = compare_recurrent_state(left, right)
            self.assertEqual(result["classification"],
                             "first_difference_layer1_recurrent_prefix_state")
            self.assertEqual(result["first_difference"]["first_element_index"], 154)
            self.assertEqual(result["first_difference"]["mismatch_count"], 1)
            self.assertEqual(result["first_difference"]["right_value"], 1.0)
            self.assertEqual(result["first_difference"]["maximum_absolute_difference"], 1.0)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.recurrent_state_fixture(directory, ROLES["text"][0][0], "left", layer=0)
            right = self.recurrent_state_fixture(directory, ROLES["text"][1][0], "right", layer=0)
            result = compare_recurrent_state(left, right)
            self.assertEqual(result["classification"], "layer0_recurrent_prefix_state_exact")
            self.assertEqual(result["text_layer"], 0)

    def test_recurrent_state_rejects_frontier_role_hash_and_nonfinite(self):
        mutations = (("state_frontier", 127, "field differs"),
                     ("linear_state_slot", 1, "field differs"),
                     ("role", ROLES["target"][1][0], "field differs"),
                     ("sidecar_path", "/tmp/not-the-matching-sidecar.bin", "identity"),
                     ("sidecar_fnv1a64", "0" * 16, "bytes/hash"))
        for key, value, message in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                left = self.recurrent_state_fixture(directory, ROLES["text"][0][0], "left")
                right = self.recurrent_state_fixture(directory, ROLES["text"][1][0], "right")
                manifest = json.loads(right.read_text())
                manifest[key] = value
                right.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, message):
                    compare_recurrent_state(left, right)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            left = self.recurrent_state_fixture(directory, ROLES["text"][0][0], "left")
            right = self.recurrent_state_fixture(directory, ROLES["text"][1][0], "right",
                                                  (0, 0x7f800000))
            with self.assertRaisesRegex(RuntimeError, "nonfinite recurrent-state"):
                compare_recurrent_state(left, right)


if __name__ == "__main__":
    unittest.main()
