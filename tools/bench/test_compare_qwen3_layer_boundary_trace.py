import json
import struct
import tempfile
import unittest
from pathlib import Path

from compare_qwen3_layer_boundary_trace import PAYLOAD_BYTES, ROLES, compare, fnv1a64


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


if __name__ == "__main__":
    unittest.main()
