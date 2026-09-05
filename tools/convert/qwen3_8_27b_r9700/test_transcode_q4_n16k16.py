from __future__ import annotations

import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.artifact.container import Artifact, MAGIC, PAYLOAD_ALIGNMENT, PREFIX
from tools.artifact.layouts import align_up
from tools.convert.qwen3.common.inventory import ResourceSpec, TensorSpec

from .transcode_q4_n16k16 import transcode


class Q4N16K16TranscodeTest(unittest.TestCase):
    def test_exact_offline_migration_and_non_q4_copy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "legacy.ninfer"; output = root / "n16.ninfer"
            n, k, groups = 16, 128, 2
            old_codes = bytes((index * 37 + 11) & 0xFF for index in range(n * k // 2))
            old_scales = b"".join(struct.pack("<H", 0x3800 + index) for index in range(n * groups))
            q4 = old_codes + old_scales
            directory = {"identity": {"model_id": "qwen3.8-27b",
                                      "weights_id": "r9700-q4g64-f8e4m3-four-role-eval"},
                         "objects": [
                             {"name": "weight", "kind": "tensor", "shape": [n, k],
                              "format": "Q4G64_F16S", "layout": "row-split-k128-v1",
                              "offset": 0, "bytes": len(q4)},
                             {"name": "frontend/tokenizer.json", "kind": "resource",
                              "encoding": "raw-bytes-v1", "offset": len(q4), "bytes": 2}]}
            encoded = json.dumps(directory, separators=(",", ":")).encode()
            payload_offset = align_up(PREFIX.size + len(encoded), PAYLOAD_ALIGNMENT)
            with source.open("wb") as stream:
                stream.write(PREFIX.pack(MAGIC, len(encoded))); stream.write(encoded)
                stream.write(bytes(payload_offset - PREFIX.size - len(encoded)))
                stream.write(q4); stream.write(b"{}")
            exact_inventory = (
                TensorSpec("weight", (n, k), "Q4G64_F16S",
                           "r9700-q4g64-n16-k16-v1"),
                ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1"),
            )
            with patch.object(transcode.__globals__["fp8_hybrid_inventory"],
                              "OBJECT_SPECS", exact_inventory):
                report = transcode(source, output)
            self.assertEqual(report["q4_objects"], 1)
            self.assertEqual(report["source_identity"]["weights_id"],
                             "r9700-q4g64-f8e4m3-four-role-eval")
            self.assertEqual(report["identity"]["weights_id"],
                             "r9700-q4g64-f8e4m3-four-role-n16k16-eval")
            self.assertEqual(source.read_bytes()[payload_offset + len(q4):
                                                 payload_offset + len(q4) + 2], b"{}")
            with Artifact.open(output) as artifact:
                self.assertEqual(artifact.identity.weights_id,
                                 "r9700-q4g64-f8e4m3-four-role-n16k16-eval")
                self.assertEqual(artifact.find("weight").layout,
                                 "r9700-q4g64-n16-k16-v1")
                self.assertEqual(bytes(artifact.payload("frontend/tokenizer.json")), b"{}")

    def test_shifted_or_extended_descriptor_never_publishes_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary); n, k = 16, 128
            q4 = bytes(n * k // 2 + n * (k // 64) * 2)
            exact_inventory = (
                TensorSpec("weight", (n, k), "Q4G64_F16S",
                           "r9700-q4g64-n16-k16-v1"),
                ResourceSpec("trailing", "raw-bytes-v1"),
            )
            base = {"identity": {"model_id": "qwen3.8-27b",
                                 "weights_id": "r9700-q4g64-f8e4m3-four-role-eval"},
                    "objects": [{"name": "weight", "kind": "tensor", "shape": [n, k],
                                 "format": "Q4G64_F16S", "layout": "row-split-k128-v1",
                                 "offset": 0, "bytes": len(q4)},
                                {"name": "trailing", "kind": "resource",
                                 "encoding": "raw-bytes-v1", "offset": len(q4), "bytes": 2}]}
            variants = []
            shifted = json.loads(json.dumps(base)); shifted["objects"][1]["offset"] += 1
            variants.append(shifted)
            extended = json.loads(json.dumps(base)); extended["objects"][0]["extra"] = True
            variants.append(extended)
            with patch.object(transcode.__globals__["fp8_hybrid_inventory"],
                              "OBJECT_SPECS", exact_inventory):
                for index, directory in enumerate(variants):
                    source = root / f"legacy-{index}.ninfer"; output = root / f"n16-{index}.ninfer"
                    encoded = json.dumps(directory, separators=(",", ":")).encode()
                    payload_offset = align_up(PREFIX.size + len(encoded), PAYLOAD_ALIGNMENT)
                    with source.open("wb") as stream:
                        stream.write(PREFIX.pack(MAGIC, len(encoded))); stream.write(encoded)
                        stream.write(bytes(payload_offset - PREFIX.size - len(encoded)))
                        stream.write(q4); stream.write(b"{}")
                    with self.assertRaisesRegex(ValueError, "differs|unexpected"):
                        transcode(source, output)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
