"""Exact inventory/copy boundaries and independent represented-source codec checks."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec as StoredTensor
from tools.convert.qwen3.common.inventory import BF16, W8, TensorSpec, tensor_spec, ResourceSpec as InventoryResource
from . import codec, q4_inventory, selective_protected_inventory as inventory
from . import convert_selective_protected as conversion


class SelectiveProtectedTest(unittest.TestCase):
    def test_fixed_inventory_and_unchanged_draft_vision_mtp(self):
        inventory.validate_inventory()
        changed = {new.name for old, new in zip(q4_inventory.TENSOR_SPECS, inventory.TENSOR_SPECS)
                   if old != new}
        self.assertEqual(changed, set(inventory.CHANGED_FORMATS))
        self.assertEqual(len(changed), 28)
        self.assertEqual(inventory.FORMAT_COUNTS, {W8: 2, BF16: 597, "FP32": 96,
            "Q4G64_F16S": 411, "F8E4M3_ROW_F32S": 11, "I32": 1})
        self.assertEqual(inventory.TENSOR_ENCODED_BYTES, 17_665_263_520)
        self.assertEqual(inventory.DEVICE_ARENA_BYTES, 17_665_277_440)
        base = {s.name: s for s in q4_inventory.TENSOR_SPECS}
        for spec in inventory.TENSOR_SPECS:
            if spec.name not in changed:
                self.assertEqual(spec, base[spec.name])

    def test_changed_encoders_match_independent_scalar_oracles(self):
        initialized = torch.cuda.is_initialized()
        # W8 crosses the streaming row boundary and exercises K padding.
        for fmt, rows, columns in ((W8, 257, 130), (W8, 2, 5120),
                                  (inventory.F8E4M3_ROW_F32S, 2, 6144),
                                  (inventory.F8E4M3_ROW_F32S, 2, 17408), (BF16, 3, 128)):
            with self.subTest(format=fmt, rows=rows, columns=columns):
                tensor = torch.tensor([((i * 37) % 257 - 128) / 19
                                       for i in range(rows * columns)], dtype=torch.bfloat16).reshape(rows, columns)
                spec = (TensorSpec("matrix", (rows, columns), fmt, inventory.ROW_SCALED_LAYOUT)
                        if fmt == inventory.F8E4M3_ROW_F32S else tensor_spec("matrix", (rows, columns), fmt))
                represented = tensor.float().flatten().tolist()
                payload = b"".join(conversion.encode_changed(tensor, spec))
                if fmt == W8:
                    expected = codec.encode_w8g32_reference(represented, rows, columns)
                    codes, scales = codec.decode_w8g32_reference(payload, rows, columns)
                    self.assertEqual(len(codes), rows * ((columns + 127) // 128 * 128))
                    self.assertTrue(all(struct.unpack("<e", struct.pack("<H", scale))[0] >= 0 for scale in scales))
                elif fmt == BF16:
                    expected = b"".join(struct.pack("<I", struct.unpack("<I", struct.pack("<f", v))[0])[2:]
                                        for v in represented)
                else:
                    expected = codec.encode_e4m3_rowwise_reference(represented, rows, columns)
                    decoded, _, _ = codec.decode_e4m3_rowwise_reference(payload, rows, columns)
                    squared = sum((a - b) ** 2 for a, b in zip(represented, decoded))
                    self.assertLess(squared / sum(v*v for v in represented), 0.002)
                self.assertEqual(payload, expected)
        self.assertEqual(torch.cuda.is_initialized(), initialized)

    def test_rejects_nonrepresented_or_nonfinite_inputs(self):
        spec = tensor_spec("matrix", (2, 128), BF16)
        with self.assertRaises(ValueError):
            list(conversion.encode_changed(torch.zeros((2, 128)), spec))
        with self.assertRaises(ValueError):
            list(conversion.encode_changed(torch.full((2, 128), float("nan"), dtype=torch.bfloat16), spec))

    def test_real_container_copy_and_replacement_are_byte_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, output = root / "base.ninfer", root / "output.ninfer"
            names = ("text/draft_head", "text/draft_head_token_ids", "vision/fixed", "mtp/fixed")
            specs = (ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 7),
                     *(StoredTensor(name, (4,), "I32", "contiguous-le-v1") for name in names),
                     StoredTensor("text/token_embedding", (2, 128), BF16, "contiguous-le-v1"))
            payloads = [b"tokeniz", *(struct.pack("<4i", i, 3, 2, 1) for i in range(4)), bytes(512)]
            with ArtifactWriter(base, ArtifactIdentity(inventory.MODEL_ID, q4_inventory.WEIGHTS_ID), specs) as writer:
                for spec, payload in zip(specs, payloads):
                    writer.write(spec.name, payload)
            target = tensor_spec("text/token_embedding", (2, 128), W8)
            tensor = torch.arange(256, dtype=torch.bfloat16).reshape(2, 128)
            output_specs = specs[:-1] + (StoredTensor(target.name, target.shape, target.format, target.layout),)
            with patch.object(inventory, "TENSOR_SPECS", (target,)), patch.object(
                    conversion.source_recipe, "materialize_recipe", return_value=tensor) as materialize:
                with Artifact(base) as artifact, ArtifactWriter(output, ArtifactIdentity(
                        inventory.MODEL_ID, inventory.WEIGHTS_ID), output_specs) as writer:
                    records = conversion.write_payloads(artifact, writer, None)
            self.assertEqual(materialize.call_count, 1)
            with Artifact(output) as artifact:
                for obj, payload, record in zip(artifact.objects[:-1], payloads[:-1], records[:-1]):
                    self.assertEqual(bytes(artifact.payload(obj)), payload)
                    self.assertEqual(record["operation"], "copy-exact")
                    self.assertEqual(record["payload_sha256"], hashlib.sha256(payload).hexdigest())
                actual = bytes(artifact.payload("text/token_embedding"))
                self.assertEqual(actual, codec.encode_w8g32_reference(tensor.float().flatten().tolist(), 2, 128))
            self.assertEqual(records[-1]["operation"], "source-bf16-reencode")
            with self.assertRaises(FileExistsError):
                conversion.preflight(base, root, output)
            with Artifact(output) as artifact:
                self.assertEqual(bytes(artifact.payload("text/token_embedding")), actual)
            original_specs = (InventoryResource(specs[0].name),) + tuple(
                TensorSpec(s.name, s.shape, s.format, s.layout) for s in specs[1:])
            changed_specs = original_specs[:-1] + (target,)
            report = {
                "artifact_type": "ninfer_r9700_selective_protected_conversion", "schema_version": 1,
                "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
                "recipe_id": inventory.RECIPE_ID, "weight_recipe_selected": False,
                "changed_formats": {target.name: W8},
                "format_counts": inventory.FORMAT_COUNTS,
                "tensor_encoded_bytes": inventory.TENSOR_ENCODED_BYTES,
                "device_arena_bytes": inventory.DEVICE_ARENA_BYTES,
                "projected_file_bytes": output.stat().st_size,
                "source": {"model_path": str(root.resolve()), "checkpoint": {}},
                "base": {"path": str(base.resolve()), "bytes": base.stat().st_size,
                         "sha256": conversion.sha(base)},
                "artifact": {"path": str(output.resolve()), "bytes": output.stat().st_size,
                             "sha256": conversion.sha(output)},
                "objects": records,
            }
            receipt = Path(str(output) + ".conversion.json")
            receipt.write_text(json.dumps(report))
            with patch.object(inventory, "OBJECT_SPECS", changed_specs), \
                    patch.object(q4_inventory, "OBJECT_SPECS", original_specs), \
                    patch.object(inventory, "CHANGED_FORMATS", {target.name: W8}), \
                    patch.object(conversion, "_checkpoint_receipt", return_value={}):
                conversion.validate(output, base, root)
                with Artifact(output) as artifact:
                    obj = artifact.find("text/draft_head_token_ids")
                    offset = artifact.payload_offset + obj.offset
                # Even a self-consistent output hash/receipt cannot relabel a
                # corrupted draft map as an exact copy of the retained base.
                with output.open("r+b") as stream:
                    stream.seek(offset)
                    stream.write(struct.pack("<i", 42))
                with Artifact(output) as artifact:
                    report["objects"][2]["payload_sha256"] = hashlib.sha256(
                        artifact.payload("text/draft_head_token_ids")).hexdigest()
                report["artifact"]["sha256"] = conversion.sha(output)
                receipt.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "not byte-exact"):
                    conversion.validate(output, base, root)


if __name__ == "__main__":
    unittest.main()
