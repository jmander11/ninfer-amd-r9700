from __future__ import annotations

import json
from pathlib import Path
import struct
import hashlib
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.artifact.container import Artifact, MAGIC, PAYLOAD_ALIGNMENT, PREFIX
from tools.artifact.layouts import align_up
from tools.convert.qwen3.common.inventory import ResourceSpec, TensorSpec
from tools.ppl.run import _validate_hybrid_migration_ancestry

from . import (fp8_hybrid_inventory, q4_inventory, q4_w8_mse_inventory,
               transcode_q4_n16k16 as transcode_module)
from .transcode_q4_n16k16 import (MigrationProfile, PROFILES_BY_SOURCE,
                                  _validate_migration_pair, preflight,
                                  publish_hybrid_migration_receipt, transcode)


class Q4N16K16TranscodeTest(unittest.TestCase):
    def write_source(
        self, path: Path, source_weights_id: str,
        *, trailing: bytes = b"{}", extra_payload: bytes = b"",
    ) -> tuple[tuple[object, ...], int, bytes]:
        n, k, groups = 16, 128, 2
        old_codes = bytes((index * 37 + 11) & 0xFF for index in range(n * k // 2))
        old_scales = b"".join(struct.pack("<H", 0x3800 + index)
                              for index in range(n * groups))
        q4 = old_codes + old_scales
        directory = {"identity": {"model_id": "qwen3.8-27b",
                                  "weights_id": source_weights_id},
                     "objects": [
                         {"name": "weight", "kind": "tensor", "shape": [n, k],
                          "format": "Q4G64_F16S", "layout": "row-split-k128-v1",
                          "offset": 0, "bytes": len(q4)},
                         {"name": "frontend/tokenizer.json", "kind": "resource",
                          "encoding": "raw-bytes-v1", "offset": len(q4),
                          "bytes": len(trailing)}]}
        encoded = json.dumps(directory, separators=(",", ":")).encode()
        payload_offset = align_up(PREFIX.size + len(encoded), PAYLOAD_ALIGNMENT)
        with path.open("wb") as stream:
            stream.write(PREFIX.pack(MAGIC, len(encoded))); stream.write(encoded)
            stream.write(bytes(payload_offset - PREFIX.size - len(encoded)))
            stream.write(q4); stream.write(trailing); stream.write(extra_payload)
        inventory = (
            TensorSpec("weight", (n, k), "Q4G64_F16S",
                       "r9700-q4g64-n16-k16-v1"),
            ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1"),
        )
        return inventory, payload_offset, q4

    def test_closed_three_profile_allowlist(self) -> None:
        expected = {
            "r9700-q4g64-eval":
                ("r9700-q4g64-n16k16-eval", q4_inventory.OBJECT_SPECS),
            "r9700-q4-w8-mse-eval":
                ("r9700-q4-w8-mse-n16k16-eval", q4_w8_mse_inventory.OBJECT_SPECS),
            "r9700-q4g64-f8e4m3-four-role-eval":
                ("r9700-q4g64-f8e4m3-four-role-n16k16-eval",
                 fp8_hybrid_inventory.OBJECT_SPECS),
        }
        self.assertEqual(set(PROFILES_BY_SOURCE), set(expected))
        for source, (output, inventory) in expected.items():
            self.assertEqual(PROFILES_BY_SOURCE[source].output_weights_id, output)
            self.assertIs(PROFILES_BY_SOURCE[source].object_specs, inventory)

    def test_exact_offline_migration_and_non_q4_copy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "legacy.ninfer"; output = root / "n16.ninfer"
            exact_inventory, payload_offset, q4 = self.write_source(
                source, "r9700-q4g64-f8e4m3-four-role-eval")
            profile = MigrationProfile(
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval", exact_inventory)
            with patch.dict(PROFILES_BY_SOURCE, {
                    "r9700-q4g64-f8e4m3-four-role-eval": profile}, clear=True):
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

    def test_preflight_maps_each_identity_without_publishing(self) -> None:
        mappings = {
            "r9700-q4g64-eval": "r9700-q4g64-n16k16-eval",
            "r9700-q4-w8-mse-eval": "r9700-q4-w8-mse-n16k16-eval",
            "r9700-q4g64-f8e4m3-four-role-eval":
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (source_id, output_id) in enumerate(mappings.items()):
                source, output = root / f"source-{index}.ninfer", root / f"out-{index}.ninfer"
                inventory, _, _ = self.write_source(source, source_id)
                with patch.dict(PROFILES_BY_SOURCE, {
                        source_id: MigrationProfile(output_id, inventory)}, clear=True):
                    report = preflight(source, output)
                self.assertEqual(report["mode"], "preflight-only")
                self.assertEqual(report["identity"]["weights_id"], output_id)
                self.assertEqual(report["q4_objects"], 1)
                self.assertFalse(output.exists())

    def test_preflight_rejects_unlisted_identity_and_payload_extent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "unknown.ninfer"
            self.write_source(source, "r9700-q4g64-n16k16-eval")
            with self.assertRaisesRegex(ValueError, "allowed legacy"):
                preflight(source)
            source = root / "trailing.ninfer"
            inventory, _, _ = self.write_source(
                source, "r9700-q4g64-eval", extra_payload=b"x")
            with patch.dict(PROFILES_BY_SOURCE, {
                    "r9700-q4g64-eval": MigrationProfile(
                        "r9700-q4g64-n16k16-eval", inventory)}, clear=True):
                with self.assertRaisesRegex(ValueError, "payload extent"):
                    preflight(source)

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
            profile = MigrationProfile(
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval", exact_inventory)
            with patch.dict(PROFILES_BY_SOURCE, {
                    "r9700-q4g64-f8e4m3-four-role-eval": profile}, clear=True):
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

    def test_receipt_only_publishes_distinct_migration_authority_create_only(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "legacy.ninfer", root / "n16.ninfer"
            source.write_bytes(b"legacy"); output.write_bytes(b"n16")
            old_plan, new_plan = "1" * 64, "2" * 64
            validation = {
                "source": str(source.resolve()), "output": str(output.resolve()),
                "source_identity": {"model_id": "qwen3.8-27b", "weights_id":
                                    "r9700-q4g64-f8e4m3-four-role-eval"},
                "identity": {"model_id": "qwen3.8-27b", "weights_id":
                             "r9700-q4g64-f8e4m3-four-role-n16k16-eval"},
                "source_bytes": 6, "output_bytes": 3,
                "source_sha256": hashlib.sha256(b"legacy").hexdigest(),
                "output_sha256": hashlib.sha256(b"n16").hexdigest(),
                "objects": 1124, "q4_objects": 295,
                "source_object_plan_sha256": old_plan,
                "object_plan_sha256": new_plan,
                "source_file_identity": (source.stat().st_dev, source.stat().st_ino,
                                         source.stat().st_uid),
                "output_file_identity": (output.stat().st_dev, output.stat().st_ino,
                                         output.stat().st_uid),
            }
            upstream = {
                "identity": validation["source_identity"],
                "target_key": fp8_hybrid_inventory.TARGET_KEY,
                "recipe_id": "r9700-q4g64-f8e4m3-four-role-eval-v0",
                "source": {"index_sha256": "3" * 64, "ranking_sha256": "4" * 64},
                "artifact": {"path": str(source.resolve()), "bytes": 6,
                             "sha256": validation["source_sha256"]},
                "candidate": {
                    "status": "registered-evaluation-only", "weight_recipe_selected": False,
                    "selection_sha256": fp8_hybrid_inventory.SELECTION_SHA256,
                    "format_counts": fp8_hybrid_inventory.FORMAT_COUNTS,
                    "format_encoded_bytes": fp8_hybrid_inventory.FORMAT_ENCODED_BYTES,
                    "tensor_encoded_bytes": fp8_hybrid_inventory.TENSOR_ENCODED_BYTES,
                    "device_arena_bytes": fp8_hybrid_inventory.DEVICE_ARENA_BYTES,
                    "object_plan_sha256": old_plan,
                },
            }
            Path(str(source) + ".conversion.json").write_text(json.dumps(upstream))
            with patch.object(transcode_module, "_validate_migration_pair",
                              return_value=validation):
                result = publish_hybrid_migration_receipt(source, output)
                receipt_path = Path(result["receipt"])
                receipt = json.loads(receipt_path.read_text())
                self.assertEqual(receipt["identity"], validation["identity"])
                self.assertEqual(receipt["candidate"]["object_plan_sha256"], new_plan)
                self.assertEqual(receipt["migration"]["source_artifact"]["sha256"],
                                 validation["source_sha256"])
                self.assertEqual(receipt["migration"]["storage_transform"], {
                    "from": "row-split-k128-v1", "to": "r9700-q4g64-n16-k16-v1"})
                _validate_hybrid_migration_ancestry(receipt)
                receipt["migration"]["transcoder"]["sha256"] = "5" * 64
                with self.assertRaisesRegex(SystemExit, "ancestry differs"):
                    _validate_hybrid_migration_ancestry(receipt)
                receipt["migration"]["transcoder"]["sha256"] = hashlib.sha256(
                    Path(receipt["migration"]["transcoder"]["path"]).read_bytes()
                ).hexdigest()
                original_transcoder = receipt["migration"]["transcoder"]["path"]
                receipt["migration"]["transcoder"]["path"] = str(source)
                with self.assertRaisesRegex(SystemExit, "ancestry differs"):
                    _validate_hybrid_migration_ancestry(receipt)
                receipt["migration"]["transcoder"]["path"] = original_transcoder
                source.write_bytes(b"LEGACY")
                with self.assertRaisesRegex(SystemExit, "ancestry differs"):
                    _validate_hybrid_migration_ancestry(receipt)
                source.write_bytes(b"legacy")
                upstream_path = Path(receipt["migration"]["source_conversion_receipt"]["path"])
                upstream_bytes = upstream_path.read_bytes()
                upstream_copy = root / "upstream-copy.json"
                upstream_copy.write_bytes(upstream_bytes)
                upstream_path.unlink(); upstream_path.symlink_to(upstream_copy)
                with self.assertRaisesRegex(SystemExit, "regular files"):
                    _validate_hybrid_migration_ancestry(receipt)
                upstream_path.unlink(); upstream_path.write_bytes(upstream_bytes)
                receipt["migration"]["q4_objects"] = 294
                with self.assertRaisesRegex(SystemExit, "ancestry differs"):
                    _validate_hybrid_migration_ancestry(receipt)
                with self.assertRaisesRegex(ValueError, "must not exist"):
                    publish_hybrid_migration_receipt(source, output)

    def test_pair_validation_rejects_alternate_output_offsets(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "legacy.ninfer"; output = root / "n16.ninfer"
            inventory, _, _ = self.write_source(
                source, "r9700-q4g64-f8e4m3-four-role-eval")
            profile = MigrationProfile(
                "r9700-q4g64-f8e4m3-four-role-n16k16-eval", inventory)
            with patch.dict(PROFILES_BY_SOURCE, {
                    "r9700-q4g64-f8e4m3-four-role-eval": profile}, clear=True):
                transcode(source, output)
                with output.open("r+b") as stream:
                    magic, size = PREFIX.unpack(stream.read(PREFIX.size))
                    directory = json.loads(stream.read(size))
                    directory["objects"][1]["offset"] += 1
                    encoded = json.dumps(directory, separators=(",", ":")).encode()
                    self.assertEqual(len(encoded), size)
                    stream.seek(PREFIX.size); stream.write(encoded)
                    stream.seek(0, 2); stream.write(b"\x00")
                with self.assertRaisesRegex(ValueError, "exact N16 object plan"):
                    _validate_migration_pair(source, output)

    def test_receipt_only_rejects_post_validation_path_replacement(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "legacy"; output = root / "output"
            source.write_bytes(b"legacy"); output.write_bytes(b"n16")
            validation = {
                "source": str(source.resolve()), "output": str(output.resolve()),
                "identity": {"model_id": "qwen3.8-27b", "weights_id":
                             "r9700-q4g64-f8e4m3-four-role-n16k16-eval"},
                "source_file_identity": (source.stat().st_dev, source.stat().st_ino,
                                         source.stat().st_uid),
                "output_file_identity": (output.stat().st_dev, output.stat().st_ino,
                                         output.stat().st_uid),
            }
            replacement = root / "replacement"; replacement.write_bytes(b"new")
            def replace_after_validation(*_args):
                output.unlink(); replacement.rename(output)
                return validation
            with patch.object(transcode_module, "_validate_migration_pair",
                              side_effect=replace_after_validation), self.assertRaisesRegex(
                                  RuntimeError, "identity changed"):
                publish_hybrid_migration_receipt(source, output)
            self.assertFalse(Path(str(output) + ".conversion.json").exists())

    def test_receipt_rolls_back_on_late_authority_mutation(self) -> None:
        for mutation in ("source", "output", "upstream", "transcoder",
                         "earlier-owner-during-later-hash", "receipt"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / "legacy", root / "output"
                source.write_bytes(b"legacy"); output.write_bytes(b"n16")
                fake_transcoder = root / "transcoder.py"; fake_transcoder.write_bytes(b"tool")
                validation = {
                    "source": str(source.resolve()), "output": str(output.resolve()),
                    "source_identity": {"model_id": "qwen3.8-27b", "weights_id":
                                        "r9700-q4g64-f8e4m3-four-role-eval"},
                    "identity": {"model_id": "qwen3.8-27b", "weights_id":
                                 "r9700-q4g64-f8e4m3-four-role-n16k16-eval"},
                    "source_bytes": 6, "output_bytes": 3,
                    "source_sha256": hashlib.sha256(b"legacy").hexdigest(),
                    "output_sha256": hashlib.sha256(b"n16").hexdigest(),
                    "objects": 1124, "q4_objects": 295,
                    "source_object_plan_sha256": "1" * 64,
                    "object_plan_sha256": "2" * 64,
                    "source_file_identity": (source.stat().st_dev, source.stat().st_ino,
                                             source.stat().st_uid),
                    "output_file_identity": (output.stat().st_dev, output.stat().st_ino,
                                             output.stat().st_uid),
                }
                upstream = {
                    "identity": validation["source_identity"],
                    "target_key": fp8_hybrid_inventory.TARGET_KEY,
                    "recipe_id": "r9700-q4g64-f8e4m3-four-role-eval-v0",
                    "source": {"index_sha256": "3" * 64, "ranking_sha256": "4" * 64},
                    "artifact": {"path": str(source.resolve()), "bytes": 6,
                                 "sha256": validation["source_sha256"]},
                    "candidate": {
                        "status": "registered-evaluation-only",
                        "weight_recipe_selected": False,
                        "selection_sha256": fp8_hybrid_inventory.SELECTION_SHA256,
                        "format_counts": fp8_hybrid_inventory.FORMAT_COUNTS,
                        "format_encoded_bytes": fp8_hybrid_inventory.FORMAT_ENCODED_BYTES,
                        "tensor_encoded_bytes": fp8_hybrid_inventory.TENSOR_ENCODED_BYTES,
                        "device_arena_bytes": fp8_hybrid_inventory.DEVICE_ARENA_BYTES,
                        "object_plan_sha256": validation["source_object_plan_sha256"],
                    },
                }
                upstream_path = Path(str(source) + ".conversion.json")
                upstream_path.write_text(json.dumps(upstream))
                publish = transcode_module._publish_json_create_only
                hash_open_fd = transcode_module._hash_open_fd
                hash_calls = 0
                def hash_with_late_earlier_mutation(descriptor):
                    nonlocal hash_calls
                    result = hash_open_fd(descriptor)
                    hash_calls += 1
                    if mutation == "earlier-owner-during-later-hash" and hash_calls == 6:
                        source.write_bytes(b"LEGACY")
                    return result
                def mutate_then_publish(path, payload):
                    if mutation == "source": source.write_bytes(b"LEGACY")
                    elif mutation == "output": output.write_bytes(b"N16")
                    elif mutation == "upstream": upstream_path.write_bytes(
                        upstream_path.read_bytes() + b" ")
                    elif mutation == "transcoder": fake_transcoder.write_bytes(b"TOOL")
                    result = publish(path, payload)
                    if mutation == "receipt":
                        path.unlink(); path.write_bytes(b"foreign")
                    return result
                with patch.object(transcode_module, "_validate_migration_pair",
                                  return_value=validation), patch.object(
                                      transcode_module, "_publish_json_create_only",
                                      side_effect=mutate_then_publish), patch.object(
                                          transcode_module, "_hash_open_fd",
                                          side_effect=hash_with_late_earlier_mutation), patch.object(
                                              transcode_module, "__file__", str(fake_transcoder)), \
                        self.assertRaisesRegex(
                            RuntimeError, "authority changed|identity changed|receipt changed"):
                    publish_hybrid_migration_receipt(source, output)
                receipt_path = Path(str(output) + ".conversion.json")
                if mutation == "receipt":
                    self.assertEqual(receipt_path.read_bytes(), b"foreign")
                else:
                    self.assertFalse(receipt_path.exists())


if __name__ == "__main__":
    unittest.main()
