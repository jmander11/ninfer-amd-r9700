from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import preflight_identity


class ConverterPreflightIdentityTest(unittest.TestCase):
    def test_source_checkpoint_binds_exact_index_and_eighteen_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            mapping = {f"tensor.{index}": preflight_identity.SOURCE_SHARDS[index % 18]
                       for index in range(preflight_identity.SOURCE_TENSORS)}
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": preflight_identity.bf16_protocol.SOURCE_TOTAL_BYTES},
                            "weight_map": mapping}))
            for index, name in enumerate(preflight_identity.SOURCE_SHARDS, 1):
                (root / name).write_bytes(bytes([index]))
            with patch.object(preflight_identity.bf16_protocol, "validate_checkpoint_files",
                              return_value=mapping):
                value = preflight_identity.source_checkpoint(root)
            self.assertEqual(value["indexed_tensor_count"], 1199)
            self.assertEqual([row["name"] for row in value["shards"]],
                             list(preflight_identity.SOURCE_SHARDS))
            self.assertEqual(value["shard_total_file_bytes"], 18)
            self.assertEqual(value["config"]["sha256"], hashlib.sha256(b"{}").hexdigest())

    def test_resources_and_object_plan_are_ordered_and_hashed(self) -> None:
        resources = tuple(SimpleNamespace(name=f"frontend/{index}", data=bytes([index]))
                          for index in range(6))
        rows = preflight_identity.frontend_resources(resources)
        self.assertEqual([row["name"] for row in rows], [f"frontend/{i}" for i in range(6)])
        objects = [SimpleNamespace(to_json=lambda: {"name": "a"})]
        self.assertEqual(preflight_identity.object_plan_sha256(objects), hashlib.sha256(
            b'[{"name":"a"}]').hexdigest())

    def test_rejects_incomplete_source_and_resource_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "config.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
            with patch.object(preflight_identity.bf16_protocol, "validate_checkpoint_files",
                              return_value={}), self.assertRaisesRegex(ValueError, "tensor byte"):
                preflight_identity.source_checkpoint(root)
        with self.assertRaisesRegex(ValueError, "six"):
            preflight_identity.frontend_resources([])


if __name__ == "__main__":
    unittest.main()
