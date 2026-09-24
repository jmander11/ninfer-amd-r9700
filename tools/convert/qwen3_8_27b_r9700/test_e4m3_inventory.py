"""CPU checks for the target-private FP8 inventory and gate/up fixture."""

from __future__ import annotations

import hashlib
import struct
import tempfile
from pathlib import Path
import unittest

from tools.artifact.container import Artifact, TensorObject
from tools.artifact.layouts import encoded_size
from tools.convert.qwen3.common.inventory import DIRECT_FORMATS

from . import codec, e4m3_inventory, source_inventory
from .e4m3_gate_up_fixture import (
    FIXTURE_IDENTITY,
    FIXTURE_SPEC,
    write_gate_up_fixture,
)


class E4M3InventoryTest(unittest.TestCase):
    def test_all_intended_matrix_objects_map_to_rowwise_e4m3(self) -> None:
        e4m3_inventory.validate_inventory()
        self.assertEqual(len(e4m3_inventory.TENSOR_SPECS), 1118)
        self.assertEqual(len(e4m3_inventory.OBJECT_SPECS), 1124)
        self.assertEqual(
            e4m3_inventory.RESOURCE_SPECS, source_inventory.RESOURCE_SPECS
        )
        for source, candidate in zip(
            source_inventory.TENSOR_SPECS,
            e4m3_inventory.TENSOR_SPECS,
            strict=True,
        ):
            self.assertEqual((candidate.name, candidate.shape), (source.name, source.shape))
            if source.format in DIRECT_FORMATS:
                self.assertEqual(candidate, source)
            else:
                self.assertEqual(candidate.format, e4m3_inventory.F8E4M3_ROW_F32S)
                self.assertEqual(candidate.layout, e4m3_inventory.ROW_SCALED_LAYOUT)
        self.assertEqual(
            e4m3_inventory.FORMAT_COUNTS,
            {"BF16": 582, "FP32": 96, "I32": 1, "F8E4M3_ROW_F32S": 439},
        )
        self.assertEqual(e4m3_inventory.TENSOR_ENCODED_BYTES, 28_503_368_096)
        self.assertEqual(e4m3_inventory.DEVICE_ARENA_BYTES, 28_503_382_016)

    def test_gate_up_fixture_spec_is_the_exact_inventory_object(self) -> None:
        inventory_spec = next(
            spec
            for spec in e4m3_inventory.TENSOR_SPECS
            if spec.name == FIXTURE_SPEC.name
        )
        self.assertEqual(
            (
                FIXTURE_SPEC.name,
                FIXTURE_SPEC.shape,
                FIXTURE_SPEC.format,
                FIXTURE_SPEC.layout,
            ),
            (
                inventory_spec.name,
                inventory_spec.shape,
                inventory_spec.format,
                inventory_spec.layout,
            ),
        )
        self.assertEqual(FIXTURE_SPEC.shape, (34816, 5120))
        self.assertEqual(FIXTURE_SPEC.format, "F8E4M3_ROW_F32S")
        self.assertEqual(FIXTURE_SPEC.layout, "row-scaled-k128-v1")
        self.assertEqual(
            encoded_size(FIXTURE_SPEC.layout, FIXTURE_SPEC.format, FIXTURE_SPEC.shape),
            178_397_184,
        )


class E4M3GateUpFixtureWriterTest(unittest.TestCase):
    def test_writer_preserves_exact_full_object_payload_bytes(self) -> None:
        rows, columns = FIXTURE_SPEC.shape
        first_row = codec.encode_e4m3_rowwise_reference(
            [448.0, 1.0625, 1.1875, -0.0] + [0.0] * (columns - 4),
            1,
            columns,
        )
        first_codes = first_row[:columns]
        first_scale = first_row[columns : columns + 4]
        code_bytes = rows * columns
        scale_bytes = rows * 4
        zero_chunk = bytes(1 << 20)

        def zeros(count: int):
            while count:
                chunk = zero_chunk[: min(count, len(zero_chunk))]
                count -= len(chunk)
                yield chunk

        def payload():
            yield first_codes
            yield from zeros(code_bytes - len(first_codes))
            yield first_scale
            yield from zeros(scale_bytes - len(first_scale))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate-up.ninfer"
            write_gate_up_fixture(path, payload())
            with Artifact.open(path) as artifact:
                self.assertEqual(artifact.identity, FIXTURE_IDENTITY)
                self.assertEqual(len(artifact.objects), 1)
                obj = artifact.objects[0]
                self.assertIsInstance(obj, TensorObject)
                self.assertEqual(
                    (obj.name, obj.shape, obj.format, obj.layout, obj.bytes),
                    (
                        FIXTURE_SPEC.name,
                        FIXTURE_SPEC.shape,
                        FIXTURE_SPEC.format,
                        FIXTURE_SPEC.layout,
                        178_397_184,
                    ),
                )
                stored = artifact.payload(obj)
                self.assertEqual(bytes(stored[:4]), bytes((0x7E, 0x38, 0x3A, 0x80)))
                self.assertEqual(bytes(stored[columns : columns + 64]), bytes(64))
                self.assertEqual(bytes(stored[code_bytes : code_bytes + 4]), struct.pack("<f", 1.0))
                expected_hash = hashlib.sha256()
                for chunk in payload():
                    expected_hash.update(chunk)
                self.assertEqual(hashlib.sha256(stored).digest(), expected_hash.digest())
                stored.release()


if __name__ == "__main__":
    unittest.main()
