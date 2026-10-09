from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.artifact.container import (
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    Artifact,
    ArtifactError,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
    write_artifact,
)
from tools.artifact.inspect import artifact_summary
from tools.artifact.layouts import align_up, encoded_size


def _small_specs():
    return [
        ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2),
        TensorSpec("direct/bf16", (2,), "BF16", "contiguous-le-v1"),
        TensorSpec("direct/fp32", (1,), "FP32", "contiguous-le-v1"),
        TensorSpec("direct/i32", (1,), "I32", "contiguous-le-v1"),
        TensorSpec("quant/q4", (16, 64), "Q4G64_F16S", "r9700-q4g64-n16-k16-v1"),
        TensorSpec("quant/q5", (1, 64), "Q5G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/q6", (1, 64), "Q6G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/w8", (1, 32), "W8G32_F16S", "row-split-k128-v1"),
        TensorSpec(
            "eval/fp8-row",
            (2, 130),
            "F8E4M3_ROW_F32S",
            "row-scaled-k128-v1",
        ),
    ]


def _payload(spec):
    if isinstance(spec, ResourceSpec):
        return b"{}"
    size = encoded_size(spec.layout, spec.format, spec.shape)
    return bytes(size)


def _write_raw(
    path,
    metadata: dict[str, object],
    payload: bytes = b"",
    *,
    magic: bytes = MAGIC,
) -> None:
    encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    payload_offset = align_up(PREFIX.size + len(encoded), PAYLOAD_ALIGNMENT)
    path.write_bytes(
        PREFIX.pack(magic, len(encoded))
        + encoded
        + bytes(payload_offset - PREFIX.size - len(encoded))
        + payload
    )


class ContainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_v2_round_trip_covers_every_registered_storage(self) -> None:
        path = self.tmp_path / "small.ninfer"
        specs = _small_specs()
        entries = [(spec, _payload(spec)) for spec in specs]
        identity = ArtifactIdentity("test-model", "test-weights")
        planned = write_artifact(path, identity, entries)

        prefix = path.read_bytes()[: PREFIX.size]
        magic, json_bytes = PREFIX.unpack(prefix)
        self.assertEqual(magic, MAGIC)
        self.assertEqual(prefix[8:], struct.pack("<Q", json_bytes))

        with Artifact.open(path) as artifact:
            self.assertEqual(artifact.identity, identity)
            self.assertEqual(
                artifact.payload_offset, align_up(PREFIX.size + json_bytes, PAYLOAD_ALIGNMENT)
            )
            self.assertEqual(artifact.objects, planned)
            for spec, expected in entries:
                self.assertEqual(bytes(artifact.payload(spec.name)), expected)
            summary = artifact_summary(artifact)
            self.assertEqual(summary["model_id"], "test-model")
            self.assertEqual(summary["weights_id"], "test-weights")
            self.assertEqual(summary["objects"], 9)
            self.assertEqual(
                summary["formats"],
                {
                    "BF16": 1,
                    "FP32": 1,
                    "I32": 1,
                    "Q4G64_F16S": 1,
                    "Q5G64_F16S": 1,
                    "Q6G64_F16S": 1,
                    "W8G32_F16S": 1,
                    "F8E4M3_ROW_F32S": 1,
                },
            )

    def test_writer_never_replaces_destination_created_while_staging(self) -> None:
        path = self.tmp_path / "raced.ninfer"
        spec = ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2)
        writer = ArtifactWriter(path, ArtifactIdentity("test-model", "candidate"), [spec])
        writer.write(spec.name, b"{}")

        competing_payload = b"created by another writer"
        path.write_bytes(competing_payload)
        with self.assertRaises(FileExistsError):
            writer.finish()

        self.assertEqual(path.read_bytes(), competing_payload)
        self.assertEqual(list(self.tmp_path.iterdir()), [path])

    def test_failed_writer_leaves_no_partial_destination_or_staging_file(self) -> None:
        path = self.tmp_path / "failed.ninfer"
        spec = ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2)

        with (
            self.assertRaisesRegex(ArtifactError, "has 1 bytes; expected 2"),
            ArtifactWriter(path, ArtifactIdentity("test-model", "candidate"), [spec]) as writer,
        ):
            writer.write(spec.name, b"{")

        self.assertFalse(path.exists())
        self.assertEqual(list(self.tmp_path.iterdir()), [])

    def test_staging_cleanup_fault_does_not_make_publication_ambiguous(self) -> None:
        path = self.tmp_path / "complete.ninfer"
        spec = ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2)
        writer = ArtifactWriter(path, ArtifactIdentity("test-model", "candidate"), [spec])
        writer.write(spec.name, b"{}")
        staging = next(self.tmp_path.iterdir())
        original_unlink = type(staging).unlink
        injected = False

        def fail_first_staging_unlink(candidate, *args, **kwargs):
            nonlocal injected
            if candidate == staging and not injected:
                injected = True
                raise PermissionError("injected staging cleanup failure")
            return original_unlink(candidate, *args, **kwargs)

        with mock.patch.object(type(staging), "unlink", fail_first_staging_unlink):
            writer.finish()
            self.assertTrue(injected)
            self.assertTrue(path.read_bytes().endswith(b"{}"))
            self.assertTrue(staging.exists())

            writer.finish()
            self.assertFalse(staging.exists())

        with Artifact.open(path) as artifact:
            self.assertEqual(artifact.identity, ArtifactIdentity("test-model", "candidate"))
            self.assertEqual(bytes(artifact.payload(spec.name)), b"{}")

    def test_reader_rejects_invalid_framing_schema_and_geometry(self) -> None:
        path = self.tmp_path / "invalid.ninfer"

        path.write_bytes(PREFIX.pack(MAGIC, 100) + b"{}")
        with self.assertRaisesRegex(ArtifactError, "beyond the file"):
            Artifact.open(path)

        root = {
            "identity": {
                "model_id": "test-model",
                "weights_id": "test-weights",
            },
            "objects": [
                {
                    "name": "a",
                    "kind": "tensor",
                    "shape": [1],
                    "format": "I32",
                    "layout": "contiguous-le-v1",
                    "offset": 0,
                    "bytes": 4,
                },
                {
                    "name": "b",
                    "kind": "resource",
                    "encoding": "raw-bytes-v1",
                    "offset": 2,
                    "bytes": 2,
                },
            ],
        }
        _write_raw(path, root, b"\x00" * 4)
        with self.assertRaisesRegex(ArtifactError, "overlaps"):
            Artifact.open(path)

        root["source_recipe"] = "must not enter the container"
        _write_raw(path, root, b"\x00" * 4)
        with self.assertRaisesRegex(ArtifactError, "exactly"):
            Artifact.open(path)

        root = {
            "identity": {
                "model_id": "test-model",
                "weights_id": "test-weights",
            },
            "objects": [
                {
                    "name": "bad-size",
                    "kind": "tensor",
                    "shape": [2],
                    "format": "BF16",
                    "layout": "contiguous-le-v1",
                    "offset": 0,
                    "bytes": 2,
                }
            ],
        }
        _write_raw(path, root, b"\x00" * 2)
        with self.assertRaisesRegex(ArtifactError, "layout requires"):
            Artifact.open(path)

    def test_reader_rejects_unsupported_v1(self) -> None:
        path = self.tmp_path / "legacy.ninfer"
        _write_raw(
            path,
            {"model_id": "test-model", "objects": [{"unused": True}]},
            magic=b"NINFER\x00\x01",
        )
        with self.assertRaisesRegex(ArtifactError, r"NInfer artifact v1 is no longer supported"):
            Artifact.open(path)


if __name__ == "__main__":
    unittest.main()
