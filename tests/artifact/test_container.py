from __future__ import annotations

import json
import struct
from unittest import mock

import pytest

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


def test_v2_round_trip_covers_every_registered_storage(tmp_path):
    path = tmp_path / "small.ninfer"
    specs = _small_specs()
    entries = [(spec, _payload(spec)) for spec in specs]
    identity = ArtifactIdentity("test-model", "test-weights")
    planned = write_artifact(path, identity, entries)

    prefix = path.read_bytes()[: PREFIX.size]
    magic, json_bytes = PREFIX.unpack(prefix)
    assert magic == MAGIC
    assert prefix[8:] == struct.pack("<Q", json_bytes)

    with Artifact.open(path) as artifact:
        assert artifact.identity == identity
        assert artifact.payload_offset == align_up(PREFIX.size + json_bytes, PAYLOAD_ALIGNMENT)
        assert artifact.objects == planned
        for spec, expected in entries:
            assert bytes(artifact.payload(spec.name)) == expected
        summary = artifact_summary(artifact)
        assert summary["model_id"] == "test-model"
        assert summary["weights_id"] == "test-weights"
        assert summary["objects"] == 9
        assert summary["formats"] == {
            "BF16": 1,
            "FP32": 1,
            "I32": 1,
            "Q4G64_F16S": 1,
            "Q5G64_F16S": 1,
            "Q6G64_F16S": 1,
            "W8G32_F16S": 1,
            "F8E4M3_ROW_F32S": 1,
        }


def test_writer_never_replaces_destination_created_while_staging(tmp_path):
    path = tmp_path / "raced.ninfer"
    spec = ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2)
    writer = ArtifactWriter(path, ArtifactIdentity("test-model", "candidate"), [spec])
    writer.write(spec.name, b"{}")

    competing_payload = b"created by another writer"
    path.write_bytes(competing_payload)
    with pytest.raises(FileExistsError):
        writer.finish()

    assert path.read_bytes() == competing_payload
    assert list(tmp_path.iterdir()) == [path]


def test_failed_writer_leaves_no_partial_destination_or_staging_file(tmp_path):
    path = tmp_path / "failed.ninfer"
    spec = ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2)

    with pytest.raises(ArtifactError, match="has 1 bytes; expected 2"):
        with ArtifactWriter(
            path, ArtifactIdentity("test-model", "candidate"), [spec]
        ) as writer:
            writer.write(spec.name, b"{")

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_staging_cleanup_fault_does_not_make_publication_ambiguous(tmp_path):
    path = tmp_path / "complete.ninfer"
    spec = ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2)
    writer = ArtifactWriter(path, ArtifactIdentity("test-model", "candidate"), [spec])
    writer.write(spec.name, b"{}")
    staging = next(tmp_path.iterdir())
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
        assert injected
        assert path.read_bytes().endswith(b"{}")
        assert staging.exists()

        writer.finish()
        assert not staging.exists()

    with Artifact.open(path) as artifact:
        assert artifact.identity == ArtifactIdentity("test-model", "candidate")
        assert bytes(artifact.payload(spec.name)) == b"{}"


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


def test_reader_rejects_invalid_framing_schema_and_geometry(tmp_path):
    path = tmp_path / "invalid.ninfer"

    path.write_bytes(PREFIX.pack(MAGIC, 100) + b"{}")
    with pytest.raises(ArtifactError, match="beyond the file"):
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
    with pytest.raises(ArtifactError, match="overlaps"):
        Artifact.open(path)

    root["source_recipe"] = "must not enter the container"
    _write_raw(path, root, b"\x00" * 4)
    with pytest.raises(ArtifactError, match="exactly"):
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
    with pytest.raises(ArtifactError, match="layout requires"):
        Artifact.open(path)


def test_reader_rejects_unsupported_v1(tmp_path):
    path = tmp_path / "legacy.ninfer"
    _write_raw(
        path,
        {"model_id": "test-model", "objects": [{"unused": True}]},
        magic=b"NINFER\x00\x01",
    )
    with pytest.raises(ArtifactError, match=r"NInfer artifact v1 is no longer supported"):
        Artifact.open(path)
