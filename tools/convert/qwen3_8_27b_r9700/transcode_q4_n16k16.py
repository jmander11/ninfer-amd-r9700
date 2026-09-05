"""One-shot offline migration of legacy Q4 planes to the production N16/K16 layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import secrets
import stat
import struct
from typing import Iterator

from tools.artifact.container import (MAGIC, PAYLOAD_ALIGNMENT, PREFIX, Artifact,
                                      ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec,
                                      )
from tools.artifact.layouts import align_up, encoded_size, get_layout, q4_n16k16_geometry

from . import fp8_hybrid_inventory


OLD_Q4_LAYOUT = "row-split-k128-v1"
NEW_Q4_LAYOUT = "r9700-q4g64-n16-k16-v1"
MODEL_ID = "qwen3.8-27b"
SOURCE_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-eval"
OUTPUT_WEIGHTS_ID = "r9700-q4g64-f8e4m3-four-role-n16k16-eval"


def _legacy_directory(path: Path):
    file = path.open("rb")
    try:
        size = path.stat().st_size
        magic, json_bytes = PREFIX.unpack(file.read(PREFIX.size))
        if magic != MAGIC or json_bytes <= 0:
            raise ValueError("source is not a NInfer v2 artifact")
        raw = json.loads(file.read(json_bytes).decode("utf-8"))
        if set(raw) != {"identity", "objects"} or set(raw["identity"]) != {"model_id", "weights_id"}:
            raise ValueError("source directory has an unexpected schema")
        payload_offset = align_up(PREFIX.size + json_bytes, PAYLOAD_ALIGNMENT)
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        return file, mapping, raw, payload_offset, size
    except BaseException:
        file.close()
        raise


def _q4_chunks(source: memoryview, n: int, k: int) -> Iterator[bytes]:
    geometry = q4_n16k16_geometry((n, k))
    groups, row_bytes = geometry.groups_per_row, geometry.k_pad // 2
    old_scale = align_up(n * row_bytes, 256)
    for tile in range(n // 16):
        out = bytearray(groups * 16 * 32)
        cursor = 0
        for group in range(groups):
            for pair in range(4):
                for lane in range(16):
                    begin = (tile * 16 + lane) * row_bytes + group * 32 + pair * 8
                    out[cursor:cursor + 8] = source[begin:begin + 8]
                    cursor += 8
        yield bytes(out)
    for tile in range(n // 16):
        out = bytearray(groups * 16 * 2)
        cursor = 0
        for group in range(groups):
            for lane in range(16):
                begin = old_scale + ((tile * 16 + lane) * groups + group) * 2
                out[cursor:cursor + 2] = source[begin:begin + 2]
                cursor += 2
        yield bytes(out)


def _logical_hashes(payload: memoryview, n: int, k: int, *, tiled: bool) -> tuple[str, str]:
    geometry = q4_n16k16_geometry((n, k)); groups = geometry.groups_per_row
    codes, scales = hashlib.sha256(), hashlib.sha256()
    if not tiled:
        codes.update(payload[:geometry.base_bytes])
        scales.update(payload[geometry.scale_offset:geometry.scale_offset + geometry.scale_bytes])
        return codes.hexdigest(), scales.hexdigest()
    for row in range(n):
        row_data = bytearray(groups * 32)
        for group in range(groups):
            for pair in range(4):
                source = (((row // 16) * groups + group) * 4 + pair) * 16 * 8 + (row % 16) * 8
                target = group * 32 + pair * 8
                row_data[target:target + 8] = payload[source:source + 8]
        codes.update(row_data)
    for row in range(n):
        row_data = bytearray(groups * 2)
        for group in range(groups):
            source = geometry.scale_offset + (((row // 16) * groups + group) * 16 + row % 16) * 2
            row_data[group * 2:group * 2 + 2] = payload[source:source + 2]
        scales.update(row_data)
    return codes.hexdigest(), scales.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> tuple[int, int, int]:
    observed = os.lstat(path)
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"artifact path is not a regular file: {path}")
    return observed.st_dev, observed.st_ino, observed.st_uid


def _unlink_owned(path: Path, expected: tuple[int, int, int] | None) -> None:
    if expected is None:
        return
    try:
        observed = os.lstat(path)
        if (stat.S_ISREG(observed.st_mode) and
                (observed.st_dev, observed.st_ino, observed.st_uid) == expected):
            os.unlink(path)
    except FileNotFoundError:
        pass


def _stage_path(parent: Path, final_name: str) -> Path:
    for _ in range(128):
        candidate = parent / f".{final_name}.{secrets.token_hex(16)}.stage"
        if not os.path.lexists(candidate):
            return candidate
    raise RuntimeError("could not allocate an unused staging leaf")


def _validate_selected_inventory(raw_objects: object) -> None:
    if not isinstance(raw_objects, list) or len(raw_objects) != len(fp8_hybrid_inventory.OBJECT_SPECS):
        raise ValueError("source artifact does not have the exact selected hybrid inventory")
    cursor = 0
    for raw, expected in zip(
            raw_objects, fp8_hybrid_inventory.OBJECT_SPECS, strict=True):
        if not isinstance(raw, dict) or raw.get("name") != expected.name:
            raise ValueError("source artifact object order/name differs from selected inventory")
        if expected.kind == "tensor":
            if set(raw) != {"name", "kind", "shape", "format", "layout", "offset", "bytes"}:
                raise ValueError(f"source tensor {expected.name} has unexpected descriptor members")
            expected_layout = OLD_Q4_LAYOUT if expected.format == "Q4G64_F16S" else expected.layout
            required = {"kind": "tensor", "shape": list(expected.shape),
                        "format": expected.format, "layout": expected_layout,
                        "offset": align_up(cursor, get_layout(expected.layout).alignment),
                        "bytes": encoded_size(expected.layout, expected.format, expected.shape)}
        else:
            if set(raw) != {"name", "kind", "encoding", "offset", "bytes"}:
                raise ValueError(f"source resource {expected.name} has unexpected descriptor members")
            if type(raw.get("bytes")) is not int or raw["bytes"] <= 0:
                raise ValueError(f"source resource {expected.name} has invalid byte size")
            required = {"kind": "resource", "encoding": expected.encoding,
                        "offset": cursor, "bytes": raw["bytes"]}
        if any(raw.get(key) != value for key, value in required.items()):
            raise ValueError(f"source object {expected.name} differs from selected inventory")
        cursor = required["offset"] + required["bytes"]


def transcode(source_path: Path, output_path: Path) -> dict[str, object]:
    source_path = source_path.resolve(strict=True)
    output_path = Path(os.path.abspath(os.fspath(output_path)))
    parent = output_path.parent
    parent_stat = os.lstat(parent)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("output parent must be a real existing directory")
    if source_path == output_path or os.path.lexists(output_path):
        raise ValueError("output must be a distinct nonexistent artifact path")
    stage_path = _stage_path(parent, output_path.name)
    file, mapping, raw, payload_offset, file_bytes = _legacy_directory(source_path)
    source_view = memoryview(mapping)
    source_closed = False
    stage_identity: tuple[int, int, int] | None = None
    published = False
    entries = []
    try:
        source_identity = raw["identity"]
        if source_identity != {"model_id": MODEL_ID, "weights_id": SOURCE_WEIGHTS_ID}:
            raise ValueError("source artifact identity is not the selected legacy Q4/FP8 artifact")
        _validate_selected_inventory(raw["objects"])
        identity = ArtifactIdentity(MODEL_ID, OUTPUT_WEIGHTS_ID)
        specs, entries, q4_names, source_hashes = [], [], [], {}
        cursor = 0
        for item in raw["objects"]:
            if not isinstance(item, dict) or item.get("offset") is None or item.get("bytes") is None:
                raise ValueError("malformed source object")
            offset, size = int(item["offset"]), int(item["bytes"])
            if offset < cursor or payload_offset + offset + size > file_bytes:
                raise ValueError(f"invalid source range for {item.get('name')}")
            cursor = offset + size
            payload = source_view[payload_offset + offset:payload_offset + offset + size]
            if item.get("kind") == "resource":
                spec = ResourceSpec(item["name"], item["encoding"], size)
                entries.append((item["name"], payload))
            elif item.get("kind") == "tensor":
                shape, fmt, layout = tuple(item["shape"]), item["format"], item["layout"]
                if fmt == "Q4G64_F16S":
                    if layout != OLD_Q4_LAYOUT or len(shape) != 2 or shape[0] % 16:
                        raise ValueError(f"Q4 source object {item['name']} is not migratable")
                    expected = q4_n16k16_geometry(shape).payload_bytes
                    if size != expected:
                        raise ValueError(f"Q4 source object {item['name']} has wrong size")
                    spec = TensorSpec(item["name"], shape, fmt, NEW_Q4_LAYOUT)
                    entries.append((item["name"], _q4_chunks(payload, *shape)))
                    q4_names.append(item["name"])
                    source_hashes[item["name"]] = _logical_hashes(payload, *shape, tiled=False)
                else:
                    if size != encoded_size(layout, fmt, shape):
                        raise ValueError(f"non-Q4 source object {item['name']} has wrong size")
                    spec = TensorSpec(item["name"], shape, fmt, layout)
                    entries.append((item["name"], payload))
            else:
                raise ValueError("unknown source object kind")
            specs.append(spec)
        if not q4_names:
            raise ValueError("source artifact contains no legacy Q4 objects")
        with ArtifactWriter(stage_path, identity, specs) as writer:
            for name, payload in entries:
                writer.write(name, payload)
        del payload
        stage_identity = _identity(stage_path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        stage_fd = os.open(stage_path, flags)
        try:
            observed = os.fstat(stage_fd)
            if (observed.st_dev, observed.st_ino, observed.st_uid) != stage_identity:
                raise RuntimeError("staging artifact identity changed before verification")
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        with Artifact.open(stage_path) as output:
            if output.identity != identity or len(output.objects) != len(specs):
                raise RuntimeError("output identity or inventory changed")
            for old, new in zip(raw["objects"], output.objects, strict=True):
                new_payload = output.payload(new)
                if old.get("format") == "Q4G64_F16S":
                    if _logical_hashes(new_payload, *old["shape"], tiled=True) != source_hashes[old["name"]]:
                        raise RuntimeError(f"Q4 decode equivalence failed for {old['name']}")
                else:
                    old_payload = source_view[payload_offset + old["offset"]:payload_offset + old["offset"] + old["bytes"]]
                    if hashlib.sha256(new_payload).digest() != hashlib.sha256(old_payload).digest():
                        raise RuntimeError(f"non-Q4 payload changed for {old['name']}")
                    old_payload.release()
                new_payload.release()
        del new_payload
        stage_sha256 = _file_sha256(stage_path)
        if _identity(stage_path) != stage_identity:
            raise RuntimeError("staging artifact identity changed during readback")
        entries.clear()
        source_view.release()
        mapping.close()
        file.close()
        source_closed = True
        os.link(stage_path, output_path, follow_symlinks=False)
        if _identity(output_path) != stage_identity:
            raise RuntimeError("published artifact does not name the verified staging inode")
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if _file_sha256(output_path) != stage_sha256 or _identity(output_path) != stage_identity:
            raise RuntimeError("published artifact failed immutable readback verification")
        return {"source": str(source_path), "output": str(output_path),
                "source_identity": source_identity,
                "identity": {"model_id": identity.model_id, "weights_id": identity.weights_id},
                "objects": len(specs), "q4_objects": len(q4_names),
                "output_sha256": stage_sha256,
                "verification": "exact logical Q4 code/scale hashes and exact non-Q4 payload hashes"}
    except BaseException:
        if published:
            _unlink_owned(output_path, stage_identity)
        raise
    finally:
        entries.clear()
        try:
            if not source_closed:
                source_view.release()
                mapping.close()
                file.close()
        finally:
            _unlink_owned(stage_path, stage_identity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(transcode(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
