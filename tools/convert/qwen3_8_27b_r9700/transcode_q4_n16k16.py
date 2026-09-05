"""Offline migration of the three legacy Q4 evaluations to production N16/K16 storage."""

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
from dataclasses import dataclass
from typing import Iterator, Sequence

from tools.artifact.container import (MAGIC, PAYLOAD_ALIGNMENT, PREFIX, Artifact,
                                      ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec,
                                      plan_objects,
                                      )
from tools.artifact.layouts import align_up, encoded_size, get_layout, q4_n16k16_geometry

from . import (fp8_hybrid_decision, fp8_hybrid_inventory, q4_inventory,
               q4_w8_mse_inventory)


OLD_Q4_LAYOUT = "row-split-k128-v1"
NEW_Q4_LAYOUT = "r9700-q4g64-n16-k16-v1"
MODEL_ID = "qwen3.8-27b"


@dataclass(frozen=True)
class MigrationProfile:
    output_weights_id: str
    object_specs: tuple[object, ...]


PROFILES_BY_SOURCE = {
    "r9700-q4g64-eval": MigrationProfile(
        "r9700-q4g64-n16k16-eval", q4_inventory.OBJECT_SPECS),
    "r9700-q4-w8-mse-eval": MigrationProfile(
        "r9700-q4-w8-mse-n16k16-eval", q4_w8_mse_inventory.OBJECT_SPECS),
    "r9700-q4g64-f8e4m3-four-role-eval": MigrationProfile(
        "r9700-q4g64-f8e4m3-four-role-n16k16-eval",
        fp8_hybrid_inventory.OBJECT_SPECS),
}


def _legacy_directory(path: Path):
    file = path.open("rb")
    try:
        size = os.fstat(file.fileno()).st_size
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


def _file_sha256_identity(path: Path, expected: tuple[int, int, int]) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        observed = (before.st_dev, before.st_ino, before.st_uid)
        if not stat.S_ISREG(before.st_mode) or observed != expected:
            raise RuntimeError(f"file identity changed before hashing: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 8 << 20):
            digest.update(block)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_uid) != expected:
            raise RuntimeError(f"file identity changed during hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_bytes_identity(path: Path, expected: tuple[int, int, int]) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino, before.st_uid) != expected):
            raise RuntimeError(f"file identity changed before readback: {path}")
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_uid) != expected:
            raise RuntimeError(f"file identity changed during readback: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_uid, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _unlink_owned_fingerprint(path: Path, expected: tuple[int, ...]) -> None:
    try:
        observed = os.lstat(path)
        if stat.S_ISREG(observed.st_mode) and _stat_fingerprint(observed) == expected:
            os.unlink(path)
    except FileNotFoundError:
        pass


def _hash_open_fd(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while block := os.read(descriptor, 8 << 20):
        digest.update(block)
    return digest.hexdigest()


def _object_plan_sha256(objects: object) -> str:
    return hashlib.sha256(json.dumps(
        objects, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


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


def _validate_registered_inventory(
    raw_objects: object, profile: MigrationProfile
) -> tuple[int, int]:
    if not isinstance(raw_objects, list) or len(raw_objects) != len(profile.object_specs):
        raise ValueError("source artifact does not have the exact registered evaluation inventory")
    cursor = 0
    q4_objects = 0
    for raw, expected in zip(raw_objects, profile.object_specs, strict=True):
        if not isinstance(raw, dict) or raw.get("name") != expected.name:
            raise ValueError("source artifact object order/name differs from registered inventory")
        if expected.kind == "tensor":
            if set(raw) != {"name", "kind", "shape", "format", "layout", "offset", "bytes"}:
                raise ValueError(f"source tensor {expected.name} has unexpected descriptor members")
            expected_layout = OLD_Q4_LAYOUT if expected.format == "Q4G64_F16S" else expected.layout
            q4_objects += int(expected.format == "Q4G64_F16S")
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
            raise ValueError(f"source object {expected.name} differs from registered inventory")
        cursor = required["offset"] + required["bytes"]
    if not q4_objects:
        raise ValueError("source artifact contains no legacy Q4 objects")
    return q4_objects, cursor


def _profile(raw_identity: object) -> tuple[dict[str, str], MigrationProfile]:
    if not isinstance(raw_identity, dict) or set(raw_identity) != {"model_id", "weights_id"}:
        raise ValueError("source artifact identity is malformed")
    source_identity = {
        "model_id": raw_identity["model_id"],
        "weights_id": raw_identity["weights_id"],
    }
    if not all(isinstance(value, str) for value in source_identity.values()):
        raise ValueError("source artifact identity values are malformed")
    if source_identity["model_id"] != MODEL_ID:
        raise ValueError("source artifact model identity is unsupported")
    profile = PROFILES_BY_SOURCE.get(source_identity["weights_id"])
    if profile is None:
        raise ValueError("source artifact identity is not an allowed legacy evaluation profile")
    return source_identity, profile


def _output_path(path: Path, source_path: Path) -> tuple[Path, Path]:
    output_path = Path(os.path.abspath(os.fspath(path)))
    parent = output_path.parent
    parent_stat = os.lstat(parent)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("output parent must be a real existing directory")
    if source_path == output_path or os.path.lexists(output_path):
        raise ValueError("output must be a distinct nonexistent artifact path")
    return output_path, parent


def preflight(source_path: Path, output_path: Path | None = None) -> dict[str, object]:
    source_path = source_path.resolve(strict=True)
    file, mapping, raw, payload_offset, file_bytes = _legacy_directory(source_path)
    try:
        source_identity, profile = _profile(raw["identity"])
        q4_objects, payload_bytes = _validate_registered_inventory(raw["objects"], profile)
        if payload_offset + payload_bytes != file_bytes:
            raise ValueError("source artifact payload extent differs from registered inventory")
        if output_path is not None:
            output_path, _ = _output_path(output_path, source_path)
        return {
            "mode": "preflight-only",
            "source": str(source_path),
            "output": None if output_path is None else str(output_path),
            "source_identity": source_identity,
            "identity": {"model_id": MODEL_ID,
                         "weights_id": profile.output_weights_id},
            "source_file_size_bytes": file_bytes,
            "source_payload_offset": payload_offset,
            "source_payload_bytes": payload_bytes,
            "objects": len(raw["objects"]),
            "q4_objects": q4_objects,
            "verification": (
                "exact source identity, descriptor inventory, offsets, formats, layouts, "
                "and sizes; payload bytes not read"
            ),
        }
    finally:
        mapping.close()
        file.close()


def transcode(source_path: Path, output_path: Path) -> dict[str, object]:
    source_path = source_path.resolve(strict=True)
    output_path, parent = _output_path(output_path, source_path)
    stage_path = _stage_path(parent, output_path.name)
    file, mapping, raw, payload_offset, file_bytes = _legacy_directory(source_path)
    source_view = memoryview(mapping)
    source_closed = False
    stage_identity: tuple[int, int, int] | None = None
    published = False
    entries = []
    try:
        source_identity, profile = _profile(raw["identity"])
        expected_q4_objects, payload_bytes = _validate_registered_inventory(
            raw["objects"], profile)
        if payload_offset + payload_bytes != file_bytes:
            raise ValueError("source artifact payload extent differs from registered inventory")
        identity = ArtifactIdentity(MODEL_ID, profile.output_weights_id)
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
        if len(q4_names) != expected_q4_objects:
            raise RuntimeError("migrated Q4 object count differs from registered inventory")
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


def _validate_migration_pair(source_path: Path, output_path: Path) -> dict[str, object]:
    """Reopen a completed migration and independently verify every represented object."""

    source_path = Path(os.path.abspath(os.fspath(source_path)))
    output_path = Path(os.path.abspath(os.fspath(output_path)))
    source_file_identity = _identity(source_path)
    output_file_identity = _identity(output_path)
    file, mapping, raw, payload_offset, file_bytes = _legacy_directory(source_path)
    source_open = os.fstat(file.fileno())
    if (source_open.st_dev, source_open.st_ino, source_open.st_uid) != source_file_identity:
        mapping.close()
        file.close()
        raise RuntimeError("source artifact identity changed while opening")
    source_view = memoryview(mapping)
    try:
        source_identity, profile = _profile(raw["identity"])
        expected_q4, payload_bytes = _validate_registered_inventory(raw["objects"], profile)
        if payload_offset + payload_bytes != file_bytes:
            raise ValueError("source artifact payload extent differs from registered inventory")
        q4_objects = 0
        expected_specs = []
        for old in raw["objects"]:
            if old["kind"] == "tensor":
                expected_specs.append(TensorSpec(
                    old["name"], tuple(old["shape"]), old["format"],
                    NEW_Q4_LAYOUT if old["format"] == "Q4G64_F16S" else old["layout"],
                ))
            else:
                expected_specs.append(ResourceSpec(old["name"], old["encoding"], old["bytes"]))
        expected_objects = plan_objects(expected_specs)
        with Artifact.open(output_path) as output:
            output_open = os.fstat(output._file.fileno())
            if (output_open.st_dev, output_open.st_ino,
                    output_open.st_uid) != output_file_identity:
                raise RuntimeError("output artifact identity changed while opening")
            expected_identity = ArtifactIdentity(MODEL_ID, profile.output_weights_id)
            if output.identity != expected_identity or output.objects != expected_objects:
                raise ValueError("migrated artifact identity or exact N16 object plan differs")
            for old, new in zip(raw["objects"], output.objects, strict=True):
                if new.name != old["name"] or new.kind != old["kind"] or new.bytes != old["bytes"]:
                    raise ValueError(f"migrated descriptor differs for {old['name']}")
                old_payload = source_view[
                    payload_offset + old["offset"]:
                    payload_offset + old["offset"] + old["bytes"]
                ]
                new_payload = output.payload(new)
                try:
                    if old.get("format") == "Q4G64_F16S":
                        if (new.format != old["format"] or tuple(new.shape) != tuple(old["shape"])
                                or old["layout"] != OLD_Q4_LAYOUT
                                or new.layout != NEW_Q4_LAYOUT
                                or _logical_hashes(old_payload, *old["shape"], tiled=False)
                                != _logical_hashes(new_payload, *old["shape"], tiled=True)):
                            raise ValueError(f"logical Q4 migration differs for {old['name']}")
                        q4_objects += 1
                    else:
                        if old["kind"] == "tensor":
                            descriptor_matches = (
                                new.format == old["format"] and new.layout == old["layout"]
                                and tuple(new.shape) == tuple(old["shape"])
                            )
                        else:
                            descriptor_matches = new.encoding == old["encoding"]
                        if (not descriptor_matches
                                or hashlib.sha256(old_payload).digest()
                                != hashlib.sha256(new_payload).digest()):
                            raise ValueError(f"non-Q4 migration differs for {old['name']}")
                finally:
                    new_payload.release()
                    old_payload.release()
            output_objects = [obj.to_json() for obj in output.objects]
        if q4_objects != expected_q4:
            raise ValueError("migrated Q4 object count differs")
        return {
            "source": str(source_path),
            "output": str(output_path),
            "source_identity": source_identity,
            "identity": {"model_id": MODEL_ID, "weights_id": profile.output_weights_id},
            "source_bytes": file_bytes,
            "output_bytes": output_path.stat().st_size,
            "source_sha256": _file_sha256_identity(source_path, source_file_identity),
            "output_sha256": _file_sha256_identity(output_path, output_file_identity),
            "objects": len(raw["objects"]),
            "q4_objects": q4_objects,
            "source_object_plan_sha256": _object_plan_sha256(raw["objects"]),
            "object_plan_sha256": _object_plan_sha256(output_objects),
            "source_file_identity": source_file_identity,
            "output_file_identity": output_file_identity,
        }
    finally:
        source_view.release()
        mapping.close()
        file.close()


def _publish_json_create_only(
    path: Path, payload: dict[str, object]
) -> tuple[str, tuple[int, int, int], tuple[int, ...]]:
    path = Path(os.path.abspath(os.fspath(path)))
    parent = path.parent
    parent_stat = os.lstat(parent)
    if not stat.S_ISDIR(parent_stat.st_mode) or os.path.lexists(path):
        raise ValueError("receipt parent must be real and final receipt must not exist")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    stage = _stage_path(parent, path.name)
    stage_identity = None
    published = False
    try:
        descriptor = os.open(
            stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        try:
            observed = os.fstat(descriptor)
            stage_identity = (observed.st_dev, observed.st_ino, observed.st_uid)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(stage, path, follow_symlinks=False)
        if _identity(path) != stage_identity:
            raise RuntimeError("published receipt does not name the verified staging inode")
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if path.read_bytes() != encoded or _identity(path) != stage_identity:
            raise RuntimeError("published receipt failed immutable readback")
        _unlink_owned(stage, stage_identity)
        return (hashlib.sha256(encoded).hexdigest(), stage_identity,
                _stat_fingerprint(os.lstat(path)))
    except BaseException:
        if published:
            _unlink_owned(path, stage_identity)
        raise
    finally:
        _unlink_owned(stage, stage_identity)


def publish_hybrid_migration_receipt(source_path: Path, output_path: Path) -> dict[str, object]:
    """Publish the adjacent authority for an already-validated four-role migration."""

    validation = _validate_migration_pair(source_path, output_path)
    decision = fp8_hybrid_decision.DECISION
    if validation["identity"] != {"model_id": MODEL_ID, "weights_id": decision.weights_id}:
        raise ValueError("migration receipt is supported only for the selected four-role identity")
    source = Path(str(validation["source"]))
    output = Path(str(validation["output"]))
    if (_identity(source) != validation["source_file_identity"]
            or _identity(output) != validation["output_file_identity"]):
        raise RuntimeError("artifact identity changed after migration validation")
    upstream_path = Path(str(source) + ".conversion.json")
    if upstream_path.is_symlink() or not upstream_path.is_file():
        raise ValueError("legacy four-role conversion receipt is missing or nonregular")
    upstream_identity = _identity(upstream_path)
    transcoder_path = Path(__file__).resolve(strict=True)
    transcoder_identity = _identity(transcoder_path)
    upstream_bytes = _read_bytes_identity(upstream_path, upstream_identity)
    transcoder_bytes = _read_bytes_identity(transcoder_path, transcoder_identity)
    upstream = json.loads(upstream_bytes.decode("utf-8"))
    candidate = upstream.get("candidate") if isinstance(upstream, dict) else None
    artifact = upstream.get("artifact") if isinstance(upstream, dict) else None
    original_source = upstream.get("source") if isinstance(upstream, dict) else None
    expected_legacy_recipe = "r9700-q4g64-f8e4m3-four-role-eval-v0"
    if (
        upstream.get("identity") != validation["source_identity"]
        or upstream.get("target_key") != fp8_hybrid_inventory.TARGET_KEY
        or upstream.get("recipe_id") != expected_legacy_recipe
        or not isinstance(candidate, dict)
        or candidate.get("status") != "registered-evaluation-only"
        or candidate.get("weight_recipe_selected") is not False
        or candidate.get("selection_sha256") != decision.selection_sha256
        or candidate.get("format_counts") != fp8_hybrid_inventory.FORMAT_COUNTS
        or candidate.get("format_encoded_bytes") != fp8_hybrid_inventory.FORMAT_ENCODED_BYTES
        or candidate.get("tensor_encoded_bytes") != fp8_hybrid_inventory.TENSOR_ENCODED_BYTES
        or candidate.get("device_arena_bytes") != fp8_hybrid_inventory.DEVICE_ARENA_BYTES
        or candidate.get("object_plan_sha256") != validation["source_object_plan_sha256"]
        or not isinstance(artifact, dict)
        or Path(str(artifact.get("path"))).resolve(strict=True) != source
        or artifact.get("bytes") != validation["source_bytes"]
        or artifact.get("sha256") != validation["source_sha256"]
        or not isinstance(original_source, dict)
    ):
        raise ValueError("legacy four-role conversion receipt differs from actual source authority")
    # The active receipt contract retains the original checkpoint/ranking ancestry
    # while the migration block records the immediate stored-byte transformation.
    new_candidate = dict(candidate)
    new_candidate["object_plan_sha256"] = validation["object_plan_sha256"]
    receipt = {
        "artifact_type": "ninfer_qwen3_8_27b_r9700_q4_n16k16_migration_receipt",
        "schema_version": 1,
        "identity": validation["identity"],
        "target_key": fp8_hybrid_inventory.TARGET_KEY,
        "recipe_id": decision.recipe_id,
        "source": original_source,
        "artifact": {
            "path": str(output), "bytes": validation["output_bytes"],
            "sha256": validation["output_sha256"],
        },
        "candidate": new_candidate,
        "migration": {
            "source_artifact": {
                "path": str(source), "identity": validation["source_identity"],
                "bytes": validation["source_bytes"], "sha256": validation["source_sha256"],
            },
            "source_conversion_receipt": {
                "path": str(upstream_path.resolve(strict=True)),
                "sha256": hashlib.sha256(upstream_bytes).hexdigest(),
            },
            "transcoder": {
                "path": str(transcoder_path),
                "sha256": hashlib.sha256(transcoder_bytes).hexdigest(),
            },
            "storage_transform": {"from": OLD_Q4_LAYOUT, "to": NEW_Q4_LAYOUT},
            "verification": "exact logical Q4 code/scale hashes and exact non-Q4 payload hashes",
            "objects": validation["objects"],
            "q4_objects": validation["q4_objects"],
        },
    }
    receipt_path = Path(str(output) + ".conversion.json")
    owners = (
        (source, validation["source_file_identity"], validation["source_sha256"]),
        (output, validation["output_file_identity"], validation["output_sha256"]),
        (upstream_path, upstream_identity, hashlib.sha256(upstream_bytes).hexdigest()),
        (transcoder_path, transcoder_identity, hashlib.sha256(transcoder_bytes).hexdigest()),
    )
    opened_owners = []
    try:
        for path, identity, digest in owners:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            observed = os.fstat(descriptor)
            if (observed.st_dev, observed.st_ino, observed.st_uid) != identity:
                os.close(descriptor)
                raise RuntimeError("migration owner changed while opening transaction")
            opened_owners.append((path, descriptor, _stat_fingerprint(observed), digest))

        def owners_stable() -> bool:
            for path, descriptor, fingerprint, digest in opened_owners:
                if (_identity(path) != fingerprint[:3]
                        or _stat_fingerprint(os.fstat(descriptor)) != fingerprint
                        or _hash_open_fd(descriptor) != digest
                        or _stat_fingerprint(os.fstat(descriptor)) != fingerprint):
                    return False
            return all(
                _identity(path) == fingerprint[:3]
                and _stat_fingerprint(os.fstat(descriptor)) == fingerprint
                for path, descriptor, fingerprint, _ in opened_owners
            )

        if not owners_stable():
            raise RuntimeError("artifact authority changed before receipt publication")
        receipt_sha256, receipt_identity, receipt_fingerprint = _publish_json_create_only(
            receipt_path, receipt)
        try:
            if not owners_stable():
                raise RuntimeError("migration authority changed after receipt publication")
            expected_receipt = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
            actual_receipt = _read_bytes_identity(receipt_path, receipt_identity)
            if (actual_receipt != expected_receipt
                    or hashlib.sha256(actual_receipt).hexdigest() != receipt_sha256
                    or _identity(receipt_path) != receipt_identity):
                raise RuntimeError("published receipt changed before transaction completion")
        except BaseException:
            _unlink_owned_fingerprint(receipt_path, receipt_fingerprint)
            raise
    finally:
        for _, descriptor, _, _ in opened_owners:
            os.close(descriptor)
    report = {"receipt": str(receipt_path), "receipt_sha256": receipt_sha256, **validation}
    report.pop("source_file_identity")
    report.pop("output_file_identity")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate identity/inventory and projected output without reading payloads")
    parser.add_argument("--publish-hybrid-receipt-only", action="store_true",
                        help="validate an existing four-role migration and publish its receipt")
    args = parser.parse_args(argv)
    if args.preflight_only and args.publish_hybrid_receipt_only:
        parser.error("receipt-only and preflight-only are mutually exclusive")
    if not args.preflight_only and args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    try:
        report = (preflight(args.source, args.output) if args.preflight_only else
                  publish_hybrid_migration_receipt(args.source, args.output)
                  if args.publish_hybrid_receipt_only else
                  transcode(args.source, args.output))
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
