"""Common immutable identities for the three terminal converter preflights."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Iterable

from tools.reference.qwen3_8_27b_bf16 import protocol as bf16_protocol


SOURCE_SHARDS = tuple(f"model-{index:05d}-of-00018.safetensors" for index in range(1, 19))
SOURCE_TENSORS = 1199


def _regular(path: Path, label: str) -> Path:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    return path.resolve(strict=True)


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_checkpoint(model_dir: Path) -> dict[str, object]:
    model = model_dir.resolve(strict=True)
    config = _regular(model / "config.json", "source config")
    index = _regular(model / "model.safetensors.index.json", "source index")
    value = json.loads(index.read_text(encoding="utf-8"))
    weight_map = bf16_protocol.validate_checkpoint_files(model)
    total_tensor_bytes = value.get("metadata", {}).get("total_size")
    if (isinstance(total_tensor_bytes, bool) or not isinstance(total_tensor_bytes, (int, float))
            or not float(total_tensor_bytes).is_integer()
            or int(total_tensor_bytes) != bf16_protocol.SOURCE_TOTAL_BYTES):
        raise ValueError("source index has invalid exact tensor byte count")
    names = tuple(sorted(set(weight_map.values())))
    if names != SOURCE_SHARDS:
        raise ValueError("source index does not bind the exact 18-shard set")
    shards = []
    for name in names:
        path = _regular(model / name, f"source shard {name}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"source shard is empty: {name}")
        shards.append({"name": name, "bytes": size})
    return {
        "config": {"path": str(config), "bytes": config.stat().st_size, "sha256": _sha(config)},
        "index": {"path": str(index), "bytes": index.stat().st_size, "sha256": _sha(index)},
        "indexed_tensor_count": SOURCE_TENSORS,
        "index_total_tensor_bytes": int(total_tensor_bytes),
        "shards": shards,
        "shard_total_file_bytes": sum(row["bytes"] for row in shards),
        "shard_payload_sha256": None,
        "shard_payload_hash_policy": (
            "not computed: exact names and nonempty regular files are the target-owned contract"),
    }


def frontend_resources(resources: Iterable[object]) -> list[dict[str, object]]:
    rows = []
    for resource in resources:
        name, data = resource.name, resource.data
        if not isinstance(name, str) or not isinstance(data, bytes):
            raise ValueError("frontend resource preflight is malformed")
        rows.append({"name": name, "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()})
    if len(rows) != 6 or len({row["name"] for row in rows}) != 6:
        raise ValueError("converter preflight requires exactly six frontend resources")
    return rows


def object_plan_sha256(objects: Iterable[object]) -> str:
    encoded = json.dumps([obj.to_json() for obj in objects], sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["SOURCE_SHARDS", "SOURCE_TENSORS", "frontend_resources",
           "object_plan_sha256", "source_checkpoint"]
