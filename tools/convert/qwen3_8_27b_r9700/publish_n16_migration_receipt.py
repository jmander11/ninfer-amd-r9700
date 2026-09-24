"""Publish and validate the retained all-Q4/mixed N16 migration authorities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Sequence

from . import q4_inventory, q4_w8_mse_inventory
from . import transcode_q4_n16k16 as migration


MODEL_ID = "qwen3.8-27b"
ARTIFACT_TYPE = "ninfer_qwen3_8_27b_r9700_q4_n16k16_migration_receipt"
PROFILES = {
    q4_inventory.WEIGHTS_ID: {
        "source_weights_id": "r9700-q4g64-eval",
        "source_recipe_id": "r9700-all-q4g64-eval-v0",
        "inventory": q4_inventory,
        "q4_objects": 439,
    },
    q4_w8_mse_inventory.WEIGHTS_ID: {
        "source_weights_id": "r9700-q4-w8-mse-eval",
        "source_recipe_id": "r9700-source-q4-promoted-w8-source-mse8-eval-v0",
        "inventory": q4_w8_mse_inventory,
        "q4_objects": 183,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> tuple[int, int, int]:
    try:
        value = os.lstat(path)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return value.st_dev, value.st_ino, value.st_uid


def _valid_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _directory_snapshot(path: Path, label: str) -> tuple[dict, int, int]:
    """Read only the framed directory while holding the exact regular inode."""
    identity = _regular(path, label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_uid) != identity:
            raise ValueError(f"{label} identity changed while opening")
        prefix = os.read(descriptor, migration.PREFIX.size)
        if len(prefix) != migration.PREFIX.size:
            raise ValueError(f"{label} has a truncated prefix")
        magic, directory_bytes = migration.PREFIX.unpack(prefix)
        if magic != migration.MAGIC or directory_bytes <= 0:
            raise ValueError(f"{label} is not a NInfer v2 artifact")
        payload = bytearray()
        while len(payload) < directory_bytes:
            block = os.read(descriptor, directory_bytes - len(payload))
            if not block:
                raise ValueError(f"{label} directory is truncated")
            payload.extend(block)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} directory is invalid") from error
        final = os.fstat(descriptor)
        if ((final.st_dev, final.st_ino, final.st_uid) != identity
                or _regular(path, label) != identity):
            raise ValueError(f"{label} identity changed during directory read")
        return value, final.st_size, directory_bytes
    finally:
        os.close(descriptor)


def _validate_exact_plans(source_path: Path, output_path: Path,
                          profile: dict[str, Any]) -> tuple[str, str]:
    source, source_bytes, source_directory_bytes = _directory_snapshot(
        source_path, "legacy source artifact")
    output, output_bytes, output_directory_bytes = _directory_snapshot(
        output_path, "N16 artifact")
    source_identity, migration_profile = migration._profile(source.get("identity"))
    if (source_identity != {"model_id": MODEL_ID,
                            "weights_id": profile["source_weights_id"]}):
        raise ValueError("legacy source identity differs from receipt profile")
    _, source_payload_bytes = migration._validate_registered_inventory(
        source.get("objects"), migration_profile)
    source_payload_offset = migration.align_up(
        migration.PREFIX.size + source_directory_bytes, migration.PAYLOAD_ALIGNMENT)
    expected_specs = []
    for old in source["objects"]:
        if old["kind"] == "tensor":
            expected_specs.append(migration.TensorSpec(
                old["name"], tuple(old["shape"]), old["format"],
                migration.NEW_Q4_LAYOUT
                if old["format"] == "Q4G64_F16S" else old["layout"],
            ))
        else:
            expected_specs.append(migration.ResourceSpec(
                old["name"], old["encoding"], old["bytes"]))
    expected_objects = [obj.to_json() for obj in migration.plan_objects(expected_specs)]
    expected_output = {"identity": {"model_id": MODEL_ID,
                                      "weights_id": profile["inventory"].WEIGHTS_ID},
                       "objects": expected_objects}
    if output != expected_output:
        raise ValueError("N16 artifact directory differs from its registered exact plan")
    output_payload_offset = migration.align_up(
        migration.PREFIX.size + output_directory_bytes, migration.PAYLOAD_ALIGNMENT)
    output_payload_bytes = max(obj["offset"] + obj["bytes"] for obj in expected_objects)
    if (source_payload_offset + source_payload_bytes != source_bytes
            or output_payload_offset + output_payload_bytes != output_bytes):
        raise ValueError("migration artifact extent differs from its exact object plan")
    return (migration._object_plan_sha256(source["objects"]),
            migration._object_plan_sha256(expected_objects))


def _producer() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _transcoder() -> dict[str, object]:
    path = Path(migration.__file__).resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _upstream(source: Path, validation: dict[str, object], profile: dict[str, Any]) -> tuple[Path, dict, str]:
    path = Path(str(source) + ".conversion.json")
    _regular(path, "legacy conversion receipt")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("legacy conversion receipt is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("legacy conversion receipt root must be an object")
    artifact = value.get("artifact") if isinstance(value, dict) else None
    candidate = value.get("candidate") if isinstance(value, dict) else None
    if (
        value.get("identity") != {"model_id": MODEL_ID,
                                  "weights_id": profile["source_weights_id"]}
        or value.get("target_key") != profile["inventory"].TARGET_KEY
        or value.get("recipe_id") != profile["source_recipe_id"]
        or not isinstance(artifact, dict)
        or Path(str(artifact.get("path"))).resolve(strict=True) != source
        or artifact.get("bytes") != validation["source_bytes"]
        or (artifact.get("sha256") is not None
            and artifact.get("sha256") != validation["source_sha256"])
        or not isinstance(candidate, dict)
        or candidate.get("status") != "registered-evaluation-only"
        or candidate.get("weight_recipe_selected") is not False
        or not isinstance(value.get("source"), dict)
    ):
        raise ValueError("legacy conversion receipt differs from the migrated source")
    return path.resolve(strict=True), value, hashlib.sha256(payload).hexdigest()


def _payload(source_path: Path, output_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    validation = migration._validate_migration_pair(source_path, output_path)
    profile = PROFILES.get(validation["identity"]["weights_id"])
    if profile is None:
        raise ValueError("common receipt producer supports only all-Q4 and mixed N16 artifacts")
    source = Path(str(validation["source"])); output = Path(str(validation["output"]))
    upstream_path, upstream, upstream_sha = _upstream(source, validation, profile)
    inventory = profile["inventory"]
    producer = _producer(); transcoder = _transcoder()
    receipt = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "identity": validation["identity"],
        "target_key": inventory.TARGET_KEY,
        "recipe_id": inventory.RECIPE_ID,
        "source": upstream["source"],
        "artifact": {"path": str(output), "bytes": validation["output_bytes"],
                     "sha256": validation["output_sha256"]},
        "candidate": {
            "status": "registered-evaluation-only", "weight_recipe_selected": False,
            "format_counts": inventory.FORMAT_COUNTS,
            "format_encoded_bytes": inventory.FORMAT_ENCODED_BYTES,
            "tensor_encoded_bytes": inventory.TENSOR_ENCODED_BYTES,
            "device_arena_bytes": inventory.DEVICE_ARENA_BYTES,
            "object_plan_sha256": validation["object_plan_sha256"],
        },
        "migration": {
            "source_artifact": {"path": str(source),
                "identity": validation["source_identity"], "bytes": validation["source_bytes"],
                "sha256": validation["source_sha256"],
                "object_plan_sha256": validation["source_object_plan_sha256"]},
            "source_conversion_receipt": {"path": str(upstream_path), "sha256": upstream_sha},
            "transcoder": transcoder,
            "receipt_producer": producer,
            "storage_transform": {"from": migration.OLD_Q4_LAYOUT,
                                  "to": migration.NEW_Q4_LAYOUT},
            "verification": "exact logical Q4 code/scale hashes and exact non-Q4 payload hashes",
            "objects": validation["objects"], "q4_objects": validation["q4_objects"],
        },
    }
    return receipt, validation


def validate_receipt(path: Path, artifact: dict[str, object]) -> dict[str, object]:
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    artifact_path = Path(os.path.abspath(os.fspath(str(artifact.get("path", "")))))
    if lexical_path != Path(str(artifact_path) + ".conversion.json"):
        raise ValueError("N16 migration receipt is not adjacent to its artifact")
    path = lexical_path
    receipt_identity = _regular(path, "N16 migration receipt")
    try:
        receipt_bytes = migration._read_bytes_identity(path, receipt_identity)
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("N16 migration receipt is invalid") from error
    if not isinstance(receipt, dict):
        raise ValueError("N16 migration receipt root must be an object")
    profile = PROFILES.get(artifact.get("weights_id"))
    if profile is None:
        raise ValueError("N16 migration receipt identity is unsupported")
    inventory = profile["inventory"]
    block = receipt.get("migration")
    candidate = receipt.get("candidate")
    output = receipt.get("artifact")
    if not all(isinstance(value, dict) for value in (block, candidate, output)):
        raise ValueError("N16 migration receipt blocks must be objects")
    source = block.get("source_artifact")
    upstream = block.get("source_conversion_receipt")
    transcoder = block.get("transcoder")
    producer = block.get("receipt_producer")
    if not all(isinstance(value, dict) for value in (
        source, upstream, transcoder, producer,
    )):
        raise ValueError("N16 migration receipt ancestry blocks must be objects")
    source_path = Path(str(source.get("path"))); upstream_path = Path(str(upstream.get("path")))
    transcoder_path = Path(str(transcoder.get("path"))); producer_path = Path(str(producer.get("path")))
    expected_transcoder = Path(migration.__file__).resolve(strict=True)
    expected_producer = Path(__file__).resolve(strict=True)
    for authority_path, label in (
        (source_path, "legacy source artifact"),
        (upstream_path, "legacy conversion receipt"),
        (transcoder_path, "N16 transcoder"),
        (producer_path, "N16 receipt producer"),
    ):
        _regular(authority_path, label)
    source_plan_sha256, output_plan_sha256 = _validate_exact_plans(
        source_path, Path(str(output.get("path"))), profile)
    validated_upstream_path, validated_upstream, validated_upstream_sha = _upstream(
        source_path,
        {"source_bytes": source.get("bytes"), "source_sha256": source.get("sha256")},
        profile,
    )
    if (
        receipt.get("artifact_type") != ARTIFACT_TYPE or receipt.get("schema_version") != 1
        or receipt.get("identity") != {"model_id": MODEL_ID,
                                       "weights_id": artifact["weights_id"]}
        or receipt.get("target_key") != inventory.TARGET_KEY
        or receipt.get("recipe_id") != inventory.RECIPE_ID
        or output.get("path") != str(artifact_path) or output.get("bytes") != artifact.get("bytes")
        or output.get("sha256") != artifact.get("sha256")
        or candidate.get("status") != "registered-evaluation-only"
        or candidate.get("weight_recipe_selected") is not False
        or candidate.get("format_counts") != inventory.FORMAT_COUNTS
        or candidate.get("format_encoded_bytes") != inventory.FORMAT_ENCODED_BYTES
        or candidate.get("tensor_encoded_bytes") != inventory.TENSOR_ENCODED_BYTES
        or candidate.get("device_arena_bytes") != inventory.DEVICE_ARENA_BYTES
        or candidate.get("object_plan_sha256") != output_plan_sha256
        or source.get("identity") != {"model_id": MODEL_ID,
                                      "weights_id": profile["source_weights_id"]}
        or source.get("object_plan_sha256") != source_plan_sha256
        or upstream_path != validated_upstream_path
        or upstream.get("sha256") != validated_upstream_sha
        or receipt.get("source") != validated_upstream.get("source")
        or block.get("storage_transform") != {"from": migration.OLD_Q4_LAYOUT,
                                              "to": migration.NEW_Q4_LAYOUT}
        or block.get("verification")
        != "exact logical Q4 code/scale hashes and exact non-Q4 payload hashes"
        or block.get("objects") != 1124 or block.get("q4_objects") != profile["q4_objects"]
        or transcoder_path != expected_transcoder or producer_path != expected_producer
        or transcoder.get("sha256") != _sha256(expected_transcoder)
        or producer.get("sha256") != _sha256(expected_producer)
        or upstream_path != Path(str(source_path) + ".conversion.json")
        or upstream.get("sha256") != _sha256(upstream_path)
        or source.get("bytes") != source_path.stat().st_size
        or source.get("sha256") != _sha256(source_path)
    ):
        raise ValueError("N16 migration receipt differs from its authority")
    if _regular(path, "N16 migration receipt") != receipt_identity:
        raise ValueError("N16 migration receipt changed during validation")
    return {"path": str(Path(os.path.abspath(os.fspath(path)))),
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "recipe_id": inventory.RECIPE_ID,
            "object_plan_sha256": candidate["object_plan_sha256"],
            "source_artifact_sha256": source["sha256"],
            "source_receipt_sha256": upstream["sha256"],
            "transcoder_sha256": transcoder["sha256"],
            "receipt_producer_sha256": producer["sha256"]}


def publish(source: Path, output: Path) -> dict[str, object]:
    receipt, validation = _payload(source, output)
    output_path = Path(str(validation["output"]))
    path = Path(str(output_path) + ".conversion.json")
    block = receipt["migration"]
    owners = (
        (Path(str(validation["source"])), validation["source_file_identity"],
         validation["source_sha256"]),
        (Path(str(validation["output"])), validation["output_file_identity"],
         validation["output_sha256"]),
        (Path(block["source_conversion_receipt"]["path"]), None,
         block["source_conversion_receipt"]["sha256"]),
        (Path(block["transcoder"]["path"]), None, block["transcoder"]["sha256"]),
        (Path(block["receipt_producer"]["path"]), None,
         block["receipt_producer"]["sha256"]),
    )
    opened = []
    try:
        for owner_path, expected_identity, expected_hash in owners:
            descriptor = os.open(owner_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            observed = os.fstat(descriptor)
            fingerprint = migration._stat_fingerprint(observed)
            if (not stat.S_ISREG(observed.st_mode)
                    or (expected_identity is not None
                        and fingerprint[:3] != expected_identity)):
                os.close(descriptor)
                raise RuntimeError("migration owner changed while opening receipt transaction")
            opened.append((owner_path, descriptor, fingerprint, expected_hash))

        def stable() -> bool:
            for owner_path, descriptor, fingerprint, expected_hash in opened:
                if (migration._stat_fingerprint(os.fstat(descriptor)) != fingerprint
                        or migration._identity(owner_path) != fingerprint[:3]
                        or migration._hash_open_fd(descriptor) != expected_hash
                        or migration._stat_fingerprint(os.fstat(descriptor)) != fingerprint):
                    return False
            return all(migration._identity(owner_path) == fingerprint[:3]
                       for owner_path, _, fingerprint, _ in opened)

        if not stable():
            raise RuntimeError("migration authorities changed before receipt publication")
        digest, receipt_identity, receipt_fingerprint = migration._publish_json_create_only(
            path, receipt)
        if not stable():
            raise RuntimeError("migration authorities changed after receipt publication")
        artifact = {"path": str(output_path), "bytes": validation["output_bytes"],
                    "sha256": validation["output_sha256"],
                    "weights_id": validation["identity"]["weights_id"]}
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
        actual = migration._read_bytes_identity(path, receipt_identity)
        summary = validate_receipt(path, artifact)
        if (actual != encoded or hashlib.sha256(actual).hexdigest() != digest
                or migration._identity(path) != receipt_identity
                or summary["sha256"] != digest or not stable()):
            raise RuntimeError("published N16 migration receipt changed")
        return summary
    except BaseException:
        if "receipt_fingerprint" in locals():
            migration._unlink_owned_fingerprint(path, receipt_fingerprint)
        raise
    finally:
        for _, descriptor, _, _ in opened:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(publish(args.source, args.output), indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
