#!/usr/bin/env python3
"""Publish the no-artifact converter preflight for the exact schema-v7 winner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from tools.convert.qwen3_8_27b_r9700 import (
    convert_fp8_hybrid,
    convert_q4,
    convert_q4_w8_mse,
    preflight_identity,
    fp8_hybrid_inventory,
    q4_inventory,
    q4_w8_mse_inventory,
    resources as target_resources,
)

REPO = Path(__file__).resolve().parents[2]
RESOLVER = REPO / "profiles/bench/post-terminal-focused-verification-20260905/resolve.py"
RECIPES = {
    "r9700-q4g64-n16k16-eval": (convert_q4, q4_inventory, "all_q4"),
    "r9700-q4-w8-mse-n16k16-eval": (convert_q4_w8_mse, q4_w8_mse_inventory, "mixed_source_mse"),
    "r9700-q4g64-f8e4m3-four-role-n16k16-eval": (
        convert_fp8_hybrid, fp8_hybrid_inventory, "four_role_fp8"),
}
PREFLIGHT_TYPES = {
    "all_q4": "ninfer_qwen3_8_27b_r9700_q4_conversion_preflight",
    "mixed_source_mse": "ninfer_qwen3_8_27b_r9700_q4_w8_mse_conversion_preflight",
    "four_role_fp8": "ninfer_qwen3_8_27b_r9700_fp8_q4_hybrid_conversion_preflight",
}
SHARD_HASH_POLICY = (
    "not computed: exact names and nonempty regular files are the target-owned contract")


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def snapshot(path: Path, label: str) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    path = path.resolve(strict=True)
    return {"path": str(path), "file_size_bytes": path.stat().st_size, "sha256": sha(path)}


def file_receipt(path: Path, label: str) -> dict[str, Any]:
    value = snapshot(path, label)
    return {"path": value["path"], "bytes": value["file_size_bytes"],
            "sha256": value["sha256"]}


def resolve(selection: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("ninfer_converter_route", RESOLVER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load terminal route resolver")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.resolve(selection)


def validate_prospective_artifact(path: Path) -> Path:
    if not path.is_absolute() or path.suffix != ".ninfer":
        raise ValueError("prospective final artifact must be an absolute .ninfer path")
    output = path.absolute()
    report = Path(str(output) + ".conversion.json")
    if os.path.lexists(output) or os.path.lexists(report):
        raise ValueError("prospective final artifact and conversion report must be absent")
    candidate = output.parent
    while not os.path.lexists(candidate):
        parent = candidate.parent
        if parent == candidate:
            raise ValueError("prospective artifact has no existing filesystem ancestor")
        candidate = parent
    metadata = candidate.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("prospective artifact ancestor must be a real directory")
    if not os.access(candidate, os.W_OK | os.X_OK):
        raise ValueError("prospective artifact filesystem is not writable")
    return output


def _validate_summary(summary: dict[str, Any], weights_id: str, inventory: object, model: Path,
                      ranking: Path, prospective: Path) -> None:
    branch = RECIPES[weights_id][2]
    source = summary.get("source")
    checkpoint = summary.get("checkpoint")
    resources = summary.get("frontend_resources")
    objects = summary.get("objects")
    ranking_value = summary.get("ranking")
    expected_checkpoint_keys = {"config", "index", "indexed_tensor_count",
                                "index_total_tensor_bytes", "shards",
                                "shard_total_file_bytes", "shard_payload_sha256",
                                "shard_payload_hash_policy"}
    if (
        summary.get("artifact_type") != PREFLIGHT_TYPES[branch]
        or summary.get("schema_version") != 1
        or summary.get("identity") != {"model_id": "qwen3.8-27b", "weights_id": weights_id}
        or summary.get("target_key") != inventory.TARGET_KEY
        or summary.get("recipe_id") != inventory.RECIPE_ID
        or Path(str(summary.get("model_path", ""))).resolve() != model
        or not isinstance(source, dict) or source.get("tensors") != 1199
        or source.get("shards") != 18 or source.get("dtypes") != {"BF16": 1199}
        or not isinstance(checkpoint, dict) or set(checkpoint) != expected_checkpoint_keys
        or checkpoint.get("config") != file_receipt(model / "config.json", "source config")
        or checkpoint.get("index") != file_receipt(
            model / "model.safetensors.index.json", "source index")
        or checkpoint.get("indexed_tensor_count") != 1199
        or checkpoint.get("index_total_tensor_bytes") != preflight_identity.bf16_protocol.SOURCE_TOTAL_BYTES
        or not isinstance(checkpoint.get("shards"), list) or len(checkpoint["shards"]) != 18
        or [row.get("name") for row in checkpoint["shards"]] != list(preflight_identity.SOURCE_SHARDS)
        or any(not isinstance(row.get("bytes"), int) or row["bytes"] <= 0
               for row in checkpoint["shards"])
        or checkpoint.get("shard_total_file_bytes") != sum(
            row["bytes"] for row in checkpoint["shards"])
        or checkpoint.get("shard_payload_sha256") is not None
        or checkpoint.get("shard_payload_hash_policy") != SHARD_HASH_POLICY
        or not isinstance(resources, list) or len(resources) != 6
        or any(set(row) != {"name", "bytes", "sha256"} for row in resources)
        or [row["name"] for row in resources] != list(target_resources.OFFICIAL_RESOURCE_SHA256)
        or any(row["sha256"] != target_resources.OFFICIAL_RESOURCE_SHA256[row["name"]]
               or not isinstance(row["bytes"], int) or row["bytes"] <= 0 for row in resources)
        or not isinstance(objects, dict) or objects.get("count") != 1124
        or not isinstance(summary.get("object_plan_sha256"), str)
        or len(summary["object_plan_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in summary["object_plan_sha256"])
        or not isinstance(ranking_value, dict)
        or Path(str(ranking_value.get("path", ""))).resolve() != ranking
        or ranking_value.get("sha256") != sha(ranking)
        or summary.get("format_counts") != inventory.FORMAT_COUNTS
        or summary.get("format_encoded_bytes") != inventory.FORMAT_ENCODED_BYTES
        or summary.get("tensor_encoded_bytes") != inventory.TENSOR_ENCODED_BYTES
        or summary.get("device_arena_bytes") != inventory.DEVICE_ARENA_BYTES
    ):
        raise ValueError("selected converter preflight summary is incomplete")
    provenance = Path(str(ranking_value.get("provenance_path", "")))
    if (not provenance.is_absolute()
            or ranking_value.get("provenance_sha256") != sha(provenance)):
        raise ValueError("selected converter preflight ranking provenance is incomplete")
    if weights_id == "r9700-q4g64-f8e4m3-four-role-n16k16-eval":
        destination = summary.get("destination", {})
        if (Path(str(destination.get("artifact_path", ""))).resolve() != prospective
                or destination.get("destination_exists") is not False
                or summary.get("selection_sha256") is None):
            raise ValueError("four-role preflight lacks its absent destination/selection identity")
    elif "destination" in summary:
        raise ValueError("non-hybrid converter preflight unexpectedly owns a destination")


def build_value(selection: Path, model: Path, ranking: Path,
                prospective_artifact: Path) -> dict[str, Any]:
    selection = selection.resolve(strict=True)
    route = resolve(selection)
    weights_id = route["artifact"]["weights_id"]
    if weights_id not in RECIPES:
        raise ValueError("terminal winner has no cutover converter")
    model = model.resolve(strict=True); ranking = ranking.resolve(strict=True)
    prospective = validate_prospective_artifact(prospective_artifact)
    converter, inventory, branch = RECIPES[weights_id]
    preflight = converter.preflight_conversion(model, ranking)
    if converter is convert_fp8_hybrid:
        destination = converter.preflight_destination(preflight, prospective)
        summary = converter.preflight_summary(preflight, destination)
    else:
        summary = converter.preflight_summary(preflight)
    _validate_summary(summary, weights_id, inventory, model, ranking, prospective)
    return {
        "artifact_type": "ninfer_r9700_selected_converter_preflight",
        "schema_version": 1, "status": "passed_no_artifact_no_device",
        "selected_route": {
            "terminal_selection": snapshot(selection, "terminal selection"),
            "winner": route["winner"], "evaluation_artifact": route["artifact"],
            "weights_id": weights_id, "branch": branch,
        },
        "prospective_final_artifact": str(prospective),
        "preflight": summary,
        "authorities": {
            "converter": snapshot(Path(converter.__file__), "selected converter"),
            "inventory": snapshot(Path(converter.__file__).with_name(
                {"all_q4": "q4_inventory.py", "mixed_source_mse": "q4_w8_mse_inventory.py",
                 "four_role_fp8": "fp8_hybrid_inventory.py"}[branch]), "selected inventory"),
            "identity_helper": snapshot(Path(preflight_identity.__file__), "identity helper"),
            "resolver": snapshot(RESOLVER, "terminal resolver"),
        },
    }


def _owned(path: Path, owner: tuple[int, int]) -> bool:
    try: metadata = path.lstat()
    except FileNotFoundError: return False
    return stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == owner


def _stable(value: dict[str, Any]) -> dict[str, Any]:
    copy = json.loads(json.dumps(value))
    destination = copy.get("preflight", {}).get("destination")
    if isinstance(destination, dict):
        destination.pop("available_free_bytes", None)
    return copy


def revalidate(path: Path, selection: Path) -> dict[str, Any]:
    snapshot(path, "selected converter preflight")
    value = json.loads(path.read_text(encoding="utf-8"))
    preflight = value.get("preflight", {})
    ranking = Path(str(preflight.get("ranking", {}).get("path", "")))
    model = Path(str(preflight.get("model_path", "")))
    prospective = Path(str(value.get("prospective_final_artifact", "")))
    rebuilt = build_value(selection, model, ranking, prospective)
    if _stable(rebuilt) != _stable(value):
        raise ValueError("selected converter preflight does not revalidate")
    destination = rebuilt.get("preflight", {}).get("destination")
    if isinstance(destination, dict) and destination.get("available_free_bytes", 0) < destination.get(
            "required_free_bytes", 1):
        raise ValueError("four-role destination no longer has required free space")
    return value


def publish(selection: Path, model: Path, ranking: Path, prospective: Path, out: Path) -> dict:
    if os.path.lexists(out):
        raise ValueError("refusing to overwrite selected converter preflight")
    value = build_value(selection, model, ranking, prospective)
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    temporary = Path(temporary_name); owner = None; published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            metadata = os.fstat(stream.fileno()); owner = (metadata.st_dev, metadata.st_ino)
        if not _owned(temporary, owner): raise ValueError("pending preflight inode changed")
        os.link(temporary, out); published = True
        if not _owned(out, owner) or json.loads(out.read_text(encoding="utf-8")) != value:
            raise ValueError("published preflight differs from owned pending inode")
        if _stable(build_value(selection, model, ranking, prospective)) != _stable(value):
            raise ValueError("converter inputs changed during preflight publication")
        directory = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
        if _owned(temporary, owner): temporary.unlink()
        return value
    except Exception:
        if owner is not None:
            if published and _owned(out, owner): out.unlink()
            if _owned(temporary, owner): temporary.unlink()
        raise
    finally:
        if owner is None and os.path.lexists(temporary):
            metadata = temporary.lstat()
            if stat.S_ISREG(metadata.st_mode):
                temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--draft-ranking", required=True, type=Path)
    parser.add_argument("--prospective-artifact", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try: publish(args.selection, args.model, args.draft_ranking, args.prospective_artifact, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__": main()
