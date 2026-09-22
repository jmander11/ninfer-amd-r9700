"""Create-only CPU conversion of the fixed 28-object selective-protected recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from tools.artifact.container import (
    Artifact, ArtifactIdentity, ArtifactWriter, TensorObject, TensorSpec as ArtifactTensorSpec,
    ResourceSpec as ArtifactResourceSpec, encode_directory, plan_objects, PREFIX_BYTES,
    PAYLOAD_ALIGNMENT,
)
from tools.artifact.layouts import align_up, encode_direct
from tools.convert.common.quantize import quantize_and_encode
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common.inventory import BF16, W8, TensorSpec
from . import q4_inventory, source, source_recipe, selective_protected_inventory as inventory
from .convert_q4_w8_mse import _checkpoint_receipt
from .e4m3_rowwise import encode_e4m3_rowwise_chunks


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _validate_objects(artifact, expected) -> None:
    if len(artifact.objects) != len(expected):
        raise ValueError("artifact object inventory count differs")
    for obj, spec in zip(artifact.objects, expected, strict=True):
        if obj.name != spec.name:
            raise ValueError("artifact object order differs")
        if isinstance(spec, TensorSpec):
            if not isinstance(obj, TensorObject) or (obj.shape, obj.format, obj.layout) != (
                    spec.shape, spec.format, spec.layout):
                raise ValueError(f"artifact tensor contract differs: {spec.name}")
        elif isinstance(obj, TensorObject) or obj.encoding != spec.encoding:
            raise ValueError(f"artifact resource contract differs: {spec.name}")


def _output_specs(base):
    tensors = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    return tuple(
        ArtifactTensorSpec(obj.name, tensors[obj.name].shape, tensors[obj.name].format,
                           tensors[obj.name].layout)
        if isinstance(obj, TensorObject) else ArtifactResourceSpec(obj.name, obj.encoding, obj.bytes)
        for obj in base.objects
    )


def preflight(base: Path, model: Path, output: Path) -> dict:
    inventory.validate_inventory()
    if output.suffix != ".ninfer":
        raise ValueError("output must have .ninfer suffix")
    for path in (output, Path(str(output) + ".conversion.json")):
        if os.path.lexists(path):
            raise FileExistsError(path)
    source.validate_config(json.loads((model / "config.json").read_text()))
    metadata = source_recipe.preflight_sources(model)
    if (metadata.source_tensor_count != 1199 or metadata.source_shard_count != 18
            or metadata.source_dtype_counts != {"BF16": 1199}):
        raise ValueError("requires the original complete 1199-tensor BF16 source")
    with Artifact(base) as artifact:
        if artifact.identity != ArtifactIdentity(inventory.MODEL_ID, q4_inventory.WEIGHTS_ID):
            raise ValueError("base must be the registered all-Q4 N16 evaluator")
        _validate_objects(artifact, q4_inventory.OBJECT_SPECS)
        objects = plan_objects(_output_specs(artifact))
        directory = encode_directory(ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID), objects)
        projected = align_up(PREFIX_BYTES + len(directory), PAYLOAD_ALIGNMENT) + objects[-1].offset + objects[-1].bytes
    return {
        "artifact_type": "ninfer_r9700_selective_protected_conversion", "schema_version": 1,
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
        "recipe_id": inventory.RECIPE_ID, "weight_recipe_selected": False,
        "base": {"path": str(base.resolve()), "bytes": base.stat().st_size,
                 "weights_id": q4_inventory.WEIGHTS_ID},
        "source": {"model_path": str(model.resolve()), "checkpoint": _checkpoint_receipt(model)},
        "changed_formats": dict(sorted(inventory.CHANGED_FORMATS.items())),
        "format_counts": inventory.FORMAT_COUNTS,
        "tensor_encoded_bytes": inventory.TENSOR_ENCODED_BYTES,
        "device_arena_bytes": inventory.DEVICE_ARENA_BYTES,
        "projected_file_bytes": projected,
    }


def encode_changed(tensor, spec):
    """Reuse canonical encoders; only chunk allocation, never quantization groups."""
    if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != spec.shape:
        raise ValueError("changed matrices require exact represented CPU BF16 source and shape")
    if spec.format == inventory.F8E4M3_ROW_F32S:
        yield from encode_e4m3_rowwise_chunks(tensor)
    elif spec.format == BF16:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("BF16 source matrix contains nonfinite values")
        yield encode_direct(tensor, BF16)
    elif spec.format == W8:
        rows, columns = spec.shape
        padded = align_up(columns, 128)
        scales = bytearray()
        for begin in range(0, rows, 256):
            part = tensor[begin:begin + 256]
            payload = quantize_and_encode(part, W8, device="cpu")
            code_bytes = part.shape[0] * padded
            yield payload[:code_bytes]
            scales.extend(payload[align_up(code_bytes, 256):])
        yield bytes(align_up(rows * padded, 256) - rows * padded)
        yield bytes(scales)
    else:
        raise ValueError("format is not a selective-protected replacement")


def _recorded_chunks(chunks, record):
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
        yield chunk
    record["payload_sha256"] = digest.hexdigest()


def _copied_chunks(artifact, obj):
    with artifact.payload(obj) as payload:
        for begin in range(0, len(payload), 8 << 20):
            yield payload[begin:begin + (8 << 20)]


def write_payloads(base, writer, reader):
    """One ordered copy/replace pass; returned per-object hashes bind actual bytes written."""
    specs = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    records = []
    for obj in base.objects:
        record = {"name": obj.name, "operation": "copy-exact"}
        if obj.name in inventory.CHANGED_FORMATS:
            spec = specs[obj.name]
            tensor = source_recipe.materialize_recipe(source_recipe.RECIPES_BY_NAME[obj.name], reader)
            record.update(operation="source-bf16-reencode", format=spec.format,
                          source_bf16_sha256=hashlib.sha256(
                              memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast("B")
                          ).hexdigest())
            writer.write(obj.name, _recorded_chunks(encode_changed(tensor, spec), record))
            del tensor
        else:
            writer.write(obj.name, _recorded_chunks(_copied_chunks(base, obj), record))
        records.append(record)
    return records


def convert(base: Path, model: Path, output: Path) -> Path:
    report = preflight(base, model, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["base"]["sha256"] = sha(base)
    with Artifact(base) as artifact, ShardReader(model) as reader:
        with ArtifactWriter(output, ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
                            _output_specs(artifact)) as writer:
            report["objects"] = write_payloads(artifact, writer, reader)
    report["artifact"] = {"path": str(output.resolve()), "bytes": output.stat().st_size,
                          "sha256": sha(output)}
    report["converter"] = {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))}
    _atomic_json(Path(str(output) + ".conversion.json"), report)
    return output


def validate(path: Path, base: Path, model: Path) -> dict:
    report = json.loads(Path(str(path) + ".conversion.json").read_text())
    if (report.get("artifact_type") != "ninfer_r9700_selective_protected_conversion"
            or report.get("schema_version") != 1
            or report.get("identity") != {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID}
            or report.get("recipe_id") != inventory.RECIPE_ID
            or report.get("weight_recipe_selected") is not False
            or report.get("changed_formats") != inventory.CHANGED_FORMATS
            or report.get("format_counts") != inventory.FORMAT_COUNTS
            or report.get("tensor_encoded_bytes") != inventory.TENSOR_ENCODED_BYTES
            or report.get("device_arena_bytes") != inventory.DEVICE_ARENA_BYTES
            or report.get("projected_file_bytes") != path.stat().st_size
            or report.get("source", {}).get("model_path") != str(model.resolve())
            or report.get("source", {}).get("checkpoint") != _checkpoint_receipt(model)
            or report.get("base", {}).get("path") != str(base.resolve())
            or report.get("base", {}).get("bytes") != base.stat().st_size
            or report.get("base", {}).get("sha256") != sha(base)
            or report.get("artifact") != {"path": str(path.resolve()), "bytes": path.stat().st_size,
                                         "sha256": sha(path)}):
        raise ValueError("selective-protected conversion receipt differs")
    with Artifact(path) as artifact, Artifact(base) as base_artifact:
        if base_artifact.identity != ArtifactIdentity(inventory.MODEL_ID, q4_inventory.WEIGHTS_ID):
            raise ValueError("base artifact identity differs")
        _validate_objects(base_artifact, q4_inventory.OBJECT_SPECS)
        if artifact.identity != ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID):
            raise ValueError("selective-protected artifact identity differs")
        _validate_objects(artifact, inventory.OBJECT_SPECS)
        records = report.get("objects", [])
        if [row.get("name") for row in records] != [obj.name for obj in artifact.objects]:
            raise ValueError("conversion object evidence is incomplete")
        for obj, row in zip(artifact.objects, records, strict=True):
            changed = obj.name in inventory.CHANGED_FORMATS
            if row.get("operation") != ("source-bf16-reencode" if changed else "copy-exact"):
                raise ValueError("conversion copy/replacement map differs")
            with artifact.payload(obj) as payload:
                if hashlib.sha256(payload).hexdigest() != row.get("payload_sha256"):
                    raise ValueError(f"converted object hash differs: {obj.name}")
                if not changed:
                    with base_artifact.payload(obj.name) as original:
                        if payload != original:
                            raise ValueError(f"unchanged payload is not byte-exact: {obj.name}")
            if changed and (row.get("format") != inventory.CHANGED_FORMATS[obj.name]
                            or not isinstance(row.get("source_bf16_sha256"), str)
                            or len(row["source_bf16_sha256"]) != 64):
                raise ValueError("changed object lacks represented BF16 provenance")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true", help="CPU preflight only; create nothing")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(preflight(args.base, args.model, args.out), indent=2))
    else:
        print(convert(args.base, args.model, args.out))


if __name__ == "__main__":
    main()
